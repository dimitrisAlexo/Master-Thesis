from pymongo import MongoClient
import numpy as np
import pickle
from tqdm import tqdm
from datetime import datetime

# Import preprocessing functions
from preprocessTypingAccelData import (
    is_valid_recording,
    resample_signal,
    remove_gravity,
    segment_signal,
)
from preprocessTypingKeyboardData import (
    compute_hold_time,
    compute_flight_time,
    create_histogram_features,
)


def parse_datetime(doc):
    """
    Extract datetime from document.

    Args:
        doc: MongoDB document

    Returns:
        datetime object or None
    """
    if "datetime" in doc and doc["datetime"] is not None:
        return doc["datetime"]
    return None


def process_accelerometer_window(accel_x, accel_y, accel_z, accel_t):
    """
    Process accelerometer data for a single window.
    Uses preprocessing functions from preprocessTypingAccelData.py

    Applies complete preprocessing pipeline:
    - Validation: duration ≥20s, sampling rate ≥50Hz, no extreme values
    - Resampling to 100Hz using polyphase resampling
    - Gravity removal using high-pass FIR filter (512 order, 1Hz cutoff)
    - Exclude first/last 5 seconds
    - Segmentation into 10s windows
    - Energy filtering (≥0.30 m/s²)²

    Returns:
        (segments, reason) where segments is list of arrays (1000, 3) or None if invalid
    """
    # Validate recording
    is_valid, sampling_rate, duration, reason = is_valid_recording(
        accel_x, accel_y, accel_z, accel_t
    )

    if not is_valid:
        return None, reason

    # Resample to 100 Hz
    accel_x_resampled = resample_signal(accel_x, accel_t, target_rate=100)
    accel_y_resampled = resample_signal(accel_y, accel_t, target_rate=100)
    accel_z_resampled = resample_signal(accel_z, accel_t, target_rate=100)

    # Remove gravity component
    accel_x_filtered = remove_gravity(accel_x_resampled, sampling_rate=100)
    accel_y_filtered = remove_gravity(accel_y_resampled, sampling_rate=100)
    accel_z_filtered = remove_gravity(accel_z_resampled, sampling_rate=100)

    # Exclude first and last 5 seconds (500 samples at 100Hz)
    samples_to_exclude = 500
    if len(accel_x_filtered) > 2 * samples_to_exclude:
        accel_x_filtered = accel_x_filtered[samples_to_exclude:-samples_to_exclude]
        accel_y_filtered = accel_y_filtered[samples_to_exclude:-samples_to_exclude]
        accel_z_filtered = accel_z_filtered[samples_to_exclude:-samples_to_exclude]

    # Segment into 10s windows
    segments = segment_signal(accel_x_filtered, accel_y_filtered, accel_z_filtered)

    return segments, None


def process_typing_session(doc):
    """
    Process typing session to extract histogram features.
    Uses preprocessing functions from preprocessTypingKeyboardData.py

    Applies same filtering:
    - Minimum 40 keystrokes per session
    - Valid hold times and flight times
    - Creates 502-dimensional histogram (101 HT bins + 401 FT bins)

    Returns:
        Feature vector (502,) or None if invalid
    """
    MIN_KEYSTROKES = 40

    if "payload" not in doc:
        return None

    payload = doc["payload"]
    down_time = payload.get("DownTime", [])
    up_time = payload.get("UpTime", [])

    # Check minimum keystroke requirement
    if len(down_time) < MIN_KEYSTROKES:
        return None

    # Compute hold times
    hold_times = compute_hold_time(down_time, up_time)
    if hold_times is None:
        return None

    # Compute flight times
    flight_times = compute_flight_time(down_time)

    # Filter out negative values
    hold_times = hold_times[hold_times >= 0]
    flight_times = flight_times[flight_times >= 0]

    # Check if we still have enough data after filtering
    if len(hold_times) < MIN_KEYSTROKES or len(flight_times) < (MIN_KEYSTROKES - 1):
        return None

    # Create histogram features
    feature_vector = create_histogram_features(hold_times, flight_times)

    return feature_vector


def find_temporal_matches(
    max_batches=None,
    batch_size=1000,
    time_tolerance_seconds=5,
    output_filepath="../data/bimodal_dataset.pickle",
    save_every_n_batches=5,
):
    """
    Find temporal matches between typing sessions and accelerometer recordings.
    Saves incrementally to pickle file to avoid memory issues with large datasets.

    Args:
        max_batches: Maximum number of IMU batches to process (None = all)
        batch_size: Number of IMU documents per batch
        time_tolerance_seconds: Maximum time difference for a match (default: 5 seconds)
        output_filepath: Path to save the pickle file
        save_every_n_batches: Save to disk after processing this many batches (default: 5)

    Returns:
        List of matched pairs: [{'serial': str, 'typing_features': array(502,),
                                  'accel_segments': array(n, 1000, 3),
                                  'typing_datetime': datetime, 'accel_datetime': datetime}, ...]
    """
    import os

    client = MongoClient("mongodb://localhost:27017/")
    db = client["iprog"]
    typing_col = db["gdata_keyboard_raw"]
    imu_col = db["gdata_imu_raw"]

    print("=" * 80)
    print("CREATING BIMODAL DATASET - TEMPORAL MATCHING")
    print("=" * 80)
    print(f"Time tolerance for matching: {time_tolerance_seconds} seconds")
    print(f"Batch size: {batch_size:,}")
    if max_batches:
        print(f"Maximum batches to process: {max_batches:,}")
    else:
        print(f"Processing ALL batches")
    print(f"Saving to: {output_filepath}")
    print(f"Incremental save every: {save_every_n_batches} batches")
    print()

    # Load existing data if file exists
    checkpoint_path = output_filepath.replace(".pickle", "_checkpoint.pickle")
    processed_imu_ids = set()

    if os.path.exists(output_filepath):
        print(f"Found existing dataset at {output_filepath}")
        with open(output_filepath, "rb") as f:
            checkpoint_data = pickle.load(f)

        # Handle both old format (list) and new format (dict with metadata)
        if isinstance(checkpoint_data, dict):
            all_matched_pairs = checkpoint_data.get("matched_pairs", [])
            processed_imu_ids = checkpoint_data.get("processed_imu_ids", set())
            print(f"Loaded {len(all_matched_pairs)} existing matched pairs")
            print(f"Loaded {len(processed_imu_ids)} processed IMU document IDs")
        else:
            all_matched_pairs = checkpoint_data
            print(
                f"Loaded {len(all_matched_pairs)} existing matched pairs (old format)"
            )
            print(f"Warning: No processed IDs found, may reprocess some documents")

        print("Will resume processing from last checkpoint\n")
    else:
        all_matched_pairs = []
        print("Starting fresh dataset\n")

    current_batch_matches = []  # Accumulate matches between saves
    processed_imu_count = 0
    batch_count = 0

    # Statistics
    stats = {
        "imu_processed": 0,
        "imu_valid": 0,
        "imu_invalid": 0,
        "typing_queries": 0,
        "matches_found": 0,
        "total_accel_segments": 0,
    }

    # Create cursor for IMU data (Keyboard origin only)
    print("Fetching IMU documents with Origin='Keyboard'...")
    cursor = imu_col.find({"payload.Origin": "Keyboard"}).batch_size(batch_size)

    # Process in batches
    current_batch = []

    with tqdm(desc="Processing IMU documents", unit=" docs") as pbar:
        for imu_doc in cursor:
            # Skip if already processed
            imu_id = str(imu_doc["_id"])
            if imu_id in processed_imu_ids:
                pbar.update(1)
                continue

            current_batch.append(imu_doc)
            stats["imu_processed"] += 1
            pbar.update(1)

            # Process batch when full
            if len(current_batch) >= batch_size:
                batch_count += 1

                # Check if we've reached max batches
                if max_batches is not None and batch_count > max_batches:
                    print(f"\nReached maximum batch limit ({max_batches})")
                    break

                print(f"\n{'='*80}")
                print(f"PROCESSING BATCH {batch_count}")
                print(f"{'='*80}")

                # Process this batch
                batch_matches, batch_imu_ids = process_batch(
                    current_batch, typing_col, time_tolerance_seconds, stats
                )
                current_batch_matches.extend(batch_matches)
                processed_imu_ids.update(batch_imu_ids)

                # Clear batch
                current_batch = []

                # Print batch statistics
                print(f"\nBatch {batch_count} results:")
                print(f"  Matches found in this batch: {len(batch_matches)}")
                print(
                    f"  Matches accumulated since last save: {len(current_batch_matches)}"
                )
                print(
                    f"  Total matches so far: {len(all_matched_pairs) + len(current_batch_matches)}"
                )
                print(f"  Total IMU processed: {stats['imu_processed']:,}")
                print(f"  Valid IMU: {stats['imu_valid']:,}")
                print(f"  Invalid IMU: {stats['imu_invalid']:,}")
                print(f"  Total accel segments: {stats['total_accel_segments']:,}")

                # Save incrementally every N batches
                if batch_count % save_every_n_batches == 0:
                    all_matched_pairs.extend(current_batch_matches)
                    print(f"\n  💾 Saving checkpoint after {batch_count} batches...")
                    print(f"  Total pairs in dataset: {len(all_matched_pairs)}")
                    print(f"  Total processed IMU IDs: {len(processed_imu_ids)}")

                    checkpoint_data = {
                        "matched_pairs": all_matched_pairs,
                        "processed_imu_ids": processed_imu_ids,
                        "last_batch": batch_count,
                        "stats": stats.copy(),
                    }
                    with open(output_filepath, "wb") as f:
                        pickle.dump(checkpoint_data, f)
                    print(f"  ✓ Checkpoint saved to {output_filepath}")
                    current_batch_matches = []  # Clear accumulated matches

            # Check if we've reached max batches
            if max_batches is not None and batch_count >= max_batches:
                break

        # Process remaining documents in last incomplete batch
        if current_batch and (max_batches is None or batch_count < max_batches):
            batch_count += 1
            print(f"\n{'='*80}")
            print(
                f"PROCESSING FINAL BATCH {batch_count} ({len(current_batch)} documents)"
            )
            print(f"{'='*80}")

            batch_matches, batch_imu_ids = process_batch(
                current_batch, typing_col, time_tolerance_seconds, stats
            )
            current_batch_matches.extend(batch_matches)
            processed_imu_ids.update(batch_imu_ids)

    cursor.close()

    # Save any remaining matches
    if current_batch_matches:
        all_matched_pairs.extend(current_batch_matches)
        print(f"\n💾 Saving final matches...")
        checkpoint_data = {
            "matched_pairs": all_matched_pairs,
            "processed_imu_ids": processed_imu_ids,
            "last_batch": batch_count,
            "stats": stats.copy(),
        }
        with open(output_filepath, "wb") as f:
            pickle.dump(checkpoint_data, f)
        print(f"✓ Final save complete")

    # Final statistics
    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    print(f"Total IMU documents processed: {stats['imu_processed']:,}")
    print(f"Valid IMU recordings: {stats['imu_valid']:,}")
    print(f"Invalid IMU recordings: {stats['imu_invalid']:,}")
    print(f"Typing database queries: {stats['typing_queries']:,}")
    print(f"Total matched pairs: {len(all_matched_pairs):,}")
    print(f"Total accelerometer segments: {stats['total_accel_segments']:,}")

    if len(all_matched_pairs) > 0:
        segments_per_match = [m["accel_segments"].shape[0] for m in all_matched_pairs]
        print(f"\nSegments per matched pair:")
        print(f"  Min: {np.min(segments_per_match)}")
        print(f"  Max: {np.max(segments_per_match)}")
        print(f"  Mean: {np.mean(segments_per_match):.2f}")
        print(f"  Median: {np.median(segments_per_match):.2f}")

    client.close()

    return all_matched_pairs


def process_batch(imu_batch, typing_col, time_tolerance_seconds, stats):
    """
    Process a batch of IMU documents to find temporal matches.

    Args:
        imu_batch: List of IMU documents
        typing_col: MongoDB collection for typing data
        time_tolerance_seconds: Time tolerance for matching
        stats: Dictionary to update with statistics

    Returns:
        Tuple of (list of matched pairs, set of processed IMU document IDs)
    """
    batch_matches = []
    batch_imu_ids = set()

    for imu_doc in tqdm(imu_batch, desc="Matching IMU with typing", leave=False):
        imu_id = str(imu_doc["_id"])
        batch_imu_ids.add(imu_id)
        serial = imu_doc["serial"]
        imu_datetime = parse_datetime(imu_doc)

        if imu_datetime is None:
            stats["imu_invalid"] += 1
            continue

        # Extract accelerometer data
        payload = imu_doc.get("payload", {})
        accel_x = np.array(payload.get("AccelX", []))
        accel_y = np.array(payload.get("AccelY", []))
        accel_z = np.array(payload.get("AccelZ", []))
        accel_t = np.array(payload.get("AccelT", []))

        # Process accelerometer data
        if len(accel_x) == 0 or len(accel_t) == 0:
            stats["imu_invalid"] += 1
            continue

        segments, reason = process_accelerometer_window(
            accel_x, accel_y, accel_z, accel_t
        )

        if segments is None or len(segments) == 0:
            stats["imu_invalid"] += 1
            continue

        stats["imu_valid"] += 1

        # Find typing sessions within time tolerance
        time_min = imu_datetime.timestamp() - time_tolerance_seconds
        time_max = imu_datetime.timestamp() + time_tolerance_seconds

        # Query typing database for matching sessions
        query = {
            "serial": serial,
            "datetime": {
                "$gte": datetime.fromtimestamp(time_min),
                "$lte": datetime.fromtimestamp(time_max),
            },
        }

        stats["typing_queries"] += 1
        typing_docs = list(typing_col.find(query))

        # Process each matching typing session
        for typing_doc in typing_docs:
            typing_features = process_typing_session(typing_doc)

            if typing_features is not None:
                typing_datetime = parse_datetime(typing_doc)

                # Create matched pair
                matched_pair = {
                    "serial": serial,
                    "typing_features": typing_features,  # Shape: (502,)
                    "accel_segments": np.stack(
                        segments, axis=0
                    ),  # Shape: (n_segments, 1000, 3)
                    "typing_datetime": typing_datetime,
                    "accel_datetime": imu_datetime,
                    "time_diff_seconds": abs(
                        (typing_datetime - imu_datetime).total_seconds()
                    ),
                }

                batch_matches.append(matched_pair)
                stats["matches_found"] += 1
                stats["total_accel_segments"] += len(segments)

    return batch_matches, batch_imu_ids


def save_bimodal_dataset(matched_pairs, filepath="../data/bimodal_dataset.pickle"):
    """Save the bimodal dataset to pickle file."""
    print(f"\nSaving bimodal dataset to {filepath}...")
    with open(filepath, "wb") as f:
        pickle.dump(matched_pairs, f)
    print(f"✓ Dataset saved successfully")
    print(f"  Total matched pairs: {len(matched_pairs)}")


def print_dataset_summary(matched_pairs):
    """Print summary of the bimodal dataset."""
    print("\n" + "=" * 80)
    print("BIMODAL DATASET SUMMARY")
    print("=" * 80)

    # Handle both old format (list) and new format (dict with metadata)
    if isinstance(matched_pairs, dict):
        actual_pairs = matched_pairs.get("matched_pairs", [])
        print(f"Checkpoint info:")
        print(f"  Last batch processed: {matched_pairs.get('last_batch', 'N/A')}")
        print(
            f"  Processed IMU documents: {len(matched_pairs.get('processed_imu_ids', set()))}"
        )
        matched_pairs = actual_pairs

    if len(matched_pairs) == 0:
        print("No matched pairs in dataset")
        return

    print(f"\nTotal matched pairs: {len(matched_pairs)}")

    # Unique serials
    unique_serials = set(m["serial"] for m in matched_pairs)
    print(f"Unique subjects (serials): {len(unique_serials)}")

    # Segments per match
    segments_per_match = [m["accel_segments"].shape[0] for m in matched_pairs]
    print(f"\nAccelerometer segments per match:")
    print(f"  Min: {np.min(segments_per_match)}")
    print(f"  Max: {np.max(segments_per_match)}")
    print(f"  Mean: {np.mean(segments_per_match):.2f}")
    print(f"  Median: {np.median(segments_per_match):.2f}")

    # Time differences
    time_diffs = [m["time_diff_seconds"] for m in matched_pairs]
    print(f"\nTemporal alignment (time difference):")
    print(f"  Min: {np.min(time_diffs):.1f} seconds")
    print(f"  Max: {np.max(time_diffs):.1f} seconds")
    print(f"  Mean: {np.mean(time_diffs):.1f} seconds")
    print(f"  Median: {np.median(time_diffs):.1f} seconds")

    # Example
    print(f"\nExample matched pair (first entry):")
    example = matched_pairs[0]
    print(f"  Serial: {example['serial']}")
    print(f"  Typing features shape: {example['typing_features'].shape}")
    print(f"  Accel segments shape: {example['accel_segments'].shape}")
    print(f"  Typing datetime: {example['typing_datetime']}")
    print(f"  Accel datetime: {example['accel_datetime']}")
    print(f"  Time difference: {example['time_diff_seconds']:.1f} seconds")

    print("\n" + "=" * 80)


def visualize_matched_pairs(matched_pairs, num_pairs=3, max_accel_segments=3):
    """
    Visualize sample matched pairs showing typing histogram and accelerometer segments.

    Args:
        matched_pairs: List of matched pair dictionaries
        num_pairs: Number of pairs to visualize
        max_accel_segments: Maximum number of accelerometer segments to show per pair
    """
    import matplotlib.pyplot as plt

    print("\n" + "=" * 80)
    print("VISUALIZING MATCHED PAIRS")
    print("=" * 80)

    if len(matched_pairs) == 0:
        print("No matched pairs to visualize")
        return

    num_pairs = min(num_pairs, len(matched_pairs))

    for pair_idx in range(num_pairs):
        pair = matched_pairs[pair_idx]
        serial = pair["serial"]
        typing_features = pair["typing_features"]
        accel_segments = pair["accel_segments"]
        time_diff = pair["time_diff_seconds"]

        num_segments = min(max_accel_segments, accel_segments.shape[0])

        print(f"\nPair {pair_idx + 1}/{num_pairs}:")
        print(f"  Serial: {serial}")
        print(f"  Time difference: {time_diff:.2f} seconds")
        print(f"  Typing datetime: {pair['typing_datetime']}")
        print(f"  Accel datetime: {pair['accel_datetime']}")
        print(f"  Total accel segments: {accel_segments.shape[0]}")
        print(f"  Showing {num_segments} segments")

        # Create figure with subplots: 1 for typing histogram + N for accel segments
        fig = plt.figure(figsize=(15, 3 * (num_segments + 1)))
        gs = fig.add_gridspec(num_segments + 1, 1, hspace=0.4)

        # Plot typing histogram at top
        ax_typing = fig.add_subplot(gs[0, 0])

        # Split typing features into HT and FT
        ht_features = typing_features[:101]  # Hold Time (101 bins)
        ft_features = typing_features[101:]  # Flight Time (401 bins)

        x_axis = np.arange(502)
        ax_typing.plot(x_axis, typing_features, "b-", linewidth=1, alpha=0.7)
        ax_typing.axvline(
            x=100.5,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label="HT/FT boundary",
        )

        # Highlight overflow bins
        ax_typing.scatter(
            [100],
            [ht_features[-1]],
            color="orange",
            s=100,
            marker="o",
            zorder=5,
            label="HT overflow",
        )
        ax_typing.scatter(
            [501],
            [ft_features[-1]],
            color="purple",
            s=100,
            marker="o",
            zorder=5,
            label="FT overflow",
        )

        ax_typing.set_title(
            f"Pair {pair_idx + 1} - Serial: {serial} | Time diff: {time_diff:.2f}s\n"
            f"Typing Histogram (502 features)",
            fontsize=12,
            fontweight="bold",
        )
        ax_typing.set_xlabel(
            "Feature Index (0-100: Hold Time, 101-501: Flight Time)", fontsize=10
        )
        ax_typing.set_ylabel("Normalized Frequency", fontsize=10)
        ax_typing.grid(True, alpha=0.3)
        ax_typing.legend(loc="upper right", fontsize=8)

        # Add annotations
        ax_typing.text(
            50,
            max(typing_features) * 0.95,
            "Hold Time\n(101 bins)",
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax_typing.text(
            300,
            max(typing_features) * 0.95,
            "Flight Time\n(401 bins)",
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )

        # Plot accelerometer segments below
        for seg_idx in range(num_segments):
            ax_accel = fig.add_subplot(gs[seg_idx + 1, 0])

            segment = accel_segments[seg_idx]  # Shape: (1000, 3)
            time = np.linspace(0, 10, 1000)

            # Plot each channel
            ax_accel.plot(
                time, segment[:, 0], "r-", label="Accel X", alpha=0.7, linewidth=0.8
            )
            ax_accel.plot(
                time, segment[:, 1], "g-", label="Accel Y", alpha=0.7, linewidth=0.8
            )
            ax_accel.plot(
                time, segment[:, 2], "b-", label="Accel Z", alpha=0.7, linewidth=0.8
            )

            ax_accel.set_xlabel("Time (seconds)", fontsize=10)
            ax_accel.set_ylabel("Acceleration (m/s²)", fontsize=10)
            ax_accel.set_title(
                f"Accelerometer Segment {seg_idx + 1}/{accel_segments.shape[0]} "
                f"(10s window, 100Hz)",
                fontsize=11,
            )
            ax_accel.legend(loc="upper right", fontsize=8)
            ax_accel.grid(True, alpha=0.3)

            # Calculate and display energy
            energy = (
                np.sum(segment[:, 0] ** 2)
                + np.sum(segment[:, 1] ** 2)
                + np.sum(segment[:, 2] ** 2)
            ) / 1000
            ax_accel.text(
                0.02,
                0.98,
                f"Energy: {energy:.3f} (m/s²)²",
                transform=ax_accel.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        plt.tight_layout()

        # Save figure
        output_path = f"../results/bimodal_pair_{pair_idx + 1}_{serial}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved to {output_path}")
        plt.close()

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import os
    import sys

    # Create bimodal dataset with incremental saving
    # Start with a small number of batches for testing
    # For full dataset: set max_batches=None
    output_path = "../data/bimodal_dataset.pickle"

    # Check if dataset already exists
    if os.path.exists(output_path):
        print("=" * 80)
        print("LOADING EXISTING BIMODAL DATASET")
        print("=" * 80)
        print(f"Found existing dataset at {output_path}")

        # Ask user whether to load or regenerate
        print("\nOptions:")
        print("  1. Load existing dataset (show summary & visualizations)")
        print("  2. Continue processing (resume from checkpoint)")
        print("  3. Delete and start fresh")

        choice = input("\nEnter choice (1/2/3) [default=1]: ").strip() or "1"

        if choice == "1":
            print("\nLoading existing dataset...")
            with open(output_path, "rb") as f:
                checkpoint_data = pickle.load(f)
            print("✓ Dataset loaded successfully")

        elif choice == "2":
            print("\nResuming processing from checkpoint...")
            checkpoint_data = find_temporal_matches(
                max_batches=None,
                batch_size=1000,
                time_tolerance_seconds=5,
                output_filepath=output_path,
                save_every_n_batches=5,
            )

        elif choice == "3":
            print(f"\nDeleting {output_path}...")
            os.remove(output_path)
            print("Starting fresh dataset creation...")
            checkpoint_data = find_temporal_matches(
                max_batches=None,
                batch_size=1000,
                time_tolerance_seconds=5,
                output_filepath=output_path,
                save_every_n_batches=5,
            )
        else:
            print("Invalid choice. Exiting.")
            sys.exit(1)
    else:
        print("=" * 80)
        print("CREATING NEW BIMODAL DATASET")
        print("=" * 80)
        checkpoint_data = find_temporal_matches(
            max_batches=None,
            batch_size=1000,
            time_tolerance_seconds=5,  # 5 seconds tolerance
            output_filepath=output_path,
            save_every_n_batches=5,  # Save every 5 batches to avoid memory issues
        )

    # Print summary (handles both dict and list formats)
    print_dataset_summary(checkpoint_data)

    # Extract matched pairs for visualization
    if isinstance(checkpoint_data, dict):
        matched_pairs = checkpoint_data.get("matched_pairs", [])
    else:
        matched_pairs = checkpoint_data

    # Visualize sample pairs (randomly selected each time)
    if len(matched_pairs) > 0:
        import random

        # Randomly select pairs to visualize
        num_pairs_to_show = min(3, len(matched_pairs))
        if len(matched_pairs) > num_pairs_to_show:
            random_indices = random.sample(range(len(matched_pairs)), num_pairs_to_show)
            random_pairs = [matched_pairs[i] for i in sorted(random_indices)]
            print(f"\nRandomly selected pairs at indices: {sorted(random_indices)}")
        else:
            random_pairs = matched_pairs

        visualize_matched_pairs(
            random_pairs, num_pairs=num_pairs_to_show, max_accel_segments=3
        )

    print("\n" + "=" * 80)
    print("BIMODAL DATASET PROCESSING COMPLETE")
    print(f"Dataset saved to: {output_path}")
    print("=" * 80)
