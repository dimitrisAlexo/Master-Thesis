"""
Bimodal Dataset Labeling Tool

This script helps manually label bimodal pairs (typing + accelerometer) for
supervised training. It visualizes pairs and allows you to assign labels (0/1)
to create a labeled dataset.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_bimodal_dataset(filepath="../data/bimodal_dataset.pickle"):
    """
    Load the bimodal dataset from pickle file and unpack into individual pairs.
    Each typing histogram + accelerometer window combination becomes a separate pair.

    Args:
        filepath: Path to the bimodal dataset pickle file

    Returns:
        List of unpacked pairs (one histogram, one window per pair)
    """
    print(f"Loading bimodal dataset from {filepath}...")
    with open(filepath, "rb") as f:
        checkpoint_data = pickle.load(f)

    # Handle both old format (list) and new format (dict with metadata)
    if isinstance(checkpoint_data, dict):
        matched_pairs = checkpoint_data.get("matched_pairs", [])
        print(f"✓ Loaded {len(matched_pairs)} matched pairs from checkpoint")
    else:
        matched_pairs = checkpoint_data
        print(f"✓ Loaded {len(matched_pairs)} matched pairs")

    # Unpack pairs: one histogram can have multiple windows
    # Create separate pair for each (histogram, window) combination
    unpacked_pairs = []
    for pair_idx, pair in enumerate(matched_pairs):
        typing_features = pair["typing_features"]  # Shape: (502,)
        accel_segments = pair["accel_segments"]  # Shape: (n_segments, 1000, 3)

        # Create one pair for each accelerometer segment
        for seg_idx, segment in enumerate(accel_segments):
            unpacked_pair = {
                "original_pair_index": pair_idx,
                "segment_index": seg_idx,
                "serial": pair["serial"],
                "typing_features": typing_features,
                "accel_segment": segment,  # Shape: (1000, 3) - single window
                "typing_datetime": pair["typing_datetime"],
                "accel_datetime": pair["accel_datetime"],
                "time_diff_seconds": pair["time_diff_seconds"],
            }
            unpacked_pairs.append(unpacked_pair)

    print(
        f"✓ Unpacked into {len(unpacked_pairs)} individual pairs (histogram + single window)"
    )
    return unpacked_pairs


def visualize_pairs_batch(pairs_with_indices, batch_size=5):
    """
    Visualize multiple bimodal pairs in a single figure for faster labeling.

    Args:
        pairs_with_indices: List of tuples (pair_dict, pair_index)
        batch_size: Number of pairs to show at once (default: 5)
    """
    num_pairs = len(pairs_with_indices)

    # Create figure with rows: each pair gets 1 row with 2 columns (histogram | accel)
    fig = plt.figure(figsize=(20, 2 * num_pairs))
    gs = fig.add_gridspec(num_pairs, 2, hspace=0.25, wspace=0.25)

    for i, (pair, pair_index) in enumerate(pairs_with_indices):
        serial = pair["serial"]
        typing_features = pair["typing_features"]
        accel_segment = pair["accel_segment"]
        time_diff = pair["time_diff_seconds"]
        original_idx = pair.get("original_pair_index", "N/A")
        segment_idx = pair.get("segment_index", 0)

        # Plot typing histogram (left column)
        ax_typing = fig.add_subplot(gs[i, 0])

        ht_features = typing_features[:101]
        ft_features = typing_features[101:]

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

        ax_typing.scatter(
            [100],
            [ht_features[-1]],
            color="orange",
            s=50,
            marker="o",
            zorder=5,
            label="HT",
        )
        ax_typing.scatter(
            [501],
            [ft_features[-1]],
            color="purple",
            s=50,
            marker="o",
            zorder=5,
            label="FT",
        )

        ax_typing.set_title(
            f"PAIR {i+1}/{num_pairs} - IDX {pair_index} | {serial}",
            fontsize=9,
            fontweight="bold",
            color="darkblue",
        )
        ax_typing.set_xlabel("Feature", fontsize=7)
        ax_typing.set_ylabel("Freq", fontsize=7)
        ax_typing.grid(True, alpha=0.2)
        ax_typing.legend(loc="upper right", fontsize=5, ncol=2)
        ax_typing.tick_params(axis="both", which="major", labelsize=6)

        # Plot accelerometer segment (right column)
        ax_accel = fig.add_subplot(gs[i, 1])

        time = np.linspace(0, 10, 1000)
        ax_accel.plot(
            time, accel_segment[:, 0], "r-", label="X", alpha=0.7, linewidth=0.7
        )
        ax_accel.plot(
            time, accel_segment[:, 1], "g-", label="Y", alpha=0.7, linewidth=0.7
        )
        ax_accel.plot(
            time, accel_segment[:, 2], "b-", label="Z", alpha=0.7, linewidth=0.7
        )

        ax_accel.set_xlabel("Time (s)", fontsize=7)
        ax_accel.set_ylabel("Accel", fontsize=7)
        ax_accel.legend(loc="upper right", fontsize=5, ncol=3)
        ax_accel.grid(True, alpha=0.2)
        ax_accel.tick_params(axis="both", which="major", labelsize=6)

        # Calculate and display energy
        energy = (
            np.sum(accel_segment[:, 0] ** 2)
            + np.sum(accel_segment[:, 1] ** 2)
            + np.sum(accel_segment[:, 2] ** 2)
        ) / 1000
        ax_accel.text(
            0.02,
            0.98,
            f"E:{energy:.2f}",
            transform=ax_accel.transAxes,
            fontsize=6,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
        )

    plt.tight_layout()
    plt.show(block=True)
    plt.close()


def visualize_pair(pair, pair_index):
    """
    Visualize a single bimodal pair showing typing histogram and ONE accelerometer segment.

    Args:
        pair: Dictionary containing matched pair data (unpacked format)
        pair_index: Index of the pair in the dataset
    """
    serial = pair["serial"]
    typing_features = pair["typing_features"]
    accel_segment = pair["accel_segment"]  # Single window: (1000, 3)
    time_diff = pair["time_diff_seconds"]
    original_idx = pair.get("original_pair_index", "N/A")
    segment_idx = pair.get("segment_index", 0)

    # Create figure with 2 subplots: 1 for typing histogram + 1 for accel segment
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(2, 1, hspace=0.4)

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
        f"PAIR INDEX {pair_index} (Original: {original_idx}, Segment: {segment_idx}) - Serial: {serial} | Time diff: {time_diff:.2f}s\n"
        f"Typing Histogram (502 features)",
        fontsize=14,
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

    # Plot single accelerometer segment
    ax_accel = fig.add_subplot(gs[1, 0])

    time = np.linspace(0, 10, 1000)

    # Plot each channel
    ax_accel.plot(
        time, accel_segment[:, 0], "r-", label="Accel X", alpha=0.7, linewidth=0.8
    )
    ax_accel.plot(
        time, accel_segment[:, 1], "g-", label="Accel Y", alpha=0.7, linewidth=0.8
    )
    ax_accel.plot(
        time, accel_segment[:, 2], "b-", label="Accel Z", alpha=0.7, linewidth=0.8
    )

    ax_accel.set_xlabel("Time (seconds)", fontsize=10)
    ax_accel.set_ylabel("Acceleration (m/s²)", fontsize=10)
    ax_accel.set_title(
        f"Accelerometer Window (10s window, 100Hz)",
        fontsize=11,
    )
    ax_accel.legend(loc="upper right", fontsize=8)
    ax_accel.grid(True, alpha=0.3)

    # Calculate and display energy
    energy = (
        np.sum(accel_segment[:, 0] ** 2)
        + np.sum(accel_segment[:, 1] ** 2)
        + np.sum(accel_segment[:, 2] ** 2)
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
    plt.show(block=True)
    plt.close()


def visualize_pairs_for_labeling(matched_pairs, pair_indices):
    """
    Visualize multiple pairs one by one for manual labeling.

    Args:
        matched_pairs: List of all matched pairs
        pair_indices: List of indices to visualize
    """
    print("\n" + "=" * 80)
    print("VISUALIZING PAIRS FOR LABELING")
    print("=" * 80)
    print(f"Total pairs to visualize: {len(pair_indices)}")
    print("Close each plot window to move to the next pair")
    print("=" * 80 + "\n")

    for i, idx in enumerate(pair_indices, 1):
        if idx >= len(matched_pairs):
            print(
                f"Warning: Index {idx} out of range (dataset has {len(matched_pairs)} pairs)"
            )
            continue

        print(f"\nShowing pair {i}/{len(pair_indices)} (Index: {idx})")
        pair = matched_pairs[idx]
        print(f"  Serial: {pair['serial']}")
        print(f"  Typing datetime: {pair['typing_datetime']}")
        print(f"  Accel datetime: {pair['accel_datetime']}")
        print(f"  Time difference: {pair['time_diff_seconds']:.2f} seconds")
        print(f"  Original pair index: {pair.get('original_pair_index', 'N/A')}")
        print(f"  Segment index: {pair.get('segment_index', 0)}")

        visualize_pair(pair, idx)


def create_labeled_dataset(
    matched_pairs, labels_dict, output_filepath="../data/labeled_bimodal_dataset.pickle"
):
    """
    Create a labeled bimodal dataset from selected pairs and their labels.

    Args:
        matched_pairs: List of all matched pairs
        labels_dict: Dictionary mapping pair indices to labels {index: label}
                    where label is 0 or 1
        output_filepath: Path to save the labeled dataset

    Returns:
        Dictionary containing the labeled dataset
    """
    print("\n" + "=" * 80)
    print("CREATING LABELED BIMODAL DATASET")
    print("=" * 80)

    labeled_data = []

    for pair_idx, label in labels_dict.items():
        if pair_idx >= len(matched_pairs):
            print(f"Warning: Index {pair_idx} out of range, skipping")
            continue

        if label not in [0, 1]:
            print(f"Warning: Invalid label {label} for index {pair_idx}, skipping")
            continue

        pair = matched_pairs[pair_idx]

        # Create labeled entry (already unpacked format)
        labeled_entry = {
            "pair_index": pair_idx,
            "original_pair_index": pair.get("original_pair_index", "N/A"),
            "segment_index": pair.get("segment_index", 0),
            "serial": pair["serial"],
            "typing_features": pair["typing_features"],  # Shape: (502,)
            "accel_segment": pair["accel_segment"],  # Shape: (1000, 3) - single window
            "label": label,  # 0 or 1
            "typing_datetime": pair["typing_datetime"],
            "accel_datetime": pair["accel_datetime"],
            "time_diff_seconds": pair["time_diff_seconds"],
        }

        labeled_data.append(labeled_entry)

    # Create dataset dictionary
    labeled_dataset = {
        "data": labeled_data,
        "total_pairs": len(labeled_data),
        "label_distribution": {
            0: sum(1 for entry in labeled_data if entry["label"] == 0),
            1: sum(1 for entry in labeled_data if entry["label"] == 1),
        },
        "source_dataset_size": len(matched_pairs),
    }

    # Save to pickle file
    print(f"\nSaving labeled dataset to {output_filepath}...")
    with open(output_filepath, "wb") as f:
        pickle.dump(labeled_dataset, f)

    print(f"✓ Labeled dataset saved successfully")
    print(f"\nDataset Summary:")
    print(f"  Total labeled pairs: {labeled_dataset['total_pairs']}")
    print(f"  Label 0 (No tremor/FMI): {labeled_dataset['label_distribution'][0]}")
    print(f"  Label 1 (Tremor/FMI): {labeled_dataset['label_distribution'][1]}")
    print(f"  Source dataset size: {labeled_dataset['source_dataset_size']}")

    print("\n" + "=" * 80)

    return labeled_dataset


def load_labeled_dataset(filepath="../data/labeled_bimodal_dataset.pickle"):
    """
    Load a labeled bimodal dataset.

    Args:
        filepath: Path to the labeled dataset pickle file

    Returns:
        Dictionary containing the labeled dataset
    """
    print(f"Loading labeled dataset from {filepath}...")
    with open(filepath, "rb") as f:
        labeled_dataset = pickle.load(f)

    print(f"✓ Loaded {labeled_dataset['total_pairs']} labeled pairs")
    print(f"  Label 0: {labeled_dataset['label_distribution'][0]}")
    print(f"  Label 1: {labeled_dataset['label_distribution'][1]}")

    return labeled_dataset


# Example usage functions
def example_workflow():
    """
    Example workflow for labeling bimodal pairs.
    """
    print("\n" + "=" * 80)
    print("BIMODAL DATASET LABELING - EXAMPLE WORKFLOW")
    print("=" * 80)

    # Step 1: Load the bimodal dataset
    matched_pairs = load_bimodal_dataset()

    print(f"\nDataset contains {len(matched_pairs)} pairs")
    print("You can now:")
    print("  1. Visualize specific pairs by index")
    print("  2. Create labels for pairs")
    print("  3. Generate a labeled dataset")

    return matched_pairs


def interactive_labeling_session(
    matched_pairs,
    start_idx=0,
    num_pairs=100,
    checkpoint_filepath="../src/labeling_checkpoint.pickle",
):
    """
    Interactive labeling session where you can label pairs one by one.
    Automatically saves progress and resumes from where you left off.

    Args:
        matched_pairs: List of all matched pairs
        start_idx: Starting index for labeling (will be overridden if checkpoint exists)
        num_pairs: Number of pairs to label
        checkpoint_filepath: Path to save labeling progress

    Returns:
        Dictionary of labels {index: label}
    """
    import os

    # Load existing labels if checkpoint exists
    labels_dict = {}
    if os.path.exists(checkpoint_filepath):
        print(f"Found existing labeling checkpoint at {checkpoint_filepath}")
        with open(checkpoint_filepath, "rb") as f:
            checkpoint_data = pickle.load(f)
        labels_dict = checkpoint_data.get("labels", {})
        start_idx = checkpoint_data.get("next_index", start_idx)
        print(f"✓ Loaded {len(labels_dict)} existing labels")
        print(f"✓ Resuming from index {start_idx}")

    print("\n" + "=" * 80)
    print("INTERACTIVE LABELING SESSION")
    print("=" * 80)
    print(f"Target: Label {num_pairs} pairs total")
    print(f"Progress: {len(labels_dict)}/{num_pairs} already labeled")
    print(f"Starting from index: {start_idx}")
    print("\nInstructions:")
    print("  - Each pair will be displayed")
    print("  - Enter 0 for No FMI/Tremor")
    print("  - Enter 1 for FMI/Tremor present")
    print("  - Enter 's' to skip")
    print("  - Enter 'q' to quit (progress will be saved)")
    print("=" * 80 + "\n")

    # Keep going until we have num_pairs labels (not num_pairs plotted)
    idx = start_idx
    batch_size = 5

    while len(labels_dict) < num_pairs and idx < len(matched_pairs):
        # Collect batch of pairs to visualize
        batch_pairs = []
        batch_indices = []

        for i in range(batch_size):
            if idx + i >= len(matched_pairs):
                break
            batch_pairs.append(matched_pairs[idx + i])
            batch_indices.append(idx + i)

        if not batch_pairs:
            break

        print(f"\n{'='*80}")
        print(f"Labeling progress: {len(labels_dict)}/{num_pairs}")
        print(f"Showing batch: indices {batch_indices[0]} to {batch_indices[-1]}")
        print(f"{'='*80}")

        # Visualize batch of pairs
        pairs_with_indices = [
            (batch_pairs[i], batch_indices[i]) for i in range(len(batch_pairs))
        ]
        visualize_pairs_batch(pairs_with_indices, batch_size=batch_size)

        # Label each pair in the batch
        for batch_idx in range(len(batch_pairs)):
            current_idx = batch_indices[batch_idx]
            pair = batch_pairs[batch_idx]

            print(
                f"\n--- Pair {batch_idx + 1}/{len(batch_pairs)} (Index: {current_idx}) ---"
            )
            print(f"Serial: {pair['serial']}")
            print(
                f"Original pair index: {pair.get('original_pair_index', 'N/A')}, Segment: {pair.get('segment_index', 0)}"
            )

            # Get label from user
            while True:
                label_input = (
                    input(f"Enter label for pair {current_idx} (0/1/s/q): ")
                    .strip()
                    .lower()
                )

                if label_input == "q":
                    print("\nQuitting labeling session...")
                    # Save checkpoint
                    checkpoint_data = {
                        "labels": labels_dict,
                        "next_index": current_idx,
                        "total_target": num_pairs,
                    }
                    with open(checkpoint_filepath, "wb") as f:
                        pickle.dump(checkpoint_data, f)
                    print(f"✓ Progress saved to {checkpoint_filepath}")
                    print(f"✓ Labeled {len(labels_dict)}/{num_pairs} pairs so far")
                    print(f"✓ Will resume from index {current_idx} next time")
                    return labels_dict
                elif label_input == "s":
                    print("Skipping this pair (not counted toward 100)")
                    break
                elif label_input in ["0", "1"]:
                    label = int(label_input)
                    labels_dict[current_idx] = label
                    print(f"✓ Labeled pair {current_idx} as {label}")
                    print(f"Progress: {len(labels_dict)}/{num_pairs}")

                    # Check if we've reached target
                    if len(labels_dict) >= num_pairs:
                        print(f"\n🎉 Target reached! Labeled {len(labels_dict)} pairs")
                        # Save final checkpoint immediately
                        checkpoint_data = {
                            "labels": labels_dict,
                            "next_index": current_idx + 1,
                            "total_target": num_pairs,
                            "completed": True,
                        }
                        with open(checkpoint_filepath, "wb") as f:
                            pickle.dump(checkpoint_data, f)
                        print(f"✓ Final progress saved to {checkpoint_filepath}")
                        return labels_dict
                    break
                else:
                    print("Invalid input. Please enter 0, 1, s, or q")

        # Move to next batch
        idx += batch_size

    print(f"\n{'='*80}")
    print("LABELING SESSION COMPLETE")
    print(f"{'='*80}")
    print(f"Total pairs labeled: {len(labels_dict)}/{num_pairs}")
    print(f"  Label 0: {sum(1 for v in labels_dict.values() if v == 0)}")
    print(f"  Label 1: {sum(1 for v in labels_dict.values() if v == 1)}")

    # Save final checkpoint
    checkpoint_data = {
        "labels": labels_dict,
        "next_index": idx,
        "total_target": num_pairs,
        "completed": len(labels_dict) >= num_pairs,
    }
    with open(checkpoint_filepath, "wb") as f:
        pickle.dump(checkpoint_data, f)
    print(f"✓ Final progress saved to {checkpoint_filepath}")

    return labels_dict


if __name__ == "__main__":
    # Example: Load dataset
    matched_pairs = load_bimodal_dataset()

    print("\n" + "=" * 80)
    print("USAGE OPTIONS")
    print("=" * 80)
    print("\n1. Visualize specific pairs:")
    print("   visualize_pairs_for_labeling(matched_pairs, [0, 10, 20, 30])")

    print("\n2. Interactive labeling session:")
    print(
        "   labels = interactive_labeling_session(matched_pairs, start_idx=0, num_pairs=100)"
    )

    print("\n3. Create labeled dataset from labels:")
    print("   labels_dict = {0: 1, 10: 0, 20: 1, 30: 0}  # Your labels")
    print("   labeled_dataset = create_labeled_dataset(matched_pairs, labels_dict)")

    print("\n4. Or provide pre-defined labels:")
    print("   # Example: Label 100 pairs")
    print("   labels = {")
    print("       0: 1, 1: 0, 2: 1, 3: 0, 4: 1,  # First 5 pairs")
    print("       # ... add more labels ...")
    print("   }")
    print("   labeled_dataset = create_labeled_dataset(matched_pairs, labels)")

    print("\n" + "=" * 80)

    # Uncomment to start interactive labeling
    labels = interactive_labeling_session(matched_pairs, start_idx=0, num_pairs=100)
    if labels:
        labeled_dataset = create_labeled_dataset(matched_pairs, labels)
