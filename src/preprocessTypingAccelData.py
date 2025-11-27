from pymongo import MongoClient
import numpy as np
from tqdm import tqdm
from scipy import signal
import pickle
import matplotlib.pyplot as plt


def is_valid_recording(accel_x, accel_y, accel_z, accel_t):
    """
    Check if recording passes quality criteria.

    Reject if:
    - Duration < 20s
    - Sampling frequency < 50 Hz
    - Extreme acceleration values > 100 m/s^2
    - Too many missing values

    Returns:
        (is_valid, sampling_rate, duration, reason)
    """
    if len(accel_t) < 2:
        return False, 0, 0, "short_duration"

    # Calculate duration (timestamps are in nanoseconds)
    duration_sec = (accel_t[-1] - accel_t[0]) / 1_000_000_000

    # Check duration >= 20s
    if duration_sec < 20:
        return False, 0, duration_sec, "short_duration"

    # Calculate sampling rate
    sampling_rate = len(accel_t) / duration_sec

    # Check sampling rate >= 50 Hz
    if sampling_rate < 50:
        return False, sampling_rate, duration_sec, "low_sampling_rate"

    # Check for extreme values > 100 m/s^2
    max_accel = max(
        np.max(np.abs(accel_x)), np.max(np.abs(accel_y)), np.max(np.abs(accel_z))
    )
    if max_accel > 100:
        return False, sampling_rate, duration_sec, "extreme_values"

    # Check for too many missing values (if arrays have different lengths)
    if not (len(accel_x) == len(accel_y) == len(accel_z) == len(accel_t)):
        return False, sampling_rate, duration_sec, "mismatched_lengths"

    # Check for NaN or inf values
    if (
        np.any(np.isnan(accel_x))
        or np.any(np.isnan(accel_y))
        or np.any(np.isnan(accel_z))
        or np.any(np.isinf(accel_x))
        or np.any(np.isinf(accel_y))
        or np.any(np.isinf(accel_z))
    ):
        return False, sampling_rate, duration_sec, "nan_inf_values"

    return True, sampling_rate, duration_sec, None


def resample_signal(signal_data, timestamps, target_rate=100):
    """
    Resample signal to target rate using polyphase resampling.

    Args:
        signal_data: Array of signal values
        timestamps: Array of timestamps in microseconds
        target_rate: Target sampling rate in Hz

    Returns:
        Resampled signal array
    """
    # Calculate original sampling rate (timestamps in nanoseconds)
    duration_sec = (timestamps[-1] - timestamps[0]) / 1_000_000_000

    # Calculate resampling factor
    num_samples_target = int(duration_sec * target_rate)

    # Use scipy's resample for polyphase resampling
    resampled = signal.resample(signal_data, num_samples_target)

    return resampled


def remove_gravity(signal_data, sampling_rate=100, cutoff_freq=1.0, filter_order=512):
    """
    Remove gravitational component using high-pass FIR filter.

    Args:
        signal_data: Signal array
        sampling_rate: Sampling rate in Hz
        cutoff_freq: Cutoff frequency in Hz
        filter_order: FIR filter order

    Returns:
        Filtered signal
    """
    # Design high-pass FIR filter
    nyquist = sampling_rate / 2.0
    normalized_cutoff = cutoff_freq / nyquist

    # Use firwin to design the filter
    fir_coeff = signal.firwin(filter_order + 1, normalized_cutoff, pass_zero=False)

    # Apply filter
    filtered = signal.filtfilt(fir_coeff, 1.0, signal_data)

    return filtered


def segment_signal(accel_x, accel_y, accel_z, window_size=1000, energy_threshold=0.30):
    """
    Segment signal into non-overlapping windows and filter by energy.

    Args:
        accel_x, accel_y, accel_z: Filtered acceleration signals (already resampled to 100Hz)
        window_size: Window size in samples (1000 samples = 10s at 100Hz)
        energy_threshold: Minimum energy threshold in (m/s^2)^2

    Returns:
        List of segments, each segment is array of shape (window_size, 3)
    """
    segments = []

    # Calculate number of complete windows
    num_windows = len(accel_x) // window_size

    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size

        # Extract window
        x_window = accel_x[start_idx:end_idx]
        y_window = accel_y[start_idx:end_idx]
        z_window = accel_z[start_idx:end_idx]

        # Calculate energy across channels
        # Energy = sum of squared values across all channels
        energy = (
            np.sum(x_window**2) + np.sum(y_window**2) + np.sum(z_window**2)
        ) / window_size

        # Keep segment if energy is above threshold
        if energy >= energy_threshold:
            # Stack channels: shape (window_size, 3)
            segment = np.stack([x_window, y_window, z_window], axis=1)
            segments.append(segment)

    return segments


def process_accelerometer_recording(doc):
    """
    Process a single accelerometer recording document.

    Returns:
        (serial, segments, reason) where segments is list of numpy arrays (1000, 3)
        Returns (None, None, reason) if recording is invalid
    """
    serial = doc["serial"]
    payload = doc.get("payload", {})

    accel_x = np.array(payload.get("AccelX", []))
    accel_y = np.array(payload.get("AccelY", []))
    accel_z = np.array(payload.get("AccelZ", []))
    accel_t = np.array(payload.get("AccelT", []))

    # Check if arrays are empty
    if len(accel_x) == 0 or len(accel_t) == 0:
        return None, None, "empty_arrays"

    # Validate recording
    is_valid, sampling_rate, duration, reason = is_valid_recording(
        accel_x, accel_y, accel_z, accel_t
    )

    if not is_valid:
        return None, None, reason

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

    if len(segments) == 0:
        return None, None, "no_segments"

    return serial, segments, None


def create_accelerometer_bags(max_documents=None, batch_size=1000):
    """
    Create accelerometer bags dataset from Keyboard origin recordings.

    Args:
        max_documents: Maximum number of documents to process (None = process all)
        batch_size: Number of documents to fetch per batch

    Returns:
        List of dicts with structure: [{'serial': '...', 'data': np.array of shape (n_segments, 1000, 3)}, ...]
    """
    client = MongoClient("mongodb://localhost:27017/")
    db = client["iprog"]
    col = db["gdata_imu_raw"]

    print("=" * 80)
    print("CREATING ACCELEROMETER BAGS DATASET")
    print("=" * 80)

    # Determine how many to process
    if max_documents is not None:
        print(f"\nProcessing limit set to: {max_documents:,} documents")
    else:
        print(f"\nProcessing ALL Keyboard documents in database")

    # Process in batches
    print(f"Processing in batches of {batch_size:,} documents...")
    serial_segments_map = {}  # Map serial -> list of segments

    valid_count = 0
    invalid_count = 0
    processed_count = 0

    # Debug counters for rejection reasons
    rejection_reasons = {
        "empty_arrays": 0,
        "short_duration": 0,
        "low_sampling_rate": 0,
        "extreme_values": 0,
        "mismatched_lengths": 0,
        "nan_inf_values": 0,
        "no_segments": 0,
    }

    # Create cursor for batch processing
    cursor = col.find({"payload.Origin": "Keyboard"}).batch_size(batch_size)

    # Progress bar
    pbar = tqdm(desc="Processing recordings", unit=" docs")

    for doc in cursor:
        if max_documents is not None and processed_count >= max_documents:
            break

        serial, segments, reason = process_accelerometer_recording(doc)

        if serial is not None and segments is not None:
            valid_count += 1
            if serial not in serial_segments_map:
                serial_segments_map[serial] = []
            serial_segments_map[serial].extend(segments)
        else:
            invalid_count += 1
            if reason in rejection_reasons:
                rejection_reasons[reason] += 1

        processed_count += 1
        pbar.update(1)

    pbar.close()
    cursor.close()

    print(f"\n✓ Valid recordings: {valid_count:,}")
    print(f"✗ Invalid recordings: {invalid_count:,}")
    print(f"Total subjects: {len(serial_segments_map):,}")

    # Print rejection reasons
    if invalid_count > 0:
        print(f"\nRejection reasons:")
        for reason, count in rejection_reasons.items():
            if count > 0:
                percentage = (count / invalid_count) * 100
                print(f"  {reason}: {count:,} ({percentage:.1f}%)")

    # Create dataset structure
    dataset = []
    for serial, segments in serial_segments_map.items():
        # Stack segments into array of shape (n_segments, 1000, 3)
        data_array = np.stack(segments, axis=0)
        dataset.append({"serial": serial, "data": data_array})

    # Print statistics
    print(f"\nDataset statistics:")
    print(f"  Number of subjects: {len(dataset)}")
    total_segments = sum(item["data"].shape[0] for item in dataset)
    print(f"  Total segments: {total_segments}")

    if len(dataset) > 0:
        segments_per_subject = [item["data"].shape[0] for item in dataset]
        print(
            f"  Segments per subject - Min: {np.min(segments_per_subject)}, "
            f"Max: {np.max(segments_per_subject)}, Mean: {np.mean(segments_per_subject):.2f}"
        )
        print(f"  Segment shape: {dataset[0]['data'].shape[1:]} (samples, channels)")

    client.close()

    return dataset


def save_accelerometer_bags(dataset, filepath="../data/accelerometer_bags.pickle"):
    """Save accelerometer bags to pickle file."""
    print(f"\nSaving dataset to {filepath}...")
    with open(filepath, "wb") as f:
        pickle.dump(dataset, f)
    print(f"✓ Dataset saved successfully")


def visualize_acceleration_windows(dataset, num_windows=3):
    """
    Visualize sample acceleration windows from the dataset.

    Args:
        dataset: List of dicts with 'serial' and 'data' keys
        num_windows: Number of windows to visualize
    """
    if len(dataset) == 0:
        print("No data to visualize")
        return

    # Find a subject with at least num_windows segments
    selected_item = None
    for item in dataset:
        if item["data"].shape[0] >= num_windows:
            selected_item = item
            break

    if selected_item is None:
        selected_item = dataset[0]
        num_windows = min(num_windows, selected_item["data"].shape[0])

    serial = selected_item["serial"]
    data = selected_item["data"]

    print(f"\nVisualizing {num_windows} windows from subject: {serial}")

    # Create subplots
    fig, axes = plt.subplots(num_windows, 1, figsize=(12, 4 * num_windows))

    if num_windows == 1:
        axes = [axes]

    for i in range(num_windows):
        ax = axes[i]
        segment = data[i]
        time = np.linspace(0, 10, 1000)

        ax.plot(time, segment[:, 0], "r-", label="Accel X", alpha=0.7, linewidth=0.8)
        ax.plot(time, segment[:, 1], "g-", label="Accel Y", alpha=0.7, linewidth=0.8)
        ax.plot(time, segment[:, 2], "b-", label="Accel Z", alpha=0.7, linewidth=0.8)

        ax.set_xlabel("Time (seconds)", fontsize=10)
        ax.set_ylabel("Acceleration (m/s²)", fontsize=10)
        ax.set_title(f"Window {i+1}/{num_windows} - Serial: {serial}", fontsize=11)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = "../results/accelerometer_windows_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Visualization saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    # Create accelerometer bags dataset
    # For debugging: set max_documents to limit processing (e.g., 1000, 5000)
    # For full dataset: set max_documents=None
    dataset = create_accelerometer_bags(max_documents=1000, batch_size=1000)

    # Save to pickle
    save_accelerometer_bags(dataset)

    # Visualize sample windows
    visualize_acceleration_windows(dataset, num_windows=3)
