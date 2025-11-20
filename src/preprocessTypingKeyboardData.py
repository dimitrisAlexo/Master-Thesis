from pymongo import MongoClient
import numpy as np
import pickle
from tqdm import tqdm

# Configuration parameters
MIN_KEYSTROKES = 40  # Minimum keystrokes per session
HT_BINS = 100  # Number of bins for Hold Time histogram (0 to 1s in 10ms bins)
FT_BINS = 400  # Number of bins for Flight Time histogram (0 to 4s in 10ms bins)
HT_RANGE = (0, 1.0)  # Hold Time range in seconds [0, 1]s
FT_RANGE = (0, 4.0)  # Flight Time range in seconds [0, 4]s
HT_BIN_WIDTH = 0.01  # 10ms bins for Hold Time
FT_BIN_WIDTH = 0.01  # 10ms bins for Flight Time
# Total feature dimensions: 100 + 1 (overflow) + 400 + 1 (overflow) = 502


def compute_hold_time(down_time, up_time):
    """
    Compute Hold Time (HT) for each keystroke.
    HT = UpTime - DownTime (time key is pressed down)

    Args:
        down_time: List of key press timestamps (in milliseconds)
        up_time: List of key release timestamps (in milliseconds)

    Returns:
        numpy array of hold times in seconds
    """
    if len(down_time) != len(up_time):
        return None

    # Convert to seconds and compute hold time
    hold_times = (np.array(up_time) - np.array(down_time)) / 1000.0
    return hold_times


def compute_flight_time(down_time):
    """
    Compute Flight Time (FT) between consecutive keystrokes.
    FT = time between release of one key and press of next key
    FT[i] = DownTime[i+1] - DownTime[i] (approximately, using down times)

    Args:
        down_time: List of key press timestamps (in milliseconds), must be sorted

    Returns:
        numpy array of flight times in seconds
    """
    # Sort timestamps first
    down_time_sorted = np.sort(down_time)

    # Compute differences between consecutive keystrokes
    flight_times = np.diff(down_time_sorted) / 1000.0
    return flight_times


def create_histogram_features(hold_times, flight_times):
    """
    Create normalized histograms for hold times and flight times.
    Includes overflow bins for values beyond the range.

    Args:
        hold_times: numpy array of hold times in seconds
        flight_times: numpy array of flight times in seconds

    Returns:
        Concatenated feature vector of normalized histograms (502 dimensions)
    """
    # Create bins: [0, 0.01, 0.02, ..., 1.0] for HT and [0, 0.01, 0.02, ..., 4.0] for FT
    ht_bins = np.arange(HT_RANGE[0], HT_RANGE[1] + HT_BIN_WIDTH, HT_BIN_WIDTH)
    ft_bins = np.arange(FT_RANGE[0], FT_RANGE[1] + FT_BIN_WIDTH, FT_BIN_WIDTH)

    # Ensure we have exactly the right number of bins
    ht_bins = ht_bins[: HT_BINS + 1]
    ft_bins = ft_bins[: FT_BINS + 1]

    # Compute histograms (values within range)
    ht_hist, _ = np.histogram(hold_times, bins=ht_bins)
    ft_hist, _ = np.histogram(flight_times, bins=ft_bins)

    # Count overflow values (beyond the range)
    ht_overflow = np.sum(hold_times > HT_RANGE[1])
    ft_overflow = np.sum(flight_times > FT_RANGE[1])

    # Append overflow bins
    ht_hist_with_overflow = np.append(ht_hist, ht_overflow)
    ft_hist_with_overflow = np.append(ft_hist, ft_overflow)

    # Normalize histograms
    ht_hist_norm = (
        ht_hist_with_overflow / np.sum(ht_hist_with_overflow)
        if np.sum(ht_hist_with_overflow) > 0
        else ht_hist_with_overflow
    )
    ft_hist_norm = (
        ft_hist_with_overflow / np.sum(ft_hist_with_overflow)
        if np.sum(ft_hist_with_overflow) > 0
        else ft_hist_with_overflow
    )

    # Concatenate to form single feature vector: 101 + 401 = 502 dimensions
    feature_vector = np.concatenate([ht_hist_norm, ft_hist_norm])

    return feature_vector


def process_typing_session(document):
    """
    Process a single typing session document to extract features.

    Args:
        document: MongoDB document containing typing session data

    Returns:
        Feature vector if session is valid, None otherwise
    """
    if "payload" not in document:
        return None

    payload = document["payload"]
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

    # Filter out only negative values (keep values beyond range for overflow bin)
    hold_times = hold_times[hold_times >= 0]
    flight_times = flight_times[flight_times >= 0]

    # Check if we still have enough data after filtering
    if len(hold_times) < MIN_KEYSTROKES or len(flight_times) < (MIN_KEYSTROKES - 1):
        return None

    # Create histogram features
    feature_vector = create_histogram_features(hold_times, flight_times)

    return feature_vector


def create_typing_dynamics_bags():
    """
    Create typing dynamics bags for all subjects in the database.
    Each bag contains all valid typing sessions for a subject.

    Returns:
        List of dictionaries, one per subject with structure:
        [{'serial': serial_number, 'data': bag_of_sessions}, ...]
    """
    client = MongoClient("mongodb://localhost:27017/")
    db = client["iprog"]
    col = db["gdata_keyboard_raw"]

    print("=" * 80)
    print("CREATING TYPING DYNAMICS BAGS")
    print("=" * 80)
    print(f"Minimum keystrokes per session: {MIN_KEYSTROKES}")
    print(f"Minimum valid sessions per subject: 5")
    print(f"Hold Time bins: {HT_BINS} + 1 overflow (range: {HT_RANGE}s)")
    print(f"Flight Time bins: {FT_BINS} + 1 overflow (range: {FT_RANGE}s)")
    print(f"Total feature dimensions per session: {HT_BINS + 1 + FT_BINS + 1} = 502")
    print()

    # Get all unique serials
    serials = col.distinct("serial")
    print(f"Total unique subjects (serials): {len(serials)}")
    print()

    # List to store subject dictionaries
    typing_bags_list = []

    # Process each subject
    for serial in tqdm(serials, desc="Processing subjects"):
        # Get all documents for this serial
        docs = list(col.find({"serial": serial}))

        # Process each session
        session_features = []
        for doc in docs:
            features = process_typing_session(doc)
            if features is not None:
                session_features.append(features)

        # Store the bag only if it has at least 5 valid sessions
        if len(session_features) >= 5:
            subject_dict = {"serial": serial, "data": np.array(session_features)}
            typing_bags_list.append(subject_dict)

    print()
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Subjects with valid data: {len(typing_bags_list)}")

    # Calculate statistics
    bag_sizes = [len(subject["data"]) for subject in typing_bags_list]
    if bag_sizes:
        print(f"\nSessions per subject:")
        print(f"  Min: {np.min(bag_sizes)}")
        print(f"  Max: {np.max(bag_sizes)}")
        print(f"  Mean: {np.mean(bag_sizes):.2f}")
        print(f"  Median: {np.median(bag_sizes):.2f}")
        print(f"  Total sessions: {np.sum(bag_sizes)}")

    return typing_bags_list


def save_typing_bags(typing_bags, output_path="../data/typing_dynamics_bags.pickle"):
    """
    Save the typing dynamics bags to a pickle file.

    Args:
        typing_bags: Dictionary of typing bags
        output_path: Path to save the pickle file
    """
    print()
    print(f"Saving typing bags to: {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(typing_bags, f)
    print("Done!")


def visualize_histograms(typing_bags, subject_idx=0, num_histograms=10):
    """
    Visualize typing histograms for a specific subject.

    Args:
        typing_bags: List of subject dictionaries
        subject_idx: Index of subject to visualize
        num_histograms: Number of histograms to plot (default: 10)
    """
    import matplotlib.pyplot as plt

    if subject_idx >= len(typing_bags):
        print(f"Subject index {subject_idx} out of range. Max: {len(typing_bags)-1}")
        return

    subject = typing_bags[subject_idx]
    serial = subject["serial"]
    data = subject["data"]

    num_to_plot = min(num_histograms, len(data))

    # Create figure with subplots
    fig, axes = plt.subplots(num_to_plot, 1, figsize=(15, 3 * num_to_plot))
    if num_to_plot == 1:
        axes = [axes]

    fig.suptitle(
        f"Subject {serial} - Typing Histograms ({len(data)} total sessions)",
        fontsize=16,
        fontweight="bold",
    )

    for i in range(num_to_plot):
        ax = axes[i]
        histogram = data[i]

        # Split into hold time and flight time parts
        ht_part = histogram[:101]  # First 101 values (including overflow)
        ft_part = histogram[101:]  # Remaining 401 values (including overflow)

        # Create x-axis
        x_axis = np.arange(502)

        # Plot the histogram
        ax.plot(x_axis, histogram, "b-", linewidth=1, alpha=0.7, label="Full histogram")

        # Add vertical line to separate hold time from flight time
        ax.axvline(
            x=100.5,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label="HT/FT boundary",
        )

        # Highlight overflow bins
        ax.scatter(
            [100],
            [ht_part[-1]],
            color="orange",
            s=100,
            marker="o",
            zorder=5,
            label="HT overflow",
        )
        ax.scatter(
            [501],
            [ft_part[-1]],
            color="purple",
            s=100,
            marker="o",
            zorder=5,
            label="FT overflow",
        )

        ax.set_title(f"Session {i+1}/{len(data)}", fontsize=12, fontweight="bold")
        ax.set_xlabel(
            "Feature Index (0-100: Hold Time, 101-501: Flight Time)", fontsize=10
        )
        ax.set_ylabel("Normalized Frequency", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

        # Add text annotations
        ax.text(
            50,
            max(histogram) * 0.95,
            "Hold Time\n(101 bins)",
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax.text(
            300,
            max(histogram) * 0.95,
            "Flight Time\n(401 bins)",
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )

    plt.tight_layout()

    # Save figure
    output_path = f"typing_histograms_subject_{subject_idx}_{serial}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to: {output_path}")
    plt.close()


def print_dimensions_summary(typing_bags):
    """
    Print a clear summary of dataset dimensions.

    Args:
        typing_bags: List of subject dictionaries
    """
    print()
    print("=" * 80)
    print("DATASET DIMENSIONS SUMMARY")
    print("=" * 80)

    # Dataset level
    print(f"\n📊 DATASET LEVEL:")
    print(f"   • Total number of subjects: {len(typing_bags)}")
    print(f"   • Data structure: List of {len(typing_bags)} dictionaries")
    print(
        f"   • Each dictionary has keys: {list(typing_bags[0].keys()) if typing_bags else []}"
    )

    # Bag level statistics
    if typing_bags:
        bag_sizes = [len(subject["data"]) for subject in typing_bags]
        print(f"\n🎒 BAG LEVEL (per subject):")
        print(f"   • Minimum sessions per bag: {np.min(bag_sizes)}")
        print(f"   • Maximum sessions per bag: {np.max(bag_sizes)}")
        print(f"   • Average sessions per bag: {np.mean(bag_sizes):.2f}")
        print(f"   • Median sessions per bag: {np.median(bag_sizes):.2f}")
        print(f"   • Total sessions across all bags: {np.sum(bag_sizes)}")

        # Show examples
        print(f"\n   Examples:")
        for i in range(min(5, len(typing_bags))):
            subject = typing_bags[i]
            print(
                f"   • Subject {i} (serial: {subject['serial']}): "
                f"bag shape = {subject['data'].shape}"
            )

        # Histogram level
        example_histogram = typing_bags[0]["data"][0]
        print(f"\n📈 HISTOGRAM LEVEL (per session):")
        print(f"   • Histogram shape: {example_histogram.shape}")
        print(f"   • Total features per histogram: {len(example_histogram)}")
        print(f"   • Hold Time features: 101 (bins 0-100, last is overflow)")
        print(f"   • Flight Time features: 401 (bins 101-501, last is overflow)")
        print(f"   • Feature split: HT bins [0:101] | FT bins [101:502]")

        # Verify normalization
        ht_sum = np.sum(example_histogram[:101])
        ft_sum = np.sum(example_histogram[101:])
        total_sum = np.sum(example_histogram)
        print(f"\n   Normalization check (first histogram of first subject):")
        print(f"   • Hold Time histogram sum: {ht_sum:.6f} (should be ~1.0)")
        print(f"   • Flight Time histogram sum: {ft_sum:.6f} (should be ~1.0)")
        print(f"   • Total sum: {total_sum:.6f} (should be ~2.0)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Create typing dynamics bags
    typing_bags = create_typing_dynamics_bags()

    # Save to file
    save_typing_bags(typing_bags)

    # Show example
    print()
    print("=" * 80)
    print("EXAMPLE DATA")
    print("=" * 80)
    if typing_bags:
        example_subject = typing_bags[0]
        example_serial = example_subject["serial"]
        example_bag = example_subject["data"]
        print(f"Example subject serial: {example_serial}")
        print(f"Number of sessions: {len(example_bag)}")
        print(f"Feature vector shape per session: {example_bag[0].shape}")
        print(f"First session features (first 10 values):")
        print(example_bag[0][:10])
        print(f"Last 5 values (should include overflow bins):")
        print(example_bag[0][-5:])
        print()
        print(f"Bag shape: {example_bag.shape}")
        print(f"\nData structure: List of {len(typing_bags)} dictionaries")
        print(f"Each dict has keys: {list(example_subject.keys())}")
    print("=" * 80)

    # Print dimensions summary
    print_dimensions_summary(typing_bags)

    # Visualize histograms for multiple subjects
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    # Visualize first 3 subjects with their first 5 sessions
    for idx in range(min(3, len(typing_bags))):
        print(f"\nVisualizing subject {idx}...")
        visualize_histograms(typing_bags, subject_idx=idx, num_histograms=5)

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
