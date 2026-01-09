"""
Create additional_typing_sdataset.pickle from TypingData folder
Dataset: Keystroke timing and pressure data from PD patients (n=18) and healthy controls (n=15)
"""

import numpy as np
import pickle as pkl
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Base path to the TypingData folder
DATA_PATH = Path(__file__).parent.parent / "data" / "TypingData"

# Subject labels mapping
SUBJECT_LABELS = {
    "S01": "PD",
    "S02": "PD",
    "S03": "PD",
    "S04": "PD",
    "S05": "PD",
    "S06": "PD",
    "S07": "PD",
    "S08": "PD",
    "S09": "PD",
    "S10": "PD",
    "S11": "Control",
    "S12": "Control",
    "S13": "PD",
    "S14": "PD",
    "S15": "Control",
    "S16": "PD",
    "S17": "PD",
    "S18": "PD",
    "S19": "PD",
    "S20": "PD",
    "S21": "PD",
    "S22": "Control",
    "S23": "Control",
    "S24": "Control",
    "S25": "Control",
    "S26": "Control",
    "S27": "Control",
    "S28": "Control",
    "S29": "Control",
    "S30": "Control",
    "S31": "Control",
    "S32": "Control",
    "S33": "Control",
}

# Histogram parameters
HT_BINS = 100  # Hold Time histogram bins (0 to 1s in 10ms bins)
FT_BINS = 400  # Flight Time histogram bins (0 to 4s in 10ms bins)
HT_RANGE = (0, 1.0)  # Hold Time range in seconds
FT_RANGE = (0, 4.0)  # Flight Time range in seconds
HT_BIN_WIDTH = 0.01  # 10ms bins
FT_BIN_WIDTH = 0.01  # 10ms bins
MIN_KEYSTROKES = 40  # Minimum keystrokes per session
K2 = 500  # Number of typing sessions per bag (zero-padded if fewer)


def parse_typing_file(file_path):
    """Parse a typing data file and extract press/release timestamps."""
    try:
        press_times = []
        release_times = []
        pressures = []

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("Press"):
                    continue

                parts = line.split(",")
                if len(parts) < 4:
                    continue

                press_time = int(parts[1].strip())
                release_part = parts[2].strip()
                if release_part.startswith("Release"):
                    release_time = int(release_part.split()[1])
                else:
                    release_time = int(release_part)
                pressure = float(parts[3].strip())

                press_times.append(press_time)
                release_times.append(release_time)
                pressures.append(pressure)

        if len(press_times) < MIN_KEYSTROKES:
            return None

        return (np.array(press_times), np.array(release_times), np.array(pressures))

    except Exception:
        return None


def compute_hold_time(press_times, release_times):
    """Compute Hold Time (HT) for each keystroke in seconds."""
    if len(press_times) != len(release_times):
        return None
    return (release_times - press_times) / 1000.0


def compute_flight_time(press_times):
    """Compute Flight Time (FT) between consecutive keystrokes in seconds."""
    press_times_sorted = np.sort(press_times)
    return np.diff(press_times_sorted) / 1000.0


def create_histogram_features(hold_times, flight_times):
    """Create normalized histograms for hold times and flight times (502 dimensions)."""
    ht_bins = np.arange(HT_RANGE[0], HT_RANGE[1] + HT_BIN_WIDTH, HT_BIN_WIDTH)[
        : HT_BINS + 1
    ]
    ft_bins = np.arange(FT_RANGE[0], FT_RANGE[1] + FT_BIN_WIDTH, FT_BIN_WIDTH)[
        : FT_BINS + 1
    ]

    ht_hist, _ = np.histogram(hold_times, bins=ht_bins)
    ft_hist, _ = np.histogram(flight_times, bins=ft_bins)

    ht_overflow = np.sum(hold_times > HT_RANGE[1])
    ft_overflow = np.sum(flight_times > FT_RANGE[1])

    ht_hist_with_overflow = np.append(ht_hist, ht_overflow)
    ft_hist_with_overflow = np.append(ft_hist, ft_overflow)

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

    return np.concatenate([ht_hist_norm, ft_hist_norm])


def create_additional_typing_dataset():
    """Create additional_typing_sdataset.pickle from TypingData folder."""
    subject_folders = sorted([f for f in DATA_PATH.iterdir() if f.is_dir()])
    data = []

    for subject_folder in tqdm(subject_folders, desc="Processing subjects"):
        subject_id = subject_folder.name
        txt_files = sorted(subject_folder.glob("*.txt"))

        session_features = []
        for txt_file in txt_files:
            result = parse_typing_file(txt_file)
            if result is None:
                continue

            press_times, release_times, pressures = result
            hold_times = compute_hold_time(press_times, release_times)
            if hold_times is None:
                continue

            flight_times = compute_flight_time(press_times)
            hold_times = hold_times[hold_times >= 0]
            flight_times = flight_times[flight_times >= 0]

            if len(hold_times) < MIN_KEYSTROKES or len(flight_times) < (
                MIN_KEYSTROKES - 1
            ):
                continue

            features = create_histogram_features(hold_times, flight_times)
            session_features.append(features)

        if len(session_features) >= 5:
            label = 1 if SUBJECT_LABELS[subject_id] == "PD" else 0

            # Take first K2 sessions
            bag = session_features[:K2]

            # Zero-pad if less than K2 sessions
            if len(bag) < K2:
                padding = [np.zeros(502) for _ in range(K2 - len(bag))]
                bag.extend(padding)

            X = np.array(bag)
            data.append((subject_id, X, label))

    return pd.DataFrame(data, columns=["subject_id", "X", "y"])


def save_dataset(dataset, output_path="data/additional_typing_sdataset.pickle"):
    """Save the typing dataset DataFrame to a pickle file."""
    project_root = Path(__file__).parent.parent
    output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pkl.dump(dataset, f)

    print(f"Saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB\n")


def print_statistics(dataset):
    """Print dataset statistics."""
    print("Sample of first 3 subjects:")
    for i in range(min(3, len(dataset))):
        row = dataset.iloc[i]
        label_str = "PD" if row["y"] == 1 else "Control"
        print(
            f"  {i+1}. {row['subject_id']}: X.shape={row['X'].shape}, y={row['y']} ({label_str})"
        )

    print("\nLabel distribution:")
    print(dataset["y"].value_counts().sort_index())
    print(f"  • PD patients: {(dataset['y'] == 1).sum()}")
    print(f"  • Healthy controls: {(dataset['y'] == 0).sum()}")

    print("\nData quality checks:")
    first_hist = dataset.iloc[0]["X"][0]
    ht_sum = np.sum(first_hist[:101])
    ft_sum = np.sum(first_hist[101:])
    print(
        f"  • Histogram normalization (first session of {dataset.iloc[0]['subject_id']}):"
    )
    print(f"    - HoldTime sum: {ht_sum:.6f} (expected: ~1.0)")
    print(f"    - FlightTime sum: {ft_sum:.6f} (expected: ~1.0)")
    print(f"    - Total: {ht_sum + ft_sum:.6f} (expected: ~2.0)")

    print("\nSessions per subject statistics:")
    num_sessions = [len(row["X"]) for _, row in dataset.iterrows()]
    print(f"  • Min: {np.min(num_sessions)}")
    print(f"  • Max: {np.max(num_sessions)}")
    print(f"  • Mean: {np.mean(num_sessions):.2f}")
    print(f"  • Median: {int(np.median(num_sessions))}")


if __name__ == "__main__":
    print("Creating additional_typing_sdataset.pickle...\n")
    dataset = create_additional_typing_dataset()
    save_dataset(dataset)
    # print_statistics(dataset)

    typing_sdataset = pkl.load(open("typing_sdataset.pickle", "rb"))
    additional_sdataset = pkl.load(open("../data/additional_typing_sdataset.pickle", "rb"))

    print("Typing dataset statistics:")
    print_statistics(typing_sdataset)

    print("\nAdditional Typing dataset statistics:")
    print_statistics(additional_sdataset)
