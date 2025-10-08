"""
Helper script to visualize typing histograms for people with and without FMI.
The typing data consists of concatenated hold time and flight time histograms.
"""

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import random
from utils import unpickle_data


def load_typing_dataset():
    """Load the typing supervised dataset"""
    try:
        with open("typing_sdataset.pickle", "rb") as f:
            dataset = pkl.load(f)
        return dataset
    except FileNotFoundError:
        print("typing_sdataset.pickle not found. Creating it...")
        # Try to create the dataset if it doesn't exist
        import os

        sdata_path = os.path.join("..", "data", "typing_sdata.pickle")
        typing_sdata = unpickle_data(sdata_path)
        from utils import form_typing_dataset

        dataset = form_typing_dataset(typing_sdata, K2=100)
        return dataset


def visualize_typing_histograms_simple(dataset, n_samples=3):
    """
    Simple visualization of typing histograms for people with and without FMI.
    Shows the full 502-dimensional histogram for each person.

    Args:
        dataset: DataFrame with columns ["subject_id", "X", "y"]
        n_samples: Number of samples to show for each class
    """
    # Separate subjects by label (FMI: 1, No FMI: 0)
    fmi_subjects = dataset[dataset["y"] == 1]
    no_fmi_subjects = dataset[dataset["y"] == 0]

    print(f"Total subjects with FMI: {len(fmi_subjects)}")
    print(f"Total subjects without FMI: {len(no_fmi_subjects)}")

    # Sample random subjects from each group
    fmi_samples = fmi_subjects.sample(min(n_samples, len(fmi_subjects)))
    no_fmi_samples = no_fmi_subjects.sample(min(n_samples, len(no_fmi_subjects)))

    # Create figure with 2 rows, 3 columns
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 8))
    fig.suptitle("Typing Histograms: FMI vs No FMI", fontsize=16)

    # Plot FMI subjects (top row)
    for i, (_, row) in enumerate(fmi_samples.iterrows()):
        if i >= n_samples:
            break
        subject_id = row["subject_id"]
        bag = row["X"]  # This is a bag of K2 histograms

        # Select a random histogram from the bag
        random_histogram = bag[random.randint(0, len(bag) - 1)]

        # Plot the full histogram
        x_axis = range(502)  # 0 to 501
        axes[0, i].plot(x_axis, random_histogram, "b-", linewidth=1)
        axes[0, i].set_title(f"FMI Subject {subject_id}", fontsize=12)
        axes[0, i].set_xlabel("Histogram Bin (0-502)")
        axes[0, i].set_ylabel("Frequency")
        axes[0, i].grid(True, alpha=0.3)

        # Add vertical line to separate hold time (0-100) from flight time (101-501)
        axes[0, i].axvline(
            x=101, color="red", linestyle="--", alpha=0.5, label="Hold/Flight boundary"
        )
        axes[0, i].legend(fontsize=8)

    # Plot No FMI subjects (bottom row)
    for i, (_, row) in enumerate(no_fmi_samples.iterrows()):
        if i >= n_samples:
            break
        subject_id = row["subject_id"]
        bag = row["X"]  # This is a bag of K2 histograms

        # Select a random histogram from the bag
        random_histogram = bag[random.randint(0, len(bag) - 1)]

        # Plot the full histogram
        x_axis = range(502)  # 0 to 501
        axes[1, i].plot(x_axis, random_histogram, "g-", linewidth=1)
        axes[1, i].set_title(f"No FMI Subject {subject_id}", fontsize=12)
        axes[1, i].set_xlabel("Histogram Bin (0-502)")
        axes[1, i].set_ylabel("Frequency")
        axes[1, i].grid(True, alpha=0.3)

        # Add vertical line to separate hold time (0-100) from flight time (101-501)
        axes[1, i].axvline(
            x=101, color="red", linestyle="--", alpha=0.5, label="Hold/Flight boundary"
        )
        axes[1, i].legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_average_histograms_simple(dataset):
    """
    Plot simple average histograms for FMI vs No FMI subjects.
    """
    # Separate subjects by label
    fmi_subjects = dataset[dataset["y"] == 1]
    no_fmi_subjects = dataset[dataset["y"] == 0]

    # Collect all histograms for each group
    fmi_histograms = []
    no_fmi_histograms = []

    for _, row in fmi_subjects.iterrows():
        bag = row["X"]
        fmi_histograms.extend(bag)

    for _, row in no_fmi_subjects.iterrows():
        bag = row["X"]
        no_fmi_histograms.extend(bag)

    # Calculate average histograms
    fmi_avg = np.mean(fmi_histograms, axis=0)
    no_fmi_avg = np.mean(no_fmi_histograms, axis=0)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Average Typing Histograms: FMI vs No FMI", fontsize=16)

    x_axis = range(502)  # 0 to 501

    # Plot FMI average
    ax1.plot(x_axis, fmi_avg, "b-", linewidth=2, label="FMI Average")
    ax1.set_title("Average - With FMI")
    ax1.set_xlabel("Histogram Bin (0-502)")
    ax1.set_ylabel("Average Frequency")
    ax1.grid(True, alpha=0.3)
    ax1.axvline(
        x=101, color="red", linestyle="--", alpha=0.5, label="Hold/Flight boundary"
    )
    ax1.legend()

    # Plot No FMI average
    ax2.plot(x_axis, no_fmi_avg, "g-", linewidth=2, label="No FMI Average")
    ax2.set_title("Average - Without FMI")
    ax2.set_xlabel("Histogram Bin (0-502)")
    ax2.set_ylabel("Average Frequency")
    ax2.grid(True, alpha=0.3)
    ax2.axvline(
        x=101, color="red", linestyle="--", alpha=0.5, label="Hold/Flight boundary"
    )
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Print some statistics
    print(f"\nStatistics:")
    print(f"Total FMI histograms: {len(fmi_histograms)}")
    print(f"Total no-FMI histograms: {len(no_fmi_histograms)}")
    print(f"Average hold time activity (FMI): {np.sum(fmi_avg[:101]):.2f}")
    print(f"Average hold time activity (no FMI): {np.sum(no_fmi_avg[:101]):.2f}")
    print(f"Average flight time activity (FMI): {np.sum(fmi_avg[101:]):.2f}")
    print(f"Average flight time activity (no FMI): {np.sum(no_fmi_avg[101:]):.2f}")


def main():
    """Main function to run the visualization"""
    print("Loading typing dataset...")
    dataset = load_typing_dataset()

    print(f"Dataset shape: {dataset.shape}")
    print(f"Columns: {dataset.columns.tolist()}")
    print(f"Label distribution:")
    print(dataset["y"].value_counts())

    # Visualize individual histograms (3 with FMI, 3 without)
    print("\nVisualizing individual typing histograms...")
    visualize_typing_histograms_simple(dataset, n_samples=3)

    # Plot average histograms
    print("\nPlotting average histograms...")
    plot_average_histograms_simple(dataset)


if __name__ == "__main__":
    main()
