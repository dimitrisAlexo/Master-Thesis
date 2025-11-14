from utils import *
import time
import os
import matplotlib.pyplot as plt
import numpy as np

start = time.time()

# Load typing supervised dataset
# First, let's try to load existing typing_sdataset.pickle
try:
    with open("typing_sdataset.pickle", "rb") as f:
        print("Loading typing_sdataset...")
        typing_sdataset = pkl.load(f)
except FileNotFoundError:
    print("typing_sdataset.pickle not found. Creating it...")
    # Load typing data and create the dataset
    sdata_path = os.path.join("..", "data", "typing_sdata.pickle")
    typing_sdata = unpickle_data(sdata_path)

    # Create typing dataset with K2=100 histograms per subject
    K2 = 100
    typing_sdataset = form_typing_dataset(typing_sdata, K2)

print("Typing dataset shape:", typing_sdataset.shape)
print("Columns:", typing_sdataset.columns.tolist())
print("Label distribution:")
print(typing_sdataset["y"].value_counts())
print("\nFirst few rows:")
print(typing_sdataset.head())


def plot_typing_histograms(X, subject_idx, save_fig=False):
    """
    Visualizes 100 typing histograms for the given subject index.
    Each histogram is 502-dimensional (hold time + flight time concatenated).

    Parameters:
    - X: The input data containing typing histograms [100, 502]
    - subject_idx: The index of the subject in the dataset to visualize
    - save_fig: If True, saves the figure as 'labeled_typing_data_<subject_idx>.png'
    """
    subject_histograms = X.iloc[
        subject_idx
    ]  # Access the histograms for the subject (bag of K2=100 histograms)

    fig, axs = plt.subplots(10, 10, figsize=(25, 25))
    fig.suptitle(
        f"Subject {subject_idx} - Typing Histograms (FMI: {typing_sdataset.iloc[subject_idx]['y']})",
        fontsize=20,
    )

    for i in range(100):  # Loop over the 100 histograms
        ax = axs[i // 10, i % 10]  # Get the subplot (10x10 grid)
        histogram_data = subject_histograms[i]  # Get the specific histogram [502]

        # Create x-axis for the full histogram (0-501)
        x_axis = range(502)

        # Plot the full histogram
        ax.plot(x_axis, histogram_data, "b-", linewidth=0.8, alpha=0.7)

        # Add vertical line to separate hold time (0-100) from flight time (101-501)
        ax.axvline(x=101, color="red", linestyle="--", alpha=0.6, linewidth=1)

        ax.set_title(f"Histogram {i}", fontsize=8)
        ax.set_xlabel("Feature Index", fontsize=6)
        ax.set_ylabel("Frequency", fontsize=6)
        ax.tick_params(axis="both", which="major", labelsize=5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title

    if save_fig:
        plt.savefig(
            f"labeled_typing_data_{subject_idx}.png", dpi=150, bbox_inches="tight"
        )

    # plt.show()
    return


# Get the histogram data (X column contains bags of histograms)
X = typing_sdataset["X"]

print(f"\nDataset contains {len(X)} subjects")
print(f"Each subject has a bag of {len(X.iloc[0]) if len(X) > 0 else 0} histograms")
print(f"Each histogram has {len(X.iloc[0][0]) if len(X) > 0 else 0} features")

# Display some basic statistics about the dataset
print("\nSubject-wise label distribution:")
for idx, row in typing_sdataset.iterrows():
    print(f"Subject {idx} (ID: {row['subject_id']}): FMI = {row['y']}")

# You can uncomment these lines to visualize specific subjects
# Note: Start with a few subjects to manually inspect and annotate

print("\n" + "=" * 50)
print("MANUAL ANNOTATION INSTRUCTIONS:")
print("=" * 50)
print("1. Uncomment the plot_typing_histograms() lines below to visualize subjects")
print("2. Look at the histograms and identify which ones show clear FMI patterns")
print("3. For FMI=1 subjects, look for histograms with distinctive patterns")
print("4. For FMI=0 subjects, look for histograms with normal typing patterns")
print("5. Note down the histogram indices (0-99) that show clear patterns")
print("6. Update the label_histograms() calls with your annotations")
print("=" * 50)

# Example plots - uncomment to start annotation process
# plot_typing_histograms(X, 0, save_fig=True)
# plot_typing_histograms(X, 1, save_fig=True)
# plot_typing_histograms(X, 2, save_fig=True)
# plot_typing_histograms(X, 4, save_fig=True)
# plot_typing_histograms(X, 5, save_fig=True)
# plot_typing_histograms(X, 6, save_fig=True)
# plot_typing_histograms(X, 7, save_fig=True)
# plot_typing_histograms(X, 8, save_fig=True)
# plot_typing_histograms(X, 9, save_fig=True)
# plot_typing_histograms(X, 10, save_fig=True)

# plot_typing_histograms(X, 11, save_fig=True)
# plot_typing_histograms(X, 12, save_fig=True)
# plot_typing_histograms(X, 13, save_fig=True)
# plot_typing_histograms(X, 14, save_fig=True)
# plot_typing_histograms(X, 16, save_fig=True)
# plot_typing_histograms(X, 21, save_fig=True)
# plot_typing_histograms(X, 22, save_fig=True)
# plot_typing_histograms(X, 23, save_fig=True)


# Initialize the empty DataFrame for labeled histograms
labeled_histograms_dataset = pd.DataFrame(columns=["X", "y"])


def label_histograms(labeled_dataset, X, subject_idx, histogram_indices, label):
    """
    Function to label histograms and add them to labeled_histograms_dataset

    Parameters:
    - labeled_dataset: The dataset to append to
    - X: The input data containing histogram bags
    - subject_idx: The index of the subject in the dataset
    - histogram_indices: List of histogram indices (0-99) to label
    - label: The label to assign (0 or 1)
    """
    for hist_idx in histogram_indices:
        # Extract the specific histogram data
        histogram_data = X.iloc[subject_idx][hist_idx]

        # Append the histogram and label to the dataset
        new_row = pd.DataFrame(
            {"X": [histogram_data], "y": [label]}  # Storing as list to avoid expansion
        )

        # Concatenate the new row to the labeled_dataset
        labeled_dataset = pd.concat([labeled_dataset, new_row], ignore_index=True)

    return labeled_dataset


# Manual annotation section - update these based on your visual inspection
# IMPORTANT: Uncomment and run the plotting functions above first to identify patterns

print("\n" + "=" * 50)
print("ANNOTATION SECTION - UPDATE AFTER VISUAL INSPECTION")
print("=" * 50)

# Example annotation template - replace with your actual annotations
# After inspecting the plots, uncomment and update these with the actual histogram indices

# FMI = 1
labeled_histograms_dataset = label_histograms(
    labeled_histograms_dataset,
    X,
    subject_idx=1,
    histogram_indices=[0, 2, 3, 4, 5, 13, 17, 38, 68, 52],
    label=1,
)

labeled_histograms_dataset = label_histograms(
    labeled_histograms_dataset,
    X,
    subject_idx=2,
    histogram_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    label=1,
)

labeled_histograms_dataset = label_histograms(
    labeled_histograms_dataset,
    X,
    subject_idx=5,
    histogram_indices=[4, 5, 6, 8, 9, 13, 15, 18, 22, 23],
    label=1,
)

labeled_histograms_dataset = label_histograms(
    labeled_histograms_dataset,
    X,
    subject_idx=6,
    histogram_indices=[1, 3, 36, 37, 46, 49, 61, 63, 71, 77],
    label=1,
)

labeled_histograms_dataset = label_histograms(
    labeled_histograms_dataset,
    X,
    subject_idx=7,
    histogram_indices=[0, 2, 5, 7, 15, 34, 48, 49, 62, 82],
    label=1,
)

labeled_histograms_dataset = label_histograms(
    labeled_histograms_dataset,
    X,
    subject_idx=9,
    histogram_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    label=1,
)

# FMI = 0
# labeled_histograms_dataset = label_histograms(
#     labeled_histograms_dataset,
#     X,
#     subject_idx=11,
#     histogram_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#     label=0,
# )

labeled_histograms_dataset = label_histograms(
    labeled_histograms_dataset,
    X,
    subject_idx=12,
    histogram_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    label=0,
)

labeled_histograms_dataset = label_histograms(
    labeled_histograms_dataset,
    X,
    subject_idx=13,
    histogram_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    label=0,
)

labeled_histograms_dataset = label_histograms(
    labeled_histograms_dataset,
    X,
    subject_idx=22,
    histogram_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    label=0,
)

labeled_histograms_dataset = label_histograms(
    labeled_histograms_dataset,
    X,
    subject_idx=23,
    histogram_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    label=0,
)

# Continue adding more subjects and histograms...
# Aim for balanced dataset (similar number of label 0 and label 1 samples)

# After completing manual annotation, save the dataset
if len(labeled_histograms_dataset) > 0:
    labeled_histograms_dataset.to_pickle("labeled_typing_histograms_dataset.pickle")
    print(
        f"\nLabeled histograms dataset has been saved to 'labeled_typing_histograms_dataset.pickle'."
    )
    print(f"Dataset contains {len(labeled_histograms_dataset)} labeled histograms")
    print("Label distribution:")
    print(labeled_histograms_dataset["y"].value_counts())
else:
    print("\nNo histograms have been labeled yet.")
    print(
        "Please uncomment the plotting functions, inspect the data, and add your annotations."
    )

print(labeled_histograms_dataset)

print(f"\nTotal processing time: {time.time() - start:.2f} seconds")
