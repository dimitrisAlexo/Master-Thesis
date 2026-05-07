"""
Print statistics for tremor and typing datasets (both labeled and unlabeled)
"""

import pickle as pkl
import numpy as np
import pandas as pd

print("=" * 60)
print("DATASET STATISTICS")
print("=" * 60)

# ============================================================================
# TREMOR DATASETS
# ============================================================================
print("\n" + "=" * 60)
print("TREMOR DATASETS")
print("=" * 60)

# Tremor labeled dataset (sdataset)
try:
    with open("datasets/sdataset.pickle", "rb") as f:
        tremor_sdataset = pkl.load(f)

    print("\n--- Tremor Labeled Dataset (datasets/sdataset.pickle) ---")
    print(f"Number of subjects: {len(tremor_sdataset)}")

    # Get label distribution
    labels = tremor_sdataset["y_train"].values
    unique, counts = np.unique(labels, return_counts=True)
    print("\nLabel distribution:")
    for label, count in zip(unique, counts):
        label_name = "Control" if label == 0 else "Tremor"
        print(
            f"  {label_name} (label={label}): {count} subjects ({count/len(labels)*100:.1f}%)"
        )

    # Print shape information
    if len(tremor_sdataset) > 0:
        sample_shape = np.array(tremor_sdataset["X"].iloc[0]).shape
        print(f"\nSample bag shape: {sample_shape}")

except FileNotFoundError:
    print("\n--- Tremor Labeled Dataset (datasets/sdataset.pickle) ---")
    print("File not found!")
except Exception as e:
    print(f"Error loading tremor labeled dataset: {e}")

# Tremor unlabeled dataset
try:
    with open("datasets/unlabeled_tremor_data.pickle", "rb") as f:
        tremor_unlabeled = pkl.load(f)

    print("\n--- Tremor Unlabeled Dataset (datasets/unlabeled_tremor_data.pickle) ---")
    print(f"Number of samples: {len(tremor_unlabeled)}")

    if len(tremor_unlabeled) > 0:
        sample_shape = tremor_unlabeled[0].shape
        print(f"Sample shape: {sample_shape}")

except FileNotFoundError:
    print("\n--- Tremor Unlabeled Dataset (datasets/unlabeled_tremor_data.pickle) ---")
    print("File not found!")
except Exception as e:
    print(f"Error loading tremor unlabeled dataset: {e}")

# ============================================================================
# TYPING DATASETS
# ============================================================================
print("\n" + "=" * 60)
print("TYPING DATASETS")
print("=" * 60)

# Typing labeled dataset (sdataset)
try:
    with open("datasets/typing_sdataset.pickle", "rb") as f:
        typing_sdataset = pkl.load(f)

    print("\n--- Typing Labeled Dataset (datasets/typing_sdataset.pickle) ---")
    print(f"Number of subjects: {len(typing_sdataset)}")

    # Get label distribution
    labels = typing_sdataset["y"].values
    unique, counts = np.unique(labels, return_counts=True)
    print("\nLabel distribution:")
    for label, count in zip(unique, counts):
        label_name = "Control (No FMI)" if label == 0 else "FMI"
        print(
            f"  {label_name} (label={label}): {count} subjects ({count/len(labels)*100:.1f}%)"
        )

    # Print shape information
    if len(typing_sdataset) > 0:
        sample_shape = np.array(typing_sdataset["X"].iloc[0]).shape
        print(f"\nSample bag shape: {sample_shape}")

except FileNotFoundError:
    print("\n--- Typing Labeled Dataset (datasets/typing_sdataset.pickle) ---")
    print("File not found!")
except Exception as e:
    print(f"Error loading typing labeled dataset: {e}")

# Typing unlabeled dataset
try:
    with open("datasets/unlabeled_typing_data.pickle", "rb") as f:
        typing_unlabeled = pkl.load(f)

    print("\n--- Typing Unlabeled Dataset (datasets/unlabeled_typing_data.pickle) ---")
    print(f"Number of samples: {len(typing_unlabeled)}")

    if len(typing_unlabeled) > 0:
        sample_shape = typing_unlabeled[0].shape
        print(f"Sample shape: {sample_shape}")

except FileNotFoundError:
    print("\n--- Typing Unlabeled Dataset (datasets/unlabeled_typing_data.pickle) ---")
    print("File not found!")
except Exception as e:
    print(f"Error loading typing unlabeled dataset: {e}")

# ============================================================================
# ADDITIONAL TYPING DATASET
# ============================================================================
print("\n" + "=" * 60)
print("ADDITIONAL TYPING DATASET")
print("=" * 60)

try:
    # Temporarily patch numpy._core to numpy.core for compatibility
    import sys

    if "numpy._core" not in sys.modules:
        import numpy.core

        sys.modules["numpy._core"] = numpy.core
        sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
        sys.modules["numpy._core.umath"] = numpy.core.umath

    with open("../data/additional_typing_sdataset.pickle", "rb") as f:
        additional_typing = pkl.load(f)

    print("\n--- Additional Typing Dataset (additional_typing_sdataset.pickle) ---")
    print(f"Number of subjects: {len(additional_typing)}")

    # Get label distribution
    labels = additional_typing["y"].values
    unique, counts = np.unique(labels, return_counts=True)
    print("\nLabel distribution:")
    for label, count in zip(unique, counts):
        label_name = "Control (No FMI)" if label == 0 else "FMI"
        print(
            f"  {label_name} (label={label}): {count} subjects ({count/len(labels)*100:.1f}%)"
        )

    # Print shape information
    if len(additional_typing) > 0:
        sample_shape = np.array(additional_typing["X"].iloc[0]).shape
        print(f"\nSample bag shape: {sample_shape}")

except FileNotFoundError:
    print("\n--- Additional Typing Dataset (additional_typing_sdataset.pickle) ---")
    print("File not found!")
except Exception as e:
    print(f"Error loading additional typing dataset: {e}")

print("\n" + "=" * 60)
print("END OF STATISTICS")
print("=" * 60)
