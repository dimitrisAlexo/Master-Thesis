"""
Generate t-SNE visualizations for SimCLR embeddings
Produces publication-quality PDF plots for IEEE paper
"""

import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tensorflow as tf
import keras
from keras import layers

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("Loading datasets...")

# Load tremor labeled dataset
with open("datasets/labeled_windows_dataset.pickle", "rb") as f:
    tremor_dataset = pkl.load(f)
print(f"Tremor dataset: {len(tremor_dataset)} samples")
print(f"Tremor label distribution:\n{tremor_dataset['y'].value_counts()}")

# Load typing labeled dataset
with open("datasets/labeled_typing_histograms_dataset.pickle", "rb") as f:
    typing_dataset = pkl.load(f)
print(f"Typing dataset: {len(typing_dataset)} samples")
print(f"Typing label distribution:\n{typing_dataset['y'].value_counts()}")


# Define tremor encoder architecture (must match tremorSimCLRlabeled.py)
def tremor_embeddings_function(M=64):
    return keras.Sequential(
        [
            # Layer 1
            layers.ZeroPadding1D(padding=1),
            layers.Conv1D(filters=32, kernel_size=8, padding="valid"),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            layers.MaxPooling1D(pool_size=2),
            # Layer 2
            layers.ZeroPadding1D(padding=1),
            layers.Conv1D(filters=32, kernel_size=8, padding="valid"),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            layers.MaxPooling1D(pool_size=2),
            # Layer 3
            layers.ZeroPadding1D(padding=1),
            layers.Conv1D(filters=16, kernel_size=16, padding="valid"),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            layers.MaxPooling1D(pool_size=2),
            # Layer 4
            layers.ZeroPadding1D(padding=1),
            layers.Conv1D(filters=16, kernel_size=16, padding="valid"),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            layers.MaxPooling1D(pool_size=2),
            # Flatten and Dense layer
            layers.Flatten(),
            layers.Dense(M),
        ],
        name="tremor_embeddings",
    )


# Define typing encoder architecture (must match typingSimCLR.py)
def typing_embeddings_function(M=64):
    return keras.Sequential(
        [
            layers.Input(shape=(502,)),
            # Layer 1
            layers.Dense(100),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.1),
            # Layer 2
            layers.Dense(50),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.1),
            # Output layer
            layers.Dense(M),
        ],
        name="typing_embeddings",
    )


print("\nCreating encoder models...")

# Create tremor encoder and load weights
tremor_encoder = tremor_embeddings_function(M=64)
tremor_encoder.build((None, 1000, 3))  # Build with tremor input shape
tremor_encoder.load_weights("weights/tremor/tremor_simclr_embeddings.weights.h5")
print("Loaded tremor encoder weights")

# Create typing encoder and load weights
typing_encoder = typing_embeddings_function(M=64)
typing_encoder.build((None, 502))  # Build with typing input shape
typing_encoder.load_weights("weights/typing/typing_simclr_embeddings.weights.h5")
print("Loaded typing encoder weights")

print("\nGenerating embeddings...")

# Generate tremor embeddings
tremor_data = np.array(tremor_dataset["X"].tolist())
tremor_labels = np.array(tremor_dataset["y"].tolist())
tremor_embeddings = tremor_encoder.predict(tremor_data, verbose=0)
print(f"Tremor embeddings shape: {tremor_embeddings.shape}")

# Generate typing embeddings
typing_data = np.array(typing_dataset["X"].tolist())
typing_labels = np.array(typing_dataset["y"].tolist())
typing_embeddings = typing_encoder.predict(typing_data, verbose=0)
print(f"Typing embeddings shape: {typing_embeddings.shape}")

print("\nRunning t-SNE...")

# Run t-SNE for tremor embeddings
tremor_tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    random_state=42
)
tremor_reduced = tremor_tsne.fit_transform(tremor_embeddings)

# Run t-SNE for typing embeddings
typing_tsne = TSNE(
    n_components=2,
    perplexity=55,
    learning_rate='auto',
    n_iter=550,
)
typing_reduced = typing_tsne.fit_transform(typing_embeddings)

print("t-SNE complete")

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

print("\nGenerating publication-quality plots...")

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (5, 4),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Plot tremor t-SNE
fig, ax = plt.subplots(figsize=(5, 4))
scatter_no_tremor = ax.scatter(
    tremor_reduced[tremor_labels == 0, 0],
    tremor_reduced[tremor_labels == 0, 1],
    c="#6fa2c5",  # Blue
    alpha=0.6,
    s=30,
    edgecolors='none',
    label='No Tremor'
)
scatter_tremor = ax.scatter(
    tremor_reduced[tremor_labels == 1, 0],
    tremor_reduced[tremor_labels == 1, 1],
    c='#e74c3c',  # Red
    alpha=0.6,
    s=30,
    edgecolors='none',
    label='Tremor'
)
# Remove all axes, labels, ticks, and grid
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
plt.savefig('plots/tremor_tsne.pdf', format='pdf', bbox_inches='tight')
print("Saved: plots/tremor_tsne.pdf")
plt.close()

# Plot typing t-SNE
fig, ax = plt.subplots(figsize=(5, 4))
scatter_no_fmi = ax.scatter(
    typing_reduced[typing_labels == 0, 0],
    typing_reduced[typing_labels == 0, 1],
    c='#3498db',  # Blue
    alpha=0.6,
    s=30,
    edgecolors='none',
    label='No FMI'
)
scatter_fmi = ax.scatter(
    typing_reduced[typing_labels == 1, 0],
    typing_reduced[typing_labels == 1, 1],
    c='#e74c3c',  # Red
    alpha=0.6,
    s=30,
    edgecolors='none',
    label='FMI'
)
# Remove all axes, labels, ticks, and grid
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
plt.savefig('plots/typing_tsne.pdf', format='pdf', bbox_inches='tight')
print("Saved: plots/typing_tsne.pdf")
plt.close()

print("\nAll plots generated successfully!")
print("Plots saved in 'plots/' directory as PDF files suitable for IEEE paper.")
