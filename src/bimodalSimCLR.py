"""
## Bimodal SimCLR: Teacher-Student Approach
## Teacher: Pretrained tremor encoder (frozen)
## Student: Typing encoder (trainable)
"""

import os
import sys
import time
import pickle as pkl
import resource
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from sklearn.manifold import TSNE
from keras import ops
from keras import layers
from keras import callbacks
from tf_keras import mixed_precision

from utils import *

start = time.time()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_BACKEND"] = "tensorflow"

np.set_printoptions(threshold=sys.maxsize)

# Make sure we are able to handle large datasets
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

plt.ion()

os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

if tf.config.list_physical_devices("GPU"):
    print("Using GPU...")
else:
    print("Using CPU...")

# Mixed precision policy
policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)
print("Using mixed precision...")

"""
## Hyperparameter setup
"""

dataset_size = 10240
M = 64
batch_size = 512
labeled_batch_size = 4
num_epochs = 200
temperature = 0.01
learning_rate = 0.001

"""
## Dataset
"""

# Load bimodal dataset
print("Loading bimodal dataset...")
with open("../data/bimodal_dataset.pickle", "rb") as f:
    bimodal_data = pkl.load(f)

# Handle both old format (list) and new format (dict with metadata)
if isinstance(bimodal_data, dict):
    matched_pairs = bimodal_data.get("matched_pairs", [])
    print(f"Loaded {len(matched_pairs)} matched pairs from checkpoint")
else:
    matched_pairs = bimodal_data
    print(f"Loaded {len(matched_pairs)} matched pairs")

# Extract typing histograms and accelerometer segments
typing_histograms = []
accel_windows = []

for pair in matched_pairs:
    typing_features = pair["typing_features"]  # Shape: (502,)
    accel_segments = pair["accel_segments"]  # Shape: (n_segments, 1000, 3)

    # For each accelerometer segment, create a pair with the typing histogram
    for segment in accel_segments:
        typing_histograms.append(typing_features)
        accel_windows.append(segment)

typing_histograms = np.array(typing_histograms)
accel_windows = np.array(accel_windows)

print(f"Typing histograms shape: {typing_histograms.shape}")
print(f"Accelerometer windows shape: {accel_windows.shape}")

# Normalize accelerometer data
accel_windows = normalize(accel_windows)

# Limit dataset size if needed
if len(typing_histograms) > dataset_size:
    indices = np.random.choice(len(typing_histograms), dataset_size, replace=False)
    typing_histograms = typing_histograms[indices]
    accel_windows = accel_windows[indices]
    print(f"Limited dataset to {dataset_size} pairs")

print(f"Final dataset size: {len(typing_histograms)}")

# Create TensorFlow dataset for unlabeled data
dataset = tf.data.Dataset.from_tensor_slices((typing_histograms, accel_windows))
dataset = (
    dataset.shuffle(buffer_size=len(typing_histograms))
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Load labeled bimodal dataset
print("\nLoading labeled bimodal dataset...")
with open("../data/labeled_bimodal_dataset.pickle", "rb") as f:
    labeled_dataset = pkl.load(f)

print(f"Labeled dataset: {labeled_dataset['total_pairs']} pairs")
print(f"Label distribution: {labeled_dataset['label_distribution']}")

# Extract labeled data
labeled_typing = np.array(
    [entry["typing_features"] for entry in labeled_dataset["data"]]
)
labeled_accel = np.array([entry["accel_segment"] for entry in labeled_dataset["data"]])
labeled_labels = np.array([entry["label"] for entry in labeled_dataset["data"]])

# Normalize labeled accelerometer data
labeled_accel = normalize(labeled_accel)

print(f"Labeled typing shape: {labeled_typing.shape}")
print(f"Labeled accel shape: {labeled_accel.shape}")
print(f"Labeled labels shape: {labeled_labels.shape}")

# Split into train and test (80/20)
split_idx = int(0.8 * len(labeled_typing))
indices = np.random.permutation(len(labeled_typing))

train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

labeled_train_typing = labeled_typing[train_indices]
labeled_train_accel = labeled_accel[train_indices]
labeled_train_labels = labeled_labels[train_indices]

labeled_test_typing = labeled_typing[test_indices]
labeled_test_accel = labeled_accel[test_indices]
labeled_test_labels = labeled_labels[test_indices]

print(f"\nTrain set: {len(labeled_train_typing)} pairs")
print(f"Test set: {len(labeled_test_typing)} pairs")

# Create TensorFlow datasets for labeled data
labeled_train_dataset = tf.data.Dataset.from_tensor_slices(
    ((labeled_train_typing, labeled_train_accel), labeled_train_labels)
)
labeled_train_dataset = (
    labeled_train_dataset.shuffle(buffer_size=len(labeled_train_typing))
    .batch(labeled_batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

labeled_test_dataset = tf.data.Dataset.from_tensor_slices(
    ((labeled_test_typing, labeled_test_accel), labeled_test_labels)
)
labeled_test_dataset = labeled_test_dataset.batch(10).prefetch(
    buffer_size=tf.data.AUTOTUNE
)

# Create combined training dataset (unlabeled + labeled)
train_dataset = tf.data.Dataset.zip((dataset, labeled_train_dataset)).prefetch(
    buffer_size=tf.data.AUTOTUNE
)

"""
## Encoder architectures
"""


# Tremor encoder (teacher) - matches the architecture from tremorSimCLRlabeled.py
def tremor_encoder(M):
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
            # Flatten and Dense layer to get M-dimensional output
            layers.Flatten(),
            layers.Dense(M),
        ],
        name="tremor_encoder",
    )


# Typing encoder (student) - matches the architecture from typingSimCLR.py
def typing_encoder(M):
    return keras.Sequential(
        [
            # Input layer to define shape (502 for typing histograms)
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
        name="typing_encoder",
    )


"""
## Teacher-Student Contrastive Model
"""


class BimodalContrastiveModel(keras.Model):
    def __init__(self):
        super().__init__()

        self.temperature = temperature

        # Student encoder (trainable)
        self.typing_encoder = typing_encoder(M)

        # Teacher encoder (frozen)
        self.tremor_encoder = tremor_encoder(M)

        # Build the tremor encoder by passing a dummy input
        dummy_accel_input = tf.zeros((1, 1000, 3))
        _ = self.tremor_encoder(dummy_accel_input)

        # Load pretrained weights for tremor encoder
        print("Loading pretrained tremor encoder weights...")
        self.tremor_encoder.load_weights("tremor_simclr_embeddings.weights.h5")
        print("✓ Tremor encoder weights loaded successfully")

        # Freeze tremor encoder (teacher)
        self.tremor_encoder.trainable = False
        print("✓ Tremor encoder frozen (teacher)")

        # Non-linear MLP as projection head for typing encoder
        self.projection_head = keras.Sequential(
            [
                keras.Input(shape=(M,)),
                layers.Dense(M, activation="relu"),
                layers.Dense(M),
            ],
            name="projection_head",
        )

        # Linear probe for classification
        self.linear_probe = keras.Sequential(
            [
                layers.Input(shape=(M,)),
                layers.Dropout(0.1),
                layers.Dense(2, kernel_regularizer=keras.regularizers.L2(1e-4)),
            ],
            name="linear_probe",
        )

        self.typing_encoder.summary()
        self.tremor_encoder.summary()
        self.projection_head.summary()
        self.linear_probe.summary()

    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super().compile(**kwargs)
        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer

        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="c_acc"
        )

        self.probe_loss_tracker = keras.metrics.Mean(name="p_loss")
        self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy(name="p_acc")

    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
            self.probe_loss_tracker,
            self.probe_accuracy,
        ]

    def contrastive_loss(self, projections_1, projections_2):
        """
        InfoNCE loss (information noise-contrastive estimation)
        NT-Xent loss (normalized temperature-scaled cross entropy)

        projections_1: embeddings from typing encoder (student)
        projections_2: embeddings from tremor encoder (teacher)
        """
        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.nn.l2_normalize(projections_1, axis=1)
        projections_2 = tf.nn.l2_normalize(projections_2, axis=1)
        similarities = (
            ops.matmul(projections_1, ops.transpose(projections_2)) / self.temperature
        )

        # The similarity between the typing and tremor representations from the
        # same temporal window should be higher than with other windows
        batch_size = ops.shape(projections_1)[0]
        contrastive_labels = ops.arange(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(
            contrastive_labels, ops.transpose(similarities)
        )

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, ops.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2

    def train_step(self, data):
        # Unpack unlabeled and labeled data
        unlabeled_data = data[0]
        labeled_data = data[1]

        typing_data, accel_data = unlabeled_data
        (labeled_typing, labeled_accel), labels = labeled_data

        # Contrastive learning step
        with tf.GradientTape() as tape:
            # Get embeddings from student (typing) encoder
            typing_embeddings = self.typing_encoder(typing_data, training=True)
            typing_projections = self.projection_head(typing_embeddings, training=True)

            # Get embeddings from teacher (tremor) encoder (no gradients computed)
            tremor_embeddings = self.tremor_encoder(accel_data, training=False)

            # Compute contrastive loss
            contrastive_loss = self.contrastive_loss(
                typing_projections, tremor_embeddings
            )

        # Compute gradients only for typing encoder and projection head
        trainable_weights = (
            self.typing_encoder.trainable_weights
            + self.projection_head.trainable_weights
        )
        gradients = tape.gradient(contrastive_loss, trainable_weights)
        self.contrastive_optimizer.apply_gradients(zip(gradients, trainable_weights))

        # Update contrastive metrics
        self.contrastive_loss_tracker.update_state(contrastive_loss)
        self.contrastive_accuracy.update_state(
            tf.range(tf.shape(typing_embeddings)[0]),
            tf.matmul(
                tf.nn.l2_normalize(typing_projections, axis=1),
                tf.nn.l2_normalize(tremor_embeddings, axis=1),
                transpose_b=True,
            )
            / self.temperature,
        )

        # Linear probe step
        with tf.GradientTape() as tape:
            # Encode labeled data (typing encoder in inference mode)
            labeled_typing_embeddings = self.typing_encoder(
                labeled_typing, training=False
            )
            # Classify
            class_logits = self.linear_probe(labeled_typing_embeddings, training=True)
            probe_loss = self.probe_loss(labels, class_logits)

        # Update only linear probe weights
        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(gradients, self.linear_probe.trainable_weights)
        )

        # Update probe metrics
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack labeled test data
        (labeled_typing, labeled_accel), labels = data

        # Encode and classify (inference mode)
        labeled_typing_embeddings = self.typing_encoder(labeled_typing, training=False)
        class_logits = self.linear_probe(labeled_typing_embeddings, training=False)
        probe_loss = self.probe_loss(labels, class_logits)

        # Update probe metrics
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)

        # Only return probe metrics for test
        return {m.name: m.result() for m in self.metrics[2:]}

    def plot_contrastive_loss(self, pretraining_history):
        """
        Plots contrastive loss per epoch.
        """
        plt.figure(figsize=(6, 5))
        plt.plot(
            pretraining_history.history["c_loss"],
            label="Contrastive Loss",
            color="blue",
        )
        plt.title("Teacher-Student Contrastive Loss per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        # Plot validation probe loss
        plt.figure(figsize=(6, 5))
        plt.plot(
            pretraining_history.history["val_p_loss"],
            label="Validation Probe Loss",
            color="red",
        )
        plt.title("Validation Probe Loss per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_contrastive_accuracy(self, pretraining_history):
        """
        Plots contrastive accuracy per epoch.
        """
        plt.figure(figsize=(6, 5))
        plt.plot(
            pretraining_history.history["c_acc"],
            label="Contrastive Accuracy",
            color="green",
        )
        plt.title("Teacher-Student Contrastive Accuracy per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

        # Plot validation probe accuracy
        plt.figure(figsize=(6, 5))
        plt.plot(
            pretraining_history.history["val_p_acc"],
            label="Validation Probe Accuracy",
            color="orange",
        )
        plt.title("Validation Probe Accuracy per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()


"""
## Training
"""

# Initialize model
print("\n" + "=" * 80)
print("INITIALIZING BIMODAL TEACHER-STUDENT MODEL")
print("=" * 80)
pretraining_model = BimodalContrastiveModel()
pretraining_model.compile(
    contrastive_optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    probe_optimizer=keras.optimizers.Adam(learning_rate=5e-4),
)

# Callbacks
checkpoint = callbacks.ModelCheckpoint(
    filepath="typing_bimodal_best_model.weights.h5",
    monitor="val_p_loss",
    mode="min",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)


def lr_schedule(epoch, lr):
    total_epochs = num_epochs
    decay_start_epoch = total_epochs // 2
    if epoch == 0:
        return learning_rate
    elif epoch >= decay_start_epoch:
        return lr * 0.99
    return lr


lr_scheduler = callbacks.LearningRateScheduler(lr_schedule)

# Train the model
print("\n" + "=" * 80)
print("STARTING TEACHER-STUDENT TRAINING")
print("=" * 80)
pretraining_history = pretraining_model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=labeled_test_dataset,
    batch_size=batch_size,
    callbacks=[checkpoint, lr_scheduler],
    verbose=1,
)

# Load best weights
print("\nLoading best weights...")
pretraining_model.load_weights("typing_bimodal_best_model.weights.h5")

print(
    "Maximal contrastive accuracy: {:.2f}%".format(
        max(pretraining_history.history["c_acc"]) * 100
    )
)

print(
    "Minimum contrastive loss: {:.2f}".format(
        min(pretraining_history.history["c_loss"])
    )
)

# Save the typing encoder weights
pretraining_model.typing_encoder.save_weights("typing_bimodal_embeddings.weights.h5")
print("✓ Typing encoder weights saved to typing_bimodal_embeddings.weights.h5")

# Plot results
pretraining_model.plot_contrastive_loss(pretraining_history)
pretraining_model.plot_contrastive_accuracy(pretraining_history)


def get_labeled_embeddings(pretraining_model, labeled_dataset):
    """
    Generate embeddings for labeled data using the trained typing encoder.
    """
    typing_features = []
    labels = []

    # Iterate through the batched dataset and collect features and labels
    for batch in labeled_dataset:
        typing_batch = batch[0][0]  # Extract the typing features from the batch
        labels_batch = batch[1]  # Extract the labels from the batch

        # Predict the embeddings for the entire batch of typing features
        embeddings_batch = pretraining_model.typing_encoder.predict(
            typing_batch, verbose=0
        )

        # Append the embeddings and labels to the lists
        typing_features.append(embeddings_batch)
        labels.append(labels_batch.numpy())  # Convert TensorFlow tensor to numpy array

    # Stack the results to form a full matrix
    embeddings = np.vstack(typing_features)  # Convert list of arrays into a full array
    labels = np.hstack(labels)  # Flatten list of label arrays into a single array

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings, labels


def visualize_embeddings(
    embeddings, labels, n_components=2, perplexity=10, learning_rate="auto", n_iter=250
):
    """
    Visualize the embeddings using t-SNE, colored by their class labels.
    """
    # Initialize t-SNE model
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=42,
    )

    # Apply t-SNE to embeddings
    reduced_embeddings = tsne.fit_transform(embeddings)

    # 2D Visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(
        reduced_embeddings[labels == 0, 0],
        reduced_embeddings[labels == 0, 1],
        label="No FMI",
        c="b",
        alpha=0.5,
    )
    plt.scatter(
        reduced_embeddings[labels == 1, 0],
        reduced_embeddings[labels == 1, 1],
        label="FMI",
        c="r",
        alpha=0.5,
    )
    plt.title("2D t-SNE Visualization of Bimodal Typing Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.show()


# Prepare labeled dataset for embedding visualization
labeled_dataset_viz = tf.data.Dataset.from_tensor_slices(
    ((labeled_typing, labeled_accel), labeled_labels)
)
labeled_dataset_viz = (
    labeled_dataset_viz.shuffle(buffer_size=len(labeled_typing))
    .batch(10)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Use the function to get embeddings and visualize them
print("\nGenerating t-SNE visualization...")
embeddings, labels = get_labeled_embeddings(pretraining_model, labeled_dataset_viz)
visualize_embeddings(embeddings, labels, n_components=2)

print(f"\nTotal training time: {time.time() - start:.2f} seconds")

# Alarm
os.system('powershell.exe -c "[console]::beep(999,1000)"')

input("Press Enter to exit...")
