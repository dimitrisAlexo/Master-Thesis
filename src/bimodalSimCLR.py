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

# Create TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((typing_histograms, accel_windows))
dataset = (
    dataset.shuffle(buffer_size=len(typing_histograms))
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
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

        self.typing_encoder.summary()
        self.tremor_encoder.summary()
        self.projection_head.summary()

    def compile(self, contrastive_optimizer, **kwargs):
        super().compile(**kwargs)
        self.contrastive_optimizer = contrastive_optimizer
        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="c_acc"
        )

    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
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
        typing_data, accel_data = data

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

        # Update metrics
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        return {m.name: m.result() for m in self.metrics}

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
)

# Callbacks
checkpoint = callbacks.ModelCheckpoint(
    filepath="typing_bimodal_best_model.weights.h5",
    monitor="c_loss",
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
    dataset,
    epochs=num_epochs,
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

print(f"\nTotal training time: {time.time() - start:.2f} seconds")

# Alarm
os.system('powershell.exe -c "[console]::beep(999,1000)"')

input("Press Enter to exit...")
