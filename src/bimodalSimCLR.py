"""
## Bimodal FOCAL: Factorized Orthogonal Contrastive Learning
## Joint training of typing and tremor encoders from pretrained initialization
## Uses shared/private space decomposition with orthogonality constraints
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
from augmentations import Augmentation

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
# policy = mixed_precision.Policy("mixed_float16")
# mixed_precision.set_global_policy(policy)
# print("Using mixed precision...")

"""
## Hyperparameter setup
"""

dataset_size = 10240
M = 64
batch_size = 256
labeled_batch_size = 2
num_epochs = 500
temperature = 0.01
learning_rate = 1e-3
probe_learning_rate = 2e-4

# FOCAL hyperparameters (based on FOCAL paper)
lambda_shared = 1.0  # Main cross-modal alignment objective
lambda_private = 0.5  # Within-modality augmentation consistency
lambda_orthogonal = 0.1  # Orthogonality regularization

"""
## Lightweight Augmentations for FOCAL L_private
"""


class LightweightTypingAugmentation:
    """Lightweight augmentations for typing histograms in FOCAL private space."""

    def __init__(
        self,
        noise_factor=0.02,  # Increased from 0.01 for stronger augmentation
        dropout_rate=0.01,  # Increased from 0.005
        scale_range=(0.9, 1.1),  # Increased from (0.95, 1.05)
        n_perm_seg=8,  # Number of permutation segments
    ):
        self.noise_factor = noise_factor
        self.dropout_rate = dropout_rate
        self.scale_range = scale_range
        self.n_perm_seg = n_perm_seg

    def add_noise(self, data):
        """Add light Gaussian noise to typing histograms."""
        # Cast data to float32 to match noise dtype
        data = tf.cast(data, tf.float32)
        noise = tf.random.normal(
            tf.shape(data), mean=0.0, stddev=self.noise_factor, dtype=tf.float32
        )
        return data + noise

    def dropout_features(self, data):
        """Randomly zero out some features in the histogram."""
        mask = tf.random.uniform(tf.shape(data)) > self.dropout_rate
        return data * tf.cast(mask, tf.float32)

    def random_scaling(self, data):
        """Randomly scale histogram magnitude per sample."""
        batch_size = tf.shape(data)[0]
        scale = tf.random.uniform(
            [batch_size, 1],
            minval=self.scale_range[0],
            maxval=self.scale_range[1],
            dtype=tf.float32,
        )
        return data * scale

    def normalize_histogram(self, data):
        """
        Normalize histograms so that each section sums to 1.
        - Hold time section (0-100): normalized to sum to 1
        - Flight time section (101-501): normalized to sum to 1
        """
        hold_time_data = data[:, :101]
        flight_time_data = data[:, 101:]

        hold_time_sum = tf.reduce_sum(hold_time_data, axis=1, keepdims=True)
        hold_time_sum = tf.maximum(hold_time_sum, 1e-8)
        normalized_hold_time = hold_time_data / hold_time_sum

        flight_time_sum = tf.reduce_sum(flight_time_data, axis=1, keepdims=True)
        flight_time_sum = tf.maximum(flight_time_sum, 1e-8)
        normalized_flight_time = flight_time_data / flight_time_sum

        return tf.concat([normalized_hold_time, normalized_flight_time], axis=1)

    def random_permutation(self, data):
        """
        Randomly permute chunks of histogram features.
        Permutes hold time (0-100) and flight time (101-501) sections separately.
        """
        batch_size = tf.shape(data)[0]
        hold_time_data = data[:, :101]
        flight_time_data = data[:, 101:]

        # Calculate chunk size based on n_perm_seg
        hold_chunk_size = 101 // self.n_perm_seg
        flight_chunk_size = 401 // self.n_perm_seg

        # Permute hold time chunks
        num_hold_chunks = 101 // hold_chunk_size
        hold_chunks = tf.reshape(
            hold_time_data[:, : num_hold_chunks * hold_chunk_size],
            [batch_size, num_hold_chunks, hold_chunk_size],
        )
        # Random permutation per sample
        perm_indices = tf.argsort(
            tf.random.uniform([batch_size, num_hold_chunks]), axis=1
        )
        hold_chunks_permuted = tf.gather(hold_chunks, perm_indices, batch_dims=1)
        hold_permuted = tf.reshape(hold_chunks_permuted, [batch_size, -1])
        hold_remainder = hold_time_data[:, num_hold_chunks * hold_chunk_size :]
        hold_final = tf.concat([hold_permuted, hold_remainder], axis=1)

        # Permute flight time chunks
        num_flight_chunks = 401 // flight_chunk_size
        flight_chunks = tf.reshape(
            flight_time_data[:, : num_flight_chunks * flight_chunk_size],
            [batch_size, num_flight_chunks, flight_chunk_size],
        )
        perm_indices = tf.argsort(
            tf.random.uniform([batch_size, num_flight_chunks]), axis=1
        )
        flight_chunks_permuted = tf.gather(flight_chunks, perm_indices, batch_dims=1)
        flight_permuted = tf.reshape(flight_chunks_permuted, [batch_size, -1])
        flight_remainder = flight_time_data[:, num_flight_chunks * flight_chunk_size :]
        flight_final = tf.concat([flight_permuted, flight_remainder], axis=1)

        return tf.concat([hold_final, flight_final], axis=1)

    def __call__(self, data):
        """Apply augmentation pipeline."""
        data = self.add_noise(data)
        data = self.dropout_features(data)
        data = self.random_scaling(data)
        data = self.random_permutation(data)
        data = self.normalize_histogram(data)
        return data


class LightweightTremorAugmentation:
    """Lightweight augmentations for tremor accelerometer data in FOCAL private space."""

    def __init__(
        self,
        flip_probability=0.5,  # Increased from 0.3 for stronger augmentation
        rotation_angle=np.pi / 3,  # Increased from pi/4 (45° -> 60°)
        noise_factor=0.02,  # Increased from 0.01
        n_perm_seg=8,  # Number of permutation segments
    ):
        self.flip_probability = flip_probability
        self.rotation_angle = rotation_angle
        self.noise_factor = noise_factor
        self.n_perm_seg = n_perm_seg

    def add_noise(self, data):
        """Add light Gaussian noise."""
        # Cast data to float32 to match noise dtype
        data = tf.cast(data, tf.float32)
        noise = tf.random.normal(tf.shape(data), mean=0.0, stddev=self.noise_factor)
        return data + noise

    def bidirectional_flipping(self, data):
        """Flip accelerometer axes with probability."""
        batch_size = tf.shape(data)[0]
        random_mask = tf.random.uniform((batch_size, 1, 1), minval=0.0, maxval=1.0)
        flip_mask = random_mask < self.flip_probability
        flipped_data = data * -1
        return tf.where(flip_mask, flipped_data, data)

    def rotate_axis(self, data):
        """Light rotation around random axis."""

        def rotate_single_sample(sample):
            axis = tf.random.uniform([3], minval=-1.0, maxval=1.0, dtype=tf.float32)
            axis = axis / tf.norm(axis)
            angle = tf.random.uniform(
                [],
                minval=-self.rotation_angle,
                maxval=self.rotation_angle,
                dtype=tf.float32,
            )

            cos_angle = tf.cos(angle)
            sin_angle = tf.sin(angle)
            one_minus_cos = 1.0 - cos_angle
            x, y, z = axis[0], axis[1], axis[2]

            rotation_matrix = tf.convert_to_tensor(
                [
                    [
                        cos_angle + x * x * one_minus_cos,
                        x * y * one_minus_cos - z * sin_angle,
                        x * z * one_minus_cos + y * sin_angle,
                    ],
                    [
                        y * x * one_minus_cos + z * sin_angle,
                        cos_angle + y * y * one_minus_cos,
                        y * z * one_minus_cos - x * sin_angle,
                    ],
                    [
                        z * x * one_minus_cos - y * sin_angle,
                        z * y * one_minus_cos + x * sin_angle,
                        cos_angle + z * z * one_minus_cos,
                    ],
                ],
                dtype=tf.float32,
            )

            return tf.matmul(sample, rotation_matrix)

        return tf.map_fn(rotate_single_sample, data)

    def temporal_permutation(self, data):
        """
        Randomly permute temporal segments of accelerometer data.
        Divides the 1000 timesteps into n_perm_seg segments and permutes their order.
        """
        batch_size = tf.shape(data)[0]
        timesteps = tf.shape(data)[1]
        channels = tf.shape(data)[2]

        # Calculate segment length based on n_perm_seg
        segment_length = timesteps // self.n_perm_seg

        # Reshape into segments: [batch, num_segments, segment_length, channels]
        num_segments = timesteps // segment_length
        segments = tf.reshape(
            data[:, : num_segments * segment_length, :],
            [batch_size, num_segments, segment_length, channels],
        )

        # Generate random permutation indices for each sample in batch
        perm_indices = tf.argsort(tf.random.uniform([batch_size, num_segments]), axis=1)

        # Apply permutation
        segments_permuted = tf.gather(segments, perm_indices, batch_dims=1)

        # Reshape back to original shape
        data_permuted = tf.reshape(
            segments_permuted, [batch_size, num_segments * segment_length, channels]
        )

        # Concatenate with remainder if any
        remainder = data[:, num_segments * segment_length :, :]
        return tf.concat([data_permuted, remainder], axis=1)

    def __call__(self, data):
        """Apply augmentation pipeline."""
        data = self.add_noise(data)
        data = self.bidirectional_flipping(data)
        data = self.rotate_axis(data)
        data = self.temporal_permutation(data)
        return data


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
print("Note: Typing histograms are NOT normalized (already sum to 2 as per design)")

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


# Simple accel encoder - symmetric with typing encoder
def tremor_encoder(M):
    return keras.Sequential(
        [
            # Conv layers to extract temporal patterns
            layers.Conv1D(filters=32, kernel_size=16, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.MaxPooling1D(pool_size=4),  # 1000 -> 250
            layers.Conv1D(filters=16, kernel_size=8, padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.GlobalAveragePooling1D(),
            # Dense layers matching typing encoder
            layers.Dense(100),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.1),
            layers.Dense(50),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.1),
            layers.Dense(M),
        ],
        name="accel_encoder",
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
## FOCAL Bimodal Contrastive Model
"""


class BimodalContrastiveModel(keras.Model):
    def __init__(self):
        super().__init__()

        self.temperature = temperature

        # Both encoders for typing data (student-student symmetric setup)
        self.typing_encoder = typing_encoder(M)
        self.tremor_encoder = tremor_encoder(M)

        # Build the tremor encoder by passing a dummy input
        dummy_accel_input = tf.zeros((1, 1000, 3))
        _ = self.tremor_encoder(dummy_accel_input)

        # Both encoders trainable with random initialization
        self.tremor_encoder.trainable = True
        self.typing_encoder.trainable = True

        # Initialize lightweight augmenters for L_private
        self.typing_augmenter = LightweightTypingAugmentation()
        self.tremor_augmenter = LightweightTremorAugmentation()

        # FOCAL-style MLP projectors: Split into shared and private spaces
        # Shared space projector for typing (for cross-modal consistency)
        self.typing_shared_projector = keras.Sequential(
            [
                layers.Dense(2 * M),
                layers.LeakyReLU(negative_slope=0.2),
                layers.Dense(M // 2),  # Shared space dimension
            ],
            name="typing_shared_projector",
        )

        # Private space projector for typing (for augmentation consistency)
        self.typing_private_projector = keras.Sequential(
            [
                layers.Dense(2 * M),
                layers.LeakyReLU(negative_slope=0.2),
                layers.Dense(M // 2),  # Private space dimension
            ],
            name="typing_private_projector",
        )

        # Shared space projector for tremor
        self.tremor_shared_projector = keras.Sequential(
            [
                layers.Dense(2 * M),
                layers.LeakyReLU(negative_slope=0.2),
                layers.Dense(M // 2),
            ],
            name="tremor_shared_projector",
        )

        # Private space projector for tremor
        self.tremor_private_projector = keras.Sequential(
            [
                layers.Dense(2 * M),
                layers.LeakyReLU(negative_slope=0.2),
                layers.Dense(M // 2),
            ],
            name="tremor_private_projector",
        )

        # Linear probe for classification on concatenated embeddings
        # Input is 2*M because we concatenate typing and tremor embeddings
        self.linear_probe = keras.Sequential(
            [
                layers.Input(shape=(2 * M,)),  # Concatenated: typing + tremor
                layers.Dropout(0.1),
                layers.Dense(2, kernel_regularizer=keras.regularizers.L2(1e-4)),
            ],
            name="linear_probe",
        )

        self.typing_encoder.summary()
        self.tremor_encoder.summary()
        self.typing_shared_projector.summary()
        self.typing_private_projector.summary()
        self.tremor_shared_projector.summary()
        self.tremor_private_projector.summary()
        self.linear_probe.summary()

    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super().compile(**kwargs)
        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer

        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # FOCAL loss component trackers
        self.shared_loss_tracker = keras.metrics.Mean(name="shared_loss")
        self.private_loss_tracker = keras.metrics.Mean(name="private_loss")
        self.orthogonal_loss_tracker = keras.metrics.Mean(name="ortho_loss")
        self.contrastive_loss_tracker = keras.metrics.Mean(
            name="c_loss"
        )  # Total contrastive
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="c_acc"
        )

        self.probe_loss_tracker = keras.metrics.Mean(name="p_loss")
        self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy(name="p_acc")

    @property
    def metrics(self):
        return [
            self.shared_loss_tracker,
            self.private_loss_tracker,
            self.orthogonal_loss_tracker,
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
            self.probe_loss_tracker,
            self.probe_accuracy,
        ]

    def shared_space_loss(self, h_shared_typing, h_shared_tremor):
        """
        L_shared: InfoNCE loss for cross-modal consistency in shared space.
        Positive pairs: (h_shared_typing[i], h_shared_tremor[i]) - same time, different modality
        Negative pairs: All other cross-modal pairs in the batch
        """
        # L2 normalize
        h_shared_typing = tf.nn.l2_normalize(h_shared_typing, axis=1)
        h_shared_tremor = tf.nn.l2_normalize(h_shared_tremor, axis=1)

        # Compute similarity matrix
        similarities = (
            ops.matmul(h_shared_typing, ops.transpose(h_shared_tremor))
            / self.temperature
        )

        batch_size = ops.shape(h_shared_typing)[0]
        labels = ops.arange(batch_size)

        # Symmetric InfoNCE loss
        loss_typing_to_tremor = keras.losses.sparse_categorical_crossentropy(
            labels, similarities, from_logits=True
        )
        loss_tremor_to_typing = keras.losses.sparse_categorical_crossentropy(
            labels, ops.transpose(similarities), from_logits=True
        )

        return (loss_typing_to_tremor + loss_tremor_to_typing) / 2

    def private_space_loss(self, h_private_1, h_private_2):
        """
        L_private: NT-Xent loss for augmentation consistency in private space.
        Positive pairs: (h_private[i], h_private_aug[i]) - same time, same modality, different augmentation
        Negative pairs: All other samples in the batch (2B - 2 negatives per anchor)
        """
        # L2 normalize
        h_private_1 = tf.nn.l2_normalize(h_private_1, axis=1)
        h_private_2 = tf.nn.l2_normalize(h_private_2, axis=1)

        batch_size = ops.shape(h_private_1)[0]

        # Compute similarities between augmented versions
        # Positive similarities: h_private_1[i] · h_private_2[i]
        positive_sim = (
            ops.sum(h_private_1 * h_private_2, axis=1, keepdims=True) / self.temperature
        )

        # Negative similarities: h_private_1[i] · h_private_1[j] (j ≠ i) and h_private_1[i] · h_private_2[j]
        neg_sim_1 = (
            ops.matmul(h_private_1, ops.transpose(h_private_1)) / self.temperature
        )
        neg_sim_2 = (
            ops.matmul(h_private_1, ops.transpose(h_private_2)) / self.temperature
        )

        # Mask out diagonal for neg_sim_1 (self-similarity)
        mask = 1 - tf.eye(batch_size)
        neg_sim_1 = neg_sim_1 * mask

        # Concatenate all similarities: [positive | negatives_1 | negatives_2]
        # Shape: [batch_size, 1 + (batch_size - 1) + batch_size]
        logits = ops.concatenate([positive_sim, neg_sim_1, neg_sim_2], axis=1)

        # Labels are 0 (first position is positive)
        labels = ops.zeros(batch_size, dtype="int32")

        # Compute cross-entropy loss
        loss = keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True
        )

        return loss

    def orthogonality_loss(
        self, h_shared_typing, h_private_typing, h_shared_tremor, h_private_tremor
    ):
        """
        L_orthogonal: Enforce orthogonality between:
        1. Shared and private of same modality: <h_shared_typing, h_private_typing>
        2. Private of different modalities: <h_private_typing, h_private_tremor>
        """
        # Orthogonality between shared and private of same modality
        ortho_typing = ops.sum(h_shared_typing * h_private_typing, axis=1)
        ortho_tremor = ops.sum(h_shared_tremor * h_private_tremor, axis=1)

        # Orthogonality between private spaces of different modalities
        ortho_cross = ops.sum(h_private_typing * h_private_tremor, axis=1)

        # Sum of absolute cosine similarities (want them to be 0)
        loss = (
            ops.mean(ops.abs(ortho_typing))
            + ops.mean(ops.abs(ortho_tremor))
            + ops.mean(ops.abs(ortho_cross))
        )

        return loss

    def train_step(self, data):
        # Unpack unlabeled and labeled data
        unlabeled_data = data[0]
        labeled_data = data[1]

        typing_data, accel_data = unlabeled_data
        (labeled_typing, labeled_accel), labels = labeled_data

        # FOCAL contrastive learning step
        with tf.GradientTape() as tape:
            # === Original embeddings (for shared space) ===
            typing_embeddings = self.typing_encoder(typing_data, training=True)
            tremor_embeddings = self.tremor_encoder(accel_data, training=True)

            # Project to shared space (both projectors trainable)
            h_shared_typing = self.typing_shared_projector(
                typing_embeddings, training=True
            )
            h_shared_tremor = self.tremor_shared_projector(
                tremor_embeddings, training=True  # Changed to True
            )

            # === Augmented embeddings (for private space) ===
            # Use lightweight versions of typing and tremor augmentations
            typing_aug = self.typing_augmenter(typing_data)
            accel_aug = self.tremor_augmenter(accel_data)

            typing_embeddings_aug = self.typing_encoder(typing_aug, training=True)
            tremor_embeddings_aug = self.tremor_encoder(accel_aug, training=True)

            # Project to private space (original and augmented)
            h_private_typing = self.typing_private_projector(
                typing_embeddings, training=True
            )
            h_private_typing_aug = self.typing_private_projector(
                typing_embeddings_aug, training=True
            )

            h_private_tremor = self.tremor_private_projector(
                tremor_embeddings, training=True  # Changed to True
            )
            h_private_tremor_aug = self.tremor_private_projector(
                tremor_embeddings_aug, training=True  # Changed to True
            )

            # === Compute FOCAL losses ===
            # L_shared: Cross-modal consistency in shared space
            loss_shared = self.shared_space_loss(h_shared_typing, h_shared_tremor)

            # L_private: Augmentation consistency in private space (per modality)
            loss_private_typing = self.private_space_loss(
                h_private_typing, h_private_typing_aug
            )
            loss_private_tremor = self.private_space_loss(
                h_private_tremor, h_private_tremor_aug
            )
            loss_private = (loss_private_typing + loss_private_tremor) / 2

            # L_orthogonal: Enforce orthogonality constraints
            loss_orthogonal = self.orthogonality_loss(
                h_shared_typing, h_private_typing, h_shared_tremor, h_private_tremor
            )

            # Total contrastive loss
            contrastive_loss = (
                lambda_shared * loss_shared
                + lambda_private * loss_private
                + lambda_orthogonal * loss_orthogonal
            )

        # Compute gradients for both encoders and all projectors
        trainable_weights = (
            self.typing_encoder.trainable_weights
            + self.tremor_encoder.trainable_weights
            + self.typing_shared_projector.trainable_weights
            + self.typing_private_projector.trainable_weights
            + self.tremor_shared_projector.trainable_weights
            + self.tremor_private_projector.trainable_weights
        )
        gradients = tape.gradient(contrastive_loss, trainable_weights)
        self.contrastive_optimizer.apply_gradients(zip(gradients, trainable_weights))

        # Update FOCAL metrics
        self.shared_loss_tracker.update_state(loss_shared)
        self.private_loss_tracker.update_state(loss_private)
        self.orthogonal_loss_tracker.update_state(loss_orthogonal)
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        # Update contrastive accuracy (based on shared space cross-modal similarity)
        self.contrastive_accuracy.update_state(
            tf.range(tf.shape(h_shared_typing)[0]),
            tf.matmul(
                tf.nn.l2_normalize(h_shared_typing, axis=1),
                tf.nn.l2_normalize(h_shared_tremor, axis=1),
                transpose_b=True,
            )
            / self.temperature,
        )

        # Linear probe step with fusion of both modalities
        with tf.GradientTape() as tape:
            # Encode labeled data from both modalities (both in inference mode)
            labeled_typing_embeddings = self.typing_encoder(
                labeled_typing, training=False
            )
            labeled_tremor_embeddings = self.tremor_encoder(
                labeled_accel, training=False
            )
            # Concatenate embeddings from both modalities
            fused_embeddings = tf.concat(
                [labeled_typing_embeddings, labeled_tremor_embeddings], axis=1
            )
            # Classify on fused representation
            class_logits = self.linear_probe(fused_embeddings, training=True)
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

        # Encode from both modalities (inference mode)
        labeled_typing_embeddings = self.typing_encoder(labeled_typing, training=False)
        labeled_tremor_embeddings = self.tremor_encoder(labeled_accel, training=False)
        # Concatenate embeddings
        fused_embeddings = tf.concat(
            [labeled_typing_embeddings, labeled_tremor_embeddings], axis=1
        )
        # Classify on fused representation
        class_logits = self.linear_probe(fused_embeddings, training=False)
        probe_loss = self.probe_loss(labels, class_logits)

        # Update only probe metrics
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)

        # Return only probe metrics (don't include contrastive metrics at all)
        return {
            self.probe_loss_tracker.name: self.probe_loss_tracker.result(),
            self.probe_accuracy.name: self.probe_accuracy.result(),
        }

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
        plt.title("FOCAL Contrastive Loss per Epoch")
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
        plt.title("FOCAL Contrastive Accuracy per Epoch")
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
print("INITIALIZING BIMODAL FOCAL MODEL")
print("=" * 80)
pretraining_model = BimodalContrastiveModel()
pretraining_model.compile(
    contrastive_optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    probe_optimizer=keras.optimizers.Adam(learning_rate=probe_learning_rate),
)

# Callbacks
checkpoint = callbacks.ModelCheckpoint(
    filepath="weights/fusion/typing_bimodal_best_model.weights.h5",
    monitor="val_p_loss",  # Changed from c_loss - care about downstream task
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
        return lr * 1.0
    return lr


lr_scheduler = callbacks.LearningRateScheduler(lr_schedule)

# Train the model
print("\n" + "=" * 80)
print("STARTING FOCAL BIMODAL TRAINING")
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
pretraining_model.load_weights("weights/fusion/typing_bimodal_best_model.weights.h5")

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
pretraining_model.typing_encoder.save_weights("weights/fusion/typing_bimodal_embeddings.weights.h5")
print("✓ Typing encoder weights saved to weights/fusion/typing_bimodal_embeddings.weights.h5")

# Plot results
pretraining_model.plot_contrastive_loss(pretraining_history)
pretraining_model.plot_contrastive_accuracy(pretraining_history)


def get_labeled_embeddings(pretraining_model, labeled_dataset):
    """
    Generate fused embeddings (typing + tremor) for labeled data.
    This matches what the linear probe sees during training.
    """
    fused_features = []
    labels = []

    # Iterate through the batched dataset and collect features and labels
    for batch in labeled_dataset:
        typing_batch = batch[0][0]  # Extract the typing features from the batch
        accel_batch = batch[0][1]  # Extract the accelerometer features from the batch
        labels_batch = batch[1]  # Extract the labels from the batch

        # Get embeddings from both encoders
        typing_embeddings = pretraining_model.typing_encoder.predict(
            typing_batch, verbose=0
        )
        tremor_embeddings = pretraining_model.tremor_encoder.predict(
            accel_batch, verbose=0
        )

        # Concatenate embeddings (same as what linear probe uses)
        fused_embeddings = np.concatenate(
            [typing_embeddings, tremor_embeddings], axis=1
        )

        # Append the fused embeddings and labels to the lists
        fused_features.append(fused_embeddings)
        labels.append(labels_batch.numpy())  # Convert TensorFlow tensor to numpy array

    # Stack the results to form a full matrix
    embeddings = np.vstack(fused_features)  # Convert list of arrays into a full array
    labels = np.hstack(labels)  # Flatten list of label arrays into a single array

    print(f"Fused embeddings shape: {embeddings.shape}")
    return embeddings, labels


def visualize_embeddings(
    embeddings, labels, n_components=2, perplexity=10, learning_rate="auto", n_iter=500
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
