"""
## Hyperparameter setup
"""

import numpy as np
import tensorflow as tf
import keras
from keras import layers

unlabeled_dataset_size = 5120
labeled_dataset_size = 450

M = 64
E_thres = 0.15 * 3
Kt = 100
batch_size = 512
labeled_gdataset_batch_size = 45
num_epochs = 200
temperature = 0.01
learning_rate = 0.001


class Augmentation:
    def __init__(
        self,
        overlap=0.90,
        flip_probability=0.5,
        rotation_angle=np.pi,
        gravity_factor=0.1,
        n_perm_seg=10,
    ):
        # Set default parameters for each augmentation
        self.overlap = overlap
        self.flip_probability = flip_probability
        self.rotation_angle = rotation_angle
        self.gravity_factor = gravity_factor
        self.n_perm_seg = n_perm_seg

    def left_to_right_flipping(self, data):
        """
        Perform left-to-right flipping of 3D accelerometer data with a probability.
        """
        # Generate a random boolean mask of shape (batch_size, 1, 1)
        batch_size = tf.shape(data)[0]
        random_mask = tf.random.uniform((batch_size, 1, 1), minval=0.0, maxval=1.0)
        flip_mask = random_mask < self.flip_probability

        # Flip only the windows that have True in the flip_mask
        flipped_data = tf.reverse(data, axis=[1])
        output = tf.where(flip_mask, flipped_data, data)

        return output

    def bidirectional_flipping(self, data):
        """
        Perform bidirectional flipping of 3D accelerometer data.
        The time-series data is mirrored along the channel axis (axis 2).
        """
        batch_size = tf.shape(data)[0]
        random_mask = tf.random.uniform((batch_size, 1, 1), minval=0.0, maxval=1.0)
        flip_mask = random_mask < self.flip_probability

        # Flip only the windows that have True in the flip_mask
        flipped_data = data * -1
        output = tf.where(flip_mask, flipped_data, data)

        return output

    def rotate_axis(self, data):
        data = tf.cast(data, tf.float32)
        batch_size = tf.shape(data)[0]

        # Sample a random unit axis and angle for every batch element at once —
        # no per-sample Python loop, fully XLA-compatible.
        axes = tf.random.uniform([batch_size, 3], minval=-1.0, maxval=1.0, dtype=tf.float32)
        axes = axes / tf.norm(axes, axis=1, keepdims=True)
        angles = tf.random.uniform(
            [batch_size], minval=-self.rotation_angle, maxval=self.rotation_angle, dtype=tf.float32
        )

        cos_a = tf.cos(angles)      # (B,)
        sin_a = tf.sin(angles)      # (B,)
        omc   = 1.0 - cos_a         # (B,)
        x, y, z = axes[:, 0], axes[:, 1], axes[:, 2]

        # Build (B, 3, 3) rotation matrices via Rodrigues' formula (vectorised)
        row0 = tf.stack([cos_a + x*x*omc,   x*y*omc - z*sin_a, x*z*omc + y*sin_a], axis=1)
        row1 = tf.stack([y*x*omc + z*sin_a, cos_a + y*y*omc,   y*z*omc - x*sin_a], axis=1)
        row2 = tf.stack([z*x*omc - y*sin_a, z*y*omc + x*sin_a, cos_a + z*z*omc  ], axis=1)
        R = tf.stack([row0, row1, row2], axis=1)  # (B, 3, 3)

        # data: (B, T, 3)  @  R: (B, 3, 3)  →  (B, T, 3)
        return tf.matmul(data, R)

    def add_gravity(self, data):
        """Adds a random gravity component to the 3D accelerometer data."""
        data = tf.cast(data, tf.float32)
        batch_size = tf.shape(data)[0]

        # Generate one gravity vector per sample, all at once (no map_fn)
        dirs = tf.random.uniform([batch_size, 3], minval=-1.0, maxval=1.0, dtype=tf.float32)
        dirs = dirs / tf.norm(dirs, axis=1, keepdims=True)
        gravity = self.gravity_factor * 10.0 * dirs  # (B, 3)

        # Broadcast over time axis: (B, 1, 3) added to (B, T, 3)
        return data + gravity[:, tf.newaxis, :]

    def permute_segments(self, data):
        """Permute segments of the input data along the time axis."""
        batch_size = tf.shape(data)[0]
        time_steps = tf.shape(data)[1]
        channels   = tf.shape(data)[2]
        n_full     = self.n_perm_seg - 1  # Python int — known at trace time

        divisor  = time_steps // self.n_perm_seg
        remainder = time_steps % self.n_perm_seg

        # (B, n_full, divisor, C)
        segments = tf.reshape(
            data[:, : divisor * n_full, :],
            [batch_size, n_full, divisor, channels],
        )
        last = data[:, divisor * n_full:, :]  # (B, divisor+remainder, C)

        # Vectorised batch permutation: argsort of uniform noise → random perm per sample
        perm_indices = tf.argsort(tf.random.uniform([batch_size, n_full]), axis=1)  # (B, n_full)
        batch_idx  = tf.tile(tf.range(batch_size)[:, tf.newaxis], [1, n_full])
        gather_idx = tf.stack([batch_idx, perm_indices], axis=2)   # (B, n_full, 2)
        permuted   = tf.gather_nd(segments, gather_idx)            # (B, n_full, divisor, C)

        permuted_flat = tf.reshape(permuted, [batch_size, divisor * n_full, channels])
        return tf.concat([permuted_flat, last], axis=1)

    def shift_windows_fun(self, data):
        """
        Extracts two overlapping windows of size (500, 3) from each sample in the input data.
        """
        batch_size, time_steps, channels = (
            tf.shape(data)[0],
            tf.shape(data)[1],
            tf.shape(data)[2],
        )
        window_size = 1000
        overlap_size = int(self.overlap * window_size)
        max_start = time_steps - window_size - (window_size - overlap_size)

        # Generate random starting index for the first window
        start_indices = tf.random.uniform(
            shape=[batch_size], minval=0, maxval=max_start + 1, dtype=tf.int32
        )

        # Calculate the start index for the second window with 10% overlap
        second_start_indices = start_indices + (window_size - overlap_size)

        # Create index ranges for the first and second windows
        indices_1 = tf.range(window_size)[tf.newaxis, :] + start_indices[:, tf.newaxis]
        indices_2 = (
            tf.range(window_size)[tf.newaxis, :] + second_start_indices[:, tf.newaxis]
        )

        # Extract the two windows
        window_batch_1 = tf.gather(data, indices_1, axis=1, batch_dims=1)
        window_batch_2 = tf.gather(data, indices_2, axis=1, batch_dims=1)

        return window_batch_1, window_batch_2

    def shift_windows(self):
        return layers.Lambda(self.shift_windows_fun)

    class CustomNormalizer(layers.Layer):
        def call(self, inputs):
            """
            Normalize inputs between 0 and 1 for each sample in the batch.
            """
            min_val = tf.reduce_min(
                inputs, axis=[1, 2], keepdims=True
            )  # Minimum value per batch sample
            max_val = tf.reduce_max(
                inputs, axis=[1, 2], keepdims=True
            )  # Maximum value per batch sample

            # Normalize each batch sample independently
            normalized = 2 * (inputs - min_val) / (max_val - min_val + 1e-8) - 1

            return normalized

    def get_contrastive_augmenter(self):
        """Combine several augmentations into a single sequential model."""
        return keras.Sequential(
            [
                layers.Lambda(self.left_to_right_flipping),
                layers.Lambda(self.bidirectional_flipping),
                layers.Lambda(self.rotate_axis),
                layers.Lambda(self.add_gravity),
                layers.Lambda(self.permute_segments),
                self.CustomNormalizer(),
            ]
        )

    def get_classification_augmenter(self):
        return keras.Sequential(
            [
                layers.Lambda(self.left_to_right_flipping),
                layers.Lambda(self.bidirectional_flipping),
                layers.Lambda(self.rotate_axis),
                # layers.Lambda(self.add_gravity),
                layers.Lambda(self.permute_segments),
                self.CustomNormalizer(),
            ]
        )
