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
        # Generate a random rotation matrix for each sample in the batch
        def rotate_single_sample(sample):
            # Generate a random axis for rotation (normalized) per sample
            axis = tf.random.uniform([3], minval=-1.0, maxval=1.0, dtype=tf.float32)
            axis = axis / tf.norm(axis)

            # Generate a random rotation angle per sample
            angle = tf.random.uniform(
                [],
                minval=-self.rotation_angle,
                maxval=self.rotation_angle,
                dtype=tf.float32,
            )

            # Compute components of the rotation matrix using the axis-angle formula
            cos_angle = tf.cos(angle)
            sin_angle = tf.sin(angle)
            one_minus_cos = 1.0 - cos_angle

            x, y, z = axis[0], axis[1], axis[2]

            # Rotation matrix for an arbitrary axis (Rodrigues' rotation formula)
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

            # Apply the rotation matrix to the sample
            sample = tf.cast(sample, tf.float32)
            return tf.linalg.matmul(sample, rotation_matrix)

        # Apply the rotate_single_sample function to each sample in the batch using tf.map_fn
        rotated_batch = tf.map_fn(rotate_single_sample, data, dtype=tf.float32)

        return rotated_batch

    def add_gravity(self, data):
        """
        Adds a random gravity component to the 3D accelerometer data.
        """

        def add_gravity_to_sample(sample):
            # Generate a random direction vector (normalized) for gravity
            gravity_direction = tf.random.uniform(
                [3], minval=-1.0, maxval=1.0, dtype=tf.float32
            )
            gravity_direction = gravity_direction / tf.norm(gravity_direction)

            # Calculate the gravity vector with the specified magnitude
            gravity_magnitude = self.gravity_factor * 10.0  # assuming g = 10 m/s^2
            gravity_vector = gravity_magnitude * gravity_direction

            # Add the gravity vector to each time step of the sample
            sample = tf.cast(sample, tf.float32)
            gravity_vector = tf.cast(gravity_vector, tf.float32)
            return sample + gravity_vector

        # Apply the add_gravity_to_sample function to each sample in the batch using tf.map_fn
        gravity_augmented_batch = tf.map_fn(
            add_gravity_to_sample, data, dtype=tf.float32
        )

        return gravity_augmented_batch

    def permute_segments(self, data):
        """
        Permute segments of the input data along the time axis.
        """
        batch_size, time_steps, channels = (
            tf.shape(data)[0],
            tf.shape(data)[1],
            tf.shape(data)[2],
        )

        # Calculate the divisor and remainder
        divisor = time_steps // self.n_perm_seg
        remainder = time_steps % self.n_perm_seg

        # tf.print("divisor: ", divisor)
        # tf.print("remainder: ", remainder)

        # Reshape the first n_perm_seg - 1 segments with size divisor
        reshaped_data_1 = tf.reshape(
            data[:, : divisor * (self.n_perm_seg - 1), :],
            [batch_size, self.n_perm_seg - 1, divisor, channels],
        )

        # tf.print("reshaped_data_1 shape: ", tf.shape(reshaped_data_1))

        # Reshape the last segment to include the remainder (divisor + remainder)
        last_segment_start = divisor * (self.n_perm_seg - 1)
        reshaped_data_2 = tf.reshape(
            data[:, last_segment_start:, :], [batch_size, divisor + remainder, channels]
        )

        # tf.print("reshaped_data_2 shape: ", tf.shape(reshaped_data_2))

        # Generate a random permutation of the segment indices for each sample in the batch
        permuted_indices = tf.map_fn(
            lambda _: tf.random.shuffle(tf.range(self.n_perm_seg - 1)),
            tf.zeros([batch_size], dtype=tf.int32),
            fn_output_signature=tf.int32,
        )

        # Gather the segments in the new permuted order for each batch
        permuted_data = tf.map_fn(
            lambda x: tf.gather(x[0], x[1]),
            (reshaped_data_1, permuted_indices),
            fn_output_signature=tf.float32,
        )

        # Reshape back to the original shape (batch_size, time_steps, channels)
        permuted_data = tf.reshape(
            permuted_data, [batch_size, time_steps - divisor - remainder, channels]
        )

        # tf.print("permuted_data shape: ", tf.shape(permuted_data))

        permuted_data = tf.concat([permuted_data, reshaped_data_2], axis=1)

        # tf.print("permuted_data shape: ", tf.shape(permuted_data))

        return permuted_data

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
