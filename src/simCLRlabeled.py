"""
## Setup
"""

import os
import time

import numpy as np
import sys
import random

start = time.time()

os.environ["KERAS_BACKEND"] = "tensorflow"

np.set_printoptions(threshold=sys.maxsize)

# Make sure we are able to handle large datasets
import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

import keras
from keras import ops
from keras import layers
from keras import callbacks
from tf_keras import mixed_precision

from sklearn.manifold import TSNE

from utils import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # run on CPU

os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'

if tf.config.list_physical_devices('GPU'):
    print("Using GPU...")
else:
    print("Using CPU...")

# Mixed precision policy
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print("Using mixed precision...")

"""
## Hyperparameter setup
"""

unlabeled_dataset_size = 5000
labeled_dataset_size = 500

M = 64
E_thres = 0.15 * 2
Kt = 100
batch_size = 200
labeled_gdataset_batch_size = 20
num_epochs = 200
temperature = 0.1
learning_rate = 0.001

"""
## Dataset
"""

# Adjust the paths to be relative to the current script location
# gdata_path = os.path.join('..', 'data', 'tremor_gdata.pickle')
# tremor_gdata = unpickle_data(gdata_path)
# gdataset = form_unlabeled_dataset(tremor_gdata, E_thres, Kt)

with open("unlabeled_data.pickle", 'rb') as f:
    gdataset = pkl.load(f)

print(np.shape(gdataset))

gdataset = normalize(gdataset)

gdataset = gdataset[:unlabeled_dataset_size]

print(np.shape(gdataset))

gdataset = tf.data.Dataset.from_tensor_slices(gdataset)
print("Length of gdataset: ", len(gdataset))
gdataset = gdataset.shuffle(buffer_size=len(gdataset)).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
# gdataset = gdataset.batch(batch_size).shuffle(buffer_size=int(len(gdataset) / batch_size)).prefetch(
#     buffer_size=tf.data.AUTOTUNE)

with open("labeled_windows_dataset.pickle", 'rb') as f:
    labeled_gdataset = pkl.load(f)

labeled_gdataset['X'] = labeled_gdataset['X'].apply(lambda window: normalize_window(window))

print(type(labeled_gdataset))
# labeled_gdataset = pd.concat([labeled_gdataset, labeled_gdataset], ignore_index=True)
print(np.shape(labeled_gdataset))

# labeled_gdataset = tf.data.Dataset.from_tensor_slices((list(labeled_gdataset['X']), list(labeled_gdataset['y'])))
# labeled_gdataset = labeled_gdataset.shuffle(buffer_size=len(labeled_gdataset)).batch(5).prefetch(
#     buffer_size=tf.data.AUTOTUNE)
#
# print(labeled_gdataset.element_spec)
#
# train_dataset = tf.data.Dataset.zip(
#     (gdataset, labeled_gdataset)
# ).prefetch(buffer_size=tf.data.AUTOTUNE)

"""
## Augmentations
"""


class Augmentation:
    def __init__(self, overlap=0.95,
                 jitter_factor=0.1,
                 flip_probability=0.5,
                 rotation_angle=np.pi,
                 gravity_factor=0.1,
                 sliding_factor=0.1,
                 block_size_ratio=0.1,
                 crop_ratio=0.1,
                 lambda_amp=0.1,
                 n_perm_seg=5):
        # Set default parameters for each augmentation
        self.overlap = overlap
        self.jitter_factor = jitter_factor
        self.flip_probability = flip_probability
        self.rotation_angle = rotation_angle
        self.gravity_factor = gravity_factor
        self.sliding_factor = sliding_factor
        self.block_size_ratio = block_size_ratio
        self.crop_ratio = crop_ratio
        self.lambda_amp = lambda_amp
        self.n_perm_seg = n_perm_seg

    def shift_window(self, data):
        """
        Randomly picks a window of size (500, 3) from each sample in the batch.
        """
        batch_size, time_steps, channels = tf.shape(data)[0], tf.shape(data)[1], tf.shape(data)[2]
        window_size = 500

        # Generate a random start index for each sample in the batch
        start_indices = tf.random.uniform(shape=[batch_size], minval=0, maxval=(time_steps - window_size + 1),
                                          dtype=tf.int32)

        # Create an array of indices for selecting windows
        # This creates a range of indices for each sample
        indices = tf.range(window_size)[tf.newaxis, :] + start_indices[:, tf.newaxis]

        # Use tf.gather to extract windows based on the calculated indices
        shifted_data = tf.gather(data, indices, axis=1, batch_dims=1)

        return shifted_data

    def jitter(self, data):
        """Add random noise to the data."""
        noise = tf.random.normal(shape=tf.shape(data), mean=0.0, stddev=self.jitter_factor)
        return data + noise

    # def left_to_right_flipping(self, data):
    #     """
    #     Perform left-to-right flipping of 3D accelerometer data.
    #     The time-series data is reversed along the time axis (axis 1).
    #     """
    #     return tf.reverse(data, axis=[1])

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

    def random_channel_permutation(self, data):
        """
        Randomly permutes the channels (X, Y, Z) in the 3D accelerometer data.
        The time dimension remains unchanged during this augmentation.
        """
        channels = tf.shape(data)[2]

        # Generate a random permutation of channel indices
        permuted_indices = tf.random.shuffle(tf.range(channels))

        # Apply the permutation along the channel dimension
        return tf.gather(data, permuted_indices, axis=2)

    def rotate_axis(self, data):
        # Generate a random rotation matrix for each sample in the batch
        def rotate_single_sample(sample):
            # Generate a random axis for rotation (normalized) per sample
            axis = tf.random.uniform([3], minval=-1.0, maxval=1.0)
            axis = axis / tf.norm(axis)

            # Generate a random rotation angle per sample
            angle = tf.random.uniform([], minval=-self.rotation_angle, maxval=self.rotation_angle)

            # Compute components of the rotation matrix using the axis-angle formula
            cos_angle = tf.cos(angle)
            sin_angle = tf.sin(angle)
            one_minus_cos = 1.0 - cos_angle

            x, y, z = axis[0], axis[1], axis[2]

            # Rotation matrix for an arbitrary axis (Rodrigues' rotation formula)
            rotation_matrix = tf.convert_to_tensor([
                [cos_angle + x * x * one_minus_cos,
                 x * y * one_minus_cos - z * sin_angle,
                 x * z * one_minus_cos + y * sin_angle],

                [y * x * one_minus_cos + z * sin_angle,
                 cos_angle + y * y * one_minus_cos,
                 y * z * one_minus_cos - x * sin_angle],

                [z * x * one_minus_cos - y * sin_angle,
                 z * y * one_minus_cos + x * sin_angle,
                 cos_angle + z * z * one_minus_cos]
            ])

            # Apply the rotation matrix to the sample
            return tf.linalg.matmul(sample, rotation_matrix)

        # Apply the rotate_single_sample function to each sample in the batch using tf.map_fn
        rotated_batch = tf.map_fn(rotate_single_sample, data)

        return rotated_batch

    def add_gravity(self, data):
        """
        Adds a random gravity component to the 3D accelerometer data.
        """

        def add_gravity_to_sample(sample):
            # Generate a random direction vector (normalized) for gravity
            gravity_direction = tf.random.uniform([3], minval=-1.0, maxval=1.0)
            gravity_direction = gravity_direction / tf.norm(gravity_direction)

            # Calculate the gravity vector with the specified magnitude
            gravity_magnitude = self.gravity_factor * 10.0  # assuming g = 10 m/s^2
            gravity_vector = gravity_magnitude * gravity_direction

            # Add the gravity vector to each time step of the sample
            return sample + gravity_vector

        # Apply the add_gravity_to_sample function to each sample in the batch using tf.map_fn
        gravity_augmented_batch = tf.map_fn(add_gravity_to_sample, data)

        return gravity_augmented_batch

    def slide_window(self, data):
        """
        Randomly slides the values of the window left or right.
        Values that do not fit are wrapped around to the other side.
        """
        window_length = tf.shape(data)[1]

        def slide_single_sample(sample):
            # Calculate the maximum number of positions to slide based on sliding_factor
            max_slide = tf.cast(tf.round(self.sliding_factor * tf.cast(window_length, tf.float32)), tf.int32)

            # Generate a random sliding factor between -max_slide and +max_slide
            slide = tf.random.uniform([], minval=-max_slide, maxval=max_slide + 1, dtype=tf.int32)

            # Use tf.roll to shift the window with wrapping
            slid_sample = tf.roll(sample, shift=slide, axis=0)

            return slid_sample

        # Apply the slide_single_sample function to each sample in the batch using tf.map_fn
        slid_batch = tf.map_fn(slide_single_sample, data)

        return slid_batch

    def blockout(self, data):
        """
        Apply blockout augmentation by randomly setting a block of neighboring elements to zero.
        """
        batch_size, time_steps, channels = tf.shape(data)[0], tf.shape(data)[1], tf.shape(data)[2]
        block_size = tf.cast(self.block_size_ratio * tf.cast(time_steps, tf.float32), tf.int32)

        # Randomly select the starting index for the block for each sample in the batch
        block_start = tf.random.uniform(shape=(batch_size,), minval=0, maxval=time_steps - block_size, dtype=tf.int32)

        time_indices = tf.range(time_steps)

        block_start_expanded = tf.expand_dims(block_start, axis=-1)

        # Create the mask by zeroing out the block
        block_mask = tf.logical_and(time_indices >= block_start_expanded,
                                    time_indices < block_start_expanded + block_size)

        block_mask = tf.cast(tf.expand_dims(block_mask, axis=-1), tf.float32)

        data_masked = data * (1 - block_mask)

        return data_masked

    @tf.function
    def crop_and_resize(self, data):
        """
        Randomly crop a portion of the time-series data and resize it to the original length.
        """
        batch_size, time_steps, channels = tf.shape(data)[0], tf.shape(data)[1], tf.shape(data)[2]

        # Ensure at least half of the original time series is kept
        min_crop_size = tf.cast((1 - self.crop_ratio) * tf.cast(time_steps, tf.float32), tf.int32)
        max_crop_size = time_steps

        # Randomly select the crop size
        crop_size = tf.random.uniform(shape=(), minval=min_crop_size, maxval=max_crop_size, dtype=tf.int32)

        # Randomly select the starting index for the crop for each sample in the batch
        crop_start = tf.random.uniform(shape=(), minval=0, maxval=time_steps - crop_size, dtype=tf.int32)

        # Create the cropped data
        def crop_fn(i):
            return data[i, crop_start:crop_start + crop_size, :]

        cropped_data = tf.map_fn(crop_fn, tf.range(batch_size), fn_output_signature=tf.float32)

        # Resize cropped data to original time_steps using linear interpolation
        cropped_data = tf.expand_dims(cropped_data, axis=2)
        resized_data = tf.image.resize(cropped_data, [time_steps, 1], method="bilinear")
        resized_data = tf.squeeze(resized_data, axis=2)

        return resized_data

    def magnitude_warping(self, data, num_periods=3):
        """
        Apply magnitude warping to the input data by scaling the signal with a sine wave matrix.
        """
        batch_size, time_steps, channels = tf.shape(data)[0], tf.shape(data)[1], tf.shape(data)[2]

        # Create a time array for the sine wave (normalized to 0 to 2π)
        time_steps_range = tf.linspace(0.0, 2.0 * np.pi * num_periods, time_steps)
        time_steps_range = tf.reshape(time_steps_range, (time_steps, 1))

        # Generate random phase shifts for each channel in each batch
        random_phase_shifts = tf.random.uniform(shape=(batch_size, channels), minval=0.0, maxval=2.0 * np.pi)

        def apply_sine_warping(inputs):
            signal, phase_shifts = inputs
            # Create the sine wave for each channel with the corresponding phase shift
            sine_waves = tf.sin(time_steps_range + phase_shifts)
            # Scale the sine wave matrix and add 1
            sine_warping = 1.0 + self.lambda_amp * sine_waves
            # Apply the sine warping to the signal
            return signal * sine_warping

        # Apply the sine warping to each sample in the batch using tf.map_fn
        warped_data = tf.map_fn(apply_sine_warping, (data, random_phase_shifts), fn_output_signature=tf.float32)

        return warped_data

    def time_warping(self, data):
        """
        Apply time warping to the time-series data.
        - Compress some parts of the data by discarding points.
        - Stretch some parts using linear interpolation.
        """
        batch_size, time_steps, channels = tf.shape(data)[0], tf.shape(data)[1], tf.shape(data)[2]

        tf.print("Running on:", data.device)

        # Generate a random vector of length time_steps with values -1, 0, 1 for each sample in the batch
        random_warp = tf.random.uniform(shape=(time_steps,), minval=-1, maxval=2, dtype=tf.int32)

        # Precompute the number of elements that will be kept or interpolated
        num_kept = tf.reduce_sum(tf.cast(random_warp == 0, tf.int32))
        num_stretched = tf.reduce_sum(
            tf.cast(random_warp == 1, tf.int32)) * 2  # Each stretch adds an interpolated point
        final_size = num_kept + num_stretched

        random_warp = tf.tile(tf.expand_dims(random_warp, axis=0), [batch_size, 1])

        def warp_single_sample(inputs):
            sample_data, warp_vector = inputs

            # Initialize a TensorArray with a fixed size
            warped_data = tf.TensorArray(dtype=tf.float32, size=final_size, dynamic_size=False)

            def loop_body(i, j, warped_data):
                warp_value = warp_vector[i]

                # Compress (discard): if warp_value == -1, skip the current time step
                def compress_step(j, warped_data):
                    return j, warped_data

                # Stretch (interpolate): if warp_value == 1, average the current and the next time step
                def stretch_step(j, warped_data):
                    next_i = tf.minimum(i + 1, time_steps - 1)
                    interpolated_value = (sample_data[i, :] + sample_data[next_i, :]) / 2
                    warped_data = warped_data.write(j, sample_data[i, :])  # Add original point
                    warped_data = warped_data.write(j + 1, interpolated_value)  # Add interpolated point
                    return j + 2, warped_data

                # No action: if warp_value == 0, keep the data as-is
                def keep_step(j, warped_data):
                    warped_data = warped_data.write(j, sample_data[i, :])  # Add original point
                    return j + 1, warped_data

                # Choose action based on the warp_value using tf.cond
                j, warped_data = tf.cond(
                    warp_value == -1,
                    lambda: compress_step(j, warped_data),
                    lambda: tf.cond(warp_value == 1, lambda: stretch_step(j, warped_data),
                                    lambda: keep_step(j, warped_data))
                )

                return i + 1, j, warped_data

            # Use tf.while_loop to iterate over time_steps
            i = tf.constant(0)
            j = tf.constant(0)  # Tracks the index for the preallocated TensorArray
            _, _, warped_data = tf.while_loop(lambda i, j, _: i < time_steps, loop_body, [i, j, warped_data])

            # Stack the warped_data TensorArray back into a tensor
            warped_data = warped_data.stack()

            # Resize only the time dimension (axis 0) to match the original time_steps using bilinear interpolation
            warped_data_resized = tf.image.resize(tf.expand_dims(warped_data, axis=1), [time_steps, 1],
                                                  method="bilinear")

            return tf.squeeze(warped_data_resized, axis=1)

        # Apply the time warping to each sample in the batch
        warped_batch = tf.map_fn(warp_single_sample, (data, random_warp), fn_output_signature=tf.float32)
        warped_batch = tf.reshape(warped_batch, [batch_size, time_steps, channels])

        return warped_batch

    def random_smoothing(self, data):
        """
        Apply random smoothing to time-series data using a FIR filter with a random smoothing factor.
        """
        batch_size, time_steps, channels = tf.shape(data)[0], tf.shape(data)[1], tf.shape(data)[2]

        lambda_val = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)

        # FIR filter coefficients based on λ
        filter_coeffs = tf.stack([lambda_val / 2, 1 - lambda_val, lambda_val / 2])
        filter_coeffs = tf.reshape(filter_coeffs, [3, 1, 1])  # Shape: [filter_size, 1, 1]

        data_reshaped = tf.reshape(data, [batch_size * channels, time_steps, 1])

        smoothed_data = tf.nn.conv1d(data_reshaped, filters=filter_coeffs, stride=1, padding='SAME')
        smoothed_data = tf.reshape(smoothed_data, [batch_size, time_steps, channels])

        return smoothed_data

    def permute_segments(self, data):
        """
        Permute segments of the input data along the time axis.
        """
        batch_size, time_steps, channels = tf.shape(data)[0], tf.shape(data)[1], tf.shape(data)[2]

        # Calculate the divisor and remainder
        divisor = time_steps // self.n_perm_seg
        remainder = time_steps % self.n_perm_seg

        # tf.print("divisor: ", divisor)
        # tf.print("remainder: ", remainder)

        # Reshape the first n_perm_seg - 1 segments with size divisor
        reshaped_data_1 = tf.reshape(data[:, :divisor * (self.n_perm_seg - 1), :],
                                     [batch_size, self.n_perm_seg - 1, divisor, channels])

        # tf.print("reshaped_data_1 shape: ", tf.shape(reshaped_data_1))

        # Reshape the last segment to include the remainder (divisor + remainder)
        last_segment_start = divisor * (self.n_perm_seg - 1)
        reshaped_data_2 = tf.reshape(data[:, last_segment_start:, :],
                                     [batch_size, divisor + remainder, channels])

        # tf.print("reshaped_data_2 shape: ", tf.shape(reshaped_data_2))

        # Generate a random permutation of the segment indices for each sample in the batch
        permuted_indices = tf.map_fn(
            lambda _: tf.random.shuffle(tf.range(self.n_perm_seg - 1)),
            tf.zeros([batch_size], dtype=tf.int32),
            fn_output_signature=tf.int32
        )

        # Gather the segments in the new permuted order for each batch
        permuted_data = tf.map_fn(
            lambda x: tf.gather(x[0], x[1]),
            (reshaped_data_1, permuted_indices),
            fn_output_signature=tf.float32
        )

        # Reshape back to the original shape (batch_size, time_steps, channels)
        permuted_data = tf.reshape(permuted_data, [batch_size, time_steps - divisor - remainder, channels])

        # tf.print("permuted_data shape: ", tf.shape(permuted_data))

        permuted_data = tf.concat([permuted_data, reshaped_data_2], axis=1)

        # tf.print("permuted_data shape: ", tf.shape(permuted_data))

        return permuted_data

    def shift_windows_fun(self, data):
        """
        Extracts two overlapping windows of size (500, 3) from each sample in the input data.
        """
        batch_size, time_steps, channels = tf.shape(data)[0], tf.shape(data)[1], tf.shape(data)[2]
        window_size = 500
        overlap_size = int(self.overlap * window_size)
        max_start = time_steps - window_size - (window_size - overlap_size)

        # Generate random starting index for the first window
        start_indices = tf.random.uniform(shape=[batch_size], minval=0, maxval=max_start + 1, dtype=tf.int32)

        # Calculate the start index for the second window with 10% overlap
        second_start_indices = start_indices + (window_size - overlap_size)

        # Create index ranges for the first and second windows
        indices_1 = tf.range(window_size)[tf.newaxis, :] + start_indices[:, tf.newaxis]
        indices_2 = tf.range(window_size)[tf.newaxis, :] + second_start_indices[:, tf.newaxis]

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
            min_val = tf.reduce_min(inputs, axis=[1, 2], keepdims=True)  # Minimum value per batch sample
            max_val = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)  # Maximum value per batch sample

            # Normalize each batch sample independently
            normalized = 2 * (inputs - min_val) / (max_val - min_val + 1e-8) - 1

            return normalized

    def get_augmenter(self):
        """Combine several augmentations into a single sequential model."""
        return keras.Sequential(
            [
                # layers.Lambda(self.jitter),
                layers.Lambda(self.left_to_right_flipping),
                layers.Lambda(self.bidirectional_flipping),
                # layers.Lambda(self.random_channel_permutation),
                layers.Lambda(self.rotate_axis),
                layers.Lambda(self.add_gravity),
                # layers.Lambda(self.slide_window),
                # layers.Lambda(self.blockout),
                # layers.Lambda(self.crop_and_resize),
                # layers.Lambda(self.magnitude_warping),
                # layers.Lambda(self.time_warping),
                # layers.Lambda(self.random_smoothing),
                layers.Lambda(self.permute_segments),
                self.CustomNormalizer(),
            ]
        )


# Visualization function
def visualize_augmentations(gdataset, augmentation, num_windows):
    augmenter = augmentation.get_augmenter()

    gdataset_np = next(iter(gdataset)).numpy()  # Convert from tensor to numpy array

    # Choose 3 random windows from the batch
    random_indices = random.sample(range(len(gdataset_np)), num_windows)
    original_windows = [gdataset_np[i] for i in random_indices]

    # Apply augmentation to the selected windows
    augmented_windows = augmenter(np.array(original_windows))

    # Plotting
    fig, axs = plt.subplots(2, num_windows, figsize=(15, 6))

    for i in range(num_windows):
        # Plot original data
        axs[0, i].plot(original_windows[i][:, 0], label='X', color='r')  # X-axis (red)
        axs[0, i].plot(original_windows[i][:, 1], label='Y', color='g')  # Y-axis (green)
        axs[0, i].plot(original_windows[i][:, 2], label='Z', color='b')  # Z-axis (blue)
        axs[0, i].set_title("Original")
        axs[0, i].legend(loc='upper right')

        # Plot augmented data
        axs[1, i].plot(augmented_windows[i][:, 0], label='X', color='r')  # X-axis (red)
        axs[1, i].plot(augmented_windows[i][:, 1], label='Y', color='g')  # Y-axis (green)
        axs[1, i].plot(augmented_windows[i][:, 2], label='Z', color='b')  # Z-axis (blue)
        axs[1, i].set_title("Augmented")
        axs[1, i].legend(loc='upper right')

    plt.tight_layout()
    plt.show()


augmentation = Augmentation()
visualize_augmentations(gdataset, augmentation, num_windows=3)


def augment_and_extend_dataset(df, get_augmenter, num_extensions=1):
    """
    Augments the dataset and extends it by a specified number of times.
    """

    # Extract acceleration windows and labels
    X_original = tf.convert_to_tensor(df["X"].to_list(), dtype=tf.float32)
    y_original = tf.convert_to_tensor(df["y"].to_list(), dtype=tf.int32)

    # Get augmenter
    augmenter = get_augmenter()

    # Initialize lists to store extended data
    X_combined = list(df["X"])  # Start with the original data
    y_combined = list(df["y"])  # Start with the original labels

    # Perform augmentation num_extensions times
    for _ in range(num_extensions):
        X_augmented = augmenter(X_original)
        X_combined.extend(X_augmented.numpy().tolist())  # Add augmented data
        y_combined.extend(y_original.numpy().tolist())  # Add corresponding labels

    # Create a new DataFrame
    extended_df = pd.DataFrame({"X": X_combined, "y": y_combined})

    return extended_df


num_extensions = labeled_dataset_size // 100 - 1
labeled_gdataset_ext = augment_and_extend_dataset(labeled_gdataset, augmentation.get_augmenter,
                                              num_extensions=num_extensions)
print("shape of labeled_gdataset: ", np.shape(labeled_gdataset_ext))

labeled_gdataset_ext = tf.data.Dataset.from_tensor_slices((list(labeled_gdataset_ext['X']),
                                                           list(labeled_gdataset_ext['y'])))
labeled_gdataset_ext = (labeled_gdataset_ext.shuffle(buffer_size=len(labeled_gdataset_ext))
                        .batch(labeled_gdataset_batch_size).prefetch(buffer_size=tf.data.AUTOTUNE))

print(labeled_gdataset_ext.element_spec)

train_dataset = tf.data.Dataset.zip(
    (gdataset, labeled_gdataset_ext)
).prefetch(buffer_size=tf.data.AUTOTUNE)

"""
## Encoder architecture
"""


# Define the encoder architecture
def embeddings_function(M):
    return keras.Sequential(
        [
            # Layer 1
            layers.ZeroPadding1D(padding=1),
            layers.Conv1D(filters=32, kernel_size=8, padding='valid'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            # layers.Dropout(0.05),
            layers.MaxPooling1D(pool_size=2),

            # Layer 2
            layers.ZeroPadding1D(padding=1),
            layers.Conv1D(filters=32, kernel_size=8, padding='valid'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            # layers.Dropout(0.05),
            layers.MaxPooling1D(pool_size=2),

            # Layer 3
            layers.ZeroPadding1D(padding=1),
            layers.Conv1D(filters=16, kernel_size=16, padding='valid'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            # layers.Dropout(0.2),
            layers.MaxPooling1D(pool_size=2),

            # Layer 4
            layers.ZeroPadding1D(padding=1),
            layers.Conv1D(filters=16, kernel_size=16, padding='valid'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            # layers.Dropout(0.2),
            layers.MaxPooling1D(pool_size=2),

            # Flatten and Dense layer to get M-dimensional output
            layers.Flatten(),
            layers.Dense(M),
        ],
        name="embeddings_function",
    )


"""
## Self-supervised model for contrastive pretraining
"""


# Define the contrastive model with model-subclassing
class ContrastiveModel(keras.Model):
    def __init__(self):
        super().__init__()

        self.temperature = temperature
        self.contrastive_augmenter = augmentation.get_augmenter()
        self.classification_augmenter = augmentation.get_augmenter()
        self.shift_windows = augmentation.shift_windows()
        self.encoder = embeddings_function(M)

        self.current_index = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.similarity_values = tf.Variable(tf.zeros((1000, 2)),
                                             trainable=False)

        # Non-linear MLP as projection head
        self.projection_head = keras.Sequential(
            [
                keras.Input(shape=(M,)),
                layers.Dense(M, activation="relu"),
                # layers.BatchNormalization(scale=False, center=False, momentum=0.99, epsilon=1e-3),
                # layers.BatchNormalization(),
                # layers.Dropout(0.05),
                # layers.Dense(M, activation="relu"),
                # layers.BatchNormalization(),
                # layers.Dropout(0.05),
                layers.Dense(M),  # Keep this linear layer at the end for better contrastive loss
            ],
            name="projection_head",
        )

        # Single dense layer for linear probing
        self.linear_probe = keras.Sequential(
            [
                layers.Input(shape=(M,)),
                layers.Dense(2, kernel_regularizer=keras.regularizers.L2(1e-4))
            ],
            name="linear_probe",
        )

        self.encoder.summary()
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
            self.probe_accuracy
        ]

    def contrastive_loss_with_regularization(self, projections_1, projections_2, regularization_weight=0.0):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        # projections_1 = ops.normalize(projections_1, axis=1)
        # projections_2 = ops.normalize(projections_2, axis=1)
        projections_1 = tf.nn.l2_normalize(projections_1, axis=1)
        projections_2 = tf.nn.l2_normalize(projections_2, axis=1)
        similarities = (
                ops.matmul(projections_1, ops.transpose(projections_2)) / self.temperature
        )

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = ops.shape(projections_1)[0]
        contrastive_labels = ops.arange(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(
            contrastive_labels, ops.transpose(similarities)
        )

        # Mask to exclude the diagonal (positive pairs)
        mask = tf.eye(batch_size)
        negative_similarities = tf.where(mask == 0, similarities, 0)

        # Regularization term: penalize negative similarities that are too high
        regularization_term = tf.reduce_sum(
            tf.square(negative_similarities))  # L2 regularization on the negative similarities
        regularization_loss = regularization_weight * regularization_term

        # Calculate positive pair similarities
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(contrastive_labels, similarities, from_logits=True)
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(contrastive_labels, ops.transpose(similarities),
                                                                from_logits=True)

        # Combine the regularization term with the contrastive loss
        contrastive_loss = (loss_1_2 + loss_2_1) / 2
        combined_loss = contrastive_loss + regularization_loss

        return combined_loss

    def contrastive_loss(self, projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = ops.normalize(projections_1, axis=1)
        projections_2 = ops.normalize(projections_2, axis=1)
        similarities = (
                ops.matmul(projections_1, ops.transpose(projections_2)) / self.temperature
        )

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
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

    def compute_similarity_metrics(self, projections_1, projections_2):
        # Normalize the projections
        projections_1 = tf.nn.l2_normalize(projections_1, axis=1)
        projections_2 = tf.nn.l2_normalize(projections_2, axis=1)

        # Compute cosine similarities
        similarities = tf.matmul(projections_1, tf.transpose(projections_2))

        # Get dynamic shape of the batch (since similarities.shape[0] might return None)
        batch_size = tf.shape(similarities)[0]

        # Extract positive pair similarities (diagonal)
        positive_similarities = tf.linalg.diag_part(similarities)

        # Extract negative pair similarities (off-diagonal)
        negative_similarities = tf.reshape(similarities, [-1])

        # Create a mask to remove diagonal elements (positive pairs)
        negative_mask = tf.not_equal(
            tf.tile(tf.range(batch_size), [batch_size]),
            tf.repeat(tf.range(batch_size), batch_size)
        )

        # Apply the mask to extract only negative similarities
        negative_similarities = tf.boolean_mask(negative_similarities, negative_mask)

        return tf.reduce_mean(positive_similarities), tf.reduce_mean(negative_similarities)

    def train_step(self, data):
        unlabeled_data = data[0]
        labeled_data = data[1][0]
        labels = data[1][1]

        # print("Unlabeled data shape: ", unlabeled_data)
        # print("Labeled data shape: ", labeled_data)
        # print("Labels shape: ", labels)
        # Each window is augmented twice, differently
        augmented_data_1, augmented_data2 = self.shift_windows(unlabeled_data, training=True)
        augmented_data_1 = self.contrastive_augmenter(augmented_data_1, training=True)
        augmented_data_2 = self.contrastive_augmenter(augmented_data2, training=True)

        with tf.GradientTape() as tape:
            # Pass both augmented versions of the images through the encoder
            features_1 = self.encoder(augmented_data_1, training=True)
            features_2 = self.encoder(augmented_data_2, training=True)

            # The representations are passed through a projection MLP
            # projections_1 = self.projection_head(features_1, training=True)
            # projections_2 = self.projection_head(features_2, training=True)

            # Compute the contrastive loss
            contrastive_loss = self.contrastive_loss_with_regularization(features_1, features_2)

            # # SIMILARITY METRICS
            # tf.print("Current index: ", self.current_index)
            # positive_sim, negative_sim = self.compute_similarity_metrics(features_1, features_2)
            # new_values = tf.stack([positive_sim, negative_sim])
            # self.similarity_values[self.current_index].assign(new_values)
            # self.current_index.assign_add(1)

        # Compute gradients of the contrastive loss and update the encoder and projection head
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights,
            )
        )

        # Update the contrastive loss tracker for monitoring
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        # Labels are only used in evalutation for an on-the-fly logistic regression
        preprocessed_data = self.classification_augmenter(labeled_data, training=True)

        with tf.GradientTape() as tape:
            # the encoder is used in inference mode here to avoid regularization
            # and updating the batch normalization paramers if they are used
            features = self.encoder(preprocessed_data, training=True)
            class_logits = self.linear_probe(features, training=True)
            probe_loss = self.probe_loss(labels, class_logits)

        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)

        self.probe_optimizer.apply_gradients(
            zip(gradients, self.linear_probe.trainable_weights)
        )

        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        labeled_data, labels = data

        # For testing the components are used with a training=False flag
        preprocessed_data = self.classification_augmenter(
            labeled_data, training=False
        )
        features = self.encoder(preprocessed_data, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)

        # Only the probe metrics are logged at test time
        return {m.name: m.result() for m in self.metrics[2:]}

    def plot_contrastive_loss(self, pretraining_history):
        """
        Plots contrastive loss per epoch.
        """
        plt.figure(figsize=(6, 5))
        plt.plot(pretraining_history.history["c_loss"], label="Contrastive Loss", color='blue')
        plt.title("Contrastive Loss per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_validation_accuracy(self, pretraining_history):
        """
        Plots validation accuracy per epoch.
        """
        plt.figure(figsize=(6, 5))
        plt.plot(pretraining_history.history["val_p_acc"], label="Linear Probing Accuracy", color='orange')
        plt.title("Linear Probing Accuracy per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()


# Contrastive pretraining
pretraining_model = ContrastiveModel()
pretraining_model.compile(
    contrastive_optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    probe_optimizer=keras.optimizers.Adam(learning_rate=1e-4, beta_2=0.99, weight_decay=1e-5)
)

early_stopping = callbacks.EarlyStopping(
    monitor="c_loss",
    patience=30,
    mode="min",
    verbose=1,
    start_from_epoch=0,
    restore_best_weights=True
)

checkpoint = callbacks.ModelCheckpoint(
    filepath="simclr_best_model.weights.h5",
    monitor="c_loss",
    mode="min",
    save_best_only=True,
    save_weights_only=True,
    verbose=0
)


def lr_schedule(epoch, lr):
    total_epochs = num_epochs
    decay_start_epoch = total_epochs // 2  # Start decay at the halfway point of the training
    if epoch == 0:  # For the first epoch, keep the initial learning rate
        return learning_rate
    elif epoch >= decay_start_epoch:
        return lr * 0.99  # Decay logic
    return lr


lr_scheduler = callbacks.LearningRateScheduler(lr_schedule)

pretraining_history = pretraining_model.fit(
    train_dataset, epochs=num_epochs, validation_data=labeled_gdataset_ext, batch_size=batch_size,
    callbacks=[checkpoint, lr_scheduler]
)

# Load best weights.
print("Loading best weights...")
pretraining_model.load_weights("simclr_best_model.weights.h5")

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

# for pos_sim, neg_sim in pretraining_model.similarity_values:
#     print("Positive similarity: {:.2f}, Negative similarity: {:.2f}".format(pos_sim, neg_sim))

pretraining_model.get_layer("embeddings_function").save_weights("embeddings.weights.h5")

# pretraining_model.get_layer("embeddings_function").load_weights("embeddings.weights.h5")

pretraining_model.plot_contrastive_loss(pretraining_history)
pretraining_model.plot_validation_accuracy(pretraining_history)


def get_labeled_embeddings(pretraining_model, labeled_gdataset):
    windows = []
    labels = []

    # Iterate through the batched dataset and collect windows and labels
    for batch in labeled_gdataset:
        windows_batch = batch[0]  # Extract the windows from the batch
        labels_batch = batch[1]  # Extract the labels from the batch

        # Predict the embeddings for the entire batch of windows
        embeddings_batch = pretraining_model.get_layer("embeddings_function").predict(windows_batch)

        # Append the embeddings and labels to the lists
        windows.append(embeddings_batch)
        labels.append(labels_batch.numpy())  # Convert TensorFlow tensor to numpy array

    # Stack the results to form a full matrix
    embeddings = np.vstack(windows)  # Convert list of arrays into a full array
    labels = np.hstack(labels)  # Flatten list of label arrays into a single array

    print("Embeddings shape: {}".format(embeddings.shape))
    return embeddings, labels


# Function to visualize the embeddings
def visualize_embeddings(embeddings, labels, n_components=2, perplexity=5, learning_rate="auto", n_iter=500):
    """
    Visualize the embeddings using t-SNE, colored by their class labels.
    """
    # Initialize t-SNE model
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter,
                random_state=42)

    # Apply t-SNE to embeddings
    reduced_embeddings = tsne.fit_transform(embeddings)

    # 2D Visualization
    if n_components == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_embeddings[labels == 0, 0], reduced_embeddings[labels == 0, 1], label="No tremor", c='b',
                    alpha=0.5)
        plt.scatter(reduced_embeddings[labels == 1, 0], reduced_embeddings[labels == 1, 1], label="Tremor", c='r',
                    alpha=0.5)
        plt.title("2D t-SNE Visualization of Embeddings")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend()
        plt.show()

    # 3D Visualization
    elif n_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced_embeddings[labels == 0, 0], reduced_embeddings[labels == 0, 1],
                   reduced_embeddings[labels == 0, 2], label="Class 0", c='b', alpha=0.5)
        ax.scatter(reduced_embeddings[labels == 1, 0], reduced_embeddings[labels == 1, 1],
                   reduced_embeddings[labels == 1, 2], label="Class 1", c='r', alpha=0.5)
        ax.set_title("3D t-SNE Visualization of Embeddings")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_zlabel("t-SNE Dimension 3")
        ax.legend()
        plt.show()
    else:
        raise ValueError("n_components must be 2 or 3 for visualization.")



labeled_gdataset = tf.data.Dataset.from_tensor_slices((list(labeled_gdataset['X']), list(labeled_gdataset['y'])))
labeled_gdataset = (labeled_gdataset.shuffle(buffer_size=len(labeled_gdataset)).batch(10)
                    .prefetch(buffer_size=tf.data.AUTOTUNE))


# Use the function to get embeddings and visualize them
embeddings, labels = get_labeled_embeddings(pretraining_model, labeled_gdataset)
visualize_embeddings(embeddings, labels, n_components=2)

print(time.time() - start)

# Alarm
os.system('play -nq -t alsa synth {} sine {}'.format(1, 999))
