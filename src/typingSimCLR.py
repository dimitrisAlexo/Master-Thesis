"""
## Setup
"""

import time
import sys
import resource
import random
import os
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

start = time.time()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_BACKEND"] = "tensorflow"

np.set_printoptions(threshold=sys.maxsize)

# Make sure we are able to handle large datasets
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))
from keras import ops
from keras import layers
from keras import callbacks
from tf_keras import mixed_precision

from sklearn.manifold import TSNE

from utils import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # run on CPU

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

M = 64
K2 = 100
unlabeled_dataset_size = 5120
batch_size = 512
num_epochs = 200
temperature = 0.01
learning_rate = 0.001

"""
## Dataset
"""

# Load or create unlabeled typing dataset
# sdata_path = os.path.join("..", "data", "typing_sdata.pickle")
# gdata_path = os.path.join("..", "data", "typing_gdata.pickle")
# typing_sdata = unpickle_data(sdata_path)
# typing_gdata = unpickle_data(gdata_path)
# gdataset = form_unlabeled_typing_dataset(typing_gdata, typing_sdata, K2)

with open("unlabeled_typing_data.pickle", "rb") as f:
    gdataset = pkl.load(f)

print(f"Original dataset shape: {gdataset.shape}")

gdataset = gdataset[:unlabeled_dataset_size]

print(f"Processed dataset shape: {gdataset.shape}")

gdataset = tf.data.Dataset.from_tensor_slices(gdataset)
print(f"Length of gdataset: {len(gdataset)}")
gdataset = (
    gdataset.shuffle(buffer_size=len(gdataset))
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

"""
## Typing Data Augmentations
"""


class TypingAugmentation:
    def __init__(
        self,
        noise_factor=0.01,
        dropout_rate=0.1,
        scale_factor=0.2,
        shift_factor=0.1,
        n_perm_seg=5,
    ):
        self.noise_factor = noise_factor
        self.dropout_rate = dropout_rate
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor
        self.n_perm_seg = n_perm_seg

    def add_noise(self, data):
        """Add Gaussian noise to typing histograms"""
        noise = tf.random.normal(
            tf.shape(data), mean=0.0, stddev=self.noise_factor, dtype=tf.float32
        )
        return data + noise

    def dropout_features(self, data):
        """Randomly zero out some features in the histogram"""
        mask = tf.random.uniform(tf.shape(data)) > self.dropout_rate
        return data * tf.cast(mask, tf.float32)

    def scale_histogram(self, data):
        """Scale the histogram values by a random factor"""
        scale = tf.random.uniform(
            [tf.shape(data)[0], 1],
            minval=1.0 - self.scale_factor,
            maxval=1.0 + self.scale_factor,
            dtype=tf.float32,
        )
        return data * scale

    def shift_values(self, data):
        """Add a random shift to all values in the histogram"""
        shift = tf.random.uniform(
            [tf.shape(data)[0], 1],
            minval=-self.shift_factor,
            maxval=self.shift_factor,
            dtype=tf.float32,
        )
        return data + shift

    @tf.function
    def permute_histogram_segments(self, data):
        """
        Permute segments of the histogram features.
        For typing data, this permutes segments of the 502-dimensional histogram.
        This disrupts the temporal structure of the histogram bins while preserving
        the overall distribution characteristics.
        """
        batch_size, features = tf.shape(data)[0], tf.shape(data)[1]

        # Calculate the divisor and remainder for segmentation
        divisor = features // self.n_perm_seg
        remainder = features % self.n_perm_seg

        # Use tf.cond instead of Python if statement
        def permute_segments():
            # Reshape the first n_perm_seg - 1 segments with size divisor
            reshaped_data_1 = tf.reshape(
                data[:, : divisor * (self.n_perm_seg - 1)],
                [batch_size, self.n_perm_seg - 1, divisor],
            )

            # Reshape the last segment to include the remainder (divisor + remainder)
            last_segment_start = divisor * (self.n_perm_seg - 1)
            reshaped_data_2 = tf.reshape(
                data[:, last_segment_start:], [batch_size, divisor + remainder]
            )

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

            # Reshape back to the flattened form
            permuted_data = tf.reshape(
                permuted_data, [batch_size, features - divisor - remainder]
            )

            # Concatenate with the last segment
            result = tf.concat([permuted_data, reshaped_data_2], axis=1)
            return result

        def return_original():
            return data

        # Use tf.cond to conditionally apply permutation
        return tf.cond(divisor >= 1, permute_segments, return_original)

    def normalize_histogram(self, data):
        """Normalize histograms between -1 and 1"""
        min_val = tf.reduce_min(data, axis=1, keepdims=True)
        max_val = tf.reduce_max(data, axis=1, keepdims=True)
        # Avoid division by zero
        denominator = max_val - min_val + 1e-8
        normalized = (data - min_val) / denominator
        return normalized

    def get_contrastive_augmenter(self):
        """Combine several augmentations into a single sequential model."""
        return keras.Sequential(
            [
                layers.Lambda(self.add_noise),
                layers.Lambda(self.dropout_features),
                layers.Lambda(self.permute_histogram_segments),
                # layers.Lambda(self.scale_histogram),
                # layers.Lambda(self.shift_values),
                # layers.Lambda(self.normalize_histogram),
            ]
        )


# Visualization function
def visualize_typing_augmentations(gdataset, augmentation, num_histograms=3):
    augmenter = augmentation.get_contrastive_augmenter()

    gdataset_np = next(iter(gdataset)).numpy()  # Convert from tensor to numpy array

    # Choose random histograms from the batch
    random_indices = random.sample(range(len(gdataset_np)), num_histograms)
    original_histograms = [gdataset_np[i] for i in random_indices]

    # Apply augmentation to the selected histograms
    augmented_histograms = augmenter(np.array(original_histograms))

    # Plotting
    fig, axs = plt.subplots(2, num_histograms, figsize=(15, 6))

    for i in range(num_histograms):
        # Plot original histogram
        axs[0, i].plot(original_histograms[i], color="b")
        axs[0, i].set_title(f"Original Histogram {i+1}")
        axs[0, i].set_xlabel("Feature Index")
        axs[0, i].set_ylabel("Value")

        # Plot augmented histogram
        axs[1, i].plot(augmented_histograms[i], color="r")
        axs[1, i].set_title(f"Augmented Histogram {i+1}")
        axs[1, i].set_xlabel("Feature Index")
        axs[1, i].set_ylabel("Value")

    plt.tight_layout()
    plt.show()


augmentation = TypingAugmentation()
visualize_typing_augmentations(gdataset, augmentation, num_histograms=3)

"""
## Encoder architecture
"""


# Define the encoder architecture for typing data (502-dimensional histograms)
def embeddings_function(M):
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
        self.contrastive_augmenter = augmentation.get_contrastive_augmenter()
        self.encoder = embeddings_function(M)

        # Non-linear MLP as projection head
        self.projection_head = keras.Sequential(
            [
                keras.Input(shape=(M,)),
                layers.Dense(M, activation="relu"),
                layers.Dense(M),
            ],
            name="projection_head",
        )

        self.encoder.summary()
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
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.nn.l2_normalize(projections_1, axis=1)
        projections_2 = tf.nn.l2_normalize(projections_2, axis=1)
        similarities = (
            ops.matmul(projections_1, ops.transpose(projections_2)) / self.temperature
        )

        # The similarity between the representations of two augmented views of the
        # same histogram should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)

        # Symmetric loss
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, ops.transpose(similarities), from_logits=True
        )

        return (loss_1_2 + loss_2_1) / 2

    def train_step(self, data):
        histograms = data

        with tf.GradientTape() as tape:
            # Apply two different augmentations to the same histograms
            augmented_histograms_1 = self.contrastive_augmenter(histograms)
            augmented_histograms_2 = self.contrastive_augmenter(histograms)

            # Generate embeddings for both augmented versions
            embeddings_1 = self.encoder(augmented_histograms_1, training=True)
            embeddings_2 = self.encoder(augmented_histograms_2, training=True)

            # Generate projections for contrastive learning
            # projections_1 = self.projection_head(embeddings_1, training=True)
            # projections_2 = self.projection_head(embeddings_2, training=True)

            # Compute contrastive loss
            contrastive_loss = self.contrastive_loss(embeddings_1, embeddings_2)

        # Apply gradients to encoder and projection head
        trainable_vars = (
            self.encoder.trainable_variables + self.projection_head.trainable_variables
        )
        gradients = tape.gradient(contrastive_loss, trainable_vars)
        self.contrastive_optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        # Compute accuracy (how often the correct positive pair has highest similarity)
        projections_1_normalized = tf.nn.l2_normalize(embeddings_1, axis=1)
        projections_2_normalized = tf.nn.l2_normalize(embeddings_2, axis=1)
        similarities = (
            ops.matmul(
                projections_1_normalized, ops.transpose(projections_2_normalized)
            )
            / self.temperature
        )
        batch_size = tf.shape(embeddings_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)

        return {
            "c_loss": self.contrastive_loss_tracker.result(),
            "c_acc": self.contrastive_accuracy.result(),
        }

    def test_step(self, data):
        # For typing data, we don't have labeled validation data
        # So we just return the training metrics
        return self.train_step(data)

    def plot_contrastive_loss(self, pretraining_history):
        plt.figure(figsize=(8, 5))
        plt.plot(pretraining_history.history["c_loss"])
        plt.title("Contrastive Loss During Pretraining")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Contrastive Loss"], loc="upper right")
        plt.show()


# Contrastive pretraining
pretraining_model = ContrastiveModel()
pretraining_model.compile(
    contrastive_optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
)

checkpoint = callbacks.ModelCheckpoint(
    filepath="typing_simclr_best_model.weights.h5",
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
        return lr
    elif epoch >= decay_start_epoch:
        return lr * 1.0
    return lr


lr_scheduler = callbacks.LearningRateScheduler(lr_schedule)

# Train the model
pretraining_history = pretraining_model.fit(
    gdataset,
    epochs=num_epochs,
    batch_size=batch_size,
    callbacks=[checkpoint, lr_scheduler],
)

# Load best weights
print("Loading best weights...")
pretraining_model.load_weights("typing_simclr_best_model.weights.h5")

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

# Save the encoder weights for use in the MIL model
pretraining_model.get_layer("embeddings_function").save_weights(
    "typing_embeddings.weights.h5"
)

# Plot results
pretraining_model.plot_contrastive_loss(pretraining_history)

print(f"Total training time: {time.time() - start} seconds")

# Alarm
os.system('powershell.exe -c "[console]::beep(999,1000)"')

input("Press Enter to exit...")
