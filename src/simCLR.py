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
from tf_keras import mixed_precision

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

M = 64
E_thres = 0.15
Kt = 100
batch_size = 100
num_epochs = 20
temperature = 0.1

"""
## Dataset
"""

# Adjust the paths to be relative to the current script location
gdata_path = os.path.join('..', 'data', 'tremor_gdata.pickle')
tremor_gdata = unpickle_data(gdata_path)

# gdataset = form_unlabeled_dataset(tremor_gdata, E_thres, Kt)

with open("unlabeled_data.pickle", 'rb') as f:
    gdataset = pkl.load(f)

print(np.shape(gdataset))

gdataset = tf.data.Dataset.from_tensor_slices(gdataset)
gdataset = gdataset.shuffle(buffer_size=len(gdataset)).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

print(gdataset.element_spec)


"""
## Augmentations
"""


def get_augmenter():
    return keras.Sequential(
        [
            layers.Lambda(lambda x: x),
            layers.GaussianNoise(0.5)
        ]
    )


# Visualization function
def visualize_augmentations(gdataset, num_windows):

    augmenter = get_augmenter()

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


visualize_augmentations(gdataset, num_windows=3)


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
            layers.LeakyReLU(negative_slope=0.2),
            layers.MaxPooling1D(pool_size=2),

            # Layer 2
            layers.ZeroPadding1D(padding=1),
            layers.Conv1D(filters=32, kernel_size=8, padding='valid'),
            layers.LeakyReLU(negative_slope=0.2),
            layers.MaxPooling1D(pool_size=2),

            # Layer 3
            layers.ZeroPadding1D(padding=1),
            layers.Conv1D(filters=16, kernel_size=16, padding='valid'),
            layers.LeakyReLU(negative_slope=0.2),
            layers.MaxPooling1D(pool_size=2),

            # Layer 4
            layers.ZeroPadding1D(padding=1),
            layers.Conv1D(filters=16, kernel_size=16, padding='valid'),
            layers.LeakyReLU(negative_slope=0.2),
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
        self.contrastive_augmenter = get_augmenter()
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
            self.contrastive_accuracy
        ]

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

    def train_step(self, data):

        # Each window is augmented twice, differently
        augmented_data_1 = self.contrastive_augmenter(data, training=True)
        augmented_data_2 = self.contrastive_augmenter(data, training=True)

        with tf.GradientTape() as tape:
            # Pass both augmented versions of the images through the encoder
            features_1 = self.encoder(augmented_data_1, training=True)
            features_2 = self.encoder(augmented_data_2, training=True)

            # The representations are passed through a projection MLP
            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)

            # Compute the contrastive loss
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)

        # Compute gradients of the contrastive loss and update the encoder and projection head
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )

        # Update the contrastive loss tracker for monitoring
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):

        # Augment the windows twice for contrastive testing
        augmented_data_1 = self.contrastive_augmenter(data, training=False)
        augmented_data_2 = self.contrastive_augmenter(data, training=False)

        # Extract features from both augmented views using the encoder
        features_1 = self.encoder(augmented_data_1, training=False)
        features_2 = self.encoder(augmented_data_2, training=False)

        # Pass the features through the projection head
        projections_1 = self.projection_head(features_1, training=False)
        projections_2 = self.projection_head(features_2, training=False)

        # Calculate contrastive loss (during testing, we don't apply gradients)
        contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        # Return only contrastive loss tracker for evaluation
        return {m.name: m.result() for m in self.metrics}


# Contrastive pretraining
pretraining_model = ContrastiveModel()
pretraining_model.compile(
    contrastive_optimizer=keras.optimizers.Adam()
)

pretraining_history = pretraining_model.fit(
    gdataset, epochs=num_epochs, validation_data=gdataset, batch_size=batch_size
)
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


print(time.time() - start)

# Alarm
os.system('play -nq -t alsa synth {} sine {}'.format(1, 999))
