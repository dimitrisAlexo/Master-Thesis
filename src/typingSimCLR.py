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

unlabeled_dataset_size = 5120
labeled_dataset_size = 450  # Will be adjusted based on actual labeled data

M = 64
K2 = 100
batch_size = 512
labeled_batch_size = 45  # Batch size for labeled data
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

# Load labeled typing histograms dataset
with open("labeled_typing_histograms_dataset.pickle", "rb") as f:
    labeled_gdataset = pkl.load(f)

labeled_gdataset = labeled_gdataset.sample(frac=1).reset_index(drop=True)

print(f"Labeled dataset shape: {labeled_gdataset.shape}")
print("Label distribution:")
print(labeled_gdataset["y"].value_counts())

# Split labeled dataset into train and test
labeled_gdataset_train = labeled_gdataset[: int(len(labeled_gdataset) * 0.8)]
labeled_gdataset_test = labeled_gdataset[int(len(labeled_gdataset) * 0.8) :]

print(f"Labeled train shape: {labeled_gdataset_train.shape}")
print(f"Labeled test shape: {labeled_gdataset_test.shape}")

# Convert to TensorFlow datasets
labeled_gdataset_test = tf.data.Dataset.from_tensor_slices(
    (list(labeled_gdataset_test["X"]), list(labeled_gdataset_test["y"]))
)
labeled_gdataset_test = (
    labeled_gdataset_test.shuffle(buffer_size=len(labeled_gdataset_test))
    .batch(1)
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
        n_perm_seg=5,
        redistribution_threshold=0.03,
    ):
        self.noise_factor = noise_factor
        self.dropout_rate = dropout_rate
        self.n_perm_seg = n_perm_seg
        self.redistribution_threshold = redistribution_threshold

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

    def permute_histogram_segments(self, data):
        """
        Permute segments of the histogram features separately for hold time and flight time.
        For typing data with 502-dimensional histogram:
        - Hold time section (0-100): 101 features
        - Flight time section (101-501): 401 features
        This disrupts the temporal structure within each section while preserving
        the separation between hold time and flight time characteristics.
        """
        batch_size = tf.shape(data)[0]

        # Split the data into hold time (0-100) and flight time (101-501) sections
        hold_time_data = data[:, :101]  # First 101 features (0-100)
        flight_time_data = data[:, 101:]  # Remaining 401 features (101-501)

        def permute_section(section_data):
            """Permute a single section (hold time or flight time)"""
            section_features = tf.shape(section_data)[1]

            # Calculate the divisor and remainder for segmentation
            divisor = section_features // self.n_perm_seg
            remainder = section_features % self.n_perm_seg

            def permute_segments():
                # Reshape the first n_perm_seg - 1 segments with size divisor
                reshaped_data_1 = tf.reshape(
                    section_data[:, : divisor * (self.n_perm_seg - 1)],
                    [batch_size, self.n_perm_seg - 1, divisor],
                )

                # Reshape the last segment to include the remainder (divisor + remainder)
                last_segment_start = divisor * (self.n_perm_seg - 1)
                reshaped_data_2 = tf.reshape(
                    section_data[:, last_segment_start:],
                    [batch_size, divisor + remainder],
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
                    permuted_data, [batch_size, section_features - divisor - remainder]
                )

                # Concatenate with the last segment
                result = tf.concat([permuted_data, reshaped_data_2], axis=1)
                return result

            def return_original():
                return section_data

            # Use tf.cond to conditionally apply permutation
            return tf.cond(divisor >= 1, permute_segments, return_original)

        # Permute hold time and flight time sections separately
        permuted_hold_time = permute_section(hold_time_data)
        permuted_flight_time = permute_section(flight_time_data)

        # Concatenate the permuted sections back together
        return tf.concat([permuted_hold_time, permuted_flight_time], axis=1)

    def normalize_histogram(self, data):
        """
        Normalize histograms so that each section sums to 1.
        For typing data:
        - Hold time section (0-100): normalized to sum to 1
        - Flight time section (101-501): normalized to sum to 1
        """
        # Split the data into hold time (0-100) and flight time (101-501) sections
        hold_time_data = data[:, :101]  # First 101 features (0-100)
        flight_time_data = data[:, 101:]  # Remaining 401 features (101-501)

        # Normalize hold time section to sum to 1
        hold_time_sum = tf.reduce_sum(hold_time_data, axis=1, keepdims=True)
        # Avoid division by zero
        hold_time_sum = tf.maximum(hold_time_sum, 1e-8)
        normalized_hold_time = hold_time_data / hold_time_sum

        # Normalize flight time section to sum to 1
        flight_time_sum = tf.reduce_sum(flight_time_data, axis=1, keepdims=True)
        # Avoid division by zero
        flight_time_sum = tf.maximum(flight_time_sum, 1e-8)
        normalized_flight_time = flight_time_data / flight_time_sum

        # Concatenate the normalized sections back together
        return tf.concat([normalized_hold_time, normalized_flight_time], axis=1)

    def uniform_redistribution(self, data):
        """
        Redistribute probability mass uniformly among histogram bins above threshold.
        This augmentation flattens the distribution while preserving active regions
        and maintaining the normalization property (sum = 1 for each section).
        """
        threshold = self.redistribution_threshold

        # Split the data into hold time (0-100) and flight time (101-501) sections
        hold_time_data = data[:, :101]  # First 101 features (0-100)
        flight_time_data = data[:, 101:]  # Remaining 401 features (101-501)

        def redistribute_section(section_data):
            """Redistribute probability mass within a section"""
            batch_size = tf.shape(section_data)[0]
            section_features = tf.shape(section_data)[1]

            # Create mask for bins above threshold
            above_threshold = section_data > threshold

            # Count number of active bins per sample
            num_active_bins = tf.reduce_sum(
                tf.cast(above_threshold, tf.float32), axis=1, keepdims=True
            )

            # Avoid division by zero - if no bins are above threshold, keep original
            num_active_bins = tf.maximum(num_active_bins, 1.0)

            # Calculate current sum for normalization
            current_sum = tf.reduce_sum(section_data, axis=1, keepdims=True)
            current_sum = tf.maximum(current_sum, 1e-8)

            # Calculate uniform value for active bins
            uniform_value = current_sum / num_active_bins

            # Create redistributed section: uniform value for active bins, zero for inactive
            redistributed = tf.where(above_threshold, uniform_value, 0.0)

            return redistributed

        # Apply redistribution to both sections
        redistributed_hold_time = redistribute_section(hold_time_data)
        redistributed_flight_time = redistribute_section(flight_time_data)

        # Concatenate the redistributed sections back together
        return tf.concat([redistributed_hold_time, redistributed_flight_time], axis=1)

    def get_contrastive_augmenter(self):
        """Combine several augmentations into a single sequential model."""
        return keras.Sequential(
            [
                layers.Lambda(self.add_noise),
                # layers.Lambda(self.dropout_features),
                layers.Lambda(self.permute_histogram_segments),
                layers.Lambda(self.uniform_redistribution),
                layers.Lambda(self.normalize_histogram),
            ]
        )

    def get_classification_augmenter(self):
        """Lighter augmentation for classification/linear probe training."""
        return keras.Sequential(
            [
                layers.Lambda(self.add_noise),
                # layers.Lambda(self.dropout_features),
                # layers.Lambda(self.permute_histogram_segments),
                layers.Lambda(self.uniform_redistribution),
                layers.Lambda(self.normalize_histogram),
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
    plt.show(block=False)


augmentation = TypingAugmentation()
visualize_typing_augmentations(gdataset, augmentation, num_histograms=3)


def augment_and_extend_dataset(df, get_contrastive_augmenter, num_extensions=1):
    """
    Augments the dataset and extends it by a specified number of times.
    """
    # Extract histograms and labels
    X_original = tf.convert_to_tensor(df["X"].to_list(), dtype=tf.float32)
    y_original = tf.convert_to_tensor(df["y"].to_list(), dtype=tf.int32)

    # Get augmenter
    augmenter = get_contrastive_augmenter()

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


# Extend labeled training dataset through augmentation
num_extensions = max(1, labeled_dataset_size // len(labeled_gdataset_train) - 1)
labeled_gdataset_train = augment_and_extend_dataset(
    labeled_gdataset_train,
    augmentation.get_contrastive_augmenter,
    num_extensions=num_extensions,
)
print(f"Extended labeled training dataset shape: {labeled_gdataset_train.shape}")

# Convert to TensorFlow dataset
labeled_gdataset_train = tf.data.Dataset.from_tensor_slices(
    (list(labeled_gdataset_train["X"]), list(labeled_gdataset_train["y"]))
)
labeled_gdataset_train = (
    labeled_gdataset_train.shuffle(buffer_size=len(labeled_gdataset_train))
    .batch(labeled_batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

print(labeled_gdataset_train.element_spec)

# Create combined training dataset
train_dataset = tf.data.Dataset.zip((gdataset, labeled_gdataset_train)).prefetch(
    buffer_size=tf.data.AUTOTUNE
)

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
        self.classification_augmenter = augmentation.get_classification_augmenter()
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

        # Single dense layer for linear probing
        self.linear_probe = keras.Sequential(
            [
                layers.Input(shape=(M,)),
                layers.Dropout(0.1),
                layers.Dense(2, kernel_regularizer=keras.regularizers.L2(1e-4)),
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
            self.probe_accuracy,
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
        unlabeled_data = data[0]
        labeled_data = data[1][0]
        labels = data[1][1]

        with tf.GradientTape() as tape:
            # Apply two different augmentations to the same histograms
            augmented_histograms_1 = self.contrastive_augmenter(unlabeled_data)
            augmented_histograms_2 = self.contrastive_augmenter(unlabeled_data)

            # Generate embeddings for both augmented versions
            embeddings_1 = self.encoder(augmented_histograms_1, training=True)
            embeddings_2 = self.encoder(augmented_histograms_2, training=True)

            # Compute contrastive loss
            contrastive_loss = self.contrastive_loss(embeddings_1, embeddings_2)

        # Compute gradients of the contrastive loss and update the encoder
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

        # Update the contrastive loss tracker
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

        # Labels are used for linear probing
        preprocessed_data = self.classification_augmenter(labeled_data, training=True)

        with tf.GradientTape() as tape:
            # The encoder is used in inference mode here to avoid updating batch norm
            features = self.encoder(preprocessed_data, training=False)
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
        preprocessed_data = self.classification_augmenter(labeled_data, training=False)
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
        # Plot Contrastive Loss
        plt.figure(figsize=(6, 5))
        plt.plot(
            pretraining_history.history["c_loss"],
            label="Contrastive Loss",
            color="blue",
        )
        plt.title("Contrastive Loss per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        # Plot Validation Loss
        plt.figure(figsize=(6, 5))
        plt.plot(
            pretraining_history.history["val_p_loss"],
            label="Validation Loss",
            color="red",
        )
        plt.title("Validation Loss per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_validation_accuracy(self, pretraining_history):
        """
        Plots validation accuracy per epoch.
        """
        plt.figure(figsize=(6, 5))
        plt.plot(
            pretraining_history.history["val_p_acc"],
            label="Linear Probing Accuracy",
            color="orange",
        )
        plt.title("Linear Probing Accuracy per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()


# Contrastive pretraining
pretraining_model = ContrastiveModel()
pretraining_model.compile(
    contrastive_optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    probe_optimizer=keras.optimizers.Adam(learning_rate=5e-4),
)

checkpoint = callbacks.ModelCheckpoint(
    filepath="typing_simclr_best_model.weights.h5",
    monitor="val_p_loss",
    mode="min",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)


def lr_schedule(epoch, lr):
    total_epochs = num_epochs
    decay_start_epoch = total_epochs // 2
    if epoch == 0:  # For the first epoch, keep the initial learning rate
        return learning_rate
    elif epoch >= decay_start_epoch:
        return lr * 0.99  # Decay logic
    return lr


lr_scheduler = callbacks.LearningRateScheduler(lr_schedule)

# Train the model
pretraining_history = pretraining_model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=labeled_gdataset_test,
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
pretraining_model.plot_validation_accuracy(pretraining_history)


def get_labeled_embeddings(pretraining_model, labeled_gdataset):
    histograms = []
    labels = []

    # Iterate through the batched dataset and collect histograms and labels
    for batch in labeled_gdataset:
        histograms_batch = batch[0]  # Extract the histograms from the batch
        labels_batch = batch[1]  # Extract the labels from the batch

        # Predict the embeddings for the entire batch of histograms
        embeddings_batch = pretraining_model.get_layer("embeddings_function").predict(
            histograms_batch
        )

        # Append the embeddings and labels to the lists
        histograms.append(embeddings_batch)
        labels.append(labels_batch.numpy())  # Convert TensorFlow tensor to numpy array

    # Stack the results to form a full matrix
    embeddings = np.vstack(histograms)  # Convert list of arrays into a full array
    labels = np.hstack(labels)  # Flatten list of label arrays into a single array

    print("Embeddings shape: {}".format(embeddings.shape))
    return embeddings, labels


# Function to visualize the embeddings
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
    plt.title("2D t-SNE Visualization of Typing Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.show()



# Prepare labeled dataset for embedding visualization
labeled_gdataset_viz = tf.data.Dataset.from_tensor_slices(
    (list(labeled_gdataset["X"]), list(labeled_gdataset["y"]))
)
labeled_gdataset_viz = (
    labeled_gdataset_viz.shuffle(buffer_size=len(labeled_gdataset_viz))
    .batch(10)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Use the function to get embeddings and visualize them
embeddings, labels = get_labeled_embeddings(pretraining_model, labeled_gdataset_viz)
visualize_embeddings(embeddings, labels, n_components=2)

print(f"Total training time: {time.time() - start} seconds")

# Alarm
os.system('powershell.exe -c "[console]::beep(999,1000)"')

input("Press Enter to exit...")
