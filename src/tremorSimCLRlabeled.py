"""
## Setup
"""

import os
import time

import numpy as np
import sys
import random

start = time.time()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_BACKEND"] = "tensorflow"

np.set_printoptions(threshold=sys.maxsize)

# Make sure we are able to handle large datasets
import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

import math
import matplotlib.pyplot as plt
import tensorflow as tf

import keras
from keras import ops
from keras import layers
from keras import callbacks
from tf_keras import mixed_precision

from sklearn.manifold import TSNE

from utils import *
from augmentations import Augmentation

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

unlabeled_dataset_size = 10240
labeled_dataset_size = 450

M = 64
E_thres = 0.15 * 3
Kt = 200
batch_size = 512
labeled_gdataset_batch_size = 22
num_epochs = 200
temperature = 0.01
learning_rate = 0.001

"""
## Dataset
"""

# # Adjust the paths to be relative to the current script location
# gdata_path = os.path.join("..", "data", "imu_gdata.pickle")
# sdata_path = os.path.join("..", "data", "imu_sdata.pickle")
# tremor_gdata = unpickle_data(gdata_path)
# tremor_sdata = unpickle_data(sdata_path)
# gdataset = form_unlabeled_tremor_dataset(tremor_gdata, tremor_sdata, E_thres, Kt)

with open("unlabeled_data.pickle", "rb") as f:
    gdataset = pkl.load(f)

print(np.shape(gdataset))

gdataset = normalize(gdataset)

gdataset = gdataset[:unlabeled_dataset_size]

print(np.shape(gdataset))

gdataset = tf.data.Dataset.from_tensor_slices(gdataset)
print("Length of gdataset: ", len(gdataset))
gdataset = (
    gdataset.shuffle(buffer_size=len(gdataset))
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
# gdataset = gdataset.batch(batch_size).shuffle(buffer_size=int(len(gdataset) / batch_size)).prefetch(
#     buffer_size=tf.data.AUTOTUNE)

with open("labeled_windows_dataset.pickle", "rb") as f:
    labeled_gdataset = pkl.load(f)

labeled_gdataset = labeled_gdataset.sample(frac=1).reset_index(drop=True)

print(labeled_gdataset)

labeled_gdataset_train = labeled_gdataset[:150]
labeled_gdataset_test = labeled_gdataset[150:]

print(type(labeled_gdataset))
print("labeled_gdataset shape: ", np.shape(labeled_gdataset))
print("labeled_gdataset_test shape: ", np.shape(labeled_gdataset_test))
print("labeled_gdataset_train shape: ", np.shape(labeled_gdataset_train))

labeled_gdataset_test = tf.data.Dataset.from_tensor_slices(
    (list(labeled_gdataset_test["X"]), list(labeled_gdataset_test["y"]))
)
labeled_gdataset_test = (
    labeled_gdataset_test.shuffle(buffer_size=len(labeled_gdataset_test))
    .batch(1)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# labeled_gdataset = tf.data.Dataset.from_tensor_slices((list(labeled_gdataset['X']), list(labeled_gdataset['y'])))
# labeled_gdataset = labeled_gdataset.shuffle(buffer_size=len(labeled_gdataset)).batch(5).prefetch(
#     buffer_size=tf.data.AUTOTUNE)
#
# print(labeled_gdataset.element_spec)
#
# train_dataset = tf.data.Dataset.zip(
#     (gdataset, labeled_gdataset)
# ).prefetch(buffer_size=tf.data.AUTOTUNE)


# Visualization function
def visualize_augmentations(gdataset, augmentation, num_windows):
    augmenter = augmentation.get_contrastive_augmenter()

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
        axs[0, i].plot(original_windows[i][:, 0], label="X", color="r")  # X-axis (red)
        axs[0, i].plot(
            original_windows[i][:, 1], label="Y", color="g"
        )  # Y-axis (green)
        axs[0, i].plot(original_windows[i][:, 2], label="Z", color="b")  # Z-axis (blue)
        axs[0, i].set_title("Original")
        axs[0, i].legend(loc="upper right")

        # Plot augmented data
        axs[1, i].plot(augmented_windows[i][:, 0], label="X", color="r")  # X-axis (red)
        axs[1, i].plot(
            augmented_windows[i][:, 1], label="Y", color="g"
        )  # Y-axis (green)
        axs[1, i].plot(
            augmented_windows[i][:, 2], label="Z", color="b"
        )  # Z-axis (blue)
        axs[1, i].set_title("Augmented")
        axs[1, i].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


augmentation = Augmentation()
visualize_augmentations(gdataset, augmentation, num_windows=3)


def augment_and_extend_dataset(df, get_contrastive_augmenter, num_extensions=1):
    """
    Augments the dataset and extends it by a specified number of times.
    """

    # Extract acceleration windows and labels
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


num_extensions = labeled_dataset_size // 150 - 1
labeled_gdataset_train = augment_and_extend_dataset(
    labeled_gdataset_train,
    augmentation.get_contrastive_augmenter,
    num_extensions=num_extensions,
)
print("shape of labeled_gdataset: ", np.shape(labeled_gdataset_train))

labeled_gdataset_train = tf.data.Dataset.from_tensor_slices(
    (list(labeled_gdataset_train["X"]), list(labeled_gdataset_train["y"]))
)
labeled_gdataset_train = (
    labeled_gdataset_train.shuffle(buffer_size=len(labeled_gdataset_train))
    .batch(labeled_gdataset_batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

print(labeled_gdataset_train.element_spec)

train_dataset = tf.data.Dataset.zip((gdataset, labeled_gdataset_train)).prefetch(
    buffer_size=tf.data.AUTOTUNE
)

"""
## Encoder architecture
"""


# Define the encoder architecture
def embeddings_function(M):
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
        self.shift_windows = augmentation.shift_windows()
        self.encoder = embeddings_function(M)

        self.current_index = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.similarity_values = tf.Variable(tf.zeros((1000, 2)), trainable=False)

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
            tf.repeat(tf.range(batch_size), batch_size),
        )

        # Apply the mask to extract only negative similarities
        negative_similarities = tf.boolean_mask(negative_similarities, negative_mask)

        return tf.reduce_mean(positive_similarities), tf.reduce_mean(
            negative_similarities
        )

    def train_step(self, data):
        unlabeled_data = data[0]
        labeled_data = data[1][0]
        labels = data[1][1]

        # print("Unlabeled data shape: ", unlabeled_data)
        # print("Labeled data shape: ", labeled_data)
        # print("Labels shape: ", labels)
        # Each window is augmented twice, differently
        augmented_data_1, augmented_data2 = self.shift_windows(
            unlabeled_data, training=True
        )
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
            contrastive_loss = self.contrastive_loss(features_1, features_2)

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
    probe_optimizer=keras.optimizers.Adam(learning_rate=1e-4),
)

checkpoint = callbacks.ModelCheckpoint(
    filepath="simclr_best_model.weights.h5",
    monitor="val_p_loss",
    mode="min",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)


def lr_schedule(epoch, lr):
    total_epochs = num_epochs
    decay_start_epoch = (
        total_epochs // 2
    )  # Start decay at the halfway point of the training
    if epoch == 0:  # For the first epoch, keep the initial learning rate
        return learning_rate
    elif epoch >= decay_start_epoch:
        return lr * 0.99  # Decay logic
    return lr


lr_scheduler = callbacks.LearningRateScheduler(lr_schedule)

pretraining_history = pretraining_model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=labeled_gdataset_test,
    batch_size=batch_size,
    callbacks=[checkpoint, lr_scheduler],
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

pretraining_model.get_layer("embeddings_function").save_weights(
    "tremor_simclr_embeddings.weights.h5"
)
print("Encoder weights saved to tremor_simclr_embeddings.weights.h5")

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
        embeddings_batch = pretraining_model.get_layer("embeddings_function").predict(
            windows_batch
        )

        # Append the embeddings and labels to the lists
        windows.append(embeddings_batch)
        labels.append(labels_batch.numpy())  # Convert TensorFlow tensor to numpy array

    # Stack the results to form a full matrix
    embeddings = np.vstack(windows)  # Convert list of arrays into a full array
    labels = np.hstack(labels)  # Flatten list of label arrays into a single array

    print("Embeddings shape: {}".format(embeddings.shape))
    return embeddings, labels


# Function to visualize the embeddings
def visualize_embeddings(
    embeddings, labels, n_components=2, perplexity=5, learning_rate="auto", n_iter=500
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
    if n_components == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(
            reduced_embeddings[labels == 0, 0],
            reduced_embeddings[labels == 0, 1],
            label="No tremor",
            c="b",
            alpha=0.5,
        )
        plt.scatter(
            reduced_embeddings[labels == 1, 0],
            reduced_embeddings[labels == 1, 1],
            label="Tremor",
            c="r",
            alpha=0.5,
        )
        plt.title("2D t-SNE Visualization of Embeddings")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend()
        plt.show()

    # 3D Visualization
    elif n_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            reduced_embeddings[labels == 0, 0],
            reduced_embeddings[labels == 0, 1],
            reduced_embeddings[labels == 0, 2],
            label="Class 0",
            c="b",
            alpha=0.5,
        )
        ax.scatter(
            reduced_embeddings[labels == 1, 0],
            reduced_embeddings[labels == 1, 1],
            reduced_embeddings[labels == 1, 2],
            label="Class 1",
            c="r",
            alpha=0.5,
        )
        ax.set_title("3D t-SNE Visualization of Embeddings")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_zlabel("t-SNE Dimension 3")
        ax.legend()
        plt.show()
    else:
        raise ValueError("n_components must be 2 or 3 for visualization.")


labeled_gdataset = tf.data.Dataset.from_tensor_slices(
    (list(labeled_gdataset["X"]), list(labeled_gdataset["y"]))
)
labeled_gdataset = (
    labeled_gdataset.shuffle(buffer_size=len(labeled_gdataset))
    .batch(10)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Use the function to get embeddings and visualize them
embeddings, labels = get_labeled_embeddings(pretraining_model, labeled_gdataset)
visualize_embeddings(embeddings, labels, n_components=2)

print(time.time() - start)

# Alarm
os.system('powershell.exe -c "[console]::beep(999,1000)"')

input("Press Enter to exit...")
