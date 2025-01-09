import collections
from collections.abc import Callable
import pickle as pkl
import numpy as np
import time
import os
import gc
import subprocess
import signal
import tensorflow as tf
import tensorflow_federated as tff
import matplotlib.pyplot as plt
from contrastiveModel import Augmentation
from sklearn.manifold import TSNE
from tensorflow.keras import layers
from utils import form_federated_dataset, unpickle_data

start = time.time()

os.environ['TF_CUDNN_USE_AUTOTUNE'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # run on CPU

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# tff.backends.native.set_sync_local_cpp_execution_context()
# policy = tf.keras.mixed_precision.Policy('float32')
# tf.keras.mixed_precision.set_global_policy(policy)

# tff.backends.native.set_sync_local_cpp_execution_context()

if tf.config.list_physical_devices('GPU'):
    print("Using GPU...")
else:
    print("Using CPU...")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled GPU(s): {len(gpus)}")
    except RuntimeError as e:
        print(f"Error setting GPU memory growth: {e}")
else:
    print("No GPUs found, running on CPU.")

E_thres = 0.15*3
Kt = 100


NUM_CLIENTS = 25
NUM_EPOCHS = 5
NUM_ROUNDS = 30
BATCH_SIZE = 50
SHUFFLE_BUFFER = Kt
TEMPERATURE = 0.01
LEARNING_RATE = 0.001
M = 64

# gdata_path = os.path.join('..', 'data', 'tremor_gdata.pickle')
# sdata_path = os.path.join('..', 'data', 'tremor_sdata.pickle')
# tremor_gdata = unpickle_data(gdata_path)
# tremor_sdata = unpickle_data(sdata_path)
# federated_data = form_federated_dataset(tremor_gdata, tremor_sdata, E_thres=E_thres, Kt=Kt, num_clients=NUM_CLIENTS)

# Load the dataset
with open('federated_data.pickle', 'rb') as f:
    federated_data = pkl.load(f)
    f.close()

with open("labeled_windows_dataset.pickle", 'rb') as f:
    labeled_gdataset = pkl.load(f)
    f.close()

labeled_gdataset = tf.data.Dataset.from_tensor_slices((list(labeled_gdataset['X']), list(labeled_gdataset['y'])))
labeled_gdataset = (labeled_gdataset.shuffle(buffer_size=len(labeled_gdataset)).batch(10)
                    .prefetch(buffer_size=tf.data.AUTOTUNE))


# Preprocess each client dataset
def preprocess(dataset):
    def batch_format_fn(element):
        return collections.OrderedDict(
            x=tf.cast(element, tf.float32),  # Input features
        )

    return (tf.data.Dataset.from_tensor_slices(dataset)
            .repeat(NUM_EPOCHS)
            .shuffle(SHUFFLE_BUFFER)
            .batch(BATCH_SIZE)
            .map(batch_format_fn)
            .prefetch(buffer_size=tf.data.AUTOTUNE))

    # return (tf.data.Dataset.from_tensor_slices(dataset)
    #             .shuffle(SHUFFLE_BUFFER, seed=1)
    #             .batch(BATCH_SIZE)
    #             .map(batch_format_fn)
    #             .repeat(NUM_EPOCHS)
    #             .prefetch(buffer_size=tf.data.AUTOTUNE))


def make_federated_data(client_data):
    return [preprocess(client_data[i]) for i in range(len(client_data))]


# Prepare federated datasets
federated_train_data = make_federated_data(federated_data)

print(f'Number of client datasets: {len(federated_train_data)}')
print(f'First dataset: {federated_train_data[0]}')
print(f'Length of first dataset: {len(federated_train_data[0])}')
print(f'Length of second dataset: {len(federated_train_data[1])}')
print(f'Length of third dataset: {len(federated_train_data[2])}')

MnistVariables = collections.namedtuple(
    'MnistVariables', 'encoder num_examples loss_sum accuracy_sum')


class EmbeddingsModel(tf.keras.Model):
    def __init__(self, M):
        super(EmbeddingsModel, self).__init__()
        self.model = tf.keras.Sequential(
            [
                # Layer 1
                tf.keras.layers.ZeroPadding1D(padding=1),
                tf.keras.layers.Conv1D(filters=32, kernel_size=8, padding='valid'),
                # tf.keras.layers.BatchNormalization(center=False, scale=False),
                tf.keras.layers.GroupNormalization(groups=1, axis=-1),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.MaxPooling1D(pool_size=2),

                # Layer 2
                tf.keras.layers.ZeroPadding1D(padding=1),
                tf.keras.layers.Conv1D(filters=32, kernel_size=8, padding='valid'),
                # tf.keras.layers.BatchNormalization(center=False, scale=False),
                tf.keras.layers.GroupNormalization(groups=1, axis=-1),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.MaxPooling1D(pool_size=2),

                # Layer 3
                tf.keras.layers.ZeroPadding1D(padding=1),
                tf.keras.layers.Conv1D(filters=16, kernel_size=16, padding='valid'),
                # tf.keras.layers.BatchNormalization(center=False, scale=False),
                tf.keras.layers.GroupNormalization(groups=1, axis=-1),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.MaxPooling1D(pool_size=2),

                # Layer 4
                tf.keras.layers.ZeroPadding1D(padding=1),
                tf.keras.layers.Conv1D(filters=16, kernel_size=16, padding='valid'),
                # tf.keras.layers.BatchNormalization(center=False, scale=False),
                tf.keras.layers.GroupNormalization(groups=1, axis=-1),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.MaxPooling1D(pool_size=2),

                # Flatten and Dense layer to get M-dimensional output
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(M),
            ],
            name="embeddings_function"
        )

    def call(self, inputs, **kwargs):
        return self.model(inputs, **kwargs)


def normalize_data(data):
    # Find the min and max values for each batch (across time_steps and channels)
    min_val = tf.reduce_min(data, axis=(1, 2), keepdims=True)  # shape: (batch_size, 1, 1)
    max_val = tf.reduce_max(data, axis=(1, 2), keepdims=True)  # shape: (batch_size, 1, 1)

    # Apply the normalization formula
    data_normalized = 2 * (data - min_val) / (max_val - min_val) - 1

    return data_normalized


def apply_augmentations(data, augmenter):
    """
    Applies a sequence of augmentations to the input data.
    """
    data = augmenter.left_to_right_flipping(data)
    data = augmenter.bidirectional_flipping(data)
    data = augmenter.rotate_axis(data)
    data = augmenter.add_gravity(data)
    data = augmenter.permute_segments(data)
    data = normalize_data(data)
    return data


def create_mnist_variables():
    encoder = EmbeddingsModel(M)
    encoder.build(input_shape=(None, 1000, 3))
    return MnistVariables(
        encoder=encoder,
        num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
        loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
        accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False),
    )


def contrastive_loss(projections_1, projections_2):
    # Normalize the projections
    projections_1 = tf.nn.l2_normalize(projections_1, axis=1)
    projections_2 = tf.nn.l2_normalize(projections_2, axis=1)

    # Compute similarities
    similarities = tf.matmul(projections_1, projections_2, transpose_b=True) / TEMPERATURE

    # Create contrastive labels
    batch_size = tf.shape(projections_1)[0]
    contrastive_labels = tf.range(batch_size)

    # Compute contrastive loss
    loss_1_2 = tf.keras.losses.sparse_categorical_crossentropy(contrastive_labels, similarities, from_logits=True)
    loss_2_1 = tf.keras.losses.sparse_categorical_crossentropy(contrastive_labels, tf.transpose(similarities),
                                                               from_logits=True)

    loss = tf.reduce_mean(loss_1_2 + loss_2_1) / 2

    # Compute predictions
    predictions = tf.argmax(similarities, axis=-1, output_type=tf.int32)

    # Compute accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, contrastive_labels), tf.float32))

    return loss, accuracy, predictions


def predict_on_batch(variables, x):
    return variables.encoder(x, training=True)


def mnist_forward_pass(variables, batch):
    # Augmentation
    augmenter = Augmentation()

    augmented_data_1, augmented_data_2 = augmenter.shift_windows_fun(batch['x'])

    augmented_data_1 = apply_augmentations(augmented_data_1, augmenter)
    augmented_data_2 = apply_augmentations(augmented_data_2, augmenter)

    # Forward pass
    y_1 = predict_on_batch(variables, augmented_data_1)
    y_2 = predict_on_batch(variables, augmented_data_2)

    print(f'y_1: {y_1}')
    print(f'y_2: {y_2}')

    loss, accuracy, predictions = contrastive_loss(y_1, y_2)

    print(f'loss: {loss}')
    print(f'accuracy: {accuracy}')
    print(f'predictions: {predictions}')

    num_examples = tf.cast(tf.size(batch['x']), tf.float32)

    variables.num_examples.assign_add(num_examples)
    variables.loss_sum.assign_add(loss * num_examples)
    variables.accuracy_sum.assign_add(accuracy * num_examples)

    return loss, predictions


def get_local_unfinalized_metrics(variables):
    return collections.OrderedDict(
        num_examples=[variables.num_examples],
        loss=[variables.loss_sum, variables.num_examples],
        accuracy=[variables.accuracy_sum, variables.num_examples])


def get_metric_finalizers():
    return collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x[0]),
        loss=tf.function(func=lambda x: x[0] / x[1]),
        accuracy=tf.function(func=lambda x: x[0] / x[1]))


class MnistModel(tff.learning.models.VariableModel):

    def __init__(self):
        self._variables = create_mnist_variables()

    @property
    def trainable_variables(self):
        return self._variables.encoder.trainable_variables

    @property
    def non_trainable_variables(self):
        return self._variables.encoder.non_trainable_variables

    @property
    def local_variables(self):
        return [
            self._variables.num_examples, self._variables.loss_sum,
            self._variables.accuracy_sum
        ]

    @property
    def input_spec(self):
        return collections.OrderedDict(
            x=tf.TensorSpec([None, 1500, 3], tf.float32))

    @tf.function
    def predict_on_batch(self, x, training=True):
        del training
        return predict_on_batch(self._variables, x)

    @tf.function
    def forward_pass(self, batch, training=True):
        del training
        loss, predictions = mnist_forward_pass(self._variables, batch)
        num_examples = tf.shape(batch['x'])[0]
        return tff.learning.models.BatchOutput(
            loss=loss, predictions=predictions, num_examples=num_examples)

    @tf.function
    def report_local_unfinalized_metrics(
            self) -> collections.OrderedDict[str, list[tf.Tensor]]:
        """Creates an `OrderedDict` of metric names to unfinalized values."""
        return get_local_unfinalized_metrics(self._variables)

    def metric_finalizers(
            self) -> collections.OrderedDict[str, Callable[[list[tf.Tensor]], tf.Tensor]]:
        """Creates an `OrderedDict` of metric names to finalizers."""
        return get_metric_finalizers()

    @tf.function
    def reset_metrics(self):
        """Resets metrics variables to initial value."""
        for var in self.local_variables:
            var.assign(tf.zeros_like(var))


training_process = tff.learning.algorithms.build_weighted_fed_avg(
    MnistModel,
    client_optimizer_fn=tff.learning.optimizers.build_adam(learning_rate=1e-3),
    server_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=1.0))

train_state = training_process.initialize()

result = training_process.next(train_state, federated_train_data)
train_state = result.state
metrics = result.metrics
print('round  1, metrics={}'.format(metrics))

for round_num in range(2, NUM_ROUNDS + 1):
    result = training_process.next(train_state, federated_train_data)
    train_state = result.state
    metrics = result.metrics
    print('round {:2d}, metrics={}'.format(round_num, metrics))

# SAVE WEIGHTS

# Extract weights from the federated model
model_weights = training_process.get_model_weights(train_state)
all_model_weights = model_weights.trainable + model_weights.non_trainable

# Create and build the embeddings model
embeddings_model = EmbeddingsModel(M)
embeddings_model.build(input_shape=(None, 1000, 3))

# Inspect the shapes of the federated and local model weights
# print("Federated weights:")
# for i, weight in enumerate(all_model_weights):
#     print(f"Weight {i}: Name = {weight.name if hasattr(weight, 'name') else 'Unnamed'}, Shape = {weight.shape}")
#
# print("Type of embeddings_model.weights: ", type(embeddings_model.weights))
# print("Type of embeddings_model.weights[0]: ", type(embeddings_model.weights[0]))
#
# print("\nLocal model expected weights:")
# for i, weight in enumerate(embeddings_model.weights):
#     print(f"Weight {i}: Name = {weight.name}, Shape = {weight.shape}")

# Reorder federated weights to match the local model
reordered_weights = []
reordered_weight_names = []
used_indices = set()  # To track matched indices in `all_model_weights`

for lw in embeddings_model.weights:  # Iterate over local model weights
    lw_name = lw.name  # Get the name of the local weight
    for idx, fw in enumerate(all_model_weights):  # Iterate over federated weights
        if idx not in used_indices and lw.shape == fw.shape:
            reordered_weights.append(fw)  # Add the federated weight
            reordered_weight_names.append(lw_name)  # Add the corresponding name
            used_indices.add(idx)  # Mark this index as used
            break  # Exit the inner loop once a match is found

# Print the reordered weights and their corresponding local weight names
# print("Reordered weights with corresponding local weight names:")
# for idx, (weight, name) in enumerate(zip(reordered_weights, reordered_weight_names)):
#     print(f"Reordered Weight {idx}: Shape = {weight.shape}, Local Weight Name = {name}")
#
# print("Type of reordered_weights: ", type(reordered_weights))
# print("Type of reordered_weights[0]: ", type(reordered_weights[0]))

# Set the reordered weights and save
embeddings_model.set_weights(reordered_weights)

# print("\nReordered weights:")
# for i, weight in enumerate(embeddings_model.weights):
#     print(f"Weight {i}: Name = {weight.name}, Shape = {weight.shape}")
#
# print("type of embeddings_model.weights: ", type(embeddings_model.weights))
# print("type of embeddings_model.weights[0]: ", type(embeddings_model.weights[0]))

embeddings_model.save_weights("federated.weights.h5")
print("Embeddings function weights saved successfully.")

# embeddings_model.load_weights("federated.weights.h5")


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
    print("Labels shape: {}".format(labels.shape))
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


embeddings, labels = get_labeled_embeddings(embeddings_model, labeled_gdataset)
visualize_embeddings(embeddings, labels, n_components=2)


try:
    # Find all processes with 'worker_binary' in the name
    output = subprocess.check_output(['pgrep', '-f', 'worker_binary'])
    pids = output.decode().strip().split('\n')
    for pid in pids:
        os.kill(int(pid), signal.SIGTERM)  # Send SIGTERM to gracefully terminate
    print("TFF worker processes cleaned up.")
except subprocess.CalledProcessError:
    print("No TFF worker processes found.")

print(time.time() - start)

# Alarm
os.system('play -nq -t alsa synth {} sine {}'.format(1, 999))
