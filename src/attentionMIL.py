import numpy as np

from utils import *
import os
import time
import gc
import sys

import tensorflow as tf
import keras
from keras import layers
from keras import ops
from keras import callbacks
from keras import optimizers
from keras import metrics
from tf_keras import backend as k
from tf_keras import mixed_precision
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import confusion_matrix

start = time.time()

np.set_printoptions(threshold=sys.maxsize)

"""
## Create dataset

We will create a set of bags and assign their labels according to their contents.
If at least one positive instance
is available in a bag, the bag is considered as a positive bag. If it does not contain any
positive instance, the bag will be considered as negative.

### Configuration parameters

- `POSITIVE_CLASS`: The desired class to be kept in the positive bag.
- `BAG_COUNT`: The number of training bags.
- `VAL_BAG_COUNT`: The number of validation bags.
- `BAG_SIZE`: The number of instances in a bag.
- `PLOT_SIZE`: The number of bags to plot.
- `ENSEMBLE_AVG_COUNT`: The number of models to create and average together. (Optional:
often results in better performance - set to 1 for single model)
"""

ENSEMBLE_AVG_COUNT = 1

# os.environ["tf_gpu_allocator"] = "cuda_malloc_async"

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # run on CPU

if tf.config.list_physical_devices('GPU'):
    print("Using GPU...")
else:
    print("Using CPU...")


# # Mixed precision policy
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)
# print("Using mixed precision...")

# k.set_floatx("float16")
# print(k.floatx())


class MILAttentionLayer(layers.Layer):
    """Implementation of the attention-based Deep MIL layer.

    Args:
      weight_params_dim: Positive Integer. Dimension of the weight matrix.
      kernel_initializer: Initializer for the `kernel` matrix.
      kernel_regularizer: Regularizer function applied to the `kernel` matrix.
      use_gated: Boolean, whether or not to use the gated mechanism.

    Returns:
      List of 2D tensors with BAG_SIZE length.
      The tensors are the attention scores after softmax with shape `(batch_size, 1)`.
    """

    def __init__(
            self,
            weight_params_dim,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=None,
            use_gated=False,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.v_init = self.kernel_initializer
        self.w_init = self.kernel_initializer
        self.u_init = self.kernel_initializer

        self.v_regularizer = self.kernel_regularizer
        self.w_regularizer = self.kernel_regularizer
        self.u_regularizer = self.kernel_regularizer

    def build(self, input_shape):
        # Input shape.
        input_dim = M

        self.v_weight_params = self.add_weight(
            shape=(input_dim, self.weight_params_dim),
            initializer=self.v_init,
            name="v",
            regularizer=self.v_regularizer,
            trainable=True,
        )

        self.w_weight_params = self.add_weight(
            shape=(self.weight_params_dim, 1),
            initializer=self.w_init,
            name="w",
            regularizer=self.w_regularizer,
            trainable=True,
        )

        if self.use_gated:
            self.u_weight_params = self.add_weight(
                shape=(input_dim, self.weight_params_dim),
                initializer=self.u_init,
                name="u",
                regularizer=self.u_regularizer,
                trainable=True,
            )
        else:
            self.u_weight_params = None

        self.input_built = True

    def call(self, inputs, mask_layer):
        # Assigning variables from the number of inputs.
        instance_weights = self.compute_attention_scores(inputs)

        # Apply masking
        # masked_weights = layers.Add()([mask_layer, instance_weights])

        # Apply softmax over instances such that the output summation is equal to 1.
        alpha = ops.softmax(instance_weights, axis=0)

        # Split to recreate the same array of tensors we had as inputs.
        return alpha

    def compute_attention_scores(self, instance):
        # Reserve in-case "gated mechanism" used.
        original_instance = instance

        # tanh(v*h_k^T)
        instance = ops.tanh(ops.tensordot(instance, self.v_weight_params, axes=1))

        # for learning non-linear relations efficiently.
        if self.use_gated:
            instance = instance * ops.sigmoid(
                ops.tensordot(original_instance, self.u_weight_params, axes=1)
            )

        # w^T*(tanh(v*h_k^T)) / w^T*(tanh(v*h_k^T)*sigmoid(u*h_k^T))
        return ops.tensordot(instance, self.w_weight_params, axes=1)


def embeddings_function(embeddings_input, M):
    # Layer 1
    x = layers.Conv1D(filters=32, kernel_size=8, padding='same')(embeddings_input)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Layer 2
    x = layers.Conv1D(filters=32, kernel_size=8, padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Layer 3
    x = layers.Conv1D(filters=16, kernel_size=16, padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Layer 4
    x = layers.Conv1D(filters=16, kernel_size=16, padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Flatten and Dense layer to get M-dimensional output
    x = layers.Flatten()(x)
    x = layers.Dense(M)(x)

    return x


def final_classifier(concat):
    # Layer 1: Dense M → 32, Leaky-ReLU (α = 0.2), Dropout p = 0.2
    dense_1 = layers.Dense(32)(concat)
    leaky_relu_1 = layers.LeakyReLU(negative_slope=0.2)(dense_1)
    dropout_1 = layers.Dropout(0.2)(leaky_relu_1)

    # Layer 2: Dense 32 → 16, Leaky-ReLU (α = 0.2), Dropout p = 0.2
    dense_2 = layers.Dense(16)(dropout_1)
    leaky_relu_2 = layers.LeakyReLU(negative_slope=0.2)(dense_2)
    dropout_2 = layers.Dropout(0.2)(leaky_relu_2)

    # Layer 3: Dense 16 → 2, 2-way softmax
    output = layers.Dense(2, activation='softmax')(dropout_2)

    # output = layers.Dense(2, activation="softmax")(concat)

    return output


def create_model(input_shape):
    # Extract features from inputs.
    model_input = layers.Input(shape=input_shape)

    def create_mask_layer(inputs):
        # Sum the features along the last two dimensions (500, 3)
        summed_features = tf.reduce_sum(inputs, axis=[2, 3], keepdims=True)

        # Squeeze to remove the extra dimension
        summed_features = tf.squeeze(summed_features, axis=-1)

        # Create a mask where summed_features is zero
        mask = tf.where(tf.abs(summed_features) < 1e-6, -1e30, summed_features)

        return mask

    mask_layer = layers.Lambda(lambda x: create_mask_layer(x))(model_input)

    embeddings = layers.Lambda(lambda x: tf.reshape(x, (-1, Ws, C)))(model_input)
    embeddings = embeddings_function(embeddings, M)
    embeddings = layers.Lambda(lambda x: tf.reshape(x, (-1, Kt, M)))(embeddings)

    # Invoke the attention layer.
    alpha = MILAttentionLayer(
        weight_params_dim=16,
        kernel_regularizer=keras.regularizers.L2(0.01),
        use_gated=True,
        name="alpha",
    )(embeddings, mask_layer)

    # Multiply attention weights with the input layers.
    weighted_embeddings = layers.multiply([alpha, embeddings])

    # Sum the weighted embeddings
    z = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(weighted_embeddings)

    # Classification output node.
    output = final_classifier(z)

    return keras.Model(model_input, output)


def compute_class_weights(labels):
    # Count number of postive and negative bags.
    negative_count = len(np.where(labels == 0)[0])
    positive_count = len(np.where(labels == 1)[0])
    total_count = negative_count + positive_count

    # Build class weight dictionary.
    return {
        0: (1 / negative_count) * (total_count / 2),
        1: (1 / positive_count) * (total_count / 2),
    }


class ClearMemory(callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()


def train(train_data, train_labels, val_data, val_labels, model):
    # Train model.
    # Prepare callbacks.
    # Path where to save best weights.

    # Take the file name from the wrapper.
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../best_model.weights.h5")

    # Initialize model checkpoint callback.
    model_checkpoint = callbacks.ModelCheckpoint(
        file_path,
        monitor="val_loss",
        verbose=0,
        mode="min",
        save_best_only=True,
        save_weights_only=True,
    )

    # Initialize early stopping callback.
    # The model performance is monitored across the validation data and stops training
    # when the generalization error cease to decrease.
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min",
        verbose=1
    )

    f1_score = metrics.F1Score(
        average="weighted",
        threshold=0.5,
        name='f1_score'
    )

    clear_memory = ClearMemory()

    # Compile model.
    model.compile(
        optimizer="adam",
        # optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Fit model.
    model.fit(
        train_data,
        train_labels,
        validation_data=(val_data, val_labels),
        epochs=100,
        class_weight=compute_class_weights(train_labels),
        batch_size=8,
        callbacks=[early_stopping, model_checkpoint],
        verbose=0,
    )

    # Load best weights.
    model.load_weights(file_path)

    return model


# Adjust the paths to be relative to the current script location
sdata_path = os.path.join('..', 'data', 'tremor_sdata.pickle')
gdata_path = os.path.join('..', 'data', 'tremor_gdata.pickle')
tremor_sdata, tremor_gdata = unpickle_data(sdata_path, gdata_path)

E_thres = 0.15
Kt = 100
sdataset, mask = form_dataset(tremor_sdata, E_thres, Kt)

print(sdataset)
print(mask)
print(np.shape(mask))

# print(min(len(sdataset['X'][i]) for i in range(len(sdataset['X']))))
#
# count = 0
# for index, row in sdataset.iterrows():
#         bag = row['X']
#         num_instances = len(bag)
#         print(f"Bag {index} contains {num_instances} instances.")
#         count += 1
# print(count)

# Split the dataset into training and validation sets with an 80/20 split
train_df, val_df = train_test_split(sdataset, test_size=0.2, random_state=42)

train_data = np.array(train_df['X'].tolist())
train_labels = np.array([np.array([label]) for label in train_df['y'].tolist()])

val_data = np.array(val_df['X'].tolist())
val_labels = np.array([np.array([label]) for label in val_df['y'].tolist()])

# Print lengths to verify the split
print(f'Training data size: {np.shape(np.array(train_data))}')

# train_data = list(np.transpose(train_data, (1, 0, 2, 3)))

print(f'Training data: {np.shape(np.array(train_data))}')

print(f'Training labels: {np.shape(train_labels)}')
print(f'Validation data: {len(val_data)}')
print(f'Validation labels: {val_labels.shape}')

print(type(train_data))
print(type(train_data[0]))
print(type(train_data[0][0]))
print(type(train_labels))
print(type(train_labels[0]))
print(np.shape(train_data))
print(np.shape(train_labels))

# # Print the splits
# print(f'Training data: {train_data}')
# print(f'Training labels: {train_labels}')
# print(f'Validation data: {val_data}')
# print(f'Validation labels: {val_labels}')

# Building model(s).
B, Kt, Ws, C = train_data.shape
print(B, Kt, Ws, C)
input_shape = (Kt, Ws, C)
print(input_shape)
M = 64
models = [create_model(input_shape) for _ in range(ENSEMBLE_AVG_COUNT)]

# print(models[0].input)
# print(models[0].output)

# Show single model architecture.
print(models[0].summary())

# Training model(s).
trained_models = [
    train(train_data, train_labels, val_data, val_labels, model)
    for model in tqdm(models)
]


def predict(data, labels, trained_models):
    # Collect info per model.
    models_predictions = []
    models_attention_weights = []
    models_losses = []
    models_accuracies = []

    for model in trained_models:
        # Predict output classes on data.
        predictions = model.predict(data)
        models_predictions.append(predictions)

        # Create intermediate model to get MIL attention layer weights.
        intermediate_model = keras.Model(model.input, model.get_layer("alpha").output)

        # Predict MIL attention layer weights.
        intermediate_predictions = intermediate_model.predict(data)

        attention_weights = np.squeeze(np.swapaxes(intermediate_predictions, 1, 0))
        models_attention_weights.append(attention_weights)

        loss, accuracy = model.evaluate(data, labels, verbose=0)
        models_losses.append(loss)
        models_accuracies.append(accuracy)

    print(
        f"The average loss and accuracy are {np.sum(models_losses, axis=0) / ENSEMBLE_AVG_COUNT:.2f}"
        f" and {100 * np.sum(models_accuracies, axis=0) / ENSEMBLE_AVG_COUNT:.2f} % resp."
    )

    return (
        np.sum(models_predictions, axis=0) / ENSEMBLE_AVG_COUNT,
        np.sum(models_attention_weights, axis=0) / ENSEMBLE_AVG_COUNT,
    )


# Evaluate and predict classes and attention scores on validation data.
class_predictions, attention_params = predict(val_data, val_labels, trained_models)


def loso_evaluate(data):
    # Extract the bags and labels
    bags = data['X'].tolist()
    labels = data['y'].tolist()

    # Initialize LeaveOneOut
    loo = LeaveOneOut()
    overall_accuracies = []

    tn, fp, fn, tp = 0, 0, 0, 0

    for train_index, test_index in loo.split(bags):
        # Split the data into training and validation sets
        train_bags = [bags[i] for i in train_index]
        train_labels = [labels[i] for i in train_index]
        val_bag = [bags[i] for i in test_index]
        val_label = [labels[i] for i in test_index]

        # Prepare train data
        train_data = np.array([np.array(instance) for instance in train_bags])
        train_data = list(np.transpose(train_data, (1, 0, 2, 3)))
        train_labels = np.array([np.array([label]) for label in train_labels])

        # Prepare validation data
        val_data = np.array([np.array(instance) for instance in val_bag])
        val_data = list(np.transpose(val_data, (1, 0, 2, 3)))
        val_labels = np.array([np.array([label]) for label in val_label])

        models = [create_model(input_shape) for _ in range(ENSEMBLE_AVG_COUNT)]

        # Train the models on the training data
        trained_models = [train(train_data, train_labels, val_data, val_labels, model) for model in models]

        # Evaluate the model on the validation data
        class_predictions, attention_params = predict(val_data, val_labels, trained_models)

        # Compute confusion matrix
        predicted_label = np.argmax(class_predictions, axis=1).flatten()
        true_label = val_labels.flatten()

        print("predicted_labels:", predicted_label)
        print("true_labels:", true_label)

        if predicted_label[0] == true_label[0]:
            if predicted_label[0] == 0:
                tn += 1
            else:
                tp += 1
        else:
            if predicted_label[0] == 0:
                fn += 1
            else:
                fp += 1

    # Calculate the final accuracy across all subjects
    final_accuracy = (tp + tn) / (tp + tn + fp + fn)
    final_f1_score = 2 * tp / (2 * tp + fp + fn)
    print(f"Final average accuracy across all subjects: {final_accuracy * 100:.2f}%")
    print(f"Final average F1-score across all subjects: {final_f1_score * 100:.2f}%")

    return final_accuracy


def rkf_evaluate(data, k, n_repeats):
    # Extract the bags and labels
    bags = data['X'].tolist()
    labels = data['y'].tolist()

    # Initialize RepeatedKFold
    rkf = RepeatedKFold(n_splits=k, n_repeats=n_repeats, random_state=42)
    overall_accuracies = []
    overall_f1_scores = []

    for train_index, test_index in rkf.split(bags):
        # Split the data into training and validation sets
        train_bags = [bags[i] for i in train_index]
        train_labels = [labels[i] for i in train_index]
        val_bags = [bags[i] for i in test_index]
        val_label = [labels[i] for i in test_index]

        train_data = np.array(train_bags)
        train_labels = np.array([np.array([label]) for label in train_labels])

        val_data = np.array(val_bags)
        val_labels = np.array([np.array([label]) for label in val_label])

        # # Prepare train data
        # train_data = np.array([np.array(instance) for instance in train_bags])
        # train_data = list(np.transpose(train_data, (1, 0, 2, 3)))
        # train_labels = np.array([np.array([label]) for label in train_labels])
        #
        # # Prepare validation data
        # val_data = np.array([np.array(instance) for instance in val_bag])
        # val_data = list(np.transpose(val_data, (1, 0, 2, 3)))
        # val_labels = np.array([np.array([label]) for label in val_label])

        models = [create_model(input_shape) for _ in range(ENSEMBLE_AVG_COUNT)]

        # Train the models on the training data
        trained_models = [train(train_data, train_labels, val_data, val_labels, model) for model in models]

        # Evaluate the model on the validation data
        class_predictions, attention_params = predict(val_data, val_labels, trained_models)

        # Compute confusion matrix
        predicted_labels = np.argmax(class_predictions, axis=1).flatten()
        true_labels = val_labels.flatten()

        print("predicted_labels:", predicted_labels)
        print("true_labels:", true_labels)

        tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
        print("tn:", tn)
        print("fp:", fp)
        print("fn:", fn)
        print("tp:", tp)

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1_score = (2 * tp) / (2 * tp + fp + fn)

        print("accuracy:", accuracy)
        print("f1_score:", f1_score)

        overall_accuracies.append(accuracy)
        overall_f1_scores.append(f1_score)

    # Calculate the final accuracy across all folds and repetitions
    final_accuracy = np.mean(overall_accuracies)
    final_f1_score = np.mean(overall_f1_scores)
    print(f"Final average accuracy across all subjects: {final_accuracy * 100:.2f}%")
    print(f"Final average F1-score across all subjects: {final_f1_score * 100:.2f}%")

    return final_accuracy


# loso_evaluate(sdataset)
rkf_evaluate(sdataset, k=5, n_repeats=2)

print(time.time() - start)
