import numpy as np
import psutil

from utils import *
from visualization import *
import os
import time
import gc
import sys
import resource

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

os.environ["tf_gpu_allocator"] = "cuda_malloc_async"


def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / (1024 ** 2):.2f} MB")


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


# k.set_floatx("float16")


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
        masked_weights = layers.Add()([mask_layer, instance_weights])

        # Apply softmax over instances such that the output summation is equal to 1.
        alpha = ops.softmax(masked_weights, axis=1)

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


# def embeddings_function(M):
#     return keras.Sequential(
#         [
#             # Layer 1
#             layers.ZeroPadding1D(padding=1),
#             layers.Conv1D(filters=32, kernel_size=8, padding='valid'),
#             layers.LeakyReLU(negative_slope=0.2),
#             layers.MaxPooling1D(pool_size=2),
#
#             # Layer 2
#             layers.ZeroPadding1D(padding=1),
#             layers.Conv1D(filters=32, kernel_size=8, padding='valid'),
#             layers.LeakyReLU(negative_slope=0.2),
#             layers.MaxPooling1D(pool_size=2),
#
#             # Layer 3
#             layers.ZeroPadding1D(padding=1),
#             layers.Conv1D(filters=16, kernel_size=16, padding='valid'),
#             layers.LeakyReLU(negative_slope=0.2),
#             layers.MaxPooling1D(pool_size=2),
#
#             # Layer 4
#             layers.ZeroPadding1D(padding=1),
#             layers.Conv1D(filters=16, kernel_size=16, padding='valid'),
#             layers.LeakyReLU(negative_slope=0.2),
#             layers.MaxPooling1D(pool_size=2),
#
#             # Flatten and Dense layer to get M-dimensional output
#             layers.Flatten(),
#             layers.Dense(M),
#         ],
#         name="embeddings_function",
#     )


def embeddings_function(M):
    return keras.Sequential(
        [
            # Layer 1
            layers.ZeroPadding1D(padding=1),
            layers.Conv1D(filters=32, kernel_size=8, padding='valid'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            layers.MaxPooling1D(pool_size=2),

            # Layer 2
            layers.ZeroPadding1D(padding=1),
            layers.Conv1D(filters=32, kernel_size=8, padding='valid'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            layers.MaxPooling1D(pool_size=2),

            # Layer 3
            layers.ZeroPadding1D(padding=1),
            layers.Conv1D(filters=16, kernel_size=16, padding='valid'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            layers.MaxPooling1D(pool_size=2),

            # Layer 4
            layers.ZeroPadding1D(padding=1),
            layers.Conv1D(filters=16, kernel_size=16, padding='valid'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            layers.MaxPooling1D(pool_size=2),

            # Flatten and Dense layer to get M-dimensional output
            layers.Flatten(),
            layers.Dense(M),
        ],
        name="embeddings_function",
    )


def final_classifier():
    return keras.Sequential(
        [
            # Layer 1: Dense M → 32, Leaky-ReLU (α = 0.2), Dropout p = 0.2
            layers.Dense(32, name="dense_1"),
            layers.LeakyReLU(negative_slope=0.2, name="leaky_relu_1"),
            layers.Dropout(0.2, name="dropout_1"),

            # Layer 2: Dense 32 → 16, Leaky-ReLU (α = 0.2), Dropout p = 0.2
            layers.Dense(16, name="dense_2"),
            layers.LeakyReLU(negative_slope=0.2, name="leaky_relu_2"),
            layers.Dropout(0.2, name="dropout_2"),

            # Layer 3: Dense 16 → 2, 2-way softmax
            layers.Dense(2, activation='softmax', name="output")
        ],
        name="final_classifier",
    )


class MILModel(keras.Model):
    def __init__(self, input_shape, M, weight_params_dim=16, use_gated=False, **kwargs):
        super(MILModel, self).__init__(**kwargs)

        # Store parameters
        self.M = M
        self.Kt, self.Ws, self.C = input_shape
        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated

        # Define encoder optimizer
        self.embeddings_learning_rate = 1e-5
        self.optimizer_embeddings = keras.optimizers.Adam(learning_rate=self.embeddings_learning_rate)

        # Define model components
        self.mask_layer = layers.Lambda(self.create_mask_layer, name="mask_layer")
        self.reshape_to_embeddings = layers.Lambda(lambda x: tf.reshape(x, (-1, self.Ws, self.C)))
        self.embeddings_function = embeddings_function(self.M)
        self.reshape_to_attention = layers.Lambda(lambda x: tf.reshape(x, (-1, self.Kt, self.M)),
                                                  name="reshape_attention")
        self.attention_layer = MILAttentionLayer(
            weight_params_dim=self.weight_params_dim,
            kernel_regularizer=keras.regularizers.L2(0.01),
            use_gated=self.use_gated,
            name="alpha",
        )
        self.weighted_embeddings_layer = layers.Multiply(name="weighted_embeddings")
        self.sum_layer = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), name="sum_layer")
        self.classifier = final_classifier()

        # Finetune
        self.finetune()

    @property
    def optimizer(self):
        return self.optimizer_other

    @optimizer.setter
    def optimizer(self, value):
        self.optimizer_other = value

    def create_mask_layer(self, inputs):
        # Sum the features along the last two dimensions (500, 3)
        summed_features = tf.reduce_sum(inputs, axis=[2, 3], keepdims=True)

        # Squeeze to remove the extra dimension
        summed_features = tf.squeeze(summed_features, axis=-1)

        # Create a mask where summed_features is zero
        mask = tf.where(tf.abs(summed_features) < 1e-3, -np.inf, 0)

        return mask

    def call(self, inputs):
        # Forward pass through the model components
        mask_layer = self.mask_layer(inputs)
        embeddings = self.reshape_to_embeddings(inputs)
        embeddings = self.embeddings_function(embeddings)
        embeddings = self.reshape_to_attention(embeddings)

        # Attention
        alpha = self.attention_layer(embeddings, mask_layer)

        # Weighted embeddings
        weighted_embeddings = self.weighted_embeddings_layer([alpha, embeddings])
        z = self.sum_layer(weighted_embeddings)

        # Classification
        output = self.classifier(z)

        return output

    def finetune(self):
        """Load pre-trained weights for the embeddings function."""
        try:
            self.embeddings_function.build(input_shape=(None, self.Ws, self.C))
            self.embeddings_function.load_weights("embeddings.weights.h5")
            self.embeddings_function.trainable = True  # Freeze encoder
            print("Successfully loaded SimCLR weights into encoder.")
        except Exception as e:
            print(f"Failed to load SimCLR weights: {e}")

    # def train_step(self, data):
    #     # Unpack data
    #     x, y = data
    #
    #     with tf.GradientTape() as tape:
    #         # Forward pass
    #         y_pred = self(x, training=True)
    #         # Compute the loss using model.compute_loss
    #         loss = self.compute_loss(x, y, y_pred)
    #
    #     # Compute gradients
    #     gradients = tape.gradient(loss, self.trainable_variables)
    #     # Apply gradients
    #     self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    #
    #     # Manually update metrics
    #     for metric in self.metrics:
    #         metric.update_state(y, y_pred)
    #
    #     # Return a dictionary mapping metric names to their current value
    #     return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        # Unpack data
        x, y = data

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)
            # Compute the loss
            loss = self.compute_loss(x, y, y_pred)

        # print("all trainable variables: ", self.trainable_variables)
        # print("length of trainable variables: ", len(self.trainable_variables))

        # Separate parameters for different learning rates
        embeddings_vars = self.embeddings_function.trainable_variables
        # print("Type of embeddings_vars: ", type(embeddings_vars))
        # print("Length of embeddings_vars: ", len(embeddings_vars))
        # print("embeddings_vars: ", embeddings_vars)

        other_vars = self.trainable_variables[len(embeddings_vars):]
        # print("Type of other_vars: ", type(other_vars))
        # print("Length of other_vars: ", len(other_vars))
        # print("other_vars: ", other_vars)

        # Compute gradients for both parameter groups
        gradients = tape.gradient(loss, self.trainable_variables)
        # print("Length of gradients: ", len(gradients))
        # print("gradients: ", gradients)
        embeddings_grads = gradients[:len(embeddings_vars)]
        # print("Length of embeddings_grads: ", len(embeddings_grads))
        # print("embeddings_grads: ", embeddings_grads)
        other_grads = gradients[len(embeddings_vars):]
        # print("Length of other_grads: ", len(other_grads))
        # print("other_grads: ", other_grads)

        # Apply gradients with different learning rates
        self.optimizer_embeddings.apply_gradients(zip(embeddings_grads, embeddings_vars))
        self.optimizer_other.apply_gradients(zip(other_grads, other_vars))

        # Manually update metrics
        for metric in self.metrics:
            metric.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


class ClearMemory(callbacks.Callback):

    def on_train_begin(self, logs=None):
        k.clear_session()
        gc.collect()

        print("Memory cleared.")


def lr_schedule(epoch, lr):
    total_epochs = num_epochs
    decay_start_epoch = total_epochs // 2  # Start decay at the halfway point of the training
    if epoch >= decay_start_epoch:
        return lr * 0.90  # Decay the learning rate by a factor of 0.9
    return lr


def train(train_dataset, val_dataset, model):
    # Train model.
    # Prepare callbacks.
    # Path where to save best weights.

    # print(model.get_layer("embeddings_function").summary())

    # Take the file name from the wrapper.
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../best_model.weights.h5")

    # Initialize model checkpoint callback.
    model_checkpoint = callbacks.ModelCheckpoint(
        file_path,
        monitor="val_loss",
        verbose=1,
        mode="min",
        save_best_only=True,
        save_weights_only=True,
    )

    # Initialize early stopping callback.
    # The model performance is monitored across the validation data and stops training
    # when the generalization error cease to decrease.
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=1,
        start_from_epoch=20,
        restore_best_weights=False
    )

    # Callbacks
    clear_memory = ClearMemory()
    lr_scheduler = callbacks.LearningRateScheduler(lr_schedule)

    # Compile model.
    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        auto_scale_loss=True
    )

    # Fit model.
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=[lr_scheduler, clear_memory],
        verbose=1,
    )

    # Load best weights.
    # print("Loading best weights...")
    # model.load_weights(file_path)

    return model


# Adjust the paths to be relative to the current script location
# sdata_path = os.path.join('..', 'data', 'tremor_sdata.pickle')
# tremor_sdata = unpickle_data(sdata_path)

E_thres = 0.15
Kt = 100
num_epochs = 50
batch_size = 1
# {'updrs16', 'updrs20', 'updrs21', 'tremor_manual'}
# sdataset = form_dataset(tremor_sdata, E_thres, Kt, 'tremor_manual', 'tremor_manual')

with open("sdataset.pickle", 'rb') as f:
    print("Loading sdataset...")
    sdataset = pkl.load(f)

print(sdataset)

# Building model(s).
Kt, Ws, C = np.array(sdataset['X'])[0].shape
input_shape = (Kt, Ws, C)
print(input_shape)
M = 64
model = MILModel(input_shape=input_shape, M=M)

# Show single model architecture.
print(model.summary())


def predict(dataset, trained_model):
    # Predict output classes on data.
    predictions = trained_model.predict(dataset)

    loss, accuracy = trained_model.evaluate(dataset, verbose=0)

    print(
        f"The average loss and accuracy are {loss}"
        f" and {100 * accuracy} % resp."
    )

    return predictions


def loso_evaluate(data):
    # Extract the bags and labels
    bags = data['X'].tolist()
    y_train = data['y_train'].tolist()
    y_test = data['y_test'].tolist()

    # Initialize LeaveOneOut
    loo = LeaveOneOut()

    tn, fp, fn, tp = 0, 0, 0, 0

    for train_index, test_index in loo.split(bags):
        # Split the data into training and validation sets
        train_bags = [bags[i] for i in train_index]
        train_labels = [y_train[i] for i in train_index]
        val_bag = [bags[i] for i in test_index]
        val_label = [y_test[i] for i in test_index]

        train_data = np.array(train_bags)
        train_data = normalize_mil(train_data)
        train_labels = np.array([np.array([label]) for label in train_labels])

        val_data = np.array(val_bag)
        val_data = normalize_mil(val_data)
        val_labels = np.array([np.array([label]) for label in val_label])

        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
        train_dataset = train_dataset.shuffle(buffer_size=len(train_data)).batch(batch_size).prefetch(
            buffer_size=tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
        val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        current_model = MILModel(input_shape=input_shape, M=M)

        # Train the models on the training data
        trained_model = train(train_dataset, val_dataset, current_model)

        print_memory_usage()

        # Evaluate the model on the validation data
        class_predictions = predict(val_dataset, trained_model)

        del trained_model

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

    # Calculate metrics
    accuracy, sensitivity, specificity, precision, f1_score = calculate_metrics(tn, fp, fn, tp)

    print(f"Final accuracy across all subjects: {accuracy * 100:.2f}%")
    print(f"Final sensitivity across all subjects: {sensitivity * 100:.2f}%")
    print(f"Final specificity across all subjects: {specificity * 100:.2f}%")
    print(f"Final precision across all subjects: {precision * 100:.2f}%")
    print(f"Final F1-score across all subjects: {f1_score * 100:.2f}%")

    return


def rkf_evaluate(data, k, n_repeats):
    # Extract the bags and labels
    bags = data['X'].tolist()
    y_train = data['y_train'].tolist()
    y_test = data['y_test'].tolist()

    # Initialize RepeatedKFold
    rkf = RepeatedKFold(n_splits=k, n_repeats=n_repeats)
    overall_accuracies = []
    overall_sensitivities = []
    overall_specificities = []
    overall_precisions = []
    overall_f1_scores = []

    all_true_labels = []
    all_predicted_labels = []
    all_predicted_probs = []

    for idx, (train_index, test_index) in enumerate(rkf.split(bags), 1):
        print(f"\033[91mIteration {idx}/{k * n_repeats}\033[0m")

        # Split the data into training and validation sets
        train_bags = [bags[i] for i in train_index]
        train_labels = [y_train[i] for i in train_index]
        val_bags = [bags[i] for i in test_index]
        val_label = [y_test[i] for i in test_index]

        train_data = np.array(train_bags)
        train_data = normalize_mil(train_data)
        train_labels = np.array([np.array([label]) for label in train_labels])

        val_data = np.array(val_bags)
        val_data = normalize_mil(val_data)
        val_labels = np.array([np.array([label]) for label in val_label])

        print_memory_usage()

        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
        train_dataset = train_dataset.shuffle(buffer_size=10 * len(train_data)).batch(batch_size).prefetch(
            buffer_size=tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
        val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        current_model = MILModel(input_shape=input_shape, M=M)

        # Train the models on the training data
        trained_model = train(train_dataset, train_dataset, current_model)

        # Evaluate the model on the validation data
        class_predictions = predict(val_dataset, trained_model)

        del trained_model
        gc.collect()

        # Compute confusion matrix
        predicted_labels = np.argmax(class_predictions, axis=1).flatten()
        true_labels = val_labels.flatten()

        # Store true and predicted labels for plotting later
        all_true_labels.extend(true_labels)
        all_predicted_labels.extend(predicted_labels)
        predicted_probs = class_predictions[:, 1].flatten()
        all_predicted_probs.extend(predicted_probs)

        print("predicted_labels:", predicted_labels)
        print("true_labels:", true_labels)

        tn, fp, fn, tp = safe_confusion_matrix(true_labels, predicted_labels)
        print("tn:", tn)
        print("fp:", fp)
        print("fn:", fn)
        print("tp:", tp)

        # Calculate metrics
        accuracy, sensitivity, specificity, precision, f1_score = calculate_metrics(tn, fp, fn, tp)

        print("accuracy:", accuracy)
        print("sensitivity:", sensitivity)
        print("specificity:", specificity)
        print("precision:", precision)
        print("f1_score:", f1_score)

        overall_accuracies.append(accuracy)
        overall_sensitivities.append(sensitivity)
        overall_specificities.append(specificity)
        overall_precisions.append(precision)
        overall_f1_scores.append(f1_score)

    # Calculate the final accuracy across all folds and repetitions
    final_accuracy = np.nanmean(overall_accuracies)
    final_sensitivity = np.nanmean(overall_sensitivities)
    final_specificity = np.nanmean(overall_specificities)
    final_precision = np.nanmean(overall_precisions)
    final_f1_score = np.nanmean(overall_f1_scores)
    print(f"Final average accuracy across all subjects: {final_accuracy * 100:.2f}%")
    print(f"Final average sensitivity across all subjects: {final_sensitivity * 100:.2f}%")
    print(f"Final average specificity across all subjects: {final_specificity * 100:.2f}%")
    print(f"Final average precision across all subjects: {final_precision * 100:.2f}%")
    print(f"Final average F1-score across all subjects: {final_f1_score * 100:.2f}%")

    # Convert lists to arrays for plotting
    all_true_labels = np.array(all_true_labels)
    all_predicted_probs = np.array(all_predicted_probs)
    all_predicted_labels = np.array(all_predicted_labels)
    valid_indices = ~np.isnan(all_true_labels) & ~np.isnan(all_predicted_probs)
    all_true_labels = all_true_labels[valid_indices]
    all_predicted_probs = all_predicted_probs[valid_indices]
    all_predicted_labels = all_predicted_labels[valid_indices]

    results = {
        'final_accuracy': final_accuracy,
        'final_sensitivity': final_sensitivity,
        'final_specificity': final_specificity,
        'final_precision': final_precision,
        'final_f1_score': final_f1_score
    }

    return all_true_labels, all_predicted_probs, all_predicted_labels, results


def rkf_evaluate_with_validation(data, k, n_repeats):
    # Extract the bags and labels
    bags = data['X'].tolist()
    y_train = data['y_train'].tolist()

    # Initialize RepeatedKFold
    rkf = RepeatedKFold(n_splits=k, n_repeats=n_repeats, random_state=42)
    overall_accuracies = []
    overall_sensitivities = []
    overall_specificities = []
    overall_precisions = []
    overall_f1_scores = []

    all_true_labels = []
    all_predicted_labels = []
    all_predicted_probs = []

    for idx, (train_val_index, test_index) in enumerate(rkf.split(bags), 1):
        print(f"\033[91mIteration {idx}/{k * n_repeats}\033[0m")

        # Split the data into train+val and test sets
        train_val_bags = [bags[i] for i in train_val_index]
        train_val_labels = [y_train[i] for i in train_val_index]
        test_bags = [bags[i] for i in test_index]
        test_labels = [y_train[i] for i in test_index]

        # Further split train_val into train and val
        rkf_inner = RepeatedKFold(n_splits=k - 1, n_repeats=1, random_state=42)
        train_index, val_index = next(rkf_inner.split(train_val_bags))

        # Prepare the training, validation, and testing datasets
        train_bags = [train_val_bags[i] for i in train_index]
        train_labels = [train_val_labels[i] for i in train_index]
        val_bags = [train_val_bags[i] for i in val_index]
        val_labels = [train_val_labels[i] for i in val_index]

        # Convert all sets to numpy arrays
        train_data = np.array(train_bags)
        train_data = normalize_mil(train_data)
        train_labels = np.array([np.array([label]) for label in train_labels])

        val_data = np.array(val_bags)
        val_data = normalize_mil(val_data)
        val_labels = np.array([np.array([label]) for label in val_labels])

        test_data = np.array(test_bags)
        test_labels = np.array([np.array([label]) for label in test_labels])

        print_memory_usage()

        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
        train_dataset = train_dataset.shuffle(buffer_size=10 * len(train_data), seed=42).batch(batch_size).prefetch(
            buffer_size=tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
        val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
        test_dataset = test_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

        # Create the model
        current_model = MILModel(input_shape=input_shape, M=M)

        # Train the model on the training and validation data
        trained_model = train(train_dataset, val_dataset, current_model)

        # Evaluate the model on the test data
        class_predictions = predict(test_dataset, trained_model)

        del trained_model
        gc.collect()

        # Compute confusion matrix
        predicted_labels = np.argmax(class_predictions, axis=1).flatten()
        true_labels = test_labels.flatten()

        # Store true and predicted labels for plotting later
        all_true_labels.extend(true_labels)
        all_predicted_labels.extend(predicted_labels)
        predicted_probs = class_predictions[:, 1].flatten()
        all_predicted_probs.extend(predicted_probs)

        print("predicted_labels:", predicted_labels)
        print("true_labels:", true_labels)
        print("validation_labels:", val_labels.flatten())

        tn, fp, fn, tp = safe_confusion_matrix(true_labels, predicted_labels)
        print("tn:", tn)
        print("fp:", fp)
        print("fn:", fn)
        print("tp:", tp)

        # Calculate metrics
        accuracy, sensitivity, specificity, precision, f1_score = calculate_metrics(tn, fp, fn, tp)

        print("accuracy:", accuracy)
        print("sensitivity:", sensitivity)
        print("specificity:", specificity)
        print("precision:", precision)
        print("f1_score:", f1_score)

        overall_accuracies.append(accuracy)
        overall_sensitivities.append(sensitivity)
        overall_specificities.append(specificity)
        overall_precisions.append(precision)
        overall_f1_scores.append(f1_score)

    # Calculate the final accuracy across all folds and repetitions
    final_accuracy = np.nanmean(overall_accuracies)
    final_sensitivity = np.nanmean(overall_sensitivities)
    final_specificity = np.nanmean(overall_specificities)
    final_precision = np.nanmean(overall_precisions)
    final_f1_score = np.nanmean(overall_f1_scores)
    print(f"Final average accuracy across all subjects: {final_accuracy * 100:.2f}%")
    print(f"Final average sensitivity across all subjects: {final_sensitivity * 100:.2f}%")
    print(f"Final average specificity across all subjects: {final_specificity * 100:.2f}%")
    print(f"Final average precision across all subjects: {final_precision * 100:.2f}%")
    print(f"Final average F1-score across all subjects: {final_f1_score * 100:.2f}%")

    # Convert lists to arrays for plotting
    all_true_labels = np.array(all_true_labels)
    all_predicted_probs = np.array(all_predicted_probs)
    all_predicted_labels = np.array(all_predicted_labels)
    valid_indices = ~np.isnan(all_true_labels) & ~np.isnan(all_predicted_probs)
    all_true_labels = all_true_labels[valid_indices]
    all_predicted_probs = all_predicted_probs[valid_indices]
    all_predicted_labels = all_predicted_labels[valid_indices]

    return all_true_labels, all_predicted_probs, all_predicted_labels


if __name__ == "__main__":
    # loso_evaluate(sdataset)
    true_labels, predicted_probs, predicted_labels, results = rkf_evaluate(sdataset, k=5, n_repeats=4)
    # true_labels, predicted_probs, predicted_labels = rkf_evaluate_with_validation(sdataset, k=5, n_repeats=4)

    # np.savez("roc_curve_pretraining.npz", true_labels=true_labels, predicted_probs=predicted_probs)
    # np.savez("roc_curve_no_pretraining.npz", true_labels=true_labels, predicted_probs=predicted_probs)

    plot_roc_curve(true_labels, predicted_probs)
    plot_confusion_matrix(true_labels, predicted_labels)

    print(time.time() - start)

    # Alarm
    os.system('play -nq -t alsa synth {} sine {}'.format(1, 999))
