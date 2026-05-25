import numpy as np
import psutil

from utils import *
from visualization import *
import os
import time
import gc
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import keras

from keras import layers
from keras import ops
from keras import callbacks
from keras import optimizers
from tf_keras import backend as k
from tf_keras import mixed_precision
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
import argparse

# Default parameters - can be overridden when importing
DEFAULT_K2 = 500
DEFAULT_PRETRAIN_NUM_EPOCHS = 100
DEFAULT_NUM_EPOCHS = 50
DEFAULT_BATCH_SIZE = 4
DEFAULT_M = 64


def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / (1024 ** 2):.2f} MB")


def setup_environment():
    """Setup environment configurations for training"""
    start = time.time()

    plt.ion()

    np.set_printoptions(threshold=sys.maxsize)

    os.environ["tf_gpu_allocator"] = "cuda_malloc_async"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # run on CPU
    os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

    # Configure GPU memory growth to prevent over-allocation
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Using GPU with memory growth enabled...")
        except RuntimeError as e:
            print(f"GPU memory growth setup failed: {e}")
    else:
        print("Using CPU...")

    # Mixed precision policy
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)
    print("Using mixed precision...")

    return start


# k.set_floatx("float16")


# === MODE SELECTION ===
# Default MODE - can be overridden when importing
MODE = "baseline"  # "baseline", "simclr", "subject_simclr"


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
        input_dim = input_shape[-1]

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


class MILModel(keras.Model):
    def __init__(
        self,
        input_shape,
        M,
        weight_params_dim=16,
        use_gated=False,
        mode=None,
        use_bimodal=False,
        **kwargs,
    ):
        super(MILModel, self).__init__(**kwargs)

        # Store parameters
        self.M = M
        self.K2, self.B = input_shape
        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated
        self.mode = (
            mode if mode is not None else MODE
        )  # Use parameter or fall back to global
        self.use_bimodal = use_bimodal

        # Mode-specific batchnorm
        self.embeddings_learning_rate = 5e-4 if self.mode != "baseline" else 1e-3
        self.optimizer_embeddings = keras.optimizers.Adam(
            learning_rate=self.embeddings_learning_rate
        )

        # Define model components
        self.mask_layer = layers.Lambda(self.create_mask_layer, name="mask_layer")
        self.embeddings_network = self.embeddings_function(self.M)
        self.reshape_to_attention = layers.Lambda(
            lambda x: tf.reshape(x, (-1, self.K2, self.M)), name="reshape_attention"
        )
        self.attention_layer = MILAttentionLayer(
            weight_params_dim=self.weight_params_dim,
            kernel_regularizer=keras.regularizers.L2(0.01),
            use_gated=self.use_gated,
            name="alpha",
        )
        self.weighted_embeddings_layer = layers.Multiply(name="weighted_embeddings")
        self.sum_layer = layers.Lambda(
            lambda x: tf.reduce_sum(x, axis=1), name="sum_layer"
        )
        self.classifier = self.final_classifier()

        # Finetune only for simclr/subject_simclr
        if self.mode in ["simclr", "subject_simclr"]:
            self.finetune()
        else:
            print("Finetune skipped for baseline mode.")

    def embeddings_function(self, M):
        # input_dim is 502 for typing data
        return keras.Sequential(
            [
                # Layer 1:
                keras.Input(shape=(self.K2, self.B)),  # Use class attributes
                layers.Dense(100),
                layers.LeakyReLU(negative_slope=0.2),
                layers.Dropout(0.1),
                # Layer 2:
                layers.Dense(50),
                layers.LeakyReLU(negative_slope=0.2),
                layers.Dropout(0.1),
                layers.Dense(M),
            ],
            name="embeddings_function",
        )

    def final_classifier(self):
        return keras.Sequential(
            [
                # Layer 1:
                layers.Dense(30, name="dense_1"),
                layers.LeakyReLU(negative_slope=0.2, name="leaky_relu_1"),
                # Layer 2:
                layers.Dense(10, name="dense_2"),
                layers.LeakyReLU(negative_slope=0.2, name="leaky_relu_2"),
                # Layer 3
                layers.Dense(2, activation="softmax", name="output"),
            ],
            name="final_classifier",
        )

    @property
    def optimizer(self):
        return self.optimizer_other

    @optimizer.setter
    def optimizer(self, value):
        self.optimizer_other = value

    def create_mask_layer(self, inputs):
        summed_features = tf.reduce_sum(inputs, axis=2, keepdims=True)

        # Create a mask where summed_features is zero
        mask = tf.where(tf.abs(summed_features) < 1e-3, -np.inf, 0)

        return mask

    def call(self, inputs):
        # Forward pass through the model components
        mask_layer = self.mask_layer(inputs)
        embeddings = self.embeddings_network(inputs)
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
        """Load pre-trained weights for the embeddings function (and attention for subject_simclr)."""
        if self.mode == "simclr":
            # Use bimodal weights if use_bimodal flag is set, otherwise use standard SimCLR weights
            if self.use_bimodal:
                weights_file = "weights/fusion/typing_bimodal_embeddings.weights.h5"
            else:
                weights_file = "weights/typing/typing_simclr_embeddings.weights.h5"
        elif self.mode == "subject_simclr":
            weights_file = "weights/typing/typing_subject_simclr_embeddings.weights.h5"
        else:
            return  # No finetune for baseline
        try:
            self.embeddings_network.build(input_shape=(None, self.K2, self.B))
            self.embeddings_network.load_weights(weights_file)
            self.embeddings_network.trainable = False  # Freeze encoder
            print(f"Successfully loaded weights from '{weights_file}' into encoder.")
        except Exception as e:
            print(f"Failed to load encoder weights: {e}")

        if self.mode == "subject_simclr":
            # Also load attention weights (warm-start; kept trainable for fine-tuning)
            attention_file = "weights/typing/typing_subject_simclr_attention.weights.pkl"
            try:
                dummy_embs = tf.zeros((1, self.K2, self.M))
                dummy_mask = tf.zeros((1, self.K2, 1))
                self.attention_layer(dummy_embs, dummy_mask)  # Build before loading
                import pickle as _pkl
                with open(attention_file, "rb") as _f:
                    self.attention_layer.set_weights(_pkl.load(_f))
                print(f"Successfully loaded attention weights from '{attention_file}'.")
            except Exception as e:
                print(f"Failed to load attention weights: {e}")

    # Baseline train step
    def train_step_baseline(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer_other.apply_gradients(zip(gradients, self.trainable_variables))
        for metric in self.metrics:
            metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    # SimCLR train step (with encoder freezing/unfreezing)
    def freeze_encoder(self):
        """Freeze the encoder by setting trainable=False."""
        self.embeddings_network.trainable = False

    def unfreeze_encoder(self):
        """Unfreeze the encoder by setting trainable=True."""
        self.embeddings_network.trainable = True

    def train_step(self, data):
        # Unpack data
        x, y = data
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)
            # Compute the loss
            loss = self.compute_loss(x, y, y_pred)

        # Separate parameters for different learning rates
        embeddings_vars = self.embeddings_network.trainable_variables
        other_vars = self.trainable_variables[len(embeddings_vars) :]
        gradients = tape.gradient(loss, self.trainable_variables)
        embeddings_grads = gradients[: len(embeddings_vars)]
        other_grads = gradients[len(embeddings_vars) :]
        if embeddings_vars:
            self.optimizer_embeddings.apply_gradients(
                zip(embeddings_grads, embeddings_vars)
            )
        self.optimizer_other.apply_gradients(zip(other_grads, other_vars))
        for metric in self.metrics:
            metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


class ClearMemory(callbacks.Callback):

    def on_train_begin(self, logs=None):
        k.clear_session()
        gc.collect()

        print("Memory cleared.")


def lr_schedule(epoch, lr, total_epochs=100):
    decay_start_epoch = (
        total_epochs // 2
    )  # Start decay at the halfway point of the training
    if epoch >= decay_start_epoch:
        return lr * 1.0  # decay
    return lr


def train(
    train_dataset,
    val_dataset,
    model,
    num_epochs=DEFAULT_PRETRAIN_NUM_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    mode=None,
):
    # Train model.
    # Prepare callbacks.
    # Path where to save best weights.

    # Use model's mode if available, otherwise fall back to global MODE
    current_mode = mode if mode is not None else (getattr(model, "mode", MODE))

    # Callbacks
    clear_memory = ClearMemory()
    lr_scheduler = callbacks.LearningRateScheduler(
        lambda epoch, lr: lr_schedule(epoch, lr, num_epochs)
    )

    # Mode-specific training logic
    if current_mode == "baseline":
        # No encoder freezing, no fine-tuning, single phase
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            auto_scale_loss=True,
            run_eagerly=False,
        )
        # Patch the baseline train_step
        model.train_step = model.train_step_baseline
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=DEFAULT_PRETRAIN_NUM_EPOCHS,
            batch_size=batch_size,
            callbacks=[lr_scheduler, clear_memory],
            verbose=1,
        )
        return model

    # simclr/subject_simclr: freeze encoder, train, then unfreeze and fine-tune
    model.freeze_encoder()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        auto_scale_loss=True,
        run_eagerly=False,
    )
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=DEFAULT_PRETRAIN_NUM_EPOCHS,
        batch_size=batch_size,
        callbacks=[lr_scheduler, clear_memory],
        verbose=1,
    )

    print("Finetuning model...")
    model.unfreeze_encoder()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        auto_scale_loss=True,
        run_eagerly=False,
    )
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=DEFAULT_NUM_EPOCHS,
        batch_size=batch_size,
        callbacks=[lr_scheduler, clear_memory],
        verbose=1,
    )
    return model


def predict(dataset, trained_model):
    # Predict output classes on data.
    predictions = trained_model.predict(dataset)

    loss, accuracy = trained_model.evaluate(dataset, verbose=0)

    print(f"The average loss and accuracy are {loss}" f" and {100 * accuracy} % resp.")

    return predictions


def loso_evaluate(
    data,
    input_shape=None,
    M=DEFAULT_M,
    batch_size=DEFAULT_BATCH_SIZE,
    additional_data=None,
    eval_subject_ids=None,
):
    # Extract the bags and labels from primary dataset
    bags = data["X"].tolist()
    y = data["y"].tolist()
    subject_ids = data["subject_id"].tolist()

    # Set default input shape if not provided
    if input_shape is None:
        K2, B = np.array(data["X"])[0].shape
        input_shape = (K2, B)

    # Filter to only use common subjects if specified
    if eval_subject_ids is not None:
        eval_indices = [
            i for i, sid in enumerate(subject_ids) if sid in eval_subject_ids
        ]
        bags = [bags[i] for i in eval_indices]
        y = [y[i] for i in eval_indices]
        print(f"Evaluating on {len(bags)} common subjects (tremor+typing)")

    # Extract additional data if provided (for training augmentation)
    additional_bags = []
    additional_labels = []
    if additional_data is not None:
        additional_bags = additional_data["X"].tolist()
        additional_labels = additional_data["y"].tolist()
        print(f"Using additional {len(additional_bags)} subjects for training")

    # Initialize LeaveOneOut
    loo = LeaveOneOut()

    tn, fp, fn, tp = 0, 0, 0, 0
    all_true_labels = []
    all_predicted_labels = []
    all_predicted_probs = []

    n_subjects = len(bags)
    for idx, (train_index, test_index) in enumerate(loo.split(bags), 1):
        print(f"\033[91mIteration {idx}/{n_subjects}\033[0m")
        # Split the data into training and validation sets
        train_bags = [bags[i] for i in train_index]
        train_labels = [y[i] for i in train_index]
        val_bag = [bags[i] for i in test_index]
        val_label = [y[i] for i in test_index]

        # Add all additional subjects to training set
        if additional_data is not None:
            train_bags.extend(additional_bags)
            train_labels.extend(additional_labels)

        train_data = np.array(train_bags)
        train_labels = np.array([np.array([label]) for label in train_labels])

        val_data = np.array(val_bag)
        val_labels = np.array([np.array([label]) for label in val_label])

        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
        train_dataset = (
            train_dataset.shuffle(buffer_size=len(train_data))
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
        val_dataset = val_dataset.batch(batch_size).prefetch(
            buffer_size=tf.data.AUTOTUNE
        )

        current_model = MILModel(input_shape=input_shape, M=M, use_gated=True)

        # Train the models on the training data
        trained_model = train(train_dataset, train_dataset, current_model)

        print_memory_usage()

        # Evaluate the model on the validation data
        class_predictions = predict(val_dataset, trained_model)

        del trained_model

        # Compute confusion matrix
        predicted_label = np.argmax(class_predictions, axis=1).flatten()
        true_label = val_labels.flatten()

        # Store for plotting
        all_true_labels.extend(true_label)
        all_predicted_labels.extend(predicted_label)
        predicted_probs = class_predictions[:, 1].flatten()
        all_predicted_probs.extend(predicted_probs)

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
    accuracy, sensitivity, specificity, precision, f1_score = calculate_metrics(
        tn, fp, fn, tp
    )

    print(f"Final accuracy across all subjects: {accuracy * 100:.2f}%")
    print(f"Final sensitivity across all subjects: {sensitivity * 100:.2f}%")
    print(f"Final specificity across all subjects: {specificity * 100:.2f}%")
    print(f"Final precision across all subjects: {precision * 100:.2f}%")
    print(f"Final F1-score across all subjects: {f1_score * 100:.2f}%")

    # Convert lists to arrays for plotting
    all_true_labels = np.array(all_true_labels)
    all_predicted_probs = np.array(all_predicted_probs)
    all_predicted_labels = np.array(all_predicted_labels)
    valid_indices = ~np.isnan(all_true_labels) & ~np.isnan(all_predicted_probs)
    all_true_labels = all_true_labels[valid_indices]
    all_predicted_probs = all_predicted_probs[valid_indices]
    all_predicted_labels = all_predicted_labels[valid_indices]

    results = {
        "final_accuracy": accuracy,
        "final_sensitivity": sensitivity,
        "final_specificity": specificity,
        "final_precision": precision,
        "final_f1_score": f1_score,
    }

    return all_true_labels, all_predicted_probs, all_predicted_labels, results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Typing MIL single LOSO evaluation")
    parser.add_argument(
        "--model",
        choices=["baseline", "simclr", "subject_simclr"],
        default="baseline",
        help="Model mode: 'baseline', 'simclr', or 'subject_simclr' (subject-level SimCLR)",
    )
    args = parser.parse_args()
    MODE = args.model

    # Setup environment and start timer
    start = setup_environment()

    # Parameters
    batch_size = DEFAULT_BATCH_SIZE
    M = DEFAULT_M

    with open("datasets/typing_sdataset.pickle", "rb") as f:
        print("Loading sdataset...")
        sdataset = pkl.load(f)

    print(sdataset)

    # Building model(s).
    K2, B = np.array(sdataset["X"])[0].shape
    input_shape = (K2, B)
    print("input shape:", input_shape)

    # Run single LOSO evaluation
    true_labels, predicted_probs, predicted_labels, results = loso_evaluate(
        sdataset, input_shape=input_shape, M=M, batch_size=batch_size
    )

    plot_roc_curve(true_labels, predicted_probs)
    plot_confusion_matrix(true_labels, predicted_labels)

    print(time.time() - start)

    input("Press Enter to exit...")
