import numpy as np
import psutil
import os
import time
import gc
import sys
import pickle as pkl
import subprocess

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
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# Import utility functions
from utils import *
from visualization import *

# Import MIL components from both tremor and typing files
# We'll create renamed classes to avoid conflicts and clarify their purpose
from tremorSimCLRattentionMIL import MILAttentionLayer as TremorMILAttentionLayer
from tremorSimCLRattentionMIL import MILModel as TremorMILModel
from tremorSimCLRattentionMIL import train as tremor_train
from tremorSimCLRattentionMIL import ClearMemory, lr_schedule

from typingSimCLRattentionMIL import MILAttentionLayer as TypingMILAttentionLayer
from typingSimCLRattentionMIL import MILModel as TypingMILModel
from typingSimCLRattentionMIL import train as typing_train

start = time.time()

plt.ion()

np.set_printoptions(threshold=sys.maxsize)

os.environ["tf_gpu_allocator"] = "cuda_malloc_async"


def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / (1024 ** 2):.2f} MB")


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # run on CPU

os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

if tf.config.list_physical_devices("GPU"):
    print("Using GPU...")
else:
    print("Using CPU...")

# Mixed precision policy
policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)
print("Using mixed precision...")


# === FUSION MODEL PARAMETERS ===
MODE = "baseline"
assert MODE in ["baseline", "simclr", "federated"], f"Invalid MODE: {MODE}"
print(f"Using MODE: {MODE}")
print(
    f"SimCLR weight loading: {'ENABLED' if MODE in ['simclr', 'federated'] else 'DISABLED'}"
)

# Tremor parameters
TREMOR_E_THRES = 0.15 * 2
TREMOR_KT = 200
TREMOR_NUM_EPOCHS = 50
TREMOR_BATCH_SIZE = 8  # Paper specifies batch size of 8
TREMOR_M = 64

# Typing parameters
TYPING_K2 = 500
TYPING_NUM_EPOCHS = 100
TYPING_BATCH_SIZE = 8  # Paper specifies batch size of 8
TYPING_M = 64

# Fusion parameters
FUSION_NUM_EPOCHS = 100
FUSION_BATCH_SIZE = 8  # Paper specifies batch size of 8


def load_tremor_dataset():
    """Load tremor dataset from pickle file"""
    print("Loading tremor dataset...")
    with open("sdataset.pickle", "rb") as f:
        tremor_dataset = pkl.load(f)
    # print(f"Tremor dataset loaded: {tremor_dataset}")
    return tremor_dataset


def load_typing_dataset():
    """Load typing dataset from pickle file"""
    print("Loading typing dataset...")
    with open("typing_sdataset.pickle", "rb") as f:
        typing_dataset = pkl.load(f)
    # print(f"Typing dataset loaded: {typing_dataset}")
    return typing_dataset


def pretrain_tremor_branch(subject_exclude_id=None):
    """
    Pretrain the tremor branch using all available tremor data.
    Returns the trained model for later weight extraction.
    """
    print("\n" + "=" * 50)
    print("PRETRAINING TREMOR BRANCH")
    print("=" * 50)

    # Load tremor dataset
    tremor_dataset = load_tremor_dataset()

    # Exclude subject if specified
    if subject_exclude_id is not None:
        tremor_dataset = tremor_dataset[
            tremor_dataset["subject_id"] != subject_exclude_id
        ]
        print(
            f"Excluded subject {subject_exclude_id}. Remaining subjects: {len(tremor_dataset)}"
        )

    # Get input shape and create model
    Kt, Ws, C = np.array(tremor_dataset["X"])[0].shape
    tremor_input_shape = (Kt, Ws, C)
    print(f"Tremor input shape: {tremor_input_shape}")

    # Create tremor model
    tremor_model = TremorMILModel(input_shape=tremor_input_shape, M=TREMOR_M, mode=MODE)

    # Prepare all data for training (using all subjects for pretraining)
    all_bags = tremor_dataset["X"].tolist()
    all_labels = tremor_dataset["y_train"].tolist()

    # Convert to numpy arrays and normalize
    train_data = np.array(all_bags)
    train_data = normalize_mil(train_data)
    train_labels = np.array([np.array([label]) for label in all_labels])

    print(f"Tremor training data shape: {train_data.shape}")
    print(f"Tremor training labels shape: {train_labels.shape}")

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    train_dataset = (
        train_dataset.shuffle(buffer_size=len(train_data))
        .batch(TREMOR_BATCH_SIZE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Set global variables for tremor training
    global num_epochs, batch_size
    num_epochs = TREMOR_NUM_EPOCHS
    batch_size = TREMOR_BATCH_SIZE

    # Train the tremor model
    print("Starting tremor branch training...")
    trained_tremor_model = tremor_train(
        train_dataset,
        train_dataset,
        tremor_model,
        num_epochs=num_epochs,
        batch_size=batch_size,
        mode=MODE,
    )

    # Save tremor branch weights
    print("Saving tremor branch weights...")
    trained_tremor_model.embeddings_network.save_weights("tremor_embeddings.weights.h5")

    # Save attention layer weights separately if needed
    attention_weights = trained_tremor_model.attention_layer.get_weights()
    with open("tremor_attention.weights.pkl", "wb") as f:
        pkl.dump(attention_weights, f)

    print("Tremor branch pretraining completed!")
    print_memory_usage()

    return trained_tremor_model


def pretrain_typing_branch(subject_exclude_id=None):
    """
    Pretrain the typing branch using all available typing data.
    Returns the trained model for later weight extraction.
    """
    print("\n" + "=" * 50)
    print("PRETRAINING TYPING BRANCH")
    print("=" * 50)

    # Load typing dataset
    typing_dataset = load_typing_dataset()

    # Exclude subject if specified
    if subject_exclude_id is not None:
        typing_dataset = typing_dataset[
            typing_dataset["subject_id"] != subject_exclude_id
        ]
        print(
            f"Excluded subject {subject_exclude_id}. Remaining subjects: {len(typing_dataset)}"
        )

    # Get input shape and create model
    K2, B = np.array(typing_dataset["X"])[0].shape
    typing_input_shape = (K2, B)
    print(f"Typing input shape: {typing_input_shape}")

    # Create typing model
    typing_model = TypingMILModel(input_shape=typing_input_shape, M=TYPING_M, mode=MODE)

    # Prepare all data for training (using all subjects for pretraining)
    all_bags = typing_dataset["X"].tolist()
    all_labels = typing_dataset["y"].tolist()  # Using labels for binary classification

    # Convert to numpy arrays (no normalization needed for typing data based on original code)
    train_data = np.array(all_bags)
    train_labels = np.array([np.array([label]) for label in all_labels])

    print(f"Typing training data shape: {train_data.shape}")
    print(f"Typing training labels shape: {train_labels.shape}")

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    train_dataset = (
        train_dataset.shuffle(buffer_size=len(train_data))
        .batch(TYPING_BATCH_SIZE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Set global variables for typing training
    global num_epochs, batch_size
    num_epochs = TYPING_NUM_EPOCHS
    batch_size = TYPING_BATCH_SIZE

    # Train the typing model
    print("Starting typing branch training...")
    trained_typing_model = typing_train(
        train_dataset,
        train_dataset,
        typing_model,
        num_epochs=num_epochs,
        batch_size=batch_size,
        mode=MODE,
    )

    # Save typing branch weights
    print("Saving typing branch weights...")
    trained_typing_model.embeddings_network.save_weights("typing_embeddings.weights.h5")

    # Save attention layer weights separately if needed
    attention_weights = trained_typing_model.attention_layer.get_weights()
    with open("typing_attention.weights.pkl", "wb") as f:
        pkl.dump(attention_weights, f)

    print("Typing branch pretraining completed!")
    print_memory_usage()

    return trained_typing_model


def run_pretraining(subject_exclude_id=None):
    """
    Run the complete pretraining process for both branches
    """
    print("Starting fusion model pretraining process...")
    if subject_exclude_id is not None:
        print(f"Excluding subject {subject_exclude_id} from pretraining")
    print(
        f"Tremor parameters: Kt={TREMOR_KT}, M={TREMOR_M}, epochs={TREMOR_NUM_EPOCHS}"
    )
    print(
        f"Typing parameters: K2={TYPING_K2}, M={TYPING_M}, epochs={TYPING_NUM_EPOCHS}"
    )

    # Step 1: Pretrain tremor branch
    tremor_model = pretrain_tremor_branch(subject_exclude_id)

    # Clear memory
    del tremor_model
    gc.collect()
    k.clear_session()

    # Step 2: Pretrain typing branch
    typing_model = pretrain_typing_branch(subject_exclude_id)

    # Clear memory
    del typing_model
    gc.collect()
    k.clear_session()

    print("\n" + "=" * 50)
    print("PRETRAINING COMPLETED")
    print("=" * 50)


def load_raw_data():
    """Load raw tremor and typing subject dictionaries"""
    tremor_path = os.path.join("..", "data", "imu_sdata.pickle")
    typing_path = os.path.join("..", "data", "typing_sdata.pickle")

    with open(tremor_path, "rb") as f:
        tremor_data = pkl.load(f)
    with open(typing_path, "rb") as f:
        typing_data = pkl.load(f)

    return tremor_data, typing_data


def create_endtask_dataset():
    """Create endtask dataset using form_fusion_dataset from utils.

    This creates the fusion dataset with common subjects from both modalities:
    - X1: tremor bags (shape: K1, 1000, 3)
    - X2: typing bags (shape: K2, 502)
    - y: multi-label array [Tremor, FMI, PD] where PD = Tremor AND FMI

    Returns: pandas DataFrame with columns ["X1", "X2", "y"]
    """

    # Option 1: Load pre-computed dataset from pickle
    print("Loading fusion dataset from pickle...")
    with open("fusion_dataset.pickle", "rb") as f:
        fusion_df = pkl.load(f)
    print(f"Loaded dataset: {len(fusion_df)} subjects")
    y_array = np.array(fusion_df["y"].tolist(), dtype=int)
    for i, label_name in enumerate(["Tremor", "FMI", "PD"]):
        print(f"  {label_name} positives: {np.sum(y_array[:, i])}/{len(y_array)}")
    print("X1 shape:", np.array(fusion_df["X1"].tolist()).shape)
    print("X2 shape:", np.array(fusion_df["X2"].tolist()).shape)
    print("y shape:", y_array.shape)
    return fusion_df

    # # Option 2: Create dataset from raw data (current approach)
    # tremor_data, typing_data = load_raw_data()

    # fusion_df = form_fusion_dataset(
    #     tremor_data=tremor_data,
    #     typing_sdata=typing_data,
    #     E_thres=TREMOR_E_THRES,
    #     K1=TREMOR_KT,
    #     K2=TYPING_K2,
    #     tremor_label_str="tremor_manual",  # Use tremor_manual as per paper methodology
    # )

    print(f"Final dataset: {len(fusion_df)} subjects")
    y_array = np.array(fusion_df["y"].tolist(), dtype=int)
    for i, label_name in enumerate(["Tremor", "FMI", "PD"]):
        print(f"  {label_name} positives: {np.sum(y_array[:, i])}/{len(y_array)}")

    return fusion_df


class FusionModel(keras.Model):
    """Fusion model combining pretrained tremor & typing MIL branches.

    Returns only multi-label prediction in forward pass (shape (batch,3)).
    Use get_bag_embeddings() if fused embedding is required externally.
    """

    def __init__(
        self, tremor_input_shape, typing_input_shape, M=64, mode="baseline", **kwargs
    ):
        super(FusionModel, self).__init__(**kwargs)

        self.M = M
        self.tremor_input_shape = tremor_input_shape
        self.typing_input_shape = typing_input_shape

        # Create tremor and typing branches
        self.tremor_branch = TremorMILModel(
            input_shape=tremor_input_shape, M=M, mode=mode
        )
        self.typing_branch = TypingMILModel(
            input_shape=typing_input_shape, M=M, mode=mode
        )

        # Multi-label classifier (define BEFORE loading weights/freeze so attribute exists)
        self.multilabel_classifier = keras.Sequential(
            [
                layers.Dense(32, name="fusion_dense_1"),
                layers.LeakyReLU(negative_slope=0.2, name="fusion_leaky_relu_1"),
                layers.Dropout(0.2, name="fusion_dropout_1"),
                layers.Dense(16, name="fusion_dense_2"),
                layers.LeakyReLU(negative_slope=0.2, name="fusion_leaky_relu_2"),
                layers.Dropout(0.2, name="fusion_dropout_2"),
                layers.Dense(
                    3, activation="sigmoid", name="fusion_output"
                ),  # 3 outputs: Tremor, FMI, PD
            ],
            name="multilabel_classifier",
        )

        # Build multilabel classifier (expects fused embedding shape (None, M))
        if not self.multilabel_classifier.built:
            self.multilabel_classifier.build((None, self.M))
        try:
            print("Tremor embeddings network summary:")
            self.tremor_branch.embeddings_network.summary(
                print_fn=lambda x: print("  " + x)
            )
            print("Typing embeddings network summary:")
            self.typing_branch.embeddings_network.summary(
                print_fn=lambda x: print("  " + x)
            )
        except Exception:
            pass

        # Load pretrained weights and freeze
        self.load_pretrained_weights()
        self.freeze_pretrained_layers()

    def load_pretrained_weights(self):
        """Load pretrained weights for both branches"""
        try:
            # Ensure tremor embeddings network is built
            if not self.tremor_branch.embeddings_network.built:
                self.tremor_branch.embeddings_network.build(
                    (None, self.tremor_branch.Ws, self.tremor_branch.C)
                )
            self.tremor_branch.embeddings_network.load_weights(
                "tremor_embeddings.weights.h5"
            )
            print(f"Loaded tremor embeddings weights from tremor_embeddings.weights.h5")

            # Ensure typing embeddings network is built
            if not self.typing_branch.embeddings_network.built:
                # Typing embeddings sequential already has an Input layer but guard anyway
                self.typing_branch.embeddings_network.build(
                    (None, self.typing_branch.K2, self.typing_branch.B)
                )
            self.typing_branch.embeddings_network.load_weights(
                "typing_embeddings.weights.h5"
            )
            print(f"Loaded typing embeddings weights from typing_embeddings.weights.h5")

            # Load attention weights if needed
            # Build attention layers before setting weights
            if not self.tremor_branch.attention_layer.built:
                self.tremor_branch.attention_layer.build(
                    (None, self.tremor_branch.Kt, self.tremor_branch.M)
                )
            with open("tremor_attention.weights.pkl", "rb") as f:
                tremor_attention_weights = pkl.load(f)
                self.tremor_branch.attention_layer.set_weights(tremor_attention_weights)
            print("Loaded tremor attention weights from tremor_attention.weights.pkl")

            if not self.typing_branch.attention_layer.built:
                self.typing_branch.attention_layer.build(
                    (None, self.typing_branch.K2, self.typing_branch.M)
                )
            with open("typing_attention.weights.pkl", "rb") as f:
                typing_attention_weights = pkl.load(f)
                self.typing_branch.attention_layer.set_weights(typing_attention_weights)
            print("Loaded typing attention weights from typing_attention.weights.pkl")

            print("Pretrained weights loaded successfully!")

        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")

    def freeze_pretrained_layers(self):
        """Freeze the pretrained embedding and attention layers"""
        self.tremor_branch.embeddings_network.trainable = False
        self.tremor_branch.attention_layer.trainable = False

        self.typing_branch.embeddings_network.trainable = False
        self.typing_branch.attention_layer.trainable = False

        print("Pretrained layers frozen")

    def get_bag_embeddings(self, tremor_input, typing_input):
        """Get bag-level embeddings from both branches"""
        # Get tremor bag embedding
        tremor_mask = self.tremor_branch.mask_layer(tremor_input)
        tremor_embeddings = self.tremor_branch.reshape_to_embeddings(tremor_input)
        tremor_embeddings = self.tremor_branch.embeddings_network(tremor_embeddings)
        tremor_embeddings = self.tremor_branch.reshape_to_attention(tremor_embeddings)

        tremor_alpha = self.tremor_branch.attention_layer(
            tremor_embeddings, tremor_mask
        )
        tremor_weighted = self.tremor_branch.weighted_embeddings_layer(
            [tremor_alpha, tremor_embeddings]
        )
        tremor_bag_embedding = self.tremor_branch.sum_layer(tremor_weighted)

        # Get typing bag embedding
        typing_mask = self.typing_branch.mask_layer(typing_input)
        typing_embeddings = self.typing_branch.embeddings_network(typing_input)
        typing_embeddings = self.typing_branch.reshape_to_attention(typing_embeddings)

        typing_alpha = self.typing_branch.attention_layer(
            typing_embeddings, typing_mask
        )
        typing_weighted = self.typing_branch.weighted_embeddings_layer(
            [typing_alpha, typing_embeddings]
        )
        typing_bag_embedding = self.typing_branch.sum_layer(typing_weighted)

        return tremor_bag_embedding, typing_bag_embedding

    def call(self, inputs):
        tremor_input, typing_input = inputs
        tremor_embedding, typing_embedding = self.get_bag_embeddings(
            tremor_input, typing_input
        )
        fused_embedding = layers.Add()([tremor_embedding, typing_embedding])
        return self.multilabel_classifier(fused_embedding)

    def compute_loss(self, x, y, y_pred, sample_weight=None):
        """
        Custom loss function as per paper: Ltotal = Ltremor + Lfmi + Lpd
        where Li = -yi * log(p(yi|X1,X2)) for i in {tremor, fmi, pd}
        """
        # Split predictions and targets for each task
        tremor_pred = y_pred[:, 0:1]  # Shape: (batch, 1)
        fmi_pred = y_pred[:, 1:2]  # Shape: (batch, 1)
        pd_pred = y_pred[:, 2:3]  # Shape: (batch, 1)

        tremor_true = tf.cast(y[:, 0:1], tf.float32)
        fmi_true = tf.cast(y[:, 1:2], tf.float32)
        pd_true = tf.cast(y[:, 2:3], tf.float32)

        # Compute individual binary cross-entropy losses
        tremor_loss = tf.keras.losses.binary_crossentropy(tremor_true, tremor_pred)
        fmi_loss = tf.keras.losses.binary_crossentropy(fmi_true, fmi_pred)
        pd_loss = tf.keras.losses.binary_crossentropy(pd_true, pd_pred)

        # Sum the three losses as specified in paper
        total_loss = tremor_loss + fmi_loss + pd_loss

        return tf.reduce_mean(total_loss)


def fusion_loso_evaluate(endtask_df):
    """Leave-One-Subject-Out (LOSO) evaluation for fusion model using DataFrame format."""
    print("\n" + "=" * 50)
    print("FUSION LOSO EVALUATION")
    print("=" * 50)

    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import multilabel_confusion_matrix, classification_report

    n_subjects = len(endtask_df)
    loo = LeaveOneOut()

    # Get input shapes from DataFrame
    tremor_sample = np.array(endtask_df.iloc[0]["X1"])
    typing_sample = np.array(endtask_df.iloc[0]["X2"])
    tremor_input_shape = tremor_sample.shape
    typing_input_shape = typing_sample.shape

    print(f"Tremor input shape: {tremor_input_shape}")
    print(f"Typing input shape: {typing_input_shape}")

    all_predictions = []
    all_true_labels = []

    subject_indices = list(range(n_subjects))

    for fold, (train_idx, test_idx) in enumerate(loo.split(subject_indices)):
        print(f"\nFold {fold + 1}/{n_subjects}")
        print(f"Training subjects: {len(train_idx)}, Test subject: {test_idx[0]}")

        # Get test subject ID for exclusion from pretraining
        test_subject_id = endtask_df.iloc[test_idx[0]]["subject_id"]
        print(f"Test subject ID: {test_subject_id}")

        # Prepare training data from DataFrame
        train_tremor_data = [endtask_df.iloc[i]["X1"] for i in train_idx]
        train_typing_data = [endtask_df.iloc[i]["X2"] for i in train_idx]
        train_labels = [endtask_df.iloc[i]["y"] for i in train_idx]

        # Prepare test data from DataFrame
        test_tremor_data = [endtask_df.iloc[i]["X1"] for i in test_idx]
        test_typing_data = [endtask_df.iloc[i]["X2"] for i in test_idx]
        test_labels = [endtask_df.iloc[i]["y"] for i in test_idx]

        # Convert to numpy arrays
        train_tremor_array = np.array(train_tremor_data)
        train_typing_array = np.array(train_typing_data)
        train_labels_array = np.array(train_labels)

        test_tremor_array = np.array(test_tremor_data)
        test_typing_array = np.array(test_typing_data)
        test_labels_array = np.array(test_labels)

        # Normalize tremor data (typing doesn't need normalization)
        train_tremor_array = normalize_mil(train_tremor_array)
        test_tremor_array = normalize_mil(test_tremor_array)

        # Create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices(
            ((train_tremor_array, train_typing_array), train_labels_array)
        )
        train_dataset = train_dataset.batch(FUSION_BATCH_SIZE).prefetch(
            tf.data.AUTOTUNE
        )

        test_dataset = tf.data.Dataset.from_tensor_slices(
            ((test_tremor_array, test_typing_array), test_labels_array)
        )
        test_dataset = test_dataset.batch(FUSION_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # Run pretraining for this fold
        print("Running pretraining for this fold...")
        run_pretraining(subject_exclude_id=test_subject_id)

        # Create and train fusion model
        fusion_model = FusionModel(
            tremor_input_shape=tremor_input_shape,
            typing_input_shape=typing_input_shape,
            M=TREMOR_M,  # Assuming same M for both branches
            mode=MODE,
        )

        # Compile model for multi-label classification
        # Paper specifies reduced learning rate of 0.0005 for fusion finetuning
        # Using custom loss that sums three separate binary cross-entropy losses
        fusion_model.compile(
            optimizer=optimizers.Adam(learning_rate=5e-4),  # 0.0005 as per paper
            loss=fusion_model.compute_loss,  # Custom loss: Ltotal = Ltremor + Lfmi + Lpd
            metrics=["accuracy"],
        )

        # Train the fusion model (only the classifier part)
        print("Training fusion model...")
        fusion_epochs = int(os.getenv("FUSION_EPOCHS", FUSION_NUM_EPOCHS))
        history = fusion_model.fit(train_dataset, epochs=fusion_epochs, verbose=1)

        # Evaluate on test subject
        print("Evaluating on test subject...")
        test_predictions = fusion_model.predict(test_dataset)

        # Convert predictions to binary (threshold 0.5)
        test_predictions_binary = (test_predictions > 0.5).astype(int)

        all_predictions.extend(test_predictions_binary)
        all_true_labels.extend(test_labels_array)

        print(f"Test subject predictions: {test_predictions_binary[0]}")
        print(f"Test subject true labels: {test_labels_array[0]}")

        # Clear memory
        del fusion_model
        gc.collect()
        k.clear_session()

    # Calculate overall metrics
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)

    print("\n" + "=" * 50)
    print("OVERALL RESULTS")
    print("=" * 50)

    label_names = ["Tremor", "FMI", "PD"]

    # Calculate confusion matrix-based metrics for each label
    from sklearn.metrics import confusion_matrix
    from utils import safe_confusion_matrix, calculate_metrics

    for i, label_name in enumerate(label_names):
        y_true = all_true_labels[:, i]
        y_pred = all_predictions[:, i]

        # Get confusion matrix values
        tn, fp, fn, tp = safe_confusion_matrix(y_true, y_pred)

        # Calculate metrics
        accuracy, sensitivity, specificity, precision, f1_score = calculate_metrics(
            tn, fp, fn, tp
        )

        print(f"\n{label_name} Results:")
        print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"  Sensitivity (Recall): {sensitivity:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  F1-Score: {f1_score:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")

    # Return metrics for this single run
    metrics_dict = {}
    for i, label_name in enumerate(label_names):
        y_true = all_true_labels[:, i]
        y_pred = all_predictions[:, i]
        tn, fp, fn, tp = safe_confusion_matrix(y_true, y_pred)
        accuracy, sensitivity, specificity, precision, f1_score = calculate_metrics(
            tn, fp, fn, tp
        )

        metrics_dict[f"{label_name.lower()}_accuracy"] = accuracy
        metrics_dict[f"{label_name.lower()}_sensitivity"] = sensitivity
        metrics_dict[f"{label_name.lower()}_specificity"] = specificity
        metrics_dict[f"{label_name.lower()}_precision"] = precision
        metrics_dict[f"{label_name.lower()}_f1_score"] = f1_score

    return all_predictions, all_true_labels, metrics_dict


def run_multiple_fusion_experiments(
    endtask_df,
    repetitions=10,
    save_path="../results/200_500_results_fusion_baseline.json",
    restart_interval=1,
):
    """
    Run the fusion LOSO experiment multiple times, calculate the average and standard deviation
    of metrics for each label, similar to results.py pattern. Restarts process every restart_interval repetitions.
    """
    import json

    # Check if we should restart the process
    start_rep = int(os.environ.get("RESULTS_START_REP", "0"))

    # Load existing results if available
    if os.path.exists(save_path):
        with open(save_path, "r") as file:
            saved_data = json.load(file)
            start_iteration = saved_data.get("completed_runs", 0)
            all_metrics = saved_data.get("all_metrics", [])
            start_iteration = max(start_iteration, start_rep)
    else:
        start_iteration = start_rep
        all_metrics = []

    # If we've completed all repetitions, print final results and exit
    if start_iteration >= repetitions:
        print("All repetitions completed!")

        # Helper function to safely compute mean and std, ignoring NaN values
        def safe_mean_std(values):
            if len(values) == 0 or all(np.isnan(values)):
                return np.nan, np.nan
            return np.nanmean(values), np.nanstd(values)

        # Calculate mean and standard deviation for each metric
        if len(all_metrics) > 0:
            metrics_summary = {}

            # Get all metric names from first run
            metric_names = list(all_metrics[0].keys())

            for metric_name in metric_names:
                values = [
                    run_metrics[metric_name]
                    for run_metrics in all_metrics
                    if not np.isnan(run_metrics[metric_name])
                ]
                mean_val, std_val = safe_mean_std(values)
                metrics_summary[f"{metric_name}_mean"] = mean_val
                metrics_summary[f"{metric_name}_std"] = std_val

            # Print the metrics summary
            print("\n" + "=" * 50)
            print("FINAL FUSION EXPERIMENTS SUMMARY")
            print("=" * 50)

            label_names = ["tremor", "fmi", "pd"]
            metric_types = [
                "accuracy",
                "sensitivity",
                "specificity",
                "precision",
                "f1_score",
            ]

            for label in label_names:
                print(f"\n{label.upper()} Results:")
                for metric in metric_types:
                    mean_key = f"{label}_{metric}_mean"
                    std_key = f"{label}_{metric}_std"
                    if mean_key in metrics_summary:
                        mean_val = metrics_summary[mean_key]
                        std_val = metrics_summary[std_key]
                        if not np.isnan(mean_val):
                            print(
                                f"  {metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}"
                            )
                        else:
                            print(f"  {metric.capitalize()}: No valid values (all NaN)")
        return

    # Calculate end iteration for this session
    end_iteration = min(start_iteration + restart_interval, repetitions)

    # Helper function to safely compute mean and std, ignoring NaN values
    def safe_mean_std(values):
        if len(values) == 0 or all(np.isnan(values)):
            return np.nan, np.nan
        return np.nanmean(values), np.nanstd(values)

    # Run the experiment for this session
    for i in range(start_iteration, end_iteration):
        print(f"\033[91mFusion Repetition {i + 1}/{repetitions}\033[0m")
        try:
            # Perform the fusion LOSO evaluation
            _, _, metrics = fusion_loso_evaluate(endtask_df)
            all_metrics.append(metrics)

            # Save results after every repetition
            with open(save_path, "w") as file:
                json.dump(
                    {
                        "completed_runs": i + 1,
                        "all_metrics": all_metrics,
                    },
                    file,
                )

        except Exception as e:
            print(f"Error during fusion repetition {i + 1}: {e}")
            import traceback

            traceback.print_exc()

        # Memory management
        gc.collect()
        k.clear_session()

    # If we haven't completed all repetitions, restart the process
    if end_iteration < repetitions:
        print(f"Completed {end_iteration} repetitions. Restarting process...")
        # Set environment variable for next start position
        env = os.environ.copy()
        env["RESULTS_START_REP"] = str(end_iteration)

        # Restart the script and exit current process immediately
        subprocess.Popen([sys.executable] + sys.argv, env=env)
        sys.exit(0)

    # Calculate mean and standard deviation for each metric (final run)
    if len(all_metrics) > 0:
        metrics_summary = {}

        # Get all metric names from first run
        metric_names = list(all_metrics[0].keys())

        for metric_name in metric_names:
            values = [
                run_metrics[metric_name]
                for run_metrics in all_metrics
                if not np.isnan(run_metrics[metric_name])
            ]
            mean_val, std_val = safe_mean_std(values)
            metrics_summary[f"{metric_name}_mean"] = mean_val
            metrics_summary[f"{metric_name}_std"] = std_val

        # Print the metrics summary
        print("\n" + "=" * 50)
        print("MULTIPLE FUSION EXPERIMENTS SUMMARY")
        print("=" * 50)

        label_names = ["tremor", "fmi", "pd"]
        metric_types = [
            "accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "f1_score",
        ]

        for label in label_names:
            print(f"\n{label.upper()} Results:")
            for metric in metric_types:
                mean_key = f"{label}_{metric}_mean"
                std_key = f"{label}_{metric}_std"
                if mean_key in metrics_summary:
                    mean_val = metrics_summary[mean_key]
                    std_val = metrics_summary[std_key]
                    if not np.isnan(mean_val):
                        print(
                            f"  {metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}"
                        )
                    else:
                        print(f"  {metric.capitalize()}: No valid values (all NaN)")

        # Save final summary
        with open(save_path, "w") as file:
            json.dump(
                {
                    "completed_runs": len(all_metrics),
                    "all_metrics": all_metrics,
                    "summary": metrics_summary,
                },
                file,
            )

        return metrics_summary
    else:
        print("No successful runs completed.")
        return {}


def run_fusion_experiment(repetitions=1):
    """Run the complete fusion experiment with optional multiple repetitions"""
    print("Starting fusion experiment...")

    # Create endtask dataset
    endtask_df = create_endtask_dataset()

    # print("Endtask dataset: ")
    # print(endtask_df)

    if repetitions == 1:
        # Single run
        predictions, true_labels, metrics = fusion_loso_evaluate(endtask_df)
        print("\nSingle fusion experiment completed!")
        return predictions, true_labels, metrics
    else:
        # Multiple runs with statistics
        summary = run_multiple_fusion_experiments(endtask_df, repetitions=repetitions)
        print("\nMultiple fusion experiments completed!")
        return summary


if __name__ == "__main__":
    # Choose what to run
    run_fusion_phase = True  # Set to True to run fusion experiment
    fusion_repetitions = 10  # Number of LOSO repetitions to run

    if run_fusion_phase:
        print("Running fusion phase...")
        results = run_fusion_experiment(repetitions=fusion_repetitions)
        print(f"Fusion experiment results: {type(results)}")

    print(f"\nTotal execution time: {time.time() - start:.2f} seconds")

    # Alarm (commented out for Linux compatibility)
    os.system('powershell.exe -c "[console]::beep(999,1000)"')
