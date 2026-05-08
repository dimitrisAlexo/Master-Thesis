"""
tremorSubjectSimCLR.py — Subject-level SimCLR pretraining for the tremor branch.

Instead of contrasting individual windows (as in tremorSimCLR.py), this script
contrasts subject-level embeddings produced by the encoder + attention MIL stack.

Positive pairs: two random 90% sub-samples of the same subject's windows.
Negative pairs: all other subjects in the batch.

Saved weights:
    weights/tremor/tremor_subject_simclr_embeddings.weights.h5  (encoder)
    weights/tremor/tremor_subject_simclr_attention.weights.h5   (attention layer)
"""

import os
import time
import pickle as pkl
import queue
import threading

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KERAS_BACKEND"] = "tensorflow"

import resource

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

import tensorflow as tf
import keras
from keras import layers, ops
from tf_keras import mixed_precision

from tremorSimCLRattentionMIL import MILAttentionLayer
from augmentations import Augmentation

# ── Environment ─────────────────────────────────────────────────────────────

os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

if tf.config.list_physical_devices("GPU"):
    print("Using GPU...")
else:
    print("Using CPU...")

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)
print("Using mixed precision...")

# ── Hyperparameters ──────────────────────────────────────────────────────────

K1 = 200           # Windows per subject view (after padding)
M = 64             # Embedding dimension
Ws = 1000          # Window length (time steps)
C = 3              # Channels (x, y, z)
batch_size = 4     # Subjects per batch — GPU limit: batch×K1 windows held under GradientTape
num_epochs = 50
temperature = 0.1
learning_rate = 0.001
sample_frac = 0.75  # Fraction of subject windows to sample per view
encoder_chunk_size = 50  # Max windows per encoder forward pass
USE_AUGMENTATION = False   # Set to False to disable window-level augmentations

# ── Dataset ──────────────────────────────────────────────────────────────────

DATASET_PATH = "datasets/unlabeled_subject_data.pickle"
GDATA_PATH = os.path.join("..", "data", "tremor_gdata.pickle")
SDATA_PATH = os.path.join("..", "data", "tremor_sdata.pickle")
E_THRES = 0.30

if not os.path.exists(DATASET_PATH):
    print(f"'{DATASET_PATH}' not found — building from raw data...")
    with open(GDATA_PATH, "rb") as f:
        tremor_gdata = pkl.load(f)
    with open(SDATA_PATH, "rb") as f:
        tremor_sdata = pkl.load(f)
    from utils import form_unlabeled_subject_tremor_dataset
    form_unlabeled_subject_tremor_dataset(tremor_gdata, tremor_sdata, E_THRES, K1)
    print("Dataset created.")

with open(DATASET_PATH, "rb") as f:
    subject_data = pkl.load(f)

print(f"Loaded {len(subject_data)} subjects.")
print(f"Window counts per subject (first 5): {[len(s) for s in subject_data[:5]]}")

# ── View creation ────────────────────────────────────────────────────────────


def create_subject_view(windows: np.ndarray, k1: int, frac: float) -> tuple:
    """Sample a random subset of windows and pad to k1.

    Args:
        windows: Array of shape (N, Ws, C) — real (unpadded) windows for one subject.
        k1:      Target number of windows after padding.
        frac:    Fraction of windows to sample.

    Returns:
        view:  np.ndarray of shape (k1, Ws, C)
        mask:  np.ndarray of shape (k1,) — True for real windows, False for padding
    """
    n = len(windows)
    n_sample = max(1, int(n * frac))
    idx = np.random.choice(n, size=n_sample, replace=False)
    sampled = windows[idx]  # (n_sample, Ws, C)

    if n_sample >= k1:
        view = sampled[:k1]
        mask = np.ones(k1, dtype=bool)
    else:
        padding = np.zeros((k1 - n_sample, Ws, C), dtype=sampled.dtype)
        view = np.concatenate([sampled, padding], axis=0)
        mask = np.array([True] * n_sample + [False] * (k1 - n_sample), dtype=bool)

    return view, mask


# ── Encoder architecture ─────────────────────────────────────────────────────


def build_encoder(m: int) -> keras.Sequential:
    """Conv1D encoder — identical to tremorSimCLR.py's embeddings_function."""
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
            # Flatten → M-dim embedding
            layers.Flatten(),
            layers.Dense(m),
        ],
        name="embeddings_function",
    )


# ── Subject-level contrastive model ─────────────────────────────────────────


class SubjectContrastiveModel(keras.Model):
    """Encoder + attention trained with subject-level NT-Xent contrastive loss."""

    def __init__(self, m: int, k1: int, ws: int, c: int, temp: float, chunk_size: int = 256):
        super().__init__()
        self.m = m
        self.k1 = k1
        self.ws = ws
        self.c = c
        self.temperature = temp
        self.encoder_chunk_size = chunk_size

        self.encoder = build_encoder(m)

        self.attention_layer = MILAttentionLayer(
            weight_params_dim=16,
            kernel_regularizer=keras.regularizers.L2(0.01),
            use_gated=False,
            name="alpha",
        )

        self.projection_head = keras.Sequential(
            [
                keras.Input(shape=(m,)),
                layers.Dense(m, activation="relu"),
                layers.Dense(m),
            ],
            name="projection_head",
        )

        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="c_acc"
        )
        # Window-level augmenter (skip shift_windows — windows are already 1000 steps)
        self.augmenter = Augmentation().get_contrastive_augmenter()

    # ── Forward helpers ──────────────────────────────────────────────────────

    def _build_mask(self, windows_batch: tf.Tensor) -> tf.Tensor:
        """Detect zero-padded windows.

        Args:
            windows_batch: (batch, k1, ws, c)

        Returns:
            mask: (batch, k1, 1) — 0.0 for real windows, -inf for padding
        """
        summed = tf.reduce_sum(tf.abs(windows_batch), axis=[2, 3], keepdims=True)
        summed = tf.squeeze(summed, axis=-1)  # (batch, k1, 1)
        mask = tf.where(summed < 1e-3, -np.inf, 0.0)
        return mask

    def _encode_subjects(self, windows_batch: tf.Tensor, training: bool, mask=None) -> tf.Tensor:
        """Produce subject-level embeddings.

        Args:
            windows_batch: (batch, k1, ws, c)
            mask:          Optional pre-computed mask (batch, k1, 1). If None, computed
                           from windows_batch. Pass a pre-computed mask when augmentation
                           has been applied (padded windows may no longer be all-zeros).

        Returns:
            subject_embs: (batch, m)
        """
        n_batch = int(windows_batch.shape[0])
        n_windows = n_batch * self.k1
        flat = tf.reshape(windows_batch, (n_windows, self.ws, self.c))

        # Encode windows in chunks to avoid GPU OOM
        parts = []
        for start in range(0, n_windows, self.encoder_chunk_size):
            parts.append(
                self.encoder(flat[start : start + self.encoder_chunk_size], training=training)
            )
        window_embs = tf.concat(parts, axis=0)                          # (n_windows, m)
        window_embs = tf.reshape(window_embs, (n_batch, self.k1, self.m))  # (batch, k1, m)

        if mask is None:
            mask = self._build_mask(windows_batch)  # (batch, k1, 1)

        alpha = self.attention_layer(window_embs, mask)  # (batch, k1, 1)

        # Weighted sum → subject embedding
        subject_embs = tf.reduce_sum(alpha * window_embs, axis=1)  # (batch, m)
        return subject_embs

    # ── NT-Xent loss ─────────────────────────────────────────────────────────

    def contrastive_loss(self, proj1: tf.Tensor, proj2: tf.Tensor) -> tf.Tensor:
        proj1 = tf.nn.l2_normalize(proj1, axis=1)
        proj2 = tf.nn.l2_normalize(proj2, axis=1)

        similarities = (
            ops.matmul(proj1, ops.transpose(proj2)) / self.temperature
        )  # (batch, batch)

        batch = ops.shape(proj1)[0]
        labels = ops.arange(batch)

        self.contrastive_accuracy.update_state(labels, similarities)
        self.contrastive_accuracy.update_state(labels, ops.transpose(similarities))

        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            labels, ops.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2

    # ── Train step ───────────────────────────────────────────────────────────

    @tf.function
    def train_step_contrastive(
        self,
        views1: tf.Tensor,
        views2: tf.Tensor,
        optimizer: keras.optimizers.Optimizer,
    ) -> dict:
        """Single gradient update on one batch of subject view pairs.

        Args:
            views1: (batch, k1, ws, c) — first augmented view per subject
            views2: (batch, k1, ws, c) — second augmented view per subject
            optimizer: Keras optimizer

        Returns:
            dict with loss and accuracy values
        """
        with tf.GradientTape() as tape:
            # Compute masks from original views BEFORE augmentation.
            # Augmentation (especially CustomNormalizer) turns zero-padded windows
            # into non-zero values, so the mask must be derived from the original.
            mask1 = self._build_mask(views1)
            mask2 = self._build_mask(views2)

            # Apply window-level augmentations independently to each window
            # within the bag (flatten batch×K1 → augment → reshape back).
            orig_shape = tf.shape(views1)
            if USE_AUGMENTATION:
                flat1 = tf.reshape(views1, (-1, self.ws, self.c))
                flat2 = tf.reshape(views2, (-1, self.ws, self.c))
                flat1 = self.augmenter(flat1, training=True)
                flat2 = self.augmenter(flat2, training=True)
                views1_aug = tf.reshape(flat1, orig_shape)
                views2_aug = tf.reshape(flat2, orig_shape)
            else:
                views1_aug = views1
                views2_aug = views2

            embs1 = self._encode_subjects(views1_aug, training=True, mask=mask1)
            embs2 = self._encode_subjects(views2_aug, training=True, mask=mask2)

            proj1 = self.projection_head(embs1, training=True)
            proj2 = self.projection_head(embs2, training=True)

            loss = tf.reduce_mean(self.contrastive_loss(proj1, proj2))
            self.contrastive_loss_tracker.update_state(loss)

        trainable_vars = (
            self.encoder.trainable_variables
            + self.attention_layer.trainable_variables
            + self.projection_head.trainable_variables
        )
        grads = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))

        return {
            "c_loss": self.contrastive_loss_tracker.result(),
            "c_acc": self.contrastive_accuracy.result(),
        }


# ── Training loop ─────────────────────────────────────────────────────────────


def _prefetch_worker(
    subject_data: list,
    perm: np.ndarray,
    steps: int,
    batch_size: int,
    k1: int,
    ws: int,
    c: int,
    frac: float,
    out_queue: queue.Queue,
):
    """Background thread: build numpy view batches and enqueue them."""
    for step in range(steps):
        batch_idx = perm[step * batch_size : (step + 1) * batch_size]
        if len(batch_idx) < 2:
            out_queue.put(None)
            continue
        views1, views2 = build_batch(subject_data, batch_idx, k1, ws, c, frac)
        out_queue.put((views1, views2))


def build_batch(
    subject_data: list, batch_indices: np.ndarray, k1: int, ws: int, c: int, frac: float
) -> tuple:
    """Create two view tensors for a batch of subjects.

    Returns:
        views1: np.ndarray (batch, k1, ws, c)
        views2: np.ndarray (batch, k1, ws, c)
    """
    b = len(batch_indices)
    views1 = np.zeros((b, k1, ws, c), dtype=np.float32)
    views2 = np.zeros((b, k1, ws, c), dtype=np.float32)

    for i, idx in enumerate(batch_indices):
        windows = subject_data[idx].astype(np.float32)
        v1, _ = create_subject_view(windows, k1, frac)
        v2, _ = create_subject_view(windows, k1, frac)
        views1[i] = v1
        views2[i] = v2

    return views1, views2


def train(model: SubjectContrastiveModel, optimizer: keras.optimizers.Optimizer):
    n_subjects = len(subject_data)
    steps_per_epoch = max(1, n_subjects // batch_size)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.contrastive_loss_tracker.reset_state()
        model.contrastive_accuracy.reset_state()

        perm = np.random.permutation(n_subjects)
        progbar = keras.utils.Progbar(steps_per_epoch, stateful_metrics=["c_loss", "c_acc"])

        # Pre-fetch batches on a background CPU thread so data prep overlaps GPU training
        prefetch_q = queue.Queue(maxsize=2)
        prefetch_thread = threading.Thread(
            target=_prefetch_worker,
            args=(subject_data, perm, steps_per_epoch, batch_size, K1, Ws, C, sample_frac, prefetch_q),
            daemon=True,
        )
        prefetch_thread.start()

        for step in range(steps_per_epoch):
            batch = prefetch_q.get()
            if batch is None:
                continue

            views1, views2 = batch
            v1_t = tf.constant(views1)
            v2_t = tf.constant(views2)

            metrics = model.train_step_contrastive(v1_t, v2_t, optimizer)

            progbar.update(
                step + 1,
                values=[
                    ("c_loss", float(metrics["c_loss"])),
                    ("c_acc", float(metrics["c_acc"])),
                ],
            )

        prefetch_thread.join()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start = time.time()

    model = SubjectContrastiveModel(
        m=M, k1=K1, ws=Ws, c=C, temp=temperature, chunk_size=encoder_chunk_size
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Build the model by running one dummy batch so all weights are created
    dummy = np.zeros((2, K1, Ws, C), dtype=np.float32)
    dummy_t = tf.constant(dummy)
    _ = model._encode_subjects(dummy_t, training=False)
    _ = model.projection_head(tf.zeros((2, M), dtype=tf.float32))

    model.encoder.summary()
    model.projection_head.summary()

    train(model, optimizer)

    # ── Save weights ─────────────────────────────────────────────────────────
    os.makedirs("weights/tremor", exist_ok=True)

    encoder_path = "weights/tremor/tremor_subject_simclr_embeddings.weights.h5"
    attention_path = "weights/tremor/tremor_subject_simclr_attention.weights.pkl"

    model.encoder.save_weights(encoder_path)
    with open(attention_path, "wb") as f:
        pkl.dump(model.attention_layer.get_weights(), f)

    print(f"\nEncoder weights saved to '{encoder_path}'")
    print(f"Attention weights saved to '{attention_path}'")
    print(f"Total training time: {(time.time() - start) / 60:.1f} min")
