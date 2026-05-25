"""
tremorSubjectSimCLR.py — Subject-level SimCLR pretraining for the tremor branch.

Instead of contrasting individual windows (as in tremorSimCLR.py), this script
contrasts subject-level embeddings produced by the encoder + attention MIL stack.

Positive pairs: two random 90% sub-samples of the same subject's windows.
Negative pairs: all other subjects in the batch.

Memory bank:
    A MoCo-style circular queue (size ``queue_size``, default 460) stores
    L2-normalized subject projections from recent steps as additional negatives.
    This gives ~459 negatives per loss step instead of batch_size-1 = 3.

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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
learning_rate = 3e-4
sample_frac = 0.75  # Fraction of subject windows to sample per view
queue_size = 64     # Memory bank: stored subject projections (≈ all 464 subjects)
moco_momentum = 0.99  # EMA decay for key encoder: θ_k ← m·θ_k + (1−m)·θ_q
USE_AUGMENTATION = True    # Set to False to disable stochastic augmentations (normalisation always runs)
# Individual augmentations to include (only used when USE_AUGMENTATION = True):
#   "flipping"             — left-to-right time-series flip (time-reversal invariance)
#   "bidirectional_flipping" — sign flip (×-1); too strong when combined with rotation
#   "rotation"             — random 3-D axis-angle rotation (orientation invariance)
#   "gravity"              — add random gravity vector (mild DC-offset variation)
#   "permute_segments"     — shuffle temporal segments (destroys tremor frequency — avoid)
ACTIVE_AUGMENTATIONS = {"flipping", "rotation"}
ROTATION_ANGLE = np.pi / 4  # Max rotation angle (radians). π = 180°, π/4 = 45° (safer default)
DEBUG = False          # If True, train on a small random subset of subjects
DEBUG_SUBJECTS = 20    # Number of subjects to use when DEBUG = True
USE_TRAINING = False    # If False, skip training and load saved weights for t-SNE visualization

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

if DEBUG:
    rng = np.random.default_rng(42)
    debug_idx = rng.choice(len(subject_data), size=min(DEBUG_SUBJECTS, len(subject_data)), replace=False)
    subject_data = [subject_data[i] for i in sorted(debug_idx)]
    queue_size = min(queue_size, len(subject_data))
    print(f"[DEBUG] Using {len(subject_data)} subjects, queue_size capped to {queue_size}.")

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

    def __init__(self, m: int, k1: int, ws: int, c: int, temp: float, queue_size: int = 460, moco_momentum: float = 0.99):
        super().__init__()
        self.m = m
        self.k1 = k1
        self.ws = ws
        self.c = c
        self.temperature = temp
        self.queue_size = queue_size
        self.moco_momentum = moco_momentum

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

        # ── EMA key encoder (MoCo momentum network) ──────────────────────────
        # Updated via θ_k ← m·θ_k + (1−m)·θ_q after each step.
        # All queue entries come from this stable, slowly-drifting encoder,
        # ensuring queue consistency — the root cause of the erratic loss.
        self.key_encoder = build_encoder(m)
        self.key_attention_layer = MILAttentionLayer(
            weight_params_dim=16,
            kernel_regularizer=keras.regularizers.L2(0.01),
            use_gated=False,
            name="key_alpha",
        )
        self.key_projection_head = keras.Sequential(
            [
                keras.Input(shape=(m,)),
                layers.Dense(m, activation="relu"),
                layers.Dense(m),
            ],
            name="key_projection_head",
        )

        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="c_acc"
        )
        # Normaliser — always applied regardless of USE_AUGMENTATION, so the
        # encoder never sees raw-amplitude differences between subjects as a
        # trivial shortcut.
        self.normalizer = Augmentation.CustomNormalizer()

        # Window-level augmenter — built from ACTIVE_AUGMENTATIONS set.
        # Normalisation is intentionally NOT included here; it runs separately.
        _aug = Augmentation(rotation_angle=ROTATION_ANGLE)
        _aug_steps = []
        if "flipping"               in ACTIVE_AUGMENTATIONS: _aug_steps.append(layers.Lambda(_aug.left_to_right_flipping))
        if "bidirectional_flipping" in ACTIVE_AUGMENTATIONS: _aug_steps.append(layers.Lambda(_aug.bidirectional_flipping))
        if "rotation"               in ACTIVE_AUGMENTATIONS: _aug_steps.append(layers.Lambda(_aug.rotate_axis))
        if "gravity"                in ACTIVE_AUGMENTATIONS: _aug_steps.append(layers.Lambda(_aug.add_gravity))
        if "permute_segments"       in ACTIVE_AUGMENTATIONS: _aug_steps.append(layers.Lambda(_aug.permute_segments))
        self.augmenter = keras.Sequential(_aug_steps) if _aug_steps else None

        # ── Memory bank ──────────────────────────────────────────────────────
        # Circular queue of L2-normalized subject projections used as extra
        # negatives.  Stored in float32 regardless of mixed-precision policy.
        init_queue = tf.math.l2_normalize(
            tf.random.normal((queue_size, m), dtype=tf.float32), axis=1
        )
        self.queue = tf.Variable(
            init_queue, trainable=False, dtype=tf.float32, name="queue"
        )
        self.queue_ptr = tf.Variable(0, trainable=False, dtype=tf.int32, name="queue_ptr")

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
        flat = tf.reshape(windows_batch, (-1, self.ws, self.c))    # (batch*k1, ws, c)

        window_embs = self.encoder(flat, training=training)        # (batch*k1, m)
        window_embs = tf.reshape(window_embs, (-1, self.k1, self.m))  # (batch, k1, m)

        if mask is None:
            mask = self._build_mask(windows_batch)  # (batch, k1, 1)

        alpha = self.attention_layer(window_embs, mask)  # (batch, k1, 1)

        # Weighted sum → subject embedding
        subject_embs = tf.reduce_sum(alpha * window_embs, axis=1)  # (batch, m)
        return subject_embs

    def _encode_subjects_key(self, windows_batch: tf.Tensor, mask=None) -> tf.Tensor:
        """Produce subject-level embeddings using the EMA key encoder.

        Always runs with training=False — the key encoder is never updated by
        backpropagation, only by the EMA momentum update rule.
        """
        flat = tf.reshape(windows_batch, (-1, self.ws, self.c))
        window_embs = self.key_encoder(flat, training=False)
        window_embs = tf.reshape(window_embs, (-1, self.k1, self.m))
        if mask is None:
            mask = self._build_mask(windows_batch)
        alpha = self.key_attention_layer(window_embs, mask)
        return tf.reduce_sum(alpha * window_embs, axis=1)

    def initialize_key_encoder(self):
        """Copy query encoder weights → key encoder.  Called once before training."""
        for q_w, k_w in zip(self.encoder.weights, self.key_encoder.weights):
            k_w.assign(tf.cast(q_w, k_w.dtype))
        for q_w, k_w in zip(self.attention_layer.weights, self.key_attention_layer.weights):
            k_w.assign(tf.cast(q_w, k_w.dtype))
        for q_w, k_w in zip(self.projection_head.weights, self.key_projection_head.weights):
            k_w.assign(tf.cast(q_w, k_w.dtype))
        print("Key encoder initialized from query encoder weights.")

    def _momentum_update(self):
        """EMA update: θ_k ← moco_momentum·θ_k + (1−moco_momentum)·θ_q

        Uses .weights (trainable + non-trainable) so that BatchNorm running
        statistics (moving_mean, moving_variance) in the key encoder also track
        the query encoder, preventing stale normalisation in the key path.
        """
        m = self.moco_momentum
        for q_w, k_w in zip(self.encoder.weights, self.key_encoder.weights):
            k_w.assign(m * k_w + (1.0 - m) * tf.cast(q_w, k_w.dtype))
        for q_w, k_w in zip(self.attention_layer.weights, self.key_attention_layer.weights):
            k_w.assign(m * k_w + (1.0 - m) * tf.cast(q_w, k_w.dtype))
        for q_w, k_w in zip(self.projection_head.weights, self.key_projection_head.weights):
            k_w.assign(m * k_w + (1.0 - m) * tf.cast(q_w, k_w.dtype))

    # ── NT-Xent loss ─────────────────────────────────────────────────────────

    def contrastive_loss(
        self, proj1_q: tf.Tensor, proj2_k: tf.Tensor, queue_snapshot: tf.Tensor
    ) -> tf.Tensor:
        """Asymmetric MoCo InfoNCE loss.

        proj1_q (query, from main encoder) is matched against proj2_k (key, from
        EMA key encoder) as the positive, and all queue entries as negatives.
        Gradients flow only through the query path (proj1_q).

        Args:
            proj1_q:        (B, M) — query projections (main encoder, gradient flows here)
            proj2_k:        (B, M) — key projections (EMA encoder, stop_gradient applied by caller)
            queue_snapshot: (Q, M) float32 — memory bank negatives (no gradient)
        """
        proj1_q = tf.nn.l2_normalize(tf.cast(proj1_q, tf.float32), axis=1)
        proj2_k = tf.nn.l2_normalize(tf.cast(proj2_k, tf.float32), axis=1)

        batch = ops.shape(proj1_q)[0]
        labels = ops.arange(batch)  # positive for query i is key i (proj2_k[i])

        # Keys: current batch positives + memory bank negatives → (B + Q, M)
        keys = tf.concat([proj2_k, queue_snapshot], axis=0)

        # (B, B+Q) similarity matrix; query i's positive is at column i
        similarities = ops.matmul(proj1_q, ops.transpose(keys)) / self.temperature

        self.contrastive_accuracy.update_state(labels, similarities)
        return keras.losses.sparse_categorical_crossentropy(
            labels, similarities, from_logits=True
        )

    # ── Memory bank update ───────────────────────────────────────────────────

    @tf.function
    def _dequeue_and_enqueue(self, keys: tf.Tensor):
        """Overwrite the oldest queue entries with new L2-normalized projections.

        Args:
            keys: (B, M) float32 — normalized subject projections to enqueue
        """
        batch_size = tf.shape(keys)[0]
        ptr = self.queue_ptr
        indices = tf.math.mod(
            tf.range(ptr, ptr + batch_size, dtype=tf.int32), self.queue_size
        )
        self.queue.assign(
            tf.tensor_scatter_nd_update(
                self.queue, tf.expand_dims(indices, axis=1), keys
            )
        )
        self.queue_ptr.assign(tf.math.mod(ptr + batch_size, self.queue_size))

    # ── Train step ───────────────────────────────────────────────────────────

    @tf.function
    def train_step_contrastive(
        self,
        views1: tf.Tensor,
        views2: tf.Tensor,
        optimizer: keras.optimizers.Optimizer,
        queue_snapshot: tf.Tensor,
    ) -> tuple:
        """Single gradient update — MoCo-style query/key separation.

        Query path  (view1, inside GradientTape):
            view1 → encoder → attention → projection_head → proj1_q

        Key path    (view2, outside GradientTape, no backprop):
            view2 → key_encoder → key_attention → key_projection_head → proj2_k

        The key encoder is EMA-updated via _momentum_update() after each step
        (called from the training loop, not here, to avoid @tf.function issues).

        Returns:
            (metrics_dict, proj2_k_norm) where proj2_k_norm is (B, M) float32
            to be enqueued in the memory bank.
        """
        # ── Masks from original views (before augmentation) ──────────────────
        mask1 = self._build_mask(views1)
        mask2 = self._build_mask(views2)

        # ── Normalise + augment both views independently ─────────────────────
        # Normalisation always runs to remove per-subject amplitude bias.
        # Stochastic augmentations only run when USE_AUGMENTATION=True.
        orig_shape = tf.shape(views1)
        flat1 = tf.reshape(views1, (-1, self.ws, self.c))
        flat2 = tf.reshape(views2, (-1, self.ws, self.c))
        flat1 = self.normalizer(flat1, training=False)
        flat2 = self.normalizer(flat2, training=False)
        if USE_AUGMENTATION and self.augmenter is not None:
            flat1 = self.augmenter(flat1, training=True)
            flat2 = self.augmenter(flat2, training=True)
        views1_aug = tf.reshape(flat1, orig_shape)
        views2_aug = tf.reshape(flat2, orig_shape)

        # ── Key path (no gradients) ───────────────────────────────────────────
        # The key encoder variables are not in trainable_vars below, so TF won't
        # differentiate through them regardless; stop_gradient makes intent clear.
        embs2_k = self._encode_subjects_key(views2_aug, mask=mask2)
        proj2_k = tf.stop_gradient(self.key_projection_head(embs2_k, training=False))

        # ── Query path (with gradients) ───────────────────────────────────────
        with tf.GradientTape() as tape:
            embs1_q = self._encode_subjects(views1_aug, training=True, mask=mask1)
            proj1_q = self.projection_head(embs1_q, training=True)

            loss = tf.reduce_mean(
                self.contrastive_loss(proj1_q, proj2_k, queue_snapshot)
            )
            self.contrastive_loss_tracker.update_state(loss)

        trainable_vars = (
            self.encoder.trainable_variables
            + self.attention_layer.trainable_variables
            + self.projection_head.trainable_variables
        )
        grads = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))

        proj2_k_norm = tf.nn.l2_normalize(tf.cast(proj2_k, tf.float32), axis=1)

        return {
            "c_loss": self.contrastive_loss_tracker.result(),
            "c_acc": self.contrastive_accuracy.result(),
        }, proj2_k_norm


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
        print(f"\nEpoch {epoch + 1}/{num_epochs}  (lr={learning_rate:.2e})")
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

            metrics, proj2_k_norm = model.train_step_contrastive(
                v1_t, v2_t, optimizer, model.queue
            )
            model._momentum_update()           # EMA: θ_k ← m·θ_k + (1−m)·θ_q
            model._dequeue_and_enqueue(proj2_k_norm)

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
        m=M, k1=K1, ws=Ws, c=C, temp=temperature, queue_size=queue_size,
        moco_momentum=moco_momentum,
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Build both encoder pairs by running dummy forward passes
    dummy = np.zeros((2, K1, Ws, C), dtype=np.float32)
    dummy_t = tf.constant(dummy)
    _ = model._encode_subjects(dummy_t, training=False)
    _ = model._encode_subjects_key(dummy_t)
    _ = model.projection_head(tf.zeros((2, M), dtype=tf.float32))
    _ = model.key_projection_head(tf.zeros((2, M), dtype=tf.float32))

    encoder_path = "weights/tremor/tremor_subject_simclr_embeddings.weights.h5"
    attention_path = "weights/tremor/tremor_subject_simclr_attention.weights.pkl"

    if USE_TRAINING:
        # Seed key encoder with the same initial weights as the query encoder
        model.initialize_key_encoder()

        model.encoder.summary()
        model.projection_head.summary()

        train(model, optimizer)

        # ── Save weights ─────────────────────────────────────────────────────
        os.makedirs("weights/tremor", exist_ok=True)

        model.encoder.save_weights(encoder_path)
        with open(attention_path, "wb") as f:
            pkl.dump(model.attention_layer.get_weights(), f)

        print(f"\nEncoder weights saved to '{encoder_path}'")
        print(f"Attention weights saved to '{attention_path}'")
        print(f"Total training time: {(time.time() - start) / 60:.1f} min")
    else:
        # ── Load saved weights ────────────────────────────────────────────────
        model.encoder.load_weights(encoder_path)
        with open(attention_path, "rb") as f:
            model.attention_layer.set_weights(pkl.load(f))
        print(f"Loaded encoder weights from '{encoder_path}'")
        print(f"Loaded attention weights from '{attention_path}'")

    # ── t-SNE visualization ───────────────────────────────────────────────────
    print("\nGenerating t-SNE subject-level embeddings from labeled dataset...")
    with open("datasets/sdataset.pickle", "rb") as f:
        labeled_df = pkl.load(f)

    labels_np = np.array(labeled_df["y_train"].tolist())

    # Pad/truncate each subject bag to K1 windows
    all_bags = labeled_df["X"].tolist()
    bags_padded = np.zeros((len(all_bags), K1, Ws, C), dtype=np.float32)
    for i, bag in enumerate(all_bags):
        bag = np.array(bag, dtype=np.float32)
        n = min(len(bag), K1)
        bags_padded[i, :n] = bag[:n]

    # Encode subjects in small batches to avoid OOM
    infer_batch = 4
    subject_embs_list = []
    for i in range(0, len(bags_padded), infer_batch):
        chunk = tf.constant(bags_padded[i : i + infer_batch])
        flat = tf.reshape(chunk, (-1, Ws, C))
        flat_norm = model.normalizer(flat, training=False)
        chunk_norm = tf.reshape(flat_norm, (-1, K1, Ws, C))
        embs = model._encode_subjects(chunk_norm, training=False)
        subject_embs_list.append(embs.numpy())
    embeddings = np.concatenate(subject_embs_list, axis=0).astype(np.float32)
    print(f"Subject embeddings shape: {embeddings.shape}")

    reduced = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[labels_np == 0, 0], reduced[labels_np == 0, 1], label="No tremor", c="b", alpha=0.5)
    plt.scatter(reduced[labels_np == 1, 0], reduced[labels_np == 1, 1], label="Tremor", c="r", alpha=0.5)
    plt.title("t-SNE — Subject SimCLR Embeddings (attention-aggregated)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/tremor_subject_simclr_tsne.png", dpi=150)
    plt.show()
    print("t-SNE plot saved to 'plots/tremor_subject_simclr_tsne.png'")
