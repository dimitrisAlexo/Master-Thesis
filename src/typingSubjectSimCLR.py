"""
typingSubjectSimCLR.py — Subject-level SimCLR pretraining for the typing branch.

Instead of contrasting individual histogram windows (as in typingSimCLR.py), this script
contrasts subject-level embeddings produced by the encoder + attention MIL stack.

Positive pairs: two disjoint halves of the same subject's typing sessions
(or two random overlapping sub-samples when DISJOINT_VIEWS = False).
Negative pairs: all other subjects in the batch.

Memory bank:
    A MoCo-style circular queue (size ``queue_size``, default 64) stores
    L2-normalized subject projections from recent steps as additional negatives.
    Subject indices are stored alongside the keys so that queue entries belonging
    to a subject present in the current batch are masked out of the loss
    (false-negative removal).

Saved weights:
    weights/typing/typing_subject_simclr_embeddings.weights.h5  (encoder)
    weights/typing/typing_subject_simclr_attention.weights.pkl  (attention layer)
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

from typingSimCLRattentionMIL import MILAttentionLayer

# ── Environment ──────────────────────────────────────────────────────────────

os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

if tf.config.list_physical_devices("GPU"):
    print("Using GPU...")
else:
    print("Using CPU...")

# policy = mixed_precision.Policy("mixed_float16")
# mixed_precision.set_global_policy(policy)
# print("Using mixed precision...")

policy = mixed_precision.Policy("float32")
mixed_precision.set_global_policy(policy)
print("Using float32 precision...")

# ── Typing augmentation ───────────────────────────────────────────────────────
# Inlined from typingSimCLR.py to avoid module-level side-effects on import.


class TypingAugmentation:
    def __init__(
        self,
        noise_factor=0.02,
        dropout_rate=0.05,
        n_perm_seg=8,
        scale_range=(0.85, 1.15),
        max_shift_hold=2,
        max_shift_flight=3,
    ):
        self.noise_factor = noise_factor
        self.dropout_rate = dropout_rate
        self.n_perm_seg = n_perm_seg
        self.scale_range = scale_range
        self.max_shift_hold = max_shift_hold
        self.max_shift_flight = max_shift_flight

    def add_noise(self, data):
        noise = tf.random.normal(
            tf.shape(data), mean=0.0, stddev=self.noise_factor, dtype=tf.float32
        )
        return data + noise

    def dropout_features(self, data):
        mask = tf.random.uniform(tf.shape(data)) > self.dropout_rate
        return data * tf.cast(mask, tf.float32)

    def poisson_resample(self, data, n_virtual=100.0):
        # Each histogram is the empirical distribution of a finite typing session.
        # Poisson-bootstrap the bin counts (as if the session were re-recorded with
        # ~n_virtual keystrokes): perturbation scales with sqrt(p/N), so it is
        # proportional to the signal, never moves mass to empty bins, and leaves
        # all-zero padding rows untouched. normalize_histogram restores sum-to-1.
        counts = tf.random.poisson(shape=[], lam=tf.nn.relu(data) * n_virtual)
        return counts / n_virtual

    def random_scaling(self, data):
        batch_size = tf.shape(data)[0]
        scale = tf.random.uniform(
            [batch_size, 1],
            minval=self.scale_range[0],
            maxval=self.scale_range[1],
            dtype=tf.float32,
        )
        return data * scale

    def permute_histogram_segments(self, data):
        batch_size = tf.shape(data)[0]
        hold_time_data = data[:, :101]
        flight_time_data = data[:, 101:]

        def permute_section(section_data):
            section_features = tf.shape(section_data)[1]
            divisor = section_features // self.n_perm_seg
            remainder = section_features % self.n_perm_seg
            n_full = self.n_perm_seg - 1  # Python int — known at trace time

            def permute_segments():
                # (batch, n_full, divisor)
                segments = tf.reshape(
                    section_data[:, : divisor * n_full],
                    [batch_size, n_full, divisor],
                )
                last = section_data[:, divisor * n_full:]  # (batch, divisor+remainder)

                # Vectorised batch permutation: argsort of per-sample uniform noise
                # produces an independent random permutation for every sample with
                # no sequential per-sample loop (XLA-friendly, no tf.map_fn).
                perm_indices = tf.argsort(
                    tf.random.uniform([batch_size, n_full]), axis=1
                )  # (batch, n_full)
                batch_idx = tf.tile(tf.range(batch_size)[:, tf.newaxis], [1, n_full])
                gather_idx = tf.stack([batch_idx, perm_indices], axis=2)  # (batch, n_full, 2)
                permuted = tf.gather_nd(segments, gather_idx)             # (batch, n_full, divisor)

                permuted_flat = tf.reshape(permuted, [batch_size, divisor * n_full])
                return tf.concat([permuted_flat, last], axis=1)

            def return_original():
                return section_data

            return tf.cond(divisor >= 1, permute_segments, return_original)

        permuted_hold = permute_section(hold_time_data)
        permuted_flight = permute_section(flight_time_data)
        return tf.concat([permuted_hold, permuted_flight], axis=1)

    def normalize_histogram(self, data):
        # Histograms are non-negative; clip noise-induced negative bins so the
        # per-half sums cannot go near zero/negative and blow up the division.
        data = tf.nn.relu(data)
        hold_time_data = data[:, :101]
        flight_time_data = data[:, 101:]
        hold_sum = tf.maximum(tf.reduce_sum(hold_time_data, axis=1, keepdims=True), 1e-8)
        flight_sum = tf.maximum(tf.reduce_sum(flight_time_data, axis=1, keepdims=True), 1e-8)
        return tf.concat([hold_time_data / hold_sum, flight_time_data / flight_sum], axis=1)

    def get_contrastive_augmenter(self):
        return keras.Sequential(
            [
                layers.Lambda(self.add_noise),
                layers.Lambda(self.dropout_features),
                layers.Lambda(self.random_scaling),
                layers.Lambda(self.permute_histogram_segments),
                layers.Lambda(self.normalize_histogram),
            ]
        )


# ── Hyperparameters ──────────────────────────────────────────────────────────

K2 = 200           # Typing sessions per subject view (after padding)
M = 64             # Embedding dimension
F = 502            # Histogram feature dimension
batch_size = 16    # Subjects per batch
num_epochs = 300   # disjoint views make the pretext task harder; c_acc was still
                   # rising at epoch 50 — train until it plateaus (~1 s/epoch)
temperature = 0.1  # NT-Xent temperature; 0.07–0.2 is the standard MoCo/SimCLR range.
learning_rate = 3e-4
sample_frac = 0.80  # Fraction of subject sessions to sample per view (only used when
                    # DISJOINT_VIEWS = False)
DISJOINT_VIEWS = True  # If True, the two views are non-overlapping halves of the
                       # subject's sessions. With overlapping subsamples the model can
                       # match views by recognizing identical sessions instead of
                       # learning the subject's underlying timing distribution.
queue_size = 128   # Memory bank size (~60% of the 209 subjects as negatives)
moco_momentum = 0.99
USE_AUGMENTATION = False    # Set to False to disable stochastic augmentations (normalize_histogram always runs)
# Individual augmentations to include (only used when USE_AUGMENTATION = True):
#   "poisson_resample"           — bootstrap bin counts: simulates re-recording the same
#                                  session (noise ∝ sqrt(p/N), preserves support/padding)
#   "add_noise"                  — Gaussian noise on histogram bins (mild, safe)
#   "dropout_features"           — randomly zero out bins (mild, safe)
#   "random_scaling"             — NO-OP here: normalize_histogram divides each half by
#                                  its sum, which exactly cancels a per-sample scale
#   "permute_histogram_segments" — shuffle bin-groups (destroys subject identity — avoid)
ACTIVE_AUGMENTATIONS = {"add_noise", "dropout_features", "permute_histogram_segments", "random_scaling"}  # normalize_histogram always runs last
DEBUG = False
DEBUG_SUBJECTS = 20
USE_TRAINING = True    # If False, skip training and load saved weights for t-SNE visualization

# ── Dataset ──────────────────────────────────────────────────────────────────

DATASET_PATH = "datasets/unlabeled_subject_typing_data.pickle"
GDATA_PATH = os.path.join("..", "data", "typing_gdata.pickle")
SDATA_PATH = os.path.join("..", "data", "typing_sdata.pickle")

if not os.path.exists(DATASET_PATH):
    print(f"'{DATASET_PATH}' not found — building from raw data...")
    with open(GDATA_PATH, "rb") as f:
        typing_gdata = pkl.load(f)
    with open(SDATA_PATH, "rb") as f:
        typing_sdata = pkl.load(f)
    from utils import form_unlabeled_subject_typing_dataset
    form_unlabeled_subject_typing_dataset(typing_gdata, typing_sdata, K2)
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
print(f"Session counts per subject (first 5): {[len(s) for s in subject_data[:5]]}")

# ── View creation ─────────────────────────────────────────────────────────────


def create_subject_view(sessions: np.ndarray, k2: int, frac: float) -> tuple:
    """Sample a random subset of typing sessions and pad to k2.

    Args:
        sessions: Array of shape (N, F) — real sessions for one subject.
        k2:       Target number of sessions after padding.
        frac:     Fraction of sessions to sample.

    Returns:
        view:  np.ndarray of shape (k2, F)
        mask:  np.ndarray of shape (k2,) — True for real sessions, False for padding
    """
    n = len(sessions)
    n_sample = max(1, int(n * frac))
    idx = np.random.choice(n, size=n_sample, replace=False)
    return _pad_view(sessions[idx], k2)


def create_subject_view_pair(sessions: np.ndarray, k2: int) -> tuple:
    """Split a subject's sessions into two disjoint halves (random partition).

    The views share no sessions, so matching them requires capturing the
    subject's session-invariant typing characteristics.

    Returns:
        (view1, mask1), (view2, mask2) — each view shaped (k2, F)
    """
    n = len(sessions)
    perm = np.random.permutation(n)
    half = n // 2
    return _pad_view(sessions[perm[half:]], k2), _pad_view(sessions[perm[:half]], k2)


def _pad_view(sampled: np.ndarray, k2: int) -> tuple:
    n_sample = len(sampled)
    if n_sample >= k2:
        view = sampled[:k2]
        mask = np.ones(k2, dtype=bool)
    else:
        padding = np.zeros((k2 - n_sample, F), dtype=sampled.dtype)
        view = np.concatenate([sampled, padding], axis=0)
        mask = np.array([True] * n_sample + [False] * (k2 - n_sample), dtype=bool)

    return view, mask


# ── Encoder architecture ──────────────────────────────────────────────────────


def build_encoder(m: int) -> keras.Sequential:
    """Dense encoder for typing histograms — identical to typingSimCLR.py's embeddings_function."""
    return keras.Sequential(
        [
            layers.Input(shape=(F,)),
            layers.Dense(100),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.1),
            layers.Dense(50),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Dropout(0.1),
            layers.Dense(m),
        ],
        name="embeddings_function",
    )


# ── Subject-level contrastive model ──────────────────────────────────────────


class SubjectContrastiveModel(keras.Model):
    """Encoder + attention trained with subject-level NT-Xent contrastive loss."""

    def __init__(self, m: int, k2: int, f: int, temp: float, queue_size: int = 64, moco_momentum: float = 0.99):
        super().__init__()
        self.m = m
        self.k2 = k2
        self.f = f
        self.temperature = temp
        self.queue_size = queue_size
        self.moco_momentum = moco_momentum

        self.encoder = build_encoder(m)

        self.attention_layer = MILAttentionLayer(
            weight_params_dim=16,
            kernel_regularizer=keras.regularizers.L2(0.01),
            use_gated=True,
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

        # ── EMA key encoder ───────────────────────────────────────────────────
        self.key_encoder = build_encoder(m)
        self.key_attention_layer = MILAttentionLayer(
            weight_params_dim=16,
            kernel_regularizer=keras.regularizers.L2(0.01),
            use_gated=True,
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
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(name="c_acc")

        # Augmenter: always normalize_histogram last; stochastic steps gated by USE_AUGMENTATION
        _taug = TypingAugmentation()
        _aug_steps = []
        if USE_AUGMENTATION:
            if "poisson_resample"           in ACTIVE_AUGMENTATIONS: _aug_steps.append(layers.Lambda(_taug.poisson_resample))
            if "add_noise"                  in ACTIVE_AUGMENTATIONS: _aug_steps.append(layers.Lambda(_taug.add_noise))
            if "dropout_features"           in ACTIVE_AUGMENTATIONS: _aug_steps.append(layers.Lambda(_taug.dropout_features))
            if "random_scaling"             in ACTIVE_AUGMENTATIONS: _aug_steps.append(layers.Lambda(_taug.random_scaling))
            if "permute_histogram_segments" in ACTIVE_AUGMENTATIONS: _aug_steps.append(layers.Lambda(_taug.permute_histogram_segments))
        _aug_steps.append(layers.Lambda(_taug.normalize_histogram))  # always last
        self.augmenter = keras.Sequential(_aug_steps)

        # ── Memory bank ───────────────────────────────────────────────────────
        init_queue = tf.math.l2_normalize(
            tf.random.normal((queue_size, m), dtype=tf.float32), axis=1
        )
        self.queue = tf.Variable(
            init_queue, trainable=False, dtype=tf.float32, name="queue"
        )
        # Subject index of each queue entry (-1 = random init, never matches a
        # real subject). Used to mask out false negatives in the loss.
        self.queue_labels = tf.Variable(
            tf.fill((queue_size,), -1), trainable=False, dtype=tf.int32, name="queue_labels"
        )
        self.queue_ptr = tf.Variable(0, trainable=False, dtype=tf.int32, name="queue_ptr")

    # ── Forward helpers ───────────────────────────────────────────────────────

    def _build_mask(self, sessions_batch: tf.Tensor) -> tf.Tensor:
        """Detect zero-padded sessions.

        Args:
            sessions_batch: (batch, k2, F)

        Returns:
            mask: (batch, k2, 1) — 0.0 for real sessions, -inf for padding
        """
        summed = tf.reduce_sum(tf.abs(sessions_batch), axis=2, keepdims=True)  # (batch, k2, 1)
        mask = tf.where(summed < 1e-3, -np.inf, 0.0)
        return mask

    def _encode_subjects(self, sessions_batch: tf.Tensor, training: bool, mask=None) -> tf.Tensor:
        """Produce subject-level embeddings.

        Args:
            sessions_batch: (batch, k2, F)
            mask:           Optional pre-computed mask (batch, k2, 1).

        Returns:
            subject_embs: (batch, m)
        """
        flat = tf.reshape(sessions_batch, (-1, self.f))           # (batch*k2, F)
        window_embs = self.encoder(flat, training=training)       # (batch*k2, m)
        window_embs = tf.reshape(window_embs, (-1, self.k2, self.m))  # (batch, k2, m)

        if mask is None:
            mask = self._build_mask(sessions_batch)  # (batch, k2, 1)

        alpha = self.attention_layer(window_embs, mask)  # (batch, k2, 1)
        return tf.reduce_sum(alpha * window_embs, axis=1)  # (batch, m)

    def _encode_subjects_key(self, sessions_batch: tf.Tensor, mask=None) -> tf.Tensor:
        flat = tf.reshape(sessions_batch, (-1, self.f))
        window_embs = self.key_encoder(flat, training=False)
        window_embs = tf.reshape(window_embs, (-1, self.k2, self.m))
        if mask is None:
            mask = self._build_mask(sessions_batch)
        alpha = self.key_attention_layer(window_embs, mask)
        return tf.reduce_sum(alpha * window_embs, axis=1)

    def initialize_key_encoder(self):
        """Copy query encoder weights → key encoder. Called once before training."""
        for q_w, k_w in zip(self.encoder.weights, self.key_encoder.weights):
            k_w.assign(tf.cast(q_w, k_w.dtype))
        for q_w, k_w in zip(self.attention_layer.weights, self.key_attention_layer.weights):
            k_w.assign(tf.cast(q_w, k_w.dtype))
        for q_w, k_w in zip(self.projection_head.weights, self.key_projection_head.weights):
            k_w.assign(tf.cast(q_w, k_w.dtype))
        print("Key encoder initialized from query encoder weights.")

    def _momentum_update(self):
        """EMA update: θ_k ← moco_momentum·θ_k + (1−moco_momentum)·θ_q"""
        m = self.moco_momentum
        for q_w, k_w in zip(self.encoder.weights, self.key_encoder.weights):
            k_w.assign(m * k_w + (1.0 - m) * tf.cast(q_w, k_w.dtype))
        for q_w, k_w in zip(self.attention_layer.weights, self.key_attention_layer.weights):
            k_w.assign(m * k_w + (1.0 - m) * tf.cast(q_w, k_w.dtype))
        for q_w, k_w in zip(self.projection_head.weights, self.key_projection_head.weights):
            k_w.assign(m * k_w + (1.0 - m) * tf.cast(q_w, k_w.dtype))

    # ── NT-Xent loss ──────────────────────────────────────────────────────────

    def contrastive_loss(
        self,
        proj1_q: tf.Tensor,
        proj2_k: tf.Tensor,
        queue_snapshot: tf.Tensor,
        queue_labels: tf.Tensor,
        subject_ids: tf.Tensor,
    ) -> tf.Tensor:
        proj1_q = tf.nn.l2_normalize(tf.cast(proj1_q, tf.float32), axis=1)
        proj2_k = tf.nn.l2_normalize(tf.cast(proj2_k, tf.float32), axis=1)

        batch = ops.shape(proj1_q)[0]
        labels = ops.arange(batch)

        keys = tf.concat([proj2_k, queue_snapshot], axis=0)  # (B+Q, M)
        similarities = ops.matmul(proj1_q, ops.transpose(keys)) / self.temperature

        # Mask out queue entries from the same subject as the query (false
        # negatives that survive in the queue across epoch boundaries).
        collisions = tf.equal(
            tf.expand_dims(subject_ids, 1), tf.expand_dims(queue_labels, 0)
        )  # (B, Q)
        queue_penalty = tf.where(collisions, -1e9, 0.0)
        penalty = tf.concat([tf.zeros((batch, batch)), queue_penalty], axis=1)
        similarities = similarities + penalty

        self.contrastive_accuracy.update_state(labels, similarities)
        return keras.losses.sparse_categorical_crossentropy(
            labels, similarities, from_logits=True
        )

    # ── Memory bank update ────────────────────────────────────────────────────

    @tf.function
    def _dequeue_and_enqueue(self, keys: tf.Tensor, subject_ids: tf.Tensor):
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
        self.queue_labels.assign(
            tf.tensor_scatter_nd_update(
                self.queue_labels, tf.expand_dims(indices, axis=1), subject_ids
            )
        )
        self.queue_ptr.assign(tf.math.mod(ptr + batch_size, self.queue_size))

    # ── Train step ────────────────────────────────────────────────────────────

    @tf.function
    def train_step_contrastive(
        self,
        views1: tf.Tensor,
        views2: tf.Tensor,
        optimizer: keras.optimizers.Optimizer,
        queue_snapshot: tf.Tensor,
        queue_labels: tf.Tensor,
        subject_ids: tf.Tensor,
    ) -> tuple:
        # Masks from original views (before augmentation)
        mask1 = self._build_mask(views1)
        mask2 = self._build_mask(views2)

        # Augment both views: flatten sessions → augment → reshape back
        orig_shape = tf.shape(views1)
        flat1 = tf.reshape(views1, (-1, self.f))
        flat2 = tf.reshape(views2, (-1, self.f))
        flat1 = tf.cast(self.augmenter(flat1, training=True), tf.float32)
        flat2 = tf.cast(self.augmenter(flat2, training=True), tf.float32)
        views1_aug = tf.reshape(flat1, orig_shape)
        views2_aug = tf.reshape(flat2, orig_shape)

        # Key path (no gradients)
        embs2_k = self._encode_subjects_key(views2_aug, mask=mask2)
        proj2_k = tf.stop_gradient(self.key_projection_head(embs2_k, training=False))

        # Query path (with gradients)
        with tf.GradientTape() as tape:
            embs1_q = self._encode_subjects(views1_aug, training=True, mask=mask1)
            proj1_q = self.projection_head(embs1_q, training=True)
            loss = tf.reduce_mean(
                self.contrastive_loss(
                    proj1_q, proj2_k, queue_snapshot, queue_labels, subject_ids
                )
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
    k2: int,
    f: int,
    frac: float,
    out_queue: queue.Queue,
):
    for step in range(steps):
        batch_idx = perm[step * batch_size : (step + 1) * batch_size]
        if len(batch_idx) < 2:
            out_queue.put(None)
            continue
        views1, views2 = build_batch(subject_data, batch_idx, k2, f, frac)
        out_queue.put((views1, views2, batch_idx.astype(np.int32)))


def build_batch(
    subject_data: list, batch_indices: np.ndarray, k2: int, f: int, frac: float
) -> tuple:
    b = len(batch_indices)
    views1 = np.zeros((b, k2, f), dtype=np.float32)
    views2 = np.zeros((b, k2, f), dtype=np.float32)
    for i, idx in enumerate(batch_indices):
        sessions = subject_data[idx].astype(np.float32)
        if DISJOINT_VIEWS:
            (v1, _), (v2, _) = create_subject_view_pair(sessions, k2)
        else:
            v1, _ = create_subject_view(sessions, k2, frac)
            v2, _ = create_subject_view(sessions, k2, frac)
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

        prefetch_q = queue.Queue(maxsize=2)
        prefetch_thread = threading.Thread(
            target=_prefetch_worker,
            args=(subject_data, perm, steps_per_epoch, batch_size, K2, F, sample_frac, prefetch_q),
            daemon=True,
        )
        prefetch_thread.start()

        for step in range(steps_per_epoch):
            batch = prefetch_q.get()
            if batch is None:
                continue

            v1_t = tf.constant(batch[0])
            v2_t = tf.constant(batch[1])
            sid_t = tf.constant(batch[2])

            metrics, proj2_k_norm = model.train_step_contrastive(
                v1_t, v2_t, optimizer, model.queue, model.queue_labels, sid_t
            )
            model._momentum_update()
            model._dequeue_and_enqueue(proj2_k_norm, sid_t)

            progbar.update(
                step + 1,
                values=[
                    ("c_loss", float(metrics["c_loss"])),
                    ("c_acc", float(metrics["c_acc"])),
                ],
            )

        prefetch_thread.join()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start = time.time()

    model = SubjectContrastiveModel(
        m=M, k2=K2, f=F, temp=temperature, queue_size=queue_size, moco_momentum=moco_momentum,
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Build both encoder pairs with dummy forward passes
    dummy = np.zeros((2, K2, F), dtype=np.float32)
    dummy_t = tf.constant(dummy)
    _ = model._encode_subjects(dummy_t, training=False)
    _ = model._encode_subjects_key(dummy_t)
    _ = model.projection_head(tf.zeros((2, M), dtype=tf.float32))
    _ = model.key_projection_head(tf.zeros((2, M), dtype=tf.float32))

    encoder_path = "weights/typing/typing_subject_simclr_embeddings.weights.h5"
    attention_path = "weights/typing/typing_subject_simclr_attention.weights.pkl"

    if USE_TRAINING:
        model.initialize_key_encoder()
        model.encoder.summary()
        model.projection_head.summary()

        train(model, optimizer)

        os.makedirs("weights/typing", exist_ok=True)
        model.encoder.save_weights(encoder_path)
        with open(attention_path, "wb") as f:
            pkl.dump(model.attention_layer.get_weights(), f)

        print(f"\nEncoder weights saved to '{encoder_path}'")
        print(f"Attention weights saved to '{attention_path}'")
        print(f"Total training time: {(time.time() - start) / 60:.1f} min")
    else:
        model.encoder.load_weights(encoder_path)
        with open(attention_path, "rb") as f:
            model.attention_layer.set_weights(pkl.load(f))
        print(f"Loaded encoder weights from '{encoder_path}'")
        print(f"Loaded attention weights from '{attention_path}'")

    # ── t-SNE visualization ───────────────────────────────────────────────────
    print("\nGenerating t-SNE subject-level embeddings from labeled dataset...")
    with open("datasets/typing_sdataset.pickle", "rb") as f:
        labeled_df = pkl.load(f)

    labels_np = np.array(labeled_df["y"].tolist())

    all_bags = labeled_df["X"].tolist()
    bags_padded = np.zeros((len(all_bags), K2, F), dtype=np.float32)
    for i, bag in enumerate(all_bags):
        bag = np.array(bag, dtype=np.float32)
        n = min(len(bag), K2)
        bags_padded[i, :n] = bag[:n]

    # Encode subjects in small batches to avoid OOM
    infer_batch = 4
    subject_embs_list = []
    for i in range(0, len(bags_padded), infer_batch):
        chunk = tf.constant(bags_padded[i : i + infer_batch])
        embs = model._encode_subjects(chunk, training=False)
        subject_embs_list.append(embs.numpy())
    embeddings = np.concatenate(subject_embs_list, axis=0).astype(np.float32)
    print(f"Subject embeddings shape: {embeddings.shape}")

    # ── Quantitative separability (the real diagnostic) ───────────────────────
    # t-SNE coloring is NOT a reliable measure of whether the representation is
    # useful: with only ~25 points it cannot use a meaningful perplexity, and it
    # is non-linear so it can hide a cleanly linearly-separable space. Measure
    # separability directly with a leave-one-subject-out linear probe (AUC).
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    y_probe = labels_np.astype(int)
    loo = LeaveOneOut()
    probe_prob = np.zeros(len(y_probe))
    for tr, te in loo.split(embeddings):
        sc = StandardScaler().fit(embeddings[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(embeddings[tr]), y_probe[tr])
        probe_prob[te] = clf.predict_proba(sc.transform(embeddings[te]))[:, 1]
    probe_auc = roc_auc_score(y_probe, probe_prob)
    probe_acc = ((probe_prob > 0.5).astype(int) == y_probe).mean()
    print(f"LOSO linear-probe on frozen embeddings: AUC={probe_auc:.3f}  acc={probe_acc:.3f}")
    print("  (this is the upper bound a FROZEN encoder offers downstream; if it is")
    print("   high but end-to-end finetuning is not, finetuning is destroying the rep.)")

    # ── t-SNE visualization (qualitative only) ────────────────────────────────
    # Perplexity must be << n_samples or t-SNE degenerates to uniform scatter.
    embeddings_std = StandardScaler().fit_transform(embeddings)
    perplexity = max(2, min(5, len(embeddings) // 3))
    reduced = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(embeddings_std)

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[labels_np == 0, 0], reduced[labels_np == 0, 1], label="No FMI", c="b", alpha=0.5)
    plt.scatter(reduced[labels_np == 1, 0], reduced[labels_np == 1, 1], label="FMI", c="r", alpha=0.5)
    plt.title(f"t-SNE — Subject Typing SimCLR Embeddings (perplexity={perplexity}, probe AUC={probe_auc:.2f})")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/typing_subject_simclr_tsne.png", dpi=150)
    plt.show()
    print("t-SNE plot saved to 'plots/typing_subject_simclr_tsne.png'")
