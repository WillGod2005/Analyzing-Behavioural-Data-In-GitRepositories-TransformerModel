#!/usr/bin/env python
# coding: utf-8
"""
train.py — Main training loop for the Security Patch Detection system.

This module orchestrates the full training pipeline:

  1. Dataset loading:     Calls extract_dataset() and pad_and_fix() from
                          dataset_utils.py to produce model-ready 3-D arrays.
  2. Cross-validation:    Uses sklearn GroupKFold (k=10) to split data into
                          folds grouped by repository, preventing data leakage
                          from shared repository characteristics.
  3. Model training:      Trains either the Conv1D baseline (Farhi's architecture)
                          or the pre-norm Transformer Encoder on each fold.
  4. Ensemble evaluation: Averages predictions from all k fold models for
                          the final test set evaluation.

Key components:
  - model_selector():       Routes to the correct model factory function.
  - compute_class_weight():  Inverse-frequency class weights for imbalanced data.
  - CosineWarmupSchedule:   Linear warmup (3 epochs) then cosine decay LR schedule.
  - EMACallback:            Exponential Moving Average of weights (decay=0.999).
  - make_mixup_dataset():   Mixup data augmentation via tf.data pipeline.
  - train_model():          Full training function with callbacks and augmentation.
  - check_results():        Threshold-sweep evaluation and ROC curve plotting.
  - main():                 CLI entry point with GroupKFold cross-validation.

Command-line usage examples:
  # Train transformer with center pooling, symmetric window (50 events each side)
  python train.py -m transformer -a before -b 50 --metadata --pooling center --seed 42

  # Train Conv1D baseline with same window configuration
  python train.py -m conv1d -a before -b 50 --metadata --seed 42

  # Train transformer with causal window and last-token pooling
  python train.py -m transformer -a only_before -b 50 --metadata --pooling last --seed 42

Used by: run_comparison.sh, run_comparison_extra_seeds.sh (batch experiment scripts)
"""

import os
import sys
import logging
import math
import random

import coloredlogs
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from matplotlib import pyplot
from sklearn.metrics import roc_curve, auc, confusion_matrix

import models
from dataset_utils import (
    Aggregate,
    extract_dataset,
    pad_and_fix,
    split_repos,
    split_into_x_and_y,
)
from helper import find_best_accuracy, find_best_f1, EnumAction, safe_mkdir

logger = logging.getLogger(__name__)
coloredlogs.install(fmt="%(asctime)s %(levelname)s %(message)s")


# =============================================================================
# Model selection
# =============================================================================

def model_selector(model_name, shape1, shape2, optimizer, pooling=None, ff_dim=None,
                    d_model=None, num_layers=None, columns=None, temporal_focus=False):
    """Instantiate a Keras model by name with appropriate hyperparameters.

    Routes to the correct model factory function in models.py.  The transformer
    has its own internal AdamW optimizer with weight decay, so the generic Adam
    optimizer is only passed to non-transformer models (Conv1D, LSTM, etc.).

    Args:
        model_name:     String name matching a function in models.py (e.g.
                        "conv1d", "transformer", "lstm", "gru").
        shape1:         Sequence length (number of timesteps in the window).
        shape2:         Feature dimensionality (number of features per timestep).
        optimizer:      Keras optimizer instance (used by non-transformer models).
        pooling:        Transformer pooling strategy: "center", "last", or "avg".
        ff_dim:         Transformer feed-forward hidden dimension (default 256).
        d_model:        Transformer model/embedding dimension (default 64).
        num_layers:     Number of transformer encoder layers (default 2).
        columns:        Feature column names (needed for two-stream/temporal-focus
                        architectures to identify which features are temporal).
        temporal_focus: If True, use temporal-focus architecture where transformer
                        processes only temporal features and static features are
                        concatenated after pooling.

    Returns:
        Compiled Keras Model ready for .fit().
    """
    model_fn = getattr(models, model_name)
    if model_name == "transformer":
        # Build kwargs dict, only including non-None parameters
        kwargs = {}
        if pooling is not None:
            kwargs["pooling"] = pooling
        if ff_dim is not None:
            kwargs["ff_dim"] = ff_dim
        if d_model is not None:
            kwargs["d_model"] = d_model
        if num_layers is not None:
            kwargs["num_layers"] = num_layers
        if columns is not None:
            kwargs["columns"] = list(columns)
        if temporal_focus:
            kwargs["temporal_focus"] = True
        # Transformer uses its own AdamW by default — don't pass the generic Adam
        return model_fn(shape1, shape2, **kwargs)
    # Non-transformer models (conv1d, lstm, gru, etc.) use the passed optimizer
    return model_fn(shape1, shape2, optimizer)


# =============================================================================
# Class weight computation
# =============================================================================

def compute_class_weight(y_train):
    """Compute class weights inversely proportional to class frequency.

    Used to handle class imbalance in the training data.  With a 1:1
    benign:vuln ratio per repo, the overall dataset may still be slightly
    imbalanced due to varying repo sizes.

    Formula: weight_c = n_total / (2 x n_c)
    This gives higher weight to the minority class, making the loss function
    penalise misclassifications of rare classes more heavily.

    Note: Only used for non-transformer models.  The transformer uses plain
    binary cross-entropy without class weights, relying instead on the
    balanced sampling strategy and regularisation techniques.

    Args:
        y_train: Array of binary labels (0 = benign, 1 = vuln).

    Returns:
        Dict {0: weight_for_0, 1: weight_for_1} or None if a class is missing.
    """
    n_samples = len(y_train)
    n_positive = np.sum(y_train)
    n_negative = n_samples - n_positive
    if n_positive == 0 or n_negative == 0:
        return None
    weight_for_0 = n_samples / (2.0 * n_negative)
    weight_for_1 = n_samples / (2.0 * n_positive)
    return {0: weight_for_0, 1: weight_for_1}


# =============================================================================
# CosineWarmupSchedule — Learning rate callback for transformer training
# =============================================================================

class CosineWarmupSchedule(tf.keras.callbacks.Callback):
    """Linear warmup followed by cosine annealing learning rate schedule.

    Phase 1 — Linear warmup (first warmup_steps steps):
        LR linearly increases from min_lr to peak_lr.
        This prevents large gradient updates at the start of training when
        the randomly initialised model produces poor predictions.

    Phase 2 — Cosine decay (remaining steps):
        LR follows a cosine curve from peak_lr down to min_lr.
        This allows fine-grained optimisation near the end of training.

    Default configuration (from train_model):
        warmup_steps = 3 epochs worth of steps
        peak_lr = 1e-4, min_lr = 1e-6

    This schedule is used ONLY for the transformer model.  The Conv1D
    baseline uses ReduceLROnPlateau (halving LR when val_loss plateaus).
    """

    def __init__(self, total_steps, warmup_steps, peak_lr=5e-4, min_lr=1e-6):
        super().__init__()
        self.total_steps = total_steps    # Total training steps across all epochs
        self.warmup_steps = warmup_steps  # Steps for linear warmup phase
        self.peak_lr = peak_lr            # Maximum learning rate after warmup
        self.min_lr = min_lr              # Minimum learning rate at end of cosine decay
        self.step = 0                     # Current step counter

    def on_train_batch_begin(self, batch, logs=None):
        """Update learning rate at the start of each training batch."""
        self.step += 1
        if self.step <= self.warmup_steps:
            # Phase 1: Linear warmup from min_lr to peak_lr
            lr = self.min_lr + (self.peak_lr - self.min_lr) * (self.step / self.warmup_steps)
        else:
            # Phase 2: Cosine decay from peak_lr to min_lr
            progress = (self.step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        self.model.optimizer.learning_rate.assign(lr)


# =============================================================================
# EMACallback — Exponential Moving Average of model weights
# =============================================================================

class EMACallback(tf.keras.callbacks.Callback):
    """Exponential Moving Average of model weights for smoother convergence.

    Maintains a shadow copy of weights updated as:
        ema_w = decay x ema_w + (1 - decay) x current_w

    This produces a time-averaged version of the model that is more stable
    than any single training checkpoint.  The smoothing effect reduces the
    impact of noisy gradient updates and typically improves generalisation.

    Weight swapping protocol:
      - Before validation: swap in EMA weights (so val metrics reflect smoothed model)
      - After validation:  restore training weights (so SGD continues normally)
      - After training:    call apply_ema() to permanently install smoothed weights

    Decay of 0.999 means each weight update contributes 0.1% to the average,
    effectively averaging over the last ~1000 updates.
    """

    def __init__(self, decay=0.999):
        super().__init__()
        self.decay = decay
        self.ema_weights = None     # Shadow copy: EMA-smoothed weights
        self.train_weights = None   # Backup: current training weights during eval

    def on_train_begin(self, logs=None):
        """Initialise EMA weights from the model's initial (randomly set) weights."""
        self.ema_weights = [w.numpy().copy() for w in self.model.trainable_weights]

    def on_train_batch_end(self, batch, logs=None):
        """Update EMA weights after each batch: ema = decay*ema + (1-decay)*current."""
        for i, w in enumerate(self.model.trainable_weights):
            self.ema_weights[i] = self.decay * self.ema_weights[i] + (1 - self.decay) * w.numpy()

    def on_test_begin(self, logs=None):
        """Swap in EMA weights before validation so metrics reflect smoothed model."""
        self.train_weights = [w.numpy().copy() for w in self.model.trainable_weights]
        for i, w in enumerate(self.model.trainable_weights):
            w.assign(self.ema_weights[i])

    def on_test_end(self, logs=None):
        """Restore training weights after validation so SGD continues normally."""
        if self.train_weights is not None:
            for i, w in enumerate(self.model.trainable_weights):
                w.assign(self.train_weights[i])
            self.train_weights = None

    def apply_ema(self):
        """Permanently apply EMA weights (call after training is done).

        After this call, model.predict() uses the smoothed weights.
        This is better than any single epoch's checkpoint because it
        averages out training noise.
        """
        for i, w in enumerate(self.model.trainable_weights):
            w.assign(self.ema_weights[i])


# =============================================================================
# Mixup data augmentation
# =============================================================================

def make_mixup_dataset(X, y, batch_size, alpha=0.2):
    """Create a tf.data.Dataset that applies Mixup augmentation on-the-fly.

    Mixup (Zhang et al., 2018) linearly interpolates pairs of training
    samples and their labels:
        x_mix = λ x x_a + (1-λ) x x_b
        y_mix = λ x y_a + (1-λ) x y_b
    where λ ~ Beta(alpha, alpha).

    This creates "virtual" training examples between real samples, which:
      - Smooths the decision boundary (reduces sharp transitions)
      - Acts as a form of regularisation (prevents memorising individual samples)
      - Improves calibration of predicted probabilities

    The Beta distribution is implemented via two Gamma samples:
        λ = Gamma(α) / (Gamma(α) + Gamma(α))
    Lambda is clamped to ≥0.5 so the mixed sample stays closer to the
    "primary" sample than the "secondary" one.

    Args:
        X:          Training features array, shape (n, seq_len, features).
        y:          Training labels array, shape (n,).
        batch_size: Batch size for the tf.data pipeline.
        alpha:      Beta distribution parameter (lower = less mixing).

    Returns:
        tf.data.Dataset yielding (x_mixed, y_mixed) batches.
    """
    n = len(X)
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(n, reshuffle_each_iteration=True)

    # Batch before mixing — pairs are formed within each batch
    ds = ds.batch(batch_size, drop_remainder=True)

    def apply_mixup(x_batch, y_batch):
        """Mix pairs of samples within a batch using Beta-distributed weights."""
        batch_sz = tf.shape(x_batch)[0]
        # Sample λ from Beta(α, α) via ratio of two Gamma samples
        lam = tf.random.gamma([batch_sz], alpha=alpha, beta=alpha)
        lam2 = tf.random.gamma([batch_sz], alpha=alpha, beta=alpha)
        lam = lam / (lam + lam2 + 1e-8)  # Normalise to [0, 1]
        lam = tf.maximum(lam, 1.0 - lam)  # Clamp to ≥0.5 (stay closer to original)

        # Create mixing pairs by shuffling within the batch
        indices = tf.random.shuffle(tf.range(batch_sz))
        x_shuffled = tf.gather(x_batch, indices)
        y_shuffled = tf.gather(y_batch, indices)

        # Broadcast lambda for element-wise mixing
        lam_x = tf.reshape(lam, [-1, 1, 1])  # (batch, 1, 1) for 3-D input
        lam_y = tf.reshape(lam, [-1])         # (batch,) for 1-D labels

        # Linearly interpolate features and labels
        x_mixed = lam_x * x_batch + (1.0 - lam_x) * x_shuffled
        y_mixed = lam_y * y_batch + (1.0 - lam_y) * y_shuffled
        return x_mixed, y_mixed

    ds = ds.map(apply_mixup, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# =============================================================================
# train_model — Full training function for a single fold
# =============================================================================

def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    exp_name,
    batch_size=32,
    epochs=20,
    model_name="conv1d",
    columns=None,
    pooling=None,
    ff_dim=None,
    d_model=None,
    num_layers=None,
    use_mixup=False,
    use_ema=False,
    temporal_focus=False,
    use_window_crop=False,
):
    """Train a single model on one cross-validation fold.

    Handles the full training lifecycle for one fold:
      1. Model instantiation via model_selector()
      2. Callback setup (checkpointing, early stopping, LR schedule, EMA)
      3. Optional data augmentation (Mixup or temporal event masking)
      4. Training via model.fit()
      5. Accuracy/loss curve plotting
      6. Return trained model (either best checkpoint or EMA-smoothed)

    Callback configuration differs by model type:
      - Transformer: CosineWarmupSchedule (3-epoch warmup, cosine decay to 1e-6),
                     optional EMACallback, no class weights
      - Conv1D/others: ReduceLROnPlateau (halve LR on val_loss plateau),
                       inverse-frequency class weights

    Both model types use:
      - ModelCheckpoint: Save best model by val_accuracy
      - EarlyStopping: Stop after 8 epochs without val_accuracy improvement

    Args:
        X_train:          Training features, shape (n_train, seq_len, features).
        y_train:          Training labels, shape (n_train,).
        X_val:            Validation features for this fold.
        y_val:            Validation labels for this fold.
        exp_name:         Experiment name (for file paths and plot titles).
        batch_size:       Training batch size (default 32).
        epochs:           Maximum training epochs (default 20, early stopping may cut short).
        model_name:       Model architecture name (e.g. "conv1d", "transformer").
        columns:          Feature column names (for two-stream/temporal-focus).
        pooling:          Transformer pooling strategy.
        ff_dim:           Transformer feed-forward dimension.
        d_model:          Transformer model dimension.
        num_layers:       Number of transformer encoder layers.
        use_mixup:        Enable Mixup data augmentation.
        use_ema:          Enable EMA weight smoothing.
        temporal_focus:   Enable temporal-focus architecture.
        use_window_crop:  Enable random event masking augmentation.

    Returns:
        Trained Keras Model (either best checkpoint or EMA-smoothed).
    """
    if columns is None:
        columns = []

    is_transformer = model_name == "transformer"

    # Create optimizer (only used by non-transformer models)
    optimizer = Adam(learning_rate=0.001)
    model = model_selector(model_name, X_train.shape[1], X_train.shape[2], optimizer,
                            pooling=pooling, ff_dim=ff_dim, d_model=d_model,
                            num_layers=num_layers, columns=columns,
                            temporal_focus=temporal_focus)

    # Create output directories for model checkpoints and training plots
    safe_mkdir("models")
    safe_mkdir(os.path.join("models", exp_name))
    safe_mkdir("figs")

    best_model_path = os.path.join("models", exp_name, f"{model_name}.keras")

    # ---- Callbacks ----

    # Save the best model checkpoint by validation accuracy
    mcp_save = ModelCheckpoint(
        best_model_path,
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
    )

    # Stop training if val_accuracy doesn't improve for 8 consecutive epochs.
    # restore_best_weights=True ensures we keep the best epoch's weights.
    es = EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        mode="max",
        restore_best_weights=True,
    )

    callbacks = [mcp_save, es]

    # Learning rate schedule differs by model type
    if is_transformer:
        # Cosine warmup schedule: 3-epoch linear warmup, then cosine decay
        # peak_lr=1e-4 is conservative for the small ~147K param transformer
        steps_per_epoch = max(1, len(X_train) // batch_size)
        total_steps = steps_per_epoch * epochs
        warmup_steps = steps_per_epoch * 3  # 3 epochs of warmup
        cosine_cb = CosineWarmupSchedule(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            peak_lr=1e-4,
            min_lr=1e-6,
        )
        callbacks.append(cosine_cb)
    else:
        # ReduceLROnPlateau: halve LR when val_loss doesn't improve for 3 epochs
        lr_scheduler = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            mode="min",
        )
        callbacks.append(lr_scheduler)

    # Optional EMA callback (only for transformer)
    ema_cb = None
    if use_ema and is_transformer:
        ema_cb = EMACallback(decay=0.999)
        callbacks.append(ema_cb)

    # Class weights for non-transformer models to handle any class imbalance.
    # Transformer uses plain BCE — it relies on balanced sampling + regularisation.
    class_weight = None if is_transformer else compute_class_weight(y_train)

    verbose = 1 if logger.level < logging.CRITICAL else 0

    # ---- Training with optional augmentation ----

    if use_window_crop and is_transformer:
        # Random event masking: zero out ~15% of non-center timesteps per sample.
        # This acts as temporal dropout — the model learns not to rely on any
        # single event in the window.  The center token (target commit) is
        # always preserved so CenterTokenPooling still works correctly.
        def event_mask_augment(x, y):
            seq_len = tf.shape(x)[0]
            center = seq_len // 2
            # Random mask: keep ~85% of timesteps
            mask = tf.random.uniform([seq_len]) > 0.15
            # Always keep the center token (target commit)
            center_mask = tf.one_hot(center, seq_len, dtype=tf.bool, on_value=True, off_value=False)
            mask = tf.logical_or(mask, center_mask)
            # Zero out masked timesteps (multiply features by 0 or 1)
            x = x * tf.cast(mask, x.dtype)[:, tf.newaxis]
            return x, y

        # Build tf.data pipeline with masking applied per-sample per-epoch
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.shuffle(len(X_train)).map(
            event_mask_augment, num_parallel_calls=tf.data.AUTOTUNE
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        steps_per_epoch_crop = max(1, len(X_train) // batch_size)

        history = model.fit(
            train_ds,
            verbose=verbose,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch_crop,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
        )
    elif use_mixup and is_transformer:
        # Mixup augmentation via tf.data pipeline — creates virtual training
        # examples by linearly interpolating pairs of samples and labels.
        train_ds = make_mixup_dataset(X_train, y_train, batch_size, alpha=0.2)
        steps_per_epoch_actual = len(X_train) // batch_size
        history = model.fit(
            train_ds,
            verbose=verbose,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch_actual,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
        )
    else:
        # Standard training: pass numpy arrays directly to model.fit()
        history = model.fit(
            X_train,
            y_train,
            verbose=verbose,
            epochs=epochs,
            shuffle=True,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weight,
        )

    # ---- Plot training curves ----
    # Save accuracy curves for visual inspection of overfitting/convergence
    pyplot.figure()
    pyplot.plot(history.history["accuracy"])
    pyplot.plot(history.history["val_accuracy"])
    pyplot.title("model accuracy")
    pyplot.ylabel("accuracy")
    pyplot.xlabel("epoch")
    pyplot.legend(["train", "val"], loc="upper left")
    pyplot.savefig(os.path.join("figs", f"{exp_name}_{model_name}_accuracy.png"))
    pyplot.close()

    # ---- Return trained model ----
    if ema_cb is not None:
        # When using EMA, permanently apply smoothed weights and return
        # the in-memory model (better than any single checkpoint)
        ema_cb.apply_ema()
        return model

    # Without EMA, load the best checkpoint from disk.
    # Must pass custom_objects for all custom Keras layers used in the
    # transformer architecture so they can be deserialised correctly.
    return tf.keras.models.load_model(
        best_model_path,
        custom_objects={
            "CenterTokenPooling": models.CenterTokenPooling,
            "LastTokenPooling": models.LastTokenPooling,
            "PaddingMask": models.PaddingMask,
            "ExpandMaskDim": models.ExpandMaskDim,
            "MaskedGlobalAveragePooling": models.MaskedGlobalAveragePooling,
            "AddPositionalEmbedding": models.AddPositionalEmbedding,
            "StochasticDepth": models.StochasticDepth,
            "GatherFeatures": models.GatherFeatures,
            "ExtractStatic": models.ExtractStatic,
        },
    )


# =============================================================================
# check_results — Evaluation and reporting
# =============================================================================

def check_results(X_test, y_test, pred, exp_name, model_name, save=False):
    """Evaluate predictions via threshold sweep and optionally save results.

    Uses find_best_f1() and find_best_accuracy() to sweep 100 thresholds
    (0.00–0.99) and find the optimal decision boundary for each metric.

    When save=True (used for final ensemble evaluation), writes:
      - results/{exp_name}_{model_name}.txt with accuracy, F1, confusion matrix
      - figs/auc_{exp_name}_{model_name}.png with ROC curve

    Args:
        X_test:     Test features (unused, kept for API consistency).
        y_test:     Ground-truth labels.
        pred:       Model predictions (probabilities in [0, 1]).
        exp_name:   Experiment name for file paths.
        model_name: Model name for file paths.
        save:       If True, write results to disk and plot ROC curve.

    Returns:
        Best accuracy found across all thresholds (float).
    """
    used_y_test = np.asarray(y_test).astype("float32")

    # Find best thresholds for F1 and accuracy via exhaustive sweep
    max_f1, f1_thresh = find_best_f1(pred, used_y_test)[:2]
    max_acc, acc_thresh, _ = find_best_accuracy(pred, used_y_test)

    logger.critical(f"F1 - {max_f1}, {f1_thresh}")
    logger.critical(f"Acc - {max_acc}, {acc_thresh}")

    if save:
        # Write detailed results to text file
        safe_mkdir("results")
        with open(os.path.join("results", f"{exp_name}_{model_name}.txt"), "w") as mfile:
            mfile.write("Accuracy: %.2f%%\n" % (max_acc * 100))
            mfile.write("fscore: %.2f%%\n" % (max_f1 * 100))
            mfile.write("confusion matrix:\n")
            # Compute confusion matrix at the best accuracy threshold
            tn, fp, fn, tp = confusion_matrix(y_test, pred > acc_thresh).ravel()
            conf_matrix = f"tn={tn}, fp={fp}, fn={fn}, tp={tp}"
            mfile.write(conf_matrix)

        # Plot ROC curve for visual assessment of discrimination ability
        fpr = {}
        tpr = {}
        fpr["micro"], tpr["micro"], _ = roc_curve(used_y_test, pred)
        roc_auc = {"micro": auc(fpr["micro"], tpr["micro"])}

        pyplot.figure()
        lw = 2
        pyplot.plot(
            fpr["micro"],
            tpr["micro"],
            lw=lw,
            label="ROC curve (area = %0.2f)" % roc_auc["micro"],
        )
        pyplot.plot([0, 1], [0, 1], lw=lw, linestyle="--")  # Random baseline
        pyplot.xlim([0.0, 1.0])
        pyplot.ylim([0.0, 1.05])
        pyplot.xlabel("False Positive Rate")
        pyplot.ylabel("True Positive Rate")
        pyplot.title("Receiver operating characteristic")
        pyplot.legend(loc="lower right")
        pyplot.savefig(os.path.join("figs", f"auc_{exp_name}_{model_name}.png"))
        pyplot.close()

    return max_acc


# =============================================================================
# Command-line argument parsing
# =============================================================================

def parse_args():
    """Parse command-line arguments for the training script.

    Returns:
        argparse.Namespace with all training configuration parameters.

    Key arguments:
        -m/--model:    Model architecture ("conv1d", "transformer", etc.)
        -a/--aggr:     Window extraction mode ("before", "only_before", etc.)
        -b/--backs:    Event window size (number of events)
        -k/--kfold:    Number of cross-validation folds (default 10)
        --metadata:    Include ~388 static metadata features
        --pooling:     Transformer pooling ("center", "last", "avg")
        --seed:        Random seed for reproducibility
        --mixup:       Enable Mixup data augmentation
        --ema:         Enable EMA weight smoothing
        --window-crop: Enable random event masking augmentation
    """
    import argparse
    parser = argparse.ArgumentParser(description="")
    # Dataset parameters
    parser.add_argument("--hours", type=int, default=0, help="hours back")
    parser.add_argument("-d", "--days", type=int, default=10, help="days back")
    parser.add_argument("--resample", type=int, default=24, help="hours for time aggregation")
    parser.add_argument("-r", "--ratio", type=int, default=1, help="benign:vuln ratio")
    parser.add_argument("-a", "--aggr", type=Aggregate, action=EnumAction, default=Aggregate.before_cve)
    parser.add_argument("-b", "--backs", type=int, default=10, help="event window size parameter")
    # Training parameters
    parser.add_argument("-v", "--verbose", action="store_const", dest="loglevel", const=logging.DEBUG)
    parser.add_argument("-c", "--cache", "--cached", action="store_const", dest="cache", const=True)
    parser.add_argument("-e", "--epochs", type=int, default=50)
    parser.add_argument("-m", "--model", action="store", type=str, default="conv1d")
    parser.add_argument("-k", "--kfold", type=int, default=10)
    parser.add_argument("--comment", action="store", type=str, default="")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--metadata", action="store_true")
    parser.add_argument("--data-location", action="store", default="data_collection")
    parser.add_argument("--cache-location", action="store", default="ready_data")
    # Transformer-specific parameters
    parser.add_argument(
        "--pooling",
        action="store",
        type=str,
        default=None,
        choices=["center", "last", "avg"],
        help="transformer pooling strategy: center, last, or avg (default: avg)",
    )
    parser.add_argument(
        "--ff-dim",
        type=int,
        default=None,
        help="transformer feed-forward dimension (default: 256)",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=None,
        help="transformer model dimension (default: 64)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="number of transformer encoder layers (default: 2)",
    )
    parser.add_argument(
        "--two-stream",
        action="store_true",
        help="use two-stream architecture (separate temporal/static branches)",
    )
    # Augmentation and regularisation flags
    parser.add_argument(
        "--mixup",
        action="store_true",
        help="use Mixup data augmentation (interpolates training samples)",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help="use Exponential Moving Average of weights (decay=0.999)",
    )
    parser.add_argument(
        "--temporal-focus",
        action="store_true",
        help="temporal-focus mode: transformer processes only temporal features, static features concatenated after pooling",
    )
    parser.add_argument(
        "--window-crop",
        action="store_true",
        help="augment training data with ±3 position window shifts (7x more training data)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


# =============================================================================
# Initialisation — Seed all random number generators
# =============================================================================

def init(seed=42):
    """Set random seeds for full reproducibility across runs.

    Seeds Python's random, numpy, and TensorFlow RNGs.  Also sets
    TF_DETERMINISTIC_OPS=1 to force deterministic GPU operations where
    possible (some operations may be slower in deterministic mode).

    Args:
        seed: Integer seed value (default 42).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Force deterministic ops where possible (may reduce GPU performance)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


# =============================================================================
# GroupKFold helpers
# =============================================================================

def build_group_labels(repos):
    """Build per-sample group labels (repo index) for GroupKFold.

    GroupKFold requires a group label for each sample indicating which
    repository it belongs to.  This ensures all samples from the same
    repo appear in the same fold, preventing data leakage from shared
    repository characteristics (metadata, event patterns, etc.).

    Args:
        repos: List of Repository objects.

    Returns:
        Numpy array of integer group labels, one per sample.
        All samples from repos[i] get group label i.
    """
    groups = []
    for repo_idx, repo in enumerate(repos):
        # Each repo contributes (n_vuln + n_benign) samples
        n_samples = repo.get_num_of_vuln() + len(repo.benign_lst)
        groups.extend([repo_idx] * n_samples)
    return np.array(groups)


# =============================================================================
# main — Entry point with GroupKFold cross-validation
# =============================================================================

def main():
    """Main training pipeline with GroupKFold cross-validation.

    Execution flow:
      1. Parse CLI arguments and set random seeds
      2. Load/create dataset via extract_dataset()
      3. Pad all repos to global max sequence length
      4. Split into train+val (80%+10% of vulns) and test (10% of vulns)
      5. Run GroupKFold CV on train+val set (k=10 folds by default)
      6. For each fold:
         a. Train model on fold's training split
         b. Evaluate on fold's validation split
         c. Track best model and all fold models
      7. Evaluate on held-out test set:
         a. Single best model (by val accuracy)
         b. Ensemble of all k fold models (averaged predictions)
      8. Save final results and ROC curve
    """
    from sklearn.model_selection import GroupKFold

    args = parse_args()
    logger.level = args.loglevel or logging.CRITICAL
    init(seed=args.seed)

    print("RUNNING FILE =", os.path.abspath(__file__))
    print("ARGV =", sys.argv)
    print("DATA LOCATION =", args.data_location)

    # Include seed in cache comment so different seeds get separate cache dirs.
    # Seed 42 is the default and doesn't add a suffix for backward compatibility.
    comment = args.comment or ""
    if args.seed != 42:
        comment = f"{comment}_s{args.seed}" if comment else f"s{args.seed}"

    # ---- Step 1: Load or create dataset ----
    all_repos, exp_name, columns = extract_dataset(
        aggr_options=args.aggr,
        resample=args.resample,
        benign_vuln_ratio=args.ratio,
        hours=args.hours,
        days=args.days,
        backs=args.backs,
        cache=args.cache,
        metadata=args.metadata,
        comment=comment,
        data_location=args.data_location,
        cache_location=args.cache_location,
    )

    # ---- Step 2: Global padding ----
    # Pad all repos' windows to the same sequence length across the dataset
    all_repos, num_of_vulns = pad_and_fix(all_repos)

    # ---- Step 3: Train/val/test split by vuln count ----
    # 80% train, 10% validation, 10% test (measured by vuln sample count)
    train_size = int(0.8 * num_of_vulns)
    validation_size = int(0.1 * num_of_vulns)

    logger.info(f"Train size: {train_size}")
    logger.info(f"Validation size: {validation_size}")

    # split_repos separates by cumulative vuln count, respecting repo boundaries
    train_and_val_repos, test_repos, _ = split_repos(all_repos, train_size + validation_size)

    # Combine train+val repos into single arrays for GroupKFold
    X_all, y_all = split_into_x_and_y(train_and_val_repos)
    # Build group labels so GroupKFold keeps each repo in one fold only
    groups = build_group_labels(train_and_val_repos)

    # ---- Step 4: GroupKFold cross-validation ----
    # Limit folds to available repos (if fewer repos than k)
    n_splits = min(args.kfold, len(train_and_val_repos))
    gkf = GroupKFold(n_splits=n_splits)

    best_model = None
    best_val_accuracy = 0
    fold_models = []       # All fold models for ensemble
    fold_accuracies = []   # Val accuracy per fold
    fold = 0

    for train_idx, val_idx in gkf.split(X_all, y_all, groups):
        fold += 1
        logger.critical(f"Fold {fold}/{n_splits}")

        # Split into this fold's train and validation sets
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]

        # Train model for this fold
        model = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            exp_name,
            batch_size=args.batch,
            epochs=args.epochs,
            model_name=args.model,
            # Pass column names only if using two-stream or temporal-focus architectures
            columns=columns if (args.two_stream or args.temporal_focus) else None,
            pooling=args.pooling,
            ff_dim=args.ff_dim,
            d_model=args.d_model,
            num_layers=args.num_layers,
            use_mixup=args.mixup,
            use_ema=args.ema,
            temporal_focus=args.temporal_focus,
            use_window_crop=args.window_crop,
        )

        fold_models.append(model)

        # Evaluate this fold's model on its validation set
        pred = model.predict(X_val, verbose=0).reshape(-1)
        acc = check_results(X_val, y_val, pred, exp_name, args.model)
        fold_accuracies.append(acc)

        # Track the best single model across all folds
        if acc > best_val_accuracy:
            best_model = model
            best_val_accuracy = acc

    if best_model is None:
        raise RuntimeError("No model was successfully trained.")

    # ---- Step 5: Final evaluation on held-out test set ----
    X_test, y_test = split_into_x_and_y(test_repos)

    # Ensemble prediction: average predictions from ALL k fold models.
    # This reduces variance and typically outperforms any single model.
    ensemble_pred = np.zeros(len(y_test), dtype=np.float32)
    for m in fold_models:
        ensemble_pred += m.predict(X_test, verbose=0).reshape(-1)
    ensemble_pred /= len(fold_models)

    # Report both single-best and ensemble results
    logging.critical(f"Best val accuracy: {best_val_accuracy}")

    # Single best model evaluation (the model with highest fold val accuracy)
    pred_single = best_model.predict(X_test, verbose=0).reshape(-1)
    acc_single = check_results(X_test, y_test, pred_single, exp_name, args.model)
    logging.critical(f"Single best test accuracy: {acc_single}")

    # Ensemble evaluation (save=True writes results file and ROC plot)
    acc_ensemble = check_results(X_test, y_test, ensemble_pred, exp_name, args.model, save=True)
    logging.critical(f"Ensemble test accuracy ({len(fold_models)} models): {acc_ensemble}")


if __name__ == "__main__":
    main()
