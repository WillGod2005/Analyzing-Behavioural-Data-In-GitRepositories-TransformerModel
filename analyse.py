#!/usr/bin/env python3
"""
analyse.py — Comprehensive dissertation analysis suite.

This module runs all statistical analyses and generates publication-quality
figures for the dissertation.  It is designed to be run after training is
complete (models exist in models/ and cached datasets exist in ready_data/).

Analysis sections (selectable via --sections):

  1. STATISTICAL SIGNIFICANCE (stats):
     - Trains both transformer and conv1d via GroupKFold CV
     - Paired t-test and Wilcoxon signed-rank test per metric (acc, F1, AUC)
     - Bootstrap confidence intervals (10,000 resamples, 95% CI)
     - Cohen's d effect size with magnitude labels
     - Per-fold comparison bar charts and paired difference boxplots
     → Output: figures/statistical_significance.png, figures/paired_differences.png

  2. FEATURE ABLATION (ablation):
     - Full model (all 435 features) vs three ablation conditions:
       a) Without ALL 23 new features (zeroed out, keeping column alignment)
       b) Without 7 GraphQL recency features only
       c) Without 16 derived temporal features only
     - Statistical significance of each ablation via paired t-test
     - Horizontal bar charts comparing all conditions
     → Output: figures/feature_ablation.png

  3. PERMUTATION FEATURE IMPORTANCE (importance):
     - Two-phase approach for efficiency:
       Phase 1: Quick single-repeat scan of all 435 features
       Phase 2: Detailed multi-repeat analysis of top 60 candidates
     - Feature group importance (event types, numeric, GraphQL, temporal, metadata)
     - Highlights which of the 23 new features appear in top-20
     → Output: figures/permutation_importance.png, figures/feature_group_importance.png

  4. ERROR ANALYSIS (error):
     - Confusion matrix heatmap at optimal threshold
     - ROC curve and Precision-Recall curve
     - Prediction confidence distribution by true class
     - Feature profile comparison: TP vs FP vs FN vs TN at center timestep
     → Output: figures/error_analysis.png, figures/prediction_distribution.png

  5. MULTI-SEED ROBUSTNESS (seeds):
     - Trains transformer across 5 seeds (42, 123, 7, 256, 999)
     - Reports mean ± std accuracy and F1 across seeds
     - Bar chart showing per-seed performance
     → Output: figures/multi_seed_robustness.png

  6. WINDOW SIZE COMPARISON (windows):
     - Compares transformer vs conv1d across window sizes B=5,10,20,40
     - Shows how performance scales with context window length
     → Output: figures/window_size_comparison.png

All numerical results are appended to figures/analysis_results.txt.

Usage:
    python analyse.py --sections all                    # Run everything
    python analyse.py --sections stats ablation         # Run specific sections
    python analyse.py --sections importance --backs 20  # Custom window size
    python analyse.py --epochs 30 --kfold 5             # Faster (less accurate)
"""

import os
import sys
import pickle
import logging
import random
import argparse

import numpy as np
import tensorflow as tf
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib

# Use non-interactive backend (no display needed — all figures saved to disk)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import models  # noqa: F401 — registers custom Keras layers for model loading
from train import train_model, build_group_labels, init, check_results
from dataset_utils import (
    Aggregate,
    extract_dataset,
    pad_and_fix,
    split_repos,
    split_into_x_and_y,
)
from helper import find_best_accuracy, find_best_f1

logger = logging.getLogger(__name__)

# Output directories and files
FIGURES_DIR = "figures"
RESULTS_FILE = "analysis_results.txt"

# The 23 features added by this dissertation (7 GraphQL + 16 derived temporal).
# Used to identify which features are "new" in permutation importance plots
# and to compute ablation indices.
NEW_FEATURE_NAMES = [
    # 7 GraphQL recency/churn features (from add_graphql_features in dataset_utils.py)
    "days_since_forks",
    "days_since_issues",
    "days_since_pullRequests",
    "days_since_releases",
    "days_since_stargazers",
    "gql_additions",
    "gql_deletions",
    # 16 derived temporal features (from fix_repo_shape in dataset_utils.py)
    "time_delta_hours",
    "code_churn",
    "add_del_ratio",
    "is_commit",
    "event_entropy",
    "repo_age_days",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "type_changed",
    "cum_commits",
    "rel_position",
    "burst_rate",
    "commit_urgency",
    "churn_diversity",
]


# =============================================================================
# Feature name utilities
# =============================================================================

def load_feature_names():
    """Load feature names from the reconstructed pickle file.

    The feature_names.pkl file contains a list of ~435 feature column names
    in the exact order they appear in the model's input tensor.  This file
    is created during dataset construction and is essential for mapping
    permutation importance scores back to human-readable feature names.

    Returns:
        List of feature name strings.

    Raises:
        FileNotFoundError: If feature_names.pkl doesn't exist.
    """
    path = os.path.join(os.path.dirname(__file__), "feature_names.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "feature_names.pkl not found. Run the column name reconstruction first."
        )
    with open(path, "rb") as f:
        return list(pickle.load(f))


def get_new_feature_indices(feature_names):
    """Return column indices of the 23 features added by this dissertation.

    These indices are used for:
      - Feature ablation: zeroing out specific columns to measure impact
      - Permutation importance: highlighting new features in plots

    Args:
        feature_names: Full list of feature names (from load_feature_names).

    Returns:
        List of integer indices into the feature dimension.
    """
    indices = []
    for name in NEW_FEATURE_NAMES:
        if name in feature_names:
            indices.append(feature_names.index(name))
    return indices


def ensure_dirs():
    """Create the figures output directory if it doesn't exist."""
    os.makedirs(FIGURES_DIR, exist_ok=True)


def write_result(f, text):
    """Write analysis output to both file and stdout for live monitoring.

    Args:
        f:    Open file handle for results file.
        text: String to write.
    """
    print(text)
    f.write(text + "\n")


# =============================================================================
# 1. STATISTICAL SIGNIFICANCE TESTS
# =============================================================================

def run_kfold_collect_metrics(
    X_all,
    y_all,
    groups,
    n_splits,
    model_name,
    pooling,
    columns,
    epochs=50,
    batch_size=32,
    seed=42,
    ablation_indices=None,
):
    """Run K-fold cross-validation and collect per-fold metrics.

    Trains a model on each fold and evaluates on the held-out validation set.
    Returns per-fold accuracy, F1, and AUC scores for statistical analysis.

    For ablation studies, specific feature columns can be zeroed out via
    ablation_indices — the model still sees the full input shape but those
    features carry no signal, measuring how much the model depends on them.

    Args:
        X_all:             Full feature array, shape (n, seq_len, features).
        y_all:             Full label array, shape (n,).
        groups:            Per-sample group labels for GroupKFold.
        n_splits:          Number of cross-validation folds.
        model_name:        Model architecture name ("transformer" or "conv1d").
        pooling:           Transformer pooling strategy (or None for conv1d).
        columns:           Feature column names (for two-stream architectures).
        epochs:            Maximum training epochs per fold.
        batch_size:        Training batch size.
        seed:              Random seed for reproducibility.
        ablation_indices:  Optional list of feature indices to zero out.

    Returns:
        Dict with keys:
          - "acc": numpy array of per-fold accuracies
          - "f1":  numpy array of per-fold F1 scores
          - "auc": numpy array of per-fold ROC-AUC scores
          - "models": list of trained Keras models (one per fold)
    """
    init(seed=seed)
    gkf = GroupKFold(n_splits=n_splits)

    fold_acc = []
    fold_f1 = []
    fold_auc = []
    fold_models = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(X_all, y_all, groups), 1
    ):
        print(f"  Fold {fold_idx}/{n_splits} ({model_name})...")
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]

        # For ablation: zero out specified feature columns in both train and val.
        # The model sees zeros instead of the real values, measuring how much
        # performance drops without those features.
        if ablation_indices is not None:
            X_train = X_train.copy()
            X_val = X_val.copy()
            X_train[:, :, ablation_indices] = 0.0
            X_val[:, :, ablation_indices] = 0.0

        model = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            f"analysis_{model_name}",
            batch_size=batch_size,
            epochs=epochs,
            model_name=model_name,
            pooling=pooling,
            columns=columns if model_name == "transformer" else None,
        )

        # Evaluate this fold's model on validation set
        pred = model.predict(X_val, verbose=0).reshape(-1)
        y_val_flat = np.asarray(y_val).astype("float32")

        # Find best threshold for accuracy and F1 via exhaustive sweep
        acc, _, _ = find_best_accuracy(pred, y_val_flat)
        f1, _, _ = find_best_f1(pred, y_val_flat)

        # Compute ROC-AUC (threshold-independent discrimination measure)
        fpr, tpr, _ = roc_curve(y_val_flat, pred)
        roc_auc_val = auc(fpr, tpr)

        fold_acc.append(acc)
        fold_f1.append(f1)
        fold_auc.append(roc_auc_val)
        fold_models.append(model)

    return {
        "acc": np.array(fold_acc),
        "f1": np.array(fold_f1),
        "auc": np.array(fold_auc),
        "models": fold_models,
    }


def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval for the mean.

    Resamples the data with replacement n_bootstrap times, computes the
    mean of each resample, and returns the percentile-based CI.

    Args:
        data:        1-D array of observed values.
        n_bootstrap: Number of bootstrap resamples (default 10,000).
        ci:          Confidence level (default 0.95 for 95% CI).

    Returns:
        Array of [lower_bound, upper_bound].
    """
    boot_means = np.array(
        [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    )
    alpha = (1 - ci) / 2
    return np.percentile(boot_means, [alpha * 100, (1 - alpha) * 100])


def plot_statistical_tests(transformer_metrics, conv1d_metrics, results_file):
    """Run and plot statistical significance tests comparing transformer vs conv1d.

    For each metric (accuracy, F1, ROC-AUC):
      - Paired t-test: parametric test assuming normal distribution of differences
      - Wilcoxon signed-rank: non-parametric alternative (no normality assumption)
      - Bootstrap CI: 95% confidence intervals for each model and their difference
      - Cohen's d: standardised effect size (small <0.5, medium 0.5-0.8, large ≥0.8)

    Produces two figures:
      1. Per-fold bar chart comparing both models on each metric
      2. Boxplot of paired differences (transformer - conv1d) per fold

    Args:
        transformer_metrics: Dict from run_kfold_collect_metrics (transformer).
        conv1d_metrics:      Dict from run_kfold_collect_metrics (conv1d).
        results_file:        Open file handle for writing numerical results.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["acc", "f1", "auc"]
    titles = ["Accuracy", "F1 Score", "ROC-AUC"]

    write_result(results_file, "\n" + "=" * 70)
    write_result(results_file, "1. STATISTICAL SIGNIFICANCE TESTS")
    write_result(results_file, "=" * 70)

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        t_vals = transformer_metrics[metric]
        c_vals = conv1d_metrics[metric]

        # Paired t-test: tests H0: mean(transformer - conv1d) = 0
        # Appropriate when per-fold differences are approximately normal
        t_stat, t_pval = stats.ttest_rel(t_vals, c_vals)

        # Wilcoxon signed-rank: non-parametric alternative to paired t-test
        # More robust with small sample sizes (k=10 folds)
        try:
            w_stat, w_pval = stats.wilcoxon(t_vals, c_vals)
        except ValueError:
            # Can fail if all differences are zero
            w_stat, w_pval = float("nan"), float("nan")

        # Bootstrap confidence intervals (10,000 resamples)
        t_ci = bootstrap_ci(t_vals)
        c_ci = bootstrap_ci(c_vals)
        diff = t_vals - c_vals
        diff_ci = bootstrap_ci(diff)

        write_result(results_file, f"\n--- {title} ---")
        write_result(
            results_file,
            f"Transformer: {t_vals.mean():.4f} +/- {t_vals.std():.4f}  95% CI [{t_ci[0]:.4f}, {t_ci[1]:.4f}]",
        )
        write_result(
            results_file,
            f"Conv1D:      {c_vals.mean():.4f} +/- {c_vals.std():.4f}  95% CI [{c_ci[0]:.4f}, {c_ci[1]:.4f}]",
        )
        write_result(
            results_file,
            f"Difference:  {diff.mean():.4f} +/- {diff.std():.4f}  95% CI [{diff_ci[0]:.4f}, {diff_ci[1]:.4f}]",
        )
        # Significance stars: * p<0.05, ** p<0.01, *** p<0.001
        write_result(
            results_file,
            f"Paired t-test: t={t_stat:.4f}, p={t_pval:.4f} {'*' if t_pval < 0.05 else ''}{'*' if t_pval < 0.01 else ''}{'*' if t_pval < 0.001 else ''}",
        )
        write_result(
            results_file,
            f"Wilcoxon:      W={w_stat:.4f}, p={w_pval:.4f} {'*' if w_pval < 0.05 else ''}{'*' if w_pval < 0.01 else ''}{'*' if w_pval < 0.001 else ''}",
        )

        # Cohen's d effect size: difference in means divided by pooled std.
        # Interpretation: <0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, ≥0.8 large
        pooled_std = np.sqrt((t_vals.std() ** 2 + c_vals.std() ** 2) / 2)
        cohens_d = diff.mean() / pooled_std if pooled_std > 0 else 0
        d_label = "large" if abs(cohens_d) >= 0.8 else "medium" if abs(cohens_d) >= 0.5 else "small"
        write_result(
            results_file,
            f"Cohen's d:     {cohens_d:.4f} ({d_label} effect)",
        )

        # Plot per-fold comparison bar chart
        ax = axes[idx]
        x = np.arange(len(t_vals))
        width = 0.35
        bars1 = ax.bar(x - width / 2, t_vals, width, label="Transformer", color="#2196F3", alpha=0.8)
        bars2 = ax.bar(x + width / 2, c_vals, width, label="Conv1D", color="#FF9800", alpha=0.8)
        ax.axhline(y=t_vals.mean(), color="#1565C0", linestyle="--", alpha=0.5, label=f"Trans. mean={t_vals.mean():.3f}")
        ax.axhline(y=c_vals.mean(), color="#E65100", linestyle="--", alpha=0.5, label=f"Conv1D mean={c_vals.mean():.3f}")
        ax.set_xlabel("Fold")
        ax.set_ylabel(title)
        ax.set_title(f"{title} per Fold\n(p={t_pval:.4f}, {'sig.' if t_pval < 0.05 else 'n.s.'})")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in x])
        ax.legend(fontsize=7)
        # Set y-axis to zoom in on the range of observed values
        all_vals = np.concatenate([t_vals, c_vals])
        margin = (all_vals.max() - all_vals.min()) * 0.3
        ax.set_ylim(all_vals.min() - margin, min(all_vals.max() + margin, 1.0))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "statistical_significance.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Additional figure: boxplot of paired differences (positive = transformer better)
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_labels = ["Accuracy", "F1 Score", "ROC-AUC"]
    diffs = [transformer_metrics[m] - conv1d_metrics[m] for m in metrics]
    positions = range(len(diffs))

    bp = ax.boxplot(diffs, positions=positions, widths=0.6, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#E3F2FD")
        patch.set_edgecolor("#1565C0")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.7, label="No difference")
    ax.set_xticklabels(metrics_labels)
    ax.set_ylabel("Transformer - Conv1D (per fold)")
    ax.set_title("Paired Differences: Transformer vs Conv1D\n(positive = transformer better)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "paired_differences.png"), dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# 2. FEATURE ABLATION STUDY
# =============================================================================

def plot_feature_ablation(results_all, results_no_new, results_no_graphql, results_no_temporal, results_file):
    """Plot feature ablation study results.

    Compares four configurations:
      1. All features (435) — full model
      2. Without new features (412) — all 23 new features zeroed out
      3. Without GraphQL (428) — only 7 GraphQL features zeroed out
      4. Without temporal (419) — only 16 temporal features zeroed out

    The ablation uses feature zeroing rather than column removal to maintain
    consistent input dimensions across all conditions.

    Args:
        results_all:          Full model metrics (dict from run_kfold_collect_metrics).
        results_no_new:       Metrics without any new features.
        results_no_graphql:   Metrics without GraphQL features.
        results_no_temporal:  Metrics without temporal features.
        results_file:         Open file handle for numerical results.
    """
    write_result(results_file, "\n" + "=" * 70)
    write_result(results_file, "2. FEATURE ABLATION STUDY")
    write_result(results_file, "=" * 70)

    configs = [
        ("All features (435)", results_all),
        ("Without new features (412)", results_no_new),
        ("Without GraphQL (428)", results_no_graphql),
        ("Without temporal (419)", results_no_temporal),
    ]

    # Horizontal bar chart for each metric
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metric_keys = ["acc", "f1", "auc"]
    metric_titles = ["Accuracy", "F1 Score", "ROC-AUC"]

    for ax, metric, title in zip(axes, metric_keys, metric_titles):
        means = [r[metric].mean() for _, r in configs]
        stds = [r[metric].std() for _, r in configs]
        labels = [name for name, _ in configs]

        bars = ax.barh(range(len(configs)), means, xerr=stds, height=0.6,
                       color=["#2196F3", "#FF5722", "#FF9800", "#FFC107"], alpha=0.85,
                       capsize=4)
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(title)
        ax.set_title(f"Feature Ablation: {title}")

        # Add numeric value labels to the right of each bar
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(m + s + 0.002, i, f"{m:.4f}", va="center", fontsize=8)

        all_vals = [m + s for m, s in zip(means, stds)]
        ax.set_xlim(min(means) - max(stds) - 0.02, max(all_vals) + 0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "feature_ablation.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Statistical tests: compare each ablation to the full model
    for name, result in configs:
        t_stat, t_pval = stats.ttest_rel(results_all["acc"], result["acc"])
        write_result(results_file, f"\n{name}:")
        write_result(
            results_file,
            f"  Acc: {result['acc'].mean():.4f} +/- {result['acc'].std():.4f}  "
            f"F1: {result['f1'].mean():.4f} +/- {result['f1'].std():.4f}  "
            f"AUC: {result['auc'].mean():.4f} +/- {result['auc'].std():.4f}",
        )
        if name != configs[0][0]:
            diff = results_all["acc"].mean() - result["acc"].mean()
            write_result(
                results_file,
                f"  vs All features: delta_acc={diff:+.4f}, t={t_stat:.3f}, p={t_pval:.4f}",
            )


# =============================================================================
# 3. PERMUTATION FEATURE IMPORTANCE
# =============================================================================

def permutation_importance_analysis(
    models_list, X_test, y_test, feature_names, results_file, n_repeats=10, top_k=30
):
    """Compute permutation importance and plot top-k features.

    Permutation importance measures how much accuracy drops when a single
    feature's values are randomly shuffled across samples.  A large drop
    means the model relies heavily on that feature.

    Two-phase approach for efficiency (435 features × ensemble inference):
      Phase 1: Quick scan — 1 repeat per feature to identify candidates.
               This takes ~435 forward passes (manageable).
      Phase 2: Detailed — n_repeats per top-60 feature for stable estimates.
               This takes ~60 × n_repeats forward passes.

    Without this two-phase approach, 435 × 10 repeats = 4,350 forward passes
    would be needed, which is prohibitively slow with ensemble inference.

    Args:
        models_list:   List of trained Keras models (ensemble).
        X_test:        Test features, shape (n, seq_len, features).
        y_test:        Test labels, shape (n,).
        feature_names: List of feature name strings.
        results_file:  Open file handle for numerical results.
        n_repeats:     Repeats for Phase 2 (default 10).
        top_k:         Number of top features to display (default 30).

    Returns:
        Array of importance scores for all features.
    """
    write_result(results_file, "\n" + "=" * 70)
    write_result(results_file, "3. PERMUTATION FEATURE IMPORTANCE")
    write_result(results_file, "=" * 70)

    y_test_flat = np.asarray(y_test).astype("float32")

    # Compute baseline ensemble accuracy (unpermuted data)
    ensemble_pred = np.zeros(len(y_test), dtype=np.float32)
    for m in models_list:
        ensemble_pred += m.predict(X_test, verbose=0).reshape(-1)
    ensemble_pred /= len(models_list)
    baseline_acc, _, _ = find_best_accuracy(ensemble_pred, y_test_flat)

    n_features = X_test.shape[2]
    importance_scores = np.zeros(n_features)

    print(f"Computing permutation importance for {n_features} features ({n_repeats} repeats)...")

    # Phase 1: Quick single-repeat scan to identify candidate features
    print("  Phase 1: quick scan (1 repeat)...")
    quick_scores = np.zeros(n_features)
    for feat_idx in range(n_features):
        if feat_idx % 100 == 0:
            print(f"    Feature {feat_idx}/{n_features}...")

        # Permute one feature column across all samples
        X_permuted = X_test.copy()
        perm_idx = np.random.permutation(X_permuted.shape[0])
        X_permuted[:, :, feat_idx] = X_permuted[perm_idx, :, feat_idx]

        # Compute ensemble accuracy on permuted data
        perm_pred = np.zeros(len(y_test), dtype=np.float32)
        for m in models_list:
            perm_pred += m.predict(X_permuted, verbose=0).reshape(-1)
        perm_pred /= len(models_list)
        perm_acc, _, _ = find_best_accuracy(perm_pred, y_test_flat)

        # Importance = accuracy drop when feature is shuffled
        quick_scores[feat_idx] = baseline_acc - perm_acc

    # Phase 2: Detailed multi-repeat analysis for top 60 candidates
    top_candidates = np.argsort(quick_scores)[::-1][:60]
    print(f"  Phase 2: detailed ({n_repeats} repeats) for top {len(top_candidates)} features...")

    for feat_idx in top_candidates:
        drop_scores = []
        for _ in range(n_repeats):
            X_permuted = X_test.copy()
            perm_idx = np.random.permutation(X_permuted.shape[0])
            X_permuted[:, :, feat_idx] = X_permuted[perm_idx, :, feat_idx]

            perm_pred = np.zeros(len(y_test), dtype=np.float32)
            for m in models_list:
                perm_pred += m.predict(X_permuted, verbose=0).reshape(-1)
            perm_pred /= len(models_list)
            perm_acc, _, _ = find_best_accuracy(perm_pred, y_test_flat)
            drop_scores.append(baseline_acc - perm_acc)

        importance_scores[feat_idx] = np.mean(drop_scores)

    # For non-top features, use the Phase 1 quick scan value
    for feat_idx in range(n_features):
        if feat_idx not in top_candidates:
            importance_scores[feat_idx] = quick_scores[feat_idx]

    # Sort by importance and report
    sorted_idx = np.argsort(importance_scores)[::-1]

    write_result(results_file, f"\nBaseline ensemble accuracy: {baseline_acc:.4f}")
    write_result(results_file, f"\nTop {top_k} features by permutation importance:")
    write_result(results_file, f"{'Rank':<6}{'Feature':<35}{'Importance':>12}{'Std':>10}")
    write_result(results_file, "-" * 63)

    for rank, idx in enumerate(sorted_idx[:top_k], 1):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        is_new = " [NEW]" if name in NEW_FEATURE_NAMES else ""
        write_result(
            results_file,
            f"{rank:<6}{name + is_new:<35}{importance_scores[idx]:>12.6f}",
        )

    # Count how many new features appear in the top 20
    top20_names = [feature_names[i] for i in sorted_idx[:20]]
    n_new_in_top20 = sum(1 for n in top20_names if n in NEW_FEATURE_NAMES)
    write_result(
        results_file,
        f"\nNew features in top 20: {n_new_in_top20}/20 ({n_new_in_top20/20*100:.0f}%)",
    )

    # Plot: horizontal bar chart with new features highlighted in blue
    fig, ax = plt.subplots(figsize=(10, 8))
    top_indices = sorted_idx[:top_k]
    top_names = [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in top_indices]
    top_scores = importance_scores[top_indices]

    # Color coding: blue = new features (this work), grey = original features (Farhi)
    colors = ["#2196F3" if name in NEW_FEATURE_NAMES else "#90A4AE" for name in top_names]

    bars = ax.barh(range(top_k), top_scores[::-1], color=colors[::-1], alpha=0.85)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel("Accuracy drop when feature is permuted")
    ax.set_title(f"Top {top_k} Features by Permutation Importance\n(blue = new features added by this work)")
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)

    # Legend to distinguish new vs original features
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", alpha=0.85, label="New features (this work)"),
        Patch(facecolor="#90A4AE", alpha=0.85, label="Original features (Farhi et al.)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "permutation_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Feature group importance: aggregate individual feature importance by group
    write_result(results_file, "\nFeature group importance (sum of individual importances):")
    groups = {
        "Event type one-hots": [i for i, n in enumerate(feature_names) if n in [
            "PullRequestEvent", "PushEvent", "ReleaseEvent", "DeleteEvent",
            "issues", "CreateEvent", "releases", "IssuesEvent", "ForkEvent",
            "WatchEvent", "PullRequestReviewCommentEvent", "stargazers",
            "pullRequests", "commits", "CommitCommentEvent", "MemberEvent",
            "GollumEvent", "IssueCommentEvent", "forks", "PullRequestReviewEvent",
            "PublicEvent",
        ]],
        "Numeric (Add/Del/Files)": [i for i, n in enumerate(feature_names) if n in ["Add", "Del", "Files"]],
        "GraphQL recency": [i for i, n in enumerate(feature_names) if n.startswith("days_since_") or n.startswith("gql_")],
        "Derived temporal": [i for i, n in enumerate(feature_names) if n in [
            "time_delta_hours", "code_churn", "add_del_ratio", "is_commit",
            "event_entropy", "repo_age_days", "hour_sin", "hour_cos",
            "dow_sin", "dow_cos", "type_changed", "cum_commits",
            "rel_position", "burst_rate", "commit_urgency", "churn_diversity",
        ]],
        "Repository metadata": [i for i, n in enumerate(feature_names) if i not in
            sum([[i for i, n2 in enumerate(feature_names) if n2 in NEW_FEATURE_NAMES],
                 [i for i, n2 in enumerate(feature_names) if n2 in ["Add", "Del", "Files"]],
                 [i for i, n2 in enumerate(feature_names) if n2 in [
                     "PullRequestEvent", "PushEvent", "ReleaseEvent", "DeleteEvent",
                     "issues", "CreateEvent", "releases", "IssuesEvent", "ForkEvent",
                     "WatchEvent", "PullRequestReviewCommentEvent", "stargazers",
                     "pullRequests", "commits", "CommitCommentEvent", "MemberEvent",
                     "GollumEvent", "IssueCommentEvent", "forks", "PullRequestReviewEvent",
                     "PublicEvent",
                 ]]], [])
        ],
    }

    group_importance = {}
    for group_name, indices in groups.items():
        total_imp = sum(importance_scores[i] for i in indices if i < len(importance_scores))
        mean_imp = total_imp / max(len(indices), 1)
        group_importance[group_name] = (total_imp, mean_imp, len(indices))

    write_result(results_file, f"{'Group':<30}{'Total':>10}{'Mean':>10}{'Count':>8}")
    write_result(results_file, "-" * 58)
    for name, (total, mean, count) in sorted(group_importance.items(), key=lambda x: -x[1][0]):
        write_result(results_file, f"{name:<30}{total:>10.6f}{mean:>10.6f}{count:>8}")

    # Group importance bar chart (mean per feature, not total)
    fig, ax = plt.subplots(figsize=(8, 5))
    group_names = sorted(group_importance.keys(), key=lambda k: group_importance[k][1], reverse=True)
    group_means = [group_importance[n][1] for n in group_names]
    group_colors = ["#2196F3" if "GraphQL" in n or "temporal" in n else "#90A4AE" for n in group_names]

    ax.barh(range(len(group_names)), group_means[::-1], color=group_colors[::-1], alpha=0.85)
    ax.set_yticks(range(len(group_names)))
    ax.set_yticklabels(group_names[::-1], fontsize=9)
    ax.set_xlabel("Mean importance per feature in group")
    ax.set_title("Feature Group Importance\n(blue = groups containing new features)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "feature_group_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return importance_scores


# =============================================================================
# 4. ERROR ANALYSIS
# =============================================================================

def error_analysis(models_list, X_test, y_test, test_repos, feature_names, results_file):
    """Analyse misclassified samples and generate diagnostic plots.

    Produces:
      - Confusion matrix heatmap (TP, FP, FN, TN counts)
      - ROC curve with AUC score
      - Precision-Recall curve with Average Precision
      - Prediction confidence distribution (histogram of P(vuln) by true class)
      - Feature profile of errors (mean values of key features for TP/FP/FN/TN)

    Args:
        models_list:   List of trained models (ensemble).
        X_test:        Test features.
        y_test:        Test labels.
        test_repos:    List of test Repository objects (for provenance).
        feature_names: List of feature name strings.
        results_file:  Open file handle for numerical results.

    Returns:
        Tuple of (ensemble_pred, y_pred_binary) for downstream use.
    """
    write_result(results_file, "\n" + "=" * 70)
    write_result(results_file, "4. ERROR ANALYSIS")
    write_result(results_file, "=" * 70)

    y_test_flat = np.asarray(y_test).astype("float32")

    # Compute ensemble predictions (average across all fold models)
    ensemble_pred = np.zeros(len(y_test), dtype=np.float32)
    for m in models_list:
        ensemble_pred += m.predict(X_test, verbose=0).reshape(-1)
    ensemble_pred /= len(models_list)

    # Find optimal threshold via accuracy sweep
    best_acc, best_thresh, _ = find_best_accuracy(ensemble_pred, y_test_flat)
    y_pred_binary = (ensemble_pred > best_thresh).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_test_flat, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()

    write_result(results_file, f"\nOptimal threshold: {best_thresh:.2f}")
    write_result(results_file, f"Accuracy: {best_acc:.4f}")
    write_result(results_file, f"\nConfusion Matrix:")
    write_result(results_file, f"  TN={tn}  FP={fp}")
    write_result(results_file, f"  FN={fn}  TP={tp}")
    write_result(results_file, f"\nPrecision (vuln): {tp/(tp+fp):.4f}")
    write_result(results_file, f"Recall (vuln):    {tp/(tp+fn):.4f}")
    write_result(results_file, f"Precision (benign): {tn/(tn+fn):.4f}")
    write_result(results_file, f"Recall (benign):    {tn/(tn+fp):.4f}")

    # ---- Figure 1: Confusion matrix + ROC + PR curve ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 4a. Confusion matrix heatmap
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Vulnerable"])
    ax.set_yticklabels(["Benign", "Vulnerable"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # 4b. ROC curve (threshold-independent discrimination measure)
    ax = axes[1]
    fpr, tpr, _ = roc_curve(y_test_flat, ensemble_pred)
    roc_auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC curve (AUC = {roc_auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")  # Random baseline
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Ensemble)")
    ax.legend(loc="lower right")

    # 4c. Precision-Recall curve (more informative for imbalanced data)
    ax = axes[2]
    precision, recall, _ = precision_recall_curve(y_test_flat, ensemble_pred)
    ap = average_precision_score(y_test_flat, ensemble_pred)
    ax.plot(recall, precision, color="#4CAF50", lw=2, label=f"PR curve (AP = {ap:.3f})")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Ensemble)")
    ax.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "error_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ---- Prediction confidence analysis ----
    correct_mask = y_pred_binary == y_test_flat
    incorrect_mask = ~correct_mask

    # Confidence = distance from 0.5 (decision boundary)
    correct_confidence = np.abs(ensemble_pred[correct_mask] - 0.5) + 0.5
    incorrect_confidence = np.abs(ensemble_pred[incorrect_mask] - 0.5) + 0.5

    write_result(results_file, f"\nPrediction confidence analysis:")
    write_result(
        results_file,
        f"  Correct predictions:   mean conf={correct_confidence.mean():.4f}, "
        f"median={np.median(correct_confidence):.4f}",
    )
    write_result(
        results_file,
        f"  Incorrect predictions: mean conf={incorrect_confidence.mean():.4f}, "
        f"median={np.median(incorrect_confidence):.4f}",
    )

    # ---- Figure 2: Prediction distribution by true class ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        ensemble_pred[y_test_flat == 1],
        bins=50, alpha=0.6, color="#F44336",
        label="Vulnerable (actual)", density=True,
    )
    ax.hist(
        ensemble_pred[y_test_flat == 0],
        bins=50, alpha=0.6, color="#2196F3",
        label="Benign (actual)", density=True,
    )
    ax.axvline(x=best_thresh, color="black", linestyle="--", label=f"Threshold={best_thresh:.2f}")
    ax.set_xlabel("Predicted probability (vulnerable)")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Distribution by True Class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "prediction_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ---- Error type breakdown ----
    fp_indices = np.where((y_test_flat == 0) & (y_pred_binary == 1))[0]
    fn_indices = np.where((y_test_flat == 1) & (y_pred_binary == 0))[0]

    write_result(results_file, f"\nFalse Positives (benign predicted as vulnerable): {len(fp_indices)}")
    write_result(results_file, f"False Negatives (vulnerable predicted as benign): {len(fn_indices)}")

    # Feature profile comparison at center timestep (where the target commit lives)
    if X_test.shape[2] == len(feature_names):
        tp_indices = np.where((y_test_flat == 1) & (y_pred_binary == 1))[0]
        tn_indices = np.where((y_test_flat == 0) & (y_pred_binary == 0))[0]

        # Key features that are most interpretable for error analysis
        key_features = ["Add", "Del", "Files", "code_churn", "time_delta_hours",
                        "event_entropy", "burst_rate", "commit_urgency"]
        key_indices = [feature_names.index(f) for f in key_features if f in feature_names]
        key_names = [feature_names[i] for i in key_indices]

        if key_indices:
            center = X_test.shape[1] // 2  # Center timestep (target commit)
            write_result(results_file, f"\nMean feature values at center timestep (key features):")
            write_result(results_file, f"{'Feature':<25}{'TP':>10}{'FP':>10}{'FN':>10}{'TN':>10}")
            write_result(results_file, "-" * 65)
            for fi, fn_name in zip(key_indices, key_names):
                tp_mean = X_test[tp_indices, center, fi].mean() if len(tp_indices) > 0 else 0
                fp_mean = X_test[fp_indices, center, fi].mean() if len(fp_indices) > 0 else 0
                fn_mean = X_test[fn_indices, center, fi].mean() if len(fn_indices) > 0 else 0
                tn_mean = X_test[tn_indices, center, fi].mean() if len(tn_indices) > 0 else 0
                write_result(
                    results_file,
                    f"{fn_name:<25}{tp_mean:>10.4f}{fp_mean:>10.4f}{fn_mean:>10.4f}{tn_mean:>10.4f}",
                )

    return ensemble_pred, y_pred_binary


# =============================================================================
# 5. MULTI-SEED ROBUSTNESS ANALYSIS
# =============================================================================

def multi_seed_analysis(X_all, y_all, groups, n_splits, columns, results_file, seeds=None):
    """Run the transformer across multiple seeds and report variance.

    Tests whether the model's performance is robust to random initialisation
    by training with different seeds and comparing mean ± std across runs.

    Low variance (small std) indicates the model's architecture and training
    procedure are stable, not lucky.  High variance would suggest the results
    depend on the specific random initialisation.

    Args:
        X_all:        Full feature array.
        y_all:        Full label array.
        groups:       Per-sample group labels.
        n_splits:     Number of CV folds.
        columns:      Feature column names (or None for single-stream).
        results_file: Open file handle for results.
        seeds:        List of seed values to test (default [42, 123, 7]).
    """
    if seeds is None:
        seeds = [42, 123, 7]

    write_result(results_file, "\n" + "=" * 70)
    write_result(results_file, "5. MULTI-SEED ROBUSTNESS ANALYSIS")
    write_result(results_file, "=" * 70)

    all_accs = []
    all_f1s = []

    for seed in seeds:
        print(f"\nRunning seed {seed}...")
        result = run_kfold_collect_metrics(
            X_all, y_all, groups, n_splits,
            model_name="transformer", pooling="center",
            columns=columns, epochs=50, seed=seed,
        )
        all_accs.append(result["acc"].mean())
        all_f1s.append(result["f1"].mean())
        write_result(
            results_file,
            f"Seed {seed}: Acc={result['acc'].mean():.4f} +/- {result['acc'].std():.4f}, "
            f"F1={result['f1'].mean():.4f} +/- {result['f1'].std():.4f}",
        )

    all_accs = np.array(all_accs)
    all_f1s = np.array(all_f1s)

    write_result(results_file, f"\nAcross seeds:")
    write_result(
        results_file,
        f"Acc: {all_accs.mean():.4f} +/- {all_accs.std():.4f} (range: {all_accs.min():.4f}-{all_accs.max():.4f})",
    )
    write_result(
        results_file,
        f"F1:  {all_f1s.mean():.4f} +/- {all_f1s.std():.4f} (range: {all_f1s.min():.4f}-{all_f1s.max():.4f})",
    )

    # Bar chart: per-seed accuracy and F1 with mean lines
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(seeds))
    ax.bar([i - 0.15 for i in x], all_accs, width=0.3, label="Accuracy", color="#2196F3", alpha=0.8)
    ax.bar([i + 0.15 for i in x], all_f1s, width=0.3, label="F1 Score", color="#4CAF50", alpha=0.8)
    ax.axhline(y=all_accs.mean(), color="#1565C0", linestyle="--", alpha=0.5)
    ax.axhline(y=all_f1s.mean(), color="#2E7D32", linestyle="--", alpha=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Seed {s}" for s in seeds])
    ax.set_ylabel("Score")
    ax.set_title("Multi-Seed Robustness Analysis")
    ax.legend()
    margin = 0.05
    ax.set_ylim(min(all_accs.min(), all_f1s.min()) - margin, max(all_accs.max(), all_f1s.max()) + margin)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "multi_seed_robustness.png"), dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# 6. WINDOW SIZE COMPARISON
# =============================================================================

def window_size_comparison(results_file, seeds=None):
    """Compare performance across window sizes B=5,10,20,40.

    Tests how much temporal context the model needs to accurately classify
    security patches.  Smaller windows are faster and need less data but may
    miss important context; larger windows capture more context but risk
    including irrelevant events and increasing computational cost.

    Both transformer and conv1d are compared at each window size to see
    if the transformer's advantage scales with context length.

    Args:
        results_file: Open file handle for numerical results.
        seeds:        List of seeds (only first is used; default [42]).
    """
    if seeds is None:
        seeds = [42]

    write_result(results_file, "\n" + "=" * 70)
    write_result(results_file, "6. WINDOW SIZE COMPARISON")
    write_result(results_file, "=" * 70)

    window_sizes = [5, 10, 20, 40]
    transformer_accs = []
    conv1d_accs = []
    transformer_f1s = []
    conv1d_f1s = []

    for B in window_sizes:
        print(f"\n--- Window size B={B} ---")
        # Load dataset for this window size (from cache if available)
        repos, exp_name, columns = extract_dataset(
            aggr_options=Aggregate.before_cve,
            backs=B, metadata=True, cache=True,
            cache_location="ready_data",
        )
        repos, num_vulns = pad_and_fix(repos)
        train_size = int(0.8 * num_vulns)
        train_repos, test_repos, _ = split_repos(repos, train_size)
        X_all, y_all = split_into_x_and_y(train_repos)
        groups = build_group_labels(train_repos)
        n_splits = min(10, len(train_repos))

        # Train and evaluate both models at this window size
        t_result = run_kfold_collect_metrics(
            X_all, y_all, groups, n_splits,
            model_name="transformer", pooling="center",
            columns=None, epochs=50, seed=42,
        )
        c_result = run_kfold_collect_metrics(
            X_all, y_all, groups, n_splits,
            model_name="conv1d", pooling=None,
            columns=None, epochs=50, seed=42,
        )

        transformer_accs.append(t_result["acc"].mean())
        conv1d_accs.append(c_result["acc"].mean())
        transformer_f1s.append(t_result["f1"].mean())
        conv1d_f1s.append(c_result["f1"].mean())

        write_result(
            results_file,
            f"B={B}: Transformer Acc={t_result['acc'].mean():.4f}, Conv1D Acc={c_result['acc'].mean():.4f}  "
            f"Transformer F1={t_result['f1'].mean():.4f}, Conv1D F1={c_result['f1'].mean():.4f}",
        )

    # Line chart: accuracy and F1 vs window size for both models
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, t_vals, c_vals, title in [
        (axes[0], transformer_accs, conv1d_accs, "Accuracy"),
        (axes[1], transformer_f1s, conv1d_f1s, "F1 Score"),
    ]:
        ax.plot(window_sizes, t_vals, "o-", color="#2196F3", label="Transformer", linewidth=2, markersize=8)
        ax.plot(window_sizes, c_vals, "s-", color="#FF9800", label="Conv1D", linewidth=2, markersize=8)
        ax.set_xlabel("Window size (B)")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs Window Size")
        ax.legend()
        ax.set_xticks(window_sizes)
        all_vals = t_vals + c_vals
        margin = (max(all_vals) - min(all_vals)) * 0.3
        ax.set_ylim(min(all_vals) - margin, max(all_vals) + margin)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "window_size_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# MAIN — CLI entry point
# =============================================================================

def parse_args():
    """Parse command-line arguments for the analysis suite.

    Returns:
        argparse.Namespace with analysis configuration.
    """
    parser = argparse.ArgumentParser(description="Dissertation analysis suite")
    parser.add_argument(
        "--sections",
        nargs="+",
        default=["all"],
        choices=["all", "stats", "ablation", "importance", "error", "seeds", "windows"],
        help="Which analysis sections to run",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--kfold", type=int, default=10)
    parser.add_argument("--backs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--importance-repeats", type=int, default=5)
    return parser.parse_args()


def main():
    """Main analysis pipeline — orchestrates all analysis sections.

    Execution flow:
      1. Load feature names from pickle (for importance and ablation)
      2. Compute ablation index sets (new features, GraphQL only, temporal only)
      3. Load cached dataset (B=20, symmetric window, with metadata)
      4. Split into train+val and test sets
      5. Run selected analysis sections (in order: stats → ablation →
         importance → error → seeds → windows)
      6. Write all results to figures/analysis_results.txt
    """
    args = parse_args()
    ensure_dirs()

    sections = set(args.sections)
    run_all = "all" in sections

    # Load feature names and compute ablation index sets
    feature_names = load_feature_names()
    new_feat_indices = get_new_feature_indices(feature_names)

    # GraphQL feature indices (7 features: 5 days_since_* + 2 gql_*)
    graphql_indices = [
        feature_names.index(n)
        for n in ["days_since_forks", "days_since_issues", "days_since_pullRequests",
                   "days_since_releases", "days_since_stargazers", "gql_additions", "gql_deletions"]
        if n in feature_names
    ]
    # Derived temporal feature indices (16 features)
    temporal_indices = [
        feature_names.index(n)
        for n in ["time_delta_hours", "code_churn", "add_del_ratio", "is_commit",
                   "event_entropy", "repo_age_days", "hour_sin", "hour_cos",
                   "dow_sin", "dow_cos", "type_changed", "cum_commits",
                   "rel_position", "burst_rate", "commit_urgency", "churn_diversity"]
        if n in feature_names
    ]

    print(f"Feature names loaded: {len(feature_names)} features")
    print(f"New features: {len(new_feat_indices)} indices")
    print(f"  GraphQL: {len(graphql_indices)}")
    print(f"  Temporal: {len(temporal_indices)}")

    # Load primary dataset (B=20, symmetric window, with metadata, seed 42)
    print("\nLoading dataset...")
    init(seed=args.seed)
    repos, exp_name, _cached_columns = extract_dataset(
        aggr_options=Aggregate.before_cve,
        backs=args.backs, metadata=True, cache=True,
        cache_location="ready_data",
    )
    repos, num_vulns = pad_and_fix(repos)

    # Split: 80% train + 10% val + 10% test (matches train.py: GroupKFold runs on
    # train+val pool of 90%, the remaining 10% is held out as the test set).
    train_size = int(0.8 * num_vulns)
    validation_size = int(0.1 * num_vulns)
    train_repos, test_repos, _ = split_repos(repos, train_size + validation_size)
    X_all, y_all = split_into_x_and_y(train_repos)
    X_test, y_test = split_into_x_and_y(test_repos)
    groups = build_group_labels(train_repos)
    n_splits = min(args.kfold, len(train_repos))

    # For standard single-stream transformer, columns=None is correct.
    # Two-stream/temporal-focus modes would need the actual column names.
    columns_list = None

    print(f"Train: {X_all.shape}, Test: {X_test.shape}")
    print(f"Folds: {n_splits}")

    # Open results file in append mode (multiple runs accumulate results)
    with open(os.path.join(FIGURES_DIR, RESULTS_FILE), "a") as rf:
        write_result(rf, "DISSERTATION ANALYSIS RESULTS")
        write_result(rf, f"Dataset: B={args.backs}, symmetric window, metadata")
        write_result(rf, f"Repos: {len(repos)}, Vulns: {num_vulns}")
        write_result(rf, f"Train shape: {X_all.shape}, Test shape: {X_test.shape}")
        write_result(rf, f"K-folds: {n_splits}, Epochs: {args.epochs}, Seed: {args.seed}")

        # --- Section 1: Statistical significance (transformer vs conv1d) ---
        transformer_metrics = None
        conv1d_metrics = None

        if run_all or "stats" in sections:
            print("\n" + "=" * 50)
            print("Running: Statistical Significance Tests")
            print("=" * 50)

            transformer_metrics = run_kfold_collect_metrics(
                X_all, y_all, groups, n_splits,
                model_name="transformer", pooling="center",
                columns=columns_list, epochs=args.epochs, seed=args.seed,
            )
            conv1d_metrics = run_kfold_collect_metrics(
                X_all, y_all, groups, n_splits,
                model_name="conv1d", pooling=None,
                columns=None, epochs=args.epochs, seed=args.seed,
            )
            plot_statistical_tests(transformer_metrics, conv1d_metrics, rf)

        # --- Section 2: Feature ablation ---
        if run_all or "ablation" in sections:
            print("\n" + "=" * 50)
            print("Running: Feature Ablation Study")
            print("=" * 50)

            # Reuse full model results from stats section if available
            if transformer_metrics is None:
                results_all = run_kfold_collect_metrics(
                    X_all, y_all, groups, n_splits,
                    model_name="transformer", pooling="center",
                    columns=columns_list, epochs=args.epochs, seed=args.seed,
                )
            else:
                results_all = transformer_metrics

            # Ablation 1: Without ALL 23 new features
            print("\n  Ablation: without new features...")
            results_no_new = run_kfold_collect_metrics(
                X_all, y_all, groups, n_splits,
                model_name="transformer", pooling="center",
                columns=columns_list, epochs=args.epochs, seed=args.seed,
                ablation_indices=new_feat_indices,
            )

            # Ablation 2: Without 7 GraphQL recency features only
            print("\n  Ablation: without GraphQL features...")
            results_no_graphql = run_kfold_collect_metrics(
                X_all, y_all, groups, n_splits,
                model_name="transformer", pooling="center",
                columns=columns_list, epochs=args.epochs, seed=args.seed,
                ablation_indices=graphql_indices,
            )

            # Ablation 3: Without 16 derived temporal features only
            print("\n  Ablation: without derived temporal features...")
            results_no_temporal = run_kfold_collect_metrics(
                X_all, y_all, groups, n_splits,
                model_name="transformer", pooling="center",
                columns=columns_list, epochs=args.epochs, seed=args.seed,
                ablation_indices=temporal_indices,
            )

            plot_feature_ablation(results_all, results_no_new, results_no_graphql, results_no_temporal, rf)

        # --- Section 3: Permutation feature importance ---
        if run_all or "importance" in sections:
            print("\n" + "=" * 50)
            print("Running: Permutation Feature Importance")
            print("=" * 50)

            # Need trained models for importance analysis
            if transformer_metrics is None:
                transformer_metrics = run_kfold_collect_metrics(
                    X_all, y_all, groups, n_splits,
                    model_name="transformer", pooling="center",
                    columns=columns_list, epochs=args.epochs, seed=args.seed,
                )

            permutation_importance_analysis(
                transformer_metrics["models"], X_test, y_test, feature_names, rf,
                n_repeats=args.importance_repeats,
            )

        # --- Section 4: Error analysis ---
        if run_all or "error" in sections:
            print("\n" + "=" * 50)
            print("Running: Error Analysis")
            print("=" * 50)

            if transformer_metrics is None:
                transformer_metrics = run_kfold_collect_metrics(
                    X_all, y_all, groups, n_splits,
                    model_name="transformer", pooling="center",
                    columns=columns_list, epochs=args.epochs, seed=args.seed,
                )

            error_analysis(
                transformer_metrics["models"], X_test, y_test, test_repos, feature_names, rf,
            )

        # --- Section 5: Multi-seed robustness ---
        if run_all or "seeds" in sections:
            print("\n" + "=" * 50)
            print("Running: Multi-Seed Robustness Analysis")
            print("=" * 50)
            multi_seed_analysis(X_all, y_all, groups, n_splits, columns_list, rf, seeds=[42, 123, 7, 256, 999])

        # --- Section 6: Window size comparison ---
        if run_all or "windows" in sections:
            print("\n" + "=" * 50)
            print("Running: Window Size Comparison")
            print("=" * 50)
            window_size_comparison(rf)

        write_result(rf, "\n" + "=" * 70)
        write_result(rf, "ANALYSIS COMPLETE")
        write_result(rf, f"Figures saved to: {FIGURES_DIR}/")
        write_result(rf, f"Results saved to: {os.path.join(FIGURES_DIR, RESULTS_FILE)}")
        write_result(rf, "=" * 70)

    print(f"\nAll done. Check {FIGURES_DIR}/ for figures and {os.path.join(FIGURES_DIR, RESULTS_FILE)} for results.")


if __name__ == "__main__":
    main()
