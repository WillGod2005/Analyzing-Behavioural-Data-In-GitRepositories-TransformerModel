#!/usr/bin/env python
"""
evaluate_cvefixes.py — External validation on the CVEfixes v1.0.7 dataset.

This module evaluates trained models on an INDEPENDENT test set of repositories
scraped from the CVEfixes database (Bhandari et al., 2021).  These repos were
NOT seen during training, providing a true out-of-distribution generalisation test.

Two evaluation modes:

  1. BALANCED (default):  For each repo, extract 1:1 vuln:benign commit windows
     (same ratio as training).  This measures accuracy under controlled conditions
     and is directly comparable to the training pipeline's test metrics.

  2. REALISTIC (--realistic):  Extract windows for ALL commits in each repo,
     preserving the natural class imbalance (~1:790 vuln:benign).  This simulates
     deploying the model as a real security scanner that must process an entire
     repository timeline.  Reports ROC-AUC, Average Precision, and FPR alongside
     standard metrics.

Multi-seed support (--all-seeds):  Evaluates models trained with all 5 seeds
(42, 123, 7, 256, 999) and reports mean ± std across seeds.

The evaluation pipeline mirrors the training pipeline exactly:
  1. Read repo CSV → fillna → mark commit hashes
  2. Add metadata (~388 static features from GitHub API)
  3. Add GraphQL recency features (7 columns)
  4. Feature engineering via fix_repo_shape() (16 derived features + z-score)
  5. Extract event windows via get_event_window()
  6. Pad windows to match model's expected input shape
  7. Run model.predict() and compute metrics

Usage:
    # Balanced evaluation with single seed
    python evaluate_cvefixes.py -a before --model transformer --backs 5 --meta

    # Balanced evaluation across all 5 seeds
    python evaluate_cvefixes.py -a before --model transformer --backs 5 --meta --all-seeds

    # Realistic evaluation (natural imbalance, enriched data)
    python evaluate_cvefixes.py -a before --model transformer --backs 5 --meta --enriched --realistic

Used by: Dissertation external validation experiments
"""

import argparse
import json
import logging
import os
import sys

import coloredlogs
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    average_precision_score, roc_auc_score,
)

# Import models module to register custom Keras layers (CenterTokenPooling, etc.)
# so tf.keras.models.load_model() can deserialise them correctly.
import models  # noqa: F401 — registers custom Keras layers for model loading
from dataset_utils import (
    Aggregate,
    fix_repo_shape,
    get_event_window,
    add_graphql_features,
)
from helper import Repository, safe_mkdir, EnumAction, add_metadata

logger = logging.getLogger(__name__)
coloredlogs.install(fmt="%(asctime)s %(levelname)s %(message)s")

# Directory containing CVEfixes test repo CSVs, GraphQL data, and metadata
CVEFIXES_DIR_BASE = os.path.join("data_collection", "cvefixes_test")
CVEFIXES_DIR = CVEFIXES_DIR_BASE  # May be overridden by --enriched flag


# =============================================================================
# Model path helper
# =============================================================================

def make_model_dir_name(aggr, backs, metadata, seed=42):
    """Construct the model checkpoint directory name from training parameters.

    Must match the naming convention used by train.py so we load the correct
    model for each configuration.  The format mirrors make_new_dir_name()
    from dataset_utils.py.

    Args:
        aggr:     Aggregate enum value (e.g. Aggregate.before_cve).
        backs:    Window size parameter.
        metadata: Whether metadata features were used in training.
        seed:     Random seed used during training.

    Returns:
        String directory name, e.g. "Aggregate.before_cve_R1_B5_meta_s123"
    """
    name = f"{aggr}_R1_B{backs}"
    if metadata:
        name += "_meta"
    if seed != 42:
        name += f"_s{seed}"
    return name


# =============================================================================
# process_repo — Per-repo feature engineering and window extraction
# =============================================================================

def process_repo(csv_path, aggr_options, backs, metadata_dict, language_vocab,
                 benign_vuln_ratio=1, realistic=False):
    """Load a repo CSV and extract event windows for vuln + benign commits.

    Applies the same feature engineering pipeline as the training data:
    metadata enrichment → GraphQL features → fix_repo_shape() → window extraction.

    Args:
        csv_path:           Path to the repo's CSV file.
        aggr_options:       Aggregate enum (window extraction mode).
        backs:              Window size in events.
        metadata_dict:      Dict of repo metadata from repo_metadata.json.
        language_vocab:     Sorted list of all languages (from training data).
        benign_vuln_ratio:  Benign:vuln sampling ratio (default 1:1).
        realistic:          If True, extract windows for ALL commits (no sampling).

    Returns:
        Tuple of (samples, n_vulns, repo_name):
          - samples:    List of (window_array, label) tuples, or None if skipped.
          - n_vulns:    Number of vulnerability windows extracted.
          - repo_name:  Repository identifier string.
    """
    import random
    repo_name = os.path.basename(csv_path).replace(".csv", "")

    # Read repo CSV with same dtype specifications as training pipeline
    cur_repo = pd.read_csv(
        csv_path,
        low_memory=False,
        parse_dates=["created_at"],
        dtype={
            "type": "string",
            "name": "string",
            "Hash": "string",
            "Add": np.float64,
            "Del": np.float64,
            "Files": np.float64,
            "Vuln": np.float64,
        },
    )

    # Skip repos with too few events (need sufficient context for windows)
    if cur_repo.shape[0] < 100:
        logger.info(f"Skipping {repo_name}: only {cur_repo.shape[0]} events (<100)")
        return None, None, repo_name

    # Fill missing values (same as training pipeline)
    if "Hash" in cur_repo.columns:
        cur_repo["Hash"] = cur_repo["Hash"].fillna("")
    cur_repo = cur_repo.fillna(0)

    # Skip repos with no vulnerability-fixing commits
    number_of_vulns = cur_repo[cur_repo["Vuln"] != 0].shape[0]
    if number_of_vulns == 0:
        logger.info(f"Skipping {repo_name}: no vuln commits")
        return None, None, repo_name

    # Mark events with actual commit hashes (becomes is_commit feature)
    if "Hash" in cur_repo.columns:
        hash_str = cur_repo["Hash"].astype(str)
        cur_repo["_has_hash"] = (
            (hash_str.str.len() > 0) & (hash_str != "0")
        ).astype(float)

    # Add static metadata features (~388 columns from GitHub API)
    repo_holder = Repository()
    repo_holder.file = repo_name
    use_metadata = bool(metadata_dict)

    if metadata_dict:
        try:
            cur_repo = add_metadata(
                CVEFIXES_DIR,
                metadata_dict,
                cur_repo,
                repo_name,
                repo_holder,
                language_vocab=language_vocab,
            )
        except KeyError:
            # Repo not found in metadata dict — proceed without metadata
            logger.debug(f"{repo_name}: not in metadata dict")
            use_metadata = False

    # Add GraphQL recency/churn features (7 columns)
    cur_repo = add_graphql_features(cur_repo, CVEFIXES_DIR, repo_name)

    # Feature engineering: derive 16 temporal features + z-score normalisation
    all_set = set()
    cur_repo = fix_repo_shape(all_set, cur_repo, metadata=use_metadata)

    # Identify target commits
    vulns = cur_repo.index[cur_repo["Vuln"] == 1].tolist()
    if not vulns:
        logger.info(f"Skipping {repo_name}: no vuln commits after processing")
        return None, None, repo_name

    if realistic:
        # REALISTIC mode: test on ALL non-vuln commits (natural class imbalance)
        benigns = cur_repo.index[cur_repo["Vuln"] == 0].tolist()
    else:
        # BALANCED mode: sample benign commits at the specified ratio
        benigns = cur_repo.index[cur_repo["Vuln"] == 0].tolist()
        random.shuffle(benigns)
        benigns = benigns[:benign_vuln_ratio * len(vulns)]

    # Drop label column before window extraction
    cur_repo = cur_repo.drop(["Vuln"], axis=1)

    # Prepare index structure for window extraction
    if aggr_options in (Aggregate.before_cve, Aggregate.only_before):
        # Drop created_at, use integer idx for positional slicing
        cur_repo = cur_repo.reset_index().drop(
            ["created_at"], axis=1, errors="ignore"
        ).set_index("idx")

    # Extract windows for vulnerability-fixing commits
    vuln_windows = []
    for event_idx in vulns:
        try:
            res = get_event_window(cur_repo, event_idx, aggr_options, backs=backs)
            if res is not None and len(res) > 0:
                vuln_windows.append(res)
        except Exception:
            continue  # Skip windows that fail (e.g. too close to sequence boundary)

    # Extract windows for benign commits
    benign_windows = []
    for event_idx in benigns:
        try:
            res = get_event_window(cur_repo, event_idx, aggr_options, backs=backs)
            if res is not None and len(res) > 0:
                benign_windows.append(res)
        except Exception:
            continue

    if not vuln_windows:
        logger.info(f"Skipping {repo_name}: no valid vuln windows extracted")
        return None, None, repo_name

    # Combine vuln and benign windows with labels
    windows = vuln_windows + benign_windows
    labels = [1] * len(vuln_windows) + [0] * len(benign_windows)

    return list(zip(windows, labels)), len(vuln_windows), repo_name


# =============================================================================
# _pad_and_predict — Pad windows and run model inference
# =============================================================================

def _pad_and_predict(model, windows, expected_seq, expected_features):
    """Pad a list of variable-length windows and run model prediction.

    Handles dimension mismatches between the test data and the model's
    expected input shape.  Windows may have different sequence lengths
    (from different repos) and different feature counts (if some repos
    lack certain metadata or GraphQL features).

    Padding strategy:
      - Sequence length: Left-pad with zeros (consistent with training)
      - Feature count:   Right-pad with zeros or truncate to match model

    Args:
        model:             Trained Keras model.
        windows:           List of 2-D numpy arrays (seq_len, features).
        expected_seq:      Model's expected sequence length (or None).
        expected_features: Model's expected feature dimensionality.

    Returns:
        1-D numpy array of prediction probabilities, shape (n_windows,).
    """
    if not windows:
        return np.array([])

    # Find max dimensions across all windows in this batch
    max_len = max(w.shape[0] for w in windows)
    max_feat = max(w.shape[1] for w in windows)

    # Left-pad each window to uniform shape within this batch
    padded = []
    for w in windows:
        padded.append(np.pad(w, ((max_len - w.shape[0], 0), (0, max_feat - w.shape[1]))))
    X = np.nan_to_num(np.array(padded, dtype=np.float32))

    # Match model's expected feature dimensionality
    if max_feat < expected_features:
        # Test data has fewer features — right-pad with zeros
        X = np.pad(X, ((0, 0), (0, 0), (0, expected_features - max_feat)))
    elif max_feat > expected_features:
        # Test data has more features — truncate to model's expected count
        X = X[:, :, :expected_features]

    # Match model's expected sequence length
    if expected_seq is not None and max_len != expected_seq:
        if max_len < expected_seq:
            # Test windows shorter — left-pad with zeros
            X = np.pad(X, ((0, 0), (expected_seq - max_len, 0), (0, 0)))
        else:
            # Test windows longer — keep only the last expected_seq timesteps
            X = X[:, -expected_seq:, :]

    return model.predict(X, verbose=0).reshape(-1)


# =============================================================================
# _run_realistic — Realistic evaluation with natural class imbalance
# =============================================================================

def _run_realistic(csv_files, args, metadata_dict, language_vocab, seeds):
    """Realistic evaluation: test on ALL commits per repo.

    Processes repos one-by-one to avoid out-of-memory issues (each repo
    can have thousands of commits).  Uses a single seed model.

    Reports comprehensive metrics appropriate for imbalanced classification:
      - ROC-AUC:  Area under the ROC curve (threshold-independent)
      - Average Precision (AP):  Area under the precision-recall curve
      - FPR:  False positive rate (false alarm rate per benign commit)
      - Threshold sweep:  Tests thresholds 0.30–0.95 to find optimal operating point

    The typical class imbalance is ~1:790 (vuln:benign), making precision
    and FPR much more informative than raw accuracy.

    Args:
        csv_files:       List of paths to test repo CSVs.
        args:            Parsed CLI arguments.
        metadata_dict:   Repo metadata dict.
        language_vocab:  Sorted language vocabulary from training data.
        seeds:           List of seed values (only first is used).
    """
    # Load model for the first seed
    seed = seeds[0]
    model_dir = make_model_dir_name(args.aggregate, args.backs, args.meta, seed)
    model_path = os.path.join("models", model_dir, f"{args.model}.keras")

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    expected_features = model.input_shape[-1]
    expected_seq = model.input_shape[1]

    # Process repos one at a time to manage memory
    all_preds = []
    all_labels = []
    repos_processed = 0
    repos_skipped = 0
    total_vulns = 0
    total_benigns = 0

    for i, csv_path in enumerate(csv_files):
        samples, n_vulns, repo_name = process_repo(
            csv_path, args.aggregate, args.backs, metadata_dict, language_vocab,
            realistic=True,
        )
        if samples is None or len(samples) == 0:
            repos_skipped += 1
            continue

        # Extract windows and labels for this repo
        windows = [s[0] for s in samples]
        labels = [s[1] for s in samples]
        n_benign = len(labels) - n_vulns

        # Run inference for this repo's windows
        pred = _pad_and_predict(model, windows, expected_seq, expected_features)
        all_preds.extend(pred.tolist())
        all_labels.extend(labels)
        total_vulns += n_vulns
        total_benigns += n_benign
        repos_processed += 1

        # Progress reporting every 20 repos
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(csv_files)}] {repos_processed} repos, "
                  f"{total_vulns} vuln + {total_benigns} benign commits so far")

    print(f"\nProcessed {repos_processed} repos, skipped {repos_skipped}")
    print(f"Total: {len(all_labels)} samples ({total_vulns} vuln, {total_benigns} benign)")
    print(f"Imbalance: 1 vuln : {total_benigns / max(total_vulns, 1):.1f} benign")
    print()

    # Compute metrics at the specified threshold
    pred = np.array(all_preds)
    y_test = np.array(all_labels, dtype=np.float32)
    y_pred = (pred >= args.threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_test, pred)
    ap = average_precision_score(y_test, pred)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"=== REALISTIC RESULTS (threshold={args.threshold}) ===")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Avg Precision (AP): {ap:.4f}")
    print(f"  FPR (false alarm rate): {fpr:.4f}")
    print(f"  Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"  Mean prob (vuln):   {pred[y_test == 1].mean():.4f}")
    print(f"  Mean prob (benign): {pred[y_test == 0].mean():.4f}")
    print()

    # Threshold sweep: find the optimal operating point for F1
    # Tests coarse grid (0.30–0.95 in 0.05 steps) then refines around best
    print("=== THRESHOLD SWEEP ===")
    print(f"{'Thr':>5} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'FPR':>7}  TP     FP     FN")
    print("-" * 75)
    best_f1, best_thr = 0, 0.5
    for thr in np.arange(0.3, 0.95, 0.05):
        y_t = (pred >= thr).astype(int)
        t_acc = accuracy_score(y_test, y_t)
        t_prec = precision_score(y_test, y_t, zero_division=0)
        t_rec = recall_score(y_test, y_t, zero_division=0)
        t_f1 = f1_score(y_test, y_t, zero_division=0)
        t_tn, t_fp, t_fn, t_tp = confusion_matrix(y_test, y_t, labels=[0, 1]).ravel()
        t_fpr = t_fp / (t_fp + t_tn) if (t_fp + t_tn) > 0 else 0
        print(f"{thr:>5.2f} {t_acc:>7.4f} {t_prec:>7.4f} {t_rec:>7.4f} {t_f1:>7.4f} {t_fpr:>7.4f}  {t_tp:>5}  {t_fp:>5}  {t_fn:>5}")
        if t_f1 > best_f1:
            best_f1, best_thr = t_f1, thr

    # Finer sweep around best threshold (±0.05 in 0.01 steps)
    for thr in np.arange(max(0.1, best_thr - 0.05), min(0.95, best_thr + 0.06), 0.01):
        y_t = (pred >= thr).astype(int)
        t_f1 = f1_score(y_test, y_t, zero_division=0)
        if t_f1 > best_f1:
            best_f1, best_thr = t_f1, thr

    print(f"\nBest F1: {best_f1:.4f} at threshold={best_thr:.2f}")

    # Final summary
    print()
    print("=" * 60)
    print(f"SUMMARY (REALISTIC): {args.model} | {args.aggregate.value} | B={args.backs} | seed={seed}")
    print(f"  Test repos: {repos_processed}")
    print(f"  Total commits tested: {len(all_labels)} ({total_vulns} vuln, {total_benigns} benign)")
    print(f"  Imbalance: 1:{total_benigns / max(total_vulns, 1):.1f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Avg Prec:  {ap:.4f}")
    print(f"  Recall@0.5:    {rec:.4f} (catches {tp} of {total_vulns} vuln commits)")
    print(f"  Precision@0.5: {prec:.4f} ({tp} true alerts out of {tp + fp} total alerts)")
    print(f"  Best F1: {best_f1:.4f} at threshold={best_thr:.2f}")


# =============================================================================
# main — Entry point for CVEfixes evaluation
# =============================================================================

def main():
    """Main evaluation pipeline for CVEfixes test data.

    Execution flow:
      1. Parse CLI arguments (aggregate mode, model, backs, seed, etc.)
      2. Load metadata and language vocabulary from training data
      3. Find all test repo CSVs in the CVEfixes directory
      4. If --realistic: delegate to _run_realistic() for per-repo processing
      5. Otherwise (balanced mode):
         a. Process all repos and extract windows
         b. Pad all windows to uniform shape
         c. For each seed, load model and run inference
         d. Report per-seed and aggregate metrics
    """
    parser = argparse.ArgumentParser(description="Evaluate models on CVEfixes test data")
    # Window extraction parameters
    parser.add_argument(
        "-a", "--aggregate",
        type=Aggregate, action=EnumAction,
        default=Aggregate.before_cve,
    )
    parser.add_argument("--model", type=str, default="transformer")
    parser.add_argument("--backs", type=int, default=5)
    parser.add_argument("--meta", action="store_true")
    # Evaluation parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--enriched", action="store_true",
                        help="Use enriched CSVs from cvefixes_test/enriched/")
    parser.add_argument("--realistic", action="store_true",
                        help="Test on ALL commits (natural imbalance, no 1:1 sampling)")
    args = parser.parse_args()

    # Select seeds: all 5 experimental seeds or just the specified one
    seeds = [42, 123, 7, 256, 999] if args.all_seeds else [args.seed]

    # Set up data directory — enriched CSVs are in a subdirectory
    global CVEFIXES_DIR
    csv_dir = CVEFIXES_DIR_BASE
    if args.enriched:
        csv_dir = os.path.join(CVEFIXES_DIR_BASE, "enriched")
        print(f"Using enriched CSVs from: {csv_dir}")
    # GraphQL data and metadata always come from the base directory
    CVEFIXES_DIR = CVEFIXES_DIR_BASE

    # Load metadata for the test repos
    metadata_dict = {}
    language_vocab = []
    if args.meta:
        # Load CVEfixes-specific metadata
        meta_path = os.path.join(CVEFIXES_DIR_BASE, "repo_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata_dict = json.load(f)

        # Use TRAINING language vocab to ensure consistent column alignment.
        # Test repos may have languages not seen in training — those are ignored.
        # Training repos may have languages not in test repos — those get zeros.
        train_meta_path = os.path.join("data_collection", "repo_metadata.json")
        if os.path.exists(train_meta_path):
            with open(train_meta_path) as f:
                train_meta = json.load(f)
            language_vocab = sorted({
                lang for m in train_meta.values()
                for lang in m.get("languages_edges", [])
            })

    print(f"Metadata: {len(metadata_dict)} repos, {len(language_vocab)} languages in vocab")

    # Find all test repo CSV files
    csv_files = sorted([
        os.path.join(csv_dir, f)
        for f in os.listdir(csv_dir)
        if f.endswith(".csv")
    ])
    print(f"Found {len(csv_files)} test repo CSVs")

    # ---- Realistic mode: per-repo processing ----
    if args.realistic:
        print("REALISTIC MODE: testing on ALL commits (natural class imbalance)")
        print("Processing repo-by-repo to manage memory...")
        _run_realistic(csv_files, args, metadata_dict, language_vocab, seeds)
        return

    # ---- Balanced mode: batch processing ----
    # Process all repos and collect windows
    all_samples = []  # List of (window_array, label) tuples
    repos_processed = 0
    repos_skipped = 0
    total_vulns = 0

    for csv_path in csv_files:
        samples, n_vulns, repo_name = process_repo(
            csv_path, args.aggregate, args.backs, metadata_dict, language_vocab,
        )
        if samples is None or len(samples) == 0:
            repos_skipped += 1
            continue

        all_samples.extend(samples)
        total_vulns += n_vulns
        repos_processed += 1
        logger.info(f"  {repo_name}: {n_vulns} vuln + {len(samples) - n_vulns} benign windows")

    print(f"\nProcessed {repos_processed} repos, skipped {repos_skipped}")
    n_vuln_samples = sum(1 for _, l in all_samples if l == 1)
    n_benign_samples = sum(1 for _, l in all_samples if l == 0)
    print(f"Total samples: {len(all_samples)} ({n_vuln_samples} vuln, {n_benign_samples} benign)")

    if not all_samples:
        print("ERROR: No valid samples extracted.")
        sys.exit(1)

    # Pad all windows to uniform shape for batched model input
    windows = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]

    max_len = max(w.shape[0] for w in windows)
    max_features = max(w.shape[1] for w in windows)
    print(f"Feature dimensions: seq_len={max_len}, n_features={max_features}")

    # Left-pad sequences and right-pad features to uniform shape
    padded = []
    for w in windows:
        seq_pad = max_len - w.shape[0]     # Zero rows at top (left-pad)
        feat_pad = max_features - w.shape[1]  # Zero columns at right
        padded.append(np.pad(w, ((seq_pad, 0), (0, feat_pad))))
    X_test = np.nan_to_num(np.array(padded, dtype=np.float32))
    y_test = np.array(labels, dtype=np.float32)

    print(f"Test array shape: {X_test.shape}")
    print()

    # ---- Run inference for each seed ----
    all_results = []

    for seed in seeds:
        # Locate the model checkpoint for this seed
        model_dir = make_model_dir_name(args.aggregate, args.backs, args.meta, seed)
        model_path = os.path.join("models", model_dir, f"{args.model}.keras")

        if not os.path.exists(model_path):
            print(f"Model not found: {model_path} — skipping seed {seed}")
            continue

        print(f"--- Seed {seed} ---")
        print(f"Loading: {model_path}")

        model = tf.keras.models.load_model(model_path, compile=False)

        # Handle feature dimensionality mismatch between test data and model.
        # This can occur if test repos have different metadata/language features.
        expected_features = model.input_shape[-1]
        X_eval = X_test
        if max_features != expected_features:
            print(f"  Feature mismatch: model expects {expected_features}, data has {max_features}")
            if max_features < expected_features:
                # Test has fewer features — right-pad with zeros
                X_eval = np.pad(X_test, ((0, 0), (0, 0), (0, expected_features - max_features)))
            else:
                # Test has more features — truncate to model's expected count
                X_eval = X_test[:, :, :expected_features]

        # Handle sequence length mismatch
        expected_seq = model.input_shape[1]
        if expected_seq is not None and max_len != expected_seq:
            print(f"  Seq mismatch: model expects {expected_seq}, data has {max_len}")
            if max_len < expected_seq:
                # Test windows shorter — left-pad with zeros
                X_eval = np.pad(X_eval, ((0, 0), (expected_seq - max_len, 0), (0, 0)))
            else:
                # Test windows longer — keep last expected_seq timesteps
                X_eval = X_eval[:, -expected_seq:, :]

        # Run model inference and binarise at threshold
        pred = model.predict(X_eval, verbose=0).reshape(-1)
        y_pred = (pred >= args.threshold).astype(int)

        # Compute standard classification metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
        print(f"  Mean prob (vuln):   {pred[y_test == 1].mean():.4f}")
        if n_benign_samples > 0:
            print(f"  Mean prob (benign): {pred[y_test == 0].mean():.4f}")
        print()

        all_results.append({
            "seed": seed, "acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        })

    # ---- Aggregate results across seeds ----
    if all_results:
        accs = [r["acc"] for r in all_results]
        precs = [r["prec"] for r in all_results]
        f1s = [r["f1"] for r in all_results]
        recs = [r["rec"] for r in all_results]
        print("=" * 60)
        print(f"SUMMARY (BALANCED 1:1): {args.model} | {args.aggregate.value} | B={args.backs} | meta={args.meta}")
        print(f"  Seeds evaluated: {len(all_results)}")
        print(f"  Accuracy:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        print(f"  Precision: {np.mean(precs):.4f} +/- {np.std(precs):.4f}")
        print(f"  Recall:    {np.mean(recs):.4f} +/- {np.std(recs):.4f}")
        print(f"  F1 Score:  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        print(f"  Test repos: {repos_processed}")
        print(f"  Test samples: {len(all_samples)} ({n_vuln_samples} vuln, {n_benign_samples} benign)")


if __name__ == "__main__":
    main()
