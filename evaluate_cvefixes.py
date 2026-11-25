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
from model_outputs import predict_logits                 # RUNBOOK_v2: logits
from provenance import ordered_events_from_frame, lookup  # RUNBOOK_v2: repo + hash
import hashlib

logger = logging.getLogger(__name__)
coloredlogs.install(fmt="%(asctime)s %(levelname)s %(message)s")

# Directory containing CVEfixes test repo CSVs, GraphQL data, and metadata
CVEFIXES_DIR_BASE = os.path.join("data_collection", "cvefixes_test")
CVEFIXES_DIR = CVEFIXES_DIR_BASE  # May be overridden by --enriched flag


# =============================================================================
# Model path helper
# =============================================================================

def make_model_dir_name(aggr, backs, metadata, seed=42, comment=""):
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
    if comment:  # rerun_v3 (Commit-only benign targets): same order as train.py / make_new_dir_name
        name += f"_{comment}"
    if seed != 42:
        name += f"_s{seed}"
    return name


N_FOLDS = 10


def load_ensemble(model_dir, model_name, legacy_last_fold=False):
    """RUNBOOK_v2 Task 1: load all 10 fold models of a run (models/<dir>/<model>_folds/fold_k.keras).

    Missing fold models are a hard error: the last-fold model must not be used
    on its own.  --legacy-last-fold reproduces the pre-RUNBOOK_v2 behaviour
    (single models/<dir>/<model>.keras) for comparison only.
    """
    if legacy_last_fold:
        paths = [os.path.join("models", model_dir, f"{model_name}.keras")]
    else:
        paths = [os.path.join("models", model_dir, f"{model_name}_folds", f"fold_{k}.keras")
                 for k in range(N_FOLDS)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"FATAL: missing model files: {missing}")
        sys.exit(3)
    print(f"[ENSEMBLE] {'LEGACY last-fold model' if legacy_last_fold else f'{len(paths)} fold models'}: "
          f"{os.path.dirname(paths[0])}")
    return [tf.keras.models.load_model(p, compile=False) for p in paths]


def ensemble_predict(fold_models, X):
    """Mean probability, mean logit, and per-fold probabilities (n_folds, n)."""
    probs = np.stack([m.predict(X, verbose=0).reshape(-1) for m in fold_models])
    logits = np.stack([predict_logits(m, X) for m in fold_models])
    return probs.mean(axis=0), logits.mean(axis=0), probs


def reload_ensemble(fold_models, model_dir, args):
    """--reload-every: drop the fold models, their logit sub-models and every compiled GPU program, then load the
    same fold model files again. Batches are unchanged, so scores are unchanged; memory used by programs compiled
    for earlier repo sizes is freed (runs_v2/NOTES.md s9, s11). Checked on the GPU first
    (runs_v2/12_oom_fix/reload_check.py)."""
    import gc
    import model_outputs
    del fold_models
    model_outputs._PARTS_CACHE.clear()
    tf.keras.backend.clear_session()
    gc.collect()
    return load_ensemble(model_dir, args.model, args.legacy_last_fold)


def ensemble_predict_padded(fold_models, X, multiple=256):
    """ensemble_predict on X padded with zero rows to a multiple of `multiple`, padded rows dropped (--pad-batches).

    Every predict call then sees the same batch shapes (predict: batches of 32; logits: 256), so the GPU compiles
    each model once instead of once per repo size (runs_v2/NOTES.md s9). Rows are scored independently; the
    padding rows are discarded. Checked on the GPU before use (runs_v2/12_oom_fix/pad_check.py).
    """
    n = len(X)
    m = -(-n // multiple) * multiple
    if m != n:
        X = np.concatenate([X, np.zeros((m - n,) + X.shape[1:], dtype=X.dtype)])
    pred, logit, probs = ensemble_predict(fold_models, X)
    return pred[:n], logit[:n], probs[:, :n]


def commit_list_fingerprint(prov, labels):
    """sha256 over sorted (repo, created_at, event_idx, label) -- identical sets give identical hashes."""
    rows = sorted(zip(prov["repo"], prov["created_at"], [int(i) for i in prov["event_idx"]],
                      [int(l) for l in labels]))
    return hashlib.sha256(repr(rows).encode()).hexdigest()


def stack_provenance(prov_rows):
    """prov_rows: list of (repo, hash, event_type, created_at, event_idx, verified) tuples."""
    keys = ["repo", "hash", "event_type", "created_at", "event_idx", "prov_verified"]
    cols = list(zip(*prov_rows)) if prov_rows else [[]] * len(keys)
    out = {k: np.asarray(c) for k, c in zip(keys, cols)}
    out["event_idx"] = out["event_idx"].astype(np.int64)
    out["prov_verified"] = out["prov_verified"].astype(bool)
    return out


# =============================================================================
# process_repo — Per-repo feature engineering and window extraction
# =============================================================================

def process_repo(csv_path, aggr_options, backs, metadata_dict, language_vocab,
                 benign_vuln_ratio=1, realistic=False, sample_seed=0, benign_commit_only=False):
    """Load a repo CSV and extract event windows for vuln + benign commits.
    rerun_v3: benign_commit_only=True keeps only Commit events as benign targets (balanced and realistic).

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
    # RUNBOOK_v2: raw copy for provenance (repo, hash) recovery after windowing
    prov_raw = cur_repo[["created_at", "type", "Hash", "Vuln"]].copy()

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

    # rerun_v3 (Commit-only benign targets): row marker before fix_repo_shape drops "type"; not a feature
    if benign_commit_only:
        cur_repo["_target_is_commit"] = (cur_repo["type"].astype(str) == "Commit").astype(float)

    # Feature engineering: derive 16 temporal features + z-score normalisation
    all_set = set()
    cur_repo = fix_repo_shape(all_set, cur_repo, metadata=use_metadata)

    # Identify target commits
    vulns = cur_repo.index[cur_repo["Vuln"] == 1].tolist()
    if not vulns:
        logger.info(f"Skipping {repo_name}: no vuln commits after processing")
        return None, None, repo_name

    benign_mask = cur_repo["Vuln"] == 0
    if benign_commit_only:  # rerun_v3 (Commit-only benign targets): candidates restricted to Commit events
        benign_mask = benign_mask & (cur_repo["_target_is_commit"] == 1)
        cur_repo = cur_repo.drop(["_target_is_commit"], axis=1)
    if realistic:
        # REALISTIC mode: test on ALL non-vuln commits (natural class imbalance)
        benigns = cur_repo.index[benign_mask].tolist()
    else:
        # BALANCED mode: sample benign commits at the specified ratio
        benigns = cur_repo.index[benign_mask].tolist()
        # RUNBOOK_v2: fixed, per-repo seeded sample so the external commit list is
        # identical across every seed/model/mode (previously unseeded random.shuffle).
        random.Random(f"{sample_seed}:{repo_name}").shuffle(benigns)
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
    vuln_windows, vuln_kept = [], []
    for event_idx in vulns:
        try:
            res = get_event_window(cur_repo, event_idx, aggr_options, backs=backs)
            if res is not None and len(res) > 0:
                vuln_windows.append(res)
                vuln_kept.append(event_idx)
        except Exception:
            continue  # Skip windows that fail (e.g. too close to sequence boundary)

    # Extract windows for benign commits
    benign_windows, benign_kept = [], []
    for event_idx in benigns:
        try:
            res = get_event_window(cur_repo, event_idx, aggr_options, backs=backs)
            if res is not None and len(res) > 0:
                benign_windows.append(res)
                benign_kept.append(event_idx)
        except Exception:
            continue

    if not vuln_windows:
        logger.info(f"Skipping {repo_name}: no valid vuln windows extracted")
        return None, None, repo_name

    # Combine vuln and benign windows with labels
    windows = vuln_windows + benign_windows
    labels = [1] * len(vuln_windows) + [0] * len(benign_windows)

    # RUNBOOK_v2: provenance per window, verified against timestamp + label
    targets = [(e[0], e[1], lab) for e, lab in zip(vuln_kept + benign_kept, labels)]
    events = ordered_events_from_frame(prov_raw, CVEFIXES_DIR, repo_name)
    hashes, types, oks = lookup(events, targets)
    prov = [(repo_name, h, t, str(e[0]), int(e[1]), ok)
            for h, t, ok, e in zip(hashes, types, oks, vuln_kept + benign_kept)]

    return list(zip(windows, labels, prov)), len(vuln_windows), repo_name


# =============================================================================
# _pad_windows — Pad windows to the model's input shape
# =============================================================================

def _pad_windows(windows, expected_seq, expected_features):
    """Pad a list of variable-length windows to the model's input shape.

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

    return X


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
    model_dir = make_model_dir_name(args.aggregate, args.backs, args.meta, seed, args.model_comment)
    if not args.merge_chunks:
        fold_models = load_ensemble(model_dir, args.model, args.legacy_last_fold)
        expected_features = fold_models[0].input_shape[-1]
        expected_seq = fold_models[0].input_shape[1]
    lo, hi = (int(x) for x in args.repo_range.split(":")) if args.repo_range else (0, len(csv_files))

    # Process repos one at a time to manage memory
    all_preds = []
    all_logits = []
    all_labels = []
    all_samples_prov = []
    repos_processed = 0
    repos_skipped = 0
    total_vulns = 0
    total_benigns = 0

    for i, csv_path in enumerate(csv_files):
        if args.merge_chunks or not lo <= i < hi:
            continue  # --repo-range: another process scores this repo; --merge-chunks: all already scored
        if args.reload_every and i > 0 and i % args.reload_every == 0:
            fold_models = reload_ensemble(fold_models, model_dir, args)
        samples, n_vulns, repo_name = process_repo(
            csv_path, args.aggregate, args.backs, metadata_dict, language_vocab,
            realistic=True, sample_seed=args.sample_seed, benign_commit_only=args.benign_commit_only,
        )
        if samples is None or len(samples) == 0:
            repos_skipped += 1
            continue

        # Extract windows and labels for this repo
        windows = [s[0] for s in samples]
        labels = [s[1] for s in samples]
        n_benign = len(labels) - n_vulns

        # Run inference for this repo's windows
        X_repo = _pad_windows(windows, expected_seq, expected_features)
        pred, logit, _ = (ensemble_predict_padded if args.pad_batches else ensemble_predict)(fold_models, X_repo)
        all_preds.extend(pred.tolist())
        all_logits.extend(logit.tolist())
        all_samples_prov.extend(s[2] for s in samples)  # provenance only, not windows
        all_labels.extend(labels)
        total_vulns += n_vulns
        total_benigns += n_benign
        repos_processed += 1

        # Progress reporting every 20 repos
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(csv_files)}] {repos_processed} repos, "
                  f"{total_vulns} vuln + {total_benigns} benign commits so far")

    if args.repo_range:  # chunk process: save the raw per-repo results; the merge run reports
        import pickle
        with open(args.chunk_out, "wb") as f:
            pickle.dump({"range": (lo, hi), "csv_files": [os.path.basename(c) for c in csv_files], "preds": all_preds,
                         "logits": all_logits, "labels": all_labels, "prov": all_samples_prov,
                         "repos_processed": repos_processed, "repos_skipped": repos_skipped,
                         "total_vulns": total_vulns, "total_benigns": total_benigns}, f)
        print(f"[CHUNK] repos {lo}:{hi} of {len(csv_files)}: {repos_processed} processed, {repos_skipped} skipped, "
              f"{len(all_labels)} samples -> {args.chunk_out}")
        return
    if args.merge_chunks:  # same lists, in the same repo order, as one process would have built
        import pickle
        chunks = []
        for path in args.merge_chunks.split(","):
            with open(path, "rb") as f:
                chunks.append(pickle.load(f))
        chunks.sort(key=lambda c: c["range"][0])
        names = [os.path.basename(c) for c in csv_files]
        edges = [c["range"] for c in chunks]
        if edges[0][0] != 0 or edges[-1][1] != len(csv_files) or any(a[1] != b[0] for a, b in zip(edges, edges[1:])) \
                or any(c["csv_files"] != names for c in chunks):
            print(f"[MERGE] ERROR: chunks {edges} do not cover repos 0:{len(csv_files)} exactly, or list differs")
            sys.exit(4)
        for c in chunks:
            all_preds += c["preds"]; all_logits += c["logits"]; all_labels += c["labels"]
            all_samples_prov += c["prov"]
            repos_processed += c["repos_processed"]; repos_skipped += c["repos_skipped"]
            total_vulns += c["total_vulns"]; total_benigns += c["total_benigns"]
        print(f"[MERGE] {len(chunks)} chunks {edges} -> {len(all_labels)} samples")

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

    prov = stack_provenance(all_samples_prov)
    print(f"[PROVENANCE] {len(prov['repo'])} rows, {int(prov['prov_verified'].sum())} verified, "
          f"{int((prov['hash'] != '').sum())} with a commit hash")
    print(f"[COMMIT-LIST] n={len(y_test)} sha256={commit_list_fingerprint(prov, y_test)}")
    if args.save_preds:
        np.savez(args.save_preds, scores=pred, logits=np.array(all_logits), labels=y_test.astype(int),
                 threshold=args.threshold, **prov)
        print(f"Saved external predictions to {args.save_preds}")

    # Threshold sweep on the EXTERNAL labels. Leakage-free patch: this is kept only
    # as a diagnostic (it tunes the threshold on the test labels). The reportable
    # numbers are the ones above, at the fixed/validation threshold.
    print("=== THRESHOLD SWEEP [DIAGNOSTIC ONLY - tuned on external labels, do not report] ===")
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

    print(f"\n[DIAGNOSTIC ONLY] Best F1 on external labels: {best_f1:.4f} at threshold={best_thr:.2f}")

    # Final summary
    print()
    print("=" * 60)
    print(f"SUMMARY (REALISTIC): {args.model} | {args.aggregate.value} | B={args.backs} | seed={seed}")
    print(f"  Test repos: {repos_processed}")
    print(f"  Total commits tested: {len(all_labels)} ({total_vulns} vuln, {total_benigns} benign)")
    print(f"  Imbalance: 1:{total_benigns / max(total_vulns, 1):.1f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Avg Prec:  {ap:.4f}")
    thr = args.threshold
    print(f"  [LEAKAGE-FREE] threshold={thr:.2f}  F1={f1:.4f}  Acc={acc:.4f}")
    print(f"  Recall@{thr:.2f}:    {rec:.4f} (catches {tp} of {total_vulns} vuln commits)")
    print(f"  Precision@{thr:.2f}: {prec:.4f} ({tp} true alerts out of {tp + fp} total alerts)")
    print(f"  [DIAGNOSTIC ONLY, tuned on external labels] Best F1: {best_f1:.4f} at threshold={best_thr:.2f}")


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
    # Leakage-free patch: take the decision threshold from the internal validation
    # pool saved by train.py (results/preds_<exp>_<model>.npz, key tau_val) instead
    # of choosing it on the external labels.
    parser.add_argument("--tau-from-npz", type=str, default=None,
                        help="Read tau_val from a train.py preds .npz and use it as --threshold")
    parser.add_argument("--save-preds", type=str, default=None,
                        help="Save external scores+labels to this .npz (re-thresholding without re-running)")
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--benign-commit-only", action="store_true",
                        help="rerun_v3: benign targets only from Commit events (needs 'commitonly' in --model-comment)")
    parser.add_argument("--model-comment", default="",
                        help="rerun_v3: the train.py --comment of the models (e.g. commitonly); seed suffix added as in training")
    parser.add_argument("--sample-seed", type=int, default=0,
                        help="RUNBOOK_v2: seed for the per-repo benign sample (balanced mode)")
    parser.add_argument("--legacy-last-fold", action="store_true",
                        help="Use only models/<dir>/<model>.keras (pre-RUNBOOK_v2; comparison only)")
    parser.add_argument("--enriched", action="store_true",
                        help="Use enriched CSVs from cvefixes_test/enriched/")
    parser.add_argument("--repo-range", default=None,
                        help="realistic mode: score only repos A:B (0-based, end excluded) and save the raw results "
                             "to --chunk-out; a later --merge-chunks run reports (memory fix, runs_v2/NOTES.md s12)")
    parser.add_argument("--chunk-out", default=None, help="output .pkl for --repo-range")
    parser.add_argument("--merge-chunks", default=None,
                        help="realistic mode: comma-separated --chunk-out files; no scoring, the results are joined in "
                             "repo order and reported exactly as a single run")
    parser.add_argument("--reload-every", type=int, default=0,
                        help="realistic mode: reload the fold models every N repos to free compiled GPU programs "
                             "(memory fix, runs_v2/NOTES.md s11); scores unchanged; 0 = never")
    parser.add_argument("--pad-batches", action="store_true",
                        help="realistic mode: pad each repo's windows to a multiple of 256 rows so the GPU compiles "
                             "each model once (memory fix, runs_v2/NOTES.md s9); padded rows are dropped")
    parser.add_argument("--realistic", action="store_true",
                        help="Test on ALL commits (natural imbalance, no 1:1 sampling)")
    parser.add_argument("--exclude-train-overlap", action="store_true",
                        help="Drop CVEfixes test repos whose flat name (case-insensitive) "
                             "matches any training repo CSV in data_collection/gh_cve_proccessed/.")
    args = parser.parse_args()
    # rerun_v3 (Commit-only benign targets): Commit-only models and Commit-only targets always go together
    if args.benign_commit_only != ("commitonly" in args.model_comment):
        print(f"FATAL: --benign-commit-only={args.benign_commit_only} but --model-comment={args.model_comment!r}")
        sys.exit(2)
    if args.benign_commit_only:
        print(f"[COMMIT-ONLY] benign targets restricted to Commit events; models from *_{args.model_comment}*")

    if args.tau_from_npz:
        if not os.path.exists(args.tau_from_npz):
            print(f"FATAL: --tau-from-npz file not found: {args.tau_from_npz}")
            sys.exit(2)
        args.threshold = float(np.load(args.tau_from_npz)["tau_val"])
        print(f"[LEAKAGE-FREE] threshold = tau_val = {args.threshold:.4f} "
              f"(selected on internal validation, from {args.tau_from_npz})")
    else:
        print(f"[THRESHOLD] fixed threshold = {args.threshold:.2f} (not tuned on external labels)")

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

    # Optional: exclude any CVEfixes test repo that overlaps with training, where
    # overlap is detected by either (a) case-insensitive name match (catches
    # casing-only GitHub renames such as 'facebookresearch_ParlAI' vs training
    # 'facebookresearch_parlai'), or (b) commit-hash intersection (catches owner
    # renames such as 'util-linux_util-linux' vs training 'karelzak_util-linux'
    # — same repo, different owner). The hash check uses a 1% overlap threshold
    # against a sample of 10k training hashes to keep memory bounded.
    if args.exclude_train_overlap:
        import csv as _csv
        train_dir = os.path.join("data_collection", "gh_cve_proccessed")
        train_files = [f for f in os.listdir(train_dir) if f.endswith(".csv")]
        train_lc = {f[:-4].lower() for f in train_files}

        # Build training hash set (any commit hash from any training CSV).
        print("Building training-hash index for owner-rename detection...")
        train_hashes = set()
        for tfile in train_files:
            with open(os.path.join(train_dir, tfile), errors="ignore") as fh:
                for row in _csv.DictReader(fh):
                    h = (row.get("Hash", "") or row.get("hash", "")).strip()
                    if h and len(h) >= 7:
                        train_hashes.add(h)
        print(f"  Indexed {len(train_hashes)} distinct training commit hashes")

        before = len(csv_files)
        kept = []
        dropped_name = 0
        dropped_hash = []
        HASH_OVERLAP_THRESHOLD = 0.01  # 1% of test hashes also in training => same repo
        for p in csv_files:
            name = os.path.basename(p)[:-4]
            if name.lower() in train_lc:
                dropped_name += 1
                continue
            # Check commit-hash overlap with training
            test_hashes = set()
            with open(p, errors="ignore") as fh:
                for row in _csv.DictReader(fh):
                    h = (row.get("Hash", "") or row.get("hash", "")).strip()
                    if h and len(h) >= 7:
                        test_hashes.add(h)
            if test_hashes:
                ov = len(test_hashes & train_hashes) / len(test_hashes)
                if ov >= HASH_OVERLAP_THRESHOLD:
                    dropped_hash.append((name, ov, len(test_hashes & train_hashes)))
                    continue
            kept.append(p)
        csv_files = kept
        print(f"Excluded {dropped_name} repos by case-insensitive name match")
        if dropped_hash:
            print(f"Excluded {len(dropped_hash)} repos by hash overlap >= "
                  f"{HASH_OVERLAP_THRESHOLD:.0%} (owner-rename detection):")
            for n, ov, cnt in dropped_hash:
                print(f"    {n}: {ov:.1%} ({cnt} commit hashes shared with training)")
        print(f"Total excluded: {before - len(csv_files)}. "
              f"Remaining: {len(csv_files)} truly-unseen repos.")

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
            sample_seed=args.sample_seed, benign_commit_only=args.benign_commit_only,
        )
        if samples is None or len(samples) == 0:
            repos_skipped += 1
            continue

        all_samples.extend(samples)
        total_vulns += n_vulns
        repos_processed += 1
        logger.info(f"  {repo_name}: {n_vulns} vuln + {len(samples) - n_vulns} benign windows")

    print(f"\nProcessed {repos_processed} repos, skipped {repos_skipped}")
    n_vuln_samples = sum(1 for s in all_samples if s[1] == 1)
    n_benign_samples = sum(1 for s in all_samples if s[1] == 0)
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
    prov = stack_provenance([s[2] for s in all_samples])
    print(f"[PROVENANCE] {len(prov['repo'])} rows, {int(prov['prov_verified'].sum())} verified, "
          f"{int((prov['hash'] != '').sum())} with a commit hash")
    print(f"[COMMIT-LIST] n={len(y_test)} vuln={n_vuln_samples} benign={n_benign_samples} "
          f"sha256={commit_list_fingerprint(prov, y_test)}")
    print()

    # ---- Run inference for each seed ----
    all_results = []

    for seed in seeds:
        # Locate the model checkpoint for this seed
        model_dir = make_model_dir_name(args.aggregate, args.backs, args.meta, seed, args.model_comment)
        print(f"--- Seed {seed} ---")
        fold_models = load_ensemble(model_dir, args.model, args.legacy_last_fold)
        model = fold_models[0]  # only used for input-shape checks below

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
        pred, logit, pred_folds = ensemble_predict(fold_models, X_eval)
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
        if n_benign_samples > 0 and n_vuln_samples > 0:
            # Threshold-free, so unaffected by threshold choice
            print(f"  ROC-AUC:   {roc_auc_score(y_test, pred):.4f}")
            print(f"  Avg Precision (AP): {average_precision_score(y_test, pred):.4f}")
        if args.save_preds:
            out = args.save_preds if len(seeds) == 1 else args.save_preds.replace(".npz", f"_s{seed}.npz")
            np.savez(out, scores=pred, logits=logit, pred_folds=pred_folds, labels=y_test.astype(int),
                     threshold=args.threshold, **prov)
            print(f"  Saved external predictions to {out}")
        print(f"  Mean prob (vuln):   {pred[y_test == 1].mean():.4f}")
        if n_benign_samples > 0:
            print(f"  Mean prob (benign): {pred[y_test == 0].mean():.4f}")
        print()

        auc_v = roc_auc_score(y_test, pred) if n_benign_samples and n_vuln_samples else float("nan")
        ap_v = average_precision_score(y_test, pred) if n_benign_samples and n_vuln_samples else float("nan")
        print(f"[RESULT] seed={seed} n={len(y_test)} tau={args.threshold:.4f} acc={acc:.4f} prec={prec:.4f} "
              f"rec={rec:.4f} f1={f1:.4f} roc_auc={auc_v:.4f} ap={ap_v:.4f}")
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
        print(f"SUMMARY (BALANCED 1:1): {args.model} | {args.aggregate.value} | B={args.backs} | meta={args.meta}"
              f" | threshold={args.threshold:.2f}{' (tau_val)' if args.tau_from_npz else ''}")
        print(f"  Seeds evaluated: {len(all_results)}")
        print(f"  Accuracy:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        print(f"  Precision: {np.mean(precs):.4f} +/- {np.std(precs):.4f}")
        print(f"  Recall:    {np.mean(recs):.4f} +/- {np.std(recs):.4f}")
        print(f"  F1 Score:  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
        print(f"  Test repos: {repos_processed}")
        print(f"  Test samples: {len(all_samples)} ({n_vuln_samples} vuln, {n_benign_samples} benign)")


if __name__ == "__main__":
    main()
