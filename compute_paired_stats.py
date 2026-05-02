"""Compute paired statistical tests (Acc, F1, AUC) from per-fold logs.

Usage:
    python compute_paired_stats.py <transformer_log> <conv1d_log>

Extracts the per-fold "Acc - X", "F1 - X", "AUC - X" critical lines from each
log (one set per fold; the trailing two acc lines are the ensemble single-best
and full-ensemble test results, which are skipped). Computes paired t-test,
Wilcoxon signed-rank, bootstrap 95% CI, Cohen's d for the within-pair diffs.
"""
import re
import sys
import numpy as np
from scipy import stats


PER_FOLD_RE = re.compile(r"CRITICAL\s+(F1|Acc|AUC)\s+-\s+([0-9.eE+-]+)")


def parse_log(path: str, n_folds: int = 10) -> dict[str, np.ndarray]:
    """Return {'acc': [..n_folds..], 'f1': [..], 'auc': [..]} per fold.

    The check_results emits F1, Acc, AUC for each fold (val set), then again
    for the single-best on test, and again for the ensemble on test. We keep
    only the first n_folds occurrences of each metric — those are the per-fold
    validation scores.
    """
    f1s, accs, aucs = [], [], []
    with open(path) as f:
        for line in f:
            m = PER_FOLD_RE.search(line)
            if not m:
                continue
            metric, val = m.group(1), float(m.group(2))
            if metric == "F1":
                f1s.append(val)
            elif metric == "Acc":
                accs.append(val)
            elif metric == "AUC":
                aucs.append(val)
    return {
        "acc": np.array(accs[:n_folds], dtype=float),
        "f1": np.array(f1s[:n_folds], dtype=float),
        "auc": np.array(aucs[:n_folds], dtype=float),
    }


def cohens_d_paired(diff: np.ndarray) -> float:
    """Cohen's d for paired samples: mean(diff) / std(diff)."""
    return float(np.mean(diff) / np.std(diff, ddof=1))


def bootstrap_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(diff, size=n, replace=True)
        means[i] = sample.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(name: str, t: np.ndarray, c: np.ndarray) -> None:
    diff = t - c
    t_mean, c_mean = t.mean(), c.mean()
    t_std, c_std = t.std(ddof=1), c.std(ddof=1)
    t_stat, t_p = stats.ttest_rel(t, c)
    w_stat, w_p = stats.wilcoxon(t, c)
    d = cohens_d_paired(diff)
    lo, hi = bootstrap_ci(diff)

    print(f"\n=== {name} ===")
    print(f"  Transformer per-fold: {np.round(t, 4).tolist()}")
    print(f"  Conv1D      per-fold: {np.round(c, 4).tolist()}")
    print(f"  Transformer mean ± std: {t_mean:.4f} ± {t_std:.4f}")
    print(f"  Conv1D      mean ± std: {c_mean:.4f} ± {c_std:.4f}")
    print(f"  Mean diff (T - C):      {diff.mean():.4f}")
    print(f"  Paired t-test:    t={t_stat:.4f}, p={t_p:.6f}")
    print(f"  Wilcoxon:         W={w_stat:.4f}, p={w_p:.6f}")
    print(f"  Bootstrap 95% CI: [{lo:.4f}, {hi:.4f}]")
    print(f"  Cohen's d:        {d:.4f}")


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: compute_paired_stats.py <transformer_log> <conv1d_log>")
        sys.exit(1)

    t_path, c_path = sys.argv[1], sys.argv[2]
    t = parse_log(t_path)
    c = parse_log(c_path)

    print(f"Transformer log: {t_path}")
    print(f"Conv1D      log: {c_path}")
    print(f"Folds parsed: T={len(t['acc'])}/{len(t['f1'])}/{len(t['auc'])}, "
          f"C={len(c['acc'])}/{len(c['f1'])}/{len(c['auc'])}")

    if not (len(t['acc']) == len(c['acc']) == 10):
        print(f"WARNING: expected 10 folds each, got differently — stats may be partial")

    report("Accuracy", t["acc"], c["acc"])
    report("F1",       t["f1"],  c["f1"])
    report("ROC-AUC",  t["auc"], c["auc"])


if __name__ == "__main__":
    main()
