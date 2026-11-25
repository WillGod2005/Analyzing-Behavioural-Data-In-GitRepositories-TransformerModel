"""
Operational metrics under realistic class imbalance (addresses R1.2 / R2.4).

Input: a .npz or .csv with per-commit scores and labels for the full,
unbalanced commit population (e.g. the 2.09M-commit full-corpus run).
  .npz: keys `scores` (float in [0,1]) and `labels` (0/1); optionally `logits`
  .csv: columns `score,label`

Outputs: precision-recall curve points, precision at fixed alert budgets,
and expected alert volume. These are the measures the reviewers asked for,
which accuracy and a single-threshold precision number do not capture.

RUNBOOK_v2 Task 2 changes:
  * --score-key chooses the ranking score; default is `logits` when present
    (probabilities saturate at 1.0 in float32, creating large ties).
  * For every budget k, reports how many commits tie with the k-th score
    (in the whole corpus and inside the top k), and the best/worst number of
    true fixes the top k could contain under any ordering of those ties.
  * Ties are broken randomly with a fixed seed (--tie-seed, default 0); the
    reported precision@k uses that ordering.  (The previous version used
    np.argsort(-scores), whose order among ties is arbitrary.)

Usage:
  python3 operational_metrics.py --preds results/fullcorpus_preds.npz \
      --repos 212 --days 365 --budgets 100 500 1000 5000
"""
import argparse
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score


def load(path, score_key=None):
    """Return (scores, labels, key_used, threshold_or_None)."""
    if path.endswith(".npz"):
        d = np.load(path)
        if score_key is None:
            score_key = "logits" if "logits" in d else ("scores" if "scores" in d else "test_pred")
        key_y = "labels" if "labels" in d else ("test_y" if "test_y" in d else None)
        thr = float(d["threshold"]) if "threshold" in d else None
        return (np.asarray(d[score_key], dtype=np.float64).reshape(-1),
                np.asarray(d[key_y]).astype(int).reshape(-1), score_key, thr)
    import csv
    s, y = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            s.append(float(row["score"])); y.append(int(float(row["label"])))
    return np.asarray(s), np.asarray(y), "score", None


def ranked_order(scores, tie_seed=0):
    """Indices sorted by descending score; ties broken by a fixed-seed random key."""
    rng = np.random.default_rng(tie_seed)
    tiebreak = rng.permutation(len(scores))
    return np.lexsort((tiebreak, -scores))


def precision_at_budget(scores, labels, k, order):
    """Top-k review budget with tie accounting.

    Returns dict: precision and hits under the fixed-seed tie-break, the k-th
    score, how many commits share it (corpus-wide and inside the top k), and
    the min/max hits achievable by any ordering of the tied commits.
    """
    k = min(k, len(scores))
    top = order[:k]
    hits = int(labels[top].sum())
    kth = scores[top[-1]]
    tied_all = scores == kth
    n_tied_total = int(tied_all.sum())
    n_tied_in_topk = int((scores[top] == kth).sum())
    strictly_above = scores > kth
    hits_above = int(labels[strictly_above].sum())
    pos_in_tied = int(labels[tied_all].sum())
    neg_in_tied = n_tied_total - pos_in_tied
    best = hits_above + min(n_tied_in_topk, pos_in_tied)
    worst = hits_above + max(0, n_tied_in_topk - neg_in_tied)
    return {"k": k, "precision": hits / k, "hits": hits, "kth_score": float(kth),
            "n_tied_total": n_tied_total, "n_tied_in_topk": n_tied_in_topk,
            "hits_min": worst, "hits_max": best}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--repos", type=int, default=None, help="repo count for per-repo rates")
    ap.add_argument("--days", type=int, default=365, help="time span the commits cover")
    ap.add_argument("--budgets", type=int, nargs="+", default=[100, 500, 1000, 5000])
    ap.add_argument("--score-key", default=None, help="npz key to rank by (default: logits if present)")
    ap.add_argument("--tie-seed", type=int, default=0, help="seed for random tie-breaking")
    ap.add_argument("--out-prefix", default="", help="prefix for pr_curve.csv / budgets.csv")
    args = ap.parse_args()

    s, y, key, npz_thr = load(args.preds, args.score_key)
    n, pos = len(y), int(y.sum())
    is_logit = key == "logits"
    print(f"ranking score: {key} (tie-break seed {args.tie_seed})")
    print(f"commits={n}  positives={pos}  prevalence=1:{(n-pos)//max(pos,1)}")
    print(f"ROC-AUC={roc_auc_score(y, s):.4f}  average_precision={average_precision_score(y, s):.4f}")
    print(f"distinct score values: {len(np.unique(s))}")

    order = ranked_order(s, args.tie_seed)
    print("\nPrecision at fixed review budget (top-k by score, ties broken randomly with fixed seed):")
    print(f"{'budget':>8} {'precision':>10} {'true_fixes':>11} {'recall':>8} {'kth_score':>12} "
          f"{'tied@kth':>9} {'tied_in_topk':>13} {'fixes_min':>10} {'fixes_max':>10}")
    rows = []
    for k in args.budgets:
        r = precision_at_budget(s, y, k, order)
        rows.append(r)
        print(f"{r['k']:>8} {r['precision']:>10.4f} {r['hits']:>11} {r['hits']/max(pos,1):>8.4f} "
              f"{r['kth_score']:>12.6g} {r['n_tied_total']:>9} {r['n_tied_in_topk']:>13} "
              f"{r['hits_min']:>10} {r['hits_max']:>10}")

    # PR curve summary points
    prec, rec, thr = precision_recall_curve(y, s)
    print("\nPR curve (selected recall levels):")
    for target in [0.3, 0.5, 0.7, 0.9]:
        idx = np.argmin(np.abs(rec - target))
        print(f"  recall~{target:.1f}: precision={prec[idx]:.4f} at threshold "
              f"{thr[min(idx, len(thr)-1)]:.3f}{' (logit)' if is_logit else ''}")

    # Alert volume at probability thresholds (converted to logit space if ranking by logits)
    if args.repos:
        probs = [t for t in ([npz_thr] if npz_thr is not None else []) + [0.5, 0.9]]
        for t in probs:
            cut = np.log(t / (1 - t)) if is_logit else t
            flagged = int((s >= cut).sum())
            label = " (tau_val from npz)" if npz_thr is not None and t == npz_thr else ""
            print(f"\nAt probability threshold {t:.2f}{label}: {flagged} alerts total, "
                  f"{flagged/args.repos:.1f} per repo, "
                  f"{flagged/args.days:.1f} per day across the corpus")

    np.savetxt(f"{args.out_prefix}pr_curve.csv",
               np.column_stack([np.append(thr, np.inf if is_logit else 1.0), prec, rec]),
               delimiter=",", header="threshold,precision,recall", comments="")
    with open(f"{args.out_prefix}budgets.csv", "w") as f:
        f.write(",".join(rows[0].keys()) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r.values()) + "\n")
    print(f"\nsaved {args.out_prefix}pr_curve.csv and {args.out_prefix}budgets.csv")


if __name__ == "__main__":
    main()
