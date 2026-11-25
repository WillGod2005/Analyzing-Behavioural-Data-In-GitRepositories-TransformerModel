"""
Leakage-free threshold selection and evaluation for the behavioural
security-patch detector.

Why this exists
---------------
In the original pipeline, train.py -> check_results() calls
find_best_accuracy(pred, y_test) and find_best_f1(pred, y_test). Those
sweep 100 thresholds and return the MAXIMUM score on the labels passed in.
In the final-evaluation calls the labels passed in are the TEST labels
(train.py lines ~952 and ~956). So the reported accuracy and F1 are the
best-over-thresholds on the test set. That tunes the decision threshold on
the test labels, which is an optimistic bias.

This module selects the threshold on VALIDATION data only and applies it to
the test predictions. ROC-AUC and average precision are threshold-free and
are reported unchanged.
"""
import numpy as np
from sklearn.metrics import (f1_score, confusion_matrix, roc_auc_score,
                             average_precision_score)


def _sweep(pred, y, metric):
    pred = np.asarray(pred).reshape(-1)
    y = np.asarray(y).astype(int)
    best_t, best_v = 0.5, -1.0
    for i in range(1, 100):
        t = i / 100.0
        yp = (pred > t).astype(int)
        v = (yp == y).mean() if metric == "accuracy" else f1_score(y, yp, zero_division=0)
        if v > best_v:
            best_v, best_t = v, t
    return best_t


def select_threshold_on_val(val_pred, val_y, metric="f1"):
    """Return the threshold that maximises `metric` on VALIDATION data."""
    return _sweep(val_pred, val_y, metric)


def metrics_at_threshold(pred, y, tau):
    """All metrics at a FIXED threshold tau (no sweeping on these labels)."""
    pred = np.asarray(pred).reshape(-1)
    y = np.asarray(y).astype(int)
    yp = (pred > tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(threshold=float(tau), accuracy=acc, precision=prec, recall=rec,
                specificity=spec, f1=f1,
                roc_auc=float(roc_auc_score(y, pred)),
                avg_precision=float(average_precision_score(y, pred)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def leakage_free_report(val_pred, val_y, test_pred, test_y, metric_for_threshold="f1"):
    """Corrected metrics, plus the old biased number, plus the inflation gap."""
    tau = select_threshold_on_val(val_pred, val_y, metric_for_threshold)
    corrected = metrics_at_threshold(test_pred, test_y, tau)
    leaked_tau = _sweep(test_pred, test_y, "accuracy")   # what the old code did
    leaked = metrics_at_threshold(test_pred, test_y, leaked_tau)
    return dict(tau_val=tau, corrected=corrected, leaked_best_on_test=leaked,
                accuracy_inflation=leaked["accuracy"] - corrected["accuracy"],
                f1_inflation=leaked["f1"] - corrected["f1"])
