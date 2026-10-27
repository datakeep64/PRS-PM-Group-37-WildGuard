"""Evaluation metrics, framed around the real use case: an inspector with a limited budget.

Accuracy is meaningless at a ~1.7% positive rate (predicting "cleared" for everything scores
98.3%), so we report ranking metrics (PR-AUC, ROC-AUC), F1 at a threshold fixed on the
validation years, and triage metrics: what share of real enforcement cases is caught if
only the top-k% highest-scored records are inspected, compared with random inspection.
"""
import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

BUDGETS = (0.01, 0.05, 0.10)


def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    return float(t[np.nanargmax(f1[:-1])])


def top_k_mask(s, frac):
    k = max(1, int(round(len(s) * frac)))
    order = np.argsort(-s, kind="stable")
    m = np.zeros(len(s), bool)
    m[order[:k]] = True
    return m


def inspection_needed(y, s, target_recall):
    """Fraction of records that must be inspected (in score order) to catch target_recall."""
    order = np.argsort(-s, kind="stable")
    cum = np.cumsum(y[order]) / max(y.sum(), 1)
    return float((np.searchsorted(cum, target_recall) + 1) / len(s))


def evaluate_scores(y, s, threshold):
    y = np.asarray(y)
    s = np.asarray(s)
    pred = s >= threshold
    base = y.mean()
    out = {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "base_rate": float(base),
    }
    for b in BUDGETS:
        m = top_k_mask(s, b)
        prec = y[m].mean()
        out[f"precision@{int(b * 100)}%"] = float(prec)
        out[f"recall@{int(b * 100)}%"] = float(y[m].sum() / max(y.sum(), 1))
        out[f"lift@{int(b * 100)}%"] = float(prec / base)
    order = np.argsort(-s, kind="stable")
    for k in (100, 1000):
        out[f"precision@{k}"] = float(y[order[:k]].mean())
    out["inspect_for_50%_recall"] = inspection_needed(y, s, 0.5)
    out["inspect_for_80%_recall"] = inspection_needed(y, s, 0.8)
    return out


def capture_curve(y, s, n_points=200):
    """(fraction inspected, fraction of positives caught) - a cumulative gains curve."""
    order = np.argsort(-s, kind="stable")
    cum = np.cumsum(np.asarray(y)[order]) / max(np.sum(y), 1)
    idx = np.unique(np.linspace(0, len(s) - 1, n_points).astype(int))
    return (idx + 1) / len(s), cum[idx]


def confidence_bands(y_val, s_val, recall_floor=0.95, priority_frac=0.02):
    """Three-band triage thresholds, fixed on the validation years:
         auto-clear  : score below t_low  (chosen so >= recall_floor of validation positives
                       score above it)
         priority    : top priority_frac of validation scores
         review      : everything in between
    """
    pos = np.sort(np.asarray(s_val)[np.asarray(y_val) == 1])
    t_low = float(pos[int(np.floor((1 - recall_floor) * len(pos)))])
    t_high = float(np.quantile(s_val, 1 - priority_frac))
    return t_low, t_high


def band_report(y, s, t_low, t_high):
    y, s = np.asarray(y), np.asarray(s)
    bands = np.where(s >= t_high, "priority", np.where(s >= t_low, "review", "auto-clear"))
    rep = {}
    for b in ("auto-clear", "review", "priority"):
        m = bands == b
        rep[b] = {"share_of_records": float(m.mean()),
                  "share_of_positives": float(y[m].sum() / max(y.sum(), 1)),
                  "positive_rate": float(y[m].mean()) if m.any() else 0.0}
    return rep
