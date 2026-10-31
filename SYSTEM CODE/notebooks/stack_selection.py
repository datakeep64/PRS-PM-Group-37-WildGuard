"""Detectors and stack selection (DESIGN_QUESTIONS.md Part B, B6-B7), used by notebook 11.

B7  unsupervised candidates: the pipeline's peer-deviation score, Isolation Forest and K-Means
    distance (wildguard/models.py), plus a Gaussian mixture and DBSCAN defined here. All read the
    numeric deviation features; none uses labels.
B6  one stacking layer built by forward selection on 2011: start from the chosen supervised model,
    add the pool member that most improves PR-AUC on listed line items, stop when the best
    addition's gain has a 95% paired-bootstrap interval that includes zero.

Members are combined by a logistic-regression meta-learner over the logit of each member's
percentile rank. On 2011 it is cross-fitted: fitted on one half of the shipments and scored on
the other, so no record is scored by a meta-learner that saw it.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from wildguard.models import RANDOM_STATE, _deviation_matrix, _subsample


class GaussianMixtureScorer:
    """Soft clustering; a record's anomaly score is its negative log-likelihood. The number of
    components is chosen by BIC (B7)."""

    def __init__(self, features, k_grid=(2, 4, 6, 8, 10, 12), max_rows=100_000):
        self.features, self.k_grid, self.max_rows = features, k_grid, max_rows

    def fit(self, df):
        M = _deviation_matrix(df.iloc[_subsample(len(df), self.max_rows)], self.features)
        self.scaler = StandardScaler().fit(M)
        Z = self.scaler.transform(M)
        self.bic = {}
        for k in self.k_grid:
            gm = GaussianMixture(k, covariance_type="full", random_state=RANDOM_STATE).fit(Z)
            self.bic[k] = float(gm.bic(Z))
        self.k = min(self.bic, key=self.bic.get)
        self.model = GaussianMixture(self.k, covariance_type="full", random_state=RANDOM_STATE).fit(Z)
        return self

    def score(self, df):
        return -self.model.score_samples(self.scaler.transform(_deviation_matrix(df, self.features)))


class DBSCANScorer:
    """Density-based clustering on a sample (B7). Neighbourhood: 10 points; radius: the 95th
    percentile of each sample point's distance to its 10th neighbour. A record's anomaly score is
    its distance to the nearest core point, so DBSCAN can score records it was not fitted on."""

    def __init__(self, features, min_samples=10, max_rows=30_000):
        self.features, self.min_samples, self.max_rows = features, min_samples, max_rows

    def fit(self, df):
        M = _deviation_matrix(df.iloc[_subsample(len(df), self.max_rows)], self.features)
        self.scaler = StandardScaler().fit(M)
        Z = self.scaler.transform(M)
        d10 = NearestNeighbors(n_neighbors=self.min_samples).fit(Z).kneighbors(Z)[0][:, -1]
        self.eps = float(np.percentile(d10, 95))
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(Z)
        self.n_clusters = int(db.labels_.max() + 1)
        self.noise_share = float((db.labels_ == -1).mean())
        self.tree = cKDTree(Z[db.core_sample_indices_])
        return self

    def score(self, df):
        return self.tree.query(self.scaler.transform(_deviation_matrix(df, self.features)))[0]


# ---------------------------------------------------------------- stacking (B6)
def logit_rank(s):
    """Logit of each score's percentile rank: keeps every member's ordering, puts members on one
    scale, and stretches the tails where PR-AUC is decided."""
    p = pd.Series(s).rank(pct=True).to_numpy()
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def cross_fit(Z, y, shipments, seed=37):
    """Out-of-fold meta-learner scores: two halves of the shipments, each scored by a logistic
    regression fitted on the other."""
    codes, uniq = pd.factorize(shipments)
    half = np.random.default_rng(seed).integers(0, 2, len(uniq))[codes]
    out = np.zeros(len(y))
    for h in (0, 1):
        m = LogisticRegression(max_iter=500).fit(Z[half != h], y[half != h])
        out[half == h] = m.predict_proba(Z[half == h])[:, 1]
    return out


def forward_selection(members, y, shipments, listed, start, pool, bootstrap, log=print):
    """B6. `members`: name -> 2011 score array. Returns the selected names and one row per step."""
    Z = {k: logit_rank(v) for k, v in members.items()}
    selected, steps = [start], []
    current = cross_fit(np.column_stack([Z[start]]), y, shipments)
    while True:
        rest = [p for p in pool if p not in selected]
        if not rest:
            break
        trial = {p: cross_fit(np.column_stack([Z[k] for k in selected + [p]]), y, shipments)
                 for p in rest}
        ap = {p: average_precision_score(y[listed], s[listed]) for p, s in trial.items()}
        best = max(ap, key=ap.get)
        b = bootstrap(y[listed], {"with": trial[best][listed], "without": current[listed]},
                      shipments[listed], [("with", "without")]).iloc[0]
        added = bool(b["low_95"] > 0)
        steps.append({"step": len(steps) + 1, "current members": " + ".join(selected),
                      "best addition": best, "PR-AUC before": b["pr_auc_b"],
                      "PR-AUC with it": b["pr_auc_a"], "gain": b["difference"],
                      "95% low": b["low_95"], "95% high": b["high_95"], "added": added,
                      **{f"PR-AUC with {p}": v for p, v in ap.items()}})
        log(f"step {len(steps)}: best addition {best}, gain {b['difference']:.4f} "
            f"[{b['low_95']:.4f}, {b['high_95']:.4f}] -> {'added' if added else 'stop'}")
        if not added:
            break
        selected.append(best)
        current = trial[best]
    return selected, pd.DataFrame(steps)
