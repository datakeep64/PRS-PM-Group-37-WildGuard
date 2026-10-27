"""Step 3-5 - the pattern-recognition components.

Supervised (trained on LEMIS enforcement outcomes)
    XGBoost       gradient-boosted tree ensemble, native categorical support
    RandomForest  bagged tree ensemble (comparison model)
    LogisticReg   linear baseline (comparison model)

Unsupervised (no labels used - can run on any unlabelled trade stream, e.g. CITES)
    PeerDeviation   interpretable statistical score: how far a record sits from its peer
                    group's quantity / unit-value norms, and how rare its route, purpose and
                    source are for that taxon
    IsolationForest tree-based outlier detector on the same deviation features
    KMeansDistance  clusters trade profiles; anomaly = distance to nearest centroid,
                    normalised by that cluster's typical spread (silhouette used to pick k)

Hybrid
    HybridStacker   logistic-regression meta-learner over [supervised logit + percentile
                    ranks of the 3 unsupervised scores], fitted on the validation years.
                    Output is a calibrated probability used for the review-priority tier.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

from .config import RANDOM_STATE
from .features import to_model_matrix

XGB_PARAMS = dict(
    objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
    enable_categorical=True, max_cat_to_onehot=8, max_depth=8, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
    n_jobs=8, random_state=RANDOM_STATE,
)


# ------------------------------------------------------------------------ supervised
def train_xgb(X, y, X_val, y_val, sample_weight=None, n_estimators=1500):
    pos = max(y.sum(), 1)
    model = xgb.XGBClassifier(
        **XGB_PARAMS, n_estimators=n_estimators, early_stopping_rounds=60,
        scale_pos_weight=float(np.sqrt((len(y) - pos) / pos)))
    model.fit(X, y, sample_weight=sample_weight, eval_set=[(X_val, y_val)], verbose=False)
    return model


def _ordinal(X):
    """Trees in scikit-learn need numeric input: categories -> integer codes (-1 = missing)."""
    X = X.copy()
    for c in X.columns:
        if isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].cat.codes
    return X.fillna(-999)


def train_random_forest(X, y, max_rows=600_000):
    idx = _subsample(len(X), max_rows)
    rf = RandomForestClassifier(n_estimators=300, max_depth=18, min_samples_leaf=5,
                                class_weight="balanced_subsample", n_jobs=8,
                                random_state=RANDOM_STATE)
    rf.fit(_ordinal(X.iloc[idx]), y[idx])
    return rf


def predict_random_forest(rf, X):
    return rf.predict_proba(_ordinal(X))[:, 1]


def train_logistic(X, y, max_rows=600_000):
    idx = _subsample(len(X), max_rows)
    cats = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    nums = [c for c in X.columns if c not in cats]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cats),
        ("num", StandardScaler(), nums)])
    lr = make_pipeline(pre, LogisticRegression(max_iter=300, C=0.5, class_weight="balanced"))
    Xs = X.iloc[idx].copy()
    for c in cats:
        Xs[c] = Xs[c].astype(str)
    Xs[nums] = Xs[nums].fillna(0)
    lr.fit(Xs, y[idx])
    return lr


def predict_logistic(lr, X):
    X = X.copy()
    cats = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    for c in cats:
        X[c] = X[c].astype(str)
    nums = [c for c in X.columns if c not in cats]
    X[nums] = X[nums].fillna(0)
    return lr.predict_proba(X)[:, 1]


def _subsample(n, k):
    rng = np.random.default_rng(RANDOM_STATE)
    return np.sort(rng.choice(n, size=min(n, k), replace=False))


# ---------------------------------------------------------------------- unsupervised
def _deviation_matrix(df, features):
    M = df[features].astype("float64").copy()
    for c in ("qty_z", "unit_value_z"):
        if c in M:
            M[c] = M[c].abs().clip(upper=15)
    return M.fillna(0).to_numpy()


class PeerDeviationScorer:
    """Transparent statistical anomaly score in [0, 1]. Needs no fitting: the peer-group
    norms it compares against are learned by FeatureBuilder from the reference population."""

    def __init__(self, use_value):
        self.use_value = use_value

    def score(self, df):
        parts = [
            np.minimum(df["qty_z"].abs().fillna(0), 10) / 10,                    # quantity vs peers
            np.clip(-np.log10(df["route_share"].fillna(0) + 1e-4) / 4, 0, 1),    # rare route
            np.clip(df["combo_rarity"].fillna(7) / 7, 0, 1),                     # rare form/purpose/source
            1 - df["purpose_share_in_genus"].fillna(0),                          # unusual purpose
            1 - df["source_share_in_genus"].fillna(0),                           # unusual source
        ]
        if self.use_value:
            parts.append(np.minimum(df["unit_value_z"].abs().fillna(0), 10) / 10)  # price vs peers
        return np.mean(np.column_stack(parts), axis=1)


class IsolationForestScorer:
    def __init__(self, features, max_rows=300_000):
        self.features, self.max_rows = features, max_rows

    def fit(self, df):
        idx = _subsample(len(df), self.max_rows)
        self.model = IsolationForest(n_estimators=200, max_samples=4096, n_jobs=8,
                                     random_state=RANDOM_STATE)
        self.model.fit(_deviation_matrix(df.iloc[idx], self.features))
        return self

    def score(self, df):
        return -self.model.score_samples(_deviation_matrix(df, self.features))


class KMeansDistanceScorer:
    def __init__(self, features, k_grid=(4, 6, 8, 10, 12), max_rows=300_000):
        self.features, self.k_grid, self.max_rows = features, k_grid, max_rows

    def fit(self, df):
        idx = _subsample(len(df), self.max_rows)
        M = _deviation_matrix(df.iloc[idx], self.features)
        self.scaler = StandardScaler().fit(M)
        Z = self.scaler.transform(M)
        sil_idx = _subsample(len(Z), 20_000)
        self.silhouette = {}
        best = None
        for k in self.k_grid:
            km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5,
                                 batch_size=10_000).fit(Z)
            s = silhouette_score(Z[sil_idx], km.predict(Z[sil_idx]))
            self.silhouette[k] = float(s)
            if best is None or s > best[0]:
                best = (s, k, km)
        _, self.k, self.model = best
        d = self.model.transform(Z)
        lab, dmin = d.argmin(1), d.min(1)
        self.spread = np.array([np.percentile(dmin[lab == c], 95) if (lab == c).any() else 1.0
                                for c in range(self.k)])
        # stability: agreement of assignments between two independent fits (ARI)
        from sklearn.metrics import adjusted_rand_score
        km2 = MiniBatchKMeans(n_clusters=self.k, random_state=RANDOM_STATE + 1, n_init=5,
                              batch_size=10_000).fit(Z)
        self.stability_ari = float(adjusted_rand_score(
            self.model.predict(Z[sil_idx]), km2.predict(Z[sil_idx])))
        return self

    def score(self, df):
        Z = self.scaler.transform(_deviation_matrix(df, self.features))
        d = self.model.transform(Z)
        lab = d.argmin(1)
        return d.min(1) / self.spread[lab]

    def assign(self, df):
        return self.model.predict(self.scaler.transform(_deviation_matrix(df, self.features)))


class PercentileRanker:
    """Maps a raw score onto its percentile within a reference population, so anomaly
    scores from detectors fitted on different datasets share one 0-1 scale."""

    def fit(self, ref_scores):
        self.grid = np.quantile(ref_scores, np.linspace(0, 1, 1001))
        return self

    def transform(self, s):
        return np.interp(s, self.grid, np.linspace(0, 1, 1001))


# --------------------------------------------------------------------------- hybrid
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class HybridStacker:
    """Meta-learner combining heterogeneous supervised + unsupervised evidence."""
    INPUTS = ["sup_logit", "peer_pct", "iforest_pct", "kmeans_pct"]

    def _matrix(self, p_sup, peer_pct, if_pct, km_pct):
        return np.column_stack([_logit(p_sup), peer_pct, if_pct, km_pct])

    def fit(self, p_sup, peer_pct, if_pct, km_pct, y):
        self.model = LogisticRegression(C=1.0, max_iter=500)
        self.model.fit(self._matrix(p_sup, peer_pct, if_pct, km_pct), y)
        self.coef_ = dict(zip(self.INPUTS, self.model.coef_[0].round(4).tolist()))
        return self

    def predict(self, p_sup, peer_pct, if_pct, km_pct):
        return self.model.predict_proba(self._matrix(p_sup, peer_pct, if_pct, km_pct))[:, 1]


# ------------------------------------------------------------------------- full stack
class Component:
    """One complete scoring stack (supervised model + 3 unsupervised detectors + stacker)."""

    def __init__(self, name, features, anomaly_features, use_value):
        self.name, self.features = name, features
        self.anomaly_features, self.use_value = anomaly_features, use_value

    def fit_unsupervised(self, train_df):
        self.peer = PeerDeviationScorer(self.use_value)
        self.iforest = IsolationForestScorer(self.anomaly_features).fit(train_df)
        self.kmeans = KMeansDistanceScorer(self.anomaly_features).fit(train_df)
        sample = train_df.sample(min(len(train_df), 400_000), random_state=RANDOM_STATE)
        self.rank_peer = PercentileRanker().fit(self.peer.score(sample))
        self.rank_if = PercentileRanker().fit(self.iforest.score(sample))
        self.rank_km = PercentileRanker().fit(self.kmeans.score(sample))
        return self

    def unsupervised_scores(self, df):
        return (self.rank_peer.transform(self.peer.score(df)),
                self.rank_if.transform(self.iforest.score(df)),
                self.rank_km.transform(self.kmeans.score(df)))

    def supervised_scores(self, df):
        X, _ = to_model_matrix(df, self.features, self.categories)
        return self.xgb.predict_proba(X)[:, 1]

    def score(self, df):
        p = self.supervised_scores(df)
        peer, iso, km = self.unsupervised_scores(df)
        return {"supervised": p, "peer": peer, "iforest": iso, "kmeans": km,
                "unsup_mean": (peer + iso + km) / 3,
                "hybrid": self.stacker.predict(p, peer, iso, km)}
