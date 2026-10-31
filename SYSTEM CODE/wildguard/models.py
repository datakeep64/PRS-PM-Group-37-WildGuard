"""Steps 3-5 - the pattern-recognition components, as chosen in SYSTEM CODE/DESIGN_QUESTIONS.md
Part B (notebooks 10-14).

Supervised (trained on LEMIS enforcement outcomes)
    Ensemble        XGBoost (boosting) and a random forest (bagging) on one shared
                    representation; their scores are combined by the average of percentile
                    ranks (B5-B6). Chosen from six candidates; no class-imbalance handling,
                    which lowered PR-AUC for every strong candidate (notebook 10, T3)
    LogisticModel   logistic regression on one-hot categories and z-scored numerics: the
                    linear comparison model of the reported results

Unsupervised (no labels used)
    PeerDeviation   how far a record sits from its peer group's norms, and how rare its route,
                    purpose and source are for its genus. Shown as context on the flag card; no
                    unsupervised detector improved the ensemble, so none is part of the score (B7)
    IsolationForestScorer, KMeansDistanceScorer
                    tested in notebook 11 and rejected; kept so the notebook can be rerun

The shared representation (`Base`) codes categories as integers (an unseen value gets its own
code) and fills missing numbers with the training median, adding a flag for each filled gap.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_STATE
from .features import CAT_LEMIS_ONLY, CAT_SHARED

CATEGORICAL = set(CAT_SHARED + CAT_LEMIS_ONLY)

XGB_PARAMS = dict(
    objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
    enable_categorical=True, max_cat_to_onehot=8, max_depth=8, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
    n_jobs=8, random_state=RANDOM_STATE,
)
# the chosen members (notebook 10, configuration 5; notebook 11, H7)
CHOSEN_XGB = XGB_PARAMS | dict(max_depth=6, learning_rate=0.05, min_child_weight=20.0)
XGB_ROUNDS = 357                      # mean best round of the two cross-validation folds
CHOSEN_RF = dict(n_estimators=100, max_depth=16, min_samples_leaf=5, max_features="sqrt",
                 n_jobs=8, random_state=RANDOM_STATE)


# ---------------------------------------------------------------- representation
class Base:
    """Integer-coded categories and median-filled numerics with missing flags."""

    def fit(self, F, feats):
        self.cats = [f for f in feats if f in CATEGORICAL]
        self.nums = [f for f in feats if f not in CATEGORICAL]
        self.levels = {c: pd.Index(sorted(F[c].dropna().astype(str).unique())) for c in self.cats}
        self.median = F[self.nums].median()
        self.flagged = [c for c in self.nums if F[c].isna().any()]
        self.names = self.cats + self.nums + [f"{c}_missing" for c in self.flagged]
        return self

    def transform(self, F):
        cols = []
        for c in self.cats:
            codes = self.levels[c].get_indexer(F[c].astype(str))
            cols.append(np.where(codes < 0, len(self.levels[c]), codes))       # unseen: own code
        num = F[self.nums].astype("float64")
        cols += [num[c].fillna(self.median[c]).to_numpy() for c in self.nums]
        cols += [num[c].isna().to_numpy().astype(float) for c in self.flagged]
        return np.column_stack(cols).astype("float32")

    @property
    def n_cat(self):
        return len(self.cats)

    def cardinality(self):
        return [len(self.levels[c]) + 1 for c in self.cats]


class Native:
    """XGBoost: categories as pandas categoricals (native handling), columns named after the
    features, so its per-feature contributions map onto them."""

    def __init__(self, base):
        self.k, self.card, self.names = base.n_cat, base.cardinality(), base.names

    def fit(self, X):
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=self.names)
        for j, c in enumerate(self.names[:self.k]):
            df[c] = pd.Categorical(X[:, j].astype(int), categories=range(self.card[j]))
        return df


class OneHotZ:
    """Logistic regression: one-hot categories, z-score standardised numerics."""

    def __init__(self, base):
        self.k = base.n_cat

    def fit(self, X):
        self.oh = OneHotEncoder(handle_unknown="ignore").fit(X[:, :self.k])
        self.z = StandardScaler().fit(X[:, self.k:])
        return self

    def transform(self, X):
        return sp.hstack([self.oh.transform(X[:, :self.k]),
                          sp.csr_matrix(self.z.transform(X[:, self.k:]))]).tocsr()


def percentile_rank(s, groups=None):
    """Percentile rank of each score, within each group when `groups` is given."""
    s = pd.Series(np.asarray(s, dtype=float))
    if groups is None:
        return s.rank(pct=True).to_numpy()
    return s.groupby(np.asarray(groups)).rank(pct=True).to_numpy()


class PercentileRanker:
    """Maps a raw score onto its percentile within a reference population, for scoring a single
    record (the app's "check a shipment")."""

    def fit(self, ref_scores):
        self.grid = np.quantile(ref_scores, np.linspace(0, 1, 1001))
        return self

    def transform(self, s):
        return np.interp(s, self.grid, np.linspace(0, 1, 1001))


# ---------------------------------------------------------------- supervised
class Ensemble:
    """The chosen supervised model (B5-B6): XGBoost and a random forest, rank-averaged."""

    def __init__(self, features):
        self.features = list(features)

    def fit(self, F, y):
        self.base = Base().fit(F, self.features)
        X = self.base.transform(F)
        self.enc = Native(self.base)
        self.xgb = xgb.XGBClassifier(**CHOSEN_XGB, n_estimators=XGB_ROUNDS)
        self.xgb.fit(self.enc.transform(X), y)
        self.rf = RandomForestClassifier(**CHOSEN_RF).fit(X, y)
        return self

    def matrix(self, F):
        """XGBoost's input for F (named columns), as used for its contributions."""
        return self.enc.transform(self.base.transform(F))

    def member_scores(self, F):
        X = self.base.transform(F)
        return {"xgboost": self.xgb.predict_proba(self.enc.transform(X))[:, 1],
                "random_forest": self.rf.predict_proba(X)[:, 1]}

    def set_reference(self, F_ref):
        """Percentile scales for scoring records one at a time (no population to rank within)."""
        m = self.member_scores(F_ref)
        self.rankers = {k: PercentileRanker().fit(v) for k, v in m.items()}
        return self

    def score(self, F, groups=None, members=None):
        """The ensemble score: average of the members' percentile ranks within the scored
        population, or within each group. For a few records on their own, use score_single."""
        m = members if members is not None else self.member_scores(F)
        return (percentile_rank(m["xgboost"], groups) + percentile_rank(m["random_forest"], groups)) / 2

    def score_single(self, F):
        """The ensemble score of records against the reference population of set_reference."""
        m = self.member_scores(F)
        return (self.rankers["xgboost"].transform(m["xgboost"])
                + self.rankers["random_forest"].transform(m["random_forest"])) / 2


class LogisticModel:
    """Linear comparison model: logistic regression (L-BFGS, C = 1), no imbalance handling, as
    in notebook 10's best configuration."""

    def __init__(self, features):
        self.features = list(features)

    def fit(self, F, y):
        self.base = Base().fit(F, self.features)
        X = self.base.transform(F)
        self.enc = OneHotZ(self.base).fit(X)
        self.model = LogisticRegression(C=1.0, max_iter=500).fit(self.enc.transform(X), y)
        return self

    def predict(self, F):
        return self.model.predict_proba(self.enc.transform(self.base.transform(F)))[:, 1]


# ---------------------------------------------------------------- unsupervised
def _subsample(n, k):
    rng = np.random.default_rng(RANDOM_STATE)
    return np.sort(rng.choice(n, size=min(n, k), replace=False))


def _deviation_matrix(df, features):
    M = df[features].astype("float64").copy()
    for c in ("qty_z", "unit_value_z"):
        if c in M:
            M[c] = M[c].abs().clip(upper=15)
    return M.fillna(0).to_numpy()


class PeerDeviationScorer:
    """Transparent statistical anomaly score in [0, 1]. Needs no fitting: the peer-group
    norms it compares against are learned by FeatureBuilder from the reference population."""

    def __init__(self, use_value=False):
        self.use_value = use_value

    def parts(self, df):
        """The score's components, each in [0, 1], for the flag card."""
        out = {
            "quantity": np.minimum(df["qty_z"].abs().fillna(0), 10) / 10,                   # vs peers
            "route": np.clip(-np.log10(df["route_share"].fillna(0) + 1e-4) / 4, 0, 1),      # rare route
            "combination": np.clip(df["combo_rarity"].fillna(7) / 7, 0, 1),                 # rare form/purpose/source
            "purpose": 1 - df["purpose_share_in_genus"].fillna(0),                          # unusual purpose
            "source": 1 - df["source_share_in_genus"].fillna(0),                            # unusual source
        }
        if self.use_value:
            out["value"] = np.minimum(df["unit_value_z"].abs().fillna(0), 10) / 10           # price vs peers
        return {k: np.asarray(v, dtype=float) for k, v in out.items()}

    def score(self, df):
        return np.mean(np.column_stack(list(self.parts(df).values())), axis=1)


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
