"""Supervised model selection (DESIGN_QUESTIONS.md Part B, B1-B5), used by notebook 10.

The protocol:
  B1  tuning by time-ordered cross-validation inside the training years (fold 1: train 2000-2006,
      validate 2007-2008; fold 2: train 2000-2008, validate 2009-2010); peer norms learned on each
      fold's training years; models fitted on a sample of the training shipments; selection on 2011
  B2  candidates: logistic regression, naive Bayes, decision tree, random forest, AdaBoost, XGBoost
  B3  each candidate's encoding and scaling; class-imbalance handling searched per candidate
  B4  random search with the same budget per candidate; PR-AUC on listed line items decides
  B5  each candidate's best configuration refitted on all of 2000-2010 and scored on 2011

Every candidate starts from one shared representation (`Base`): categories as integer codes
(unseen values get their own code), numeric features with gaps filled by the training median
plus a flag for each filled gap. Each candidate's encoding is built from it, and SMOTE-NC
resamples it before encoding, so all candidates see the same information.
"""
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from wildguard.features import CAT_SHARED, FeatureBuilder
from wildguard.models import XGB_PARAMS

FOLDS = [((2000, 2006), (2007, 2008)), ((2000, 2008), (2009, 2010))]
SAMPLE_FRAC = 0.2        # share of training shipments used to fit during the search (B1)
N_CONFIGS = 20           # random-search budget per candidate (B4)
RESAMPLE_RATIO = 0.1     # undersampling and SMOTE-NC target: one case per ten cleared items
SIMPLICITY = ["logistic_regression", "naive_bayes", "decision_tree", "random_forest",
              "adaboost", "xgboost"]                     # tie order (B5)


# ---------------------------------------------------------------- shared representation
class Base:
    """Integer-coded categories and median-filled numerics with missing flags."""

    def fit(self, F, feats):
        self.cats = [f for f in feats if f in CAT_SHARED]
        self.nums = [f for f in feats if f not in CAT_SHARED]
        self.levels = {c: pd.Index(sorted(F[c].dropna().astype(str).unique())) for c in self.cats}
        self.median = F[self.nums].median()
        self.flagged = [c for c in self.nums if F[c].isna().any()]
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


# ---------------------------------------------------------------- encodings (B3)
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


class Binned:
    """Naive Bayes: category codes as they are, numerics in ten quantile bins."""

    def __init__(self, base):
        self.k, self.card = base.n_cat, base.cardinality()

    def fit(self, X):
        self.edges = [np.unique(np.quantile(X[:, j], np.linspace(0, 1, 11)[1:-1]))
                      for j in range(self.k, X.shape[1])]
        return self

    def transform(self, X):
        num = [np.searchsorted(e, X[:, self.k + i], side="right") for i, e in enumerate(self.edges)]
        return np.column_stack([X[:, :self.k].astype(int)] + num)

    def min_categories(self):
        return np.array(self.card + [len(e) + 1 for e in self.edges])


class Native:
    """XGBoost: categories as pandas categoricals (native handling)."""

    def __init__(self, base):
        self.k, self.card = base.n_cat, base.cardinality()

    def fit(self, X):
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        for j in range(self.k):
            df[j] = pd.Categorical(X[:, j].astype(int), categories=range(self.card[j]))
        df.columns = [f"f{j}" for j in range(X.shape[1])]
        return df


class Identity:
    """Tree models in scikit-learn: integer codes and numerics as they are (no scaling)."""

    def __init__(self, base):
        pass

    def fit(self, X):
        return self

    def transform(self, X):
        return X


# ---------------------------------------------------------------- candidates and search spaces
def _loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_config(name, rng):
    """One random configuration of a candidate, including its class-imbalance handling (B3)."""
    if name == "logistic_regression":
        return {"imbalance": rng.choice(["none", "class_weight", "undersample", "smotenc"]),
                "optimizer": rng.choice(["lbfgs", "sgd"]),
                "C": _loguniform(rng, 1e-3, 10), "alpha": _loguniform(rng, 1e-7, 1e-3)}
    if name == "naive_bayes":
        return {"imbalance": rng.choice(["none", "class_weight", "undersample", "smotenc"]),
                "smoothing": _loguniform(rng, 1e-2, 10)}
    if name == "decision_tree":
        return {"imbalance": rng.choice(["none", "class_weight", "undersample", "smotenc"]),
                "max_depth": int(rng.integers(4, 21)), "min_samples_leaf": int(_loguniform(rng, 5, 2000))}
    if name == "random_forest":
        return {"imbalance": rng.choice(["none", "class_weight", "undersample", "smotenc", "balanced_rf"]),
                "max_depth": int(rng.choice([12, 16, 20, 25, 30])),
                "min_samples_leaf": int(_loguniform(rng, 1, 200)),
                "max_features": rng.choice(["sqrt", "0.3", "0.5"])}
    if name == "adaboost":
        return {"imbalance": rng.choice(["none", "class_weight", "undersample", "smotenc", "rusboost"]),
                "depth": int(rng.integers(1, 4)), "n_estimators": int(rng.choice([100, 200, 300])),
                "learning_rate": _loguniform(rng, 0.05, 1.0)}
    if name == "xgboost":
        return {"imbalance": rng.choice(["none", "class_weight", "undersample", "smotenc"]),
                "max_depth": int(rng.integers(4, 11)), "learning_rate": _loguniform(rng, 0.03, 0.2),
                "subsample": float(rng.uniform(0.6, 1.0)), "colsample_bytree": float(rng.uniform(0.5, 1.0)),
                "min_child_weight": _loguniform(rng, 1, 50)}
    raise ValueError(name)


ENCODING = {"logistic_regression": OneHotZ, "naive_bayes": Binned, "decision_tree": Identity,
            "random_forest": Identity, "adaboost": Identity, "xgboost": Native}


def _sqrt_weight(y):
    pos = y.sum()
    return float(np.sqrt((len(y) - pos) / pos))


def resample(X, y, cfg, base, seed):
    if cfg["imbalance"] == "undersample":
        return RandomUnderSampler(sampling_strategy=RESAMPLE_RATIO, random_state=seed).fit_resample(X, y)
    if cfg["imbalance"] == "smotenc":
        return SMOTENC(categorical_features=list(range(base.n_cat)), sampling_strategy=RESAMPLE_RATIO,
                       random_state=seed).fit_resample(X, y)
    return X, y


def build_model(name, cfg, enc, y, seed, n_estimators=None):
    cw = "balanced" if cfg["imbalance"] == "class_weight" else None
    if name == "logistic_regression":
        if cfg["optimizer"] == "lbfgs":
            return LogisticRegression(C=cfg["C"], class_weight=cw, max_iter=500)
        return SGDClassifier(loss="log_loss", alpha=cfg["alpha"], class_weight=cw, max_iter=30,
                             tol=1e-4, learning_rate="optimal", random_state=seed)
    if name == "naive_bayes":
        return CategoricalNB(alpha=cfg["smoothing"], fit_prior=cfg["imbalance"] != "class_weight",
                             min_categories=enc.min_categories())
    if name == "decision_tree":
        return DecisionTreeClassifier(max_depth=cfg["max_depth"], min_samples_leaf=cfg["min_samples_leaf"],
                                      class_weight=cw, random_state=seed)
    if name == "random_forest":
        mf = cfg["max_features"] if cfg["max_features"] == "sqrt" else float(cfg["max_features"])
        kw = dict(n_estimators=200, max_depth=cfg["max_depth"], min_samples_leaf=cfg["min_samples_leaf"],
                  max_features=mf, n_jobs=8, random_state=seed)
        if cfg["imbalance"] == "balanced_rf":
            return BalancedRandomForestClassifier(sampling_strategy="all", replacement=True,
                                                  bootstrap=False, **kw)
        return RandomForestClassifier(class_weight="balanced_subsample" if cw else None, **kw)
    if name == "adaboost":
        tree = DecisionTreeClassifier(max_depth=cfg["depth"])
        kw = dict(n_estimators=cfg["n_estimators"], learning_rate=cfg["learning_rate"],
                  algorithm="SAMME", random_state=seed)
        if cfg["imbalance"] == "rusboost":
            return RUSBoostClassifier(estimator=tree, sampling_strategy=RESAMPLE_RATIO, **kw)
        return AdaBoostClassifier(estimator=tree, **kw)
    if name == "xgboost":
        params = XGB_PARAMS | {k: cfg[k] for k in ("max_depth", "learning_rate", "subsample",
                                                    "colsample_bytree", "min_child_weight")}
        spw = _sqrt_weight(y) if cfg["imbalance"] == "class_weight" else 1.0
        if n_estimators:
            return xgb.XGBClassifier(**(params | {"random_state": seed}), n_estimators=n_estimators,
                                     scale_pos_weight=spw)
        return xgb.XGBClassifier(**(params | {"random_state": seed}), n_estimators=1500,
                                 early_stopping_rounds=60, scale_pos_weight=spw)
    raise ValueError(name)


def fit_predict(name, cfg, base, X_tr, y_tr, X_ev, seed, X_va=None, y_va=None, n_estimators=None):
    """Resample, encode, fit and score. Returns (scores on X_ev, best XGBoost round or None)."""
    Xr, yr = resample(X_tr, y_tr, cfg, base, seed)
    enc = ENCODING[name](base).fit(Xr)
    model = build_model(name, cfg, enc, yr, seed, n_estimators)
    fit_kw = {}
    if cfg["imbalance"] == "class_weight" and name == "adaboost":
        w = np.where(yr == 1, (len(yr) - yr.sum()) / yr.sum(), 1.0)
        fit_kw["sample_weight"] = w / w.mean()
    if name == "xgboost" and n_estimators is None:
        fit_kw.update(eval_set=[(enc.transform(X_va), y_va)], verbose=False)
    model.fit(enc.transform(Xr), yr, **fit_kw)
    best = getattr(model, "best_iteration", None) if name == "xgboost" else None
    return model.predict_proba(enc.transform(X_ev))[:, 1], best


# ---------------------------------------------------------------- the search (B1, B4)
def _years(df, span):
    return df["year"].between(*span).to_numpy()


def sample_shipments(df, mask, frac, seed):
    ships = df.loc[mask, "shipment_id"].drop_duplicates()
    keep = set(ships.sample(frac=frac, random_state=seed))
    return mask & df["shipment_id"].isin(keep).to_numpy()


def prepare_fold(df, feats, tr_span, va_span, seed=37):
    """Norms from all of the fold's training years; model rows from a sample of its shipments;
    validation on all listed line items of the fold's validation years."""
    tr_all = _years(df, tr_span)
    tr = sample_shipments(df, tr_all, SAMPLE_FRAC, seed)
    va = _years(df, va_span) & df["sp_listed"].to_numpy()
    fb = FeatureBuilder(use_value=False).fit(df[tr_all])
    F_tr, F_va = fb.transform(df[tr]), fb.transform(df[va])
    base = Base().fit(F_tr, feats)
    return dict(base=base, X_tr=base.transform(F_tr), y_tr=df["label"].to_numpy()[tr],
                X_va=base.transform(F_va), y_va=df["label"].to_numpy()[va])


def search(df, feats, candidates, seed=37, log=print, cache_dir=None):
    """Random search per candidate over the two folds. Returns one row per (candidate, config).

    With `cache_dir`, each candidate's results are saved as soon as it finishes
    (search_<candidate>.parquet) and read back instead of rerun, so an interrupted search
    resumes where it stopped."""
    cached = {c: pd.read_parquet(cache_dir / f"search_{c}.parquet") for c in candidates
              if cache_dir and (cache_dir / f"search_{c}.parquet").exists()}
    todo = [c for c in candidates if c not in cached]
    if cached:
        log(f"read from cache: {list(cached)}")
    if not todo:
        return pd.concat([cached[c] for c in candidates], ignore_index=True)
    folds = [prepare_fold(df, feats, tr, va, seed) for tr, va in FOLDS]
    log(f"folds ready: {[len(f['y_tr']) for f in folds]} training rows")
    for name in todo:
        rows = []
        rng = np.random.default_rng(seed)
        for i in range(N_CONFIGS):
            cfg = sample_config(name, rng)
            t, aps, bests = time.time(), [], []
            for f in folds:
                s, best = fit_predict(name, cfg, f["base"], f["X_tr"], f["y_tr"], f["X_va"], seed,
                                      X_va=f["X_va"], y_va=f["y_va"])
                aps.append(average_precision_score(f["y_va"], s))
                bests.append(best)
            rows.append({"candidate": name, "config": i, **{k: str(v) for k, v in cfg.items()},
                         "pr_auc_fold1": aps[0], "pr_auc_fold2": aps[1], "pr_auc": float(np.mean(aps)),
                         "best_rounds": int(np.mean(bests)) if bests[0] is not None else None,
                         "seconds": time.time() - t})
            log(f"{name} {i}: PR-AUC {np.mean(aps):.3f} ({cfg['imbalance']}, {time.time() - t:.0f}s)")
        cached[name] = pd.DataFrame(rows)
        if cache_dir:
            cached[name].to_parquet(cache_dir / f"search_{name}.parquet")
    return pd.concat([cached[c] for c in candidates], ignore_index=True)


def config_from_row(row):
    """Rebuild a configuration dict from a row of the search table."""
    keys = {"logistic_regression": ["imbalance", "optimizer", "C", "alpha"],
            "naive_bayes": ["imbalance", "smoothing"],
            "decision_tree": ["imbalance", "max_depth", "min_samples_leaf"],
            "random_forest": ["imbalance", "max_depth", "min_samples_leaf", "max_features"],
            "adaboost": ["imbalance", "depth", "n_estimators", "learning_rate"],
            "xgboost": ["imbalance", "max_depth", "learning_rate", "subsample", "colsample_bytree",
                        "min_child_weight"]}[row["candidate"]]
    cfg = {}
    for k in keys:
        v = row[k]
        try:
            v = float(v)
            v = int(v) if v.is_integer() and k not in ("C", "alpha", "smoothing", "learning_rate",
                                                       "subsample", "colsample_bytree",
                                                       "min_child_weight") else v
        except (TypeError, ValueError):
            pass
        cfg[k] = v
    return cfg


def refit_and_score(df, feats, name, cfg, best_rounds, seed=37):
    """B5: refit on all of 2000-2010 (norms from the same years), score all of 2011."""
    tr, ev = _years(df, (2000, 2010)), _years(df, (2011, 2011))
    fb = FeatureBuilder(use_value=False).fit(df[tr])
    F_tr, F_ev = fb.transform(df[tr]), fb.transform(df[ev])
    base = Base().fit(F_tr, feats)
    s, _ = fit_predict(name, cfg, base, base.transform(F_tr), df["label"].to_numpy()[tr],
                       base.transform(F_ev), seed, n_estimators=best_rounds or None)
    return s
