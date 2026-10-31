"""How different is CITES from LEMIS? The shift measurements of DESIGN_QUESTIONS.md C4.

Measured here, all with the deployed model's LEMIS norms (S2.1):
  1. recording shift: LEMIS line items of listed species 2013-2014 against the CITES records the
     US reported as importer for the same years (the same imports, recorded twice; ALIGNING 2.3)
  2. deployment shift: LEMIS training line items against the CITES records WildGuard ranks
     (rule D, 2015-2024, no source I or purpose L)
  For each: a domain classifier's ROC-AUC on raw fields (0.5 = indistinguishable), the
  population stability index (PSI) of each raw field, and the quantity deviation from LEMIS
  peer norms. A PCA projection of the numeric deviation features pictures the deployment shift
  (B9).

Measured in the design notebooks and reported from there: the trade-mix shift (held-out sets of
C1, in train.py), regional transfer and ageing (notebook 13). The jurisdiction shift cannot be
measured: no labels exist outside the US.

Domain classifiers see raw fields only. Group statistics (route share, peer-group size, ...)
take one exact value per group in each population, so a tree model can recognise the
population from those values alone ("fingerprinting", AUC near 1.0 for any two samples).

Run:  python -m wildguard.domain_shift      (after train.py and cites_features.py)
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import cites_features, config, plots
from .features import ANOMALY_FEATURES_SHARED, to_model_matrix
from .train import load_lemis

RAW_COMPARABLE = ["class", "product_group", "unit_group", "purpose", "source", "appendix",
                  "origin_region", "exporter_region", "genus_top", "exporter_top", "origin_top",
                  "log_qty", "reexport_flag", "is_wild"]
PSI_FIELDS = ["class", "product_group", "unit_group", "purpose", "source", "appendix",
              "exporter_region", "origin_region"]
SAMPLE = 400_000


def domain_classifier(A, B, max_rows=250_000):
    """ROC-AUC, on a held-out 30%, of a classifier telling sample A from sample B on raw fields."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    ia = rng.choice(len(A), min(len(A), max_rows), replace=False)
    ib = rng.choice(len(B), min(len(B), max_rows), replace=False)
    XA, cats = to_model_matrix(A.iloc[ia], RAW_COMPARABLE)
    XB = to_model_matrix(B.iloc[ib], RAW_COMPARABLE, cats)[0]
    X = pd.concat([XA, XB], ignore_index=True)
    y = np.r_[np.zeros(len(ia)), np.ones(len(ib))]
    perm = rng.permutation(len(X))
    tr, te = perm[:int(0.7 * len(X))], perm[int(0.7 * len(X)):]
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist",
                            enable_categorical=True, subsample=0.8, colsample_bytree=0.8, n_jobs=8,
                            random_state=config.RANDOM_STATE).fit(X.iloc[tr], y[tr])
    return round(float(roc_auc_score(y[te], clf.predict_proba(X.iloc[te])[:, 1])), 3)


def psi(a, b, eps=1e-4):
    """Population stability index between two categorical samples."""
    pa, pb = a.astype(str).value_counts(normalize=True), b.astype(str).value_counts(normalize=True)
    idx = pa.index.union(pb.index)
    pa, pb = pa.reindex(idx).fillna(0) + eps, pb.reindex(idx).fillna(0) + eps
    return round(float(((pa - pb) * np.log(pa / pb)).sum()), 3)


def compare(A, B):
    return {"domain_classifier_auc": domain_classifier(A, B),
            "psi": {c: psi(A[c], B[c]) for c in PSI_FIELDS},
            "qty_z": {k: {"median": round(float(d["qty_z"].median()), 3),
                          "share_above_3": round(float((d["qty_z"] > 3).mean()), 4)}
                      for k, d in (("a", A), ("b", B))}}


def pca_figure(L, C):
    """Two principal components of the standardised deviation features, LEMIS against CITES."""
    import matplotlib.pyplot as plt
    plots.apply_style()
    feats = ANOMALY_FEATURES_SHARED
    ZL = L[feats].astype(float).fillna(0).to_numpy()
    ZC = C[feats].astype(float).fillna(0).to_numpy()
    sc = StandardScaler().fit(ZL)
    pca = PCA(2, random_state=config.RANDOM_STATE).fit(np.r_[sc.transform(ZL), sc.transform(ZC)])
    PL, PC = pca.transform(sc.transform(ZL)), pca.transform(sc.transform(ZC))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    lim = np.percentile(np.abs(np.r_[PL, PC]), 99.5)
    for ax, P, title in ((axes[0], PL, "LEMIS training line items"),
                         (axes[1], PC, "CITES records WildGuard ranks")):
        ax.hexbin(P[:, 0], P[:, 1], gridsize=45, bins="log", cmap="Greens", extent=(-lim, lim, -lim, lim))
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance)")
    plots.save(fig, config.FIGURES_DIR / "domain_shift_pca.png")
    return {"features": feats, "explained_variance": [round(float(v), 3) for v in pca.explained_variance_ratio_],
            "loadings": {f"PC{i + 1}": dict(zip(feats, np.round(pca.components_[i], 3).tolist()))
                         for i in range(2)}}


def main():
    t0 = time.time()
    fb = joblib.load(config.MODELS_DIR / "feature_builder_lemis.joblib")
    df = load_lemis()
    lem_test = fb.transform(df[df["year"].isin(config.TEST_YEARS) & df["sp_listed"]])
    lem_train = fb.transform(df[df["year"].isin(config.TRAIN_YEARS)].sample(
        SAMPLE, random_state=config.RANDOM_STATE))
    us = cites_features.load_cites(config.TEST_YEARS,
                                   lambda d: (d["importer"] == "US") & (d["reporter_type"] == "I"))
    us = fb.transform(us)
    cols = list(dict.fromkeys(RAW_COMPARABLE + PSI_FIELDS + ANOMALY_FEATURES_SHARED
                              + ["qty_z", "rule_d", "source", "purpose"]))
    C = cites_features.load_features(columns=cols)
    C = C[C["rule_d"] & (C["source"] != "I") & (C["purpose"] != "L")]
    C = C.sample(min(len(C), SAMPLE), random_state=config.RANDOM_STATE)
    print(f"samples ready ({time.time() - t0:.0f}s)")
    out = {"recording_shift_lemis_listed_vs_cites_us_2013_14": compare(lem_test, us)
           | {"n": [int(len(lem_test)), int(len(us))]},
           "deployment_shift_lemis_train_vs_cites_ranked": compare(lem_train, C),
           "pca_deviation_features": pca_figure(lem_train, C),
           "measured_elsewhere": {"trade_mix": "outputs/lemis_evaluation.json (held-out sets of C1)",
                                  "regional_transfer_and_ageing": "notebooks/13_transfer_and_age.ipynb",
                                  "jurisdiction": "not measurable: no labels outside the US"}}
    (config.OUTPUTS_DIR / "domain_shift.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
