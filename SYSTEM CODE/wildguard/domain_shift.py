"""Transfer robustness: how much does a LEMIS-trained model lose when the data shifts, and
which adaptation steps recover it?

CITES has no enforcement labels, so transfer *to CITES* cannot be scored directly. We
therefore run three complementary experiments:

A. Simulated domain shift inside LEMIS (labels on both sides)
   Source = shipments exported from every region except Asia; target = shipments exported
   from Asia (unseen during training). Compared on the held-out target years:
     in-domain reference     source model on source test data
     naive transfer          source model, target described with *source* peer norms
     re-normalised transfer  source model, target peer norms re-learned from unlabelled
                             target data (exactly what we do for CITES)
     importance-weighted     source rows re-weighted by p(target|x)/p(source|x) from a
                             domain classifier (covariate-shift correction)
     few-shot                source data + 1% / 5% of labelled target rows
     target oracle           trained on all labelled target rows (upper bound)

B. How different is CITES from LEMIS? (domain-classifier AUC; 0.5 = indistinguishable)
     LEMIS 2013-14  vs  CITES records the US itself reported for 2013-14 imports
                        (sanity check of the schema alignment - the US compiles its CITES
                        annual report from its own declarations, so these should look alike)
     LEMIS          vs  global CITES 2015-2024 (the deployment shift)
   plus the population stability index (PSI) of each categorical feature.

C. Deployment variant: transfer model importance-weighted towards CITES 2015-2024. It is
   deployed only if experiment A shows importance weighting helps on a labelled target AND it
   is not worse than the unweighted model on held-out LEMIS records of CITES-listed taxa.

Run:  python -m wildguard.domain_shift      (after train.py and cites_features.py)
      python -m wildguard.domain_shift --only-c   (re-use A and B from a previous run)
"""
import copy
import json
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from . import cites_features, config
from .evaluate import evaluate_scores
from .features import SHARED_FEATURES, FeatureBuilder, to_model_matrix
from .models import HybridStacker, train_xgb
from .train import load_lemis

TARGET_REGION = "Asia"
DOMAIN_DEFINING = ["exporter_region", "origin_region", "exporter_top", "origin_top"]
# Domain classifiers only see raw, directly comparable fields. Group-level statistics
# (route_share, peer_n_log, combo_rarity, ...) take one exact value per group in each
# population, so a tree model can "fingerprint" the population from those values and reach
# AUC ~ 1.0 without any real difference in trade - which also makes importance weights
# degenerate. Raw fields have the same meaning and value range in both populations.
RAW_COMPARABLE = ["class", "product_group", "unit_group", "purpose", "source", "appendix",
                  "origin_region", "exporter_region", "genus_top", "exporter_top", "origin_top",
                  "log_qty", "reexport_flag", "is_wild"]
KEEP =["pr_auc", "roc_auc", "recall@5%", "precision@1%", "lift@1%", "base_rate"]


def _metrics(y, s, thr=0.5):
    m = evaluate_scores(y, s, thr)
    return {k: round(m[k], 4) for k in KEEP}


def domain_classifier(Xa, Xb, max_rows=250_000):
    """Fit a classifier to tell sample a (label 0) from sample b (label 1); return the model
    and its ROC-AUC on a held-out 30%. AUC near 0.5 means the domains look alike."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    ia = rng.choice(len(Xa), min(len(Xa), max_rows), replace=False)
    ib = rng.choice(len(Xb), min(len(Xb), max_rows), replace=False)
    X = pd.concat([Xa.iloc[ia], Xb.iloc[ib]], ignore_index=True)
    y = np.r_[np.zeros(len(ia)), np.ones(len(ib))]
    perm = rng.permutation(len(X))
    cut = int(0.7 * len(X))
    tr, te = perm[:cut], perm[cut:]
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                            tree_method="hist", enable_categorical=True, subsample=0.8,
                            colsample_bytree=0.8, n_jobs=8, random_state=config.RANDOM_STATE)
    clf.fit(X.iloc[tr], y[tr])
    auc = roc_auc_score(y[te], clf.predict_proba(X.iloc[te])[:, 1])
    return clf, float(auc), len(ia) / len(ib)


def importance_weights(clf, X, sample_ratio, clip=(0.05, 20.0)):
    p = np.clip(clf.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
    w = np.clip(p / (1 - p) * sample_ratio, *clip)
    w = w / w.mean()
    ess = float(w.sum() ** 2 / (w ** 2).sum() / len(w))
    return w, ess


def psi(a, b, eps=1e-4):
    """Population stability index between two categorical samples."""
    pa = a.value_counts(normalize=True)
    pb = b.value_counts(normalize=True)
    idx = pa.index.union(pb.index)
    pa, pb = pa.reindex(idx).fillna(0) + eps, pb.reindex(idx).fillna(0) + eps
    return float(((pa - pb) * np.log(pa / pb)).sum())


# --------------------------------------------------------------------------- experiment A
def experiment_a(df):
    t0 = time.time()
    yr = df["year"]
    tr, va, te = (yr.isin(config.TRAIN_YEARS), yr.isin(config.VALID_YEARS),
                  yr.isin(config.TEST_YEARS))
    tgt = df["exporter_region"] == TARGET_REGION
    src_tr, src_va, src_te = df[tr & ~tgt], df[va & ~tgt], df[te & ~tgt]
    tgt_tr, tgt_va, tgt_te = df[tr & tgt], df[va & tgt], df[te & tgt]

    fb_src = FeatureBuilder(use_value=False).fit(src_tr)
    # target norms learned from unlabelled target rows only (labels never touched)
    fb_tgt = FeatureBuilder(use_value=False).fit(pd.concat([tgt_tr, tgt_va, tgt_te]),
                                                 vocab_from=fb_src)

    S_tr, S_va, S_te = (fb_src.transform(d) for d in (src_tr, src_va, src_te))
    Tte_naive = fb_src.transform(tgt_te)
    Ttr, Tva, Tte = (fb_tgt.transform(d) for d in (tgt_tr, tgt_va, tgt_te))

    Xs_tr, cats = to_model_matrix(S_tr, SHARED_FEATURES)
    M = lambda d: to_model_matrix(d, SHARED_FEATURES, cats)[0]   # noqa: E731
    Xs_va, Xs_te, Xt_naive, Xt_tr, Xt_va, Xt_te = map(M, (S_va, S_te, Tte_naive, Ttr, Tva, Tte))
    ys_tr, ys_va, ys_te = S_tr["label"].values, S_va["label"].values, S_te["label"].values
    yt_tr, yt_va, yt_te = Ttr["label"].values, Tva["label"].values, Tte["label"].values

    res = {"setup": {"target_region": TARGET_REGION, "n_source_train": int(len(S_tr)),
                     "n_target_test": int(len(Tte)), "target_test_pos_rate": float(yt_te.mean()),
                     "source_test_pos_rate": float(ys_te.mean())}}

    base = train_xgb(Xs_tr, ys_tr, Xs_va, ys_va)
    res["in_domain_reference"] = _metrics(ys_te, base.predict_proba(Xs_te)[:, 1])
    res["naive_transfer"] = _metrics(yt_te, base.predict_proba(Xt_naive)[:, 1])
    res["renormalised_transfer"] = _metrics(yt_te, base.predict_proba(Xt_te)[:, 1])
    print(f"  A: base done ({time.time() - t0:.0f}s)")

    # The domains are *defined* by exporter region, so region/country features separate them
    # perfectly (AUC 1.0) and no re-weighting can bridge that. Covariate-shift correction is
    # only meaningful on features whose support overlaps, so they are left out here.
    dom = [f for f in RAW_COMPARABLE if f not in DOMAIN_DEFINING]
    clf, auc, ratio = domain_classifier(Xs_tr[dom], pd.concat([Xt_tr, Xt_va, Xt_te])[dom])
    w, ess = importance_weights(clf, Xs_tr[dom], ratio)
    iw = train_xgb(Xs_tr, ys_tr, Xs_va, ys_va, sample_weight=w)
    res["importance_weighted"] = _metrics(yt_te, iw.predict_proba(Xt_te)[:, 1])
    res["importance_weighted"].update({"domain_auc": round(auc, 4), "ess": round(ess, 4)})
    print(f"  A: importance weighting done, domain AUC {auc:.3f} ({time.time() - t0:.0f}s)")

    rng = np.random.default_rng(config.RANDOM_STATE)
    for frac in (0.01, 0.05):
        k = rng.choice(len(Xt_tr), int(frac * len(Xt_tr)), replace=False)
        X = pd.concat([Xs_tr, Xt_tr.iloc[k]], ignore_index=True)
        y = np.r_[ys_tr, yt_tr[k]]
        sw = np.r_[np.ones(len(Xs_tr)), np.full(len(k), 5.0)]
        m = train_xgb(X, y, Xs_va, ys_va, sample_weight=sw)
        res[f"few_shot_{int(frac * 100)}pct_target_labels"] = _metrics(
            yt_te, m.predict_proba(Xt_te)[:, 1])
        res[f"few_shot_{int(frac * 100)}pct_target_labels"]["n_target_labels"] = int(len(k))

    oracle = train_xgb(Xt_tr, yt_tr, Xt_va, yt_va)
    res["target_oracle"] = _metrics(yt_te, oracle.predict_proba(Xt_te)[:, 1])
    res["target_oracle"]["n_target_labels"] = int(len(Xt_tr))
    print(f"  A: done ({time.time() - t0:.0f}s)")
    return res


# ------------------------------------------------------------------------ experiments B, C
def experiment_b(df, cites_sample):
    t0 = time.time()
    lem = df[df["year"].isin([2013, 2014])]
    us = cites_features.load_cites(range(2013, 2015),
                                   lambda d: (d["importer"] == "US") & (d["reporter_type"] == "I"))
    fb_l = FeatureBuilder(use_value=False).fit(lem)
    fb_u = FeatureBuilder(use_value=False).fit(us)
    L, U = fb_l.transform(lem), fb_u.transform(us)
    L_listed = L[L["appendix"].isin(["I", "II", "III"])]
    XL, cats = to_model_matrix(L, SHARED_FEATURES)
    M = lambda d: to_model_matrix(d, SHARED_FEATURES, cats)[0]   # noqa: E731
    out = {"n_lemis_2013_14": int(len(L)), "n_cites_us_reported_2013_14": int(len(U))}
    R = RAW_COMPARABLE
    out["auc_lemis_vs_cites_us"] = domain_classifier(XL[R], M(U)[R])[1]
    out["auc_lemis_listed_vs_cites_us"] = domain_classifier(M(L_listed)[R], M(U)[R])[1]
    out["auc_lemis_vs_cites_global_2015_24"] = domain_classifier(XL[R], M(cites_sample)[R])[1]
    # How far do CITES quantities sit from LEMIS peer norms? CITES records aggregate trade
    # per permit and taxon, so a systematic offset here is what re-normalisation absorbs.
    fb_train = joblib.load(config.MODELS_DIR / "feature_builder_lemis.joblib")
    raw = cites_features.load_cites(config.CITES_SCORE_YEARS)
    raw = raw.sample(min(len(raw), 400_000), random_state=config.RANDOM_STATE)
    q = {"lemis_test_2013_14": fb_train.transform(lem)["qty_z"],
         "cites_us_reported_2013_14": fb_train.transform(us)["qty_z"],
         "cites_global_2015_24": fb_train.transform(raw)["qty_z"]}
    out["qty_z_under_lemis_norms"] = {k: {"median": round(float(v.median()), 3),
                                          "share_above_3": round(float((v > 3).mean()), 4)}
                                      for k, v in q.items()}
    out["psi_lemis_vs_cites_global"] = {c: round(psi(L[c], cites_sample[c]), 3)
                                        for c in ["class", "product_group", "purpose", "source",
                                                  "appendix", "exporter_region", "unit_group"]}
    out["psi_lemis_listed_vs_cites_us"] = {c: round(psi(L_listed[c], U[c]), 3)
                                           for c in ["class", "product_group", "purpose",
                                                     "source", "appendix", "exporter_region"]}
    out["purpose_share"] = {
        "lemis": L["purpose"].value_counts(normalize=True).head(6).round(4).to_dict(),
        "cites_global": cites_sample["purpose"].value_counts(normalize=True).head(6).round(4).to_dict()}
    print(f"  B: done ({time.time() - t0:.0f}s)")
    return out


def experiment_c(df, cites_sample, passes_a):
    """Importance-weighted transfer stack (towards CITES 2015-2024), compared with the
    unweighted stack on held-out LEMIS records - all of them, and the CITES-like subset
    (CITES-listed taxa), which is the population the deployed model will actually face."""
    t0 = time.time()
    stack = joblib.load(config.MODELS_DIR / "transfer_stack.joblib")
    fb = joblib.load(config.MODELS_DIR / "feature_builder_lemis.joblib")
    tr = df["year"].isin(config.TRAIN_YEARS)
    va = df["year"].isin(config.VALID_YEARS)
    te = df["year"].isin(config.TEST_YEARS)
    Ftr, Fva, Fte = (fb.transform(df[m]).reset_index(drop=True) for m in (tr, va, te))
    Xtr, _ = to_model_matrix(Ftr, SHARED_FEATURES, stack.categories)
    Xva, _ = to_model_matrix(Fva, SHARED_FEATURES, stack.categories)
    Xc, _ = to_model_matrix(cites_sample, SHARED_FEATURES, stack.categories)
    clf, auc, ratio = domain_classifier(Xtr[RAW_COMPARABLE], Xc[RAW_COMPARABLE])
    w, ess = importance_weights(clf, Xtr[RAW_COMPARABLE], ratio)
    out = {"domain_auc_lemis_train_vs_cites": round(auc, 4), "ess": round(ess, 4)}

    iw_stack = copy.copy(stack)
    iw_stack.name = "transfer_iw"
    iw_stack.xgb = train_xgb(Xtr, Ftr["label"].values, Xva, Fva["label"].values, sample_weight=w)
    p_va = iw_stack.xgb.predict_proba(Xva)[:, 1]
    iw_stack.stacker = HybridStacker().fit(p_va, *iw_stack.unsupervised_scores(Fva),
                                           Fva["label"].values)
    y = Fte["label"].values
    listed = Fte["appendix"].isin(["I", "II", "III"]).to_numpy()
    scores = {"unweighted": stack.score(Fte)["hybrid"],
              "importance_weighted": iw_stack.score(Fte)["hybrid"]}
    for subset, m in (("lemis_test_all", np.ones(len(y), bool)),
                      ("lemis_test_cites_listed", listed)):
        out[subset] = {k: _metrics(y[m], s[m]) for k, s in scores.items()}
        out[subset]["n"] = int(m.sum())
    better_on_cites_like = (out["lemis_test_cites_listed"]["importance_weighted"]["pr_auc"]
                            >= out["lemis_test_cites_listed"]["unweighted"]["pr_auc"])
    deploy = passes_a and better_on_cites_like
    out["passes_experiment_a"] = bool(passes_a)
    out["not_worse_on_cites_listed_lemis"] = bool(better_on_cites_like)
    iw_path = config.MODELS_DIR / "transfer_iw_stack.joblib"
    if deploy:
        joblib.dump(iw_stack, iw_path, compress=3)
    elif iw_path.exists():
        iw_path.unlink()
    print(f"  C: done ({time.time() - t0:.0f}s)")
    return out, deploy


def _checkpoint(results):
    (config.OUTPUTS_DIR / "domain_shift.json").write_text(json.dumps(results, indent=2))


def main(only_c=False):
    cites_features.build()
    df = load_lemis()
    path = config.OUTPUTS_DIR / "domain_shift.json"
    if only_c and path.exists():          # re-use experiments A and B from a previous run
        results = {k: v for k, v in json.loads(path.read_text()).items() if k[0] in "AB"}
    else:
        results = {"A_simulated_shift": experiment_a(df)}
        _checkpoint(results)
    A = results["A_simulated_shift"]
    best_label_free = max(A["naive_transfer"]["pr_auc"], A["renormalised_transfer"]["pr_auc"])
    passes_a = A["importance_weighted"]["pr_auc"] > best_label_free + 0.005

    cs = cites_features.load_features(columns=SHARED_FEATURES + ["year"])
    cites_sample = cs.sample(min(len(cs), 400_000), random_state=config.RANDOM_STATE)
    del cs
    if "B_lemis_vs_cites" not in results:
        results["B_lemis_vs_cites"] = experiment_b(df, cites_sample)
        _checkpoint(results)
    results["C_deployment_importance_weighting"], deploy_iw = experiment_c(df, cites_sample,
                                                                           passes_a)
    deployed = "transfer_iw" if deploy_iw else "transfer"
    results["deployed_stack"] = deployed
    (config.MODELS_DIR / "deployment.json").write_text(json.dumps(
        {"stack": deployed,
         "chosen_by": "importance weighting is deployed only if (1) in experiment A it beats the "
                      "best label-free transfer (naive or re-normalised) by > 0.005 PR-AUC on the "
                      "labelled held-out target region, and (2) it is not worse than the "
                      "unweighted model on held-out LEMIS records of CITES-listed taxa"},
        indent=2))
    _checkpoint(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    import sys
    main(only_c="--only-c" in sys.argv)
