"""Steps 2-6 on LEMIS - train every component and run the primary (labelled) evaluation.

Temporal split: fit on 2000-2010, tune thresholds and the hybrid combiner on 2011-2012,
report on 2013-2014 (never touched before the final evaluation).

Compared on the same held-out years:
  baselines     random inspection; "inspect the most endangered first" listing rule
  supervised    XGBoost (full LEMIS features, and transfer = CITES-compatible features only),
                Random Forest, Logistic Regression
  unsupervised  peer-deviation score, Isolation Forest, K-Means distance (labels unused)
  hybrid        stacked supervised + unsupervised (full, and transfer = deployed to CITES)

Run:  python -m wildguard.train
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from . import config
from .evaluate import (band_report, best_f1_threshold, capture_curve, confidence_bands,
                       evaluate_scores, top_k_mask)
from .features import (ANOMALY_FEATURES_FULL, ANOMALY_FEATURES_SHARED, FULL_FEATURES,
                       SHARED_FEATURES, FeatureBuilder, to_model_matrix)
from .models import (Component, HybridStacker, predict_logistic, predict_random_forest,
                     train_logistic, train_random_forest, train_xgb)
from . import plots

LEMIS_COLS = ["shipment_id", "year", "month", "class", "genus", "taxon", "taxon_display",
              "common_name", "product_code", "product_group", "unit_group", "quantity_std",
              "value_usd", "purpose", "source", "origin", "exporter", "port", "label",
              "disposition", "appendix", "origin_region", "exporter_region"]


def load_lemis():
    df = pd.read_parquet(config.LEMIS_PARQUET, columns=LEMIS_COLS)
    df["month"] = df["month"].astype("float32")
    frac = float(os.environ.get("WILDGUARD_SAMPLE", "1"))     # e.g. 0.03 for a smoke test
    if frac < 1:
        df = df.sample(frac=frac, random_state=config.RANDOM_STATE).reset_index(drop=True)
    return df


def listing_rule_score(df, seed=0):
    """Naive manual heuristic: inspect the most endangered (Appendix I) and wild-caught first."""
    rank = df["appendix"].map({"I": 3, "II": 2, "III": 2}).fillna(1)
    jitter = np.random.default_rng(seed).random(len(df)) * 1e-3
    return (rank + 0.5 * (df["source"] == "W") + jitter).to_numpy()


def main():
    t0 = time.time()
    df = load_lemis()
    tr = df["year"].isin(config.TRAIN_YEARS).to_numpy()
    va = df["year"].isin(config.VALID_YEARS).to_numpy()
    te = df["year"].isin(config.TEST_YEARS).to_numpy()
    print(f"rows train {tr.sum():,}  valid {va.sum():,}  test {te.sum():,}")

    fb = FeatureBuilder(use_value=True).fit(df[tr])
    F = fb.transform(df)
    print(f"features built ({time.time() - t0:.0f}s)")
    y = F["label"].to_numpy()
    Ftr, Fva, Fte = F[tr].reset_index(drop=True), F[va].reset_index(drop=True), F[te].reset_index(drop=True)
    ytr, yva, yte = y[tr], y[va], y[te]

    results, test_scores, val_scores = {}, {}, {}

    # ---------------------------------------------------------------- full + transfer stacks
    stacks = {
        "full": Component("full", FULL_FEATURES, ANOMALY_FEATURES_FULL, True),
        "transfer": Component("transfer", SHARED_FEATURES, ANOMALY_FEATURES_SHARED, False),
    }
    for key, comp in stacks.items():
        Xtr, comp.categories = to_model_matrix(Ftr, comp.features)
        Xva, _ = to_model_matrix(Fva, comp.features, comp.categories)
        comp.xgb = train_xgb(Xtr, ytr, Xva, yva)
        print(f"[{key}] xgb trained, best_iteration={comp.xgb.best_iteration} "
              f"({time.time() - t0:.0f}s)")
        comp.fit_unsupervised(Ftr)
        print(f"[{key}] unsupervised fitted, k={comp.kmeans.k} ({time.time() - t0:.0f}s)")
        p_va = comp.xgb.predict_proba(Xva)[:, 1]
        u_va = comp.unsupervised_scores(Fva)
        comp.stacker = HybridStacker().fit(p_va, *u_va, yva)
        sv, st = comp.score(Fva), comp.score(Fte)
        for part in ("supervised", "peer", "iforest", "kmeans", "unsup_mean", "hybrid"):
            val_scores[f"{key}:{part}"] = sv[part]
            test_scores[f"{key}:{part}"] = st[part]

    # ------------------------------------------------------------- comparison models
    Xtr, cats = to_model_matrix(Ftr, FULL_FEATURES)
    rf = train_random_forest(Xtr, ytr)
    lr = train_logistic(Xtr, ytr)
    for name, fn, mdl in (("full:random_forest", predict_random_forest, rf),
                          ("full:logistic_regression", predict_logistic, lr)):
        val_scores[name] = fn(mdl, to_model_matrix(Fva, FULL_FEATURES, cats)[0])
        test_scores[name] = fn(mdl, to_model_matrix(Fte, FULL_FEATURES, cats)[0])
    print(f"comparison models done ({time.time() - t0:.0f}s)")
    del rf, lr

    rng = np.random.default_rng(config.RANDOM_STATE)
    val_scores["baseline:random"] = rng.random(va.sum())
    test_scores["baseline:random"] = rng.random(te.sum())
    val_scores["baseline:listing_rule"] = listing_rule_score(Fva, 1)
    test_scores["baseline:listing_rule"] = listing_rule_score(Fte, 2)

    # ---------------------------------------------------------------- evaluation
    for name in test_scores:
        thr = best_f1_threshold(yva, val_scores[name])
        results[name] = evaluate_scores(yte, test_scores[name], thr)
        results[name]["threshold_from_validation"] = thr
    table = pd.DataFrame(results).T
    cols = ["pr_auc", "roc_auc", "precision", "recall", "f1", "precision@1%", "recall@1%",
            "recall@5%", "recall@10%", "lift@1%", "precision@100", "inspect_for_50%_recall",
            "inspect_for_80%_recall"]
    print(table[cols].round(4).sort_values("pr_auc", ascending=False).to_string())

    bands = {}
    for key in ("full", "transfer"):
        t_low, t_high = confidence_bands(yva, val_scores[f"{key}:hybrid"])
        bands[key] = {"t_low": t_low, "t_high": t_high,
                      "test": band_report(yte, test_scores[f"{key}:hybrid"], t_low, t_high)}
        stacks[key].bands = (t_low, t_high)

    # recall by outcome type at a 5% inspection budget (deployed transfer hybrid)
    m5 = top_k_mask(test_scores["transfer:hybrid"], 0.05)
    disp = Fte["disposition"].to_numpy()
    by_outcome = {d: float((m5 & (disp == d)).sum() / max((disp == d).sum(), 1))
                  for d in ("S", "A", "R")}

    imp = {}
    for key in ("full", "transfer"):
        gain = stacks[key].xgb.get_booster().get_score(importance_type="total_gain")
        tot = sum(gain.values())
        imp[key] = dict(sorted(((k, v / tot) for k, v in gain.items()),
                               key=lambda kv: -kv[1])[:15])

    summary = {
        "split": {"train_years": [min(config.TRAIN_YEARS), max(config.TRAIN_YEARS)],
                  "valid_years": [min(config.VALID_YEARS), max(config.VALID_YEARS)],
                  "test_years": [min(config.TEST_YEARS), max(config.TEST_YEARS)],
                  "n_train": int(tr.sum()), "n_valid": int(va.sum()), "n_test": int(te.sum()),
                  "pos_rate_train": float(ytr.mean()), "pos_rate_test": float(yte.mean())},
        "metrics_test": results,
        "confidence_bands": bands,
        "transfer_hybrid_recall_at_5pct_by_outcome": by_outcome,
        "stacker_coefficients": {k: stacks[k].stacker.coef_ for k in stacks},
        "kmeans": {k: {"k": stacks[k].kmeans.k, "silhouette": stacks[k].kmeans.silhouette,
                       "stability_ari": stacks[k].kmeans.stability_ari} for k in stacks},
        "feature_importance_gain": imp,
        "xgb_best_iteration": {k: int(stacks[k].xgb.best_iteration) for k in stacks},
    }
    (config.OUTPUTS_DIR / "lemis_evaluation.json").write_text(json.dumps(summary, indent=2))
    table.to_csv(config.OUTPUTS_DIR / "lemis_evaluation_table.csv")

    # ------------------------------------------------------------- persist artefacts
    for key, comp in stacks.items():
        joblib.dump(comp, config.MODELS_DIR / f"{key}_stack.joblib", compress=3)
    joblib.dump(fb, config.MODELS_DIR / "feature_builder_lemis.joblib", compress=3)

    keep = ["shipment_id", "year", "taxon_display", "common_name", "product_code",
            "product_group", "quantity_std", "value_usd", "purpose", "source", "origin",
            "exporter", "port", "appendix", "label", "disposition"]
    out = Fte[keep].copy()
    for name, s in test_scores.items():
        out[name] = s.astype("float32")
    out.to_parquet(config.PROCESSED_DIR / "lemis_test_scores_full.parquet", index=False)
    slim_test_scores(out).to_parquet(config.OUTPUTS_DIR / "lemis_test_scores.parquet", index=False)

    make_figures(yte, test_scores, results, bands, imp)
    print(f"done in {time.time() - t0:.0f}s")


FEATURE_LABELS = {
    "purpose": "Declared purpose", "origin_top": "Origin country", "genus_top": "Genus",
    "exporter_top": "Exporter country", "product_group": "Product form",
    "combo_rarity": "Rarity of form/purpose/source combination", "log_qty": "Quantity",
    "source": "Source code", "class": "Taxonomic class", "qty_z": "Quantity vs peer norm",
    "route_n_log": "Route volume per million records", "origin_region": "Origin region",
    "route_share": "Route share of genus trade", "peer_n_log": "Peer-group size per million records",
    "purpose_share_in_genus": "Purpose share within genus", "port": "US entry port",
    "log_value": "Declared value", "log_unit_value": "Declared unit value",
    "unit_value_z": "Unit value vs peer norm", "exporter_region": "Exporter region",
    "source_share_in_genus": "Source share within genus", "appendix": "CITES appendix",
}

APP_SCORE_COLS = ["transfer:hybrid", "full:hybrid", "transfer:supervised", "transfer:unsup_mean",
                  "baseline:listing_rule", "baseline:random"]


def slim_test_scores(out):
    """Small copy of the held-out scores for the app's interactive evaluation page."""
    cols = ["year", "purpose", "product_group", "label", "disposition"] + APP_SCORE_COLS
    return out[cols]


def make_figures(yte, scores, results, bands, imp):
    plots.apply_style()
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    # 1. capture (gains) curve: share of enforcement cases caught vs share inspected
    fig, ax = plt.subplots(figsize=(7, 4.4))
    show = [("transfer:hybrid", "WildGuard hybrid (CITES-compatible features)", plots.SERIES[0]),
            ("full:hybrid", "Hybrid with LEMIS-only value/port features", plots.SERIES[1]),
            ("baseline:listing_rule", "Rule: most endangered + wild first", plots.SERIES[2])]
    for name, label, col in show:
        x, c = capture_curve(yte, scores[name])
        ax.plot(x * 100, c * 100, color=col, label=label)
    ax.plot([0, 100], [0, 100], color=plots.BASELINE, lw=1.5, label="Random inspection")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Share of shipments inspected (%), highest score first")
    ax.set_ylabel("Enforcement cases caught (%)")
    ax.set_title("Triage: cases caught per inspection budget (LEMIS 2013-2014, held out)")
    r5 = results["transfer:hybrid"]["recall@5%"] * 100
    ax.annotate(f"{r5:.0f}% caught at 5% inspected", xy=(5, r5), xytext=(6, 38),
                color=plots.INK_2, arrowprops=dict(arrowstyle="-", color=plots.MUTED, lw=1))
    ax.scatter([5], [r5], s=36, color=plots.SERIES[0], zorder=5,
               edgecolor=plots.SURFACE, linewidth=2)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.74))
    plots.save(fig, config.FIGURES_DIR / "capture_curve.png")

    # 2. precision-recall curves
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for name, label, col in [("transfer:hybrid", "Hybrid", plots.SERIES[0]),
                             ("transfer:supervised", "Supervised only (XGBoost)", plots.SERIES[1]),
                             ("transfer:unsup_mean", "Unsupervised only (mean of 3)", plots.SERIES[2])]:
        p, r, _ = precision_recall_curve(yte, scores[name])
        ax.plot(r, p, color=col, label=f"{label}  {results[name]['pr_auc']:.3f}")
    ax.axhline(yte.mean(), color=plots.BASELINE, lw=1.5, label=f"Random  {yte.mean():.3f}")
    ax.set_xlabel("Recall (share of enforcement cases flagged)")
    ax.set_ylabel("Precision")
    ax.set_title("Supervised vs unsupervised vs hybrid (held-out 2013-2014)")
    ax.legend(loc="upper right", title="PR-AUC (transfer feature set)", title_fontsize=9,
              alignment="left")
    plots.save(fig, config.FIGURES_DIR / "pr_curves.png")

    # 3. confidence bands
    rep = bands["transfer"]["test"]
    fig, ax = plt.subplots(figsize=(7, 2.9))
    ax.set_axisbelow(True)
    names = ["auto-clear", "review", "priority"]
    for i, (metric, lab) in enumerate((("share_of_records", "Share of shipments"),
                                       ("share_of_positives", "Share of enforcement cases"))):
        left = 0
        for b in names:
            w = rep[b][metric] * 100
            ax.barh(i, w - 0.3, left=left, height=0.5, color=plots.STATUS[b],
                    label=b if i == 0 else None)
            if w > 5:
                ax.text(left + w / 2, i, f"{w:.1f}%", ha="center", va="center", fontsize=9,
                        color="#0b0b0b" if b == "review" else "#ffffff")
            else:   # too narrow for an inside label: put it just past the bar end
                ax.text(min(left + w + 0.8, 99), i + 0.36, f"{w:.1f}%", ha="right" if left > 90
                        else "left", va="center", fontsize=8.5, color=plots.INK_2)
            left += w
    ax.set_yticks([0, 1], ["Shipments", "Enforcement\ncases"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("%")
    ax.grid(axis="y", visible=False)
    ax.set_title("Confidence-aware triage bands (thresholds fixed on 2011-2012)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3)
    plots.save(fig, config.FIGURES_DIR / "confidence_bands.png")

    # 4. feature importance (transfer model)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    items = list(imp["transfer"].items())[:12][::-1]
    ax.barh([FEATURE_LABELS.get(k, k) for k, _ in items], [v * 100 for _, v in items], height=0.6, color=plots.SERIES[0])
    ax.set_xlabel("Share of total split gain (%)")
    ax.set_title("What drives the transfer model (XGBoost gain)")
    ax.grid(axis="y", visible=False)
    plots.save(fig, config.FIGURES_DIR / "feature_importance_transfer.png")


def regenerate_figures():
    """Redraw the figures from saved scores without retraining."""
    out = pd.read_parquet(config.PROCESSED_DIR / "lemis_test_scores_full.parquet")
    ev = json.loads((config.OUTPUTS_DIR / "lemis_evaluation.json").read_text())
    scores = {c: out[c].to_numpy() for c in out.columns if ":" in c}
    make_figures(out["label"].to_numpy(), scores, ev["metrics_test"], ev["confidence_bands"],
                 ev["feature_importance_gain"])


if __name__ == "__main__":
    import sys
    regenerate_figures() if "--figures-only" in sys.argv else main()
