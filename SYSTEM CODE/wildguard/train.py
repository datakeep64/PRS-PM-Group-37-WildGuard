"""Steps 2-6 on LEMIS - train the chosen models, run the one test, refit for deployment.

The design was chosen on 2011 (SYSTEM CODE/DESIGN_QUESTIONS.md, notebooks 08-14). This step:
  1. trains on 2000-2010, animals with a class: the ensemble (XGBoost and a random forest,
     rank-averaged; the CITES-compatible features), its two members, logistic regression, and
     the same ensemble with the LEMIS-only fields added (the "full" model, for C3);
  2. runs the one test on 2013-2014 (B10), against random ranking and the "most endangered
     first" listing rule: on all line items, listed species, commercial trade, and listed
     commercial trade (C1); per line item and per shipment (C2); with bootstrap intervals;
  3. fixes the LEMIS confidence bands on the held-back year 2012;
  4. refits the ensembles on 2000-2014 for deployment (B1: performance falls with the gap
     between training and scored years, notebook 13), and records where the full model's
     scores fall among the deployment years' shipments (the app's "Check a record").

Run:  python -m wildguard.train
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from . import config, plots
from .evaluate import (band_report, best_f1_threshold, capture_curve, confidence_bands,
                       evaluate_scores, pr_auc_bootstrap, shipment_scores)
from .features import FULL_FEATURES, SHARED_FEATURES, FeatureBuilder
from .models import Ensemble, LogisticModel, tail_quantiles

SCALE_PATH = config.OUTPUTS_DIR / "lemis_record_scale.json"

LEMIS_COLS = ["shipment_id", "year", "month", "class", "genus", "taxon", "taxon_display",
              "common_name", "product_code", "product_group", "unit_group", "quantity_std",
              "value_usd", "purpose", "source", "origin", "exporter", "port", "label",
              "disposition", "appendix", "origin_region", "exporter_region", "animal", "sp_listed"]
SUBSETS = {"all": lambda d: np.ones(len(d), bool),
           "listed": lambda d: d["sp_listed"].to_numpy(),
           "commercial": lambda d: (d["purpose"] == "T").to_numpy(),
           "listed_commercial": lambda d: (d["sp_listed"] & (d["purpose"] == "T")).to_numpy()}


def load_lemis():
    """LEMIS line items of animals with a class, the training population (ALIGNING section 2.2)."""
    df = pd.read_parquet(config.LEMIS_PARQUET, columns=LEMIS_COLS)
    df = df[df["animal"]].reset_index(drop=True)
    frac = float(os.environ.get("WILDGUARD_SAMPLE", "1"))     # e.g. 0.03 for a quick test
    if frac < 1:
        keep = set(df["shipment_id"].drop_duplicates().sample(frac=frac, random_state=config.RANDOM_STATE))
        df = df[df["shipment_id"].isin(keep)].reset_index(drop=True)
    df["month"] = df["month"].astype("float32")
    return df


def listing_rule_score(df, seed=0):
    """The manual heuristic: inspect the most endangered (Appendix I) and wild-caught first,
    using the listing in force in the shipment year."""
    rank = df["appendix"].map({"I": 3, "II": 2, "III": 2}).fillna(1)
    jitter = np.random.default_rng(seed).random(len(df)) * 1e-3
    return (rank + 0.5 * (df["source"] == "W") + jitter).to_numpy()


def score_all(models, F, seed):
    """Every model's score on one population; ensemble scores are percentile ranks within it."""
    ens, full, logit = models["ensemble"], models["full"], models["logistic"]
    m = ens.member_scores(F)
    out = {"ensemble": ens.score(F, members=m), "xgboost": m["xgboost"],
           "random_forest": m["random_forest"], "logistic_regression": logit.predict(F),
           "full_ensemble": full.score(F), "baseline:listing_rule": listing_rule_score(F, seed),
           "baseline:random": np.random.default_rng(config.RANDOM_STATE + seed).random(len(F))}
    return out


def write_record_scale(full, F):
    """Where one US declaration, scored by the full model as a shipment of its own, falls among
    the deployment years' shipments (each scored by its highest line, as in C2): the
    shipment-score quantiles and the cut-offs of the CITES queue's budgets, top 1% and next 4%."""
    _, ship = shipment_scores(F["shipment_id"].to_numpy(), np.zeros(len(F)), full.score_single(F))
    SCALE_PATH.write_text(json.dumps({
        "years": [min(config.DEPLOY_YEARS), max(config.DEPLOY_YEARS)], "shipments": int(len(ship)),
        "priority_cut": float(np.quantile(ship, 0.99)), "review_cut": float(np.quantile(ship, 0.95)),
        "shipment_score": tail_quantiles(ship)}))


def main():
    t0 = time.time()
    df = load_lemis()
    yr = df["year"].to_numpy()
    tr, hb, te = (np.isin(yr, list(config.TRAIN_YEARS)), yr == config.HELD_BACK_YEAR,
                  np.isin(yr, list(config.TEST_YEARS)))
    print(f"line items: train {tr.sum():,}  held back {hb.sum():,}  test {te.sum():,}")

    fb = FeatureBuilder(use_value=True).fit(df[tr])
    F_tr, F_hb, F_te = (fb.transform(df[m]).reset_index(drop=True) for m in (tr, hb, te))
    y_tr, y_hb, y_te = F_tr["label"].to_numpy(), F_hb["label"].to_numpy(), F_te["label"].to_numpy()
    print(f"features built ({time.time() - t0:.0f}s)")

    models = {"ensemble": Ensemble(SHARED_FEATURES).fit(F_tr, y_tr),
              "full": Ensemble(FULL_FEATURES).fit(F_tr, y_tr),
              "logistic": LogisticModel(SHARED_FEATURES).fit(F_tr, y_tr)}
    print(f"models trained ({time.time() - t0:.0f}s)")

    S_hb, S_te = score_all(models, F_hb, 1), score_all(models, F_te, 2)

    # ------------------------------------------------------------- the one test (B10)
    results, intervals = {}, {}
    for sub, pick in SUBSETS.items():
        m = pick(F_te)
        for unit in ("line_items", "shipments"):
            res = {}
            for name, s in S_te.items():
                if unit == "line_items":
                    yy, ss = y_te[m], s[m]
                    thr = best_f1_threshold(y_hb[pick(F_hb)], S_hb[name][pick(F_hb)])
                else:
                    yy, ss = shipment_scores(F_te["shipment_id"].to_numpy()[m], y_te[m], s[m])
                    thr = np.inf
                res[name] = evaluate_scores(yy, ss, thr)
            results[f"{sub}:{unit}"] = res
        names = ["ensemble", "xgboost", "random_forest", "logistic_regression", "baseline:listing_rule"]
        single, pairs = pr_auc_bootstrap(
            y_te[m], {k: S_te[k][m] for k in names}, F_te["shipment_id"].to_numpy()[m],
            [("ensemble", k) for k in names[1:]], n_boot=1000 if sub == "listed" else 300)
        intervals[sub] = {"pr_auc": single.round(4).to_dict(orient="index"),
                          "ensemble_minus": pairs.round(4).to_dict(orient="records")}
        print(f"evaluated on {sub} ({time.time() - t0:.0f}s)")

    # what the LEMIS-only fields add (C3): the 2013 line items with a recorded value
    has_value = (F_te["year"] == 2013).to_numpy() & (F_te["value_usd"] > 0).to_numpy()
    c3_single, c3_pairs = pr_auc_bootstrap(
        y_te[has_value], {k: S_te[k][has_value] for k in ("full_ensemble", "ensemble")},
        F_te["shipment_id"].to_numpy()[has_value], [("full_ensemble", "ensemble")])
    c3 = {"line_items": int(has_value.sum()), "pr_auc": c3_single.round(4).to_dict(orient="index"),
          "full_minus_shared": c3_pairs.round(4).to_dict(orient="records")}

    # LEMIS confidence bands, fixed on the held-back year (percentile thresholds)
    t_low, t_high = confidence_bands(y_hb, S_hb["ensemble"])
    bands = {"t_low": t_low, "t_high": t_high, "fixed_on": config.HELD_BACK_YEAR,
             "test": band_report(y_te, S_te["ensemble"], t_low, t_high)}

    gain = models["ensemble"].xgb.get_booster().get_score(importance_type="total_gain")
    tot = sum(gain.values())
    imp = dict(sorted(((k, v / tot) for k, v in gain.items()), key=lambda kv: -kv[1])[:15])

    summary = {
        "years": {"train": [min(config.TRAIN_YEARS), max(config.TRAIN_YEARS)],
                  "held_back": config.HELD_BACK_YEAR,
                  "test": [min(config.TEST_YEARS), max(config.TEST_YEARS)],
                  "deployed_model_trained_on": [min(config.DEPLOY_YEARS), max(config.DEPLOY_YEARS)]},
        "line_items": {"train": int(tr.sum()), "held_back": int(hb.sum()), "test": int(te.sum())},
        "positive_rate": {"train": float(y_tr.mean()), "test": float(y_te.mean())},
        "metrics_test": results,
        "pr_auc_intervals": intervals,
        "lemis_only_fields_c3": c3,
        "confidence_bands": bands,
        "feature_importance_gain": imp,
    }
    (config.OUTPUTS_DIR / "lemis_evaluation.json").write_text(json.dumps(summary, indent=2))
    table = pd.DataFrame(results["listed:line_items"]).T
    table.to_csv(config.OUTPUTS_DIR / "lemis_evaluation_table.csv")

    keep = ["shipment_id", "year", "taxon_display", "common_name", "product_code", "product_group",
            "quantity_std", "value_usd", "purpose", "source", "origin", "exporter", "port",
            "appendix", "sp_listed", "label", "disposition"]
    out = F_te[keep].copy()
    for name, s in S_te.items():
        out[name] = np.asarray(s).astype("float32")
    out.to_parquet(config.PROCESSED_DIR / "lemis_test_scores_full.parquet", index=False)
    out[["year", "purpose", "product_group", "sp_listed", "label", "disposition", "shipment_id"]
        + list(S_te)].to_parquet(config.OUTPUTS_DIR / "lemis_test_scores.parquet", index=False)
    joblib.dump(models["ensemble"], config.MODELS_DIR / "tested_ensemble.joblib", compress=3)
    make_figures(F_te, S_te, results, bands, imp)
    print(f"test done ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------- deployment refit (B1)
    dp = np.isin(yr, list(config.DEPLOY_YEARS))
    fb_d = FeatureBuilder(use_value=True).fit(df[dp])
    F_d = fb_d.transform(df[dp]).reset_index(drop=True)
    y_d = F_d["label"].to_numpy()
    deployed = Ensemble(SHARED_FEATURES).fit(F_d, y_d)
    full_d = Ensemble(FULL_FEATURES).fit(F_d, y_d)
    full_d.set_reference(F_d.sample(min(len(F_d), 400_000), random_state=config.RANDOM_STATE))
    write_record_scale(full_d, F_d)
    joblib.dump(deployed, config.MODELS_DIR / "deployed_ensemble.joblib", compress=3)
    joblib.dump(full_d, config.MODELS_DIR / "full_ensemble.joblib", compress=3)
    joblib.dump(fb_d, config.MODELS_DIR / "feature_builder_lemis.joblib", compress=3)
    print(f"deployment models refitted on {min(config.DEPLOY_YEARS)}-{max(config.DEPLOY_YEARS)} "
          f"({time.time() - t0:.0f}s)")


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
    "unit_group": "Unit", "reexport_flag": "Re-exported (origin differs from exporter)",
    "is_wild": "Wild-caught source",
}


def make_figures(F_te, S, results, bands, imp):
    plots.apply_style()
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    listed = F_te["sp_listed"].to_numpy()
    y, ships = F_te["label"].to_numpy(), F_te["shipment_id"].to_numpy()

    # 1. capture curve, in shipments (C2), listed species
    fig, ax = plt.subplots(figsize=(7, 4.4))
    show = [("ensemble", "WildGuard (XGBoost + random forest)", plots.SERIES[0]),
            ("full_ensemble", "With LEMIS-only value, port and month", plots.SERIES[1]),
            ("baseline:listing_rule", "Rule: most endangered + wild first", plots.SERIES[2])]
    for name, label, col in show:
        yy, ss = shipment_scores(ships[listed], y[listed], S[name][listed])
        x, c = capture_curve(yy, ss)
        ax.plot(x * 100, c * 100, color=col, label=label)
    ax.plot([0, 100], [0, 100], color=plots.BASELINE, lw=1.5, label="Random inspection")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Share of shipments inspected (%), highest score first")
    ax.set_ylabel("Shipments with an enforcement case caught (%)")
    ax.set_title("Triage on CITES-listed species (LEMIS 2013-2014, held out)")
    r5 = results["listed:shipments"]["ensemble"]["recall@5%"] * 100
    ax.annotate(f"{r5:.0f}% caught at 5% inspected", xy=(5, r5), xytext=(6, 38),
                color=plots.INK_2, arrowprops=dict(arrowstyle="-", color=plots.MUTED, lw=1))
    ax.scatter([5], [r5], s=36, color=plots.SERIES[0], zorder=5, edgecolor=plots.SURFACE, linewidth=2)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.74))
    plots.save(fig, config.FIGURES_DIR / "capture_curve.png")

    # 2. precision-recall curves, listed line items
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for name, label, col in [("ensemble", "Ensemble", plots.SERIES[0]),
                             ("xgboost", "XGBoost alone", plots.SERIES[1]),
                             ("logistic_regression", "Logistic regression", plots.SERIES[2])]:
        p, r, _ = precision_recall_curve(y[listed], S[name][listed])
        ax.plot(r, p, color=col, label=f"{label}  {results['listed:line_items'][name]['pr_auc']:.3f}")
    base = y[listed].mean()
    ax.axhline(base, color=plots.BASELINE, lw=1.5, label=f"Random  {base:.3f}")
    ax.set_xlabel("Recall (share of enforcement cases flagged)")
    ax.set_ylabel("Precision")
    ax.set_title("CITES-listed species, held-out 2013-2014")
    ax.legend(loc="upper right", title="PR-AUC", title_fontsize=9, alignment="left")
    plots.save(fig, config.FIGURES_DIR / "pr_curves.png")

    # 3. confidence bands
    rep = bands["test"]
    fig, ax = plt.subplots(figsize=(7, 2.9))
    ax.set_axisbelow(True)
    names = ["auto-clear", "review", "priority"]
    for i, (metric, lab) in enumerate((("share_of_records", "Share of line items"),
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
    ax.set_yticks([0, 1], ["Line items", "Enforcement\ncases"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("%")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Confidence-aware triage bands (thresholds fixed on {config.HELD_BACK_YEAR})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3)
    plots.save(fig, config.FIGURES_DIR / "confidence_bands.png")

    # 4. feature importance (XGBoost member)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    items = list(imp.items())[:12][::-1]
    ax.barh([FEATURE_LABELS.get(k, k) for k, _ in items], [v * 100 for _, v in items], height=0.6,
            color=plots.SERIES[0])
    ax.set_xlabel("Share of total split gain (%)")
    ax.set_title("What drives the XGBoost member (gain)")
    ax.grid(axis="y", visible=False)
    plots.save(fig, config.FIGURES_DIR / "feature_importance_transfer.png")


if __name__ == "__main__":
    main()
