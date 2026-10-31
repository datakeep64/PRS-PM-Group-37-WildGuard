"""Supplementary analyses of the held-out LEMIS results (after train.py).

The held-out sets of DESIGN_QUESTIONS C1 (all, listed, commercial, listed commercial) and the
shipment-level results of C2 are in lemis_evaluation.json. This adds:
  personal_only  purpose P: half of LEMIS enforcement cases are personal imports
  by_year        2013 and 2014 separately (stability over time)
  calibration    the XGBoost member's probability in deciles (the ensemble score is a rank, so
                 it has no probability to calibrate)
  examples       the Indian star tortoise case (in-sample, illustrative) and the highest-ranked
                 held-out enforcement cases, with their reasons

Run:  python -m wildguard.analysis      (after train.py)
"""
import json

import joblib
import numpy as np
import pandas as pd

from . import config
from .evaluate import evaluate_scores
from .explain import explain_rows
from .features import FeatureBuilder
from .train import load_lemis

MODELS = ["ensemble", "xgboost", "random_forest", "logistic_regression", "full_ensemble",
          "baseline:listing_rule", "baseline:random"]
KEEP = ["pr_auc", "roc_auc", "precision@1%", "recall@1%", "recall@5%", "lift@1%",
        "inspect_for_80%_recall", "base_rate"]
RECORD = ["year", "taxon_display", "common_name", "product_code", "quantity_std", "value_usd",
          "purpose", "source", "origin", "exporter", "port", "disposition"]


def _clean(o):
    """Replace NaN with None so the output is strict JSON (readable by any JSON parser)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def _table(df):
    y = df["label"].to_numpy()
    return {m: {k: round(v, 4) for k, v in evaluate_scores(y, df[m].to_numpy(), np.inf).items()
                if k in KEEP} for m in MODELS}


def main():
    t = pd.read_parquet(config.PROCESSED_DIR / "lemis_test_scores_full.parquet")
    listed = t[t["sp_listed"]]
    out = {"personal_only": _table(t[t["purpose"] == "P"]),
           "share_of_test_positives": {
               "commercial": float(t.loc[t.purpose == "T", "label"].sum() / t["label"].sum()),
               "personal": float(t.loc[t.purpose == "P", "label"].sum() / t["label"].sum())},
           "by_year": {int(y): {"all": _table(g), "listed": _table(g[g["sp_listed"]])}
                       for y, g in t.groupby("year")}}
    s, y = listed["xgboost"].to_numpy(), listed["label"].to_numpy()
    dec = pd.qcut(s, 10, labels=False, duplicates="drop")
    out["calibration_xgboost_listed"] = pd.DataFrame({"predicted": s, "observed": y, "decile": dec}) \
        .groupby("decile").mean().round(5).to_dict(orient="index")

    # ------------------------------------------------------------------ worked examples
    ens = joblib.load(config.MODELS_DIR / "tested_ensemble.joblib")
    df = load_lemis()
    fb = FeatureBuilder(use_value=True).fit(df[df["year"].isin(config.TRAIN_YEARS)])  # the tested norms
    tort = fb.transform(df[df["shipment_id"] == "2000709917"]).reset_index(drop=True)
    m = ens.member_scores(tort)
    ref = {k: t[k].to_numpy() for k in ("xgboost", "random_forest")}
    pct = {k: float((ref[k] < m[k][0]).mean()) for k in ref}
    out["example_tortoise"] = {
        "record": tort[RECORD].iloc[0].to_dict(),
        "member_percentile_vs_test_years": pct,
        "ensemble_percentile_vs_test_years": (pct["xgboost"] + pct["random_forest"]) / 2,
        "reasons": explain_rows(ens.xgb, ens.matrix(tort), tort)[0],
        "note": "shipment year 2000 is a training year: illustrative, in-sample"}

    top = t[t["label"] == 1].sort_values("ensemble", ascending=False)
    top = top.drop_duplicates(["taxon_display", "product_group", "purpose", "origin"]).head(5)
    rows = df[df["shipment_id"].isin(top["shipment_id"])].drop_duplicates("shipment_id")
    Ft = fb.transform(rows).reset_index(drop=True)
    out["examples_top_heldout_cases"] = [
        {**r[RECORD].to_dict(), "reasons": rs}
        for (_, r), rs in zip(Ft.iterrows(), explain_rows(ens.xgb, ens.matrix(Ft), Ft))]

    (config.OUTPUTS_DIR / "lemis_supplementary.json").write_text(
        json.dumps(_clean(out), indent=2, default=str))
    print(json.dumps(_clean({k: out[k] for k in ("personal_only", "share_of_test_positives")}), indent=1))
    print(json.dumps(_clean(out["example_tortoise"]), indent=1, default=str))


if __name__ == "__main__":
    main()
