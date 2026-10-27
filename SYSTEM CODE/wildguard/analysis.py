"""Supplementary analyses of the held-out LEMIS results (robustness of the headline claims).

  commercial_only  evaluation restricted to purpose = T (commercial) - most CITES trade is
                   commercial, whereas half of LEMIS enforcement cases are personal imports,
                   so this is the more relevant (and harder) test for transfer to CITES
  shipment_level   line items aggregated to shipments (max score, any enforcement) - checks
                   that multi-item shipments do not inflate line-item metrics
  by_year          2013 vs 2014 separately (stability over time)
  calibration      reliability of the hybrid probability in deciles of predicted risk
  examples         the Indian star tortoise case (in-sample, illustrative) and the
                   highest-ranked held-out enforcement case

Run:  python -m wildguard.analysis      (after train.py)
"""
import json

import joblib
import numpy as np
import pandas as pd

from . import config
from .evaluate import evaluate_scores
from .explain import explain_rows
from .features import to_model_matrix
from .train import load_lemis

MODELS = ["transfer:hybrid", "transfer:supervised", "full:hybrid", "transfer:unsup_mean",
          "baseline:listing_rule", "baseline:random"]
KEEP = ["pr_auc", "roc_auc", "precision@1%", "recall@5%", "lift@1%", "inspect_for_80%_recall",
        "base_rate"]


def _clean(o):
    """Replace NaN with None so the output is strict JSON (readable by any JSON parser)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and o != o:
        return None
    return o


def _table(df):
    y = df["label"].to_numpy()
    return {m: {k: round(v, 4) for k, v in evaluate_scores(y, df[m].to_numpy(), 0.5).items()
                if k in KEEP} for m in MODELS}


def main():
    t = pd.read_parquet(config.PROCESSED_DIR / "lemis_test_scores_full.parquet")
    out = {"commercial_only": _table(t[t["purpose"] == "T"]),
           "personal_only": _table(t[t["purpose"] == "P"])}
    out["share_of_test_positives"] = {
        "commercial": float(t.loc[t.purpose == "T", "label"].sum() / t["label"].sum()),
        "personal": float(t.loc[t.purpose == "P", "label"].sum() / t["label"].sum())}

    shp = t.groupby("shipment_id").agg(label=("label", "max"), **{m: (m, "max") for m in MODELS})
    out["shipment_level"] = _table(shp)
    out["shipment_level_n"] = int(len(shp))
    out["by_year"] = {int(y): _table(g) for y, g in t.groupby("year")}

    s, y = t["transfer:hybrid"].to_numpy(), t["label"].to_numpy()
    dec = pd.qcut(s, 10, labels=False, duplicates="drop")
    cal = pd.DataFrame({"pred": s, "obs": y, "decile": dec}).groupby("decile").mean()
    out["calibration_deciles"] = cal.round(5).to_dict(orient="index")
    out["brier_transfer_hybrid"] = float(np.mean((s - y) ** 2))

    # ------------------------------------------------------------------ worked examples
    stack = joblib.load(config.MODELS_DIR / "transfer_stack.joblib")
    full = joblib.load(config.MODELS_DIR / "full_stack.joblib")
    fb = joblib.load(config.MODELS_DIR / "feature_builder_lemis.joblib")
    df = load_lemis()
    tort = df[df["shipment_id"] == "2000709917"]
    F = fb.transform(tort)
    ex = {}
    for name, st in (("transfer", stack), ("full", full)):
        sc = st.score(F)
        X, _ = to_model_matrix(F, st.features, st.categories)
        ex[name] = {"hybrid_probability": float(sc["hybrid"][0]),
                    "supervised_probability": float(sc["supervised"][0]),
                    "reasons": explain_rows(st.xgb, X, F)[0]}
        # percentile of this score among held-out test shipments' scores, same model
        ref = t["transfer:hybrid" if name == "transfer" else "full:hybrid"].to_numpy()
        ex[name]["percentile_vs_test_years"] = float((ref < sc["hybrid"][0]).mean())
    ex["record"] = tort[["taxon_display", "common_name", "product_code", "quantity_std",
                         "value_usd", "purpose", "source", "origin", "exporter", "port",
                         "disposition"]].iloc[0].to_dict()
    ex["note"] = "shipment year 2000 is a training year: illustrative, in-sample"
    out["example_tortoise"] = ex

    top = t[t["label"] == 1].sort_values("transfer:hybrid", ascending=False)
    top = top.drop_duplicates(["taxon_display", "product_group", "purpose", "origin"]).head(5)
    rows = df[df["shipment_id"].isin(top["shipment_id"])].drop_duplicates("shipment_id")
    Ft = fb.transform(rows)
    X, _ = to_model_matrix(Ft, stack.features, stack.categories)
    reasons = explain_rows(stack.xgb, X, Ft)
    out["examples_top_heldout_cases"] = [
        {**{k: (v if not isinstance(v, (np.floating, np.integer)) else v.item())
            for k, v in r[["year", "taxon_display", "common_name", "product_code", "quantity_std",
                           "value_usd", "purpose", "source", "origin", "exporter",
                           "disposition"]].to_dict().items()},
         "reasons": rs}
        for (_, r), rs in zip(Ft.iterrows(), reasons)]
    # ------------------------------------------------ deployment check: IW vs unweighted
    # Importance weighting re-focuses the transfer model on CITES-like trade, so the fair
    # comparison is on held-out LEMIS records that resemble CITES records (CITES-listed taxa).
    iw_path = config.MODELS_DIR / "transfer_iw_stack.joblib"
    if iw_path.exists():
        iw = joblib.load(iw_path)
        test = df[df["year"].isin(config.TEST_YEARS)].reset_index(drop=True)
        Fte = fb.transform(test)
        s_base, s_iw = stack.score(Fte)["hybrid"], iw.score(Fte)["hybrid"]
        y = Fte["label"].to_numpy()
        listed = Fte["appendix"].isin(["I", "II", "III"]).to_numpy()
        commercial = (Fte["purpose"] == "T").to_numpy()
        check = {}
        for name, m in (("all_test", np.ones(len(y), bool)), ("cites_listed", listed),
                        ("cites_listed_commercial", listed & commercial)):
            check[name] = {"n": int(m.sum()), "positives": int(y[m].sum())}
            for label, s in (("unweighted", s_base), ("importance_weighted", s_iw)):
                r = evaluate_scores(y[m], s[m], 0.5)
                check[name][label] = {k: round(r[k], 4) for k in KEEP}
        out["deployment_check"] = check

    (config.OUTPUTS_DIR / "lemis_supplementary.json").write_text(
        json.dumps(_clean(out), indent=2, default=str))
    print(json.dumps({k: out[k] for k in ("commercial_only", "shipment_level",
                                          "share_of_test_positives")}, indent=1))
    print(json.dumps(out["example_tortoise"], indent=1, default=str))
    print(json.dumps(out["examples_top_heldout_cases"], indent=1, default=str))


if __name__ == "__main__":
    main()
