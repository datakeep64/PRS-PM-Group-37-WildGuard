"""Human-readable explanations for each flag.

For every scored record, XGBoost's exact per-feature contributions (TreeSHAP, via
pred_contribs=True) say which inputs pushed the risk up. The top positive contributors are
turned into plain-language reasons that an analyst can check against the record, e.g. for the seized
Indian star tortoise shipment (LEMIS control number 2000709917):

    "declared purpose 'Personal' - only 5.9% of Geochelone records"
    "rare route: Malaysia accounts for 0.07% of recorded Geochelone trade"
"""
import numpy as np
import pandas as pd
import xgboost as xgb

from . import config

_CODES = pd.read_csv(config.DATA_DIR / "reference" / "lemis_codes.csv", keep_default_na=False)
COUNTRY = dict(_CODES.loc[_CODES.field == "country", ["code", "value"]].values)
PURPOSE = dict(_CODES.loc[_CODES.field == "purpose", ["code", "value"]].values)
PURPOSE.update({"N": "Reintroduction or introduction into the wild",
                "unknown": "not declared"})
SOURCE = {"W": "wild-caught", "C": "bred in captivity", "F": "born in captivity (F)",
          "R": "ranched", "D": "commercially bred (Appendix I)", "A": "artificially propagated",
          "I": "confiscated/seized specimen", "U": "unknown source", "O": "pre-Convention",
          "unknown": "not declared"}
PORT = dict(_CODES.loc[_CODES.field == "port", ["code", "value"]].values)


def country(code):
    return COUNTRY.get(code, code)


def _fmt_qty(x):
    if pd.isna(x):
        return "?"
    return f"{x:,.0f}" if x >= 10 else f"{x:,.2g}"


def contributions(model, X):
    """Per-row, per-feature contribution to the log-odds (last column = bias)."""
    dm = xgb.DMatrix(X, enable_categorical=True)
    return model.get_booster().predict(dm, pred_contribs=True)


def reason_for(feature, r):
    g = str(r.get("genus", "") or "").capitalize() or "this taxon"
    prod = str(r.get("product_group", "")).replace("_", " ")
    if feature == "qty_z":
        med = r.get("peer_qty_median")
        if pd.notna(med) and med > 0 and pd.notna(r.get("quantity_std")):
            ratio = r["quantity_std"] / med
            word = f"{ratio:,.1f}x" if ratio >= 1 else f"{ratio:.2f}x"
            return (f"quantity {_fmt_qty(r['quantity_std'])} is {word} the usual for "
                    f"{g} {prod} (peer median {_fmt_qty(med)})")
        return "quantity unusual for its peer group"
    if feature == "unit_value_z":
        med = r.get("peer_unit_value_median")
        if pd.notna(med) and med > 0 and pd.notna(r.get("value_usd")) and r.get("quantity_std"):
            uv = r["value_usd"] / r["quantity_std"]
            return (f"declared ${uv:,.2f}/unit vs peer median ${med:,.2f}/unit "
                    f"({uv / med:,.1f}x)")
        return "declared unit value unusual for its peer group"
    if feature in ("purpose", "purpose_share_in_genus"):
        share = r.get("purpose_share_in_genus")
        tail = f" - only {share:.1%} of {g} records" if pd.notna(share) and share < 0.2 else ""
        return f"declared purpose '{PURPOSE.get(r['purpose'], r['purpose'])}'{tail}"
    if feature in ("source", "source_share_in_genus", "is_wild"):
        share = r.get("source_share_in_genus")
        tail = f" - only {share:.1%} of {g} records" if pd.notna(share) and share < 0.2 else ""
        return f"source: {SOURCE.get(r['source'], r['source'])}{tail}"
    if feature in ("route_share", "route_n_log"):
        share = r.get("route_share", 0)
        if share < 0.01:
            return (f"rare route: {country(r['exporter'])} accounts for {share:.2%} of "
                    f"recorded {g} trade")
        return f"exported from {country(r['exporter'])}"
    if feature == "combo_rarity":
        return (f"uncommon combination: {prod} / {PURPOSE.get(r['purpose'], r['purpose'])} / "
                f"{SOURCE.get(r['source'], r['source'])}")
    if feature in ("exporter_top", "exporter_region"):
        return f"exported from {country(r['exporter'])}"
    if feature in ("origin_top", "origin_region"):
        return f"origin {country(r['origin'])}"
    if feature == "reexport_flag":
        return ("re-exported via a third country" if r.get("reexport_flag") == 1
                else "shipped directly from the origin country")
    if feature in ("product_group", "unit_group", "log_qty"):
        return f"product type: {prod}"
    if feature == "genus_top":
        return f"taxon: {r.get('taxon_display', g)}"
    if feature == "class":
        return f"taxonomic class {r.get('class')}"
    if feature == "appendix":
        a = r.get("appendix")
        if a == "genus_listed":
            return "species not found in CITES records; its genus is listed"
        return "species not CITES-listed" if a in ("not_listed", "N") else f"CITES Appendix {a}"
    if feature == "peer_n_log":
        return "few comparable records for this taxon/product (thin peer group)"
    if feature == "port":
        return f"entry port {PORT.get(r.get('port'), r.get('port'))}"
    if feature in ("value_missing", "log_value", "log_unit_value"):
        return ("no declared value" if r.get("value_missing") == 1
                else f"declared value ${r.get('value_usd', 0):,.0f}")
    if feature == "month":
        return "time of year"
    return feature


def explain_rows(model, X, frame, top_n=3):
    """Return a list of top_n plain-language reasons per row (positive contributions only,
    duplicates merged)."""
    contrib = contributions(model, X)[:, :-1]
    names = list(X.columns)
    out = []
    for i in range(len(X)):
        r = frame.iloc[i]
        order = np.argsort(-contrib[i])
        reasons = []
        for j in order:
            if contrib[i, j] <= 0 or len(reasons) >= top_n:
                break
            text = reason_for(names[j], r)
            if text not in reasons:
                reasons.append(text)
        out.append(reasons)
    return out
