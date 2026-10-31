"""Steps 5-7 on CITES - cross-dataset label transfer in deployment (DESIGN_QUESTIONS.md Part A).

1. The CITES records rule D selects, 2015-2024, described with the LEMIS peer norms of the
   deployed model (cites_features.py; S2.1).
2. Records already marked as confiscated (source I) or law enforcement (purpose L) are set
   aside as "excluded": telling an analyst to review what has already been seized adds nothing.
3. The deployed ensemble (XGBoost and a random forest, trained on LEMIS 2000-2014) scores every
   other record; its score is the average of the two models' percentile ranks within each year.
4. Records are grouped into consignments: those sharing an export permit, exporter, importer and
   year (a record without a permit stands alone). The countries are part of the key because the
   anonymised permit IDs collide across unrelated trades. A consignment's score is its highest
   record score (S6.1, notebook 14).
5. Consignments are ranked within each year into review tiers by inspection budget:
       priority = top 1 %, review = next 4 %, routine = the rest
   Without CITES labels the score cannot be calibrated as a probability, so it is a ranking of
   where to look first. Every record carries its consignment's tier.
6. Priority records (and a random sample of the rest) get plain-language reasons from the
   XGBoost member, the peer-deviation context, and the US-law warning where it applies (S5.2,
   S6.3).
7. Counts and shares describe exporter-reported records only (ALIGNING section 1.2).

Outputs
    DATA/processed/cites_scored.parquet        every rule D record, scored or excluded (not in git)
    DATA/samples/cites_review_queue.parquet    flagged + sample records with reasons (app)
    SYSTEM CODE/outputs/cites_aggregates/*.csv risk concentration tables (app + report)
    SYSTEM CODE/outputs/cites_record_scale.json per year: tier cut-offs and score quantiles, to
                                               place one permit or new record in its year (app)
    SYSTEM CODE/models/cites_ensemble.joblib   the deployed ensemble with CITES percentile scales

Run:  python -m wildguard.score_cites
"""
import json
import time

import joblib
import numpy as np
import pandas as pd

from . import cites_features, config
from .explain import explain_rows, marine_mammal, peer_context, us_law_warning
from .features import SHARED_FEATURES
from .models import PeerDeviationScorer, percentile_rank, tail_quantiles

PRIORITY, REVIEW = 0.01, 0.05          # cumulative inspection budgets per year, in consignments
AGG_DIR = config.OUTPUTS_DIR / "cites_aggregates"
QUEUE_PATH = config.SAMPLES_DIR / "cites_review_queue.parquet"
SCALE_PATH = config.OUTPUTS_DIR / "cites_record_scale.json"
KEEP = ["shipment_id", "year", "class", "order_name", "family", "genus", "taxon", "taxon_display",
        "product_code", "product_group", "unit_group", "quantity_std", "purpose", "source",
        "origin", "exporter", "importer", "appendix", "reporter_type", "export_permit",
        "origin_region", "exporter_region", "peer_qty_median", "qty_z", "route_share",
        "purpose_share_in_genus", "source_share_in_genus", "combo_rarity"]


def known_enforcement(df):
    return ((df["source"] == "I") | (df["purpose"] == "L")).to_numpy()


def consignment_key(df, cons=None):
    """One consignment per year, exporter, importer and export permit. An export permit is issued
    by one country for one importer, and the anonymised permit IDs collide across unrelated
    trades (a placeholder permit number gets one ID), so the ID alone would merge them."""
    cons = df["consignment"] if cons is None else cons
    return (df["year"].astype(str) + "|" + df["exporter"].astype(str) + "|"
            + df["importer"].astype(str) + "|" + cons.astype(str))


def consignment_tiers(df):
    """Consignment = export permit, else the record itself; score = highest record score;
    tiers by budget within each year. Returns (permit ID, consignment score, tier)."""
    cons = df["export_permit"].astype("object").where(df["export_permit"].notna(),
                                                      "record " + df["shipment_id"].astype(str))
    key = consignment_key(df, cons)
    c_score = df.groupby(key)["score"].transform("max").to_numpy()
    first = ~key.duplicated().to_numpy()
    c = pd.DataFrame({"year": df["year"].to_numpy()[first], "key": key.to_numpy()[first],
                      "score": c_score[first]})
    c["rank"] = c.groupby("year")["score"].rank(ascending=False, method="first")
    c["n"] = c.groupby("year")["score"].transform("size")
    c["tier"] = np.where(c["rank"] <= PRIORITY * c["n"], "priority",
                         np.where(c["rank"] <= REVIEW * c["n"], "review", "routine"))
    tier = pd.Series(c["tier"].to_numpy(), index=c["key"]).reindex(key).to_numpy()
    return cons.to_numpy(), c_score, tier


def write_record_scale(R):
    """Per year, what the app needs to place one permit, or one new record scored as a permit
    of its own, among the year's ranked permits: the consignment-score quantiles and tier
    cut-offs, and each member's score quantiles (a new record's percentile ranks within the year,
    as `Ensemble.score` computes them for the scored records)."""
    first = ~consignment_key(R).duplicated().to_numpy()
    out = {}
    for y in sorted(R["year"].unique()):
        rec = (R["year"] == y).to_numpy()
        c = R[rec & first]
        out[str(int(y))] = {
            "permits": int(len(c)),
            "priority_cut": float(c.loc[c["tier"] == "priority", "consignment_score"].min()),
            "review_cut": float(c.loc[c["tier"] != "routine", "consignment_score"].min()),
            "permit_score": tail_quantiles(c["consignment_score"]),
            "xgboost": tail_quantiles(R.loc[rec, "score_xgboost"]),
            "random_forest": tail_quantiles(R.loc[rec, "score_random_forest"])}
    SCALE_PATH.write_text(json.dumps(out))


def main():
    t0 = time.time()
    cites_features.build()
    ensemble = joblib.load(config.MODELS_DIR / "deployed_ensemble.joblib")
    cols = list(dict.fromkeys(KEEP + SHARED_FEATURES + ["rule_d"]))
    F = cites_features.load_features(columns=cols)
    F = F[F["rule_d"]].drop(columns="rule_d").reset_index(drop=True)
    eligible = ~known_enforcement(F)
    print(f"{len(F):,} rule D records, {eligible.sum():,} ranked ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------------ scores
    R = F[eligible].reset_index(drop=True)
    parts = []
    for lo in range(0, len(R), 1_000_000):
        parts.append(pd.DataFrame(ensemble.member_scores(R.iloc[lo:lo + 1_000_000])))
    M = pd.concat(parts, ignore_index=True)
    R["score_xgboost"] = M["xgboost"].astype("float32").to_numpy()
    R["score_random_forest"] = M["random_forest"].astype("float32").to_numpy()
    R["score"] = ensemble.score(R, groups=R["year"].to_numpy(),
                                members={"xgboost": M["xgboost"].to_numpy(),
                                         "random_forest": M["random_forest"].to_numpy()})
    R["consignment"], R["consignment_score"], R["tier"] = consignment_tiers(R)
    R["peer_deviation"] = PeerDeviationScorer().score(R).astype("float32")
    R["us_law_warning"] = us_law_warning(R) != ""
    print(f"scored ({time.time() - t0:.0f}s)")

    X = F[~eligible].copy()
    X["tier"] = "excluded"
    scored = pd.concat([R, X], ignore_index=True)
    out_cols = KEEP + ["score", "score_xgboost", "score_random_forest", "consignment",
                       "consignment_score", "peer_deviation", "us_law_warning", "tier"]
    scored[[c for c in out_cols if c in scored]].to_parquet(
        config.PROCESSED_DIR / "cites_scored.parquet", index=False)

    # percentile scales for scoring a single record in the app
    ensemble.set_reference(R.sample(min(len(R), 400_000), random_state=config.RANDOM_STATE))
    joblib.dump(ensemble, config.MODELS_DIR / "cites_ensemble.joblib", compress=3)
    write_record_scale(R)

    # ----------------------------------------------------------------- review queue
    rng = np.random.default_rng(config.RANDOM_STATE)
    pri = np.flatnonzero(R["tier"].to_numpy() == "priority")
    rest = np.flatnonzero(R["tier"].to_numpy() != "priority")
    pick = np.sort(np.r_[pri, rng.choice(rest, min(40_000, len(rest)), replace=False)])
    Q = R.iloc[pick].reset_index(drop=True)
    reasons = []
    for lo in range(0, len(Q), 20_000):
        part = Q.iloc[lo:lo + 20_000]
        reasons += explain_rows(ensemble.xgb, ensemble.matrix(part), part)
    Q["reasons"] = [" | ".join(r) for r in reasons]
    Q["peer_context"] = peer_context(PeerDeviationScorer().parts(Q))
    Q["warning"] = us_law_warning(Q)
    queue_cols = [c for c in out_cols if c in Q] + ["reasons", "peer_context", "warning"]
    Q[queue_cols].to_parquet(QUEUE_PATH, index=False)
    print(f"review queue: {len(Q):,} rows with reasons ({time.time() - t0:.0f}s)")

    # ----------------------------------------------------------------- aggregates
    AGG_DIR.mkdir(exist_ok=True)
    E = R[R["reporter_type"] == "E"].copy()       # counts and shares: exporter-reported records
    E["is_priority"] = (E["tier"] == "priority").astype(float)
    E["percentile"] = percentile_rank(E["score"], E["year"])
    for col, min_n in (("year", 0), ("class", 500), ("order_name", 500), ("taxon_display", 500),
                       ("exporter", 2000), ("importer", 2000), ("product_group", 500),
                       ("purpose", 500), ("source", 500), ("appendix", 0)):
        g = E.groupby(col).agg(records=("is_priority", "size"), priority_share=("is_priority", "mean"),
                               mean_risk_percentile=("percentile", "mean"))
        g = g[g["records"] >= min_n].sort_values("priority_share", ascending=False)
        g.round(5).to_csv(AGG_DIR / f"by_{col}.csv")
    routes = (E.groupby(["exporter", "importer"])
              .agg(records=("is_priority", "size"), priority_share=("is_priority", "mean"))
              .query("records >= 1000").sort_values("priority_share", ascending=False))
    routes.head(300).round(5).to_csv(AGG_DIR / "by_route.csv")

    pri_r = R[R["tier"] == "priority"]
    mm = marine_mammal(pri_r) & (pri_r["importer"] != "US").to_numpy()
    summary = {
        "records_rule_d": int(len(F)), "records_ranked": int(len(R)),
        "excluded_known_enforcement": int((~eligible).sum()),
        "years": [int(R["year"].min()), int(R["year"].max())],
        "consignments": int(consignment_key(R).nunique()),
        "priority_records": int((R["tier"] == "priority").sum()),
        "review_records": int((R["tier"] == "review").sum()),
        "priority_consignments_per_year": {
            str(k): int(v) for k, v in consignment_key(R[R["tier"] == "priority"])
            .groupby(R.loc[R["tier"] == "priority", "year"]).nunique().items()},
        # S6.3: how much of the priority tier the US-law warning covers
        "priority_share_marine_mammal_non_us_importer": float(mm.mean()) if len(pri_r) else 0.0,
        "ranked_share_marine_mammal_non_us_importer": float(
            (marine_mammal(R) & (R["importer"] != "US").to_numpy()).mean()),
    }
    (config.OUTPUTS_DIR / "cites_scoring_summary.json").write_text(json.dumps(summary, indent=2, default=int))
    print(json.dumps(summary, indent=2, default=int))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
