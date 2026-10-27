"""Steps 5-7 on CITES - cross-dataset label transfer in deployment.

1. CITES 2015-2024 animal records are described with the shared feature set, using peer
   norms learned from CITES itself (cites_features.py).
2. The supervised part of the deployed stack (trained on LEMIS outcomes) scores each record.
3. The three unsupervised detectors are re-fitted on CITES - they need no labels - so
   "anomalous" means anomalous relative to global CITES trade.
4. The LEMIS-trained hybrid combiner merges the four signals into one risk score.
5. Records that are already enforcement material - source "I" (confiscated/seized specimen) or
   purpose "L" (law enforcement / judicial / forensic) - are set aside as tier "excluded": an
   analyst gains nothing from being told to review what has already been seized.
6. The remaining records are ranked within each year into review tiers by inspection budget:
       priority = top 1 %, review = next 4 %, routine = remaining 95 %
   Budget-based tiers are used on CITES because, without CITES labels, the score cannot be
   calibrated as a probability there; it is a *ranking* of where to look first.
7. Every priority record (and a random sample of the rest) gets plain-language reasons.

Outputs
    DATA/processed/cites_scored.parquet        every scored record (large, not in git)
    DATA/samples/cites_review_queue.parquet    flagged + sample records with reasons (app)
    SYSTEM CODE/outputs/cites_aggregates/*.csv risk concentration tables (app + report)

Run:  python -m wildguard.score_cites
"""
import copy
import json
import time

import joblib
import numpy as np
import pandas as pd

from . import cites_features, config
from .explain import explain_rows
from .features import to_model_matrix

PRIORITY, REVIEW = 0.01, 0.05          # cumulative inspection budgets per year
AGG_DIR = config.OUTPUTS_DIR / "cites_aggregates"
QUEUE_PATH = config.SAMPLES_DIR / "cites_review_queue.parquet"


def deployed_stack_name():
    p = config.MODELS_DIR / "deployment.json"
    return json.loads(p.read_text())["stack"] if p.exists() else "transfer"


def load_cites_stack():
    """The deployed stack with its unsupervised detectors re-fitted on CITES."""
    path = config.MODELS_DIR / "cites_stack.joblib"
    if path.exists():
        return joblib.load(path)
    raise FileNotFoundError("run python -m wildguard.score_cites first")


def known_enforcement(df):
    """Records that are already enforcement material and so are not ranked for review."""
    return ((df["source"] == "I") | (df["purpose"] == "L")).to_numpy()


def tiers_by_year(df, score_col, eligible):
    pct = df[score_col].where(eligible).groupby(df["year"]).rank(pct=True, method="first")
    tier = np.where(~eligible, "excluded",
                    np.where(pct > 1 - PRIORITY, "priority",
                             np.where(pct > 1 - REVIEW, "review", "routine")))
    return pct.to_numpy(), tier


def main():
    t0 = time.time()
    cites_features.build()
    name = deployed_stack_name()
    stack = copy.copy(joblib.load(config.MODELS_DIR / f"{name}_stack.joblib"))
    F = cites_features.load_features()
    print(f"loaded {len(F):,} CITES records, deployed stack = {name} ({time.time() - t0:.0f}s)")

    stack.fit_unsupervised(F)
    stack.name = f"{name}_cites"
    joblib.dump(stack, config.MODELS_DIR / "cites_stack.joblib", compress=3)
    print(f"unsupervised detectors re-fitted on CITES, k={stack.kmeans.k} "
          f"({time.time() - t0:.0f}s)")

    parts = []
    for lo in range(0, len(F), 1_000_000):
        parts.append(pd.DataFrame(stack.score(F.iloc[lo:lo + 1_000_000])))
    S = pd.concat(parts, ignore_index=True)
    for c in S.columns:
        F[f"score_{c}"] = S[c].astype("float32").to_numpy()
    eligible = ~known_enforcement(F)
    F["risk_percentile"], F["tier"] = tiers_by_year(F, "score_hybrid", eligible)
    F["cluster"] = stack.kmeans.assign(F).astype("int16")
    print(f"scored ({time.time() - t0:.0f}s)")

    keep = ["shipment_id", "year", "class", "order_name", "family", "genus", "taxon",
            "taxon_display", "product_code", "product_group", "unit_group", "quantity_std",
            "purpose", "source", "origin", "exporter", "importer", "appendix", "reporter_type",
            "origin_region", "exporter_region", "peer_qty_median", "qty_z", "route_share",
            "purpose_share_in_genus", "source_share_in_genus", "cluster", "risk_percentile",
            "tier"] + [c for c in F.columns if c.startswith("score_")]
    F[keep].to_parquet(config.PROCESSED_DIR / "cites_scored.parquet", index=False)

    # ----------------------------------------------------------------- review queue
    rng = np.random.default_rng(config.RANDOM_STATE)
    pri = np.flatnonzero(F["tier"].to_numpy() == "priority")
    rest = np.flatnonzero(np.isin(F["tier"].to_numpy(), ["review", "routine"]))
    pick = np.sort(np.r_[pri, rng.choice(rest, 40_000, replace=False)])
    Q = F.iloc[pick].reset_index(drop=True)
    X, _ = to_model_matrix(Q, stack.features, stack.categories)
    reasons = []
    for lo in range(0, len(Q), 20_000):
        reasons += explain_rows(stack.xgb, X.iloc[lo:lo + 20_000], Q.iloc[lo:lo + 20_000])
    Q["reasons"] = [" | ".join(r) for r in reasons]
    queue_cols = [c for c in keep if c not in ("peer_qty_median",)] + ["peer_qty_median", "reasons"]
    Q[queue_cols].to_parquet(QUEUE_PATH, index=False)
    print(f"review queue: {len(Q):,} rows with reasons ({time.time() - t0:.0f}s)")

    # ----------------------------------------------------------------- aggregates
    AGG_DIR.mkdir(exist_ok=True)
    n_excluded = int((~eligible).sum())
    F["is_priority"] = (F["tier"] == "priority").astype(float)
    F = F[eligible]                              # aggregates describe the ranked records
    for col, min_n in (("year", 0), ("class", 500), ("order_name", 500), ("taxon_display", 500),
                       ("exporter", 2000), ("importer", 2000), ("product_group", 500),
                       ("purpose", 500), ("source", 500), ("appendix", 0), ("cluster", 0)):
        g = F.groupby(col).agg(records=("is_priority", "size"),
                               priority_share=("is_priority", "mean"),
                               mean_risk_percentile=("risk_percentile", "mean"))
        g = g[g["records"] >= min_n].sort_values("priority_share", ascending=False)
        g.round(5).to_csv(AGG_DIR / f"by_{col}.csv")
    routes = (F.groupby(["exporter", "importer"])
              .agg(records=("is_priority", "size"), priority_share=("is_priority", "mean"))
              .query("records >= 1000").sort_values("priority_share", ascending=False))
    routes.head(300).round(5).to_csv(AGG_DIR / "by_route.csv")
    summary = {"deployed_stack": name, "records_scored": int(len(F)) + n_excluded,
               "records_ranked": int(len(F)),
               "excluded_known_enforcement": n_excluded,
               "years": [int(F["year"].min()), int(F["year"].max())],
               "priority_records": int((F["tier"] == "priority").sum()),
               "review_records": int((F["tier"] == "review").sum()),
               "kmeans_k_on_cites": int(stack.kmeans.k),
               "kmeans_silhouette_on_cites": stack.kmeans.silhouette,
               "kmeans_stability_ari_on_cites": stack.kmeans.stability_ari,
               # score distribution, so a single new record can be placed on the same ranking
               "hybrid_score_quantiles": np.quantile(
                   F["score_hybrid"], np.linspace(0, 1, 1001)).round(8).tolist()}
    (config.OUTPUTS_DIR / "cites_scoring_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
