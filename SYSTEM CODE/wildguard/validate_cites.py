"""Secondary evaluation on CITES - plausibility and directional consistency, not accuracy.

CITES has no enforcement labels (the CITES Illegal Trade Database is access-restricted), so
no precision/recall claim is made here. Three checks instead:

1. Consistency with independent seizure statistics (UNODC World Wildlife Crime Report 2024,
   Fig. 2.1: share of 2015-2021 seizure records by species group). For each animal group we
   compute how over-represented it is in seizures relative to its share of legal CITES
   trade (seizure-to-trade ratio), and test with Spearman's rank correlation whether
   WildGuard flags groups more often where that ratio is higher. The same test is run on
   two baselines (listing rule, unsupervised-only) for comparison.

2. Confiscated-specimen proxy. CITES source code "I" marks confiscated/seized specimens.
   A transfer model retrained *without any source-derived feature* is asked to rank CITES
   records; ROC-AUC for source == "I" vs the rest measures whether patterns learned from
   US enforcement outcomes point toward independently-marked enforcement material.

3. Top-N qualitative review: the highest-ranked records with their reasons, for a human
   plausibility read (written up in the report).

Run:  python -m wildguard.validate_cites      (after score_cites.py)
"""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from . import cites_features, config
from .features import SHARED_FEATURES, to_model_matrix
from .models import train_xgb
from .train import listing_rule_score, load_lemis

# UNODC WWCR 2024, Fig. 2.1 - % of seizure records 2015-2021 (animal groups only)
UNODC_SEIZURE_SHARE = {
    "Corals": 16, "Crocodilians": 9, "Elephants": 6, "Bivalve molluscs": 6, "Carnivores": 5,
    "Parrots and cockatoos": 4, "Turtles and tortoises": 4, "Snakes": 4,
    "Even-toed ungulates": 2, "Sturgeons and paddlefishes": 2, "Pangolins": 2,
    "Sea snails": 2, "Lizards": 2, "Birds of prey": 1,
}


def unodc_group(df):
    o, f, c = df["order_name"], df["family"], df["class"]
    conds = [
        c.isin(["Anthozoa", "Hydrozoa"]),
        o.eq("Crocodylia"),
        f.eq("Elephantidae") | o.eq("Proboscidea"),
        c.eq("Bivalvia"),
        o.eq("Carnivora"),
        o.eq("Psittaciformes"),
        o.eq("Testudines"),
        o.eq("Serpentes"),
        o.eq("Artiodactyla"),
        o.eq("Acipenseriformes"),
        f.eq("Manidae") | o.eq("Pholidota"),
        c.eq("Gastropoda"),
        o.eq("Sauria"),
        o.isin(["Falconiformes", "Accipitriformes", "Strigiformes"]),
    ]
    return np.select(conds, list(UNODC_SEIZURE_SHARE), default="(other)")


def check_unodc(S):
    s = S[S["year"].between(2015, 2021) & (S["tier"] != "excluded")].copy()
    s["group"] = unodc_group(s)
    s = s[s["group"] != "(other)"]
    s["listing_rule"] = listing_rule_score(s)
    s["is_priority"] = (s["tier"] == "priority").astype(float)
    # baselines expressed the same way: share of records in their own top 1% per year
    for col in ("listing_rule", "score_unsup_mean", "score_supervised"):
        pct = s.groupby("year")[col].rank(pct=True, method="first")
        s[f"top1_{col}"] = (pct > 0.99).astype(float)
    g = s.groupby("group").agg(cites_records=("year", "size"),
                               wildguard_priority_share=("is_priority", "mean"),
                               supervised_only_top1=("top1_score_supervised", "mean"),
                               unsupervised_only_top1=("top1_score_unsup_mean", "mean"),
                               listing_rule_top1=("top1_listing_rule", "mean"))
    g["unodc_seizure_share"] = pd.Series(UNODC_SEIZURE_SHARE) / sum(UNODC_SEIZURE_SHARE.values())
    g["cites_trade_share"] = g["cites_records"] / g["cites_records"].sum()
    g["seizure_to_trade_ratio"] = g["unodc_seizure_share"] / g["cites_trade_share"]
    g = g.sort_values("seizure_to_trade_ratio", ascending=False)
    tests = {}
    for col in ("wildguard_priority_share", "supervised_only_top1", "unsupervised_only_top1",
                "listing_rule_top1"):
        rho, p = spearmanr(g["seizure_to_trade_ratio"], g[col])
        tests[col] = {"spearman_rho": round(float(rho), 3), "p_value": round(float(p), 4)}
    g.round(5).to_csv(config.OUTPUTS_DIR / "cites_unodc_consistency.csv")
    return {"n_groups": int(len(g)), "spearman_vs_seizure_to_trade_ratio": tests}


def check_confiscated_proxy():
    """Retrain the transfer model without source-derived features, then test whether it
    ranks CITES source='I' (confiscated/seized) records above the rest."""
    no_source = [f for f in SHARED_FEATURES
                 if f not in ("source", "is_wild", "source_share_in_genus", "combo_rarity")]
    fb = joblib.load(config.MODELS_DIR / "feature_builder_lemis.joblib")
    df = load_lemis()
    tr, va = df["year"].isin(config.TRAIN_YEARS), df["year"].isin(config.VALID_YEARS)
    Ftr, Fva = fb.transform(df[tr]), fb.transform(df[va])
    Xtr, cats = to_model_matrix(Ftr, no_source)
    Xva, _ = to_model_matrix(Fva, no_source, cats)
    model = train_xgb(Xtr, Ftr["label"].values, Xva, Fva["label"].values)
    del df, Ftr, Fva, Xtr, Xva

    F = cites_features.load_features(columns=no_source + ["source"])
    p = np.concatenate([model.predict_proba(to_model_matrix(F.iloc[i:i + 1_000_000], no_source,
                                                            cats)[0])[:, 1]
                        for i in range(0, len(F), 1_000_000)])
    y = (F["source"] == "I").to_numpy()
    rng = np.random.default_rng(config.RANDOM_STATE)
    return {"n_source_I": int(y.sum()), "share_source_I": float(y.mean()),
            "auc_no_source_transfer_model": round(float(roc_auc_score(y, p)), 4),
            "auc_random": round(float(roc_auc_score(y, rng.random(len(y)))), 4),
            "share_of_source_I_in_top_5pct": float(
                y[p >= np.quantile(p, 0.95)].sum() / max(y.sum(), 1))}


def top_n(n=25):
    q = pd.read_parquet(config.SAMPLES_DIR / "cites_review_queue.parquet")
    q = q[q["tier"] != "excluded"].sort_values("score_hybrid", ascending=False)
    q = q.drop_duplicates(["taxon_display", "product_code", "exporter", "importer", "purpose"])
    cols = ["year", "taxon_display", "product_code", "quantity_std", "unit_group", "purpose",
            "source", "exporter", "importer", "appendix", "risk_percentile", "reasons"]
    q.head(n)[cols].to_csv(config.OUTPUTS_DIR / "cites_top25_review.csv", index=False)


def make_figure():
    """Seizure-to-trade ratio (UNODC vs CITES) against WildGuard's priority share per group."""
    from . import plots
    import matplotlib.pyplot as plt
    plots.apply_style()
    g = pd.read_csv(config.OUTPUTS_DIR / "cites_unodc_consistency.csv")
    rho = json.loads((config.OUTPUTS_DIR / "cites_validation.json").read_text())[
        "unodc_consistency"]["spearman_vs_seizure_to_trade_ratio"]["wildguard_priority_share"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_axisbelow(True)
    ax.scatter(g["seizure_to_trade_ratio"], g["wildguard_priority_share"] * 100, s=42,
               color=plots.SERIES[0], edgecolor=plots.SURFACE, linewidth=2, zorder=3)
    # hand-placed label offsets (points) for the crowded lower-left region
    offsets = {"Crocodilians": (-6, -11, "right"), "Snakes": (6, -9, "left"),
               "Lizards": (-6, -9, "right"), "Birds of prey": (-6, 4, "right"),
               "Sturgeons and paddlefishes": (6, -9, "left"), "Corals": (6, 3, "left"),
               "Parrots and cockatoos": (6, -9, "left"), "Carnivores": (6, -7, "left"),
               "Even-toed ungulates": (-6, 4, "right")}
    for _, r in g.iterrows():
        dx, dy, ha = offsets.get(r["group"], (6, 4, "left"))
        ax.annotate(r["group"], (r["seizure_to_trade_ratio"], r["wildguard_priority_share"] * 100),
                    xytext=(dx, dy), textcoords="offset points", fontsize=8, ha=ha,
                    color=plots.INK_2)
    ax.axhline(1, color=plots.BASELINE, lw=1.2)
    ax.text(2500, 1.08, "uniform flagging (1%)", fontsize=8, color=plots.MUTED, ha="right",
            va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.12, 3000)
    ax.set_ylim(0.1, 60)
    ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50], ["0.1", "0.2", "0.5", "1", "2", "5", "10",
                                                        "20", "50"])
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
    ax.minorticks_off()
    ax.set_xlabel("UNODC seizure share ÷ CITES legal-trade share (log scale)")
    ax.set_ylabel("Records in WildGuard priority tier (%, log scale)")
    ax.set_title(f"Flags track independent seizure statistics (Spearman ρ = {rho['spearman_rho']}, "
                 f"p = {rho['p_value']})")
    plots.save(fig, config.FIGURES_DIR / "unodc_consistency.png")


def main():
    S = pd.read_parquet(config.PROCESSED_DIR / "cites_scored.parquet")
    out = {"unodc_consistency": check_unodc(S)}
    del S
    top_n()
    out["confiscated_proxy"] = check_confiscated_proxy()
    (config.OUTPUTS_DIR / "cites_validation.json").write_text(json.dumps(out, indent=2))
    make_figure()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import sys
    make_figure() if "--figure-only" in sys.argv else main()
