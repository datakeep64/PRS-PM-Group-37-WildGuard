"""Secondary evaluation on CITES - consistency with independent evidence, not accuracy.

CITES has no enforcement labels (the CITES Illegal Trade Database is access-restricted), so no
precision or recall claim is made here. The three checks and what counts as a pass were fixed
before the rebuild (DESIGN_QUESTIONS.md C7). A pass means the ranking is consistent with
independent evidence, not that it is accurate.

1. UNODC consistency. For the 14 animal groups of the World Wildlife Crime Report 2024 (Fig. 2.1),
   how over-represented each group is in seizures relative to its share of legal trade
   (exporter-reported CITES records 2015-2021), against WildGuard's priority share for the group.
   Pass: Spearman rho positive with a one-sided permutation p-value below 0.05, and higher than
   the rho of the "most endangered first" listing rule.
2. Confiscated-specimen proxy. The ensemble retrained without any source-derived feature ranks
   the rule D records, including the source I records the ranking sets aside. Pass: the lower
   end of the 95% bootstrap interval of ROC-AUC for source I is above 0.5.
3. Review of the top-ranked records: the 50 highest-ranked priority records, one per taxon,
   put into C7's three categories by a stated rule (classify). Pass: at least half recognised
   risk patterns, at most a fifth with no apparent reason. (C7 planned two human reviewers;
   the rule replaced them, so no agreement between reviewers is reported.)

Also reported, without a pass criterion: the share of priority flags on marine-mammal products
traded between other countries (S6.3).

Run:  python -m wildguard.validate_cites      (after score_cites.py)
"""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from . import cites_features, config
from .features import SHARED_FEATURES
from .models import Ensemble
from .train import listing_rule_score, load_lemis

SOURCE_DERIVED = ("source", "is_wild", "source_share_in_genus", "combo_rarity")
MIN_RECORDS = 500          # groups with fewer records in the window are flagged as unstable

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


def check_unodc(S, n_perm=10_000):
    s = S[S["year"].between(2015, 2021) & (S["tier"] != "excluded") & (S["reporter_type"] == "E")].copy()
    s["group"] = unodc_group(s)
    s = s[s["group"] != "(other)"]
    s["is_priority"] = (s["tier"] == "priority").astype(float)
    pct = pd.Series(listing_rule_score(s)).groupby(s["year"].to_numpy()).rank(pct=True, method="first")
    s["listing_rule_top1"] = (pct.to_numpy() > 0.99).astype(float)
    g = s.groupby("group").agg(cites_records=("year", "size"),
                               wildguard_priority_share=("is_priority", "mean"),
                               listing_rule_top1=("listing_rule_top1", "mean"))
    g["unodc_seizure_share"] = pd.Series(UNODC_SEIZURE_SHARE) / sum(UNODC_SEIZURE_SHARE.values())
    g["cites_trade_share"] = g["cites_records"] / g["cites_records"].sum()
    g["seizure_to_trade_ratio"] = g["unodc_seizure_share"] / g["cites_trade_share"]
    g["unstable_fewer_than_500_records"] = g["cites_records"] < MIN_RECORDS
    g = g.sort_values("seizure_to_trade_ratio", ascending=False)
    rho_w = float(spearmanr(g["seizure_to_trade_ratio"], g["wildguard_priority_share"])[0])
    rho_l = float(spearmanr(g["seizure_to_trade_ratio"], g["listing_rule_top1"])[0])
    rng = np.random.default_rng(config.RANDOM_STATE)
    v = g["wildguard_priority_share"].to_numpy()
    perm = np.array([spearmanr(g["seizure_to_trade_ratio"], rng.permutation(v))[0] for _ in range(n_perm)])
    p = float((np.sum(perm >= rho_w) + 1) / (n_perm + 1))
    g.round(5).to_csv(config.OUTPUTS_DIR / "cites_unodc_consistency.csv")
    return {"n_groups": int(len(g)), "spearman_rho_wildguard": round(rho_w, 3),
            "permutation_p_one_sided": round(p, 4), "spearman_rho_listing_rule": round(rho_l, 3),
            "unstable_groups": g.index[g["unstable_fewer_than_500_records"]].tolist(),
            "pass": bool(rho_w > 0 and p < 0.05 and rho_w > rho_l)}


def check_confiscated_proxy(n_boot=1000):
    feats = [f for f in SHARED_FEATURES if f not in SOURCE_DERIVED]
    fb = joblib.load(config.MODELS_DIR / "feature_builder_lemis.joblib")    # the deployed norms
    df = load_lemis()
    df = df[df["year"].isin(config.DEPLOY_YEARS)]
    F = fb.transform(df)
    model = Ensemble(feats).fit(F, F["label"].to_numpy())
    del df, F
    C = cites_features.load_features(columns=list(dict.fromkeys(SHARED_FEATURES + ["source", "rule_d", "year"])))
    C = C[C["rule_d"]].reset_index(drop=True)
    s = model.score(C, groups=C["year"].to_numpy())
    y = (C["source"] == "I").to_numpy()
    rng = np.random.default_rng(config.RANDOM_STATE)
    idx_pos, idx_neg = np.flatnonzero(y), np.flatnonzero(~y)
    boot = []
    for _ in range(n_boot):
        i = np.r_[rng.choice(idx_pos, len(idx_pos)), rng.choice(idx_neg, min(len(idx_neg), 500_000))]
        boot.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    auc = float(roc_auc_score(y, s))
    return {"features_removed": list(SOURCE_DERIVED), "n_source_I": int(y.sum()),
            "roc_auc": round(auc, 4), "low_95": round(float(lo), 4), "high_95": round(float(hi), 4),
            "share_of_source_I_in_top_5pct": round(float(y[s >= np.quantile(s, 0.95)].sum() / max(y.sum(), 1)), 4),
            "pass": bool(lo > 0.5)}


# Check 3's categories (DESIGN_QUESTIONS C7), applied as a stated rule rather than by two reviewers
WORKED_OR_HIGH_VALUE = {"ivory", "horn", "bone", "teeth_claws", "carvings_jewellery", "extracts_medicine"}
UNCLEAR_SOURCE = {"O": "pre-Convention", "U": "unknown", "unknown": "not declared"}


def classify(r):
    """(category, the part of the rule that applied) for one top-ranked record. Every record is a
    CITES record, so every taxon is listed."""
    if r["warning"]:
        return "b", "marine-mammal product traded between two other countries (US law)"
    if r["product_group"] in WORKED_OR_HIGH_VALUE:
        return "a", f"worked or high-value part ({r['product_group'].replace('_', ' ')})"
    if r["purpose"] == "P" and r["appendix"] == "I":
        return "a", "personal-purpose specimen of an Appendix I taxon"
    if r["source"] in UNCLEAR_SOURCE:
        return "a", f"source claimed as {UNCLEAR_SOURCE[r['source']]}"
    if r["origin"] == "XX":
        return "a", "unknown origin"
    return "c", "none of the recognised patterns"


def check_top_records(n=50):
    """Check 3: the n highest-ranked priority records, one per taxon, classified by the rule.
    Pass: at least half in (a), at most a fifth in (c); (b) is the jurisdiction effect."""
    q = pd.read_parquet(config.SAMPLES_DIR / "cites_review_queue.parquet")
    q = q[q["tier"] == "priority"].sort_values("score", ascending=False).drop_duplicates("taxon_display")
    out = q.head(n).reset_index(drop=True)
    out[["category", "rule"]] = [classify(r) for _, r in out.iterrows()]
    cols = ["year", "taxon_display", "product_code", "product_group", "quantity_std", "unit_group",
            "purpose", "source", "origin", "exporter", "importer", "appendix", "score", "reasons",
            "peer_context", "warning", "category", "rule"]
    out[cols].to_csv(config.OUTPUTS_DIR / "cites_top50_review.csv", index=False)
    share = out["category"].value_counts(normalize=True).reindex(["a", "b", "c"]).fillna(0)
    return {"records": int(len(out)), "file": "cites_top50_review.csv",
            "classified_by": "the stated rule in validate_cites.classify, not by human reviewers",
            "share_a_recognised_pattern": round(float(share["a"]), 3),
            "share_b_us_law_abroad": round(float(share["b"]), 3),
            "share_c_no_apparent_reason": round(float(share["c"]), 3),
            "pass": bool(share["a"] >= 0.5 and share["c"] <= 0.2)}


def make_figure():
    """Seizure-to-trade ratio (UNODC vs CITES) against WildGuard's priority share per group."""
    from . import plots
    import matplotlib.pyplot as plt
    plots.apply_style()
    g = pd.read_csv(config.OUTPUTS_DIR / "cites_unodc_consistency.csv")
    u = json.loads((config.OUTPUTS_DIR / "cites_validation.json").read_text())["unodc_consistency"]
    rho = {"spearman_rho": u["spearman_rho_wildguard"], "p_value": u["permutation_p_one_sided"]}
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
    S = pd.read_parquet(config.PROCESSED_DIR / "cites_scored.parquet",
                        columns=["year", "tier", "reporter_type", "class", "order_name", "family",
                                 "appendix", "source"])
    out = {"unodc_consistency": check_unodc(S)}
    del S
    out["top_records"] = check_top_records()
    out["confiscated_proxy"] = check_confiscated_proxy()
    summary = json.loads((config.OUTPUTS_DIR / "cites_scoring_summary.json").read_text())
    out["marine_mammal_non_us_importer"] = {
        "share_of_priority_records": summary["priority_share_marine_mammal_non_us_importer"],
        "share_of_ranked_records": summary["ranked_share_marine_mammal_non_us_importer"]}
    (config.OUTPUTS_DIR / "cites_validation.json").write_text(json.dumps(out, indent=2))
    make_figure()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import sys
    make_figure() if "--figure-only" in sys.argv else main()
