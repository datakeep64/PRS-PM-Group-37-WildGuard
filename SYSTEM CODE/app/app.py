"""WildGuard - wildlife-trade risk screening.

    streamlit run "SYSTEM CODE/app/app.py"

Each page answers one question. It leads with a plain answer and a comparison, and keeps the
measurement detail one click below, in expanders.

Pages
    Overview        what the system does, its headline results, what a flag means, the data
    Review queue    CITES export permits ranked within each year, with a flag card per permit
    Check a record  rank one trade record, or a US import declaration, and see why
    Does it work?   the one test on held-out LEMIS years, against real enforcement outcomes
    What it flags   where flags concentrate in CITES, and three checks against outside evidence
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

# Streamlit Community Cloud pulls each push into the running app without restarting Python, and
# only this file is re-read on every run: the wildguard modules already imported, and everything
# held by st.cache_data / st.cache_resource, would keep serving the previous version (an
# AttributeError on a new config constant, the previous dataset version on the Overview).
# Whenever a code, model, output or sample file has changed since the last run, both are dropped.
WATCHED = [CODE_DIR / d for d in ("wildguard", "models", "outputs")] + \
          [CODE_DIR.parent / "DATA" / d for d in ("samples", "reference")]


def fingerprint():
    return max(p.stat().st_mtime_ns for d in WATCHED for p in d.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts)


@st.cache_resource(show_spinner=False)
def last_run():
    return {}


stamp = fingerprint()
if last_run().get("stamp") != stamp:
    for name in [m for m in sys.modules if m.split(".")[0] == "wildguard"]:
        del sys.modules[name]
    st.cache_data.clear()
    st.cache_resource.clear()
    last_run()["stamp"] = stamp

from wildguard import config  # noqa: E402
from wildguard.explain import (COUNTRY, PORT, PURPOSE, SOURCE, explain_rows,  # noqa: E402
                               peer_context, us_law_warning)
from wildguard.models import SCALE_GRID, PeerDeviationScorer  # noqa: E402
from wildguard.schema_alignment import region  # noqa: E402

# "ranger station" palette, matching .streamlit/config.toml: forest green, trail-blaze orange,
# slate, stone; khaki page, olive-drab sidebar, cream lettering
GREEN, BLAZE, SLATE, GREY = "#3E5C2E", "#B5652B", "#5B7F95", "#8C8270"
KHAKI, OLIVE, CREAM = "#E3DAC2", "#4A5A3A", "#EFE6CC"
PASS, FAIL = "#2E7D32", "#C62828"
st.set_page_config(page_title="WildGuard", page_icon="🐘", layout="wide",
                   initial_sidebar_state="expanded")
# slab-serif headings like routed park signs; olive-drab sidebar with cream text
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@500;700&display=swap');
h1, h2, h3, h4, h5, [data-testid="stMetricValue"], .wg-sign {{
    font-family: 'Roboto Slab', Georgia, serif !important; }}
[data-testid="stSidebar"] {{ background-color: {OLIVE}; }}
[data-testid="stSidebar"] * {{ color: {CREAM} !important; }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ opacity: 0.75; }}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{
    background-color: rgba(239, 230, 204, 0.45) !important; }}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child > div {{
    background-color: {OLIVE} !important; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) > div:first-child {{
    background-color: {BLAZE} !important; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) > div:first-child > div {{
    background-color: {CREAM} !important; }}
.wg-step {{ background: rgba(255, 255, 255, 0.35); border-radius: 10px; padding: 10px 12px;
    min-height: 12.5rem; font-size: 0.88rem; line-height: 1.35; }}
.wg-step b {{ font-family: 'Roboto Slab', Georgia, serif; }}
</style>""", unsafe_allow_html=True)

TIER_STYLE = {"priority": ("#d03b3b", "Priority review"), "review": ("#fab219", "Review"),
              "routine": ("#0ca30c", "Routine")}
TIER_CHOICES = {"priority": "Priority · top 1% of each year",
                "review": "Review · next 4% (sample)", "routine": "Routine · the rest (sample)"}
PRODUCT_LABELS = {"live_eggs": "live eggs", "skin_pieces": "skin pieces",
                  "leather_products": "leather products", "coral_raw": "raw coral",
                  "teeth_claws": "teeth and claws", "extracts_medicine": "extracts and medicine",
                  "carvings_jewellery": "carvings and jewellery", "body_parts": "body parts"}
PRODUCT_GROUPS = ["live", "live_eggs", "bodies", "trophies", "skins", "skin_pieces",
                  "leather_products", "garments", "cloth", "specimens", "meat", "caviar", "ivory",
                  "horn", "bone", "skulls", "teeth_claws", "coral_raw", "shells", "scales",
                  "feathers", "eggs", "hair", "fins", "extracts_medicine", "carvings_jewellery",
                  "body_parts", "other"]
UNITS = ["count", "kg", "litre", "m2", "m3", "m"]
APPENDIX_LABELS = {"I": "Appendix I", "II": "Appendix II", "III": "Appendix III",
                   "not_listed": "not listed"}
CLASS_ICON = {"Reptilia": "🦎", "Mammalia": "🐘", "Aves": "🦜", "Anthozoa": "🪸",
              "Actinopteri": "🐟", "Amphibia": "🐸", "Bivalvia": "🐚", "Gastropoda": "🐌",
              "Insecta": "🦋", "Elasmobranchii": "🦈", "Hydrozoa": "🪼", "Arachnida": "🕷️",
              "Holothuroidea": "🥒", "Hirudinoidea": "🪱"}
CLASSES = ["Reptilia", "Mammalia", "Aves", "Anthozoa", "Actinopteri", "Amphibia", "Bivalvia",
           "Gastropoda", "Insecta", "Elasmobranchii", "Hydrozoa", "Arachnida"]
# the models and baselines of the one test, in the order they are shown
MODEL_NAMES = {"ensemble": "WildGuard (XGBoost + random forest)",
               "xgboost": "XGBoost alone", "random_forest": "Random forest alone",
               "logistic_regression": "Logistic regression",
               "full_ensemble": "WildGuard + US-only fields (value, port, month)",
               "baseline:listing_rule": "Listing rule (most endangered, wild first)",
               "baseline:random": "Random inspection"}
SUBSETS = {"listed": "Species listed in CITES (closest to CITES trade)",
           "all": "All imports", "commercial": "Commercial imports",
           "listed_commercial": "Commercial imports of listed species"}
# column help for the CITES preview (LEMIS's comes from its own codebook, DATA/reference)
CITES_FIELDS = {
    "Id": "Unique record identifier, unchanged across releases",
    "Year": "Year of the trade record",
    "Appendix": "CITES Appendix of the taxon (I, II, III; N = not listed)",
    "Taxon": "Scientific name", "Class": "Taxonomic class (blank for plants)",
    "Order": "Taxonomic order", "Family": "Taxonomic family", "Genus": "Genus",
    "Term": "Specimen traded (e.g. live, skins, meat)", "Quantity": "Quantity traded, in Unit",
    "Unit": "Unit of the quantity (blank = number of specimens)",
    "Importer": "Importing country (ISO 3166 alpha-2)",
    "Exporter": "Exporting or re-exporting country (ISO 3166 alpha-2)",
    "Origin": "Country of origin, for re-exports (ISO 3166 alpha-2)",
    "Purpose": "Purpose code (e.g. T commercial, P personal, Z zoo)",
    "Source": "Source code (e.g. W wild, C bred in captivity, I confiscated)",
    "Reporter.type": "Who reported the record: E exporter, I importer",
    "Import.permit.RandomID": "Anonymised ID replacing the confidential import permit number",
    "Export.permit.RandomID": "Anonymised ID replacing the confidential export or re-export "
                              "permit number (same permit, same ID)",
    "Origin.permit.RandomID": "Anonymised ID replacing the confidential permit from the "
                              "country of origin",
    "Ifs.permit.RandomID": "Anonymised ID replacing the introduction-from-the-sea certificate "
                           "number"}


# ----------------------------------------------------------------------------- loaders
@st.cache_data
def load_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


QUEUE_COLS = ["year", "class", "order_name", "family", "genus", "taxon_display", "product_code",
              "unit_group", "quantity_std", "purpose", "source", "origin", "exporter",
              "importer", "appendix", "reporter_type", "export_permit", "consignment",
              "consignment_score", "score", "score_xgboost", "score_random_forest",
              "peer_deviation", "tier", "reasons", "peer_context", "warning"]
TEST_MODELS = list(MODEL_NAMES)


def _compact(df, keep_text=()):
    """Repeated strings (codes, tiers, taxa) as categoricals - keeps the hosted app small."""
    for c in df.columns:
        if df[c].dtype == object and c not in keep_text:
            df[c] = df[c].astype("category")
    return df


# The large tables are read-only, so they are cached once as shared resources rather than
# copied per session (st.cache_data returns a fresh copy on every call).
@st.cache_resource
def load_queue():
    p = config.SAMPLES_DIR / "cites_review_queue.parquet"
    return _compact(pd.read_parquet(p, columns=QUEUE_COLS), keep_text=("reasons",)) \
        if p.exists() else None


@st.cache_resource
def load_permits():
    """The review queue as permits: one row per export permit within a year (a record without a
    permit stands alone), described by its highest-scoring record, which sets the permit's
    score. Returns (permits, each queue record's permit number)."""
    q, scale = load_queue(), load_json(config.OUTPUTS_DIR / "cites_record_scale.json")
    if q is None or scale is None:
        return None, None
    gid = q.groupby(["year", "consignment"], observed=True, sort=False).ngroup().to_numpy()
    top_row = pd.Series(q["score"].to_numpy()).groupby(gid).idxmax().to_numpy()
    P = q.iloc[top_row].reset_index(drop=True)
    P["gid"], P["row"], P["records"] = np.arange(len(P)), top_row, np.bincount(gid)
    P["taxa"] = pd.Series(q["taxon_display"].cat.codes.to_numpy()).groupby(gid).nunique().to_numpy()
    P["both_reports"] = pd.Series(q["reporter_type"].cat.codes.to_numpy()).groupby(gid).nunique().to_numpy() > 1
    P["any_warning"] = pd.Series((q["warning"] != "").to_numpy()).groupby(gid).any().to_numpy()
    # rank within the year: exact for priority permits (all of them are in the queue), placed
    # on the year's permit-score scale for the sampled lower tiers, and kept inside its tier
    n = P["year"].map(lambda y: scale[str(y)]["permits"]).to_numpy()
    est = np.ones(len(P))
    for y, idx in P.groupby("year").indices.items():
        s = scale[str(y)]
        est[idx] = 1 - np.interp(P["consignment_score"].to_numpy()[idx], s["permit_score"], SCALE_GRID)
    exact = P[P["tier"] == "priority"].groupby("year")["consignment_score"].rank(
        ascending=False, method="first")
    rank = np.maximum(np.round(est * n), 1)
    lo = np.where(P["tier"] == "review", np.floor(0.01 * n) + 1, np.floor(0.05 * n) + 1)
    hi = np.where(P["tier"] == "review", np.floor(0.05 * n), n)
    rank = np.clip(rank, lo, hi)
    rank[exact.index.to_numpy()] = exact.to_numpy()
    P["rank"], P["permits_in_year"] = rank.astype(int), n
    P["top"] = P["rank"] / n
    P["taxon_label"] = [f"{CLASS_ICON.get(c, '🌿')} {t}" + (f" +{k - 1}" if k > 1 else "")
                        for c, t, k in zip(P["class"], P["taxon_display"], P["taxa"])]
    P["route"] = P["exporter"].astype(str) + " → " + P["importer"].astype(str)
    return P, gid


@st.cache_resource
def load_test_shipments():
    """The one test's line items (LEMIS 2013-2014) as shipments, for each held-out set: a
    shipment's score is its highest line score, and it is a case if any line is (C2)."""
    p = config.OUTPUTS_DIR / "lemis_test_scores.parquet"
    if not p.exists():
        return None
    t = pq.read_table(p, columns=["shipment_id", "sp_listed", "purpose", "label"] + TEST_MODELS,
                      read_dictionary=["shipment_id", "purpose"]).to_pandas()
    masks = {"all": np.ones(len(t), bool), "listed": t["sp_listed"].to_numpy(),
             "commercial": (t["purpose"] == "T").to_numpy()}
    masks["listed_commercial"] = masks["listed"] & masks["commercial"]
    return {k: t[m].groupby("shipment_id", observed=True)[["label"] + TEST_MODELS].max()
            .reset_index(drop=True) for k, m in masks.items()}


@st.cache_data
def load_csv(name):
    p = config.OUTPUTS_DIR / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_head(name):
    """First rows of a source dataset as written by wildguard.datasets - all text, verbatim."""
    p = config.SAMPLES_DIR / f"{name}_head.csv"
    return pd.read_csv(p, dtype=str, keep_default_na=False) if p.exists() else None


@st.cache_data
def lemis_fields():
    p = config.DATA_DIR / "reference" / "lemis_metadata.csv"
    return dict(pd.read_csv(p).values) if p.exists() else {}


# The two ensembles hold 130-240 MB each once loaded, so only the one in use is kept.
@st.cache_resource(max_entries=1)
def load_scorer(kind):
    """(ensemble, feature builder): the CITES model, or the full model for US declarations."""
    ens, fb = {"cites": ("cites_ensemble.joblib", "feature_builder_cites.joblib"),
               "full": ("full_ensemble.joblib", "feature_builder_lemis.joblib")}[kind]
    paths = [config.MODELS_DIR / ens, config.MODELS_DIR / fb]
    return tuple(joblib.load(p) for p in paths) if all(p.exists() for p in paths) else None


@st.cache_data
def genus_taxonomy():
    """Order and family of each genus in the queue, for the US-law warning on a checked record."""
    q = load_queue()
    if q is None:
        return {}
    g = q[["genus", "order_name", "family"]].astype(str).drop_duplicates("genus")
    return {r.genus: (r.order_name, r.family) for r in g.itertuples()}


# ----------------------------------------------------------------------------- helpers
def plotly_layout(fig, height=360, **kw):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), **kw)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(137,135,129,0.2)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(137,135,129,0.2)", zeroline=False)
    return fig


def tier_badge(tier):
    color, label = TIER_STYLE[tier]
    text = "#0b0b0b" if tier == "review" else "#ffffff"
    return (f"<span style='background:{color};color:{text};padding:3px 10px;border-radius:12px;"
            f"font-weight:600;font-size:0.9rem'>● {label}</span>")


def check_badge(passed):
    color, label = (PASS, "PASS") if passed else (FAIL, "FAIL")
    return (f"<span style='background:{color};color:#fff;padding:2px 9px;border-radius:10px;"
            f"font-weight:700;font-size:0.8rem;letter-spacing:0.05em'>{label}</span>")


def top_text(share):
    """'top 0.37%': the share of the population ranked at or above, rounded up so it stays true."""
    pct = share * 100
    if pct < 1:
        return f"top {max(np.ceil(pct * 100) / 100, 0.01):.2f}%"
    if pct < 10:
        return f"top {np.ceil(pct * 10) / 10:.1f}%"
    return f"top {min(np.ceil(pct), 100):.0f}%"


def cname(code):
    return f"{COUNTRY.get(code, code)} ({code})"


def with_icon(cls):
    return f"{CLASS_ICON.get(cls, '🌿')} {cls}" if isinstance(cls, str) else "🌿"


def product_label(group):
    return PRODUCT_LABELS.get(group, str(group).replace("_", " "))


def prob(p):
    """Probability for display; avoids printing 100.0% for 0.9996."""
    return ">99.9%" if p > 0.999 else "<0.1%" if p < 0.001 else f"{p:.1%}"


def need(obj, what):
    if obj is None:
        st.warning(f"{what} not found - run the pipeline first: `bash \"SYSTEM CODE/run_pipeline.sh\"`")
        st.stop()


def go_to(page):
    st.session_state["nav"] = page


# ------------------------------------------------------------------------- flag card
def flag_card(r, rank_line, reasons, peer, warning, flagged):
    """The flag card shared by the review queue and Check a record: what the record is, why it
    ranks where it does, how unusual it is for its peers, and any warning."""
    st.markdown(f"{tier_badge(r['tier'])} &nbsp; <b>{rank_line}</b>", unsafe_allow_html=True)
    x = float(r["quantity_std"])
    qty = f"{x:,.0f}" if x.is_integer() or x >= 1000 else f"{x:,.4g}"
    amount = f"{qty} {r['product']}" if r["unit_group"] == "count" else \
        f"{qty} {r['unit_group']} of {r['product']}"
    where = f"{cname(r['exporter'])} → {cname(r['importer'])}"
    if r["origin"] != r["exporter"]:
        where += f" · origin {cname(r['origin'])}"
    tail = [f"{r['year']}"] if r.get("year") else []
    if r.get("reporter_type") in ("E", "I"):
        tail.append(f"reported by the {'importer' if r['reporter_type'] == 'I' else 'exporter'}")
    st.markdown(
        f"#### {CLASS_ICON.get(r['class'], '🌿')} {r['taxon_display'] or 'unnamed taxon'}\n"
        f"**{amount}** · source **{SOURCE.get(r['source'], r['source'])}** · purpose "
        f"**{PURPOSE.get(r['purpose'], r['purpose'])}** · "
        f"{APPENDIX_LABELS.get(r['appendix'], 'CITES ' + str(r['appendix']))}  \n"
        f"{where}" + (" · " + " · ".join(tail) if tail else ""))

    st.markdown("**Why flagged**" if flagged else "**What raised its score**")
    for reason in reasons or ["No single field raised its score."]:
        st.markdown(f"- {reason}")
    st.caption("What pushed the score up most, according to XGBoost, one of the two models. "
               "Shares and peer norms are those of US imports, 2000–2014, which the models "
               "learned from.")

    st.markdown("**How unusual for its peers**")
    if peer:
        for part in str(peer).split(", "):
            st.markdown(f"- {part}")
    else:
        st.markdown("Nothing stands out against its peers: the score comes from the patterns the "
                    "models learned from US enforcement outcomes.")
    st.caption("Peers: US imports of the same genus, product and unit. This is context for the "
               "reviewer; it is not part of the score.")
    if warning:
        st.warning(f"**Warning.** {warning}", icon="⚠️")


def member_lines(score, pct_x, pct_rf, p_x, peer_dev, population):
    return [f"Score {score:.4f}: the average of the two models' percentiles among {population}.",
            f"XGBoost: percentile {pct_x:.2%}. Its probability for a US import like this: "
            f"{prob(p_x)}. That is a US-conditions estimate, not a probability for this record.",
            f"Random forest: percentile {pct_rf:.2%}.",
            f"Peer-deviation score {peer_dev:.2f} (0 = typical for its peers, 1 = as unusual "
            "as the measure allows)."]


# ------------------------------------------------------------------------------- pages
def page_overview():
    animals = "🐢 🦜 🐊 🦈 🐘 🦎 🦋 🐍 🦏 🐅 🪸 🐒"
    st.markdown(
        f"<div style='background:linear-gradient(120deg,#2F4A2A 0%,{OLIVE} 60%,#5E6B3A 100%);"
        f"border-radius:14px;padding:26px 30px 18px;margin-bottom:18px;color:{CREAM};"
        f"box-shadow:inset 0 0 0 3px rgba(239,230,204,0.35)'>"
        "<div class='wg-sign' style='font-size:2.6rem;font-weight:700;line-height:1.1'>"
        "WildGuard</div>"
        "<div style='font-size:1.05rem;opacity:0.92;margin-top:6px'>Cross-dataset label transfer "
        "for wildlife-trade risk prioritisation</div>"
        "<div aria-hidden='true' style='font-size:1.7rem;letter-spacing:0.55rem;margin-top:16px;"
        f"filter:brightness(0) invert(92%) sepia(25%);opacity:0.6'>{animals}</div></div>",
        unsafe_allow_html=True)
    ev, cs = load_json(config.OUTPUTS_DIR / "lemis_evaluation.json"), load_json(
        config.OUTPUTS_DIR / "cites_scoring_summary.json")
    need(ev, "Evaluation results")
    (y0, y1) = cs["years"] if cs else (2015, 2024)
    st.markdown(
        "Customs and CITES authorities cannot inspect every wildlife shipment. WildGuard learns "
        "what came before enforcement action from **US wildlife import declarations, 2000–2014, "
        "with what officers did about each (LEMIS)**, and uses those patterns to rank **CITES "
        f"export permits from {y0}–{y1}** so reviewers can look at the riskiest first.")
    m = ev["metrics_test"]["listed:shipments"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cases found at a 5% budget", f"{m['ensemble']['recall@5%']:.0%}",
              help="Share of shipments with an enforcement case that are among the 5% of "
                   "shipments WildGuard ranks highest.")
    c1.caption(f"random inspection: 5% · listing rule: {m['baseline:listing_rule']['recall@5%']:.0%}")
    c2.metric("Real cases in the top 1%", f"{m['ensemble']['precision@1%']:.0%}",
              help="Of the 1% of shipments ranked highest, the share that had an enforcement case.")
    c2.caption(f"across all shipments: {m['ensemble']['base_rate']:.1%}")
    if cs:
        c3.metric("CITES permits ranked", f"{cs['consignments'] / 1e6:.2f} M")
        c3.caption(f"{cs['records_ranked'] / 1e6:.2f} M records, {y0}–{y1}")
        c4.metric("Priority permits", f"{sum(cs['priority_consignments_per_year'].values()):,}")
        c4.caption("top 1% of each year's permits")
    st.caption("The first two numbers come from the one test: US imports of CITES-listed species "
               "in 2013–2014, years the models never saw, counted per shipment.")

    st.subheader("What a flag means")
    st.info(
        "A flag means a permit's records **resemble trade that US officers stopped** (refused, "
        "seized, abandoned or re-exported) in 2000–2014. It is a prompt for a reviewer's limited "
        "time, **not evidence that a law was broken**: CITES records permitted trade, and most "
        "flagged permits will be lawful. Most stopped US shipments never appear in CITES at all, "
        "so a flag says a record looks like stopped trade, not that it was stopped. Some signals "
        "exist only in CITES, such as the two countries' reports of a permit disagreeing; they "
        "cannot be learned from US outcomes, so the flag card shows them as context.")

    st.subheader("How it works")
    steps = [
        ("1 · Select", "Records", "Learn from LEMIS animal imports; rank each CITES trade once."),
        ("2 · Align", "Knowledge modelling", "One code list for products, units, purposes, "
                                             "sources, countries, taxa and CITES listings."),
        ("3 · Features", "Feature engineering", "How each record departs from its peers in US "
                                                "trade: quantity, route, purpose, source."),
        ("4 · Learn", "XGBoost + random forest", "Two models, chosen from six, learn from US "
                                                 "enforcement outcomes."),
        ("5 · Context", "Peer deviation", "How unusual a record is for its peers: shown to the "
                                          "reviewer, not scored."),
        ("6 · Combine", "Rank averaging", "Score = average of the two models' ranks; reasons "
                                          "from XGBoost."),
        ("7 · Rank", "Review queue", "Permits ranked within each year: priority = top 1%, "
                                     "review = next 4%."),
    ]
    for col, (t, tech, desc) in zip(st.columns(len(steps)), steps):
        col.markdown(f"<div class='wg-step'><b>{t}</b><br><span style='color:{BLAZE};"
                     f"font-size:0.72rem;font-weight:700'>{tech.upper()}</span><br>{desc}</div>",
                     unsafe_allow_html=True)

    st.subheader("Where to go next")
    a, b, c, d = st.columns(4)
    a.button("Open the review queue", on_click=go_to, args=("Review queue",), type="primary",
             use_container_width=True)
    b.button("Check a record", on_click=go_to, args=("Check a record",), use_container_width=True)
    c.button("See how it was tested", on_click=go_to, args=("Does it work?",),
             use_container_width=True)
    d.button("See what it flags", on_click=go_to, args=("What it flags",), use_container_width=True)

    ds = load_json(config.DATASETS_JSON)
    if not ds:
        return
    # every value below is read from the raw files by wildguard.datasets, so a rebuild on a new
    # release updates this section; the table mirrors section 3.1 of the final report
    le, ci = ds["lemis"], ds["cites"]
    (s0, s1), (t0, t1) = ci["scored_years"], (config.TEST_YEARS[0], config.TEST_YEARS[-1])
    ranked = (f", of which {cs['records_ranked']:,} are ranked (each trade counted once; records "
              "already confiscated or for law enforcement set aside)") if cs else ""
    with st.expander("The two source datasets"):
        st.markdown(
            "| | LEMIS (US Fish & Wildlife Service) | CITES Trade Database |\n|---|---|---|\n"
            "| Content | Every wildlife import declared to USFWS | Trade in CITES-listed species, "
            "reported by all Parties |\n"
            f"| Version | [Zenodo record {le['record']}](https://doi.org/{le['doi']}), v{le['version']} "
            f"(published {le['published']}) | Full download v{ci['version']} (UNEP-WCMC)"
            + (f", extracted {ci['extracted']}" if ci["extracted"] else "") + " |\n"
            f"| Size | {le['line_items']:,} import line items in {le['shipments']:,} shipments | "
            f"{ci['records']:,} records ({ci['years'][0]}–{ci['years'][1]}) |\n"
            f"| Period used | {le['years'][0]}–{le['years'][1]}, animals with a taxonomic class | "
            f"Animal records {s0}–{s1} ({ci['animal_records_scored']:,}){ranked}; {t0}–{t1} "
            "US-reported records for the alignment check |\n"
            "| Labels | `action` (Cleared/Refused), `disposition` (Cleared/Seized/Abandoned/"
            "Re-exported) | none |\n"
            "| Declared value | yes (USD) | no |")
        st.markdown("**What the two original datasets look like**")
        st.caption(f"LEMIS · {le['file']}, as published by EcoHealth Alliance (the 'cleaned' in "
                   "the name is theirs) · first five rows")
        st.dataframe(load_head("lemis"), hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.Column(help=h)
                                    for c, h in lemis_fields().items()})
        st.caption(f"CITES · the first five animal records from {s0}–{s1}, in download order "
                   f"({', '.join(ci['head_files'])} of {ci['files']} files)")
        st.dataframe(load_head("cites"), hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.Column(help=h)
                                    for c, h in CITES_FIELDS.items()})
        st.caption("Values are shown as they appear in the source files: LEMIS writes missing "
                   "values as NA; CITES leaves them blank and replaces confidential permit "
                   "numbers with anonymised IDs. Hover a column name for its meaning.")


def permit_card(p, q, gid, sampled, scale):
    """Flag card for one permit of the queue, followed by the permit's records."""
    r = q.iloc[p["row"]]
    y = int(p["year"])
    approx = "" if p["tier"] == "priority" else "about "
    rank_line = (f"{top_text(p['top'])} of {y} permits · {approx}rank {p['rank']:,} of "
                 f"{p['permits_in_year']:,}")
    card = dict(r) | {"tier": p["tier"], "year": y, "product": r["product_code"]}
    flag_card(card, rank_line, [s for s in str(r["reasons"]).split(" | ") if s],
              r["peer_context"], r["warning"], p["tier"] != "routine")

    recs = q[gid == p["gid"]]
    no_permit = str(p["consignment"]).startswith("record ")
    if no_permit:
        st.markdown("**On this permit**")
        st.markdown("No export permit is recorded, so this record is ranked on its own.")
    else:
        st.markdown(f"**On this permit** · {len(recs)} record{'s' if len(recs) > 1 else ''}")
        t = recs.assign(_top=recs.index == p["row"]).sort_values(
            ["_top", "reporter_type", "score"], ascending=[False, True, False])
        st.dataframe(pd.DataFrame({
            "": np.where(t["_top"], "★", ""),
            "Reported by": t["reporter_type"].map({"E": "exporter", "I": "importer"}).astype(str),
            "Taxon": t["taxon_display"].astype(str), "Product": t["product_code"].astype(str),
            "Quantity": t["quantity_std"], "Unit": t["unit_group"].astype(str),
            "Purpose": t["purpose"].astype(str), "Source": t["source"].astype(str),
            "Origin": t["origin"].astype(str)}),
            hide_index=True, use_container_width=True, height=min(38 + 35 * len(t), 260),
            column_config={"Quantity": st.column_config.NumberColumn(format="%.4g")})
        notes = ["★ is the record that sets the permit's score (its highest)."]
        if p["both_reports"]:
            notes.append("Both countries reported this permit. An importer's record appears "
                         "only where it has no exact match in the exporter's report, so "
                         "differences between the two reports show here.")
        if sampled:
            notes.append("This permit comes from the random sample of its tier, so some of its "
                         "records may be missing.")
        st.caption(" ".join(notes))

    s = scale[str(y)]
    with st.expander("Model details"):
        lines = member_lines(r["score"], np.interp(r["score_xgboost"], s["xgboost"], SCALE_GRID),
                             np.interp(r["score_random_forest"], s["random_forest"], SCALE_GRID),
                             r["score_xgboost"], r["peer_deviation"], f"{y} CITES records")
        lines.append(f"The permit's score is its highest record's, {p['consignment_score']:.4f}. "
                     f"Cut-offs in {y}: priority from {s['priority_cut']:.4f}, review from "
                     f"{s['review_cut']:.4f}.")
        if not no_permit:
            lines.append(f"Export permit (anonymised ID): {p['consignment']}")
        for line in lines:
            st.markdown(f"- {line}")


def page_queue():
    st.title("Review queue")
    P, gid = load_permits()
    need(P, "Review queue")
    q, scale = load_queue(), load_json(config.OUTPUTS_DIR / "cites_record_scale.json")
    cs = load_json(config.OUTPUTS_DIR / "cites_scoring_summary.json")
    excl = (f" {cs['excluded_known_enforcement']:,} records already confiscated (source I) or "
            "for law enforcement (purpose L) are left out.") if cs else ""
    st.markdown(
        "CITES export permits, ranked within each year by how closely their records resemble "
        "trade that US officers stopped. Work down from the top of a year: **priority** is the "
        "top 1% of the year's permits, **review** the next 4%. Select a permit to see its flag "
        "card.")
    st.caption("A flag is a prompt for human review, not a finding of wrongdoing: most flagged "
               "permits will be lawful." + excl)

    years = sorted(P["year"].unique().tolist(), reverse=True)
    f1, f2, f3 = st.columns([1, 1.5, 2.2])
    year = f1.selectbox("Year", years + ["All years"])
    tier = f2.selectbox("Tier", list(TIER_CHOICES), format_func=TIER_CHOICES.get)
    search = f3.text_input("Search taxon", placeholder="e.g. Python, Manis, Crocodylus")
    with st.expander("More filters"):
        a, b, c = st.columns(3)
        cls = a.multiselect("Class", sorted(P["class"].dropna().unique()), format_func=with_icon,
                            placeholder="All classes")
        exp = b.multiselect("Exporter", sorted(P["exporter"].unique()), format_func=cname,
                            placeholder="All exporters")
        imp = c.multiselect("Importer", sorted(P["importer"].unique()), format_func=cname,
                            placeholder="All importers")
        a, b = st.columns(2)
        only_warn = a.checkbox("Only permits with a US-law warning")
        only_both = b.checkbox("Only permits reported by both countries")
    keep = (P["tier"] == tier).to_numpy()
    if year != "All years":
        keep &= (P["year"] == year).to_numpy()
    if search:
        keep &= np.isin(P["gid"], np.unique(
            gid[q["taxon_display"].str.contains(search, case=False, na=False).to_numpy()]))
    for values, col in ((cls, "class"), (exp, "exporter"), (imp, "importer")):
        if values:
            keep &= P[col].isin(values).to_numpy()
    if only_warn:
        keep &= P["any_warning"].to_numpy()
    if only_both:
        keep &= P["both_reports"].to_numpy()
    v = P[keep].sort_values(["top", "year"], ascending=[True, False]).reset_index(drop=True)

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown(f"**{len(v):,} permits**" + ("" if tier == "priority" else
                                                 " · a random sample of this tier"))
        cols = {"Rank": v["rank"]} | ({"Year": v["year"]} if year == "All years" else {}) | {
            "Taxon": v["taxon_label"], "Product": v["product_code"].astype(str),
            "Route": v["route"], "Records": v["records"]}
        event = st.dataframe(
            pd.DataFrame(cols).head(3000), hide_index=True, use_container_width=True, height=640,
            on_select="rerun", selection_mode="single-row",
            column_config={
                "Rank": st.column_config.NumberColumn(
                    format="%d", width="small",
                    help="Position among the year's permits, 1 = highest score. Approximate for "
                         "the sampled review and routine tiers."),
                "Year": st.column_config.NumberColumn(format="%d", width="small"),
                "Taxon": st.column_config.Column(width="medium",
                                                 help="The record that sets the permit's score; "
                                                      "+n = other taxa on the permit"),
                "Product": st.column_config.Column(width="medium"),
                "Route": st.column_config.Column(width="small", help="Exporter → importer"),
                "Records": st.column_config.NumberColumn(width="small",
                                                         help="Records on this permit")})
        if len(v) > 3000:
            st.caption("Showing the first 3,000. Narrow the list with the filters.")
    with right:
        if not len(v):
            st.info("No permits match these filters.")
            return
        rows = event.selection.rows if event and event.selection else []
        i = rows[0] if rows and rows[0] < min(len(v), 3000) else 0
        if not rows:
            st.caption("Showing the first permit in the list. Select a row to see another.")
        with st.container(border=True):
            permit_card(v.iloc[i], q, gid, tier != "priority", scale)


PRESETS = {
    "(blank)": dict(
        us=False, genus="", species="", cls="Reptilia", product="live", qty=1.0, unit="count",
        purpose="T", source="W", exporter="ID", origin="ID", importer="US", appendix="II",
        year=2024, value=100.0, port="LA", month=6),
    "Indian star tortoise: a real US case from 2000, seized": dict(
        us=True, genus="Geochelone", species="elegans", cls="Reptilia", product="live", qty=2.0,
        unit="count", purpose="P", source="W", exporter="MY", origin="MY", importer="US",
        appendix="II", year=2024, value=800.0, port="SF", month=2),
    "Saiga antelope medicine, China → Japan: a priority CITES record (2019)": dict(
        us=False, genus="Saiga", species="tatarica", cls="Mammalia", product="extracts_medicine",
        qty=23.4, unit="kg", purpose="T", source="U", exporter="CN", origin="XX", importer="JP",
        appendix="II", year=2019, value=100.0, port="LA", month=6),
    "Captive-bred ball pythons, Ghana → US, commercial": dict(
        us=False, genus="Python", species="regius", cls="Reptilia", product="live", qty=200.0,
        unit="count", purpose="T", source="C", exporter="GH", origin="GH", importer="US",
        appendix="II", year=2020, value=2400.0, port="NY", month=5),
}


def page_check():
    st.title("Check a record")
    st.markdown("Enter one trade record to see where WildGuard would rank it and why. It is "
                "ranked as if it were a permit on its own, against that year's CITES permits.")
    demo = st.query_params.get("example") == "tortoise"     # deep link used for the demo video
    names = list(PRESETS)
    preset = st.selectbox("Start from an example", names, index=1 if demo else 0)
    p = PRESETS[preset]
    us = st.toggle("Add US import declaration details (declared value, entry port, month)",
                   value=p["us"], key=f"us::{preset}",
                   help="Uses the full model, which also learned from these US-only fields. The "
                        "record is then ranked among US import shipments, 2000–2014, instead of "
                        "CITES permits.")
    cites_years = list(config.CITES_SCORE_YEARS)
    countries = sorted(COUNTRY)
    with st.form("record"):
        a, b, c, d = st.columns(4)
        genus = a.text_input("Genus", p["genus"])
        species = b.text_input("Species", p["species"], help="Leave blank for a genus-level record")
        cls = c.selectbox("Class", CLASSES, index=CLASSES.index(p["cls"]), format_func=with_icon)
        appendix = d.selectbox("CITES listing", list(APPENDIX_LABELS),
                               index=list(APPENDIX_LABELS).index(p["appendix"]),
                               format_func=APPENDIX_LABELS.get)
        a, b, c, d = st.columns(4)
        product = a.selectbox("Product", PRODUCT_GROUPS, index=PRODUCT_GROUPS.index(p["product"]),
                              format_func=product_label)
        qty = b.number_input("Quantity", min_value=0.0, value=p["qty"])
        unit = c.selectbox("Unit", UNITS, index=UNITS.index(p["unit"]))
        if us:
            month = d.number_input("Month", 1, 12, p["month"])
            year = None
        else:
            year = d.selectbox("Year", cites_years, index=cites_years.index(p["year"]),
                               help="The record is ranked against this year's permits.")
            month = None
        a, b, c, d = st.columns(4)
        purposes = list(PURPOSE)
        purpose = a.selectbox("Purpose", purposes, index=purposes.index(p["purpose"]),
                              format_func=lambda k: f"{k} - {PURPOSE[k]}")
        sources = [k for k in SOURCE if k != "unknown"]
        source = b.selectbox("Source", sources, index=sources.index(p["source"]),
                             format_func=lambda k: f"{k} - {SOURCE[k]}")
        exporter = c.selectbox("Exporter", countries, index=countries.index(p["exporter"]),
                               format_func=cname)
        origin = d.selectbox("Origin", countries, index=countries.index(p["origin"]),
                             format_func=cname)
        a, b, c, d = st.columns(4)
        if us:
            importer = "US"
            value = a.number_input("Declared value (USD)", min_value=0.0, value=p["value"])
            ports = sorted(PORT, key=PORT.get)
            port = b.selectbox("US entry port", ports, index=ports.index(p["port"]),
                               format_func=lambda k: f"{PORT[k]} ({k})")
        else:
            importer = a.selectbox("Importer", countries, index=countries.index(p["importer"]),
                                   format_func=cname)
            value, port = np.nan, None
        submitted = st.form_submit_button("Rank this record", type="primary")
    if not (submitted or demo):
        return
    if not genus.strip():
        st.warning("Enter a genus: the record's peers and its route are measured by genus.")
        return
    if not us and (source == "I" or purpose == "L"):
        st.info("This record is already enforcement material (a confiscated specimen or a "
                "law-enforcement purpose), so WildGuard does not rank it for review.")
        return

    g = genus.strip().lower()
    order_name, family = genus_taxonomy().get(g, ("", ""))
    rec = {"year": year or max(config.DEPLOY_YEARS), "month": float(month) if month else np.nan,
           "class": cls, "order_name": order_name, "family": family, "genus": g,
           "taxon": f"{genus} {species}".strip().lower(),
           "taxon_display": f"{genus} {species}".strip(), "product_group": product,
           "unit_group": unit, "quantity_std": float(qty), "value_usd": value,
           "purpose": purpose, "source": source, "origin": origin, "exporter": exporter,
           "importer": importer, "port": port or "unknown", "appendix": appendix,
           "origin_region": region(origin), "exporter_region": region(exporter)}
    scorer = load_scorer("full" if us else "cites")
    need(scorer, "Model files")
    ens, fb = scorer
    F = fb.transform(pd.DataFrame([rec]))
    m = {k: float(v[0]) for k, v in ens.member_scores(F).items()}
    if us:
        L = load_json(config.OUTPUTS_DIR / "lemis_record_scale.json")
        need(L, "US record scale")
        pct = {k: float(ens.rankers[k].transform(v)) for k, v in m.items()}
        cuts, grid = (L["priority_cut"], L["review_cut"]), L["shipment_score"]
        population = f"US import shipments, {L['years'][0]}–{L['years'][1]}"
        ranked_as = "a shipment of one line item"
    else:
        S = load_json(config.OUTPUTS_DIR / "cites_record_scale.json")
        need(S, "CITES record scale")
        S = S[str(year)]
        pct = {k: float(np.interp(v, S[k], SCALE_GRID)) for k, v in m.items()}
        cuts, grid = (S["priority_cut"], S["review_cut"]), S["permit_score"]
        population = f"{year} CITES permits"
        ranked_as = "a permit with this one record"
    score = (pct["xgboost"] + pct["random_forest"]) / 2
    tier = "priority" if score >= cuts[0] else "review" if score >= cuts[1] else "routine"
    top = 1 - float(np.interp(score, grid, SCALE_GRID))
    card = rec | {"tier": tier, "product": product_label(product), "year": year}
    with st.container(border=True):
        flag_card(card, f"{top_text(top)} of {population}",
                  explain_rows(ens.xgb, ens.matrix(F), F)[0],
                  peer_context(PeerDeviationScorer().parts(F))[0], us_law_warning(F)[0],
                  tier != "routine")
        st.caption(f"Ranked as {ranked_as}. Scored by " + (
            "the full model: the shared fields plus declared value, entry port and month."
            if us else "the CITES model, which uses only the 21 fields LEMIS and CITES share."))
        with st.expander("Model details"):
            for line in member_lines(score, pct["xgboost"], pct["random_forest"], m["xgboost"],
                                     float(PeerDeviationScorer().score(F)[0]),
                                     population.replace("permits", "records")
                                     .replace("shipments", "line items")):
                st.markdown(f"- {line}")
            st.markdown(f"- Cut-offs: priority from {cuts[0]:.4f}, review from {cuts[1]:.4f} "
                        "(the top 1% and the next 4% of the population).")
    if us:
        st.caption("Both models were trained on US imports 2000–2014, so a declaration from those "
                   "years (such as the tortoise case) is an in-sample illustration, not a test "
                   "result.")


def page_evaluation():
    st.title("Does it work?")
    ev = load_json(config.OUTPUTS_DIR / "lemis_evaluation.json")
    ships = load_test_shipments()
    need(ev, "Evaluation results")
    need(ships, "Held-out scores")
    st.markdown(
        "Only the US import records (LEMIS) say what officers actually did, so that is where "
        "WildGuard was tested: once, on **2013–2014 shipments the models never saw**, against "
        "what happened to each (refused, seized, abandoned or re-exported, versus cleared).")
    a, b = st.columns([1, 2], gap="large")
    sub = a.selectbox("Shipments", list(SUBSETS), format_func=SUBSETS.get)
    budget = a.slider("Your team can inspect this share of shipments", 0.5, 30.0, 5.0, 0.5,
                      format="%.1f%%") / 100
    v = ships[sub]
    y = v["label"].to_numpy()
    k = max(1, int(round(budget * len(v))))
    res = {}
    for col in ("ensemble", "baseline:listing_rule"):
        top = np.argsort(-v[col].to_numpy(), kind="stable")[:k]
        res[col] = (y[top].sum() / max(y.sum(), 1), y[top].mean())
    caught, per = res["ensemble"]
    a.metric("Enforcement cases found", f"{caught:.0%}",
             help="Share of the shipments with an enforcement case that fall within the budget.")
    # random inspection by its expected value: it finds the budget's share of cases
    a.caption(f"random inspection: {budget * 100:g}% · listing rule: "
              f"{res['baseline:listing_rule'][0]:.0%}")
    a.metric("Real cases per 100 inspections", f"{per * 100:.0f}")
    a.caption(f"random inspection: {y.mean() * 100:.1f}")
    a.caption(f"{len(v):,} held-out shipments, {int(y.sum()):,} with an enforcement case "
              f"({y.mean():.1%}).")
    fig = go.Figure()
    idx = np.unique(np.linspace(0, int(0.3 * len(v)), 400).astype(int))
    for col, label, color in (("ensemble", "WildGuard", GREEN),
                              ("baseline:listing_rule", MODEL_NAMES["baseline:listing_rule"], SLATE),
                              ("baseline:random", "Random inspection (expected)", GREY)):
        if col == "baseline:random":
            cum = np.arange(1, len(v) + 1) / len(v)
        else:
            cum = np.cumsum(y[np.argsort(-v[col].to_numpy(), kind="stable")]) / max(y.sum(), 1)
        fig.add_trace(go.Scatter(x=(idx + 1) / len(v) * 100, y=cum[idx] * 100, name=label,
                                 line=dict(color=color, width=2),
                                 hovertemplate="inspect %{x:.1f}% → find %{y:.1f}%<extra>"
                                               + label + "</extra>"))
    fig.add_vline(x=budget * 100, line=dict(color="#898781", width=1))
    fig.update_xaxes(title="Share of shipments inspected (%), highest score first", range=[0, 30])
    fig.update_yaxes(title="Enforcement cases found (%)", range=[0, 100])
    b.markdown("**Cases found for each inspection budget**")
    b.plotly_chart(plotly_layout(fig, 420), use_container_width=True)

    met = ev["metrics_test"]
    e_s, x_s = met[f"{sub}:shipments"]["ensemble"]["pr_auc"], met[f"{sub}:shipments"]["xgboost"]["pr_auc"]
    d = next(r for r in ev["pr_auc_intervals"][sub]["ensemble_minus"] if r["b"] == "xgboost")
    sig = d["high_95"] < 0 or d["low_95"] > 0
    st.markdown(
        f"**One result went against the design:** on these shipments XGBoost alone ranked "
        f"{'slightly better' if x_s > e_s else 'no better'} than the two-model ensemble "
        f"(PR-AUC {x_s:.3f} vs {e_s:.3f}; per line item the ensemble's difference is "
        f"{d['difference']:+.3f}, 95% interval {d['low_95']:+.3f} to {d['high_95']:+.3f}"
        f"{', a significant difference' if sig else ''}). The ensemble was chosen on 2011, "
        "before the test, where it was ahead; it was kept so that the test stays a test.")

    with st.expander("How this was measured"):
        yr = ev["years"]
        st.markdown(
            f"- **Years:** trained on {yr['train'][0]}–{yr['train'][1]}; every design choice "
            "made on 2011; 2012 held back for the confidence bands; tested once on "
            f"{yr['test'][0]}–{yr['test'][1]}. The deployed model was then refit on "
            f"{yr['deployed_model_trained_on'][0]}–{yr['deployed_model_trained_on'][1]}.\n"
            "- **Shipments:** a shipment's score is its highest line item's; it counts as a case "
            "if any of its line items drew enforcement action. A shipment's line items were never "
            "split between training and testing.\n"
            "- **The deciding measure:** PR-AUC on line items of CITES-listed species, the part of "
            "LEMIS most like CITES. It summarises the whole ranking without fixing a budget; "
            "accuracy would reward clearing everything, since cases are under 2% of line items.\n"
            "- **Listing rule:** Appendix I first, then II and III, wild-caught first within "
            "each, using the listing in force in the shipment year.")
    with st.expander("All models compared"):
        unit = st.radio("Counted per", ["shipments", "line_items"], horizontal=True,
                        format_func=lambda u: u.replace("_", " "))
        mt, iv = met[f"{sub}:{unit}"], ev["pr_auc_intervals"][sub]["pr_auc"]
        tab = pd.DataFrame({
            "PR-AUC": [mt[k]["pr_auc"] for k in TEST_MODELS],
            "ROC-AUC": [mt[k]["roc_auc"] for k in TEST_MODELS],
            "found at 1%": [mt[k]["recall@1%"] for k in TEST_MODELS],
            "found at 5%": [mt[k]["recall@5%"] for k in TEST_MODELS],
            "inspect to find 80%": [mt[k]["inspect_for_80%_recall"] for k in TEST_MODELS]},
            index=[MODEL_NAMES[k] for k in TEST_MODELS])
        fmt = {c: "{:.3f}" for c in ("PR-AUC", "ROC-AUC")} | \
              {c: "{:.1%}" for c in ("found at 1%", "found at 5%", "inspect to find 80%")}
        if unit == "line_items":
            tab.insert(1, "95% interval", [f"{iv[k]['low_95']:.3f}–{iv[k]['high_95']:.3f}"
                                           if k in iv else "" for k in TEST_MODELS])
        st.dataframe(tab.style.format(fmt), use_container_width=True)
        st.caption("Intervals resample whole shipments (1,000 bootstrap draws). 'Found at 5%': "
                   "share of cases among the top 5% ranked; 'inspect to find 80%': share that "
                   "must be inspected to find 80% of the cases.")
        diff = pd.DataFrame(ev["pr_auc_intervals"][sub]["ensemble_minus"])
        diff["compared with"] = diff["b"].map(MODEL_NAMES)
        st.markdown("**WildGuard's PR-AUC minus each other model's, per line item**")
        st.dataframe(diff.set_index("compared with")[["difference", "low_95", "high_95"]]
                     .rename(columns={"low_95": "95% low", "high_95": "95% high"})
                     .style.format("{:+.4f}"), use_container_width=True)
    c3 = ev["lemis_only_fields_c3"]
    with st.expander("What the US-only fields add (declared value, port, month)"):
        dd = c3["full_minus_shared"][0]
        st.markdown(
            f"CITES has no declared value, entry port or month, so WildGuard cannot use them. On "
            f"the {c3['line_items']:,} test line items from 2013 with a recorded value (2014 has "
            f"none), the same ensemble with these fields reaches PR-AUC "
            f"{c3['pr_auc']['full_ensemble']['pr_auc']:.3f} against "
            f"{c3['pr_auc']['ensemble']['pr_auc']:.3f} without them: **{dd['difference']:+.3f}** "
            f"(95% interval {dd['low_95']:+.3f} to {dd['high_95']:+.3f}). That is what aligning "
            "the two datasets gives up.")
    bands = ev["confidence_bands"]
    with st.expander("Confidence bands for US triage"):
        st.dataframe(pd.DataFrame(bands["test"]).T.rename(columns={
            "share_of_records": "share of line items",
            "share_of_positives": "share of enforcement cases",
            "positive_rate": "enforcement rate in band"}).style.format("{:.1%}"),
            use_container_width=True)
        st.caption(f"Thresholds fixed on {bands['fixed_on']}: auto-clear keeps 95% of that year's "
                   "cases above it; priority is the top 2% of that year's scores. Shown on the "
                   "2013–2014 test line items.")
    ds = load_json(config.OUTPUTS_DIR / "domain_shift.json")
    if ds:
        with st.expander("Does it transfer to CITES?"):
            rec, dep = (ds["recording_shift_lemis_listed_vs_cites_us_2013_14"],
                        ds["deployment_shift_lemis_train_vs_cites_ranked"])
            st.markdown(
                "CITES has no outcomes, so transfer cannot be tested directly. What can be "
                "measured:\n"
                f"- **Recording:** even for the same US trade (LEMIS listed species against US-reported "
                f"CITES records, 2013–2014), a classifier tells the two datasets apart with AUC "
                f"{rec['domain_classifier_auc']:.2f} from the raw fields (0.5 = indistinguishable).\n"
                f"- **Trade mix:** LEMIS training records against the ranked CITES records: AUC "
                f"{dep['domain_classifier_auc']:.2f}. The largest shifts are in the Appendix, "
                "class and product (table).\n"
                "- **Time:** within LEMIS, PR-AUC fell from 0.570 to 0.512 as the gap between the "
                "training and scored years grew from 1 to 7 years (notebook 13), so the deployed "
                "model was refit on 2000–2014, and later CITES years are likely ranked less "
                "well.\n"
                "- **Jurisdiction:** not measurable; there are no enforcement outcomes outside "
                "the US.")
            psi = pd.DataFrame({"same US trade, recorded twice": rec["psi"],
                                "LEMIS training vs CITES ranked": dep["psi"]})
            st.dataframe(psi.style.format("{:.3f}"), use_container_width=True)
            st.caption("Population stability index per field: above 0.25 is usually read as a "
                       "large shift.")
            fig_path = config.FIGURES_DIR / "domain_shift_pca.png"
            if fig_path.exists():
                st.image(str(fig_path), caption="The deviation features of LEMIS and CITES, "
                                                "projected on their first two principal components")


def page_patterns():
    st.title("What it flags")
    st.markdown(
        "CITES has no enforcement outcomes, so the ranking cannot be scored there. Instead it was "
        "checked against evidence from outside the model. Each pass mark was written down before "
        "the result was seen. A pass means the ranking is *consistent* with that evidence, not "
        "that it is accurate.")
    val = load_json(config.OUTPUTS_DIR / "cites_validation.json")
    need(val, "CITES checks")
    u, top50 = load_csv("cites_unodc_consistency.csv"), load_csv("cites_top50_review.csv")
    sup = load_json(config.OUTPUTS_DIR / "lemis_supplementary.json")
    uc, cp, tr = val["unodc_consistency"], val["confiscated_proxy"], val["top_records"]
    none_c = top50[top50["category"] == "c"] if top50 is not None else None
    checks = [
        (uc["pass"], "1 · Seizure data (UNODC)",
         f"Animal groups seized more often than their legal trade suggests are flagged more "
         f"often: Spearman ρ = {uc['spearman_rho_wildguard']:.2f} (one-sided p = "
         f"{uc['permutation_p_one_sided']}), above the listing rule's "
         f"{uc['spearman_rho_listing_rule']:.2f}.",
         "Pass: ρ positive with p below 0.05, and above the listing rule's."),
        (cp["pass"], "2 · Confiscated specimens",
         f"Retrained without any source information, the model still ranks confiscated "
         f"specimens above other records: ROC-AUC {cp['roc_auc']:.3f} (95% interval "
         f"{cp['low_95']:.3f}–{cp['high_95']:.3f}); {cp['share_of_source_I_in_top_5pct']:.0%} of "
         "them fall in the top 5%.",
         "Pass: the interval's lower end above 0.5."),
        (tr["pass"], "3 · The top 50 taxa",
         f"Of the {tr['records']} highest-ranked taxa, {tr['share_a_recognised_pattern']:.0%} show "
         f"a recognised risk pattern, {tr['share_b_us_law_abroad']:.0%} a pattern of US law, and "
         f"{tr['share_c_no_apparent_reason']:.0%} no apparent reason.",
         "Pass: at least 50% recognised and at most 20% with no apparent reason."),
    ]
    for col, (passed, title, result, rule) in zip(st.columns(3), checks):
        with col.container(border=True):
            st.markdown(f"{check_badge(passed)} &nbsp; **{title}**", unsafe_allow_html=True)
            st.markdown(result)
            st.caption(rule)
    if none_c is not None and len(none_c):
        same = none_c.groupby(["purpose", "source", "appendix"]).size()
        into = ", ".join(f"{COUNTRY.get(c, c)} {n}" for c, n in
                         none_c["importer"].value_counts().items())
        pattern = ""
        if len(same) == 1:
            (pu, so, ap), = same.index
            pattern = (f"all {len(none_c)} are {PURPOSE.get(pu, pu).lower()}-purpose imports of "
                       f"{SOURCE.get(so, so)} Appendix {ap} specimens (importers: {into}). ")
        personal = (f"In 2013–2014, {sup['personal_only']['ensemble']['base_rate']:.0%} of US "
                    "personal-purpose line items drew enforcement action, so the models learned "
                    "this pattern from US practice; ") if sup else ""
        st.markdown(
            f"**Why check 3 fails:** {pattern}{personal}it is not one of the patterns fixed in "
            "advance as recognised risks. The result is reported as a fail, not re-ruled. The "
            "categories were applied by a stated rule (`validate_cites.classify`), not by human "
            "reviewers.")

    if u is not None:
        with st.expander("Check 1 in detail: the 14 UNODC animal groups"):
            show = (u["wildguard_priority_share"] >= 0.01) | (u["seizure_to_trade_ratio"] >= 5)
            pos = np.where(u["group"] == "Carnivores", "bottom center", "top center")
            fig = go.Figure(go.Scatter(
                x=u["seizure_to_trade_ratio"], y=u["wildguard_priority_share"] * 100,
                mode="markers+text", text=u["group"].where(show, ""), textposition=pos,
                marker=dict(size=10, color=GREEN, line=dict(color=KHAKI, width=2)),
                customdata=np.c_[u["group"], u["cites_records"]],
                hovertemplate="%{customdata[0]}<br>seizure/trade ratio %{x:.2f}<br>priority "
                              "%{y:.2f}%<br>%{customdata[1]:,} CITES records<extra></extra>"))
            fig.update_xaxes(type="log", title="UNODC seizure share ÷ CITES legal-trade share (log)",
                             tickvals=[0.3, 1, 3, 10, 30, 100, 300, 1000],
                             ticktext=["0.3", "1", "3", "10", "30", "100", "300", "1,000"])
            fig.update_yaxes(type="log", title="Share in priority permits (%, log)",
                             tickvals=[0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10],
                             ticktext=["0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10"])
            st.plotly_chart(plotly_layout(fig, 440, showlegend=False), use_container_width=True)
            tbl = u[["group", "cites_records", "unodc_seizure_share", "cites_trade_share",
                     "seizure_to_trade_ratio", "wildguard_priority_share", "listing_rule_top1"]]
            st.dataframe(tbl.rename(columns={
                "cites_records": "CITES records", "unodc_seizure_share": "UNODC seizure share",
                "cites_trade_share": "CITES trade share", "seizure_to_trade_ratio": "seizure/trade",
                "wildguard_priority_share": "WildGuard priority share",
                "listing_rule_top1": "listing rule top 1%"}).style.format({
                    "CITES records": "{:,.0f}", "UNODC seizure share": "{:.1%}",
                    "CITES trade share": "{:.2%}", "seizure/trade": "{:.2f}",
                    "WildGuard priority share": "{:.2%}", "listing rule top 1%": "{:.2%}"}),
                hide_index=True, use_container_width=True)
            st.caption("UNODC World Wildlife Crime Report 2024 (Fig. 2.1) against exporter-reported "
                       f"CITES records 2015–2021. Unstable (fewer than 500 records): "
                       f"{', '.join(uc['unstable_groups']) or 'none'}.")
    if top50 is not None:
        with st.expander("Check 3 in detail: the 50 records and their categories"):
            cat = {"a": "recognised pattern", "b": "US law abroad", "c": "no apparent reason"}
            st.dataframe(top50.assign(category=top50["category"].map(cat))[
                ["category", "rule", "year", "taxon_display", "product_code", "quantity_std",
                 "unit_group", "purpose", "source", "exporter", "importer", "appendix", "reasons"]]
                .rename(columns={"taxon_display": "taxon", "product_code": "product",
                                 "quantity_std": "quantity", "unit_group": "unit"}),
                hide_index=True, use_container_width=True, height=400)
            st.caption("Each taxon's highest-ranked priority record, 2015–2024.")

    st.subheader("Where the flags concentrate")
    dims = {"Taxonomic class": ("by_class.csv", "class"),
            "Order": ("by_order_name.csv", "order_name"),
            "Product": ("by_product_group.csv", "product_group"),
            "Purpose": ("by_purpose.csv", "purpose"), "Source": ("by_source.csv", "source"),
            "CITES Appendix": ("by_appendix.csv", "appendix"),
            "Exporter (2,000+ records)": ("by_exporter.csv", "exporter"),
            "Importer (2,000+ records)": ("by_importer.csv", "importer"),
            "Route (1,000+ records)": ("by_route.csv", "route"),
            "Taxon (500+ records)": ("by_taxon_display.csv", "taxon_display"),
            "Year": ("by_year.csv", "year")}
    dim = st.selectbox("Break down by", list(dims))
    fname, col = dims[dim]
    d = load_csv(f"cites_aggregates/{fname}")
    need(d, "CITES aggregates")
    yrs = load_csv("cites_aggregates/by_year.csv")
    overall = float((yrs["records"] * yrs["priority_share"]).sum() / yrs["records"].sum()) \
        if yrs is not None else 0.01
    if col == "route":
        d = d.assign(route=d["exporter"].map(cname) + " → " + d["importer"].map(cname))
    d = d.sort_values("year") if col == "year" else d.head(20).iloc[::-1]
    labels = {"exporter": lambda s: s.map(cname), "importer": lambda s: s.map(cname),
              "purpose": lambda s: s.map(lambda k: PURPOSE.get(k, k)),
              "source": lambda s: s.map(lambda k: SOURCE.get(k, k)),
              "class": lambda s: s.map(with_icon), "product_group": lambda s: s.map(product_label),
              "appendix": lambda s: s.map(lambda k: APPENDIX_LABELS.get(k, k))}.get(
        col, lambda s: s.astype(str))(d[col])
    fig = go.Figure(go.Bar(
        x=d["priority_share"] * 100, y=labels, orientation="h", marker_color=GREEN,
        customdata=d["records"], width=0.6,
        hovertemplate="%{y}: %{x:.2f}% in priority permits, of %{customdata:,} records"
                      "<extra></extra>"))
    fig.add_vline(x=overall * 100, line=dict(color="#898781", width=1))
    fig.update_xaxes(title="Records in priority permits (%)")
    st.plotly_chart(plotly_layout(fig, 40 + 26 * len(d), showlegend=False),
                    use_container_width=True)
    st.caption(f"Share of each group's exporter-reported records that belong to a priority permit "
               f"(the top 20 groups, or every year). Grey line: the share across all records, "
               f"{overall:.2%}.")
    cs = load_json(config.OUTPUTS_DIR / "cites_scoring_summary.json")
    if cs:
        st.markdown(
            f"**Marine mammals traded between other countries** are "
            f"{cs['priority_share_marine_mammal_non_us_importer']:.1%} of priority records but "
            f"{cs['ranked_share_marine_mammal_non_us_importer']:.1%} of all ranked records. US "
            "law restricts imports of marine-mammal products, so their flag cards carry a "
            "warning that the pattern may reflect US law rather than risk.")


PAGES = {"Overview": page_overview, "Review queue": page_queue,
         "Check a record": page_check, "Does it work?": page_evaluation,
         "What it flags": page_patterns}
SLUGS = {"overview": 0, "queue": 1, "check": 2, "evaluation": 3, "patterns": 4}
if "nav" not in st.session_state:
    st.session_state["nav"] = list(PAGES)[SLUGS.get(st.query_params.get("page", ""), 0)]
with st.sidebar:
    st.markdown("## 🐘 WildGuard")
    choice = st.radio("Navigate", list(PAGES), key="nav", label_visibility="collapsed")
    st.caption("NUS-ISS PRS Practice Module · Group 37")
    ds = load_json(config.DATASETS_JSON)
    if ds:
        st.caption(f"Data: LEMIS (USFWS / EcoHealth Alliance, Zenodo v{ds['lemis']['version']}); "
                   f"CITES Trade Database full download v{ds['cites']['version']} "
                   "(UNEP-WCMC for the CITES Secretariat).")
PAGES[choice]()
