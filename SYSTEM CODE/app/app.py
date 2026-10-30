"""WildGuard - wildlife-trade risk screening (MVP).

    streamlit run "SYSTEM CODE/app/app.py"

Pages
    Overview          what the system does, headline results, caveats, the two source datasets
    Review queue      CITES records ranked for human review, each with plain-language reasons
    Check a shipment  score one record (CITES-style, or a US import declaration with value)
    Does it work?     triage simulation on held-out LEMIS years with real enforcement outcomes
    Risk patterns     where flagged CITES trade concentrates; consistency with UNODC data
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
from wildguard.explain import COUNTRY, PURPOSE, SOURCE, explain_rows  # noqa: E402
from wildguard.features import to_model_matrix  # noqa: E402
from wildguard.schema_alignment import region  # noqa: E402

# "ranger station" palette, matching .streamlit/config.toml: forest green, trail-blaze orange,
# slate, stone; khaki page, olive-drab sidebar, cream lettering
GREEN, BLAZE, SLATE, GREY = "#3E5C2E", "#B5652B", "#5B7F95", "#8C8270"
KHAKI, OLIVE, CREAM = "#E3DAC2", "#4A5A3A", "#EFE6CC"
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
</style>""", unsafe_allow_html=True)

TIER_STYLE = {"priority": ("#d03b3b", "Priority review"), "review": ("#fab219", "Review"),
              "routine": ("#0ca30c", "Routine"), "auto-clear": ("#0ca30c", "Auto-clear")}
PRODUCT_GROUPS = ["live", "live_eggs", "bodies", "trophies", "skins", "skin_pieces",
                  "leather_small", "leather_large", "garments", "shoes", "cloth", "specimens",
                  "meat", "caviar", "ivory", "horn", "bone", "skulls", "teeth_claws",
                  "coral_raw", "coral_live_rock", "shells", "scales", "feathers", "eggs", "hair",
                  "fins", "extracts_medicine", "carvings_jewellery", "body_parts", "other"]
CLASS_ICON = {"Reptilia": "🦎", "Mammalia": "🐘", "Aves": "🦜", "Anthozoa": "🪸",
              "Actinopteri": "🐟", "Amphibia": "🐸", "Bivalvia": "🐚", "Gastropoda": "🐌",
              "Insecta": "🦋", "Elasmobranchii": "🦈", "Hydrozoa": "🪼", "Arachnida": "🕷️",
              "Holothuroidea": "🥒", "Hirudinoidea": "🪱"}
CLASSES = ["Reptilia", "Mammalia", "Aves", "Anthozoa", "Actinopteri", "Amphibia", "Bivalvia",
           "Gastropoda", "Insecta", "Elasmobranchii", "Hydrozoa", "Arachnida"]
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


QUEUE_COLS = ["year", "class", "taxon_display", "product_code", "quantity_std", "unit_group",
              "purpose", "source", "origin", "exporter", "importer", "appendix", "reporter_type",
              "risk_percentile", "tier", "reasons", "score_supervised", "score_peer",
              "score_iforest", "score_kmeans", "score_hybrid"]
EVAL_COLS = ["purpose", "label", "transfer:hybrid", "full:hybrid", "baseline:listing_rule",
             "baseline:random"]


def _compact(df, keep_text=()):
    """Repeated strings (codes, tiers, taxa) as categoricals - keeps the hosted app small."""
    for c in df.columns:
        if df[c].dtype == object and c not in keep_text:
            df[c] = df[c].astype("category")
    return df


# The two large tables are read-only, so they are cached once as shared resources rather than
# copied per session (st.cache_data returns a fresh copy on every call).
@st.cache_resource
def load_queue():
    p = config.SAMPLES_DIR / "cites_review_queue.parquet"
    return _compact(pd.read_parquet(p, columns=QUEUE_COLS), keep_text=("reasons",)) \
        if p.exists() else None


@st.cache_resource
def load_test_scores():
    p = config.OUTPUTS_DIR / "lemis_test_scores.parquet"
    return _compact(pd.read_parquet(p, columns=EVAL_COLS)) if p.exists() else None


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


@st.cache_resource
def load_model(name):
    p = config.MODELS_DIR / name
    return joblib.load(p) if p.exists() else None


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


def cname(code):
    return f"{COUNTRY.get(code, code)} ({code})"


def with_icon(cls):
    return f"{CLASS_ICON.get(cls, '🌿')} {cls}" if isinstance(cls, str) else "🌿"


def prob(p):
    """Probability for display; avoids printing 100.0% for 0.9996."""
    return ">99.9%" if p > 0.999 else "<0.1%" if p < 0.001 else f"{p:.1%}"


def need(obj, what):
    if obj is None:
        st.warning(f"{what} not found - run the pipeline first: `bash \"SYSTEM CODE/run_pipeline.sh\"`")
        st.stop()


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
    st.markdown(
        "Customs and CITES authorities cannot inspect every wildlife shipment. WildGuard learns "
        "what historically *preceded enforcement action* from **5.5 million US import "
        "declarations with real outcomes (LEMIS)**, and uses those patterns, together with "
        "label-free anomaly detection, to rank **global CITES trade records** for human review.")
    ev, cs = load_json(config.OUTPUTS_DIR / "lemis_evaluation.json"), load_json(
        config.OUTPUTS_DIR / "cites_scoring_summary.json")
    need(ev, "Evaluation results")
    m = ev["metrics_test"]["transfer:hybrid"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cases caught at 5% inspection", f"{m['recall@5%']:.0%}")
    c1.caption("random inspection: 5%")
    c2.metric("Real cases among top-1% flags", f"{m['precision@1%']:.0%}")
    c2.caption(f"base rate: {m['base_rate']:.1%}")
    if cs:
        c3.metric("CITES records scored", f"{cs['records_scored'] / 1e6:.2f}M")
        c3.caption(f"animal trade {cs['years'][0]}-{cs['years'][1]}")
        c4.metric("Priority flags", f"{cs['priority_records']:,}")
        c4.caption("top 1% per year, for human review")
    st.caption("Held-out evaluation on LEMIS 2013-2014 (never used for training or tuning).")

    st.subheader("How it works")
    steps = [
        ("1 · Align", "Knowledge modelling", "Map LEMIS ↔ CITES codes: product forms, units, "
                                             "source/purpose codes, countries."),
        ("2 · Features", "Feature engineering", "Deviation from peer norms: quantity vs peers, "
                                                "route rarity, unusual purpose/source."),
        ("3 · Supervised", "XGBoost ensemble", "Learns from LEMIS outcomes: refused / seized / "
                                               "abandoned / re-exported vs cleared."),
        ("4 · Unsupervised", "Peer z-score · Isolation Forest · K-Means",
         "Label-free anomaly scores, re-fitted on CITES itself."),
        ("5 · Hybrid", "Stacked meta-learner", "Combines all four signals into one risk score "
                                               "and review tier, with reasons."),
    ]
    cols = st.columns(5)
    for col, (t, tech, desc) in zip(cols, steps):
        col.markdown(f"**{t}**  \n<span style='color:{BLAZE};font-size:0.8rem;font-weight:600'>"
                     f"{tech.upper()}</span>  \n{desc}", unsafe_allow_html=True)

    st.subheader("What a flag means - and does not mean")
    st.info(
        "A flag means *this record's pattern resembles shipments that historically drew "
        "enforcement action, or deviates sharply from normal trade for its species* - i.e. it "
        "is worth an analyst's limited attention. It is **not** evidence that a law was broken. "
        "LEMIS reflects US enforcement; CITES is global, mostly permitted trade, so absolute "
        "probabilities do not transfer - the system ranks CITES records by inspection budget "
        "instead.")

    ds = load_json(config.DATASETS_JSON)
    if not ds:
        return
    # every value below is read from the raw files by wildguard.datasets, so a rebuild on a new
    # release updates this section; the table mirrors section 3.1 of the final report
    le, ci = ds["lemis"], ds["cites"]
    (s0, s1), (t0, t1) = ci["scored_years"], (config.TEST_YEARS[0], config.TEST_YEARS[-1])
    st.subheader("The data")
    st.markdown(
        "| | LEMIS (US Fish & Wildlife Service) | CITES Trade Database |\n|---|---|---|\n"
        "| Content | Every wildlife import declared to USFWS | Trade in CITES-listed species, "
        "reported by all Parties |\n"
        f"| Version | [Zenodo record {le['record']}](https://doi.org/{le['doi']}), v{le['version']} "
        f"(published {le['published']}) | Full download v{ci['version']} (UNEP-WCMC)"
        + (f", extracted {ci['extracted']}" if ci["extracted"] else "") + " |\n"
        f"| Size | {le['line_items']:,} import line items in {le['shipments']:,} shipments | "
        f"{ci['records']:,} records ({ci['years'][0]}–{ci['years'][1]}) |\n"
        f"| Period used | {le['years'][0]}–{le['years'][1]} (all) | Animal records {s0}–{s1} for "
        f"scoring ({ci['animal_records_scored']:,}); {t0}–{t1} US-reported records for the "
        "alignment check |\n"
        "| Labels | `action` (Cleared/Refused), `disposition` (Cleared/Seized/Abandoned/"
        "Re-exported) | none |\n"
        "| Declared value | yes (USD) | no |")
    st.markdown("**What the two original datasets look like**")
    st.caption(f"LEMIS · {le['file']}, as published by EcoHealth Alliance (the 'cleaned' in the "
               "name is theirs) · first five rows")
    st.dataframe(load_head("lemis"), hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.Column(help=h) for c, h in lemis_fields().items()})
    st.caption(f"CITES · the first five animal records from {s0}–{s1}, in download order "
               f"({', '.join(ci['head_files'])} of {ci['files']} files)")
    st.dataframe(load_head("cites"), hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.Column(help=h) for c, h in CITES_FIELDS.items()})
    st.caption("Values are shown as they appear in the source files: LEMIS writes missing values as "
               "NA; CITES leaves them blank and replaces confidential permit numbers with "
               "anonymised IDs. Hover a column name for its meaning.")


def flag_card(r):
    st.markdown(tier_badge(r["tier"]), unsafe_allow_html=True)
    qty = f"{r['quantity_std']:,.0f}" if r["quantity_std"] >= 10 else f"{r['quantity_std']:,.3g}"
    st.markdown(
        f"#### {CLASS_ICON.get(r['class'], '🌿')} {r['taxon_display']}\n"
        f"**{qty} {r['unit_group']}** · {r['product_code']} · "
        f"source **{SOURCE.get(r['source'], r['source'])}** · purpose "
        f"**{PURPOSE.get(r['purpose'], r['purpose'])}**  \n"
        f"{cname(r['exporter'])} → {cname(r['importer'])}"
        + (f" (origin {cname(r['origin'])})" if r["origin"] != r["exporter"] else "")
        + f" · {int(r['year'])} · CITES Appendix {r['appendix']} · "
          f"reported by {'importer' if r['reporter_type'] == 'I' else 'exporter'}")
    st.markdown("**Why flagged**")
    for reason in str(r["reasons"]).split(" | "):
        if reason:
            st.markdown(f"- {reason}")
    top = max(1 - r["risk_percentile"], 0.0001)
    st.markdown(
        f"**Historical basis:** the LEMIS-trained model gives this pattern a "
        f"{prob(r['score_supervised'])} enforcement-action likelihood under US conditions; "
        f"overall it is in the **top {top:.2%}** of CITES records ranked for {int(r['year'])}.")
    c = st.columns(4)
    c[0].metric("Supervised (LEMIS pattern)", prob(r["score_supervised"]))
    c[1].metric("Peer deviation", f"{r['score_peer']:.0%} pct.")
    c[2].metric("Isolation Forest", f"{r['score_iforest']:.0%} pct.")
    c[3].metric("K-Means distance", f"{r['score_kmeans']:.0%} pct.")


def page_queue():
    st.title("CITES review queue")
    q = load_queue()
    need(q, "Review queue")
    cs = load_json(config.OUTPUTS_DIR / "cites_scoring_summary.json")
    excl = f" {cs['excluded_known_enforcement']:,} records already marked as confiscated (source I) " \
           "or law-enforcement trade (purpose L) are excluded from ranking." if cs else ""
    st.caption("All priority records (top 1% per year) plus a random sample of the rest, "
               "CITES 2015-2024 animal trade. Select a row to see its flag card." + excl)
    st.caption("A flag is a statistical prompt for human review, not a finding of wrongdoing: "
               "CITES records permitted trade, and most flagged records will be lawful.")
    f1, f2, f3, f4, f5 = st.columns([1, 1, 1.2, 1.2, 1.2])
    tier = f1.selectbox("Tier", ["priority", "review", "routine", "all"])
    years = f2.multiselect("Year", sorted(q["year"].unique()), placeholder="All years")
    cls = f3.multiselect("Class", sorted(q["class"].dropna().unique()), format_func=with_icon,
                         placeholder="All classes")
    exp = f4.multiselect("Exporter", sorted(q["exporter"].unique()), format_func=cname,
                         placeholder="All exporters")
    imp = f5.multiselect("Importer", sorted(q["importer"].unique()), format_func=cname,
                         placeholder="All importers")
    search = st.text_input("Search taxon", placeholder="e.g. Psittacus, Manis, Crocodylus")
    v = q
    if tier != "all":
        v = v[v["tier"] == tier]
    if years:
        v = v[v["year"].isin(years)]
    if cls:
        v = v[v["class"].isin(cls)]
    if exp:
        v = v[v["exporter"].isin(exp)]
    if imp:
        v = v[v["importer"].isin(imp)]
    if search:
        v = v[v["taxon_display"].str.contains(search, case=False, na=False)]
    v = v.sort_values("score_hybrid", ascending=False).reset_index(drop=True)
    st.markdown(f"**{len(v):,} records**")
    cols = ["year", "class", "taxon_display", "product_code", "quantity_std", "unit_group", "purpose",
            "source", "exporter", "importer", "appendix", "risk_percentile"]
    show = v[cols + (["tier"] if tier == "all" else [])].head(5000)
    show = show.assign(**{"class": show["class"].astype(object).map(with_icon)}).rename(
        columns={"taxon_display": "taxon", "product_code": "term", "quantity_std": "quantity",
                 "unit_group": "unit", "risk_percentile": "risk pct."})
    event = st.dataframe(
        show, use_container_width=True, hide_index=True, height=340,
        on_select="rerun", selection_mode="single-row",
        column_config={"risk pct.": st.column_config.ProgressColumn(
            "risk pct.", min_value=0.0, max_value=1.0, format="%.4f"),
            "quantity": st.column_config.NumberColumn(format="%.4g"),
            "year": st.column_config.NumberColumn(format="%d")})
    rows = event.selection.rows if event and event.selection else []
    if len(v):
        st.divider()
        if not rows:
            st.caption("Showing the highest-ranked record - select any row to inspect it instead.")
        flag_card(v.iloc[rows[0] if rows else 0])


def score_one(record, mode):
    if mode == "CITES":
        stack, fb = load_model("cites_stack.joblib"), load_model("feature_builder_cites.joblib")
    else:
        stack, fb = load_model("full_stack.joblib"), load_model("feature_builder_lemis.joblib")
    need(stack, "Model files")
    F = fb.transform(pd.DataFrame([record]))
    s = {k: float(v[0]) for k, v in stack.score(F).items()}
    X, _ = to_model_matrix(F, stack.features, stack.categories)
    reasons = explain_rows(stack.xgb, X, F)[0]
    return s, reasons, stack


def page_check():
    st.title("Check a shipment")
    demo = st.query_params.get("example") == "tortoise"     # deep link used for the demo video
    mode = st.radio("Record type", ["CITES", "LEMIS"], horizontal=True, index=1 if demo else 0,
                    format_func=lambda m: {
        "CITES": "CITES trade record (transfer model, global norms)",
        "LEMIS": "US import declaration with declared value (full LEMIS model)"}[m])
    presets = {
        "Indian star tortoise - real LEMIS case (2000), actually seized": dict(
            genus="Geochelone", species="elegans", cls="Reptilia", product="live", qty=2.0,
            unit="count", purpose="P", source="W", exporter="MY", origin="MY", appendix="II",
            value=800.0, port="SF", month=2, year=2000),
        "Routine: captive-bred ball pythons, commercial": dict(
            genus="Python", species="regius", cls="Reptilia", product="live", qty=200.0,
            unit="count", purpose="T", source="C", exporter="GH", origin="GH", appendix="II",
            value=2400.0, port="NY", month=5, year=2014),
    }
    names = ["(blank)"] + list(presets)
    preset = st.selectbox("Start from an example", names, index=1 if demo else 0)
    p = presets.get(preset, dict(genus="", species="", cls="Reptilia", product="live", qty=1.0,
                                 unit="count", purpose="T", source="W", exporter="ID",
                                 origin="ID", appendix="II", value=100.0, port="LA", month=6,
                                 year=2020))
    st.caption("The tortoise case is in LEMIS training years (2000), so its score is an "
               "in-sample illustration, not a test result.")
    with st.form("shipment"):
        a, b, c, d = st.columns(4)
        genus = a.text_input("Genus", p["genus"])
        species = b.text_input("Species", p["species"])
        cls = c.selectbox("Class", CLASSES, index=CLASSES.index(p["cls"]), format_func=with_icon)
        appendix = d.selectbox("CITES appendix", ["I", "II", "III", "not_listed"],
                               index=["I", "II", "III", "not_listed"].index(p["appendix"]))
        a, b, c, d = st.columns(4)
        product = a.selectbox("Product form", PRODUCT_GROUPS, index=PRODUCT_GROUPS.index(p["product"]))
        qty = b.number_input("Quantity", min_value=0.0, value=p["qty"])
        unit = c.selectbox("Unit", ["count", "kg", "litre", "m2", "m3", "m"],
                           index=["count", "kg", "litre", "m2", "m3", "m"].index(p["unit"]))
        year = d.number_input("Year", 2000, 2030, p["year"])
        a, b, c, d = st.columns(4)
        purposes = list(PURPOSE)
        purpose = a.selectbox("Purpose", purposes, index=purposes.index(p["purpose"]),
                              format_func=lambda k: f"{k} - {PURPOSE[k]}")
        sources = [k for k in SOURCE if k != "unknown"]
        source = b.selectbox("Source", sources, index=sources.index(p["source"]),
                             format_func=lambda k: f"{k} - {SOURCE[k]}")
        countries = sorted(COUNTRY)
        exporter = c.selectbox("Exporter", countries, index=countries.index(p["exporter"]),
                               format_func=cname)
        origin = d.selectbox("Origin", countries, index=countries.index(p["origin"]),
                             format_func=cname)
        value, port, month = None, None, None
        if mode == "LEMIS":
            a, b, c, _ = st.columns(4)
            value = a.number_input("Declared value (USD)", min_value=0.0, value=p["value"])
            port = b.text_input("US entry port code", p["port"])
            month = c.number_input("Month", 1, 12, p["month"])
        submitted = st.form_submit_button("Score this shipment", type="primary")
    if not (submitted or demo):
        return
    rec = {"year": int(year), "month": float(month) if month else np.nan, "class": cls,
           "genus": genus.strip().lower(), "taxon": f"{genus} {species}".strip().lower(),
           "taxon_display": f"{genus} {species}".strip(), "product_group": product,
           "unit_group": unit, "quantity_std": float(qty), "value_usd": value,
           "purpose": purpose, "source": source, "origin": origin, "exporter": exporter,
           "importer": "US", "port": port or "unknown", "appendix": appendix,
           "origin_region": region(origin), "exporter_region": region(exporter)}
    s, reasons, stack = score_one(rec, mode)
    if mode == "CITES" and (source == "I" or purpose == "L"):
        st.info("This record is already enforcement material (confiscated specimen or "
                "law-enforcement purpose), so WildGuard does not rank it for review.")
        return
    if mode == "CITES":
        cs = load_json(config.OUTPUTS_DIR / "cites_scoring_summary.json")
        need(cs, "CITES scoring summary")
        pct = float(np.interp(s["hybrid"], cs["hybrid_score_quantiles"], np.linspace(0, 1, 1001)))
        tier = "priority" if pct > 0.99 else "review" if pct > 0.95 else "routine"
        headline = (f"In the **top {max(1 - pct, 0.001):.1%}** of CITES records ranked for review, "
                    "2015-2024.")
    else:
        t_low, t_high = stack.bands
        tier = ("priority" if s["hybrid"] >= t_high else
                "review" if s["hybrid"] >= t_low else "auto-clear")
        headline = (f"Estimated enforcement-action probability **{s['hybrid']:.1%}** "
                    f"(US conditions; LEMIS base rate ≈ 1.7%).")
    st.markdown(tier_badge(tier), unsafe_allow_html=True)
    st.markdown(headline)
    st.markdown("**Why**" if reasons else "No feature pushed the risk up.")
    for r in reasons:
        st.markdown(f"- {r}")
    c = st.columns(4)
    c[0].metric("Supervised (LEMIS pattern)", prob(s["supervised"]))
    c[1].metric("Peer deviation", f"{s['peer']:.0%} pct.")
    c[2].metric("Isolation Forest", f"{s['iforest']:.0%} pct.")
    c[3].metric("K-Means distance", f"{s['kmeans']:.0%} pct.")


def page_evaluation():
    st.title("Does it work?")
    t = load_test_scores()
    ev = load_json(config.OUTPUTS_DIR / "lemis_evaluation.json")
    need(t, "Held-out scores")
    st.markdown(
        "Ground truth exists only in LEMIS, so the rigorous test is there: models were trained "
        "on 2000-2010, tuned on 2011-2012 and are scored here on **2013-2014 shipments they never "
        "saw**, against what US officers actually did (refused / seized / abandoned / "
        "re-exported vs cleared).")
    a, b = st.columns([1, 2])
    budget = a.slider("Your team can inspect this share of shipments", 0.5, 30.0, 5.0, 0.5,
                      format="%.1f%%") / 100
    subset = a.radio("Shipments", ["All", "Commercial purpose only"], horizontal=True)
    v = t if subset == "All" else t[t["purpose"] == "T"]
    y = v["label"].to_numpy()
    models = [("transfer:hybrid", "WildGuard (deployed transfer hybrid)", GREEN),
              ("full:hybrid", "Hybrid + LEMIS-only value/port features", BLAZE),
              ("baseline:listing_rule", "Rule: most endangered + wild first", SLATE),
              ("baseline:random", "Random inspection", GREY)]
    k = max(1, int(round(budget * len(v))))
    res = {}
    for col, label, _ in models:
        top = np.argsort(-v[col].to_numpy(), kind="stable")[:k]
        res[col] = (y[top].sum() / max(y.sum(), 1), y[top].mean())
    caught, prec = res["transfer:hybrid"]
    a.metric("Enforcement cases caught", f"{caught:.0%}")
    a.caption(f"random inspection: {res['baseline:random'][0]:.0%} · listing rule: "
              f"{res['baseline:listing_rule'][0]:.0%}")
    a.metric("Real cases per 100 inspections", f"{prec * 100:.0f}")
    a.caption(f"random inspection: {res['baseline:random'][1] * 100:.1f}")
    a.caption(f"{len(v):,} held-out shipment line items, {y.sum():,} enforcement cases "
              f"({y.mean():.1%}).")
    fig = go.Figure()
    for col, label, color in models:
        order = np.argsort(-v[col].to_numpy(), kind="stable")
        cum = np.cumsum(y[order]) / max(y.sum(), 1)
        idx = np.unique(np.linspace(0, len(v) - 1, 400).astype(int))
        fig.add_trace(go.Scatter(x=(idx + 1) / len(v) * 100, y=cum[idx] * 100, name=label,
                                 line=dict(color=color, width=2),
                                 hovertemplate="inspect %{x:.1f}% → catch %{y:.1f}%<extra>" + label + "</extra>"))
    fig.add_vline(x=budget * 100, line=dict(color="#898781", width=1))
    fig.update_xaxes(title="Share of shipments inspected (%)", range=[0, 30])
    fig.update_yaxes(title="Enforcement cases caught (%)", range=[0, 100])
    b.markdown("**Cases caught per inspection budget**")
    b.plotly_chart(plotly_layout(fig, 420), use_container_width=True)

    st.subheader("Model comparison (held-out 2013-2014)")
    tab = load_csv("lemis_evaluation_table.csv")
    if tab is not None:
        names = {
            "full:hybrid": "Hybrid - full features", "transfer:hybrid": "Hybrid - transfer (deployed)",
            "full:supervised": "XGBoost - full", "transfer:supervised": "XGBoost - transfer",
            "full:random_forest": "Random Forest - full",
            "full:logistic_regression": "Logistic Regression - full",
            "full:unsup_mean": "Unsupervised mean - full",
            "transfer:unsup_mean": "Unsupervised mean - transfer",
            "full:iforest": "Isolation Forest - full", "transfer:iforest": "Isolation Forest - transfer",
            "full:peer": "Peer deviation - full", "transfer:peer": "Peer deviation - transfer",
            "full:kmeans": "K-Means distance - full", "transfer:kmeans": "K-Means distance - transfer",
            "baseline:listing_rule": "Listing rule (most endangered first)",
            "baseline:random": "Random inspection"}
        tab = tab.rename(columns={"Unnamed: 0": "model"})
        tab["model"] = tab["model"].map(names).fillna(tab["model"])
        tab = tab.set_index("model")[["pr_auc", "roc_auc", "f1", "precision@1%", "recall@5%",
                                      "lift@1%", "inspect_for_80%_recall"]]
        tab.columns = ["PR-AUC", "ROC-AUC", "F1", "precision @ top 1%", "recall @ top 5%",
                       "lift @ top 1%", "inspect to catch 80%"]
        st.dataframe(tab.sort_values("PR-AUC", ascending=False).style.format("{:.3f}"),
                     use_container_width=True, height=598)
        st.caption("PR-AUC and recall at a fixed inspection budget are the headline metrics - at a "
                   "1.7% positive rate, accuracy would reward a model that clears everything.")
    if ev:
        st.subheader("Confidence-aware triage")
        bands = ev["confidence_bands"]["transfer"]["test"]
        st.dataframe(pd.DataFrame(bands).T.rename(columns={
            "share_of_records": "share of shipments", "share_of_positives": "share of enforcement cases",
            "positive_rate": "enforcement rate in band"}).style.format("{:.1%}"),
            use_container_width=True)
        st.caption("Thresholds were fixed on 2011-2012: auto-clear keeps ≥95% of validation "
                   "enforcement cases above it; priority = top 2% of validation scores.")
    ds = load_json(config.OUTPUTS_DIR / "domain_shift.json")
    if ds:
        st.subheader("Does it transfer? Simulated domain shift")
        A = ds["A_simulated_shift"]
        labels = {"in_domain_reference": "In-domain reference (no shift)",
                  "naive_transfer": "Naive transfer (source norms)",
                  "renormalised_transfer": "Re-normalised norms (no labels)",
                  "importance_weighted": "Importance-weighted (no labels)",
                  "few_shot_1pct_target_labels": "+ 1% of target labels",
                  "few_shot_5pct_target_labels": "+ 5% of target labels",
                  "target_oracle": "Target oracle (all target labels)"}
        rows = pd.DataFrame({labels[k]: v for k, v in A.items() if k in labels}).T
        rows = rows[["pr_auc", "roc_auc", "recall@5%", "precision@1%"]]
        rows.columns = ["PR-AUC", "ROC-AUC", "recall @ top 5%", "precision @ top 1%"]
        st.dataframe(rows.style.format("{:.3f}"), use_container_width=True)
        st.caption(f"Trained without any shipments exported from {A['setup']['target_region']}, "
                   f"tested on {A['setup']['n_target_test']:,} held-out {A['setup']['target_region']} "
                   "shipments.")


def page_patterns():
    st.title("Risk patterns in CITES trade")
    st.caption("Share of each group's records in the priority tier (top 1% per year). "
               "Uniform risk would put every group at 1%.")
    dims = {"Taxonomic class": ("by_class.csv", "class"),
            "Product form": ("by_product_group.csv", "product_group"),
            "Declared purpose": ("by_purpose.csv", "purpose"),
            "Source": ("by_source.csv", "source"),
            "Exporter (≥2,000 records)": ("by_exporter.csv", "exporter"),
            "Importer (≥2,000 records)": ("by_importer.csv", "importer"),
            "Taxon (≥500 records)": ("by_taxon_display.csv", "taxon_display"),
            "Year": ("by_year.csv", "year")}
    dim = st.selectbox("Break down by", list(dims))
    fname, col = dims[dim]
    d = load_csv(f"cites_aggregates/{fname}")
    need(d, "CITES aggregates")
    if col == "year":
        d = d.sort_values("year")
    else:
        d = d.head(20).iloc[::-1]
    labels = d[col].astype(str)
    if col in ("exporter", "importer"):
        labels = d[col].map(cname)
    elif col == "purpose":
        labels = d[col].map(lambda k: PURPOSE.get(k, k))
    elif col == "source":
        labels = d[col].map(lambda k: SOURCE.get(k, k))
    elif col == "class":
        labels = d[col].map(with_icon)
    fig = go.Figure(go.Bar(
        x=d["priority_share"] * 100, y=labels, orientation="h", marker_color=GREEN,
        customdata=d["records"], width=0.6,
        hovertemplate="%{y}: %{x:.2f}% flagged of %{customdata:,} records<extra></extra>"))
    fig.add_vline(x=1, line=dict(color="#898781", width=1))
    fig.update_xaxes(title="Records in priority tier (%)")
    st.plotly_chart(plotly_layout(fig, 40 + 26 * len(d), showlegend=False),
                    use_container_width=True)

    u = load_csv("cites_unodc_consistency.csv")
    val = load_json(config.OUTPUTS_DIR / "cites_validation.json")
    if u is not None and val:
        st.subheader("Consistency with independent seizure data (UNODC WWCR 2024)")
        rho = val["unodc_consistency"]["spearman_vs_seizure_to_trade_ratio"]["wildguard_priority_share"]
        st.markdown(
            f"Groups that are over-represented in global seizures relative to their legal trade "
            f"volume should be flagged more often. Spearman ρ = **{rho['spearman_rho']}** "
            f"(p = {rho['p_value']}, {len(u)} species groups, CITES 2015-2021).")
        # label only the groups that stand apart; the rest are named on hover and in the table
        show = (u["wildguard_priority_share"] >= 0.02) | (u["seizure_to_trade_ratio"] >= 5)
        pos = np.where(u["group"] == "Carnivores", "bottom center", "top center")
        fig = go.Figure(go.Scatter(
            x=u["seizure_to_trade_ratio"], y=u["wildguard_priority_share"] * 100,
            mode="markers+text", text=u["group"].where(show, ""), textposition=pos,
            marker=dict(size=10, color=GREEN, line=dict(color=KHAKI, width=2)),
            customdata=np.c_[u["group"], u["cites_records"]],
            hovertemplate="%{customdata[0]}<br>seizure/trade ratio %{x:.2f}<br>flagged %{y:.2f}%"
                          "<br>%{customdata[1]:,} CITES records<extra></extra>"))
        fig.add_hline(y=1, line=dict(color="#898781", width=1))
        fig.update_xaxes(type="log", title="UNODC seizure share ÷ CITES legal-trade share (log)",
                         tickvals=[0.3, 1, 3, 10, 30, 100, 300, 1000],
                         ticktext=["0.3", "1", "3", "10", "30", "100", "300", "1,000"])
        fig.update_yaxes(type="log", title="WildGuard priority share (%, log)",
                         tickvals=[0.2, 0.5, 1, 2, 5, 10, 20, 50],
                         ticktext=["0.2", "0.5", "1", "2", "5", "10", "20", "50"])
        st.plotly_chart(plotly_layout(fig, 460, showlegend=False), use_container_width=True)
        st.caption("Directional consistency only - not a precision/recall claim. Grey line = "
                   "uniform flagging (1%). Hover a point for its group.")
        tbl = u[["group", "cites_records", "unodc_seizure_share", "cites_trade_share",
                 "seizure_to_trade_ratio", "wildguard_priority_share"]].rename(columns={
            "cites_records": "CITES records", "unodc_seizure_share": "UNODC seizure share",
            "cites_trade_share": "CITES trade share", "seizure_to_trade_ratio": "seizure/trade ratio",
            "wildguard_priority_share": "WildGuard priority share"})
        st.dataframe(tbl.style.format({"CITES records": "{:,.0f}", "UNODC seizure share": "{:.1%}",
                                       "CITES trade share": "{:.2%}", "seizure/trade ratio": "{:.2f}",
                                       "WildGuard priority share": "{:.2%}"}),
                     hide_index=True, use_container_width=True)


PAGES = {"Overview": page_overview, "Review queue": page_queue,
         "Check a shipment": page_check, "Does it work?": page_evaluation,
         "Risk patterns": page_patterns}
SLUGS = {"overview": 0, "queue": 1, "check": 2, "evaluation": 3, "patterns": 4}
with st.sidebar:
    st.markdown("## 🐘 WildGuard")
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed",
                      index=SLUGS.get(st.query_params.get("page", ""), 0))
    st.caption("NUS-ISS PRS Practice Module · Group 37")
    ds = load_json(config.DATASETS_JSON)
    if ds:
        st.caption(f"Data: LEMIS (USFWS / EcoHealth Alliance, Zenodo v{ds['lemis']['version']}); "
                   f"CITES Trade Database full download v{ds['cites']['version']} "
                   "(UNEP-WCMC for the CITES Secretariat).")
PAGES[choice]()
