"""Shared setup for the design experiments (notebooks 04, 05 and 08 onward).

The experiments train the model the same way, so they share: aligned LEMIS with its population
flags, the year split, the model features, the training procedure, the CITES records a model is
applied to, and a shipment-level paired bootstrap.

Since the data step was rebuilt (branch `rework`), the listing in force and the population flags
are read from the aligned files written by wildguard/prepare_data.py instead of being derived here.
Notebooks 04 and 05 derived them themselves, with the earlier genus rule (notebook 06, G6).
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import average_precision_score

from eda_setup import config
from wildguard.features import SHARED_FEATURES, FeatureBuilder, to_model_matrix
from wildguard.models import XGB_PARAMS
from wildguard.schema_alignment import NON_ANIMAL_CLASSES as NON_ANIMAL  # noqa: F401

LEMIS_COLS = ["shipment_id", "year", "class", "genus", "taxon", "product_group", "unit_group",
              "quantity_std", "purpose", "source", "origin", "exporter", "label",
              "origin_region", "exporter_region", "appendix", "animal", "sp_listed"]
CITES_FEATURES_PARQUET = config.PROCESSED_DIR / "cites_features.parquet"


def load_lemis(con=None, sample=1.0):
    """Aligned LEMIS line items with the flags prepare_data writes: `appendix` (the listing in
    force in the shipment year), `animal` (the training population) and `sp_listed` (the species
    has its own listing in force). Adds `gen_listed` (species or genus listed). `con` is unused
    and kept for the notebooks that pass it."""
    df = pd.read_parquet(config.LEMIS_PARQUET, columns=LEMIS_COLS)
    if sample < 1:
        keep = df["shipment_id"].drop_duplicates().sample(frac=sample, random_state=37)
        df = df[df["shipment_id"].isin(keep)].reset_index(drop=True)
    df["gen_listed"] = df["appendix"] != "not_listed"
    return df


def pr_auc_bootstrap(y, scores, shipments, pairs, n_boot=1000, seed=37):
    """Paired bootstrap of PR-AUC differences, resampling shipments (line items of a shipment are
    not independent; notebook 01, L1). `scores` maps a name to a score array; `pairs` lists
    (a, b) names. Returns one row per pair: PR-AUC of a and b, and a - b with its 95% interval."""
    codes, uniq = pd.factorize(shipments)
    rng = np.random.default_rng(seed)
    diffs = {p: np.empty(n_boot) for p in pairs}
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes]
        keep = w > 0
        ap = {k: average_precision_score(y[keep], scores[k][keep], sample_weight=w[keep])
              for k in {name for p in pairs for name in p}}
        for a, b in pairs:
            diffs[(a, b)][i] = ap[a] - ap[b]
    rows = []
    for a, b in pairs:
        lo, hi = np.percentile(diffs[(a, b)], [2.5, 97.5])
        rows.append({"a": a, "b": b, "pr_auc_a": average_precision_score(y, scores[a]),
                     "pr_auc_b": average_precision_score(y, scores[b]),
                     "difference": diffs[(a, b)].mean(), "low_95": lo, "high_95": hi})
    return pd.DataFrame(rows)


def split(df):
    """The pipeline's temporal split (config.TRAIN/VALID/TEST_YEARS) as boolean masks."""
    yr = df["year"].to_numpy()
    return (np.isin(yr, list(config.TRAIN_YEARS)), np.isin(yr, list(config.VALID_YEARS)),
            np.isin(yr, list(config.TEST_YEARS)))


def build_features(df, tr):
    """Model matrix for every row. Peer norms and category sets come from all training-year rows,
    so every training population is described in the same terms."""
    F = FeatureBuilder(use_value=False).fit(df[tr]).transform(df)
    cats = to_model_matrix(F[tr], SHARED_FEATURES)[1]
    return to_model_matrix(F, SHARED_FEATURES, cats)[0], cats


def fit(X, y, train, val, seed):
    """wildguard.models.train_xgb with the seed varied: class weight from the training rows' own
    case rate, early stopping on average precision over the validation rows."""
    return fit_split(X[train], y[train], X[val], y[val], seed)


def fit_split(X_tr, y_tr, X_va, y_va, seed):
    """As fit(), for training and validation matrices built separately."""
    pos = y_tr.sum()
    model = xgb.XGBClassifier(**(XGB_PARAMS | {"random_state": seed}), n_estimators=1500,
                              early_stopping_rounds=60,
                              scale_pos_weight=float(np.sqrt((len(y_tr) - pos) / pos)))
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


# "exact counterpart" key for mirror records (notebook 05): the same year, taxon, term, unit
# group, countries, origin, purpose and source
STRICT = ["year", "taxon", "importer", "exporter", "product_code", "unit_group", "origin",
          "purpose", "source"]


def cites_records(con):
    """Table `cites_rec`: every CITES record of the features file that can be scored (Appendix N,
    source I and purpose L removed), with its row number `rid`, the fields the mirror-record rules
    use, and its export and origin permit IDs from the raw download (joined on the record Id)."""
    con.execute(f"""
        CREATE OR REPLACE TABLE cites_rec AS
        SELECT f.file_row_number AS rid, f.year, f.reporter_type, f.taxon, f.class, f.appendix,
               f.product_code, f.unit_group, f.importer, f.exporter, f.origin, f.purpose,
               f.source, f.quantity_std,
               r."Export.permit.RandomID" AS export_permit,
               r."Origin.permit.RandomID" AS origin_permit
        FROM read_parquet('{CITES_FEATURES_PARQUET}', file_row_number=true) f
        LEFT JOIN cites_raw r ON r.Id = f.shipment_id
        WHERE f.appendix <> 'N' AND f.source <> 'I' AND f.purpose <> 'L'
    """)


def queue_rids(con, rule="D"):
    """Row numbers (rid) of the CITES records a mirror-record rule puts in the review queue
    (notebook 05):
      "A"      exporter-reported only
      "C"      both reporter types
      "D"      exporter-reported, plus importer-reported records with no exact exporter-reported
               counterpart (same STRICT key): the rule chosen in notebook 05
      "D_permit"  as D, but an importer-reported record also counts as matched when the exporter
               reported the same export permit (tested and not adopted, notebook 05)
    """
    if not con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'cites_rec'").fetchone()[0]:
        cites_records(con)
    keep = {"A": "reporter_type = 'E'", "C": "TRUE",
            "D": "reporter_type = 'E' OR NOT has_e",
            "D_permit": "reporter_type = 'E' OR NOT (has_e OR has_e_permit)"}[rule]
    k = ", ".join(STRICT)
    return con.execute(f"""
        WITH e AS (SELECT DISTINCT {k} FROM cites_rec WHERE reporter_type = 'E'),
             ep AS (SELECT DISTINCT export_permit FROM cites_rec
                    WHERE reporter_type = 'E' AND export_permit IS NOT NULL)
        SELECT rid FROM (SELECT r.rid, r.reporter_type, e.year IS NOT NULL AS has_e,
                                ep.export_permit IS NOT NULL AS has_e_permit
                         FROM cites_rec r LEFT JOIN e USING ({k})
                         LEFT JOIN ep ON r.export_permit = ep.export_permit)
        WHERE {keep} ORDER BY rid
    """).df()["rid"].to_numpy()


def load_cites(cats, reporters=("E", "I"), extra=(), keep=None):
    """CITES 2015-2024 records ready to score, from the pipeline's CITES features (peer norms
    learned from CITES).

    Keeps the given reporter types; removes Appendix N, source I and purpose L
    (ALIGNING_LEMIS_AND_CITES.md section 1.2). Both datasets now use the CITES codes (ALIGNING
    section 2.1), so no recoding is needed; the features file must be rebuilt from the rebuilt
    aligned files first (wildguard/cites_features.py).
    `rid` is the record's row number in the features file, for joining back to other columns;
    `keep` (rids, e.g. from queue_rids) restricts the records further.
    Returns (records, model matrix).
    """
    cols = list(dict.fromkeys(SHARED_FEATURES + ["year", "reporter_type"] + list(extra)))
    # text columns as categoricals: 8 million rows of repeated codes would not fit in memory as text
    c = pq.read_table(CITES_FEATURES_PARQUET, columns=cols).to_pandas(strings_to_categorical=True)
    c["rid"] = np.arange(len(c))
    c = c[c["reporter_type"].isin(reporters) & (c["appendix"] != "N")
          & (c["source"] != "I") & (c["purpose"] != "L")]
    if keep is not None:
        c = c[c["rid"].isin(keep)]
    c = c.reset_index(drop=True)
    return c, to_model_matrix(c, SHARED_FEATURES, cats)[0]


def top_within_year(score, years, frac=0.01):
    """WildGuard ranks CITES records within each year: mask of each year's top `frac`."""
    out = np.zeros(len(score), bool)
    for y in np.unique(years):
        idx = np.flatnonzero(years == y)
        k = max(1, int(round(len(idx) * frac)))
        out[idx[np.argsort(-score[idx], kind="stable")[:k]]] = True
    return out
