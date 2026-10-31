"""Shared setup for the design experiments (notebooks 04 and 05).

Both notebooks train the model the same way, so they share: LEMIS with the CITES listing in force
in the shipment year, the year split, the model features, the training procedure, and the CITES
records a model is applied to. Needs a DuckDB connection from eda_setup.connect() (for the
cites_animals view).
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

from eda_setup import config
from wildguard.features import SHARED_FEATURES, FeatureBuilder, to_model_matrix
from wildguard.models import XGB_PARAMS

# classes in LEMIS that are not animals (plants, algae, fungi, bacteria; notebook 03 section 5)
NON_ANIMAL = {"Magnoliopsida", "Liliopsida", "Polypodiopsida", "Cycadopsida", "Pinopsida",
              "Ulvophyceae", "Phaeophyceae", "Gammaproteobacteria", "Chytridiomycetes",
              "Agaricomycetes"}
LEMIS_COLS = ["shipment_id", "year", "class", "genus", "taxon", "product_group", "unit_group",
              "quantity_std", "purpose", "source", "origin", "exporter", "label",
              "origin_region", "exporter_region"]
CITES_FEATURES_PARQUET = config.PROCESSED_DIR / "cites_features.parquet"


def load_lemis(con, sample=1.0):
    """Aligned LEMIS line items with the CITES listing in force in the shipment year.

    For each species and shipment year: the most recent appendix CITES recorded for that species
    in or before that year (Appendix N ignored; the strictest when a year has several), else the
    same for its genus. Adds `appendix` (replacing the pipeline's latest-year lookup), `animal`,
    `sp_listed` (species-level listing) and `gen_listed` (species- or genus-level listing).
    """
    df = pd.read_parquet(config.LEMIS_PARQUET, columns=LEMIS_COLS)
    if sample < 1:
        keep = df["shipment_id"].drop_duplicates().sample(frac=sample, random_state=37)
        df = df[df["shipment_id"].isin(keep)].reset_index(drop=True)
    con.execute("""
        CREATE OR REPLACE TABLE sp_app AS
        SELECT lower(Taxon) AS taxon, Year AS year, min(Appendix) AS app FROM cites_animals
        WHERE Appendix <> 'N' AND Genus IS NOT NULL AND Taxon NOT LIKE '%hybrid%'
        GROUP BY 1, 2;
        CREATE OR REPLACE TABLE gen_app AS
        SELECT lower(Genus) AS genus, Year AS year, min(Appendix) AS app FROM cites_animals
        WHERE Appendix <> 'N' AND Genus IS NOT NULL
        GROUP BY 1, 2;
    """)
    con.register("lemis_keys", df[["taxon", "genus", "year"]].drop_duplicates())
    lookup = con.execute("""
        SELECT k.taxon, k.genus, k.year, s.app AS sp_app, g.app AS gen_app
        FROM lemis_keys k
        ASOF LEFT JOIN sp_app s ON k.taxon = s.taxon AND k.year >= s.year
        ASOF LEFT JOIN gen_app g ON k.genus = g.genus AND k.year >= g.year
    """).df()
    df = df.merge(lookup, on=["taxon", "genus", "year"], how="left")
    df["appendix"] = df["sp_app"].fillna(df["gen_app"]).fillna("not_listed")
    df["animal"] = df["class"].notna() & ~df["class"].isin(NON_ANIMAL)
    df["sp_listed"] = df["sp_app"].notna()
    df["gen_listed"] = df["sp_listed"] | df["gen_app"].notna()
    return df


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
    pos = y[train].sum()
    model = xgb.XGBClassifier(**(XGB_PARAMS | {"random_state": seed}), n_estimators=1500,
                              early_stopping_rounds=60,
                              scale_pos_weight=float(np.sqrt((train.sum() - pos) / pos)))
    model.fit(X[train], y[train], eval_set=[(X[val], y[val])], verbose=False)
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
    (ALIGNING_LEMIS_AND_CITES.md section 1.2). The aligned file still writes pre-Convention as O and
    reintroduction as N, so these are mapped to LEMIS's P and Y (ALIGNING section 2.1).
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
    c["source"] = c["source"].astype(str).replace({"O": "P"})
    c["purpose"] = c["purpose"].astype(str).replace({"N": "Y"})
    return c, to_model_matrix(c, SHARED_FEATURES, cats)[0]


def top_within_year(score, years, frac=0.01):
    """WildGuard ranks CITES records within each year: mask of each year's top `frac`."""
    out = np.zeros(len(score), bool)
    for y in np.unique(years):
        idx = np.flatnonzero(years == y)
        k = max(1, int(round(len(idx) * frac)))
        out[idx[np.argsort(-score[idx], kind="stable")[:k]]] = True
    return out
