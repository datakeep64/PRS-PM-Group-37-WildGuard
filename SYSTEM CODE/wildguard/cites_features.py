"""Feature construction for the unlabelled CITES stream.

CITES records are described against the peer-group norms (median quantity per
genus/product/unit, route shares, purpose/source shares) learned from LEMIS's training years:
the same norms the model was trained with. Re-learning the norms from CITES itself was the
earlier design; tested on held-out LEMIS populations it was significantly worse in two of six
tests, so it was rejected (notebook 08; DATA/ALIGNING_LEMIS_AND_CITES.md section 1.2, step 11).
"""
import joblib
import pandas as pd

from . import config

CITES_COLS = ["shipment_id", "year", "class", "order_name", "family", "genus", "taxon",
              "taxon_display", "product_code", "product_group", "unit_group", "quantity_std",
              "value_usd", "purpose", "source", "origin", "exporter", "importer", "appendix",
              "reporter_type", "origin_region", "exporter_region", "rule_d", "export_permit"]

CITES_FEATURES_PARQUET = config.PROCESSED_DIR / "cites_features.parquet"
# the builder applied to CITES records (the LEMIS one), saved under its own name for the app
CITES_BUILDER = config.MODELS_DIR / "feature_builder_cites.joblib"


def load_cites(years, extra_filter=None):
    yrs = list(years)
    df = pd.read_parquet(config.CITES_PARQUET, columns=CITES_COLS,
                         filters=[("year", ">=", min(yrs)), ("year", "<=", max(yrs))])
    if extra_filter is not None:
        df = df[extra_filter(df)].reset_index(drop=True)
    return df


def build(years=config.CITES_SCORE_YEARS, force=False):
    if CITES_FEATURES_PARQUET.exists() and CITES_BUILDER.exists() and not force:
        return
    df = load_cites(years)
    fb = joblib.load(config.MODELS_DIR / "feature_builder_lemis.joblib")
    joblib.dump(fb, CITES_BUILDER, compress=3)
    parts = []
    for y in sorted(df["year"].unique()):
        parts.append(fb.transform(df[df["year"] == y]))
        print(f"  CITES features {y}: {len(parts[-1]):,} rows")
    pd.concat(parts, ignore_index=True).to_parquet(CITES_FEATURES_PARQUET, index=False)


def load_features(columns=None):
    return pd.read_parquet(CITES_FEATURES_PARQUET, columns=columns)


if __name__ == "__main__":
    build(force=True)
