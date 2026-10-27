"""Feature construction for the unlabelled CITES stream.

Peer-group norms (median quantity per genus/product/unit, route shares, purpose/source
shares) are re-learned from CITES itself - no labels are needed for that - so a feature
such as qty_z means "deviation from what is normal *in global CITES trade*", the same
relative meaning it had in LEMIS. This "re-normalisation" is the first domain-adaptation
step (its effect is measured in domain_shift.py).
"""
import joblib
import pandas as pd

from . import config
from .features import FeatureBuilder

CITES_COLS = ["shipment_id", "year", "class", "order_name", "family", "genus", "taxon",
              "taxon_display", "product_code", "product_group", "unit_group", "quantity_std",
              "value_usd", "purpose", "source", "origin", "exporter", "importer", "appendix",
              "reporter_type", "origin_region", "exporter_region"]

CITES_FEATURES_PARQUET = config.PROCESSED_DIR / "cites_features.parquet"
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
    lemis_fb = joblib.load(config.MODELS_DIR / "feature_builder_lemis.joblib")
    fb = FeatureBuilder(use_value=False).fit(df, vocab_from=lemis_fb)
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
