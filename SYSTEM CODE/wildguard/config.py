"""Paths and global constants for the WildGuard pipeline.

All paths can be overridden with the WILDGUARD_DATA_DIR environment variable so the
pipeline runs from any checkout location.
"""
import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]          # .../SYSTEM CODE
REPO_DIR = CODE_DIR.parent
DATA_DIR = Path(os.environ.get("WILDGUARD_DATA_DIR", REPO_DIR / "DATA"))

RAW_LEMIS = DATA_DIR / "raw" / "lemis" / "lemis_2000_2014_cleaned.csv"
RAW_CITES_GLOB = str(DATA_DIR / "raw" / "cites" / "trade_db_*.csv")
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"

MODELS_DIR = CODE_DIR / "models"
OUTPUTS_DIR = CODE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

LEMIS_PARQUET = PROCESSED_DIR / "lemis_aligned.parquet"
CITES_PARQUET = PROCESSED_DIR / "cites_aligned.parquet"
CITES_SCORED_PARQUET = PROCESSED_DIR / "cites_scored.parquet"

# Temporal split on LEMIS (shipment year). Test years are never used for fitting
# models, peer-group statistics, thresholds or the hybrid combiner.
TRAIN_YEARS = range(2000, 2011)     # 2000-2010
VALID_YEARS = range(2011, 2013)     # 2011-2012  (threshold + hybrid-combiner fitting)
TEST_YEARS = range(2013, 2015)      # 2013-2014  (held-out evaluation)

# CITES scoring window: animal records from the years after LEMIS coverage ends.
CITES_SCORE_YEARS = range(2015, 2025)

RANDOM_STATE = 37

for _d in (PROCESSED_DIR, SAMPLES_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
