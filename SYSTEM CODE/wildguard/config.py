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
RAW_LEMIS_RECORD = DATA_DIR / "raw" / "lemis" / "zenodo_record.json"   # written by download_data.py
RAW_CITES_GLOB = str(DATA_DIR / "raw" / "cites" / "trade_db_*.csv")
# release notes shipped in the CITES zip; they state the version and extraction date
RAW_CITES_INFO_GLOB = str(DATA_DIR / "raw" / "cites" / "*supplementary information*.docx")
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"

MODELS_DIR = CODE_DIR / "models"
OUTPUTS_DIR = CODE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

LEMIS_PARQUET = PROCESSED_DIR / "lemis_aligned.parquet"
CITES_PARQUET = PROCESSED_DIR / "cites_aligned.parquet"
CITES_SCORED_PARQUET = PROCESSED_DIR / "cites_scored.parquet"
DATASETS_JSON = OUTPUTS_DIR / "datasets.json"

# Year roles on LEMIS (shipment year; SYSTEM CODE/DESIGN_QUESTIONS.md B1). The design was
# chosen on the selection year; the test years are used once, for the reported results; the
# deployed model is then refitted on all years, because performance falls with the gap
# between the training years and the scored year (notebook 13).
TRAIN_YEARS = range(2000, 2011)     # 2000-2010  models and peer norms
SELECTION_YEAR = 2011               # design choices (notebooks 08-14)
HELD_BACK_YEAR = 2012               # confidence bands of the LEMIS triage view
VALID_YEARS = range(2011, 2013)     # 2011-2012, kept for code that still reads it
TEST_YEARS = range(2013, 2015)      # 2013-2014  the one test
DEPLOY_YEARS = range(2000, 2015)    # 2000-2014  the deployed model

# CITES scoring window: animal records from the years after LEMIS coverage ends.
CITES_SCORE_YEARS = range(2015, 2025)

RANDOM_STATE = 37

for _d in (PROCESSED_DIR, SAMPLES_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
