"""Shared setup for the EDA notebooks: raw-data caches, DuckDB views and chart style.

The notebooks describe the two datasets as published (DATA/raw/), before any alignment, so
they read the raw downloads rather than the pipeline's aligned files. How each file is read
follows the publishers' rules in DATA/USING_LEMIS_AND_CITES.md, cited in the notebooks as
"guide §x". No alignment rule is assumed: the notebooks' findings are what
DATA/ALIGNING_LEMIS_AND_CITES.md was written from.

Views created by connect():
  lemis_raw      the LEMIS CSV verbatim (every column text, 'NA' kept as a string)
  lemis          typed LEMIS: 'NA' and 'non-standard value' read as missing (guide §1.2),
                 year = shipment_year, label = enforcement case (guide §1.4)
  cites_raw      all 59 CITES CSVs verbatim (blank cells are NULL)
  cites          typed CITES (Year, Quantity numeric)
  cites_animals  CITES records with a Class (plants have none, guide §2.2)
  cites_window   animal records 2015-2024, the years WildGuard scores (Appendix N included,
                 so the notebooks can show it before filtering it out)
"""
import sys
from pathlib import Path

import duckdb
import matplotlib as mpl
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
from wildguard import config  # noqa: E402

LEMIS_CACHE = config.PROCESSED_DIR / "eda_lemis_raw.parquet"
CITES_CACHE = config.PROCESSED_DIR / "eda_cites_raw.parquet"
WINDOW = (2015, 2024)

# Chart style: the validated light categorical palette; each dataset keeps one colour.
LEMIS_C, CITES_C, THIRD_C = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# CITES code meanings (from the CITES Trade Database guide, Annexes 2-5)
CITES_SOURCE = {
    "W": "wild", "C": "bred in captivity", "F": "born in captivity (F1+)", "R": "ranched",
    "D": "commercially bred (App. I, registered operations)", "A": "artificially propagated (plants)",
    "I": "confiscated or seized specimen", "O": "pre-Convention", "U": "unknown",
    "X": "marine environment outside any State's jurisdiction", "Y": "assisted production (plants)",
}
CITES_PURPOSE = {
    "T": "commercial", "H": "hunting trophy", "P": "personal", "M": "medical / biomedical",
    "S": "scientific", "Z": "zoo", "Q": "circus / travelling exhibition", "B": "breeding",
    "E": "educational", "N": "reintroduction into the wild", "G": "botanic gardens",
    "L": "law enforcement / judicial / forensic",
}

pd.set_option("display.max_rows", 120)
pd.set_option("display.width", 160)
pd.set_option("display.float_format",
              lambda v: f"{v:,.0f}" if float(v).is_integer() else f"{v:,.4g}")


def build_caches(force=False):
    """Copy the raw CSVs to Parquet once (all columns as text, so nothing is re-coded)."""
    con = duckdb.connect()
    if force or not LEMIS_CACHE.exists():
        con.execute(f"""COPY (SELECT * FROM read_csv('{config.RAW_LEMIS}', all_varchar=true))
                        TO '{LEMIS_CACHE}' (FORMAT parquet)""")
    if force or not CITES_CACHE.exists():
        con.execute(f"""COPY (SELECT * FROM read_csv('{config.RAW_CITES_GLOB}', all_varchar=true,
                                                     union_by_name=true))
                        TO '{CITES_CACHE}' (FORMAT parquet)""")


def connect():
    build_caches()
    con = duckdb.connect()
    con.execute(f"""
        CREATE MACRO nz(x) AS CASE WHEN x IN ('NA', 'non-standard value') THEN NULL ELSE x END;

        CREATE VIEW lemis_raw AS SELECT * FROM '{LEMIS_CACHE}';
        CREATE VIEW lemis AS SELECT
            control_number, nz(species_code) AS species_code, nz(taxa) AS taxa,
            nz(class) AS class, nz(genus) AS genus, nz(species) AS species,
            nz(description) AS description,
            TRY_CAST(nz(quantity) AS DOUBLE) AS quantity, nz(unit) AS unit,
            TRY_CAST(nz(value) AS DOUBLE) AS value,
            nz(country_origin) AS country_origin, nz(country_imp_exp) AS country_imp_exp,
            nz(purpose) AS purpose, nz(source) AS source,
            nz(action) AS action, nz(disposition) AS disposition, nz(port) AS port,
            CAST(shipment_year AS INTEGER) AS year,
            coalesce(nz(action) = 'R' OR nz(disposition) IN ('S', 'A', 'R'), false)::INTEGER
                AS label
        FROM lemis_raw;

        CREATE VIEW cites_raw AS SELECT * FROM '{CITES_CACHE}';
        CREATE VIEW cites AS SELECT * REPLACE (CAST(Year AS INTEGER) AS Year,
                                               TRY_CAST(Quantity AS DOUBLE) AS Quantity)
        FROM cites_raw;
        CREATE VIEW cites_animals AS SELECT * FROM cites WHERE Class IS NOT NULL;
        CREATE VIEW cites_window AS SELECT * FROM cites_animals
        WHERE Year BETWEEN {WINDOW[0]} AND {WINDOW[1]};
    """)
    return con


def q(con, sql):
    """Run SQL and return a DataFrame."""
    return con.execute(sql).df()


def share(df, col="n", name="share"):
    """Add a percentage-of-total column next to a count column."""
    out = df.copy()
    out[name] = (100 * out[col] / out[col].sum()).round(2).astype(str) + "%"
    return out


def style():
    mpl.rcParams.update({
        "figure.dpi": 110, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2,
        "font.family": "sans-serif",
    })
