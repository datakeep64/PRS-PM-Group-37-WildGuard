"""Step 1 - acquire & align.

Reads the raw LEMIS CSV and the 59 raw CITES CSVs, maps both onto the shared schema defined
in schema_alignment.py and writes two parquet files:

    DATA/processed/lemis_aligned.parquet   one row per LEMIS import line item, with label
    DATA/processed/cites_aligned.parquet   one row per CITES animal trade record, 2000-2024

DuckDB is used so the ~6 GB of raw CSV is streamed rather than loaded into memory.

Run:  python -m wildguard.prepare_data            (both datasets)
      python -m wildguard.prepare_data cites      (CITES only)
"""
import time

import duckdb
import pandas as pd

from . import config
from .schema_alignment import (CITES_TERM_TO_GROUP, CITES_UNIT, LEMIS_DESCRIPTION_TO_GROUP,
                               LEMIS_UNIT, PURPOSE_CODES, REGION_OF, SOURCE_CODES)


def _register_lookups(con):
    con.register("lemis_desc_map", pd.DataFrame(
        list(LEMIS_DESCRIPTION_TO_GROUP.items()), columns=["code", "product_group"]))
    con.register("cites_term_map", pd.DataFrame(
        list(CITES_TERM_TO_GROUP.items()), columns=["map_term", "map_group"]))
    con.register("lemis_unit_map", pd.DataFrame(
        [(k, g, m) for k, (g, m) in LEMIS_UNIT.items()], columns=["code", "unit_group", "mult"]))
    con.register("cites_unit_map", pd.DataFrame(
        [(k, g, m) for k, (g, m) in CITES_UNIT.items() if k], columns=["map_unit", "map_unit_group", "map_mult"]))
    con.register("region_map", pd.DataFrame(
        list(REGION_OF.items()), columns=["iso2", "region"]))


def _cites_view(con):
    con.execute(f"""
        CREATE OR REPLACE VIEW cites_raw AS
        SELECT * FROM read_csv('{config.RAW_CITES_GLOB}', all_varchar=true, header=true)
    """)


def build_cites_appendix_lookup(con):
    """Most recent CITES appendix per species and per genus, used to label LEMIS taxa as
    CITES-listed or not (LEMIS itself has no appendix field)."""
    con.execute("""
        CREATE OR REPLACE TABLE appendix_species AS
        SELECT lower(Taxon) AS taxon, arg_max(Appendix, CAST(Year AS INT)) AS appendix
        FROM cites_raw WHERE Class IS NOT NULL AND Taxon NOT LIKE '% spp.%'
        GROUP BY 1
    """)
    con.execute("""
        CREATE OR REPLACE TABLE appendix_genus AS
        SELECT lower(Genus) AS genus, arg_max(Appendix, CAST(Year AS INT)) AS appendix
        FROM cites_raw WHERE Class IS NOT NULL AND Genus IS NOT NULL
        GROUP BY 1
    """)


def prepare_lemis(con):
    t = time.time()
    src = sorted(SOURCE_CODES)
    pur = sorted(PURPOSE_CODES)
    con.execute(f"""
        CREATE OR REPLACE VIEW lemis_raw AS
        SELECT * FROM read_csv('{config.RAW_LEMIS}', all_varchar=true, header=true,
                               nullstr=['NA', 'non-standard value'])
    """)
    con.execute(f"""
        COPY (
        WITH b AS (
          SELECT
            'LEMIS' AS dataset,
            control_number AS shipment_id,
            CAST(shipment_year AS INT) AS year,
            CAST(substr(shipment_date, 6, 2) AS INT) AS month,
            CASE WHEN class = 'Actinopterygii' THEN 'Actinopteri' ELSE class END AS class,
            lower(genus) AS genus,
            lower(genus || ' ' || species) AS taxon,
            upper(genus[1]) || lower(genus[2:]) || ' ' || coalesce(lower(species), 'spp.') AS taxon_display,
            coalesce(specific_name || ' / ', '') || coalesce(generic_name, '') AS common_name,
            description AS product_code,
            coalesce(dm.product_group, 'other') AS product_group,
            coalesce(um.unit_group, 'other') AS unit_group,
            TRY_CAST(quantity AS DOUBLE) * coalesce(um.mult, 1.0) AS quantity_std,
            TRY_CAST(value AS DOUBLE) AS value_usd,
            CASE WHEN purpose IN {tuple(pur)} THEN purpose ELSE 'unknown' END AS purpose,
            CASE WHEN source  IN {tuple(src)} THEN source  ELSE 'unknown' END AS source,
            coalesce(country_origin, country_imp_exp, 'XX') AS origin,
            coalesce(country_imp_exp, country_origin, 'XX') AS exporter,
            'US' AS importer,
            coalesce(port, 'unknown') AS port,
            CASE WHEN action = 'R' OR disposition IN ('S', 'A', 'R') THEN 1 ELSE 0 END AS label,
            coalesce(disposition, 'unknown') AS disposition
          FROM lemis_raw l
          LEFT JOIN lemis_desc_map dm ON l.description = dm.code
          LEFT JOIN lemis_unit_map um ON l.unit = um.code
          WHERE import_export = 'I' AND shipment_year IS NOT NULL
        )
        SELECT b.*,
               coalesce(s.appendix, g.appendix, 'not_listed') AS appendix,
               coalesce(ro.region, 'Unknown') AS origin_region,
               coalesce(re.region, 'Unknown') AS exporter_region
        FROM b
        LEFT JOIN appendix_species s ON b.taxon = s.taxon
        LEFT JOIN appendix_genus g ON b.genus = g.genus
        LEFT JOIN region_map ro ON b.origin = ro.iso2
        LEFT JOIN region_map re ON b.exporter = re.iso2
        ) TO '{config.LEMIS_PARQUET}' (FORMAT PARQUET)
    """)
    n, pos = con.execute(
        f"SELECT count(*), avg(label) FROM '{config.LEMIS_PARQUET}'").fetchone()
    print(f"LEMIS aligned: {n:,} rows, positive rate {pos:.4f}  ({time.time() - t:.0f}s)")


def prepare_cites(con):
    t = time.time()
    src = sorted(SOURCE_CODES)
    pur = sorted(PURPOSE_CODES)
    con.execute(f"""
        COPY (
        WITH b AS (
          SELECT
            'CITES' AS dataset,
            Id AS shipment_id,
            CAST(Year AS INT) AS year,
            CAST(NULL AS INT) AS month,
            Class AS class,
            lower(Genus) AS genus,
            lower(Taxon) AS taxon,
            Taxon AS taxon_display,
            CAST(NULL AS VARCHAR) AS common_name,
            Term AS product_code,
            coalesce(tm.map_group, 'other') AS product_group,
            coalesce(um.map_unit_group, CASE WHEN Unit IS NULL THEN 'count' ELSE 'other' END) AS unit_group,
            TRY_CAST(Quantity AS DOUBLE) * coalesce(um.map_mult, 1.0) AS quantity_std,
            CAST(NULL AS DOUBLE) AS value_usd,
            CASE WHEN Purpose IN {tuple(pur)} THEN Purpose ELSE 'unknown' END AS purpose,
            CASE WHEN Source  IN {tuple(src)} THEN Source  ELSE 'unknown' END AS source,
            coalesce(Origin, Exporter, 'XX') AS origin,
            coalesce(Exporter, 'XX') AS exporter,
            coalesce(Importer, 'XX') AS importer,
            CAST(NULL AS VARCHAR) AS port,
            CAST(NULL AS INT) AS label,
            CAST(NULL AS VARCHAR) AS disposition,
            Appendix AS appendix,
            "Reporter.type" AS reporter_type,
            "Order" AS order_name,
            Family AS family
          FROM cites_raw c
          LEFT JOIN cites_term_map tm ON c.Term = tm.map_term
          LEFT JOIN cites_unit_map um ON c.Unit = um.map_unit
          WHERE Class IS NOT NULL AND CAST(Year AS INT) BETWEEN 2000 AND 2024
        )
        SELECT b.*,
               coalesce(ro.region, 'Unknown') AS origin_region,
               coalesce(re.region, 'Unknown') AS exporter_region
        FROM b
        LEFT JOIN region_map ro ON b.origin = ro.iso2
        LEFT JOIN region_map re ON b.exporter = re.iso2
        ) TO '{config.CITES_PARQUET}' (FORMAT PARQUET)
    """)
    n = con.execute(f"SELECT count(*) FROM '{config.CITES_PARQUET}'").fetchone()[0]
    print(f"CITES aligned: {n:,} animal records 2000-2024  ({time.time() - t:.0f}s)")


def main(which="all"):
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    _register_lookups(con)
    _cites_view(con)
    if which in ("all", "lemis"):
        build_cites_appendix_lookup(con)
        prepare_lemis(con)
    if which in ("all", "cites"):
        prepare_cites(con)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
