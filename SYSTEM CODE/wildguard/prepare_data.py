"""Step 1 - acquire & align.

Reads the raw LEMIS CSV and the 59 raw CITES CSVs, maps both onto the shared schema defined
in schema_alignment.py and writes two parquet files:

    DATA/processed/lemis_aligned.parquet   one row per LEMIS import line item, with label
    DATA/processed/cites_aligned.parquet   one row per CITES animal trade record, 2000-2024,
                                           Appendix N removed

The rules follow DATA/ALIGNING_LEMIS_AND_CITES.md (section numbers below refer to it).
Records are not dropped for the training population or the review queue; they are flagged,
so that evaluation subsets and checks can still see every record:

    LEMIS  animal      animal with a class: the training population (section 2.2)
           sp_listed   the species had a CITES listing in force in the shipment year
    CITES  rule_d      the record is scored: exporter-reported, or importer-reported with no
                       exact exporter-reported counterpart (section 2.2)

DuckDB is used so the ~6 GB of raw CSV is streamed rather than loaded into memory.

Run:  python -m wildguard.prepare_data            (both datasets)
      python -m wildguard.prepare_data cites      (CITES only)
"""
import time

import duckdb
import pandas as pd

from . import config
from .schema_alignment import (CITES_TERM_TO_GROUP, CITES_UNIT, LEMIS_CLASS_BY_GENUS,
                               LEMIS_CLASS_TO_CITES, LEMIS_COUNTRY_TO_CITES,
                               LEMIS_DESCRIPTION_TO_GROUP, LEMIS_PLACEHOLDER_GENERA,
                               LEMIS_PURPOSE_TO_CITES, LEMIS_SOURCE_TO_CITES, LEMIS_UNIT,
                               NON_ANIMAL_CLASSES, PURPOSE_CODES, REGION_OF, SOURCE_CODES)

# the "exact counterpart" key of mirror rule D (section 2.2; notebook 05)
RULE_D_KEY = ["year", "taxon", "importer", "exporter", "product_code", "unit_group", "origin",
              "purpose", "source"]


def _sql_list(values):
    return "(" + ", ".join("'" + v.replace("'", "''") + "'" for v in sorted(values)) + ")"


def _sql_map(expr, mapping):
    """CASE expression mapping the values of `expr` through `mapping`, other values unchanged."""
    whens = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in mapping.items())
    return f"CASE {expr} {whens} ELSE {expr} END"


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
    con.register("class_by_genus", pd.DataFrame(
        [(c, g, to) for (c, g), to in LEMIS_CLASS_BY_GENUS.items()],
        columns=["map_class", "map_genus", "map_to"]))


def _cites_view(con):
    con.execute(f"""
        CREATE OR REPLACE VIEW cites_raw AS
        SELECT * FROM read_csv('{config.RAW_CITES_GLOB}', all_varchar=true, header=true)
    """)


def build_cites_appendix_lookup(con):
    """The CITES appendix recorded for each species and each genus in each year (Appendix N
    ignored; the strictest when a year has several), for the LEMIS 'listing in force'.

    Genus follows the shared higher-taxon convention: when CITES leaves Genus blank, the first
    word of Taxon (section 2.1), so LEMIS higher taxa such as Scleractinia find their listing."""
    con.execute("""
        CREATE OR REPLACE TABLE appendix_species AS
        SELECT lower(Taxon) AS taxon, CAST(Year AS INT) AS year, min(Appendix) AS appendix
        FROM cites_raw
        WHERE Class IS NOT NULL AND Appendix <> 'N' AND Genus IS NOT NULL
          AND Taxon NOT LIKE '%hybrid%'
        GROUP BY 1, 2
    """)
    con.execute("""
        CREATE OR REPLACE TABLE appendix_genus AS
        SELECT lower(coalesce(Genus, split_part(Taxon, ' ', 1))) AS genus,
               CAST(Year AS INT) AS year, min(Appendix) AS appendix
        FROM cites_raw
        WHERE Class IS NOT NULL AND Appendix <> 'N'
        GROUP BY 1, 2
    """)


def prepare_lemis(con):
    t = time.time()
    src, pur = _sql_list(SOURCE_CODES), _sql_list(PURPOSE_CODES)
    # input columns are qualified (r.) because the output reuses their names
    source = _sql_map("r.source", LEMIS_SOURCE_TO_CITES)
    purpose = _sql_map("r.purpose", LEMIS_PURPOSE_TO_CITES)
    origin = _sql_map("coalesce(r.country_origin, r.country_imp_exp, 'XX')", LEMIS_COUNTRY_TO_CITES)
    exporter = _sql_map("coalesce(r.country_imp_exp, r.country_origin, 'XX')", LEMIS_COUNTRY_TO_CITES)
    klass = _sql_map("r.class", LEMIS_CLASS_TO_CITES)
    con.execute(f"""
        CREATE OR REPLACE VIEW lemis_raw AS
        SELECT * FROM read_csv('{config.RAW_LEMIS}', all_varchar=true, header=true,
                               nullstr=['NA', 'non-standard value'])
    """)
    con.execute(f"""
        COPY (
        WITH r AS (
          SELECT *,
                 CASE WHEN genus IN {_sql_list(LEMIS_PLACEHOLDER_GENERA)} THEN NULL
                      ELSE lower(genus) END AS genus_known
          FROM lemis_raw
          WHERE import_export = 'I' AND shipment_year IS NOT NULL
        ),
        b AS (
          SELECT
            'LEMIS' AS dataset,
            control_number AS shipment_id,
            CAST(shipment_year AS INT) AS year,
            CAST(substr(shipment_date, 6, 2) AS INT) AS month,
            coalesce(cg.map_to, {klass}) AS class,
            genus_known AS genus,
            genus_known || ' ' || lower(species) AS taxon,
            species IS NOT NULL AND lower(species) NOT IN ('sp.', 'spp.', 'n.sp', 'spec') AS species_named,
            upper(r.genus[1]) || lower(r.genus[2:]) || ' ' || coalesce(lower(species), 'spp.') AS taxon_display,
            coalesce(specific_name || ' / ', '') || coalesce(generic_name, '') AS common_name,
            description AS product_code,
            coalesce(dm.product_group, 'other') AS product_group,
            coalesce(um.unit_group, 'other') AS unit_group,
            TRY_CAST(quantity AS DOUBLE) * coalesce(um.mult, 1.0) AS quantity_std,
            TRY_CAST(value AS DOUBLE) AS value_usd,
            CASE WHEN {purpose} IN {pur} THEN {purpose} ELSE 'unknown' END AS purpose,
            CASE WHEN {source} IN {src} THEN {source} ELSE 'unknown' END AS source,
            {origin} AS origin,
            {exporter} AS exporter,
            'US' AS importer,
            coalesce(port, 'unknown') AS port,
            CASE WHEN action = 'R' OR disposition IN ('S', 'A', 'R') THEN 1 ELSE 0 END AS label,
            coalesce(disposition, 'unknown') AS disposition,
            r.class IS NOT NULL AND r.class NOT IN {_sql_list(NON_ANIMAL_CLASSES)} AS animal
          FROM r
          LEFT JOIN lemis_desc_map dm ON r.description = dm.code
          LEFT JOIN lemis_unit_map um ON r.unit = um.code
          LEFT JOIN class_by_genus cg ON r.class = cg.map_class AND r.genus_known = cg.map_genus
        )
        -- listing in force (section 2.1): the most recent appendix recorded for the species in
        -- or before the shipment year. A record that names no species takes its genus's. A named
        -- species CITES has no record of is 'genus_listed' when its genus has one: it may be
        -- unlisted (white-tailed deer) or listed under another name (reticulated python), and
        -- taking a congener's appendix would make, for example, American mink Appendix I.
        SELECT b.*,
               CASE WHEN s.appendix IS NOT NULL THEN s.appendix
                    WHEN g.appendix IS NOT NULL AND b.species_named THEN 'genus_listed'
                    ELSE coalesce(g.appendix, 'not_listed') END AS appendix,
               s.appendix IS NOT NULL AS sp_listed,
               coalesce(ro.region, 'Unknown') AS origin_region,
               coalesce(re.region, 'Unknown') AS exporter_region
        FROM b
        ASOF LEFT JOIN appendix_species s ON b.taxon = s.taxon AND b.year >= s.year
        ASOF LEFT JOIN appendix_genus g ON b.genus = g.genus AND b.year >= g.year
        LEFT JOIN region_map ro ON b.origin = ro.iso2
        LEFT JOIN region_map re ON b.exporter = re.iso2
        ) TO '{config.LEMIS_PARQUET}' (FORMAT PARQUET)
    """)
    n, pos, n_b, n_l = con.execute(f"""
        SELECT count(*), avg(label), count(*) FILTER (WHERE animal),
               count(*) FILTER (WHERE animal AND sp_listed)
        FROM '{config.LEMIS_PARQUET}'""").fetchone()
    print(f"LEMIS aligned: {n:,} rows, positive rate {pos:.4f}; animals with a class {n_b:,}, "
          f"of which listed species {n_l:,}  ({time.time() - t:.0f}s)")


def prepare_cites(con):
    t = time.time()
    src, pur = _sql_list(SOURCE_CODES), _sql_list(PURPOSE_CODES)
    key = ", ".join(RULE_D_KEY)
    on = " AND ".join(f"b.{k} = e.{k}" for k in RULE_D_KEY)
    con.execute(f"""
        COPY (
        WITH b AS (
          SELECT
            'CITES' AS dataset,
            Id AS shipment_id,
            CAST(Year AS INT) AS year,
            CAST(NULL AS INT) AS month,
            Class AS class,
            lower(coalesce(Genus, split_part(Taxon, ' ', 1))) AS genus,
            lower(Taxon) AS taxon,
            Taxon AS taxon_display,
            CAST(NULL AS VARCHAR) AS common_name,
            Term AS product_code,
            coalesce(tm.map_group, 'other') AS product_group,
            coalesce(um.map_unit_group, CASE WHEN Unit IS NULL THEN 'count' ELSE 'other' END) AS unit_group,
            TRY_CAST(Quantity AS DOUBLE) * coalesce(um.map_mult, 1.0) AS quantity_std,
            CAST(NULL AS DOUBLE) AS value_usd,
            CASE WHEN Purpose IN {pur} THEN Purpose ELSE 'unknown' END AS purpose,
            CASE WHEN Source  IN {src} THEN Source  ELSE 'unknown' END AS source,
            coalesce(Origin, Exporter, 'XX') AS origin,
            coalesce(Exporter, 'XX') AS exporter,
            coalesce(Importer, 'XX') AS importer,
            CAST(NULL AS VARCHAR) AS port,
            CAST(NULL AS INT) AS label,
            CAST(NULL AS VARCHAR) AS disposition,
            Appendix AS appendix,
            "Reporter.type" AS reporter_type,
            "Order" AS order_name,
            Family AS family,
            "Export.permit.RandomID" AS export_permit,
            "Origin.permit.RandomID" AS origin_permit
          FROM cites_raw c
          LEFT JOIN cites_term_map tm ON c.Term = tm.map_term
          LEFT JOIN cites_unit_map um ON c.Unit = um.map_unit
          WHERE Class IS NOT NULL AND Appendix <> 'N' AND CAST(Year AS INT) BETWEEN 2000 AND 2024
        ),
        e AS (SELECT DISTINCT {key}, TRUE AS has_e FROM b WHERE reporter_type = 'E')
        SELECT b.*,
               b.reporter_type = 'E' OR e.has_e IS NULL AS rule_d,
               coalesce(ro.region, 'Unknown') AS origin_region,
               coalesce(re.region, 'Unknown') AS exporter_region
        FROM b
        LEFT JOIN e ON {on}
        LEFT JOIN region_map ro ON b.origin = ro.iso2
        LEFT JOIN region_map re ON b.exporter = re.iso2
        ) TO '{config.CITES_PARQUET}' (FORMAT PARQUET)
    """)
    n, n_w, n_d, n_q = con.execute(f"""
        SELECT count(*), count(*) FILTER (WHERE year BETWEEN 2015 AND 2024),
               count(*) FILTER (WHERE year BETWEEN 2015 AND 2024 AND rule_d),
               count(*) FILTER (WHERE year BETWEEN 2015 AND 2024 AND rule_d
                                AND source <> 'I' AND purpose <> 'L')
        FROM '{config.CITES_PARQUET}'""").fetchone()
    print(f"CITES aligned: {n:,} animal records 2000-2024 (Appendix N removed); 2015-2024: "
          f"{n_w:,}, of which rule D {n_d:,}, ranked (no source I / purpose L) {n_q:,}  "
          f"({time.time() - t:.0f}s)")


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
