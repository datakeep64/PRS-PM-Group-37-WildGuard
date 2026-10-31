"""The alignment checks every pipeline run should pass (DATA/ALIGNING_LEMIS_AND_CITES.md
section 2.4), run on the aligned files written by prepare_data.

Each check prints PASS, FAIL or INFO with its evidence; the details go to
outputs/alignment_checks.json. A failure does not stop the pipeline, so that one run shows
every problem at once, but a run with a FAIL should not be used for results.

Populations (ALIGNING sections 1-2):
    LEMIS training   animals with a class, training years
    CITES scored     2015-2024, rule D, without source I and purpose L
    CITES exporter   2015-2024, exporter-reported (counts and shares)

Run:  python -m wildguard.check_alignment      (after prepare_data)
"""
import json

import duckdb

from . import config
from .datasets import cites_files, cites_release, lemis_release
from .schema_alignment import (CITES_TERM_TO_GROUP, CITES_UNIT, LEMIS_DESCRIPTION_TO_GROUP,
                               LEMIS_UNIT)

L, C = f"'{config.LEMIS_PARQUET}'", f"'{config.CITES_PARQUET}'"
T0, T1 = min(config.TRAIN_YEARS), max(config.TRAIN_YEARS)
S0, S1 = min(config.CITES_SCORE_YEARS), max(config.CITES_SCORE_YEARS)
LEMIS_TRAIN = f"SELECT * FROM {L} WHERE animal AND year BETWEEN {T0} AND {T1}"
CITES_SCORED = (f"SELECT * FROM {C} WHERE year BETWEEN {S0} AND {S1} AND rule_d "
                "AND source <> 'I' AND purpose <> 'L'")
CITES_EXPORTER = f"SELECT * FROM {C} WHERE year BETWEEN {S0} AND {S1} AND reporter_type = 'E'"

UNSEEN_FIELDS = ["class", "product_group", "unit_group", "purpose", "source", "appendix",
                 "origin_region", "exporter_region"]
UNSEEN_MAX_SHARE = 1e-4      # "anything beyond a handful of records is a crosswalk error"
ONE_SIDED_MIN_SHARE = 1e-3   # a group with 0.1% on one side and nothing on the other
US_OVERLAP_MAX_TVD = 0.10    # notebook 03 (X11): about 0.06 for purpose and 0.07 for source


def q(con, sql):
    return con.execute(sql).df()


def check_files(con):
    rec = lemis_release()                     # stops the run if the LEMIS checksum does not match
    version, extracted = cites_release()
    files = cites_files()
    return "PASS", f"LEMIS v{rec['version']} (checksum ok); CITES v{version}, extracted " \
                   f"{extracted}, {len(files)} files", \
        {"lemis_version": rec["version"], "cites_version": version, "cites_extracted": extracted,
         "cites_files": len(files)}


def check_codes(con):
    """Codes outside the crosswalk's vocabularies end up as 'other' or 'unknown'."""
    out = {}
    out["lemis_description_unmapped"] = q(con, f"""
        SELECT product_code, count(*) AS n FROM {L} WHERE animal
          AND product_code NOT IN ({",".join(repr(k) for k in LEMIS_DESCRIPTION_TO_GROUP)})
        GROUP BY 1 ORDER BY 2 DESC""").to_dict("records")
    out["cites_term_unmapped"] = q(con, f"""
        SELECT product_code, count(*) AS n FROM ({CITES_SCORED})
        WHERE product_code NOT IN ({",".join(repr(k) for k in CITES_TERM_TO_GROUP)})
        GROUP BY 1 ORDER BY 2 DESC""").to_dict("records")
    out["lemis_unit_other"] = int(q(con, f"SELECT count(*) AS n FROM {L} WHERE animal AND unit_group = 'other'").n[0])
    out["cites_unit_other"] = int(q(con, f"SELECT count(*) AS n FROM ({CITES_SCORED}) WHERE unit_group = 'other'").n[0])
    out["lemis_unit_codes"] = sorted(LEMIS_UNIT)
    out["cites_unit_words"] = sorted(k for k in CITES_UNIT if k)
    n_scored = int(q(con, f"SELECT count(*) AS n FROM ({CITES_SCORED})").n[0])
    n_bad = sum(r["n"] for r in out["cites_term_unmapped"]) + out["cites_unit_other"]
    status = "PASS" if n_bad <= UNSEEN_MAX_SHARE * n_scored else "FAIL"
    return status, f"{n_bad:,} scored CITES records with an unmapped term or unit; " \
                   f"{sum(r['n'] for r in out['lemis_description_unmapped']):,} LEMIS training " \
                   f"line items with an unmapped description", out


def check_unseen(con):
    out, worst = {}, 0.0
    n_scored = int(q(con, f"SELECT count(*) AS n FROM ({CITES_SCORED})").n[0])
    for f in UNSEEN_FIELDS:
        rows = q(con, f"""
            SELECT c.{f} AS value, count(*) AS n FROM ({CITES_SCORED}) c
            WHERE c.{f} NOT IN (SELECT DISTINCT {f} FROM ({LEMIS_TRAIN}) WHERE {f} IS NOT NULL)
            GROUP BY 1 ORDER BY 2 DESC""")
        out[f] = rows.to_dict("records")
        worst = max(worst, rows.n.sum() / n_scored)
    status = "PASS" if worst <= UNSEEN_MAX_SHARE else "FAIL"
    detail = "; ".join(f + ": " + ", ".join(f"{r['value']} ({r['n']:,})" for r in v[:4])
                       for f, v in out.items() if v) or "none"
    return status, f"CITES values never seen in LEMIS training: {detail}", out


def check_group_shares(con):
    out, one_sided = {}, []
    for f in ("product_group", "class"):
        rows = q(con, f"""
            WITH l AS (SELECT {f} AS g, count(*) / sum(count(*)) OVER () AS lemis
                       FROM ({LEMIS_TRAIN}) GROUP BY 1),
                 c AS (SELECT {f} AS g, count(*) / sum(count(*)) OVER () AS cites
                       FROM ({CITES_EXPORTER}) GROUP BY 1)
            SELECT coalesce(l.g, c.g) AS g, coalesce(lemis, 0) AS lemis,
                   coalesce(cites, 0) AS cites
            FROM l FULL JOIN c ON l.g = c.g ORDER BY lemis DESC""")
        out[f] = rows.round(5).to_dict("records")
        for r in out[f]:
            if max(r["lemis"], r["cites"]) >= ONE_SIDED_MIN_SHARE and min(r["lemis"], r["cites"]) == 0:
                one_sided.append(f"{f}={r['g']} (LEMIS {r['lemis']:.2%}, CITES {r['cites']:.2%})")
    out["one_sided"] = one_sided
    status = "FAIL" if any(s.startswith("product_group") for s in one_sided) else "PASS"
    return status, "groups on one side only: " + (", ".join(one_sided) or "none"), out


def check_value_coverage(con):
    rows = q(con, f"""SELECT year, avg(coalesce(value_usd > 0, false)::INT) AS share_with_value
                      FROM {L} WHERE animal GROUP BY 1 ORDER BY 1""")
    missing = rows.loc[rows.share_with_value < 0.01, "year"].tolist()
    return "INFO", f"years without declared value: {missing}", rows.round(4).to_dict("records")


def check_higher_taxa(con):
    blank = int(q(con, f"SELECT count(*) AS n FROM {C} WHERE genus IS NULL").n[0])
    match = q(con, f"""
        SELECT avg((genus IN (SELECT DISTINCT genus FROM {C}))::INT) AS lemis_genus_found_in_cites
        FROM ({LEMIS_TRAIN}) WHERE genus IS NOT NULL""").iat[0, 0]
    both = q(con, f"""SELECT (SELECT count(*) FROM {L} WHERE genus = 'scleractinia') AS lemis,
                             (SELECT count(*) FROM {C} WHERE genus = 'scleractinia') AS cites""")
    status = "PASS" if blank == 0 and both.lemis[0] > 0 and both.cites[0] > 0 else "FAIL"
    return status, f"CITES records without genus: {blank}; Scleractinia in both: LEMIS " \
                   f"{both.lemis[0]:,}, CITES {both.cites[0]:,}; LEMIS training genera found in " \
                   f"CITES: {match:.1%} of line items", \
        {"cites_blank_genus": blank, "lemis_genus_found_in_cites": float(match)}


def check_appendix_n(con):
    n_c = int(q(con, f"SELECT count(*) AS n FROM {C} WHERE appendix = 'N'").n[0])
    vals = sorted(q(con, f"SELECT DISTINCT appendix FROM {L}").appendix)
    status = "PASS" if n_c == 0 and set(vals) <= {"I", "II", "III", "genus_listed", "not_listed"} else "FAIL"
    return status, f"CITES Appendix N records: {n_c}; LEMIS appendix values: {vals}", \
        {"cites_appendix_n": n_c, "lemis_appendix_values": vals}


def check_mirror(con):
    rows = q(con, f"""SELECT reporter_type, rule_d, count(*) AS n,
                             avg((export_permit IS NOT NULL)::INT) AS with_export_permit
                      FROM {C} WHERE year BETWEEN {S0} AND {S1} GROUP BY 1, 2 ORDER BY 1, 2""")
    n_scored = int(q(con, f"SELECT count(*) AS n FROM ({CITES_SCORED})").n[0])
    exp_not_d = int(rows.loc[(rows.reporter_type == "E") & ~rows.rule_d, "n"].sum())
    permit = float((rows.n * rows.with_export_permit).sum() / rows.n.sum())
    status = "PASS" if exp_not_d == 0 and permit > 0.9 else "FAIL"
    return status, f"scored records (rule D, no source I / purpose L): {n_scored:,}; export " \
                   f"permit on {permit:.1%} of records", \
        {"by_reporter": rows.to_dict("records"), "scored": n_scored, "export_permit_share": permit}


def check_years(con):
    rows = q(con, f"SELECT year, count(*) AS n FROM {C} WHERE reporter_type = 'E' GROUP BY 1 ORDER BY 1")
    full = rows[rows.year.between(2018, 2023)].n.median()
    low = rows.loc[rows.year.between(S0, S1) & (rows.n < 0.8 * full), "year"].tolist()
    return "INFO", f"scoring years with under 80% of the 2018-2023 median: {low}", \
        rows.to_dict("records")


def check_labels(con):
    con.execute(f"""CREATE OR REPLACE VIEW lemis_raw AS SELECT * FROM read_csv('{config.RAW_LEMIS}',
                    all_varchar=true, header=true, nullstr=['NA', 'non-standard value'])""")
    rows = q(con, """SELECT action, disposition, count(*) AS n FROM lemis_raw
                     WHERE import_export = 'I' GROUP BY 1, 2 ORDER BY 3 DESC""")
    pos = int(q(con, f"SELECT sum(label) AS n FROM {L}").n[0])
    by_rule = int(rows.loc[(rows.action == "R") | rows.disposition.isin(["S", "A", "R"]), "n"].sum())
    status = "PASS" if pos == by_rule else "FAIL"
    return status, f"enforcement cases: {pos:,} in the aligned file, {by_rule:,} by the rule", \
        rows.fillna("NA").to_dict("records")


def check_us_overlap(con):
    """2013-2014: LEMIS line items of listed species vs CITES records the US reported as importer."""
    out, tvds = {}, {}
    for f in ("purpose", "source"):
        rows = q(con, f"""
            WITH l AS (SELECT {f} AS v, count(*) / sum(count(*)) OVER () AS lemis FROM {L}
                       WHERE animal AND sp_listed AND year IN (2013, 2014) GROUP BY 1),
                 c AS (SELECT {f} AS v, count(*) / sum(count(*)) OVER () AS cites FROM {C}
                       WHERE importer = 'US' AND reporter_type = 'I' AND year IN (2013, 2014) GROUP BY 1)
            SELECT coalesce(l.v, c.v) AS v, coalesce(lemis, 0) AS lemis, coalesce(cites, 0) AS cites
            FROM l FULL JOIN c ON l.v = c.v ORDER BY lemis DESC""")
        tvds[f] = float((rows.lemis - rows.cites).abs().sum() / 2)
        out[f] = rows.round(4).to_dict("records")
    out["total_variation"] = tvds
    status = "PASS" if max(tvds.values()) <= US_OVERLAP_MAX_TVD else "FAIL"
    return status, f"mix difference (total variation): purpose {tvds['purpose']:.3f}, " \
                   f"source {tvds['source']:.3f}", out


CHECKS = [("1 file identity", check_files), ("2 codes valid", check_codes),
          ("3 no unseen categories", check_unseen), ("4 group shares", check_group_shares),
          ("5 value coverage", check_value_coverage), ("6 higher taxa", check_higher_taxa),
          ("7 Appendix N", check_appendix_n), ("8 mirror records", check_mirror),
          ("9 latest years", check_years), ("10 labels", check_labels),
          ("11 US overlap", check_us_overlap)]


def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    results = {}
    for name, fn in CHECKS:
        status, summary, detail = fn(con)
        print(f"{status:4}  {name}: {summary}")
        results[name] = {"status": status, "summary": summary, "detail": detail}
    path = config.OUTPUTS_DIR / "alignment_checks.json"
    path.write_text(json.dumps(results, indent=1, default=str))
    failed = [n for n, r in results.items() if r["status"] == "FAIL"]
    print(f"{len(failed)} check(s) failed: {failed}" if failed else "all checks passed",
          f"- details in {path.name}")


if __name__ == "__main__":
    main()
