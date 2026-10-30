"""Source datasets - version, size and first rows, for the app's Overview page.

Everything is read from the raw files themselves, so rebuilding on a new LEMIS or CITES
release updates what the app shows without a code change:

    SYSTEM CODE/outputs/datasets.json   version, size and years of each dataset
    DATA/samples/lemis_head.csv         first 5 rows of the LEMIS CSV, verbatim
    DATA/samples/cites_head.csv         first 5 animal records of the scoring years, in download
                                        order, verbatim

The LEMIS version comes from raw/lemis/zenodo_record.json (written by DATA/download_data.py)
and is only accepted if the CSV's checksum matches it; the CITES version comes from the
supplementary information document shipped in the CITES zip.

Run:  python -m wildguard.datasets
"""
import glob
import hashlib
import json
import re
import zipfile
from pathlib import Path

import duckdb
import pandas as pd

from . import config

HEAD_ROWS = 5


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def lemis_release():
    if not config.RAW_LEMIS_RECORD.exists():
        raise SystemExit(f"{config.RAW_LEMIS_RECORD} not found - run: python DATA/download_data.py")
    rec = json.loads(config.RAW_LEMIS_RECORD.read_text())
    if md5(config.RAW_LEMIS) != rec["md5"]:
        raise SystemExit(f"{config.RAW_LEMIS} does not match Zenodo version {rec['version']} - "
                         "run: python DATA/download_data.py")
    return rec


def cites_release():
    docs = glob.glob(config.RAW_CITES_INFO_GLOB)
    if not docs:
        raise SystemExit("CITES supplementary information .docx not found in DATA/raw/cites - "
                         "unzip the whole CITES download there (see DATA/download_data.py)")
    with zipfile.ZipFile(docs[0]) as z:
        text = re.sub(r"<[^>]+>", "", z.read("word/document.xml").decode())
    version = re.search(r"Version (\d{4}\.\d+)", text)
    extracted = re.search(r"extracted at the shipment level on (\d{1,2} \w+ \d{4})", text)
    if not version:
        raise SystemExit(f"no 'Version yyyy.n' found in {docs[0]}")
    return version.group(1), extracted.group(1) if extracted else None


def cites_files():
    """trade_db_1.csv, trade_db_2.csv, ... in numeric order (the download is oldest first)."""
    return sorted(glob.glob(config.RAW_CITES_GLOB),
                  key=lambda p: int(re.search(r"(\d+)\.csv$", p).group(1)))


def head(path, nrows=HEAD_ROWS):
    # strings exactly as written in the file ("NA", blanks and leading zeros kept)
    return pd.read_csv(path, dtype=str, keep_default_na=False, nrows=nrows)


def cites_head(con, files, s0, s1):
    """First HEAD_ROWS animal records (Class not blank) of the scoring years, in download order.
    DuckDB finds the files that hold any; pandas then reads those in order, verbatim."""
    hits = {Path(f).name for (f,) in con.execute(f"""
        SELECT DISTINCT filename
        FROM read_csv('{config.RAW_CITES_GLOB}', all_varchar=true, header=true, filename=true)
        WHERE Class IS NOT NULL AND CAST(Year AS INT) BETWEEN {s0} AND {s1}
    """).fetchall()}
    parts, sources, n = [], [], 0
    for f in files:
        if Path(f).name not in hits:
            continue
        d = head(f, nrows=None)
        d = d[(d["Class"] != "") & pd.to_numeric(d["Year"]).between(s0, s1)].head(HEAD_ROWS - n)
        parts.append(d)
        sources.append(Path(f).name)
        n += len(d)
        if n == HEAD_ROWS:
            break
    return pd.concat(parts), sources


def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    out = {}

    rec = lemis_release()
    n, shipments, y0, y1 = con.execute(f"""
        SELECT count(*), count(DISTINCT control_number),
               min(TRY_CAST(shipment_year AS INT)), max(TRY_CAST(shipment_year AS INT))
        FROM read_csv('{config.RAW_LEMIS}', all_varchar=true, header=true, nullstr='NA')
    """).fetchone()
    out["lemis"] = dict(rec, file=config.RAW_LEMIS.name, line_items=n, shipments=shipments,
                        years=[y0, y1])
    head(config.RAW_LEMIS).to_csv(config.SAMPLES_DIR / "lemis_head.csv", index=False)
    print(f"LEMIS v{rec['version']}: {n:,} line items in {shipments:,} shipments, {y0}-{y1}")

    version, extracted = cites_release()
    files = cites_files()
    s0, s1 = config.CITES_SCORE_YEARS[0], config.CITES_SCORE_YEARS[-1]
    n, y0, y1, scored = con.execute(f"""
        SELECT count(*), min(CAST(Year AS INT)), max(CAST(Year AS INT)),
               count(*) FILTER (WHERE Class IS NOT NULL AND CAST(Year AS INT) BETWEEN {s0} AND {s1})
        FROM read_csv('{config.RAW_CITES_GLOB}', all_varchar=true, header=true)
    """).fetchone()
    rows, sources = cites_head(con, files, s0, s1)
    out["cites"] = dict(version=version, extracted=extracted, files=len(files), records=n,
                        years=[y0, y1], animal_records_scored=scored, scored_years=[s0, s1],
                        head_files=sources)
    rows.to_csv(config.SAMPLES_DIR / "cites_head.csv", index=False)
    print(f"CITES v{version} (extracted {extracted}): {n:,} records in {len(files)} files, "
          f"{y0}-{y1}; {scored:,} animal records {s0}-{s1}")

    config.DATASETS_JSON.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
