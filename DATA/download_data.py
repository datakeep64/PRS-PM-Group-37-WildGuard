"""Download the raw datasets WildGuard is built on into DATA/raw/.

    python DATA/download_data.py

LEMIS is fetched automatically from the latest version of its Zenodo record (v1.1.0 when
this was written, CC-BY-4.0, ~1 GB). Files already present are checked against that version's
checksums, and the version is recorded in raw/lemis/zenodo_record.json, which the app shows.
The CITES full database download must be fetched manually (see below), because the
CITES Trade Database site serves it through an interactive page.
"""
import hashlib
import json
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"
LEMIS_FILES = ["lemis_2000_2014_cleaned.csv", "lemis_codes.csv", "lemis_metadata.csv"]
ZENODO_LATEST = "https://zenodo.org/api/records/3565869/versions/latest"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    lemis_dir = RAW / "lemis"
    lemis_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ZENODO_LATEST) as r:
        rec = json.load(r)
    version = rec["metadata"]["version"]
    files = {f["key"]: f for f in rec["files"]}
    print(f"LEMIS: Zenodo record {rec['id']}, version {version}")
    for name in LEMIS_FILES:
        dest = lemis_dir / name
        if dest.exists():
            print(f"exists: {dest}")
        else:
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(files[name]["links"]["self"], dest)
        if md5(dest) != files[name]["checksum"].removeprefix("md5:"):
            raise SystemExit(f"{dest} is not the file in Zenodo version {version}; "
                             "delete it and rerun to download that version.")
    (lemis_dir / "zenodo_record.json").write_text(json.dumps({
        "record": rec["id"], "version": version, "doi": rec["doi"],
        "published": rec["metadata"]["publication_date"],
        "md5": files[LEMIS_FILES[0]]["checksum"].removeprefix("md5:")}, indent=2))

    cites_dir = RAW / "cites"
    cites_dir.mkdir(parents=True, exist_ok=True)
    if not list(cites_dir.glob("trade_db_*.csv")):
        print(
            "\nCITES: download the latest 'Full CITES Trade Database Download' (version 2026.1\n"
            "when this was written) from\n"
            "  https://trade.cites.org  ->  'Full database download'\n"
            "and unzip all of it - the trade_db_*.csv files and 'Full data download_\n"
            "supplementary information.docx', which states the version - into:\n"
            f"  {cites_dir}\n")
    else:
        print(f"CITES CSVs present: {len(list(cites_dir.glob('trade_db_*.csv')))} files")
        if not list(cites_dir.glob("*supplementary information*.docx")):
            print("CITES: 'Full data download_supplementary information.docx' is missing - "
                  "copy it from the CITES zip; it states the version the app shows.")


if __name__ == "__main__":
    main()
