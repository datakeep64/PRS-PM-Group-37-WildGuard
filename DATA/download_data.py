"""Download the raw datasets WildGuard is built on into DATA/raw/.

    python DATA/download_data.py

LEMIS is fetched automatically from Zenodo (record 3565869, v1.1.0, CC-BY-4.0, ~1 GB).
The CITES full database download must be fetched manually (see below), because the
CITES Trade Database site serves it through an interactive page.
"""
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"
LEMIS_FILES = ["lemis_2000_2014_cleaned.csv", "lemis_codes.csv", "lemis_metadata.csv"]
ZENODO = "https://zenodo.org/api/records/3565869/files/{}/content"


def main():
    lemis_dir = RAW / "lemis"
    lemis_dir.mkdir(parents=True, exist_ok=True)
    for name in LEMIS_FILES:
        dest = lemis_dir / name
        if dest.exists():
            print(f"exists: {dest}")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(ZENODO.format(name), dest)

    cites_dir = RAW / "cites"
    cites_dir.mkdir(parents=True, exist_ok=True)
    if not list(cites_dir.glob("trade_db_*.csv")):
        print(
            "\nCITES: download the 'Full CITES Trade Database Download' (version 2026.1) from\n"
            "  https://trade.cites.org  ->  'Full database download'\n"
            f"unzip it, and place the 59 trade_db_*.csv files in:\n  {cites_dir}\n")
    else:
        print(f"CITES CSVs present: {len(list(cites_dir.glob('trade_db_*.csv')))} files")


if __name__ == "__main__":
    main()
