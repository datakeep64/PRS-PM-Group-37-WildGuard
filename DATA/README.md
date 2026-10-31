# DATA

WildGuard uses two public, real-world wildlife-trade datasets. They complement each other:
LEMIS has **labels without global scale**; CITES has **global scale without labels**.

To view these datasets, access the following links :

https://trade.cites.org/

https://github.com/eveskew/lemis



**Before changing how either dataset is loaded or aligned, read
[`USING_LEMIS_AND_CITES.md`](USING_LEMIS_AND_CITES.md)** (the publishers' rules for both datasets)
**and [`ALIGNING_LEMIS_AND_CITES.md`](ALIGNING_LEMIS_AND_CITES.md)** (what WildGuard does with them, written
from the EDA notebooks in `SYSTEM CODE/notebooks/`: what each dataset needs before use, the
crosswalk, the checks every pipeline should pass, and the known deviations in the current code).

|                 | LEMIS (US Fish & Wildlife Service)                                                      | CITES Trade Database                                                                                  |
| --------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| What it records | Every wildlife import declared to USFWS                                                 | Trade in CITES-listed species reported by all Parties                                                 |
| Version used    | Zenodo record 3565869, **v1.1.0** (Dec 2019), 2000–2014                          | **Full download v2026.1** (files dated June 2026)                                               |
| Size            | 5,512,706 import line items (2,088,676 shipments), 974 MB                               | 29,170,051 records (1975–2026), 4.5 GB in 59 CSVs                                                    |
| Used here       | Animals with a taxonomic class, 2000–2014, labelled (5,107,119 line items)             | Animal records 2015–2024 (8.25M), of which 5,465,862 are ranked (each trade once, rule D); 2013–2014 US-reported records for the alignment check |
| Labels          | `action` (Cleared/Refused) and `disposition` (Cleared/Seized/Abandoned/Re-exported) | none (the CITES Illegal Trade Database is access-restricted)                                          |
| Declared value  | yes (`value`, USD)                                                                    | no                                                                                                    |
| Licence / terms | CC-BY-4.0                                                                               | free for use with citation (see below)                                                                |
| Downloaded      | 27 Oct 2026                                                                             | 2026 (v2026.1)                                                                                        |

**Label used:** a line item is an *enforcement case* (positive) if it was refused
(`action = R`) or ended seized, abandoned or re-exported (`disposition ∈ {S, A, R}`).
Positive rate: 1.74% overall (1.59% in the 2013–2014 test years).

## Folder layout

```
DATA/
├── download_data.py        fetches LEMIS from Zenodo; instructions for CITES
├── raw/                    (not in git) raw CSVs - see download_data.py
│   ├── lemis/              the LEMIS files + zenodo_record.json (version, checksum)
│   └── cites/              the whole CITES zip: trade_db_*.csv + supplementary information .docx
├── processed/              (not in git) aligned parquet files rebuilt by the pipeline
├── reference/              LEMIS codebook (country, port, purpose, source, unit, product codes)
└── samples/                small extracts committed so the data can be inspected and the app runs
    ├── cites_review_queue.parquet     all priority-tier CITES records + 40k others, scored, with reasons
    ├── lemis_head.csv                 first 5 rows of the raw LEMIS CSV, verbatim (app Overview)
    └── cites_head.csv                 first 5 animal records 2015-2024 of the raw CITES download, verbatim (app Overview)
```

The raw files exceed GitHub's 100 MB file limit, so they are not committed. To rebuild
everything from scratch:

```bash
python DATA/download_data.py                 # LEMIS automatically; CITES instructions printed
bash "SYSTEM CODE/run_pipeline.sh"
```

**Moving to a new release.** The app's Overview takes each dataset's version, size and first
rows from the raw files (`SYSTEM CODE/wildguard/datasets.py`), not from the code. For a new
CITES annual release, empty `raw/cites/`, unzip the new download there in full and rerun the
pipeline; for a new LEMIS version on Zenodo, delete `raw/lemis/` and rerun `download_data.py`
and the pipeline. The models, review queue and Overview then all reflect the same release -
update the report's data section to match.

## Schema alignment (the knowledge model)

Both datasets are mapped onto one shared schema by `SYSTEM CODE/wildguard/schema_alignment.py`:

| Field           | LEMIS                                                  | CITES                                            | Shared form                                               |
| --------------- | ------------------------------------------------------ | ------------------------------------------------ | --------------------------------------------------------- |
| Product form    | 3-letter code (`LIV`, `SKI`, `IVC` …, 94 codes) | English term (`live`, `skins` …, 101 terms) | 31 product groups                                         |
| Unit            | code (`NO`, `KG`, `GM`, `ML` …)               | word (`kg`, `g`, blank = number …)          | count / kg / litre / m / m² / m³, quantities converted  |
| Source, purpose | CITES letter codes                                     | CITES letter codes                               | identical, unmapped values → `unknown`                 |
| Countries       | ISO-3166 alpha-2                                       | ISO-3166 alpha-2                                 | identical, plus world region                              |
| Taxonomy        | genus, species, class                                  | taxon, genus, family, order, class               | genus, class (*Actinopterygii* ↔ *Actinopteri*)      |
| CITES appendix  | not recorded                                           | recorded                                         | looked up for LEMIS taxa from CITES (species, then genus) |

Coverage after mapping: 0.5% of LEMIS and 0.03% of CITES animal records fall into the
residual `other` product group; 0.002% / 0.04% have an unmapped unit.

## Citations

CITES Trade Database statistics:

> CITES Trade Database 2026. Compiled by UNEP-WCMC for the CITES Secretariat. Available at:
> trade.cites.org. Last accessed 27 October 2026.

Full database download:

> Full CITES Trade Database Download. Version 2026.1. Compiled by UNEP-WCMC, Cambridge, UK for
> the CITES Secretariat, Geneva, Switzerland. Available at: trade.cites.org.

LEMIS:

> Eskew, E. A., White, A. M., Ross, N., Smith, K. M., Smith, K. F., Rodríguez, J. P., Zambrana-Torrelio, C., Karesh, W. B., & Daszak, P. (2019).
> *United States LEMIS wildlife trade data curated by EcoHealth Alliance* (Version 1.1.0) [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.3565869

> Eskew, E. A., White, A. M., Ross, N., Smith, K. M., Smith, K. F., Rodríguez, J. P., Zambrana-Torrelio, C., Karesh, W. B., & Daszak, P. (2020).
> United States wildlife and wildlife product imports from 2000–2014. *Scientific Data*, 7, 22.
> https://doi.org/10.1038/s41597-020-0354-5
