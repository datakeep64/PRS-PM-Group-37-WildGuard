# Using LEMIS and CITES data correctly

This guide records how the two source datasets are meant to be read, according to the people who
publish them. It does not depend on how WildGuard is built: **any rework of the pipeline, models or
app should follow it.** How WildGuard puts the two datasets on one schema, the checks every pipeline
should pass and the status of the current code are in
[`ALIGNING_LEMIS_AND_CITES.md`](ALIGNING_LEMIS_AND_CITES.md), written from the EDA notebooks in
`SYSTEM CODE/notebooks/`.

Every rule below comes from a publisher's documentation (listed in [Sources](#sources)) or from a
count run on the files themselves; counts are given so you can re-check them.

**What can change.** LEMIS v1.1.0 is a one-off release: its rules will not change unless EcoHealth
Alliance publishes a new version. The CITES full download is re-released every year, and records can
be removed or corrected between versions (for example after nomenclature changes). When you move to
a new CITES version, rerun the EDA notebooks (`SYSTEM CODE/notebooks/`) and the
[pipeline checks](ALIGNING_LEMIS_AND_CITES.md#24-checks-every-pipeline-should-pass).

---

## 1. LEMIS (US Fish & Wildlife Service, curated by EcoHealth Alliance)

### 1.1 The right file

| | |
|---|---|
| File | `lemis_2000_2014_cleaned.csv` (the "cleaned" is EcoHealth Alliance's) |
| Version | v1.1.0, published 6 December 2019. Identical to the data in the `lemis` R package v1.1.0 |
| Zenodo | record 3565869, DOI [10.5281/zenodo.3565869](https://doi.org/10.5281/zenodo.3565869). The all-versions DOI 10.5281/zenodo.3387660 resolves to the same latest version |
| Size and checksum | 973,799,285 bytes, MD5 `434fac9386cba5e23ad72bd1403e91c8` |
| Rows, columns | 5,512,706 rows (one per product line item), 28 columns |
| Companion files | [`reference/lemis_codes.csv`](reference/lemis_codes.csv) (every code's meaning), [`reference/lemis_metadata.csv`](reference/lemis_metadata.csv) (field definitions) |
| Licence | CC-BY-4.0 |

`python DATA/download_data.py` checks the MD5 against Zenodo. Earlier package releases (0.1.0,
0.2.0, 1.0.0) contain different data; never mix versions. The R package itself is not needed: the CSV
is the same data.

### 1.2 What "cleaned" already did (don't redo it, don't undo it)

- **Imports only.** EcoHealth removed the under 5% of raw rows not marked `I` in `import_export`.
  Every row is `I`, so filtering on it does nothing.
- **Near-duplicates removed, exact duplicates kept on purpose.** Some 2013 rows were duplicated
  with and without `value`; the copies without value were removed. Exact duplicates that remain came
  from the same USFWS file and can be real. For example, `control_number` 2000732392 repeats one
  record (quantity 1) 25 times for 25 individually packed garments. Do not de-duplicate blindly.
- **Codes checked against the USFWS code lists.** Obvious typos were fixed (for example `MAE` → `MEA`
  for meat, `L` → `LT`). Pounds were converted to kilograms (`LB` → `KG`) and gallons to litres
  (`GL` → `LT`), with quantities converted too. Codes that could not be interpreted were replaced by
  the literal string **`non-standard value`**, with the original kept in `cleaning_notes`. This was
  applied to `description`, `unit`, `country_origin`, `country_imp_exp`, `purpose`, `source`,
  `action`, `disposition` and `port`.
- **Missing values are the string `NA`.** Read the CSV with `NA` (and usually `non-standard value`)
  as null.
- **Dates.** Obviously wrong disposition dates were cleaned; shipment dates are within 2000–2014.
- **Added fields.** `taxa` (USFWS grouping), `class` (assigned by EcoHealth), `cleaning_notes`,
  `shipment_year` and `disposition_year`. Names were hand-corrected; a vague species is `sp.`.

### 1.3 Field rules

| Field | Meaning and rule |
|---|---|
| `control_number` | One shipment. A shipment's rows are its products. All rows of a shipment share `country_imp_exp`, `shipment_date`, `us_co`, `foreign_co` and `port`. Keep a shipment's rows in the same train/test split, and use it for shipment-level evaluation |
| `genus` | "Genus **(or higher-level taxonomic name)**". 64,805 rows (137 distinct names) hold a name that CITES uses as an order, family or class rather than a genus: `Scleractinia` 45,922, `Cheloniidae` 4,243, `Anatidae` 3,067, `Elephantidae` 2,397, `Crocodylidae` 903, `Cetacea` 849, and so on (counted in EDA notebook 03). Some rows hold generic labels such as "Tropical fish". Never assume this field is a genus |
| `species` | `sp.` when vague. **Synonyms were not reconciled**, and names do not follow CITES standard nomenclature, so name matching against CITES will miss synonyms |
| `class` | Assigned by EcoHealth from Catalogue of Life/ITIS for over 92% of rows. 403,596 rows (7.3%) have none. Names follow Catalogue of Life (`Actinopterygii`), whereas CITES uses `Actinopteri` |
| `taxa` | USFWS's broad grouping: covers over 99% of rows but is not a consistent taxonomy. The data paper advises using it together with `class` |
| `description` | Product form: 94 valid codes (see `lemis_codes.csv`) |
| `quantity`, `unit` | Units: `NO` number of specimens, `KG`, `GM`, `MG`, `LT`, `ML`, `C3` cm³, `M2`, `C2` cm², `M3`, **`MT` = metres (not metric tonnes)**, `CM`. Never add or compare quantities across units |
| `value` | Declared value in USD, entered by the importer. **USFWS did not provide it for 2008, 2009, 2010 or 2014** (every row is `NA`). It is also `NA` for 118,640 of 396,038 rows in 2011 and 179,749 of 410,717 in 2013. Zeros occur in other years. `NA` is not 0. Any value-based feature has no data in those years, including the whole of 2014 |
| `country_origin`, `country_imp_exp` | Declared by the importer and not verifiable. The US can legitimately be the origin of a US import (raw material processed abroad). Codes are mostly ISO-3166 alpha-2, plus USFWS codes, listed in `lemis_codes.csv` |
| `purpose`, `source` | CITES-style letter codes, defined in `lemis_codes.csv`. Do not assume they mean the same as CITES's letters; how they compare is in the [crosswalk](ALIGNING_LEMIS_AND_CITES.md#21-crosswalk) |
| `action` | `C` cleared, `R` refused |
| `disposition` | `A` abandoned, `C` cleared, `R` **re-exported**, `S` seized. `R` means "refused" in `action` and "re-exported" in `disposition` |
| `shipment_date`, `shipment_year` | The reliable time index |
| `disposition_date` | Some values are still inconsistent. Avoid it, and distrust any disposition dated before its shipment |
| `us_co`, `foreign_co` | `EXEMPTIONS 6 AND 7(C)` means redacted under FOIA (2.2% and 0.5% of rows). 17.7% and 6.9% of rows are missing |
| `port` | Port of entry. LEMIS-only, with no CITES equivalent |

**Scope.** LEMIS covers every wildlife import declared to USFWS, whether or not the species is
CITES-listed. Only 24.5% of its standard species names are subject to CITES reporting. It also
includes some plants and microorganisms.

### 1.4 Enforcement outcomes (the labels)

Observed combinations of `action` × `disposition`:

| action | disposition | rows |
|---|---|---|
| C | C | 5,417,030 |
| R | S | 53,028 |
| R | A | 22,257 |
| R | R | 18,212 |
| R | C | 2,075 |
| R | NA / non-standard | 39 / 29 |
| C | R | 23 |
| C | S | 2 |
| C | non-standard | 10 |
| NA | C | 1 |

These are **what US inspectors did**, not whether a law was broken. Refusals include paperwork
problems. Illegal shipments that were not detected appear as cleared. Outcomes also reflect US law:
for example, marine-mammal products are refused about ten times as often as average (17.3%),
consistent with the US Marine Mammal Protection Act. Seizure, abandonment and re-export are
different outcomes; if you combine them, say so.

---

## 2. CITES Trade Database: full download (UNEP-WCMC for the CITES Secretariat)

### 2.1 The right files

| | |
|---|---|
| Version used | **2026.1**: extracted 9 June 2026, containing annual reports received by 5 May 2026. 29.2 million records |
| Files | `Trade_database_download_v2026.1.zip` contains 59 `trade_db_*.csv` files (about 500,000 rows each, oldest first) and *Full data download_supplementary information.docx* (English, plus Spanish and French versions). Keep the .docx: it states the version |
| Columns | `Id, Year, Appendix, Taxon, Class, Order, Family, Genus, Term, Quantity, Unit, Importer, Exporter, Origin, Purpose, Source, Reporter.type, Import.permit.RandomID, Export.permit.RandomID, Origin.permit.RandomID, Ifs.permit.RandomID` |
| `Id` | Unique per record and **stable across versions** |
| Permit IDs | Confidential permit numbers are replaced by random IDs, and the same permit always gets the same ID. Parties often report permit numbers inconsistently, so reconstructing trade routes from them is unreliable |

UNEP-WCMC recommends always using the most recent version. Never mix versions. Report the version
and download date.

### 2.2 What a record is, and how not to misread it

- **One row is one line item of a permit or certificate (boxes 7–12), as reported by one Party.**
  `Reporter.type` is `E` (exporter) or `I` (importer).
- **The same trade often appears twice**, once reported by the exporter and once by the importer.
  Never add the two together. Exporter-reported data are often more complete, because some
  Parties do not report imports of Appendix II species. For animals in 2015–2024 there are
  4,836,000 exporter-reported and 3,411,093 importer-reported records. Whenever you count records or
  volumes per group, use one reporter type (or pair the records) and say which you did.
- **Mirror records often disagree.** Causes include a missing annual report, one Party reporting
  permits issued while the other reports actual trade, different terms, units or taxonomic levels,
  overseas territories, year boundaries, and mortality in transport.
- **Some Parties report permits issued rather than used**, so some recorded trades never happened
  and some quantities are overstated.
- **Not in the database:** illegal trade, domestic trade, and species not listed by CITES. The
  exception is **Appendix `N`**: non-CITES species reported by EU members under stricter domestic
  measures (6,511 animal records in 2015–2024). **Filter `N` out** for CITES analyses.
- **Source `I` is not a seizure.** It marks legal, permitted movement of items that were seized
  earlier, for example being returned to their origin. Seizure records are held in the separate,
  access-restricted CITES Illegal Trade Database (hosted by UNODC, open only to Party governments and
  the international enforcement partner agencies, ICCWC).
- **A blank `Origin` means direct trade** (the exporter is the origin) or that origin was not
  reported. It is blank in 33.5% of animal records for 2015–2024.
- **Units.** A blank `Unit` (historical) and `Number of specimens` both mean a count. The full
  download writes units as words (`kg`, `g`, `ml`, `microgrammes` …). Never combine units, and be
  careful combining terms (one feather is not one trophy).
- **Higher-level taxa.** Records reported above genus level have `Taxon` such as
  `Scleractinia spp.` or `Elephantidae spp.`, or are hybrids (`Felidae hybrid`), and leave `Genus`
  blank (86,569 animal records in 2015–2024).
- **Plants have no `Class`.** Select animals with `Class` not blank.
- **Names** follow CITES standard nomenclature. Nomenclature changes adopted by the Conference of
  the Parties are applied to existing records, which is one reason records change between versions.
- **The latest years are incomplete.** Annual reports are due by 31 October of the following year,
  so complete data normally lag two years. Animal records per year in v2026.1:

  | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
  |---|---|---|---|---|---|---|---|
  | 848,215 | 822,985 | 667,829 | 755,935 | 794,191 | 832,770 | **686,852** | **10,040** |

  A drop in the most recent years reflects missing reports, not less trade.

---

## Sources

**LEMIS**
- Eskew, E. A. et al. (2020). United States wildlife and wildlife product imports from 2000–2014.
  *Scientific Data*, 7, 22. https://doi.org/10.1038/s41597-020-0354-5 (Methods and Usage Notes)
- Eskew, E. A. et al. (2019). *United States LEMIS wildlife trade data curated by EcoHealth Alliance*
  (v1.1.0). Zenodo. https://doi.org/10.5281/zenodo.3565869
- `lemis` R package: https://github.com/ecohealthalliance/lemis (mirror:
  https://github.com/eveskew/lemis). See `README.md` and `NEWS.md`; the cleaning code is in
  `data-raw/clean_lemis.R` and `data-raw/taxonomy_lemis.R`
- Code list and field definitions: [`reference/lemis_codes.csv`](reference/lemis_codes.csv),
  [`reference/lemis_metadata.csv`](reference/lemis_metadata.csv)

**CITES**
- CITES Secretariat and UNEP-WCMC (2022). *A guide to using the CITES Trade Database*, Version 9.
  https://trade.cites.org/cites_trade_guidelines/en-CITES_Trade_Database_Guide.pdf
- *Full data download_supplementary information.docx*, included in each full download
  (v2026.1 here)
- Code definitions: the latest *Guidelines for the preparation and submission of CITES annual
  reports* (cites.org), summarised in Annexes 2–5 of the guide above
- CITES Illegal Trade Database: https://citesdata.un.org/
