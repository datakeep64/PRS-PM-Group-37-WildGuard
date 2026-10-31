# Aligning LEMIS and CITES for WildGuard

This document is where WildGuard starts using the two datasets together. It sets out what each
dataset needs before use, how the two are put on one schema, and the checks every pipeline should
pass. **Any rework of the pipeline, models or app should follow sections 1 and 2.** Section 3 is a
dated check of the code as it stood on 30 October 2026; update or delete it once those items are
fixed.

**How these rules were reached.** In this order:

1. [`USING_LEMIS_AND_CITES.md`](USING_LEMIS_AND_CITES.md) records how each publisher says its
   dataset should be read. These are the only rules taken as given, cited here as *guide §x*.
2. The exploratory analysis in [`SYSTEM CODE/notebooks/`](../SYSTEM%20CODE/notebooks/) looked at
   what the data actually contains:
   - [01](../SYSTEM%20CODE/notebooks/01_lemis_eda.ipynb) covers LEMIS (findings **L1–L13**);
   - [02](../SYSTEM%20CODE/notebooks/02_cites_eda.ipynb) covers CITES (findings **C1–C13**);
   - [03](../SYSTEM%20CODE/notebooks/03_lemis_vs_cites.ipynb) compares the two field by field
     (findings **X1–X12**).
3. The rules below were written from those findings. Each rule cites the findings that justify it.

All counts come from the notebooks. Unless stated otherwise, CITES counts are animal records,
2015–2024, exporter-reported, with Appendix `N` excluded. When either dataset changes version,
rerun the notebooks first, then revise these rules.

---

## 1. Preparing each dataset

### 1.1 LEMIS (training data)

1. **Check the file.** It must be v1.1.0 (MD5 in guide §1.1).
2. **Missing values.** Read `NA` and `non-standard value` as missing. Missing purpose and source
   become "unknown"; never drop the row (L3).
3. **Duplicates.** Keep the 153,485 exact duplicates: EcoHealth kept them on purpose (L2).
4. **Label.** An enforcement case is `action = R` or `disposition` ∈ {S, A, R}. That gives 95,665
   cases (1.74%). Report the outcome mix with it, and evaluate with ranking metrics rather than
   accuracy (L4).
5. **Training population.** Train on animals with a class: drop rows without a class (mostly
   placeholder genera) and plants, fungi and bacteria (§2.2, notebook 04).
6. **Splits and evaluation.**
   - Split by shipment year, keeping each shipment's line items together, and evaluate per line
     item and per shipment (L1, L5).
   - Also report a commercial-only subset. Personal imports are 3.4% of line items but 49.6% of
     cases (L7).
7. **Taxonomy.**
   - Treat placeholder genera (*Non-CITES entry*, *Other live inverts* …) as unknown (L11).
   - Higher taxa written into `genus` follow the shared convention in §2.1 (L11, X7).
8. **Quantities.** Log-transform them, and compare them only within one unit (L10).
9. **Declared value.**
   - It is missing for all of 2008–2010 and 2014, and for part of 2011 and 2013.
   - Keep `NA` and 0 apart: where value was recorded, zero values are 24.1% of cases against 9.8%
     of cleared line items.
   - Value can feed only a LEMIS-only model, in the years it was recorded (L6).

### 1.2 CITES (scoring data)

1. **Check the version.** Check the version and extraction date in the release notes, and that all
   59 files are present (C1).
2. **Select records.**
   - Keep animals (`Class` not blank) in 2015–2024. That is 8,247,093 records, both reporter types
     (C2).
   - Remove Appendix `N`, which is not a CITES listing (C5).
3. **Mirror records** (§2.2). Counts and shares use exporter-reported records only (C4). The
   records scored are those of rule D: every exporter-reported record, plus importer-reported
   records with no exact exporter-reported counterpart on field values.
4. **Units.** A blank unit and `Number of specimens` both mean a count. Keep quantities apart by
   unit, and log-transform them (C9).
5. **Higher taxa.** When `Genus` is blank, take the higher taxon from the first word of `Taxon`, so
   that 81,773 records are not pooled into one "no genus" group (C10).
6. **Origin.** A blank `Origin` means the exporter is the origin (direct trade) (C11).
7. **Missing codes.** Missing purpose and source become "unknown" (C12).
8. **Permit IDs.** Keep the export permit ID, present on 97% of records. It groups line items into
   consignments, and it links an exporter's record to the importer's record of the same permit
   (C12; notebook 05, M2). Permit IDs are not model features, because LEMIS has none (§2.3).
9. **Source `I` and purpose `L`.** Keep these records out of the ranking. Source `I` marks the legal
   movement of items seized earlier, not a seizure (C7).
10. **Ranking.** Rank within each year. Reports lag, so 2024 is incomplete and 2025 almost empty
   (C3).
11. **Peer norms.** Describe CITES records against the peer norms (what is "normal" for a peer
    group) learned from LEMIS's training years, the same norms the model was trained with. Norms
    re-learned from the records being scored were tested and rejected: on held-out LEMIS
    populations they helped in three regions but were significantly worse in two of six tests
    ([notebook 08](../SYSTEM%20CODE/notebooks/08_peer_norms.ipynb), N1–N2).

---

## 2. Putting LEMIS and CITES on one schema

### 2.1 Crosswalk

**Source.** The letters look like CITES codes, but read with their meanings side by side they
differ in one place that matters (L8, X3):

| Meaning | LEMIS | CITES | LEMIS rows | CITES records |
|---|---|---|---|---|
| Wild | W | W | 4,301,115 | 2,924,055 |
| Bred in captivity | C | C | 811,618 | 1,194,468 |
| Ranched | R | R | 229,471 | 265,211 |
| Born in captivity (F1+) | F | F | 107,475 | 325,431 |
| Commercially bred (CITES: Appendix-I animals from registered operations) | D | D | 3,213 | 77,452 |
| Confiscated or seized specimen | I | I | 1,411 | 2,684 |
| Unknown | U | U | 38,269 | 3,831 |
| Artificially propagated (plants) | A | A | 425 | — |
| **Pre-Convention** | **P** | **O** | 14,291 | 36,279 |
| Marine environment outside any State's jurisdiction | — | X | — | 1,253 |

**Map LEMIS `P` → `O`.** Unmapped, every CITES pre-Convention record carries a code the model never
saw in training. LEMIS has no equivalent of CITES's `X`, so `X` maps to unknown.

**Purpose.** All codes match except one: **reintroduction into the wild is `Y` in LEMIS (402 rows) and
`N` in CITES (2,101 records)**, so map `Y` → `N` (X3). The other codes mean the same in both:
- B breeding
- E educational
- G botanic gardens
- H hunting trophy
- L law enforcement / judicial / forensic
- M medical / biomedical
- P personal
- Q circus / travelling exhibition
- S scientific
- T commercial
- Z zoo

The letter `P` means *personal* in both purpose fields but *pre-Convention* in the LEMIS source
field.

**Units.** Read through the LEMIS code list, the two vocabularies match one to one, except that CITES
also has `microgrammes` (237 records). Convert each unit to one canonical unit per dimension, and
never compare across dimensions (L10, C9, X4):

| LEMIS | CITES | | LEMIS | CITES |
|---|---|---|---|---|
| `NO` | blank or `Number of specimens` | | `M2` | `m2` |
| `KG` | `kg` | | `C2` | `cm2` |
| `GM` | `g` | | `M3` | `m3` |
| `MG` | `mg`, and `microgrammes` (× 0.001 to mg) | | `MT` (metres) | `m` |
| `LT` | `l` | | `CM` | `cm` |
| `ML` | `ml` | | `C3` | `cm3` |

**Product form.** LEMIS uses 92 of its 94 three-letter codes; CITES uses 81 English terms in the window. Neither
maps onto the other one to one, so both map to shared groups (X5, L9, C8).
- **Define groups by how each vocabulary codes the object, not by matching names.** LEMIS has
  separate codes for shoes, garments and trim. CITES records most of these as *leather products
  (small)*. That group is 56.4% of CITES records but 10.4% of LEMIS line items, and LEMIS shoes
  (220,574 line items) have no CITES records at all.
- **Keep the high-risk products apart.** Caviar, medicinal extracts, ivory and horn have much higher
  case rates than live animals or trophies (L9).
- **After mapping, compare group shares in both datasets.** A group that is large on one side and
  empty on the other points to a coding difference, not a difference in trade.

The groups, as decided in [notebook 06](../SYSTEM%20CODE/notebooks/06_product_groups.ipynb)
(findings G1–G5; the groups are in `SYSTEM CODE/wildguard/schema_alignment.py`):
- **One leather group.** LEMIS small and large leather goods, shoes and trim, and the CITES leather
  and fur product terms, form `leather_products`. LEMIS shoes are mostly reptile leather, CITES
  never used *shoes* in 2015–2024, and CITES uses *leather products (large)* on 0.5% of leather
  records. LEMIS small goods, large goods and shoes are enforced at similar rates (2.0–3.1%).
  Garments stay separate (0.6%).
- **Live rock is raw coral.** LEMIS `ROC` is *Scleractinia*, which CITES records as *raw corals*.
- **Codes outside the animal groups.** LEMIS `VEN` is wood veneer and `FIB` plant fibre: with the
  other plant and wood codes they go to `other`. CITES *fibres* on animal records are animal fibre
  and go to `hair`.
- **Differences in trade, not coding.** The groups whose shares still differ more than tenfold
  are trophies, shells, horn, hair, scales, cloth, `other`, caviar and fins. Restricted to
  species with a CITES listing in force, all but caviar and fins fall below tenfold: LEMIS holds
  US imports of species CITES does not list (pearl-oyster shells, deer antler, badger hair,
  starfish), and the US is the largest importer of hunting trophies. Caviar uses the same term
  for the same taxa in both datasets; the sharks behind CITES fin records were listed in 2014 or
  later, when LEMIS ends.

**Countries.** Both mostly use ISO-3166 alpha-2. These LEMIS codes never occur in CITES (L12, X9):

| LEMIS | Meaning | Map to | LEMIS values (origin + exporter) |
|---|---|---|---|
| `VS` | Various | `XV` (CITES's "various") | 17,096 |
| `ZZ` | High seas | `HS` (CITES's "introduction from the sea") | 2,255 |
| `ZR` | Zaire | `CD` (ISO successor) | 622 |
| `TP` | East Timor (former ISO code) | `TL` (ISO successor) | 25 |
| `AX` | Åland Islands | leave as is (negligible) | 3 |

`XX` (unknown) and `AN` (Netherlands Antilles) occur in both datasets. `non-standard value` is
unknown.

**Taxonomy.**
- **Class names** (X6). LEMIS follows Catalogue of Life and CITES its own standard nomenclature:
  - `Actinopterygii` = `Actinopteri`;
  - LEMIS `Sarcopterygii` covers CITES `Dipneusti` and `Coelacanthi`;
  - LEMIS `Clitellata` holds both earthworms and leeches; CITES files leeches (*Hirudo*, 8,380
    records) as `Hirudinoidea`, so leech genera (*Hirudo*, *Hirudinaria*, *Placobdella*) map
    there. The EDA missed this; check 3 of §2.4 found it on the rebuilt data;
  - the 50-odd classes found only in LEMIS are taxa CITES does not list.
- **Higher taxa** (L11, C10, X7). The two datasets store them differently:
  - LEMIS writes an order, family or class name into `genus`: 64,805 line items, 137 names, e.g.
    *Scleractinia* 45,922;
  - CITES leaves `Genus` blank and writes `Taxon = "<name> spp."` (*Scleractinia spp.* 60,899).

  Use one convention for both: when CITES `Genus` is blank, take the first word of `Taxon`. Without
  it, CITES corals, elephants, sturgeon hybrids and giant clams fall into one "no genus" group,
  while each LEMIS higher taxon forms its own.
- **Name matching** (X2). LEMIS synonyms are not reconciled with CITES nomenclature. Matched by name:

  | Match | Share of LEMIS line items |
  |---|---|
  | Species matches a CITES taxon | 28.7% |
  | Genus only | 18.9% |
  | Higher taxon | 1.2% |
  | No match | 44.3% |
  | No genus | 6.9% |

  Use the fallback order species → genus → higher taxon, and report the match rate.
- **Appendix** (X8, C5). LEMIS has none. When one is attached from CITES records:
  - use the appendix CITES recorded for that species **in the shipment year**, never the latest
    one. Listings change: pangolins are Appendix II in CITES records to 2016 and Appendix I from
    2017.
  - ignore Appendix `N`.
  - for the model feature, the genus decides only when the record names no species (`sp.` or a
    higher taxon): it takes the strictest appendix in force for the genus. A record that names a
    species CITES has no record of gets the value `genus_listed` when its genus has one, and
    `not_listed` otherwise. Taking a congener's appendix instead, as notebooks 04 and 05 did,
    made American mink Appendix I (through the black-footed ferret) and mislabelled synonyms such
    as the reticulated python, which CITES records as *Malayopython reticulatus*. Those line items
    are enforced at 1.02%, close to unlisted species (0.83%) (notebook 06, G6).

  On this basis (species level, no genus fallback), LEMIS case rates are 4.0% for Appendix I, 1.8%
  for II, and 1.5% for species with no CITES record in that year. The [Checklist of CITES Species](https://checklist.cites.org) and
  [Species+](https://speciesplus.net) give listing histories.

**Fields with no counterpart** (L6, L13, X12).
- **Only in LEMIS:** `value`, `port`, shipment month, `us_co`, `foreign_co`, and the labels.
- **Only in CITES:** `Reporter.type`, the permit IDs, `Appendix`, `Order`, `Family` and `Importer`
  (LEMIS's importer is always the US).

A model meant to transfer uses only fields both datasets record. LEMIS-only fields can serve a
LEMIS-only comparison model, as a ceiling on what the shared fields lose.

### 2.2 Design decisions

Two decisions were tested against rules fixed in advance and decided on 31 October 2026. The mirror
rule was decided first, because the training-population evidence on CITES ([notebook 04](../SYSTEM%20CODE/notebooks/04_training_population.ipynb), P4–P5) is computed
on the records it selects. A third decision, about the review tool, is open.

- **Mirror records: rule D** (X1, C4; [notebook 05](../SYSTEM%20CODE/notebooks/05_mirror_records.ipynb)). A LEMIS row has one reporter, but a CITES trade is
  often reported twice. Aggregate statistics use exporter-reported records only (guide §2.2). The
  review queue scores every exporter-reported record, plus importer-reported records with no exact
  exporter-reported counterpart (same year, taxon, term, unit group, countries, origin, purpose and
  source). Evidence:
  - The two reports cannot be merged into one record. Field values pair only 0.8% of exporter
    records one-to-one (M1). The export permit ID links half of the trade, and the linked records
    agree with each other (M2), but only 33.7% pair one-to-one. Up to a quarter of field-value pairs
    carry different permit IDs, so permit numbering is inconsistent (M3).
  - Importer reports are not redundant. For US imports they match LEMIS better than the exporters'
    reports (M4). 6% of importer records describe trade no exporter reported, mostly from exporters
    that filed no export records that year (M5).
  - Review effort lost in the top 1% (field match): exporter-only 17.1%, both 16.0%, rule D 4.4%. The
    order is the same with either training population (M7).
  - Using the permit to drop importer reports (D_permit) cuts duplication but misses more, because the
    dropped report is often the one the model flags (M7). The permit is therefore used to group and
    compare records, not to drop them.
- **LEMIS training population: animals with a class** (X2, L11; [notebook 04](../SYSTEM%20CODE/notebooks/04_training_population.ipynb)). Compared with all line
  items, CITES-listed species only and listed genera, on held-out CITES-listed line items:
  - the four are indistinguishable overall (P1);
  - training on CITES-listed species only is better on the commercial part (P2) but not overall, so
    it failed the rule;
  - the choice changes about a third of the CITES top 1% (P4).

  If WildGuard is later judged mainly on commercial trade, CITES-listed species only is the
  candidate to re-test on a second year split.
- **Open: the unit of the review queue: line items or permits** ([notebook 05](../SYSTEM%20CODE/notebooks/05_mirror_records.ipynb), M9). Under rule D, 10.9% of
  the top-1% slots go to a second line of a permit already in the top 1%. That includes the other
  country's report of the same consignment. 85% of multi-line permits in the top 1% are split across
  the cut. Ranking permits by their highest-scoring line would free those slots, show whole
  consignments, and absorb most of rule D's remaining duplication. The two reports of a consignment
  disagree on quantity more often among high-risk records (M8). So a permit view would also let the
  reviewer compare what each country reported. This changes the review tool, not the data.

### 2.3 Using the LEMIS labels on CITES

LEMIS outcomes are US enforcement decisions on US imports, 2000–2014. CITES records are mostly
permitted, mostly commercial trade between all Parties, 2015 onward. A model trained on one and
applied to the other produces a **ranking for human review**, not a probability of illegality.

The EDA measured how far apart the two are (X10):
- commercial purpose: 74.5% of LEMIS → 90.4% of CITES;
- hunting trophies: 17.6% → 3.0%;
- wild source: 78% → 60%;
- Reptilia: 13% → 65%, while Mammalia goes from 29% → 8%.

The label is dominated by personal imports (L7), which are 3% of CITES (C6). Some patterns the model
learns are specific to US law, such as marine-mammal products (guide §1.4). For these reasons:
- prefer features relative to a peer group over absolute ones;
- report a commercial-only LEMIS evaluation;
- measure the shift (for example with a domain classifier).

**What the model cannot learn.** Several signals exist only in CITES, so they cannot be learned
from LEMIS enforcement outcomes and can enter only as context or unsupervised evidence:
- permit-based signals: the two countries' reports of a consignment disagreeing (notebook 05, M8),
  re-export chains traced through origin permits, and most of what travels on one permit. One
  consignment feature carries over, the number of distinct sources in a LEMIS shipment or a
  CITES permit ([notebook 09](../SYSTEM%20CODE/notebooks/09_consignment_features.ipynb)), but
  it adds nothing to the final model and is not used (notebook 12). The number of items or taxa
  does not carry over: a LEMIS shipment also holds unlisted items;
- whether the other country reported the trade at all. This depends mostly on the reporting country
  (M5, M6): Taiwan, Russia and Canada, for example, report almost no imports in these records. So
  "not confirmed by the importer" is not a risk signal on its own.

**Most enforcement cases never reach CITES** ([notebook 07](../SYSTEM%20CODE/notebooks/07_us_overlap.ipynb)).
Matched on year, species, exporter, product group, purpose and source, 72.0% of cleared 2013–14
LEMIS line items of listed species have a CITES record the US reported, but only 39.1% of
enforcement cases (seized 31%, abandoned 22%). CITES records trade that took place, and a stopped
shipment did not enter. So LEMIS outcomes cannot be attached to CITES records to form a labelled
set, and a flag on a CITES record means the record resembles trade that was stopped in the US, not
that the record itself was stopped.

Where the two datasets describe the same trade, they agree (X11). In 2013–14, LEMIS line items of
CITES-listed species (243,172) and CITES records reported by the US as importer (246,206) have
almost the same purpose mix (personal 3.29% vs 3.27%) and a similar source mix. The remaining
differences come from how taxa and products are reported, not from the trade itself.

### 2.4 Checks every pipeline should pass

Run these after any change to data loading, the crosswalk, or either dataset's version:

1. **File identity.** The LEMIS MD5 matches
   [guide §1.1](USING_LEMIS_AND_CITES.md#11-the-right-file). The CITES version and extraction date
   in the release notes match the version you cite, and all CSV files are present (C1).
2. **Codes are valid.** Every LEMIS code is in `lemis_codes.csv`, or is `NA` /
   `non-standard value`. Every CITES code is in the lists of the CITES guide (Annexes 2, 3 and 5).
3. **No unseen categories.** For every shared categorical field, list the CITES values that never
   appear in the LEMIS training data, with counts. Anything beyond a handful of records is a
   crosswalk error. This is the test that found the `P`/`O`, `Y`/`N`, `microgrammes` and country
   mismatches (X3, X4, X9).
4. **Group shares after mapping.** Compare product-group and class shares in both datasets (X5,
   X6).
5. **Value coverage by year** before using `value` (L6).
6. **Higher taxa** are handled the same way in both datasets (X7).
7. **Appendix `N`** is excluded from CITES scoring and from any appendix lookup (C5, X8).
8. **Mirror records.** Every CITES count or share uses exporter-reported records and says so (C4).
   The scored records are rule D's (§2.2), and the export permit ID is carried through.
9. **Latest-year completeness.** Compare records per year before reading trends (C3).
10. **Labels.** Report the `action` × `disposition` counts behind the label (L4).
11. **US overlap.** The 2013–14 comparison of LEMIS and US-reported CITES records still agrees on
    purpose and source (X11).

---

## 3. Status of the current WildGuard code (checked 30 October 2026)

These deviations were found by comparing `SYSTEM CODE/wildguard/` with sections 1 and 2. The measured
effects use the deployed model and the committed outputs, which score both reporter types.

**Progress on branch `rework` (31 October 2026).** The data step (`prepare_data.py`) now follows
sections 1 and 2, and `check_alignment.py` runs the §2.4 checks, all of which pass:
- **Fixed:** #1, #2, #3, #4 (including the corrected genus rule above), #5, #10, #13 (export and
  origin permit IDs), #14 (notebook 06); also the leech class name, found by check 3.
- **Flagged in the data, applied when the later steps are rebuilt:** #7 (`rule_d` on CITES
  records) and #11 (`animal` on LEMIS line items).
- **Fixed differently than first planned:** #12. Design question S2.1 (notebook 08) chose LEMIS
  norms for CITES, so `cites_features.py` now applies the LEMIS feature builder instead of
  learning norms from CITES.
- **Open:** #6, #8 and #9 (report wording).
- With the crosswalk applied, rule D leaves 5,465,862 records to rank, 55 fewer than notebook 05
  found: CITES source `X` and a blank source now both read as unknown, so a few more importer
  reports match an exporter report.

| # | Deviation | Where | Effect | Fix |
|---|---|---|---|---|
| 1 | **Pre-Convention `P` (LEMIS) vs `O` (CITES) not mapped.** Both pass through as-is, and `to_model_matrix` turns the unseen `O` into a missing value | `schema_alignment.py` (`SOURCE_CODES`), `prepare_data.py` (`prepare_lemis`), `features.py` (`to_model_matrix`) | **Large.** Rescoring only the affected records with `O` mapped to `P`: pre-Convention records in the priority tier fall from 28,762 (46.4%) to 1,913 (3.1%). Their share of the whole priority tier falls from 35.0% to 2.3%. The old and new priority tiers overlap by only half (Jaccard 0.505). This changes most CITES results: priority shares by product and source, the top-ranked records, the UNODC check and the app's risk-pattern pages. LEMIS held-out results are unaffected | Map LEMIS `P` → `O` when loading, then rerun the full pipeline |
| 2 | Reintroduction purpose `Y` (LEMIS) vs `N` (CITES) not mapped | same as #1 (`PURPOSE_CODES`) | Small: 3,261 CITES records (both reporter types) get a missing purpose | Map `Y` → `N` |
| 3 | Higher-level taxa stored differently: LEMIS name in `genus`, CITES `Genus` blank | `prepare_data.py` (`prepare_cites`) | Not measured. The 86,569 CITES records with blank `Genus` make up 23.8% of the current priority tier (22.6% of them flagged; 14.1% excluding pre-Convention records) | Fill CITES `genus` from `Taxon` when `Genus` is blank, then rerun |
| 4 | The appendix lookup takes each taxon's latest-year CITES appendix, which can be `N` and is anachronistic for 2000–2014 | `prepare_data.py` (`build_cites_appendix_lookup`) | 391,241 LEMIS rows (7.1%) get appendix `N`, and the 6,511 CITES records with `N` are scored | Use the appendix in force in the shipment year, ignoring `N`; filter `N` out of CITES scoring |
| 5 | Country codes `VS`, `ZZ`, `ZR`, `TP` not mapped, and `AN` given no region | `prepare_data.py`, `schema_alignment.py` (`REGION_OF`) | Small: about 20,000 LEMIS country values, mostly `VS` origins, get region "Unknown" and country values CITES never uses | Apply the §2.1 country table |
| 6 | `value` missing for 2008–2010 and 2014 (and parts of 2011 and 2013) not accounted for | `features.py` (`unit_value_z`), report | The deployed transfer model does not use `value`. The "full" model's test years (2013–2014) are mostly without value, which weakens claims about what value adds. The report's statistic "zero values: 24% of refused vs 10%" holds with `NA` and 0 kept apart (24.1% vs 9.8%, L6), but only for years where value was recorded | State the coverage, and say which years the statistic covers |
| 7 | CITES mirror records not de-duplicated | scoring and aggregates | Already listed as a limitation. Group shares and the UNODC "legal trade share" count both reports | Score rule D's records; compute aggregates on exporter-reported records only (§2.2) |
| 8 | CITES 2024 incomplete | year aggregates | Ranking within each year is unaffected; year-on-year comparisons are | Note it wherever 2024 appears in a trend |
| 9 | The meaning of source `I` | report wording (confiscated-specimen check) | Source `I` records are legal movements of previously seized items, not seizures. Excluding them from review stays sensible | Word the check accordingly |
| 10 | CITES unit `microgrammes` not in the unit map | `schema_alignment.py` (`CITES_UNIT`) | Small: 237 exporter-reported records get no unit | Add it, converted to mg |
| 11 | The model trains on every LEMIS line item | `train.py` | Small on CITES-listed records (notebook 04, P1), but about a quarter of the CITES top 1% differs from training on animals with a class (P4) | Train on animals with a class (§1.1) |
| 12 | CITES peer norms learned from all CITES records, both reporter types | `cites_features.py` | Not measured | Learn them from rule D's records (§1.2) |
| 13 | CITES permit IDs not loaded | `prepare_data.py` (`prepare_cites`) | Consignments cannot be grouped and mirror reports cannot be compared | Keep the export and origin permit IDs (§1.2) |
| 14 | Product groups matched by name, not by how each vocabulary codes the object | `schema_alignment.py` (LEMIS and CITES product maps) | Of the 31 groups, 2 never occur in CITES 2015–2024: `shoes` (220,574 LEMIS line items, 4.0%; CITES records shoes as *leather products (small)*) and `coral_live_rock` (19,624). What the model learns about them never applies to a CITES record. `leather_small` is 10.5% of LEMIS but 56.4% of exporter-reported CITES | Regroup as §2.1 says (LEMIS shoes with *leather products (small)*; place live rock), then compare group shares (§2.4 check 4) and explain every group large on one side only |

**Already correct (keep it this way in a rework):**
- reading `NA` and `non-standard value` as missing;
- the label rule (refused, or disposition S/A/R), which matches the code list;
- LEMIS unit conversions, including `MT` = metres;
- `Actinopterygii` → `Actinopteri`;
- selecting CITES animals with `Class` not blank;
- treating a blank `Origin` as the exporter;
- treating a blank unit as a count;
- using `shipment_year` as the time index;
- shipment-level evaluation via `control_number`;
- ranking CITES records within each year.

---

## Sources

- The EDA notebooks in [`SYSTEM CODE/notebooks/`](../SYSTEM%20CODE/notebooks/), for every
  finding cited here (L, C and X numbers).
- The publishers' documentation behind the reading rules, listed under
  [Sources](USING_LEMIS_AND_CITES.md#sources) in [`USING_LEMIS_AND_CITES.md`](USING_LEMIS_AND_CITES.md).
- Listing histories for the appendix rule: Checklist of CITES Species (https://checklist.cites.org)
  and Species+ (https://speciesplus.net).
