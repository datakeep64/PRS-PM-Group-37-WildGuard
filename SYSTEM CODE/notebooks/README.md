# Exploratory data analysis

Three notebooks explore LEMIS and CITES as published, before any alignment, to find out what each
dataset needs before use and what a shared schema has to reconcile. The only rules they take as given
are the publishers' reading rules in [`DATA/USING_LEMIS_AND_CITES.md`](../../DATA/USING_LEMIS_AND_CITES.md).
Each ends with numbered findings (L1–L13, C1–C13, X1–X12), what each implies for preprocessing and
alignment, and the report section it informs. WildGuard's alignment rules were written from these
findings, in [`DATA/ALIGNING_LEMIS_AND_CITES.md`](../../DATA/ALIGNING_LEMIS_AND_CITES.md), which
cites them by number. Notebooks 05 and 04 then test the two decisions those rules leave open
(ALIGNING §2.2), each against a decision rule written before the results were seen. The system-design
questions that remain are listed in [`../DESIGN_QUESTIONS.md`](../DESIGN_QUESTIONS.md); their
experiments will follow here as notebooks 06 onward.

| Notebook | Covers |
|---|---|
| [`01_lemis_eda.ipynb`](01_lemis_eda.ipynb) | LEMIS: file identity, line items and shipments, missing values, labels, declared value, codes, quantities, taxonomy, countries |
| [`02_cites_eda.ipynb`](02_cites_eda.ipynb) | CITES: file identity, records per year, reporter types and mirror records, appendix, purpose and source, terms and units, taxonomy, countries |
| [`03_lemis_vs_cites.ipynb`](03_lemis_vs_cites.ipynb) | Field-by-field comparison: code, unit, product, taxonomy and country mismatches; the deployment shift; the 2013–14 US overlap; the preprocessing the shared schema needs |
| [`05_mirror_records.ipynb`](05_mirror_records.ipynb) | Decision 1, mirror records: whether exporter and importer reports can be paired, which matches actual US imports better, which trade only importers report, and how much review effort each rule loses (findings M1–M6) |
| [`04_training_population.ipynb`](04_training_population.ipynb) | Decision 2, training population: the same model trained on four LEMIS populations, compared on held-out CITES-listed line items; its CITES-side checks use the Decision 1 rule (findings P1–P5) |
| [`06_product_groups.ipynb`](06_product_groups.ipynb) | Design question S1.1, product groups defined by how each vocabulary codes an object, and a correction to the appendix feature's genus rule (findings G1–G6). Reads the rebuilt aligned files and the pre-rework copies in `DATA/processed/pre_rework/` |
| [`07_us_overlap.ipynb`](07_us_overlap.ipynb) | Design question C5: whether LEMIS outcomes can be attached to the CITES records the US reported for the same 2013–14 imports, to give a labelled set in CITES's format (findings U1–U4: they cannot; most enforcement cases never reach CITES) |

They read the raw downloads in `DATA/raw/` (see [`DATA/README.md`](../../DATA/README.md)). The first
run copies them to Parquet in `DATA/processed/` (about 20 seconds); after that each notebook runs in
under 15 seconds. Notebooks 04 and 05 also read the pipeline's aligned LEMIS file and CITES features in
`DATA/processed/` (built by `run_pipeline.sh`). Notebook 04 trains twelve models and takes about
15 minutes; notebook 05 takes about 2. Shared setup (views, code meanings, chart style) is in
[`eda_setup.py`](eda_setup.py); what 04 and 05 share (listing in force, year split, features,
training, CITES records and the mirror-record rules) is in [`experiment_setup.py`](experiment_setup.py).

```bash
pip install ipykernel          # in addition to SYSTEM CODE/requirements.txt
```

Then open a notebook in VS Code or Jupyter with the project's `.venv` as the kernel.
