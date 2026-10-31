
# outputs

Everything here is written by the pipeline (`run_pipeline.sh`) and read by the report, the
slides and the app.

| File | Written by | Contents |
|---|---|---|
| `datasets.json` | `datasets.py` | Version, size and years of LEMIS and CITES as read from the raw files, and which CITES files the preview rows come from (app Overview) |
| `alignment_checks.json` | `check_alignment.py` | The alignment checks of `DATA/ALIGNING_LEMIS_AND_CITES.md` §2.4 |
| `lemis_evaluation.json` | `train.py` | The one test (2013-2014): every model and baseline on four held-out sets (all, listed species, commercial, listed commercial), per line item and per shipment; PR-AUC with 95% intervals and paired differences; the cost of the LEMIS-only fields (C3); confidence bands fixed on 2012; feature importance |
| `lemis_evaluation_table.csv` | `train.py` | The listed-species, per-line-item metrics as a table (one row per model) |
| `lemis_test_scores.parquet` | `train.py` | Per-line-item test scores of every model (app "Does it work?") |
| `lemis_record_scale.json` | `train.py` | Where one US declaration, scored by the full model, falls among the 2000-2014 shipments (app "Check a record") |
| `lemis_supplementary.json` | `analysis.py` | Personal-only and per-year metrics, calibration, worked examples |
| `domain_shift.json` | `domain_shift.py` | Recording shift (same US trade in LEMIS and CITES) and deployment shift (LEMIS training vs ranked CITES): domain-classifier AUC, PSI per field, PCA of the deviation features |
| `cites_scoring_summary.json` | `score_cites.py` | Records selected and ranked, permits, priority permits per year, marine-mammal shares |
| `cites_record_scale.json` | `score_cites.py` | Per year: tier cut-offs and score quantiles, to place one permit or new record in its year (app) |
| `cites_aggregates/*.csv` | `score_cites.py` | Share of exporter-reported records in priority permits, by class, order, taxon, exporter, importer, route, product, purpose, source, appendix and year |
| `cites_validation.json` | `validate_cites.py` | The three CITES checks, each with its pass or fail |
| `cites_unodc_consistency.csv` | `validate_cites.py` | Per UNODC animal group: CITES records, seizure and legal-trade shares, priority share, listing-rule share |
| `cites_top50_review.csv` | `validate_cites.py` | The 50 top-ranked taxa, with reasons and the category each falls in (check 3) |
| `figures/*.png` | `train.py`, `domain_shift.py`, `validate_cites.py`, `diagram.py` | Report and slide figures |
