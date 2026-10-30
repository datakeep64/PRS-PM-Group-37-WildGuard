# outputs

Everything here is written by the pipeline (`run_pipeline.sh`) and read by the report, the
slides and the app.

| File | Written by | Contents |
|---|---|---|
| `datasets.json` | `datasets.py` | Version, size and years of LEMIS and CITES as read from the raw files, and which CITES files the preview rows come from (app Overview) |
| `lemis_evaluation.json` | `train.py` | Held-out 2013-2014 metrics for every model and baseline, confidence bands, stacker weights, K-Means silhouette/stability, feature importance |
| `lemis_evaluation_table.csv` | `train.py` | The same metrics as a table (one row per model) |
| `lemis_test_scores.parquet` | `train.py` | Per-line-item held-out scores of the main models (used by the app's "Does it work?" page) |
| `lemis_supplementary.json` | `analysis.py` | Commercial-only, personal-only, shipment-level and per-year metrics; calibration; worked examples |
| `domain_shift.json` | `domain_shift.py` | Simulated regional shift (A), LEMIS vs CITES shift: domain-classifier AUC, PSI, quantity offset (B), importance-weighted deployment variant (C) |
| `cites_scoring_summary.json` | `score_cites.py` | CITES records scored, tier counts, K-Means on CITES, score quantiles |
| `cites_aggregates/*.csv` | `score_cites.py` | Priority-tier share by class, order, taxon, exporter, importer, route, product, purpose, source, appendix, year, cluster |
| `cites_validation.json` | `validate_cites.py` | UNODC consistency (Spearman) and confiscated-specimen proxy (ROC-AUC) |
| `cites_unodc_consistency.csv` | `validate_cites.py` | Per species group: CITES records, UNODC seizure share, seizure-to-trade ratio, flag rates |
| `cites_top25_review.csv` | `validate_cites.py` | Highest-ranked distinct CITES records with reasons (qualitative review) |
| `figures/*.png` | `train.py`, `diagram.py` | Report and slide figures |
| `*_log.txt` | all steps | Run logs |
