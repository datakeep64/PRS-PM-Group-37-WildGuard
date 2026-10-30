# SYSTEM CODE - the runnable WildGuard pattern-recognition system

## Run the MVP app (no raw data needed)

The trained models, evaluation outputs and the scored CITES review queue are committed, so the
app runs straight from a clone:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r "SYSTEM CODE/requirements.txt"
streamlit run "SYSTEM CODE/app/app.py"
```

| Page | What it shows |
|---|---|
| Overview | What the system does, headline results, what a flag does and does not mean; the two source datasets (version, size, period used) with their first five rows as published |
| Review queue | CITES 2015-2024 records ranked for human review; select one for its flag card and reasons |
| Check a shipment | Score any record - as a CITES record (transfer model) or a US import with declared value (full model). Includes the real Indian star tortoise seizure as a preset |
| Does it work? | Inspection-budget slider over held-out LEMIS 2013-2014 shipments: cases caught vs random and vs a listing rule; model comparison; triage bands; domain-shift results |
| Risk patterns | Where flagged CITES trade concentrates (class, product, purpose, exporter, importer, taxon, year) and the UNODC consistency check |

## Rebuild everything from raw data

```bash
python DATA/download_data.py          # LEMIS from Zenodo; prints CITES download steps
bash "SYSTEM CODE/run_pipeline.sh"    # ~35-45 min on an 8-core laptop, ~12 GB RAM peak
```

Set `WILDGUARD_SAMPLE=0.03` to smoke-test `train.py` on a 3% sample in about a minute.

## Pipeline

| Step | Module | Technique | Output |
|---|---|---|---|
| Source data | `datasets.py` | Release version (Zenodo record, CITES release notes), record counts and first rows, read from the raw files | `outputs/datasets.json`, `DATA/samples/*_head.csv` |
| 1. Acquire & align | `prepare_data.py`, `schema_alignment.py` | Knowledge modelling: LEMIS ↔ CITES code crosswalk, unit conversion, CITES-appendix lookup | `DATA/processed/*_aligned.parquet` |
| 2. Features | `features.py` | Robust peer-group statistics (median/MAD), route and purpose/source rarity | relative risk features |
| 3. Supervised | `models.py`, `train.py` | XGBoost (+ Random Forest, Logistic Regression for comparison), temporal split, early stopping | `models/*_stack.joblib` |
| 4. Unsupervised | `models.py` | Peer-deviation score, Isolation Forest, K-Means distance (k by silhouette, stability by ARI) | anomaly percentiles |
| 5. Hybrid | `models.py` | Stacked logistic-regression meta-learner over supervised + unsupervised signals | risk score |
| 6. Evaluate | `train.py`, `evaluate.py`, `domain_shift.py` | PR-AUC, recall at inspection budget, confidence bands, simulated domain shift, importance weighting | `outputs/*.json`, `outputs/figures/` |
| 7. Apply to CITES | `cites_features.py`, `score_cites.py`, `explain.py` | Re-normalised features, detectors re-fitted on CITES, already-confiscated records excluded, per-year budget tiers, TreeSHAP reasons | review queue, aggregates |
| 8. Validate on CITES | `validate_cites.py` | Spearman consistency with UNODC seizure data; confiscated-specimen proxy; top-N review | `outputs/cites_validation.json` |
| 9. Robustness | `analysis.py` | Commercial-only and shipment-level evaluation, per-year stability, calibration, worked examples | `outputs/lemis_supplementary.json` |
| MVP | `app/app.py` | Streamlit + Plotly | interactive review tool |

## Folder contents

```
SYSTEM CODE/
├── app/app.py                 Streamlit MVP
├── wildguard/                 the pipeline package (see table above)
├── models/                    trained stacks, feature builders, deployment.json
├── outputs/                   metrics (json/csv), figures, CITES aggregates, logs
├── requirements.txt
└── run_pipeline.sh
```
