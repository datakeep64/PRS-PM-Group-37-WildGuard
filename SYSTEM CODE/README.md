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
| Overview | What the system does, its headline results, what a flag does and does not mean, how it works in seven steps; the two source datasets (version, size, period used) with their first five rows as published |
| Review queue | CITES export permits 2015-2024, ranked within each year (priority = top 1%, review = next 4%); select a permit for its flag card: reasons, peer context, the permit's records with both countries' reports, and a US-law warning where it applies |
| Check a record | Rank one trade record against that year's CITES permits, or a US import declaration (with declared value, port and month) against US shipments 2000-2014. Includes the real Indian star tortoise seizure and a priority CITES record as examples |
| Does it work? | The one test on held-out LEMIS 2013-2014 shipments: an inspection-budget slider, cases found against random inspection and the listing rule, all models compared, what the US-only fields add, triage bands, and the measured shift to CITES |
| What it flags | The three checks on CITES with their pass marks (UNODC seizure data, confiscated specimens, the top 50 taxa), and where priority permits concentrate |

## Rebuild everything from raw data

```bash
python DATA/download_data.py          # LEMIS from Zenodo; prints CITES download steps
bash "SYSTEM CODE/run_pipeline.sh"    # about 40 min on an 8-core laptop, ~12 GB RAM peak
```

Set `WILDGUARD_SAMPLE=0.03` to smoke-test `train.py` on a 3% sample in about a minute.

## Pipeline

How each choice below was made, and the evidence for it, is in [`DESIGN_QUESTIONS.md`](DESIGN_QUESTIONS.md)
(the decision log at its end) and the design notebooks in [`notebooks/`](notebooks/).

| Step | Module | Technique | Output |
|---|---|---|---|
| Source data | `datasets.py` | Release version (Zenodo record, CITES release notes), record counts and first rows, read from the raw files | `outputs/datasets.json`, `DATA/samples/*_head.csv` |
| 1. Select and align | `prepare_data.py`, `schema_alignment.py`, `check_alignment.py` | Knowledge modelling: LEMIS ↔ CITES crosswalk (product groups by how each vocabulary codes the object, units, source and purpose codes, countries, higher taxa) and the CITES listing in force in the shipment year; LEMIS animals with a class; CITES records once per trade (rule D); the alignment checks | `DATA/processed/*_aligned.parquet`, `outputs/alignment_checks.json` |
| 2. Features | `features.py` | Deviation from peer norms (median and MAD per genus × product × unit), route, purpose and source rarity; the 21 fields both datasets share | relative risk features |
| 3. Supervised | `models.py`, `train.py` | XGBoost and a random forest, combined by the average of their percentile ranks, chosen from six candidates; logistic regression for comparison. Trained on 2000-2010, tested once on 2013-2014, refit on 2000-2014 for deployment | `models/cites_ensemble.joblib`, `models/full_ensemble.joblib` |
| 4. Context and reasons | `models.py`, `explain.py` | Peer-deviation score shown as context (no unsupervised detector improved the score); plain-language reasons from XGBoost's exact contributions (TreeSHAP); a US-law warning for marine-mammal products traded between other countries | flag card |
| 5. Evaluate | `train.py`, `evaluate.py`, `domain_shift.py`, `analysis.py` | PR-AUC with a shipment bootstrap, cases found at inspection budgets per shipment, four held-out sets, the cost of the LEMIS-only fields, recording and deployment shift (domain classifier, PSI, PCA) | `outputs/*.json`, `outputs/figures/` |
| 6. Rank CITES | `cites_features.py`, `score_cites.py` | LEMIS peer norms; percentile ranks within each year; export permits scored by their highest record; tiers by inspection budget; already-confiscated and law-enforcement records set aside | review queue, aggregates |
| 7. Check on CITES | `validate_cites.py` | Three checks with pass marks fixed in advance: consistency with UNODC seizure data, the confiscated-specimen proxy, review of the 50 top-ranked taxa | `outputs/cites_validation.json` |
| MVP | `app/app.py` | Streamlit + Plotly | interactive review tool |

## Folder contents

```
SYSTEM CODE/
├── app/app.py                 Streamlit MVP
├── wildguard/                 the pipeline package (see table above)
├── models/                    cites_ensemble (deployed, with CITES scales), full_ensemble (with
│                              the US-only fields), the feature builders; train.py also writes
│                              deployed_ensemble and tested_ensemble, which are not in git
├── outputs/                   metrics (json/csv), figures, CITES aggregates (see outputs/README.md)
├── notebooks/                 EDA (01-03) and design experiments (04-14)
├── DESIGN_QUESTIONS.md        the system design, how each model was chosen, the decision log
├── requirements.txt
└── run_pipeline.sh
```
