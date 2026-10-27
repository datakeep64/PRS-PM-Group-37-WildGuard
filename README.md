# WildGuard — Cross-Dataset Label Transfer for Wildlife-Trade Risk Prioritisation

**NUS-ISS Graduate Certificate in Pattern Recognition Systems — Practice Module · Group 37**

| Team member | Student ID | Role |
|---|---|---|
| Abinaya Muruganandam | A0181013X | Member |
| Adhithi Muruganandam | A0181014W | Team Leader |

WildGuard learns what preceded enforcement action in **5.5 million US wildlife import
declarations with real outcomes (LEMIS)**, and uses those patterns, together with label-free
anomaly detection, to rank **8.25 million global CITES trade records (2015–2024)** for human
review.

### Results

On held-out LEMIS shipments from 2013–2014, which were never used for training or tuning:

| If an inspector can check… | WildGuard catches | Random inspection | "Most endangered first" rule |
|---|---|---|---|
| top 1% of shipments | **50%** of enforcement cases (79% of flags are real cases) | 1% | 2% |
| top 5% of shipments | **70%** | 5% | 9% |
| to catch 80% of cases, inspect… | **13%** of shipments | 80% | 71% |

PR-AUC is **0.599**, against 0.016 for random inspection, using only fields that CITES also records.
On CITES, where no labels exist, two independent checks support the ranking. First, it agrees with
UNODC seizure statistics (Spearman ρ = 0.864).
Second, a model that never sees the source field ranks CITES's confiscated-specimen records far
above the rest (ROC-AUC 0.87).

A flag means *"this record resembles shipments that historically drew enforcement action, or
deviates sharply from normal trade for its species"*. It is a prompt for limited human attention,
not evidence that a law was broken.

## Repository structure (mapped to the submission requirements)

| Requirement | Location |
|---|---|
| 1. Runnable pattern recognition system | [`SYSTEM CODE/`](SYSTEM%20CODE/) — pipeline package `wildguard/`, Streamlit MVP `app/app.py`, trained `models/`, evaluation `outputs/` |
| 2. Datasets | [`DATA/`](DATA/) — sources, versions and citations, download script, codebook, data samples |
| 3. Code, model files, supporting documents | [`SYSTEM CODE/wildguard/`](SYSTEM%20CODE/wildguard/), [`SYSTEM CODE/models/`](SYSTEM%20CODE/models/), [`SYSTEM CODE/outputs/`](SYSTEM%20CODE/outputs/) |
| 4. Group final project report | [`Project Report/`](Project%20Report/) — `.pdf`, `.docx` and `.md` |
| 5. Presentation files | [`Presentation/`](Presentation/) — first presentation (PDF), final presentation (`.pptx`) and video script |

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r "SYSTEM CODE/requirements.txt"
streamlit run "SYSTEM CODE/app/app.py"
```

The app runs from the committed models and outputs; no raw data is needed. To rebuild everything
from the raw data (about 40 minutes on an 8-core laptop):

```bash
python DATA/download_data.py           # LEMIS from Zenodo; prints CITES download steps
bash "SYSTEM CODE/run_pipeline.sh"
```

## Module aspects demonstrated

| Aspect | In WildGuard |
|---|---|
| Supervised / unsupervised learning | XGBoost, Random Forest and Logistic Regression on LEMIS outcomes; peer-deviation score, Isolation Forest and K-Means distance without labels |
| Machine learning techniques | Gradient boosting with early stopping, bagging, clustering with silhouette selection, isolation-based outlier detection |
| Hybrid ML / ensemble | Tree ensembles, plus a stacked meta-learner over supervised and unsupervised signals |
| Sense making | LEMIS ↔ CITES knowledge model, TreeSHAP reasons on every flag, confidence-aware triage bands, risk-pattern analytics |

## Data sources

* **LEMIS**: Eskew, E. A. et al. (2019). *United States LEMIS wildlife trade data curated by
  EcoHealth Alliance* (v1.1.0). Zenodo. https://doi.org/10.5281/zenodo.3565869 (CC-BY-4.0)
* **CITES**: Full CITES Trade Database Download. Version 2026.1. Compiled by UNEP-WCMC,
  Cambridge, UK for the CITES Secretariat, Geneva, Switzerland. Available at: trade.cites.org.

See [`DATA/README.md`](DATA/README.md) for the full citations and the schema alignment.
