# WildGuard — Cross-Dataset Label Transfer for Wildlife-Trade Risk Prioritisation

**NUS-ISS Graduate Certificate in Pattern Recognition Systems — Practice Module · Group 37**

| Team member | Student ID | Role |
|---|---|---|
| Abinaya Muruganandam | A0181013X | Member |
| Adhithi Muruganandam | A0181014W | Team Leader |

WildGuard learns what preceded enforcement action in **US wildlife import declarations with real
outcomes (LEMIS, 2000–2014)**, and uses those patterns to rank **CITES export permits from
2015–2024** within each year, so that reviewers look at the riskiest trade first.

### Results

The one test: LEMIS shipments of CITES-listed species from 2013–2014, which were never used for
training or for any design choice, counted per shipment.

| If an inspector can check… | WildGuard finds | Random inspection | "Most endangered first" rule |
|---|---|---|---|
| top 1% of shipments | **26%** of the shipments with an enforcement case (99% of these flags are real cases) | 1% | 1% |
| top 5% of shipments | **63%** | 5% | 6% |
| to find 80% of cases, inspect… | **21%** of shipments | 80% | 80% |

PR-AUC per line item is **0.540** (95% interval 0.519–0.559), against 0.026 for the listing rule
and 0.020 for random inspection, using only fields that CITES also records. XGBoost alone scored
0.006 higher than the chosen two-model ensemble on this test; the ensemble had been chosen
beforehand and was kept.

On CITES, 5,465,862 records are ranked in 3,274,789 export permits; 32,741 permits fall in the
priority tier. CITES has no outcomes, so three checks were fixed in advance. One passes: a model
that never sees the source field ranks confiscated specimens above other records (ROC-AUC 0.888).
Two fail: the ranking does not clearly follow UNODC seizure statistics (Spearman ρ = 0.13 across
14 animal groups, against a pass mark of p < 0.05), and in a review of the 50 top-ranked taxa 30%
showed no recognised risk pattern (the pass mark was 20%).

A flag means *"this permit's records resemble trade that US officers stopped"*. It is a prompt for
limited human attention, not evidence that a law was broken.

## Repository structure (mapped to the submission requirements)

| Requirement | Location |
|---|---|
| 1. Runnable pattern recognition system | [`SYSTEM CODE/`](SYSTEM%20CODE/) — pipeline package `wildguard/`, Streamlit MVP `app/app.py`, trained `models/`, evaluation `outputs/`, design notebooks `notebooks/` |
| 2. Datasets | [`DATA/`](DATA/) — sources, versions and citations, download script, codebook, the rules for reading and aligning both datasets |
| 3. Code, model files, supporting documents | [`SYSTEM CODE/wildguard/`](SYSTEM%20CODE/wildguard/), [`SYSTEM CODE/models/`](SYSTEM%20CODE/models/), [`SYSTEM CODE/outputs/`](SYSTEM%20CODE/outputs/), [`SYSTEM CODE/DESIGN_QUESTIONS.md`](SYSTEM%20CODE/DESIGN_QUESTIONS.md) |
| 4. Group final project report | [`Project Report/`](Project%20Report/) — `WildGuard_Final_Report.docx` |
| 5. Presentation files | [`Presentation/`](Presentation/) — first presentation (PDF), final presentation slides and the video recording guide |

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
| Supervised / unsupervised learning | Six supervised candidates compared under one protocol (logistic regression, naive Bayes, decision tree, random forest, AdaBoost, XGBoost); five unsupervised detectors tested (peer deviation, Isolation Forest, K-Means, Gaussian mixture, DBSCAN), with the peer-deviation score kept as context |
| Machine learning techniques | Gradient boosting with early stopping, bagging, time-ordered cross-validation, class-imbalance handling compared, feature selection by groups, PCA to show the shift between the datasets |
| Hybrid ML / ensemble | XGBoost and a random forest combined by rank averaging, chosen by forward selection against a stacked meta-learner and the detectors |
| Sense making | The LEMIS ↔ CITES knowledge model, plain-language reasons from TreeSHAP on every flag, peer context, a permit-level review queue, confidence-aware triage bands |

## Data sources

* **LEMIS**: Eskew, E. A. et al. (2019). *United States LEMIS wildlife trade data curated by
  EcoHealth Alliance* (v1.1.0). Zenodo. https://doi.org/10.5281/zenodo.3565869 (CC-BY-4.0)
* **CITES**: Full CITES Trade Database Download. Version 2026.1. Compiled by UNEP-WCMC,
  Cambridge, UK for the CITES Secretariat, Geneva, Switzerland. Available at: trade.cites.org.

See [`DATA/README.md`](DATA/README.md) for the full citations,
[`DATA/USING_LEMIS_AND_CITES.md`](DATA/USING_LEMIS_AND_CITES.md) for how both datasets must be read
according to their publishers' documentation, and
[`DATA/ALIGNING_LEMIS_AND_CITES.md`](DATA/ALIGNING_LEMIS_AND_CITES.md) for how WildGuard aligns them.
Any change to the pipeline or app should follow both.
[`SYSTEM CODE/DESIGN_QUESTIONS.md`](SYSTEM%20CODE/DESIGN_QUESTIONS.md) sets out the system design
stage by stage, how each model was chosen and evaluated, and the log of every decision.
