#!/usr/bin/env bash
# Rebuilds WildGuard end to end from the raw data (about 30-40 minutes on an 8-core laptop).
# Prerequisite: raw data in DATA/raw (see DATA/README.md; python "DATA/download_data.py").
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

$PY -m wildguard.prepare_data       # 1. acquire & align LEMIS + CITES onto one schema
$PY -m wildguard.train              # 2-6. features, supervised/unsupervised/hybrid, LEMIS evaluation
$PY -m wildguard.cites_features     # CITES features with peer norms learned from CITES
$PY -m wildguard.domain_shift       # transfer-robustness experiments + deployment choice
$PY -m wildguard.score_cites        # score CITES 2015-2024, tiers, reasons, aggregates
$PY -m wildguard.validate_cites     # CITES plausibility / consistency checks
$PY -m wildguard.analysis           # robustness: commercial-only, shipment-level, calibration, examples
$PY -m wildguard.diagram            # architecture figure
echo "Done. Launch the app with:  streamlit run app/app.py"
