#!/usr/bin/env bash
# Rebuilds WildGuard end to end from the raw data (about 30-40 minutes on an 8-core laptop).
# Prerequisite: raw data in DATA/raw (see DATA/README.md; python "DATA/download_data.py").
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"

$PY -m wildguard.datasets           # source versions, sizes and first rows (app Overview)
$PY -m wildguard.prepare_data       # 1. acquire & align LEMIS + CITES onto one schema
$PY -m wildguard.check_alignment    #    the alignment checks (DATA/ALIGNING_LEMIS_AND_CITES.md 2.4)
$PY -m wildguard.train              # 2-6. features, the chosen ensemble, the one test, deployment refit
$PY -m wildguard.cites_features     # CITES features, described with the LEMIS peer norms
$PY -m wildguard.domain_shift       # how far CITES differs from LEMIS (recording, deployment shift)
$PY -m wildguard.score_cites        # rank CITES consignments 2015-2024: tiers, reasons, aggregates
$PY -m wildguard.validate_cites     # the CITES checks with pass criteria fixed in advance
$PY -m wildguard.analysis           # personal imports, by year, calibration, worked examples
$PY -m wildguard.diagram            # architecture figure
echo "Done. Launch the app with:  streamlit run app/app.py"
