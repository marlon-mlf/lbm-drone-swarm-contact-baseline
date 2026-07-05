#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-python3}
OUT=${OUT:-results/scirep_mass25_production_100runs}
$PY src/scirep_generate_tables_figures.py --output-root "$OUT"
find "$OUT/analysis_outputs" -maxdepth 3 -type f | sort
