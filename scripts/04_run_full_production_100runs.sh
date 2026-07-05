#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-python3}
OUT=${OUT:-results/scirep_mass25_production_100runs}
$PY src/scirep_generate_tables_figures.py \
  --sim-script src/lbm_swarm_contact_simulator.py \
  --output-root "$OUT" \
  --run-simulations \
  --runs 100 --steps 100 \
  --force-scheme nearest --agent-radius 5.0 --agent-mass 25.0 \
  --v0-max 0.2 --max-attempts-factor 5 --skip-existing
