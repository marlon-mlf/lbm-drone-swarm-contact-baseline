#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-python3}
OUT=${OUT:-results/smoke_test}
mkdir -p "$OUT"
$PY src/lbm_swarm_contact_simulator.py \
  --rho 1.0 --runs 1 --agents 2 --steps 5 --grid 40,40,40 \
  --force-scheme nearest --agent-radius 5.0 --agent-mass 25.0 \
  --v0-max 0.2 --max-attempts-factor 5 --output-dir "$OUT" --no-plots
ls -lh "$OUT"
