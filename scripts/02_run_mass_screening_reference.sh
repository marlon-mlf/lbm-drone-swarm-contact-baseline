#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-python3}
OUTROOT=${OUTROOT:-results/mass_screening_reference}
mkdir -p "$OUTROOT"
for m in 21 22 23 24 25; do
  echo "[mass screening] m=$m"
  $PY src/lbm_swarm_contact_simulator.py \
    --rho 4.55 --runs 20 --agents 60 --steps 100 --grid 100,100,100 \
    --force-scheme nearest --agent-radius 5.0 --agent-mass "$m" \
    --v0-max 0.2 --max-attempts-factor 5 \
    --output-dir "$OUTROOT/mass_${m}_rho4.55_N60" || true
done
