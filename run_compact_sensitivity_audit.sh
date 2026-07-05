#!/usr/bin/env bash
set -u

# Compact one-factor-at-a-time sensitivity audit for the Scientific Reports LBM swarm paper.
# Run from the root of scirep_lbm_swarm_submission_package after copying
# lbm_swarm_contact_simulator_sensitivity.py into src/.

PYTHON=${PYTHON:-python3}
SIM=${SIM:-src/lbm_swarm_contact_simulator_sensitivity.py}
OUT_ROOT=${OUT_ROOT:-results/compact_sensitivity_audit}
SEED=${SEED:-20260705}
RUNS=${RUNS:-20}
MAX_ATTEMPTS_FACTOR=${MAX_ATTEMPTS_FACTOR:-1}  # exactly RUNS matched attempts per variant

GRID=${GRID:-100,100,100}
RHO=${RHO:-4.55}
AGENTS=${AGENTS:-60}
STEPS=${STEPS:-100}
RADIUS=${RADIUS:-5.0}
V0=${V0:-0.2}

mkdir -p "$OUT_ROOT"

echo "Python: $PYTHON"
echo "Simulator: $SIM"
echo "Output root: $OUT_ROOT"
echo "Matched seed start: $SEED"
echo "Runs/attempts per variant: $RUNS"
echo "Representative condition: grid=$GRID, rho=$RHO, agents=$AGENTS, steps=$STEPS"

audit_case () {
  local factor="$1"; shift
  local variant="$1"; shift
  local out="$OUT_ROOT/${factor}__${variant}"
  mkdir -p "$out"
  echo
  echo "===== $factor :: $variant ====="
  "$PYTHON" "$SIM" \
    --rho "$RHO" \
    --runs "$RUNS" \
    --agents "$AGENTS" \
    --steps "$STEPS" \
    --grid "$GRID" \
    --seed "$SEED" \
    --agent-radius "$RADIUS" \
    --v0-max "$V0" \
    --max-attempts-factor "$MAX_ATTEMPTS_FACTOR" \
    --output-dir "$out" \
    --no-analysis --no-plots \
    "$@" || true
}

# Production reference.
audit_case production reference_m25_nearest_vmax1p0 \
  --force-scheme nearest --agent-mass 25.0 --max-vel 1.0

# One-factor mass sensitivity around the production value.
audit_case mass m23_nearest_vmax1p0 \
  --force-scheme nearest --agent-mass 23.0 --max-vel 1.0

audit_case mass m30_nearest_vmax1p0 \
  --force-scheme nearest --agent-mass 30.0 --max-vel 1.0

# One-factor force-spreading sensitivity.
audit_case forcing m25_trilinear_vmax1p0 \
  --force-scheme trilinear --agent-mass 25.0 --max-vel 1.0

# One-factor velocity-clamp sensitivity.
audit_case clamp m25_nearest_vmax0p8 \
  --force-scheme nearest --agent-mass 25.0 --max-vel 0.8

audit_case clamp m25_nearest_vmax1p2 \
  --force-scheme nearest --agent-mass 25.0 --max-vel 1.2

# Optional stronger lower-clamp stress case. Uncomment if you want a more severe clamp check.
# audit_case clamp m25_nearest_vmax0p6 \
#   --force-scheme nearest --agent-mass 25.0 --max-vel 0.6

echo

echo "Done. Now summarize with:"
echo "python3 summarize_compact_sensitivity_audit.py --input-root $OUT_ROOT --output-csv $OUT_ROOT/supplementary_table_S12_compact_sensitivity.csv"
