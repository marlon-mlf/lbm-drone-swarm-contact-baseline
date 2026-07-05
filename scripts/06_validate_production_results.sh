#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-python3}
OUT=${OUT:-results/scirep_mass25_production_100runs}

if find "$OUT" -maxdepth 2 -name summary.csv -print -quit 2>/dev/null | grep -q .; then
  echo "[validate] Regenerated production results found at: $OUT"
  "$PY" src/validate_production_outputs.py --root "$OUT" --expected-runs 100
else
  echo "[validate] No regenerated production results found at: $OUT"
  echo "[validate] Validating submitted compact source-data files instead."
  "$PY" src/validate_submitted_source_data.py --root .
fi
