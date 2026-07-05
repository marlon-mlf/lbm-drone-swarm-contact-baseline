#!/usr/bin/env bash
set -euo pipefail
PY=${PY:-python3}
"$PY" src/validate_submitted_source_data.py --root .
