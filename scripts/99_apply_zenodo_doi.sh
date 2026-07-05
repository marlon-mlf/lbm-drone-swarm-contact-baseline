#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/99_apply_zenodo_doi.sh <ZENODO_DOI> <GITHUB_REPOSITORY_URL>" >&2
  exit 1
fi

DOI="$1"
REPO_URL="$2"

python3 - <<PY
from pathlib import Path
import re
path = Path('CITATION.cff')
text = path.read_text(encoding='utf-8')
text = re.sub(r'\nrepository-code:.*', '', text)
text = re.sub(r'\ndoi:.*', '', text)
text = text.rstrip() + f"\nrepository-code: "{REPO_URL}"\ndoi: "{DOI}"\n"
path.write_text(text, encoding='utf-8')
PY

echo "Updated CITATION.cff with DOI $DOI and repository $REPO_URL"
echo "Now update manuscript Data/Code Availability and cover letter, then run scripts/07_make_checksums.sh."
