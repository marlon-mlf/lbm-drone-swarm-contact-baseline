#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/99_apply_zenodo_doi.sh <ZENODO_DOI> <GITHUB_REPOSITORY_URL>" >&2
  exit 1
fi

DOI="$1"
REPO_URL="$2"
DOI_URL="https://doi.org/${DOI#https://doi.org/}"
DOI_CLEAN="${DOI#https://doi.org/}"

python3 - "$DOI_CLEAN" "$DOI_URL" "$REPO_URL" <<'PY2'
from pathlib import Path
import re, sys

doi, doi_url, repo_url = sys.argv[1:4]
path = Path('CITATION.cff')
text = path.read_text(encoding='utf-8')
text = re.sub(r'\nrepository-code:.*', '', text)
text = re.sub(r'\ndoi:.*', '', text)
text = re.sub(r'\nurl:.*', '', text)
text = text.rstrip() + f'\nrepository-code: "{repo_url}"\ndoi: "{doi}"\nurl: "{doi_url}"\n'
path.write_text(text, encoding='utf-8')
PY2

echo "Updated CITATION.cff with DOI $DOI_CLEAN and repository $REPO_URL"
echo "Now update manuscript Data/Code Availability and cover letter if needed, then run scripts/07_make_checksums.sh."
