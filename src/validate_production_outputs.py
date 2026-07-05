#!/usr/bin/env python3
"""Validate regenerated full production outputs.

This script expects a local `results/scirep_mass25_production_100runs/` tree
containing one per-condition folder with `summary.csv` files.  It is for
post-regeneration validation.  To validate the compact submitted source-data
files included in the archive, use `src/validate_submitted_source_data.py` or
`scripts/06b_validate_submitted_source_data.sh`.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

EXPECTED_GRIDS = ["100x100x100", "130x130x130"]
EXPECTED_RHOS = ["1.0", "1.8", "2.93", "4.55"]
EXPECTED_AGENTS = [30, 40, 50, 60]

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('results/scirep_mass25_production_100runs'))
    p.add_argument('--expected-runs', type=int, default=100)
    args = p.parse_args()
    root = args.root
    missing, incomplete, unstable = [], [], []
    for grid in EXPECTED_GRIDS:
        for rho in EXPECTED_RHOS:
            for n in EXPECTED_AGENTS:
                name = f'grid_{grid}_rho_{rho}_N_{n}'
                d = root / name
                summary = d / 'summary.csv'
                if not summary.exists():
                    missing.append(name)
                    continue
                df = pd.read_csv(summary)
                requested = int(df['stable_runs_requested'].iloc[0])
                obtained = int(df['stable_runs_obtained'].iloc[0])
                unstable_attempts = int(df['unstable_attempts'].iloc[0])
                if requested != args.expected_runs or obtained != args.expected_runs:
                    incomplete.append((name, requested, obtained, unstable_attempts))
                if unstable_attempts != 0:
                    unstable.append((name, unstable_attempts))
    print(f'Missing conditions: {len(missing)}')
    for x in missing: print('  ', x)
    print(f'Incomplete conditions: {len(incomplete)}')
    for x in incomplete: print('  ', x)
    print(f'Conditions with unstable attempts: {len(unstable)}')
    for x in unstable: print('  ', x)
    print(f'Total summary.csv files: {len(list(root.glob("grid_*/summary.csv")))}')
    if missing or incomplete:
        raise SystemExit(1)
    print('Validation status: PASS')

if __name__ == '__main__':
    main()
