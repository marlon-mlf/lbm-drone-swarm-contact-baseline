# Validation notes for compact submitted source-data package

This package contains a compact set of processed source-data files for manuscript
inspection. It does not bundle the full regenerated `results/scirep_mass25_production_100runs/`
directory by default.

Use:

```bash
bash scripts/06b_validate_submitted_source_data.sh
```

or:

```bash
bash scripts/06_validate_production_results.sh
```

The latter is a compatibility wrapper. If regenerated production `summary.csv`
files are present under `results/scirep_mass25_production_100runs/`, it validates
that full output tree. If not, it validates the submitted compact source-data files.

Expected compact-source-data validation output:

```text
Attempt-log rows: 3200
Condition-summary rows: 32
Stability-audit rows: 32
Condition-step summary rows: 3200
Run-level transient rows: 3200
Maximum density in submitted attempt log: 11.6482
Maximum velocity in submitted attempt log: 1
Main figure PDFs: 9
Supplementary figure PDFs: 8
Validation status: PASS
The compact submitted source-data package is internally consistent.
```
