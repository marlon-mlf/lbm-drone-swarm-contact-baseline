# Package validation fix changelog

This revision addresses the package-readiness problem identified in the
Scientific Reports readiness report: the validation helper originally expected
a regenerated `results/scirep_mass25_production_100runs/` tree that is not
bundled in the compact submission ZIP.

Changes made:

1. Added `src/validate_submitted_source_data.py`.
   - Validates the processed source-data files included in the ZIP.
   - Checks 3200 attempt-log rows, 32 condition summaries, 32 stability-audit
     rows, 3200 condition-step summaries, 3200 run-level transient rows,
     100 runs per condition, zero unstable attempts, and density below the
     threshold.
2. Updated `scripts/06_validate_production_results.sh`.
   - If regenerated production results exist under `results/`, it validates
     those.
   - If not, it validates the submitted compact source data instead.
3. Added `scripts/06b_validate_submitted_source_data.sh`.
   - Direct command for validating the compact source-data archive.
4. Updated README, DATA_INVENTORY, REPRODUCIBILITY_CHECKLIST, and checksum
   generation instructions.
5. Regenerated `MANIFEST.tsv` and `checksums_sha256.txt`.

The compact ZIP can now be validated without first rerunning the full
production simulation.
