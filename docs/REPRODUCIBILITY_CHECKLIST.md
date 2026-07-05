# Reproducibility and release checklist

## Included in this DOI-synchronized release

- [x] Production configuration is included at `config/production_config.json`.
- [x] Processed source-data tables are included under `source_data/tables/`.
- [x] Compact sensitivity-audit outputs are included under `source_data/supplementary_tables/`.
- [x] Validation scripts are included under `src/` and `scripts/`.
- [x] `scripts/06b_validate_submitted_source_data.sh` validates the compact source-data package without regenerated `results/`.
- [x] MIT license for code is included in `LICENSE`.
- [x] CC BY 4.0 data-license notice is included in `DATA_LICENSE.txt`.
- [x] Citation metadata are included in `CITATION.cff`.
- [x] Zenodo metadata are included in `.zenodo.json`.
- [x] Manifest and checksum files are included.

## To do after Zenodo mints the DOI

- [ ] Add the Zenodo DOI and GitHub repository URL to `CITATION.cff`.
- [ ] Insert the Zenodo DOI into the manuscript Data Availability and Code Availability statements.
- [ ] Insert the Zenodo DOI into the cover letter before journal submission.
- [ ] Rebuild `checksums_sha256.txt` and `MANIFEST.tsv` after DOI-related edits.

## Validation commands

```bash
bash scripts/06b_validate_submitted_source_data.sh
bash scripts/07_make_checksums.sh
sha256sum -c checksums_sha256.txt
```
