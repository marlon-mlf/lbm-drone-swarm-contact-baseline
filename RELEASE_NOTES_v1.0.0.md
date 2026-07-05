# Release notes: v1.0.0

This DOI-synchronized release contains the code, configuration, processed source data, selected figure-source files, validation scripts, manuscript-support files, and compact sensitivity-audit outputs supporting the manuscript:

**Baseline contact-detection scaling in fluid-coupled periodic models of self-propelled drone-swarm proxies**

## Contents

- Production simulator and analysis pipeline in `src/`.
- Reproduction helper scripts in `scripts/`.
- Production configuration in `config/production_config.json`.
- Processed source-data tables, selected figure sources, and Supplementary Table S12 source data in `source_data/`.
- Synchronized manuscript-support PDFs and LaTeX sources in `manuscript_support/`.
- MIT code license, CC BY 4.0 data-license notice, citation metadata, Zenodo metadata, manifest, and checksums.

## Validation

The compact submitted source-data package is validated with:

```bash
bash scripts/06b_validate_submitted_source_data.sh
```

The checksum manifest is validated with:

```bash
sha256sum -c checksums_sha256.txt
```

## DOI

Archived release DOI: https://doi.org/10.5281/zenodo.21211257.

The DOI has been inserted into `CITATION.cff`, `README.md`, and the manuscript Data/Code Availability statements bundled in `manuscript_support/`.
