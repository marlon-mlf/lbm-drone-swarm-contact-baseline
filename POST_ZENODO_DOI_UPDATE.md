# Post-Zenodo DOI update status

The Zenodo DOI has been inserted into the synchronized release metadata and manuscript-support files.

- GitHub repository: https://github.com/marlon-mlf/lbm-drone-swarm-contact-baseline
- Zenodo DOI: https://doi.org/10.5281/zenodo.21211257

Completed updates in this package:

1. `CITATION.cff` includes `repository-code`, `doi`, and `url`.
2. `README.md` includes the DOI badge and release DOI.
3. `.zenodo.json` notes the archived DOI.
4. The manuscript Data Availability and Code Availability statements cite the DOI.
5. The supplementary source-data inventory states the archive DOI.
6. `MANIFEST.tsv` and `checksums_sha256.txt` were regenerated after synchronization.

For any future release, rerun:

```bash
bash scripts/07_make_checksums.sh
sha256sum -c checksums_sha256.txt
```
