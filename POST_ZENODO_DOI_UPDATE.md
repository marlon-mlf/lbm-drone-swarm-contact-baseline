# Post-Zenodo DOI update instructions

After Zenodo mints the archive DOI, update the repository metadata and manuscript-support files before journal submission.

Recommended steps:

1. Edit `CITATION.cff` to add the archive DOI and the public GitHub repository URL.
2. Edit the manuscript Data Availability and Code Availability statements to cite the Zenodo DOI.
3. Edit the cover letter to cite the same DOI.
4. Recompile the manuscript and cover letter.
5. Regenerate the manifest and checksum files:

```bash
bash scripts/07_make_checksums.sh
sha256sum -c checksums_sha256.txt
```

The helper script `scripts/99_apply_zenodo_doi.sh` can update `CITATION.cff` automatically if you provide the DOI and repository URL.
