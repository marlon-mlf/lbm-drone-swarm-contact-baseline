# Compact sensitivity audit

This package includes a compact one-factor-at-a-time sensitivity audit used as a supplementary robustness diagnostic for the Scientific Reports manuscript.

The audit is not a replacement for the 32-condition production dataset. It tests selected numerical safeguards around the production protocol under the hardest representative condition:

- grid: 100x100x100
- density parameter: rho0 = 4.55
- number of agents: Nd = 60
- horizon: T = 100
- radius: ri = 5.0
- initial velocity range: v0,max = 0.2
- density threshold: rho_thresh = 20

The production reference uses mi = 25.0, nearest-node reaction forcing, and Vmax = 1.0. Variants change one factor at a time: proxy mass, force-spreading rule, or velocity-clamp scale.

## Reproduction

From the archive root:

```bash
bash run_compact_sensitivity_audit.sh
python3 summarize_compact_sensitivity_audit.py \
  --input-root results/compact_sensitivity_audit \
  --output-csv results/compact_sensitivity_audit/supplementary_table_S12_compact_sensitivity.csv
```

## Submitted source-data files

- `source_data/supplementary_tables/supplementary_table_S12_compact_sensitivity.csv`
- `source_data/supplementary_tables/supplementary_table_S12_compact_sensitivity.tex`
- `source_data/supplementary_tables/compact_sensitivity_attempt_logs/*_attempt_log.csv`

The attempt logs report raw repeated detections, de-duplicated unique events, stability status, maximum density, maximum velocity, maximum pre-clamp agent speed, and clamp-activation diagnostics.
