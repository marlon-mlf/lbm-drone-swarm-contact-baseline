#!/usr/bin/env python3
"""Validate the submitted compact source-data package.

This validator is intended for the archived submission package.  It checks the
processed source-data files that are actually included in the compact ZIP,
rather than requiring the regenerated per-condition `results/` tree.  The full
production tree can still be regenerated with `scripts/04_run_full_production_100runs.sh`
and validated with `src/validate_production_outputs.py`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

EXPECTED_GRIDS = ["100x100x100", "130x130x130"]
EXPECTED_RHOS = [1.0, 1.8, 2.93, 4.55]
EXPECTED_AGENTS = [30, 40, 50, 60]
EXPECTED_CONDITIONS = len(EXPECTED_GRIDS) * len(EXPECTED_RHOS) * len(EXPECTED_AGENTS)
EXPECTED_RUNS_PER_CONDITION = 100
EXPECTED_STEPS = 100
EXPECTED_ATTEMPTS = EXPECTED_CONDITIONS * EXPECTED_RUNS_PER_CONDITION
EXPECTED_CONDITION_STEP_ROWS = EXPECTED_CONDITIONS * EXPECTED_STEPS
RHO_THRESHOLD = 20.0

REQUIRED_FILES = [
    "config/production_config.json",
    "source_data/tables/master_attempt_log.csv",
    "source_data/tables/scirep_condition_summary_raw_unique_stability.csv",
    "source_data/tables/scirep_stability_audit_table.csv",
    "source_data/tables/scirep_time_series_summary_by_condition_step.csv",
    "source_data/tables/scirep_run_level_transient_metrics.csv",
    "source_data/tables/scirep_transient_metrics_summary.csv",
    "source_data/tables/count_model_coefficients.csv",
    "source_data/tables/count_model_status.txt",
    "src/lbm_swarm_contact_simulator.py",
    "src/scirep_generate_tables_figures.py",
]

def fail(messages: list[str]) -> None:
    print("Validation status: FAIL")
    for msg in messages:
        print(f"  - {msg}")
    raise SystemExit(1)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate submitted compact source-data files for the Scientific Reports LBM swarm package."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Archive root containing config/, source_data/, src/, and scripts/.",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    errors: list[str] = []
    print(f"Archive root: {root}")

    for rel in REQUIRED_FILES:
        if not (root / rel).exists():
            errors.append(f"Missing required file: {rel}")
    if errors:
        fail(errors)

    attempt = pd.read_csv(root / "source_data/tables/master_attempt_log.csv")
    cond = pd.read_csv(root / "source_data/tables/scirep_condition_summary_raw_unique_stability.csv")
    stability = pd.read_csv(root / "source_data/tables/scirep_stability_audit_table.csv")
    ts = pd.read_csv(root / "source_data/tables/scirep_time_series_summary_by_condition_step.csv")
    run_metrics = pd.read_csv(root / "source_data/tables/scirep_run_level_transient_metrics.csv")

    print(f"Attempt-log rows: {len(attempt)}")
    print(f"Condition-summary rows: {len(cond)}")
    print(f"Stability-audit rows: {len(stability)}")
    print(f"Condition-step summary rows: {len(ts)}")
    print(f"Run-level transient rows: {len(run_metrics)}")

    if len(attempt) != EXPECTED_ATTEMPTS:
        errors.append(f"master_attempt_log.csv has {len(attempt)} rows; expected {EXPECTED_ATTEMPTS}.")
    if len(cond) != EXPECTED_CONDITIONS:
        errors.append(f"condition summary has {len(cond)} rows; expected {EXPECTED_CONDITIONS}.")
    if len(stability) != EXPECTED_CONDITIONS:
        errors.append(f"stability audit has {len(stability)} rows; expected {EXPECTED_CONDITIONS}.")
    if len(ts) != EXPECTED_CONDITION_STEP_ROWS:
        errors.append(f"time-series condition-step summary has {len(ts)} rows; expected {EXPECTED_CONDITION_STEP_ROWS}.")
    if len(run_metrics) != EXPECTED_ATTEMPTS:
        errors.append(f"run-level transient metrics has {len(run_metrics)} rows; expected {EXPECTED_ATTEMPTS}.")

    # Attempt-level checks.
    if "stable" not in attempt.columns:
        errors.append("master_attempt_log.csv is missing the stable column.")
    elif not attempt["stable"].astype(bool).all():
        errors.append("At least one attempt in master_attempt_log.csv is not stable.")
    if "stop_reason" in attempt.columns:
        bad_stop = sorted(set(attempt.loc[attempt["stop_reason"] != "completed", "stop_reason"].astype(str)))
        if bad_stop:
            errors.append(f"Non-completed stop reasons present in submitted attempt log: {bad_stop}.")
    if "max_rho_observed" in attempt.columns:
        max_rho = float(attempt["max_rho_observed"].max())
        print(f"Maximum density in submitted attempt log: {max_rho:.6g}")
        if max_rho > RHO_THRESHOLD:
            errors.append(f"Maximum density {max_rho:.6g} exceeds threshold {RHO_THRESHOLD}.")
    if "max_u_observed" in attempt.columns:
        print(f"Maximum velocity in submitted attempt log: {float(attempt['max_u_observed'].max()):.6g}")

    # Condition coverage and run counts.
    group_cols = ["grid", "rho0", "n_agents"]
    for col in group_cols:
        if col not in attempt.columns:
            errors.append(f"master_attempt_log.csv is missing {col}.")
    if all(col in attempt.columns for col in group_cols):
        counts = attempt.groupby(group_cols).size().reset_index(name="n")
        if len(counts) != EXPECTED_CONDITIONS:
            errors.append(f"Attempt log covers {len(counts)} conditions; expected {EXPECTED_CONDITIONS}.")
        bad_counts = counts[counts["n"] != EXPECTED_RUNS_PER_CONDITION]
        if not bad_counts.empty:
            errors.append("Some conditions do not have 100 attempts in master_attempt_log.csv.")
        observed = {(str(r.grid), float(r.rho0), int(r.n_agents)) for r in counts.itertuples(index=False)}
        expected = {(g, float(r), int(n)) for g in EXPECTED_GRIDS for r in EXPECTED_RHOS for n in EXPECTED_AGENTS}
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing:
            errors.append(f"Missing conditions in attempt log: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if extra:
            errors.append(f"Unexpected conditions in attempt log: {extra[:5]}{'...' if len(extra) > 5 else ''}")

    # Condition summary checks.
    for col in ["stable_runs_obtained", "unstable_attempts", "stability_fraction"]:
        if col not in cond.columns:
            errors.append(f"condition summary is missing {col}.")
    if "stable_runs_obtained" in cond.columns and not (cond["stable_runs_obtained"] == EXPECTED_RUNS_PER_CONDITION).all():
        errors.append("At least one condition summary row does not report 100 stable runs.")
    if "unstable_attempts" in cond.columns and not (cond["unstable_attempts"] == 0).all():
        errors.append("At least one condition summary row reports unstable attempts.")
    if "stability_fraction" in cond.columns and not (cond["stability_fraction"] == 1.0).all():
        errors.append("At least one condition summary row has stability_fraction != 1.")

    # Time-series summary checks.
    for col in ["grid", "rho0", "n_agents", "step", "n_runs"]:
        if col not in ts.columns:
            errors.append(f"time-series summary is missing {col}.")
    if all(col in ts.columns for col in ["grid", "rho0", "n_agents", "step", "n_runs"]):
        ts_counts = ts.groupby(["grid", "rho0", "n_agents"]).size().reset_index(name="n_steps")
        if len(ts_counts) != EXPECTED_CONDITIONS:
            errors.append(f"Time-series summary covers {len(ts_counts)} conditions; expected {EXPECTED_CONDITIONS}.")
        if not (ts_counts["n_steps"] == EXPECTED_STEPS).all():
            errors.append("At least one condition in the time-series summary does not have 100 steps.")
        if not (ts["n_runs"] == EXPECTED_RUNS_PER_CONDITION).all():
            errors.append("At least one condition-step row does not summarize 100 runs.")

    # Figure/source-table presence.
    main_figs = sorted((root / "source_data/figures").glob("*.pdf"))
    supp_figs = sorted((root / "source_data/supplementary_figures").glob("*.pdf"))
    supp_tables = sorted((root / "source_data/supplementary_tables").glob("*"))
    print(f"Main figure PDFs: {len(main_figs)}")
    print(f"Supplementary figure PDFs: {len(supp_figs)}")
    print(f"Supplementary table/source files: {len(supp_tables)}")
    if len(main_figs) < 8:
        errors.append("Fewer than 8 main-figure PDF source files were found in source_data/figures/.")
    if len(supp_figs) < 8:
        errors.append("Fewer than 8 supplementary-figure PDF source files were found in source_data/supplementary_figures/.")

    if errors:
        fail(errors)

    print("Validation status: PASS")
    print("The compact submitted source-data package is internally consistent.")
    print("Note: this validates included processed source data. To validate regenerated raw production outputs,")
    print("run src/validate_production_outputs.py after scripts/04_run_full_production_100runs.sh.")

if __name__ == "__main__":
    main()
