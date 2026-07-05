#!/usr/bin/env python3
"""
Scientific Reports analysis pipeline for the D3Q19 BGK LBM drone-swarm proxy contact-detection study.

This script is a companion/driver for:
    lbm_swarm_contact_simulator.py

It can:
1. Run the complete factorial simulation design.
2. Collect raw per-attempt and per-step time-series data across all conditions.
3. Generate manuscript-ready CSV/Markdown/LaTeX tables.
4. Regenerate the original manuscript-style figures:
   - quadratic growth curves for 100^3 and 130^3 domains,
   - heat map for the 100^3 domain,
   - domain comparison figure.
5. Generate additional time-series and robustness figures for Scientific Reports.
6. Generate extra statistical analyses: rank correlations, descriptive quadratic fits,
   domain-reduction tables, raw-vs-unique comparison, stability margins, and optional
   Poisson/negative-binomial count models when statsmodels is installed.

Recommended production candidate from the stability audit:
    --agent-radius 5.0 --agent-mass 25.0 --v0-max 0.2 --force-scheme nearest

Example stability design run with 10 stable runs per condition:
    python3 scirep_generate_tables_figures.py \
      --sim-script lbm_swarm_contact_simulator.py \
      --output-root scirep_mass25_design_10runs \
      --run-simulations \
      --runs 10 \
      --agent-radius 5.0 \
      --agent-mass 25.0 \
      --v0-max 0.2 \
      --force-scheme nearest \
      --max-attempts-factor 5

Example final production run with 100 stable runs per condition:
    python3 scirep_generate_tables_figures.py \
      --sim-script lbm_swarm_contact_simulator.py \
      --output-root scirep_mass25_production_100runs \
      --run-simulations \
      --runs 100 \
      --agent-radius 5.0 \
      --agent-mass 25.0 \
      --v0-max 0.2 \
      --force-scheme nearest \
      --max-attempts-factor 5

If simulations have already been run, analyze only:
    python3 scirep_generate_tables_figures.py \
      --output-root scirep_mass25_production_100runs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr, t

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_RHOS = [1.0, 1.8, 2.93, 4.55]
DEFAULT_AGENTS = [30, 40, 50, 60]
DEFAULT_GRIDS = ["100,100,100", "130,130,130"]
RHO_THRESHOLD = 20.0
MAX_U_THRESHOLD = 1.1


@dataclass(frozen=True)
class Condition:
    grid_cli: str
    rho: float
    n_agents: int

    @property
    def safe_grid(self) -> str:
        return self.grid_cli.replace(",", "x")

    @property
    def grid_label(self) -> str:
        return self.safe_grid

    @property
    def output_name(self) -> str:
        return f"grid_{self.safe_grid}_rho_{self.rho}_N_{self.n_agents}"


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_grid_list(text: str) -> List[str]:
    grids = []
    for item in text.split(";"):
        item = item.strip()
        if item:
            grids.append(item)
    return grids


def grid_volume(grid_label: str) -> int:
    parts = [int(x) for x in str(grid_label).replace(",", "x").split("x")]
    if len(parts) != 3:
        raise ValueError(f"Invalid grid label: {grid_label}")
    return parts[0] * parts[1] * parts[2]


def grid_side(grid_label: str) -> int:
    parts = [int(x) for x in str(grid_label).replace(",", "x").split("x")]
    return parts[0]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ci95_mean(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return (np.nan, np.nan)
    mean = float(arr.mean())
    if n == 1:
        return (mean, mean)
    sd = float(arr.std(ddof=1))
    if sd <= 0:
        return (mean, mean)
    low, high = t.interval(0.95, n - 1, loc=mean, scale=sd / math.sqrt(n))
    return (float(low), float(high))


def summarize_numeric(values: Iterable[float], prefix: str) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    out: Dict[str, float] = {f"{prefix}_n": int(n)}
    if n == 0:
        return out
    ci_low, ci_high = ci95_mean(arr)
    out.update(
        {
            f"{prefix}_mean": float(arr.mean()),
            f"{prefix}_sd": float(arr.std(ddof=1)) if n > 1 else 0.0,
            f"{prefix}_median": float(np.median(arr)),
            f"{prefix}_ci95_low": ci_low,
            f"{prefix}_ci95_high": ci_high,
            f"{prefix}_iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            f"{prefix}_min": float(arr.min()),
            f"{prefix}_max": float(arr.max()),
        }
    )
    return out


def format_mean_sd_ci_iqr(row: pd.Series, prefix: str) -> str:
    n = int(row.get(f"{prefix}_n", 0))
    if n == 0 or pd.isna(row.get(f"{prefix}_mean", np.nan)):
        return "NA"
    return (
        f"Mean: {row[f'{prefix}_mean']:.2f} "
        f"({row[f'{prefix}_sd']:.2f}), "
        f"CI: [{row[f'{prefix}_ci95_low']:.2f}, {row[f'{prefix}_ci95_high']:.2f}], "
        f"Median: {row[f'{prefix}_median']:.2f}, "
        f"IQR: {row[f'{prefix}_iqr']:.2f}"
    )


# ---------------------------------------------------------------------------
# Simulation execution
# ---------------------------------------------------------------------------


def build_conditions(grids: List[str], rhos: List[float], agents: List[int]) -> List[Condition]:
    return [Condition(grid_cli=g, rho=r, n_agents=n) for g in grids for r in rhos for n in agents]


def run_full_design(args: argparse.Namespace, conditions: List[Condition]) -> None:
    if not args.sim_script:
        raise ValueError("--sim-script is required when --run-simulations is used.")
    sim_script = Path(args.sim_script).expanduser().resolve()
    if not sim_script.exists():
        raise FileNotFoundError(f"Simulation script not found: {sim_script}")

    ensure_dir(args.output_root)
    failed_conditions: List[str] = []

    for cond in conditions:
        cond_dir = args.output_root / cond.output_name
        ensure_dir(cond_dir)
        summary_file = cond_dir / "summary.csv"
        failed_file = cond_dir / "failed_run_config.json"
        if args.skip_existing and (summary_file.exists() or failed_file.exists()):
            print(f"[skip] {cond.output_name}")
            continue

        cmd = [
            sys.executable,
            str(sim_script),
            "--rho",
            str(cond.rho),
            "--runs",
            str(args.runs),
            "--agents",
            str(cond.n_agents),
            "--steps",
            str(args.steps),
            "--grid",
            cond.grid_cli,
            "--force-scheme",
            args.force_scheme,
            "--agent-radius",
            str(args.agent_radius),
            "--agent-mass",
            str(args.agent_mass),
            "--v0-max",
            str(args.v0_max),
            "--max-attempts-factor",
            str(args.max_attempts_factor),
            "--output-dir",
            str(cond_dir),
        ]
        if args.no_condition_plots:
            cmd.append("--no-plots")
        print("\n[run]", cond.output_name)
        print(" ".join(cmd))
        ensure_dir(cond_dir)
        proc = subprocess.run(cmd, cwd=str(args.output_root), text=True)
        if proc.returncode != 0:
            failed_conditions.append(cond.output_name)
            print(f"[warning] condition failed or was unstable: {cond.output_name}")
            if not args.continue_on_failure:
                raise RuntimeError(f"Condition failed: {cond.output_name}")

    if failed_conditions:
        with open(args.output_root / "failed_conditions.txt", "w", encoding="utf-8") as f:
            for item in failed_conditions:
                f.write(item + "\n")
        print("\nFailed/unstable conditions written to failed_conditions.txt")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def read_json_optional(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_config_columns(df: pd.DataFrame, config: Dict, condition_dir: Path) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    grid = f"{config.get('nx', '')}x{config.get('ny', '')}x{config.get('nz', '')}"
    if "grid" not in out.columns or out["grid"].isna().all():
        out["grid"] = grid if grid != "xx" else condition_dir.name
    for key, value in {
        "rho0": config.get("rho0"),
        "n_agents": config.get("n_agents"),
        "steps": config.get("steps"),
        "force_scheme": config.get("force_scheme"),
        "agent_radius": config.get("agent_radius"),
        "agent_mass": config.get("agent_mass"),
        "v0_max": config.get("v0_max"),
        "rho_threshold": config.get("rho_threshold", RHO_THRESHOLD),
        "max_vel": config.get("max_vel", 1.0),
        "tau_bgk": config.get("tau_bgk"),
    }.items():
        if value is not None and (key not in out.columns or out[key].isna().all()):
            out[key] = value
    out["condition_dir"] = condition_dir.name
    out["grid_volume"] = out["grid"].map(grid_volume)
    out["grid_side"] = out["grid"].map(grid_side)
    return out


def load_all_outputs(output_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attempts_frames: List[pd.DataFrame] = []
    summary_frames: List[pd.DataFrame] = []
    ts_frames: List[pd.DataFrame] = []

    for d in sorted(output_root.iterdir()):
        if not d.is_dir():
            continue
        config = read_json_optional(d / "run_config.json")
        if not config:
            config = read_json_optional(d / "failed_run_config.json")
        if not config:
            continue

        attempt_path = d / "attempt_log.csv"
        if attempt_path.exists():
            att = pd.read_csv(attempt_path)
            attempts_frames.append(add_config_columns(att, config, d))

        summary_path = d / "summary.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            summary_frames.append(add_config_columns(summary, config, d))

        stable_ts_path = d / "stable_timeseries_long.csv"
        all_ts_path = d / "all_timeseries_long.csv"
        if stable_ts_path.exists():
            ts = pd.read_csv(stable_ts_path)
            ts_frames.append(add_config_columns(ts, config, d))
        elif all_ts_path.exists():
            ts_all = pd.read_csv(all_ts_path)
            if "stable" in ts_all.columns:
                ts = ts_all[ts_all["stable"] == True].copy()  # noqa: E712
            else:
                ts = ts_all.copy()
            ts_frames.append(add_config_columns(ts, config, d))

    attempts = pd.concat(attempts_frames, ignore_index=True) if attempts_frames else pd.DataFrame()
    summaries = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    timeseries = pd.concat(ts_frames, ignore_index=True) if ts_frames else pd.DataFrame()

    # Normalize datatypes.
    for df in [attempts, summaries, timeseries]:
        if df.empty:
            continue
        for c in ["rho0", "agent_radius", "agent_mass", "v0_max"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in ["n_agents", "steps", "seed", "step"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return attempts, summaries, timeseries


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------


def build_condition_summary_from_attempts(attempts: pd.DataFrame) -> pd.DataFrame:
    if attempts.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["grid", "rho0", "n_agents", "force_scheme", "agent_radius", "agent_mass", "v0_max"]
    for keys, group in attempts.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        stable = group[group["stable"] == True].copy()  # noqa: E712
        row["attempts_total"] = int(len(group))
        row["stable_runs_obtained"] = int(len(stable))
        row["unstable_attempts"] = int(len(group) - len(stable))
        row["stability_fraction"] = float(len(stable) / len(group)) if len(group) else np.nan
        if len(stable) > 0:
            row.update(summarize_numeric(stable["raw_contact_detections"], "raw_detections"))
            row.update(summarize_numeric(stable["unique_contact_events"], "unique_events"))
            row.update(summarize_numeric(stable["steps_completed"], "steps_completed"))
            row.update(summarize_numeric(stable["final_active_contacts"], "final_active_contacts"))
            row.update(summarize_numeric(stable["max_rho_observed"], "max_rho_observed"))
            row.update(summarize_numeric(stable["max_u_observed"], "max_u_observed"))
        else:
            row.update({"raw_detections_n": 0, "unique_events_n": 0, "steps_completed_n": 0})
        row["grid_volume"] = grid_volume(row["grid"])
        row["grid_side"] = grid_side(row["grid"])
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["grid_side", "rho0", "n_agents"]).reset_index(drop=True)
    return out


def build_formatted_contact_tables(condition_summary: pd.DataFrame, metric_prefix: str) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    if condition_summary.empty:
        return tables
    for grid, gdf in condition_summary.groupby("grid"):
        rows = []
        for _, row in gdf.sort_values(["rho0", "n_agents"]).iterrows():
            rows.append(
                {
                    "rho": row["rho0"],
                    "agents": int(row["n_agents"]),
                    "results": format_mean_sd_ci_iqr(row, metric_prefix),
                    "stable_runs": int(row["stable_runs_obtained"]),
                    "unstable_attempts": int(row["unstable_attempts"]),
                }
            )
        tables[str(grid)] = pd.DataFrame(rows)
    return tables


def fit_quadratic_tables(condition_summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    if condition_summary.empty:
        return pd.DataFrame()
    rows = []
    for (grid, rho), gdf in condition_summary.groupby(["grid", "rho0"]):
        gdf = gdf.sort_values("n_agents")
        x = gdf["n_agents"].to_numpy(dtype=float)
        y = gdf[metric].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 3:
            continue
        degree = 2 if len(x) >= 3 else 1
        coeff = np.polyfit(x, y, degree)
        yhat = np.polyval(coeff, x)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        if degree == 2:
            a, b, c = coeff
        else:
            a, b, c = 0.0, coeff[0], coeff[1]
        rows.append(
            {
                "grid": grid,
                "rho0": rho,
                "metric": metric,
                "a_quadratic": float(a),
                "b_linear": float(b),
                "c_constant": float(c),
                "r2": float(r2),
                "n_points": int(len(x)),
            }
        )
    return pd.DataFrame(rows).sort_values(["grid", "rho0", "metric"]).reset_index(drop=True)


def build_correlation_tables(condition_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if condition_summary.empty:
        return pd.DataFrame()
    metrics = ["raw_detections_mean", "unique_events_mean"]
    for grid, gdf in condition_summary.groupby("grid"):
        for metric in metrics:
            if metric not in gdf.columns:
                continue
            values = gdf[metric].astype(float)
            for predictor in ["n_agents", "rho0"]:
                x = gdf[predictor].astype(float)
                ok = np.isfinite(x) & np.isfinite(values)
                if ok.sum() < 3:
                    continue
                sp = spearmanr(x[ok], values[ok])
                kt = kendalltau(x[ok], values[ok])
                rows.append(
                    {
                        "grid": grid,
                        "metric": metric,
                        "predictor": predictor,
                        "n": int(ok.sum()),
                        "spearman_r": float(sp.statistic),
                        "spearman_p": float(sp.pvalue),
                        "kendall_tau": float(kt.statistic),
                        "kendall_p": float(kt.pvalue),
                    }
                )
    return pd.DataFrame(rows)


def build_domain_reduction_table(condition_summary: pd.DataFrame) -> pd.DataFrame:
    if condition_summary.empty:
        return pd.DataFrame()
    pivot_raw = condition_summary.pivot_table(
        index=["rho0", "n_agents"], columns="grid", values="raw_detections_mean", aggfunc="first"
    )
    pivot_unique = condition_summary.pivot_table(
        index=["rho0", "n_agents"], columns="grid", values="unique_events_mean", aggfunc="first"
    )
    grids = sorted(condition_summary["grid"].dropna().unique(), key=grid_volume)
    if len(grids) < 2:
        return pd.DataFrame()
    small, large = grids[0], grids[-1]
    rows = []
    for idx in pivot_raw.index:
        raw_small = pivot_raw.loc[idx].get(small, np.nan)
        raw_large = pivot_raw.loc[idx].get(large, np.nan)
        unique_small = pivot_unique.loc[idx].get(small, np.nan)
        unique_large = pivot_unique.loc[idx].get(large, np.nan)
        rows.append(
            {
                "rho0": idx[0],
                "n_agents": idx[1],
                "small_grid": small,
                "large_grid": large,
                "raw_mean_small_grid": raw_small,
                "raw_mean_large_grid": raw_large,
                "raw_absolute_reduction": raw_small - raw_large if pd.notna(raw_small) and pd.notna(raw_large) else np.nan,
                "raw_percent_reduction": 100 * (raw_small - raw_large) / raw_small if pd.notna(raw_small) and raw_small else np.nan,
                "unique_mean_small_grid": unique_small,
                "unique_mean_large_grid": unique_large,
                "unique_absolute_reduction": unique_small - unique_large if pd.notna(unique_small) and pd.notna(unique_large) else np.nan,
                "unique_percent_reduction": 100 * (unique_small - unique_large) / unique_small if pd.notna(unique_small) and unique_small else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["rho0", "n_agents"]).reset_index(drop=True)


def build_time_series_summary(timeseries: pd.DataFrame) -> pd.DataFrame:
    if timeseries.empty:
        return pd.DataFrame()
    metrics = [
        "raw_contact_detections",
        "unique_contact_events",
        "cumulative_raw_contact_detections",
        "cumulative_unique_contact_events",
        "active_contacts",
        "max_rho",
        "max_u",
        "mean_agent_speed",
    ]
    available = [m for m in metrics if m in timeseries.columns]
    rows = []
    group_cols = ["grid", "rho0", "n_agents", "step"]
    for keys, group in timeseries.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["n_runs"] = int(group["seed"].nunique()) if "seed" in group.columns else int(len(group))
        for metric in available:
            row.update(summarize_numeric(group[metric], metric))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["grid", "rho0", "n_agents", "step"]).reset_index(drop=True)


def build_run_transient_metrics(timeseries: pd.DataFrame) -> pd.DataFrame:
    if timeseries.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["grid", "rho0", "n_agents", "seed", "force_scheme", "agent_radius", "agent_mass", "v0_max"]
    for keys, g in timeseries.groupby(group_cols, dropna=False):
        g = g.sort_values("step")
        row = dict(zip(group_cols, keys))
        row["steps_recorded"] = int(len(g))
        row["final_step"] = int(g["step"].max()) if "step" in g else np.nan
        row["total_raw_contact_detections"] = float(g["cumulative_raw_contact_detections"].iloc[-1]) if "cumulative_raw_contact_detections" in g else float(g["raw_contact_detections"].sum())
        row["total_unique_contact_events"] = float(g["cumulative_unique_contact_events"].iloc[-1]) if "cumulative_unique_contact_events" in g else float(g["unique_contact_events"].sum())
        row["peak_active_contacts"] = float(g["active_contacts"].max()) if "active_contacts" in g else np.nan
        row["peak_max_rho"] = float(g["max_rho"].max()) if "max_rho" in g else np.nan
        row["peak_max_u"] = float(g["max_u"].max()) if "max_u" in g else np.nan
        row["minimum_rho_safety_margin"] = float((RHO_THRESHOLD - g["max_rho"]).min()) if "max_rho" in g else np.nan
        row["minimum_velocity_safety_margin"] = float((MAX_U_THRESHOLD - g["max_u"]).min()) if "max_u" in g else np.nan
        if "cumulative_raw_contact_detections" in g and (g["cumulative_raw_contact_detections"] > 0).any():
            row["first_raw_contact_step"] = int(g.loc[g["cumulative_raw_contact_detections"] > 0, "step"].iloc[0])
        else:
            row["first_raw_contact_step"] = np.nan
        if "cumulative_unique_contact_events" in g and (g["cumulative_unique_contact_events"] > 0).any():
            row["first_unique_contact_step"] = int(g.loc[g["cumulative_unique_contact_events"] > 0, "step"].iloc[0])
        else:
            row["first_unique_contact_step"] = np.nan
        if "mean_agent_speed" in g:
            row["mean_agent_speed_over_time"] = float(g["mean_agent_speed"].mean())
            row["peak_agent_speed_mean_field"] = float(g["mean_agent_speed"].max())
        rows.append(row)
    return pd.DataFrame(rows)


def build_transient_summary(run_metrics: pd.DataFrame) -> pd.DataFrame:
    if run_metrics.empty:
        return pd.DataFrame()
    rows = []
    metrics = [
        "total_raw_contact_detections",
        "total_unique_contact_events",
        "peak_active_contacts",
        "peak_max_rho",
        "peak_max_u",
        "minimum_rho_safety_margin",
        "minimum_velocity_safety_margin",
        "first_raw_contact_step",
        "first_unique_contact_step",
        "mean_agent_speed_over_time",
    ]
    available = [m for m in metrics if m in run_metrics.columns]
    for keys, g in run_metrics.groupby(["grid", "rho0", "n_agents"]):
        row = dict(zip(["grid", "rho0", "n_agents"], keys))
        row["n_runs"] = int(len(g))
        for metric in available:
            row.update(summarize_numeric(g[metric].dropna(), metric))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["grid", "rho0", "n_agents"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Count models
# ---------------------------------------------------------------------------


def fit_count_models(attempts: pd.DataFrame, out_dir: Path) -> None:
    if attempts.empty or not HAS_STATSMODELS:
        note = "statsmodels is not installed or no attempts were available; count models were not fitted.\n"
        with open(out_dir / "count_model_status.txt", "w", encoding="utf-8") as f:
            f.write(note)
        return
    stable = attempts[attempts["stable"] == True].copy()  # noqa: E712
    if stable.empty or stable["raw_contact_detections"].sum() <= 0:
        with open(out_dir / "count_model_status.txt", "w", encoding="utf-8") as f:
            f.write("No stable positive-count data were available; count models were not fitted.\n")
        return
    stable["log_grid_volume"] = np.log(stable["grid_volume"].astype(float))
    stable["n_agents_scaled"] = (stable["n_agents"].astype(float) - stable["n_agents"].mean()) / stable["n_agents"].std(ddof=0)
    stable["rho_scaled"] = (stable["rho0"].astype(float) - stable["rho0"].mean()) / stable["rho0"].std(ddof=0)
    stable["log_volume_scaled"] = (stable["log_grid_volume"] - stable["log_grid_volume"].mean()) / stable["log_grid_volume"].std(ddof=0)

    formulas = {
        "raw": "raw_contact_detections ~ n_agents_scaled + rho_scaled + log_volume_scaled + n_agents_scaled:rho_scaled + n_agents_scaled:log_volume_scaled",
        "unique": "unique_contact_events ~ n_agents_scaled + rho_scaled + log_volume_scaled + n_agents_scaled:rho_scaled + n_agents_scaled:log_volume_scaled",
    }
    reports = []
    model_tables = []

    for label, formula in formulas.items():
        ycol = "raw_contact_detections" if label == "raw" else "unique_contact_events"
        if stable[ycol].sum() <= 0:
            continue
        try:
            poisson = smf.glm(formula=formula, data=stable, family=sm.families.Poisson()).fit()
            pearson = float(sum(poisson.resid_pearson ** 2))
            df_resid = float(poisson.df_resid)
            overdispersion = pearson / df_resid if df_resid > 0 else np.nan
            nb = smf.glm(formula=formula, data=stable, family=sm.families.NegativeBinomial()).fit()
            for model_name, model, od in [("poisson", poisson, overdispersion), ("negative_binomial", nb, np.nan)]:
                conf = model.conf_int()
                for term in model.params.index:
                    model_tables.append(
                        {
                            "outcome": label,
                            "model": model_name,
                            "term": term,
                            "coef_log": float(model.params[term]),
                            "std_error": float(model.bse[term]),
                            "p_value": float(model.pvalues[term]),
                            "irr": float(np.exp(model.params[term])),
                            "irr_ci95_low": float(np.exp(conf.loc[term, 0])),
                            "irr_ci95_high": float(np.exp(conf.loc[term, 1])),
                            "aic": float(model.aic) if hasattr(model, "aic") else np.nan,
                            "poisson_overdispersion_ratio": od if model_name == "poisson" else np.nan,
                        }
                    )
            reports.append(
                f"{label}: Poisson overdispersion ratio = {overdispersion:.3g}; "
                "negative-binomial coefficients are provided as sensitivity summaries."
            )
        except Exception as exc:
            reports.append(f"{label}: model fitting failed: {exc}")

    if model_tables:
        pd.DataFrame(model_tables).to_csv(out_dir / "count_model_coefficients.csv", index=False)
    with open(out_dir / "count_model_status.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(reports) + "\n")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_quadratic_growth(condition_summary: pd.DataFrame, quadratic: pd.DataFrame, out_dir: Path, metric: str, label: str) -> None:
    if condition_summary.empty:
        return
    for grid, gdf in condition_summary.groupby("grid"):
        fig = plt.figure(figsize=(7.2, 4.8))
        for rho, rdf in gdf.groupby("rho0"):
            rdf = rdf.sort_values("n_agents")
            x = rdf["n_agents"].to_numpy(dtype=float)
            y = rdf[metric].to_numpy(dtype=float)
            plt.scatter(x, y, label=f"Mean {label}, rho={rho:g}")
            qrow = quadratic[(quadratic["grid"] == grid) & (quadratic["rho0"] == rho) & (quadratic["metric"] == metric)]
            if not qrow.empty:
                a = float(qrow["a_quadratic"].iloc[0])
                b = float(qrow["b_linear"].iloc[0])
                c = float(qrow["c_constant"].iloc[0])
                xx = np.linspace(x.min(), x.max(), 100)
                yy = a * xx * xx + b * xx + c
                plt.plot(xx, yy, linestyle="--", label=f"Quadratic fit, rho={rho:g}")
        plt.xlabel("Number of agents")
        plt.ylabel(f"Mean {label}")
        plt.title(f"{grid} periodic domain")
        plt.legend(fontsize=8)
        savefig(fig, out_dir / f"figure_quadratic_{label}_{grid}.png")


def plot_heatmap_100(condition_summary: pd.DataFrame, out_dir: Path, metric: str, label: str) -> None:
    if condition_summary.empty:
        return
    grids = sorted(condition_summary["grid"].unique(), key=grid_volume)
    if not grids:
        return
    grid = grids[0]
    data = condition_summary[condition_summary["grid"] == grid]
    if data.empty:
        return
    pivot = data.pivot_table(index="rho0", columns="n_agents", values=metric, aggfunc="first").sort_index()
    fig = plt.figure(figsize=(6.8, 4.8))
    im = plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(im, label=f"Mean {label}")
    plt.xticks(range(len(pivot.columns)), [str(int(c)) for c in pivot.columns])
    plt.yticks(range(len(pivot.index)), [f"{v:g}" for v in pivot.index])
    plt.xlabel("Number of agents")
    plt.ylabel("Fluid density parameter")
    plt.title(f"Heat map of mean {label}, {grid} domain")
    savefig(fig, out_dir / f"figure_heatmap_{label}_{grid}.png")


def plot_domain_comparison(condition_summary: pd.DataFrame, out_dir: Path, metric: str, label: str) -> None:
    if condition_summary.empty:
        return
    data = condition_summary.groupby(["grid", "n_agents"], as_index=False)[metric].mean()
    fig = plt.figure(figsize=(7.0, 4.5))
    for grid, gdf in data.groupby("grid"):
        gdf = gdf.sort_values("n_agents")
        plt.plot(gdf["n_agents"], gdf[metric], marker="o", label=grid)
    plt.xlabel("Number of agents")
    plt.ylabel(f"Mean {label} averaged over density values")
    plt.title("Domain-size comparison")
    plt.legend()
    savefig(fig, out_dir / f"figure_domain_comparison_{label}.png")


def plot_raw_unique_endpoint(condition_summary: pd.DataFrame, out_dir: Path) -> None:
    if condition_summary.empty or "raw_detections_mean" not in condition_summary or "unique_events_mean" not in condition_summary:
        return
    fig = plt.figure(figsize=(6.0, 5.0))
    plt.scatter(condition_summary["raw_detections_mean"], condition_summary["unique_events_mean"])
    maxv = np.nanmax([condition_summary["raw_detections_mean"].max(), condition_summary["unique_events_mean"].max()])
    if np.isfinite(maxv):
        plt.plot([0, maxv], [0, maxv], linestyle="--", label="raw = unique")
    plt.xlabel("Mean raw repeated detections")
    plt.ylabel("Mean unique contact events")
    plt.title("Raw detections versus de-duplicated events")
    plt.legend()
    savefig(fig, out_dir / "figure_raw_vs_unique_endpoint.png")


def plot_stability_margins(condition_summary: pd.DataFrame, out_dir: Path) -> None:
    if condition_summary.empty or "max_rho_observed_max" not in condition_summary:
        return
    data = condition_summary.sort_values(["grid", "rho0", "n_agents"])
    labels = [f"{g}\nr={r:g},N={int(n)}" for g, r, n in zip(data["grid"], data["rho0"], data["n_agents"])]
    x = np.arange(len(data))
    fig = plt.figure(figsize=(max(10, len(data) * 0.35), 4.8))
    plt.plot(x, data["max_rho_observed_max"], marker="o", label="Max observed density")
    plt.axhline(RHO_THRESHOLD, linestyle="--", label="Density threshold")
    plt.xticks(x, labels, rotation=90, fontsize=7)
    plt.ylabel("Maximum density over stable runs")
    plt.title("Density stability margins across conditions")
    plt.legend()
    savefig(fig, out_dir / "figure_density_stability_margins.png")


def plot_worstcase_time_series(ts_summary: pd.DataFrame, out_dir: Path) -> None:
    if ts_summary.empty:
        return
    # Worst case: smallest grid volume, largest rho, largest N.
    ts_summary = ts_summary.copy()
    ts_summary["grid_volume"] = ts_summary["grid"].map(grid_volume)
    min_vol = ts_summary["grid_volume"].min()
    max_rho = ts_summary["rho0"].max()
    max_n = ts_summary["n_agents"].max()
    data = ts_summary[(ts_summary["grid_volume"] == min_vol) & (ts_summary["rho0"] == max_rho) & (ts_summary["n_agents"] == max_n)].sort_values("step")
    if data.empty:
        return

    def plot_metric(metric: str, ylabel: str, filename: str, threshold: Optional[float] = None) -> None:
        mean_col = f"{metric}_mean"
        low_col = f"{metric}_ci95_low"
        high_col = f"{metric}_ci95_high"
        if mean_col not in data.columns:
            return
        fig = plt.figure(figsize=(7.2, 4.6))
        plt.plot(data["step"], data[mean_col], label="Mean")
        if low_col in data.columns and high_col in data.columns:
            plt.fill_between(data["step"], data[low_col], data[high_col], alpha=0.2, label="95% CI")
        if threshold is not None:
            plt.axhline(threshold, linestyle="--", label="Threshold")
        plt.xlabel("Time step")
        plt.ylabel(ylabel)
        plt.title(f"Worst-case time series: rho={max_rho:g}, N={int(max_n)}, grid={data['grid'].iloc[0]}")
        plt.legend()
        savefig(fig, out_dir / filename)

    plot_metric("cumulative_raw_contact_detections", "Cumulative raw detections", "figure_ts_worstcase_cumulative_raw.png")
    plot_metric("cumulative_unique_contact_events", "Cumulative unique events", "figure_ts_worstcase_cumulative_unique.png")
    plot_metric("active_contacts", "Active contacts", "figure_ts_worstcase_active_contacts.png")
    plot_metric("max_rho", "Maximum density", "figure_ts_worstcase_max_rho.png", threshold=RHO_THRESHOLD)
    plot_metric("max_u", "Maximum fluid speed", "figure_ts_worstcase_max_u.png", threshold=MAX_U_THRESHOLD)
    plot_metric("mean_agent_speed", "Mean agent speed", "figure_ts_worstcase_mean_agent_speed.png")


def plot_agent_count_time_series(ts_summary: pd.DataFrame, out_dir: Path, metric: str, label: str) -> None:
    if ts_summary.empty:
        return
    ts_summary = ts_summary.copy()
    ts_summary["grid_volume"] = ts_summary["grid"].map(grid_volume)
    min_vol = ts_summary["grid_volume"].min()
    max_rho = ts_summary["rho0"].max()
    data = ts_summary[(ts_summary["grid_volume"] == min_vol) & (ts_summary["rho0"] == max_rho)].sort_values(["n_agents", "step"])
    mean_col = f"{metric}_mean"
    if data.empty or mean_col not in data.columns:
        return
    fig = plt.figure(figsize=(7.2, 4.6))
    for n, gdf in data.groupby("n_agents"):
        plt.plot(gdf["step"], gdf[mean_col], label=f"N={int(n)}")
    plt.xlabel("Time step")
    plt.ylabel(label)
    plt.title(f"Temporal evolution by swarm size, rho={max_rho:g}, smallest domain")
    plt.legend()
    savefig(fig, out_dir / f"figure_ts_by_agents_{metric}.png")


def generate_all_figures(condition_summary: pd.DataFrame, ts_summary: pd.DataFrame, out_dir: Path) -> None:
    figs_dir = out_dir / "figures"
    ensure_dir(figs_dir)
    q_raw = fit_quadratic_tables(condition_summary, "raw_detections_mean")
    q_unique = fit_quadratic_tables(condition_summary, "unique_events_mean")
    if not q_raw.empty:
        plot_quadratic_growth(condition_summary, q_raw, figs_dir, "raw_detections_mean", "raw_detections")
    if not q_unique.empty:
        plot_quadratic_growth(condition_summary, q_unique, figs_dir, "unique_events_mean", "unique_events")
    plot_heatmap_100(condition_summary, figs_dir, "raw_detections_mean", "raw_detections")
    plot_heatmap_100(condition_summary, figs_dir, "unique_events_mean", "unique_events")
    plot_domain_comparison(condition_summary, figs_dir, "raw_detections_mean", "raw_detections")
    plot_domain_comparison(condition_summary, figs_dir, "unique_events_mean", "unique_events")
    plot_raw_unique_endpoint(condition_summary, figs_dir)
    plot_stability_margins(condition_summary, figs_dir)
    plot_worstcase_time_series(ts_summary, figs_dir)
    plot_agent_count_time_series(ts_summary, figs_dir, "cumulative_raw_contact_detections", "Cumulative raw detections")
    plot_agent_count_time_series(ts_summary, figs_dir, "cumulative_unique_contact_events", "Cumulative unique events")
    plot_agent_count_time_series(ts_summary, figs_dir, "max_rho", "Maximum density")


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


def write_tables_and_reports(output_root: Path, attempts: pd.DataFrame, summaries: pd.DataFrame, timeseries: pd.DataFrame) -> None:
    analysis_dir = output_root / "analysis_outputs"
    tables_dir = analysis_dir / "tables"
    ensure_dir(tables_dir)

    if attempts.empty:
        raise RuntimeError("No attempt_log.csv files were found. Run simulations first or check --output-root.")

    attempts.to_csv(tables_dir / "master_attempt_log.csv", index=False)
    if not summaries.empty:
        summaries.to_csv(tables_dir / "master_condition_summaries_from_simulator.csv", index=False)
    if not timeseries.empty:
        timeseries.to_csv(tables_dir / "master_stable_timeseries_long.csv", index=False)

    condition_summary = build_condition_summary_from_attempts(attempts)
    condition_summary.to_csv(tables_dir / "scirep_condition_summary_raw_unique_stability.csv", index=False)

    # Manuscript-style Tables 3 and 5 for raw detections, plus unique-event equivalents.
    for metric_prefix, label in [("raw_detections", "raw"), ("unique_events", "unique")]:
        formatted = build_formatted_contact_tables(condition_summary, metric_prefix)
        for grid, table in formatted.items():
            safe_grid = str(grid).replace("x", "x")
            table.to_csv(tables_dir / f"scirep_table_{label}_contact_summary_{safe_grid}.csv", index=False)
            table.to_markdown(tables_dir / f"scirep_table_{label}_contact_summary_{safe_grid}.md", index=False)
            table.to_latex(tables_dir / f"scirep_table_{label}_contact_summary_{safe_grid}.tex", index=False, escape=False)

    q_raw = fit_quadratic_tables(condition_summary, "raw_detections_mean")
    q_unique = fit_quadratic_tables(condition_summary, "unique_events_mean")
    q_raw.to_csv(tables_dir / "scirep_quadratic_fit_table_raw_detections.csv", index=False)
    q_unique.to_csv(tables_dir / "scirep_quadratic_fit_table_unique_events.csv", index=False)
    q_raw.to_latex(tables_dir / "scirep_quadratic_fit_table_raw_detections.tex", index=False)
    q_unique.to_latex(tables_dir / "scirep_quadratic_fit_table_unique_events.tex", index=False)

    corr = build_correlation_tables(condition_summary)
    corr.to_csv(tables_dir / "scirep_rank_correlation_table.csv", index=False)
    if not corr.empty:
        corr.to_latex(tables_dir / "scirep_rank_correlation_table.tex", index=False)

    domain_reduction = build_domain_reduction_table(condition_summary)
    domain_reduction.to_csv(tables_dir / "scirep_domain_reduction_table.csv", index=False)
    if not domain_reduction.empty:
        domain_reduction.to_latex(tables_dir / "scirep_domain_reduction_table.tex", index=False)

    if not timeseries.empty:
        ts_summary = build_time_series_summary(timeseries)
        ts_summary.to_csv(tables_dir / "scirep_time_series_summary_by_condition_step.csv", index=False)
        run_metrics = build_run_transient_metrics(timeseries)
        run_metrics.to_csv(tables_dir / "scirep_run_level_transient_metrics.csv", index=False)
        transient_summary = build_transient_summary(run_metrics)
        transient_summary.to_csv(tables_dir / "scirep_transient_metrics_summary.csv", index=False)
        if not transient_summary.empty:
            transient_summary.to_latex(tables_dir / "scirep_transient_metrics_summary.tex", index=False)
    else:
        ts_summary = pd.DataFrame()
        run_metrics = pd.DataFrame()
        transient_summary = pd.DataFrame()

    # Stability/failure table.
    stability_rows = []
    for keys, g in attempts.groupby(["grid", "rho0", "n_agents"]):
        row = dict(zip(["grid", "rho0", "n_agents"], keys))
        row["attempts_total"] = int(len(g))
        row["stable_attempts"] = int(g["stable"].sum())
        row["unstable_attempts"] = int((~g["stable"]).sum())
        row["stability_fraction"] = float(g["stable"].mean())
        row["max_rho_all_attempts"] = float(g["max_rho_observed"].max())
        row["median_max_rho_all_attempts"] = float(g["max_rho_observed"].median())
        row["max_u_all_attempts"] = float(g["max_u_observed"].max())
        stability_rows.append(row)
    stability_table = pd.DataFrame(stability_rows).sort_values(["grid", "rho0", "n_agents"])
    stability_table.to_csv(tables_dir / "scirep_stability_audit_table.csv", index=False)

    stop_reason_table = (
        attempts.groupby(["grid", "rho0", "n_agents", "stop_reason"]).size().reset_index(name="attempt_count")
    )
    stop_reason_table.to_csv(tables_dir / "scirep_stop_reason_table.csv", index=False)

    # Count model analysis.
    fit_count_models(attempts, tables_dir)

    # Figures.
    generate_all_figures(condition_summary, ts_summary, analysis_dir)

    # Manuscript-oriented README.
    write_analysis_readme(analysis_dir, condition_summary, corr, q_raw, q_unique, domain_reduction, transient_summary)


# Pandas DataFrame.to_markdown normally requires the optional package tabulate.
# To keep the pipeline self-contained on fresh Python installations, this patch
# uses pandas' implementation when available and falls back to a small GitHub-
# style markdown writer when tabulate is missing.
def _fallback_markdown(df: pd.DataFrame, index: bool = False) -> str:
    work = df.copy()
    if index:
        work = work.reset_index()
    cols = [str(c) for c in work.columns]
    rows = []
    for _, row in work.iterrows():
        rows.append(["" if pd.isna(v) else str(v) for v in row.tolist()])
    widths = []
    for j, c in enumerate(cols):
        cell_widths = [len(r[j]) for r in rows] if rows else [0]
        widths.append(max(len(c), *cell_widths, 3))
    def fmt(items):
        return "| " + " | ".join(str(x).ljust(widths[i]) for i, x in enumerate(items)) + " |"
    lines = [fmt(cols), "| " + " | ".join("-" * w for w in widths) + " |"]
    for r in rows:
        lines.append(fmt(r))
    return "\n".join(lines)


def _patch_markdown_writer() -> None:
    if getattr(pd.DataFrame, "_scirep_markdown_patched", False):
        return
    original = pd.DataFrame.to_markdown

    def to_markdown(self, buf=None, *args, **kwargs):
        try:
            text = original(self, *args, **kwargs)
        except ImportError:
            text = _fallback_markdown(self, index=kwargs.get("index", False))
        if buf is None:
            return text
        Path(buf).parent.mkdir(parents=True, exist_ok=True)
        with open(buf, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        return None

    pd.DataFrame.to_markdown = to_markdown
    pd.DataFrame._scirep_markdown_patched = True


def write_analysis_readme(
    analysis_dir: Path,
    condition_summary: pd.DataFrame,
    corr: pd.DataFrame,
    q_raw: pd.DataFrame,
    q_unique: pd.DataFrame,
    domain_reduction: pd.DataFrame,
    transient_summary: pd.DataFrame,
) -> None:
    tables_dir = analysis_dir / "tables"
    figures_dir = analysis_dir / "figures"
    lines = []
    lines.append("# Scientific Reports LBM swarm analysis outputs")
    lines.append("")
    lines.append("This folder contains manuscript-ready tables, source data, and figures generated from the final simulation protocol.")
    lines.append("")
    lines.append("## Core tables")
    lines.append("")
    lines.append("- `scirep_condition_summary_raw_unique_stability.csv`: main condition-level summary including raw detections, unique events, and stability margins.")
    lines.append("- `scirep_table_raw_contact_summary_<grid>.csv/.tex/.md`: manuscript-style raw contact-detection tables.")
    lines.append("- `scirep_table_unique_contact_summary_<grid>.csv/.tex/.md`: de-duplicated unique-event companion tables.")
    lines.append("- `scirep_quadratic_fit_table_raw_detections.csv/.tex`: quadratic growth coefficients for the original manuscript-style curves.")
    lines.append("- `scirep_rank_correlation_table.csv/.tex`: Spearman and Kendall monotonic association summaries.")
    lines.append("- `scirep_domain_reduction_table.csv/.tex`: matched comparison between the smaller and larger periodic domains.")
    lines.append("- `scirep_time_series_summary_by_condition_step.csv`: pooled time-series means and confidence intervals by condition and time step.")
    lines.append("- `scirep_transient_metrics_summary.csv/.tex`: first-contact time, peak active contacts, stability margins, and related transient diagnostics.")
    lines.append("- `count_model_coefficients.csv`: optional Poisson and negative-binomial model coefficients if statsmodels was available.")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for fig in sorted(figures_dir.glob("*.png")):
        lines.append(f"- `{fig.name}`")
    lines.append("")
    lines.append("## Recommended manuscript strengthening")
    lines.append("")
    lines.append("1. Report raw repeated detections and de-duplicated unique events side by side.")
    lines.append("2. Add a time-series figure showing cumulative raw detections, cumulative unique events, active contacts, and maximum density for the hardest stable condition.")
    lines.append("3. Add a stability-audit paragraph stating how many attempts were stable and reporting the minimum density safety margin.")
    lines.append("4. Use the domain-reduction table to quantify the effect of increasing volume from 100^3 to 130^3.")
    lines.append("5. Treat quadratic curves as descriptive summaries, not mechanistic laws.")
    lines.append("6. Use the count-model output as a sensitivity analysis, not as the primary result, unless all model diagnostics are acceptable.")
    lines.append("")
    if not condition_summary.empty:
        unstable_total = int(condition_summary["unstable_attempts"].sum()) if "unstable_attempts" in condition_summary.columns else 0
        stable_total = int(condition_summary["stable_runs_obtained"].sum()) if "stable_runs_obtained" in condition_summary.columns else 0
        lines.append("## Dataset audit summary")
        lines.append("")
        lines.append(f"- Stable runs included in condition summaries: {stable_total}")
        lines.append(f"- Unstable attempts observed: {unstable_total}")
        if "max_rho_observed_max" in condition_summary.columns:
            lines.append(f"- Largest maximum density among stable runs: {condition_summary['max_rho_observed_max'].max():.4g}")
        if "max_u_observed_max" in condition_summary.columns:
            lines.append(f"- Largest maximum fluid velocity among stable runs: {condition_summary['max_u_observed_max'].max():.4g}")
    with open(analysis_dir / "README_analysis_outputs.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and analyze full Scientific Reports LBM swarm design.")
    parser.add_argument("--sim-script", type=str, default=None, help="Path to lbm_swarm_contact_simulator.py")
    parser.add_argument("--output-root", type=Path, required=True, help="Root folder for simulation outputs and analysis_outputs")
    parser.add_argument("--run-simulations", action="store_true", help="Run the full factorial simulation design before analysis")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip conditions with summary.csv or failed_run_config.json")
    parser.add_argument("--rerun-existing", action="store_true", help="Rerun conditions even if outputs already exist")
    parser.add_argument("--continue-on-failure", action="store_true", default=True, help="Continue analysis even if some conditions fail")
    parser.add_argument("--runs", type=int, default=10, help="Stable runs requested per condition")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--grids", type=str, default=";".join(DEFAULT_GRIDS), help="Semicolon-separated list, e.g. '100,100,100;130,130,130'")
    parser.add_argument("--rhos", type=str, default=",".join(map(str, DEFAULT_RHOS)))
    parser.add_argument("--agents", type=str, default=",".join(map(str, DEFAULT_AGENTS)))
    parser.add_argument("--force-scheme", choices=["nearest", "trilinear"], default="nearest")
    parser.add_argument("--agent-radius", type=float, default=5.0)
    parser.add_argument("--agent-mass", type=float, default=25.0)
    parser.add_argument("--v0-max", type=float, default=0.2)
    parser.add_argument("--max-attempts-factor", type=int, default=5)
    parser.add_argument("--no-condition-plots", action="store_true", help="Disable per-condition figures generated by the simulator; global figures are still produced")
    args = parser.parse_args()
    if args.rerun_existing:
        args.skip_existing = False
    args.output_root = args.output_root.expanduser().resolve()
    return args


def main() -> None:
    _patch_markdown_writer()
    args = parse_args()
    ensure_dir(args.output_root)

    if ".local/share/Trash" in str(args.output_root) or (args.sim_script and ".local/share/Trash" in str(args.sim_script)):
        print("[warning] You are running from the desktop Trash folder. Move the scripts and output-root to a normal project folder before production runs.")
    rhos = parse_float_list(args.rhos)
    agents = parse_int_list(args.agents)
    grids = parse_grid_list(args.grids)
    conditions = build_conditions(grids, rhos, agents)

    with open(args.output_root / "pipeline_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "sim_script": args.sim_script,
                "run_simulations": args.run_simulations,
                "runs": args.runs,
                "steps": args.steps,
                "grids": grids,
                "rhos": rhos,
                "agents": agents,
                "force_scheme": args.force_scheme,
                "agent_radius": args.agent_radius,
                "agent_mass": args.agent_mass,
                "v0_max": args.v0_max,
                "max_attempts_factor": args.max_attempts_factor,
            },
            f,
            indent=2,
        )

    if args.run_simulations:
        run_full_design(args, conditions)

    print("\n[load] collecting simulation outputs")
    attempts, summaries, timeseries = load_all_outputs(args.output_root)
    print(f"attempt rows: {len(attempts)}")
    print(f"condition summaries: {len(summaries)}")
    print(f"stable time-series rows: {len(timeseries)}")

    print("\n[analyze] writing tables, figures, and reports")
    write_tables_and_reports(args.output_root, attempts, summaries, timeseries)
    print(f"\nDone. See: {args.output_root / 'analysis_outputs'}")


if __name__ == "__main__":
    main()
