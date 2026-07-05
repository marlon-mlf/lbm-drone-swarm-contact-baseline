#!/usr/bin/env python3
"""Summarize compact one-factor sensitivity audit outputs.

Reads attempt_log.csv files produced by lbm_swarm_contact_simulator_sensitivity.py
and writes a manuscript/supplement-ready CSV table.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd


def parse_case_name(path: Path) -> tuple[str, str]:
    name = path.name
    if "__" in name:
        factor, variant = name.split("__", 1)
    else:
        factor, variant = "unknown", name
    return factor, variant


def safe_mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan


def safe_max(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.max()) if len(s) else np.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("results/compact_sensitivity_audit"))
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-tex", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for attempt_path in sorted(args.input_root.glob("*/attempt_log.csv")):
        case_dir = attempt_path.parent
        factor, variant = parse_case_name(case_dir)
        df = pd.read_csv(attempt_path)
        if df.empty:
            continue

        stable_col = df["stable"].astype(str).str.lower().isin(["true", "1", "yes"])
        stable_df = df[stable_col].copy()
        n_attempts = int(len(df))
        n_stable = int(stable_col.sum())
        n_unstable = n_attempts - n_stable

        # Use stable runs for contact-count means, but all attempts for stability maxima.
        contact_df = stable_df if len(stable_df) else df

        stop_counts = df["stop_reason"].value_counts().to_dict() if "stop_reason" in df.columns else {}
        stop_summary = "; ".join(f"{k}: {v}" for k, v in sorted(stop_counts.items()))

        row = {
            "sensitivity_factor": factor,
            "variant": variant,
            "grid": f"{int(df['nx'].iloc[0])}x{int(df['ny'].iloc[0])}x{int(df['nz'].iloc[0])}",
            "rho0": float(df["rho0"].iloc[0]),
            "n_agents": int(df["n_agents"].iloc[0]),
            "attempts": n_attempts,
            "stable_attempts": n_stable,
            "unstable_attempts": n_unstable,
            "stability_fraction": n_stable / n_attempts if n_attempts else np.nan,
            "mean_raw_detections_stable": safe_mean(contact_df.get("raw_contact_detections", pd.Series(dtype=float))),
            "mean_unique_events_stable": safe_mean(contact_df.get("unique_contact_events", pd.Series(dtype=float))),
            "max_rho_observed_all_attempts": safe_max(df.get("max_rho_observed", pd.Series(dtype=float))),
            "max_u_observed_all_attempts": safe_max(df.get("max_u_observed", pd.Series(dtype=float))),
            "max_agent_speed_pre_clamp_all_attempts": safe_max(df.get("max_agent_speed_pre_clamp_observed", pd.Series(dtype=float))),
            "mean_agent_clamp_activation_frequency_stable": safe_mean(contact_df.get("agent_clamp_activation_frequency", pd.Series(dtype=float))),
            "stop_reasons": stop_summary,
        }

        # Pull config when available.
        for cfg_name in ["run_config.json", "failed_run_config.json"]:
            cfg_path = case_dir / cfg_name
            if cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text())
                    for key in ["force_scheme", "agent_radius", "agent_mass", "v0_max", "max_vel", "rho_threshold", "tau_bgk"]:
                        if key in cfg:
                            row[key] = cfg[key]
                except Exception:
                    pass
                break

        # Fallbacks from variant name if no JSON was written.
        row.setdefault("force_scheme", "trilinear" if "trilinear" in variant else "nearest")
        row.setdefault("agent_radius", 5.0)
        row.setdefault("agent_mass", np.nan)
        row.setdefault("v0_max", 0.2)
        row.setdefault("max_vel", np.nan)
        row.setdefault("rho_threshold", 20.0)

        if n_stable == n_attempts and row["max_rho_observed_all_attempts"] <= row.get("rho_threshold", 20.0):
            row["diagnostic_interpretation"] = "stable under matched-attempt sensitivity audit"
        elif n_stable == 0:
            row["diagnostic_interpretation"] = "unstable under matched-attempt sensitivity audit"
        else:
            row["diagnostic_interpretation"] = "partially stable; report as limitation"

        rows.append(row)

    if not rows:
        raise SystemExit(f"No attempt_log.csv files found under {args.input_root}")

    out = pd.DataFrame(rows)
    preferred_cols = [
        "sensitivity_factor", "variant", "grid", "rho0", "n_agents", "force_scheme",
        "agent_mass", "max_vel", "attempts", "stable_attempts", "unstable_attempts",
        "stability_fraction", "mean_raw_detections_stable", "mean_unique_events_stable",
        "max_rho_observed_all_attempts", "max_agent_speed_pre_clamp_all_attempts",
        "mean_agent_clamp_activation_frequency_stable", "stop_reasons", "diagnostic_interpretation",
    ]
    cols = [c for c in preferred_cols if c in out.columns] + [c for c in out.columns if c not in preferred_cols]
    out = out[cols]

    if args.output_csv is None:
        args.output_csv = args.input_root / "supplementary_table_S12_compact_sensitivity.csv"
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    if args.output_tex is None:
        args.output_tex = args.output_csv.with_suffix(".tex")
    try:
        compact = out.copy()
        for col in compact.select_dtypes(include=["float"]).columns:
            compact[col] = compact[col].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")
        compact.to_latex(args.output_tex, index=False, escape=True)
    except Exception as exc:
        print(f"[warning] could not write LaTeX table: {exc}")

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_tex}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
