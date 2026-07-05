#!/usr/bin/env python3
"""
D3Q19 BGK lattice-Boltzmann simulator for the drone-swarm proxy
contact-detection study.

This research code generates the production simulations reported in the
Scientific Reports manuscript "Baseline contact-detection scaling in
fluid-coupled periodic models of self-propelled drone-swarm proxies".
The model couples a D3Q19 single-relaxation-time BGK fluid solver to
self-propelled spherical agents in periodic three-dimensional domains.

Main modelling and audit features:

1. Pairwise contacts use the minimum-image convention, consistent with
   periodic domains.
2. Contact handling is sequential to avoid race conditions in counters and
   velocity updates.
3. Two contact metrics are recorded: raw repeated contact detections and
   de-duplicated unique contact episodes. A unique episode starts when a pair
   enters contact after being separated at the previous time step.
4. Each attempt returns stability status, stopping reason, run-level summaries,
   and per-step time series, enabling audits of unstable attempts, temporal
   contact accumulation, and density/velocity safeguards.
5. Fluid reaction forcing can be applied either to the nearest lattice node
   (the production protocol) or distributed to eight neighboring nodes using
   trilinear weights for future sensitivity studies.
6. When an output directory is provided, the script writes machine-readable
   CSV/JSON files for per-attempt diagnostics, per-run time series, pooled
   time-series summaries, run-level transient metrics, instability summaries,
   and optional diagnostic plots.

Production protocol used in the manuscript:
    agent radius r_i = 5.0
    agent mass m_i = 25.0
    v0,max = 0.2
    force scheme = nearest
    rho_threshold = 20.0
    100 stable runs for each of 32 full-factorial conditions

The code is intended for reproducibility and scientific audit of the reported
computational baseline. It is not an operational flight-safety simulator and
should not be interpreted as predicting certified UAV collision risk.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _ensure_parent(path: Path) -> Path:
    """Ensure that the parent directory of a file path exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _safe_to_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Write a DataFrame to CSV, creating parent directories if needed."""
    path = _ensure_parent(Path(path))
    df.to_csv(path, **kwargs)

from numba import njit, prange
from scipy.stats import t
from tqdm import trange

# ---------------------------------------------------------------------------
# Global numerical configuration
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42
Q = 19
TAU_BGK = 0.6
MAX_VEL = 1.0
CS2 = 1.0 / 3.0
FCOLL_MAX = 1e3
RHO_THRESHOLD = 20.0  # matches the manuscript parameter table

# ---------------------------------------------------------------------------
# D3Q19 lattice
# ---------------------------------------------------------------------------
C = np.array(
    [
        [0, 0, 0],
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
        [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
        [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
    ],
    dtype=np.int32,
)
W = np.array([1 / 3, *([1 / 18] * 6), *([1 / 36] * 12)], dtype=np.float64)


@dataclass
class RunResult:
    nx: int
    ny: int
    nz: int
    rho0: float
    n_agents: int
    steps_requested: int
    steps_completed: int
    seed: int
    force_scheme: str
    stable: bool
    stop_reason: str
    raw_contact_detections: int
    unique_contact_events: int
    final_active_contacts: int
    max_rho_observed: float
    max_u_observed: float
    max_agent_speed_pre_clamp_observed: float
    agent_clamp_activation_count: int
    agent_clamp_checks: int
    agent_clamp_activation_frequency: float


def init_positions_no_overlap(
    nx: int,
    ny: int,
    nz: int,
    rads: np.ndarray,
    rng: np.random.Generator,
    margin: float = 0.5,
    pad: float = 10.0,
    max_tries: int = 200_000,
) -> np.ndarray:
    """Generate non-overlapping centers using periodic minimum-image distances."""
    if min(nx, ny, nz) <= 2 * pad:
        raise ValueError("Domain is too small for the requested boundary pad.")

    n_agents = len(rads)
    pos = np.empty((n_agents, 3), dtype=np.float64)

    def pbc_delta(delta: float, length: float) -> float:
        delta = abs(delta)
        return delta if delta <= 0.5 * length else length - delta

    tries = 0
    for i in range(n_agents):
        while True:
            if tries >= max_tries:
                raise RuntimeError(
                    "init_positions_no_overlap exceeded max_tries; "
                    "reduce n_agents/margin/pad or enlarge the domain."
                )
            tries += 1
            candidate = np.array(
                [
                    rng.uniform(pad, nx - pad),
                    rng.uniform(pad, ny - pad),
                    rng.uniform(pad, nz - pad),
                ],
                dtype=np.float64,
            )
            ok = True
            for j in range(i):
                dx = pbc_delta(candidate[0] - pos[j, 0], nx)
                dy = pbc_delta(candidate[1] - pos[j, 1], ny)
                dz = pbc_delta(candidate[2] - pos[j, 2], nz)
                dij = (dx * dx + dy * dy + dz * dz) ** 0.5
                if dij < (rads[i] + rads[j] + margin):
                    ok = False
                    break
            if ok:
                pos[i] = candidate
                break
    return pos


@njit(parallel=True, fastmath=True)
def lbm_step_jit(f, force, tau, C, W, CS2, max_vel, q_count, fcoll_max):
    nx, ny, nz, _ = f.shape
    fnew = np.empty_like(f)
    rho = np.empty((nx, ny, nz), dtype=np.float64)
    u = np.empty((nx, ny, nz, 3), dtype=np.float64)
    inv_cs2 = 1.0 / CS2
    inv_cs4 = inv_cs2 * inv_cs2
    guo_prefactor = 1.0 - 1.0 / (2.0 * tau)

    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                dens = 0.0
                momx = 0.0
                momy = 0.0
                momz = 0.0
                for q in range(q_count):
                    fq = f[i, j, k, q]
                    dens += fq
                    momx += fq * C[q, 0]
                    momy += fq * C[q, 1]
                    momz += fq * C[q, 2]
                if dens <= 1e-14:
                    dens = 1e-14
                rho[i, j, k] = dens
                ux = (momx + 0.5 * force[i, j, k, 0]) / dens
                uy = (momy + 0.5 * force[i, j, k, 1]) / dens
                uz = (momz + 0.5 * force[i, j, k, 2]) / dens
                speed = (ux * ux + uy * uy + uz * uz) ** 0.5
                if speed > max_vel:
                    fac = max_vel / speed
                    ux *= fac
                    uy *= fac
                    uz *= fac
                u[i, j, k, 0] = ux
                u[i, j, k, 1] = uy
                u[i, j, k, 2] = uz

    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                ux = u[i, j, k, 0]
                uy = u[i, j, k, 1]
                uz = u[i, j, k, 2]
                usq = ux * ux + uy * uy + uz * uz
                Fx = force[i, j, k, 0]
                Fy = force[i, j, k, 1]
                Fz = force[i, j, k, 2]
                for q in range(q_count):
                    ci0 = C[q, 0]
                    ci1 = C[q, 1]
                    ci2 = C[q, 2]
                    cu = ux * ci0 + uy * ci1 + uz * ci2
                    feq = rho[i, j, k] * W[q] * (
                        1.0 + cu * inv_cs2 + 0.5 * cu * cu * inv_cs4 - 0.5 * usq * inv_cs2
                    )
                    fcoll = f[i, j, k, q] - (f[i, j, k, q] - feq) / tau
                    # Guo forcing term: w_i (1 - 1/(2 tau)) [((c_i-u)/cs^2) + ((c_i.u)c_i/cs^4)] . F
                    force_term = W[q] * guo_prefactor * (
                        ((ci0 - ux) * inv_cs2 + cu * ci0 * inv_cs4) * Fx
                        + ((ci1 - uy) * inv_cs2 + cu * ci1 * inv_cs4) * Fy
                        + ((ci2 - uz) * inv_cs2 + cu * ci2 * inv_cs4) * Fz
                    )
                    fcoll += force_term
                    if fcoll < -fcoll_max:
                        fcoll = -fcoll_max
                    elif fcoll > fcoll_max:
                        fcoll = fcoll_max
                    ip = (i + ci0) % nx
                    jp = (j + ci1) % ny
                    kp = (k + ci2) % nz
                    fnew[ip, jp, kp, q] = fcoll

    return fnew, rho, u


@njit(fastmath=True)
def sample_rho_trilinear(rho, x, y, z):
    nx, ny, nz = rho.shape
    i0 = int(np.floor(x)) % nx
    j0 = int(np.floor(y)) % ny
    k0 = int(np.floor(z)) % nz
    fx = x - np.floor(x)
    fy = y - np.floor(y)
    fz = z - np.floor(z)
    value = 0.0
    for dx in range(2):
        wx = (1.0 - fx) if dx == 0 else fx
        ii = (i0 + dx) % nx
        for dy in range(2):
            wy = (1.0 - fy) if dy == 0 else fy
            jj = (j0 + dy) % ny
            for dz in range(2):
                wz = (1.0 - fz) if dz == 0 else fz
                kk = (k0 + dz) % nz
                value += wx * wy * wz * rho[ii, jj, kk]
    return value


@njit(fastmath=True)
def update_drones_jit(pos, vel, C_d, thrust, area, mass, rho, dt, max_vel, force_scheme_id):
    nx, ny, nz = rho.shape
    force = np.zeros((nx, ny, nz, 3), dtype=np.float64)
    n_agents = pos.shape[0]
    max_agent_speed_pre_clamp = 0.0
    agent_clamp_activation_count = 0
    agent_clamp_checks = 0
    for d in range(n_agents):
        x = pos[d, 0]
        y = pos[d, 1]
        z = pos[d, 2]
        i = int(np.floor(x)) % nx
        j = int(np.floor(y)) % ny
        k = int(np.floor(z)) % nz

        if force_scheme_id == 1:
            local_rho = sample_rho_trilinear(rho, x, y, z)
        else:
            local_rho = rho[i, j, k]

        vx = vel[d, 0]
        vy = vel[d, 1]
        vz = vel[d, 2]
        speed = (vx * vx + vy * vy + vz * vz) ** 0.5

        fd = -0.5 * C_d[d] * local_rho * area[d] * speed / mass[d]
        Fdx = fd * vx
        Fdy = fd * vy
        Fdz = fd * vz
        if speed > 1e-14:
            thrust_scale = thrust[d] / speed
            Fpx = thrust_scale * vx
            Fpy = thrust_scale * vy
            Fpz = thrust_scale * vz
        else:
            Fpx = 0.0
            Fpy = 0.0
            Fpz = 0.0

        Fx = Fdx + Fpx
        Fy = Fdy + Fpy
        Fz = Fdz + Fpz

        nvx = vx + Fx * dt
        nvy = vy + Fy * dt
        nvz = vz + Fz * dt
        sp = (nvx * nvx + nvy * nvy + nvz * nvz) ** 0.5
        if sp > max_agent_speed_pre_clamp:
            max_agent_speed_pre_clamp = sp
        agent_clamp_checks += 1
        if sp > max_vel:
            agent_clamp_activation_count += 1
            fac = max_vel / sp
            nvx *= fac
            nvy *= fac
            nvz *= fac
        vel[d, 0] = nvx
        vel[d, 1] = nvy
        vel[d, 2] = nvz

        # Equal-and-opposite reaction force to the fluid.
        if force_scheme_id == 1:
            fx = x - np.floor(x)
            fy = y - np.floor(y)
            fz = z - np.floor(z)
            for dx in range(2):
                wx = (1.0 - fx) if dx == 0 else fx
                ii = (i + dx) % nx
                for dy in range(2):
                    wy = (1.0 - fy) if dy == 0 else fy
                    jj = (j + dy) % ny
                    for dz in range(2):
                        wz = (1.0 - fz) if dz == 0 else fz
                        kk = (k + dz) % nz
                        w = wx * wy * wz
                        force[ii, jj, kk, 0] -= w * Fx
                        force[ii, jj, kk, 1] -= w * Fy
                        force[ii, jj, kk, 2] -= w * Fz
        else:
            force[i, j, k, 0] -= Fx
            force[i, j, k, 1] -= Fy
            force[i, j, k, 2] -= Fz

        pos[d, 0] = (x + nvx * dt) % nx
        pos[d, 1] = (y + nvy * dt) % ny
        pos[d, 2] = (z + nvz * dt) % nz

    return force, max_agent_speed_pre_clamp, agent_clamp_activation_count, agent_clamp_checks


@njit(fastmath=True)
def _minimum_image(delta, length):
    if delta > 0.5 * length:
        delta -= length
    elif delta < -0.5 * length:
        delta += length
    return delta


@njit(fastmath=True)
def handle_contacts_jit(pos, vel, rads, contact_state, nx, ny, nz):
    n_agents = pos.shape[0]
    raw_count = 0
    unique_count = 0
    active_contacts = 0

    for a in range(n_agents):
        for b in range(a + 1, n_agents):
            dx = _minimum_image(pos[b, 0] - pos[a, 0], nx)
            dy = _minimum_image(pos[b, 1] - pos[a, 1], ny)
            dz = _minimum_image(pos[b, 2] - pos[a, 2], nz)
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            rsum = rads[a] + rads[b]
            in_contact = dist < rsum

            if in_contact:
                raw_count += 1
                active_contacts += 1
                if contact_state[a, b] == 0:
                    unique_count += 1
                contact_state[a, b] = 1

                if dist < 1e-12:
                    nx_ = 1.0
                    ny_ = 0.0
                    nz_ = 0.0
                    dist = 1e-12
                else:
                    nx_ = dx / dist
                    ny_ = dy / dist
                    nz_ = dz / dist

                dvx = vel[b, 0] - vel[a, 0]
                dvy = vel[b, 1] - vel[a, 1]
                dvz = vel[b, 2] - vel[a, 2]
                vn = dvx * nx_ + dvy * ny_ + dvz * nz_
                # Apply impulse only if the pair is approaching.
                if vn < 0.0:
                    vel[a, 0] += vn * nx_
                    vel[a, 1] += vn * ny_
                    vel[a, 2] += vn * nz_
                    vel[b, 0] -= vn * nx_
                    vel[b, 1] -= vn * ny_
                    vel[b, 2] -= vn * nz_

                # Minimal overlap correction to avoid indefinite repeated contacts.
                overlap = rsum - dist
                if overlap > 0.0:
                    corr = 0.5 * overlap + 1e-9
                    pos[a, 0] = (pos[a, 0] - corr * nx_) % nx
                    pos[a, 1] = (pos[a, 1] - corr * ny_) % ny
                    pos[a, 2] = (pos[a, 2] - corr * nz_) % nz
                    pos[b, 0] = (pos[b, 0] + corr * nx_) % nx
                    pos[b, 1] = (pos[b, 1] + corr * ny_) % ny
                    pos[b, 2] = (pos[b, 2] + corr * nz_) % nz
            else:
                contact_state[a, b] = 0

    return raw_count, unique_count, active_contacts


def simulate_jit(
    nx: int,
    ny: int,
    nz: int,
    rho0: float,
    n_agents: int,
    steps: int,
    tau: float = TAU_BGK,
    dt: float = 1.0,
    seed: int = DEFAULT_SEED,
    force_scheme: str = "nearest",
    agent_radius: float = 5.0,
    agent_mass: float = 1.0,
    v0_max: float = 0.2,
    max_vel: float = MAX_VEL,
) -> Tuple[RunResult, pd.DataFrame]:
    if force_scheme not in {"nearest", "trilinear"}:
        raise ValueError("force_scheme must be 'nearest' or 'trilinear'.")
    force_scheme_id = 1 if force_scheme == "trilinear" else 0
    rng = np.random.default_rng(seed)

    f = np.zeros((nx, ny, nz, Q), dtype=np.float64)
    for q in range(Q):
        f[..., q] = W[q] * rho0

    rads = np.full(n_agents, agent_radius, dtype=np.float64)
    C_d = np.full(n_agents, 1.0, dtype=np.float64)
    thrust = np.full(n_agents, 0.1, dtype=np.float64)
    area = np.full(n_agents, np.pi * agent_radius * agent_radius, dtype=np.float64)
    mass = np.full(n_agents, agent_mass, dtype=np.float64)

    pos = init_positions_no_overlap(nx, ny, nz, rads, rng=rng, margin=0.5, pad=10.0)
    vel = rng.uniform(-v0_max, v0_max, size=(n_agents, 3)).astype(np.float64)
    contact_state = np.zeros((n_agents, n_agents), dtype=np.uint8)

    total_raw = 0
    total_unique = 0
    max_rho_observed = float(rho0)
    max_u_observed = 0.0
    max_agent_speed_pre_clamp_observed = 0.0
    agent_clamp_activation_count = 0
    agent_clamp_checks = 0
    stable = True
    stop_reason = "completed"
    records: List[Dict[str, float]] = []

    for step in range(steps):
        rho_before = f.sum(axis=3)
        force, max_pre_step, clamp_count_step, clamp_checks_step = update_drones_jit(
            pos, vel, C_d, thrust, area, mass, rho_before, dt, max_vel, force_scheme_id
        )
        max_agent_speed_pre_clamp_observed = max(max_agent_speed_pre_clamp_observed, float(max_pre_step))
        agent_clamp_activation_count += int(clamp_count_step)
        agent_clamp_checks += int(clamp_checks_step)
        f, rho, u = lbm_step_jit(f, force, tau, C, W, CS2, max_vel, Q, FCOLL_MAX)

        max_rho = float(rho.max())
        mean_rho = float(rho.mean())
        u_norm = np.sqrt((u * u).sum(axis=3))
        max_u = float(u_norm.max())
        mean_u = float(u_norm.mean())
        max_rho_observed = max(max_rho_observed, max_rho)
        max_u_observed = max(max_u_observed, max_u)

        raw_step, unique_step, active_contacts = handle_contacts_jit(pos, vel, rads, contact_state, nx, ny, nz)
        total_raw += int(raw_step)
        total_unique += int(unique_step)
        agent_speed = np.sqrt((vel * vel).sum(axis=1))
        records.append(
            {
                "step": step,
                "raw_contact_detections": int(raw_step),
                "unique_contact_events": int(unique_step),
                "active_contacts": int(active_contacts),
                "cumulative_raw_contact_detections": int(total_raw),
                "cumulative_unique_contact_events": int(total_unique),
                "max_rho": max_rho,
                "mean_rho": mean_rho,
                "rho_safety_margin": float(RHO_THRESHOLD - max_rho),
                "max_u": max_u,
                "mean_u": mean_u,
                "velocity_safety_margin": float(max_vel * 1.1 - max_u),
                "max_agent_speed_pre_clamp": float(max_pre_step),
                "agent_clamp_activations_step": int(clamp_count_step),
                "agent_clamp_checks_step": int(clamp_checks_step),
                "agent_clamp_activation_frequency_cumulative": float(
                    agent_clamp_activation_count / agent_clamp_checks
                    if agent_clamp_checks > 0 else 0.0
                ),
                "mean_agent_speed": float(agent_speed.mean()),
                "max_agent_speed": float(agent_speed.max()),
            }
        )

        if max_rho > RHO_THRESHOLD:
            stable = False
            stop_reason = f"rho_threshold_exceeded_at_step_{step}"
            break
        if max_u > max_vel * 1.1:
            stable = False
            stop_reason = f"velocity_threshold_exceeded_at_step_{step}"
            break

    result = RunResult(
        nx=nx,
        ny=ny,
        nz=nz,
        rho0=float(rho0),
        n_agents=int(n_agents),
        steps_requested=int(steps),
        steps_completed=len(records),
        seed=int(seed),
        force_scheme=force_scheme,
        stable=bool(stable),
        stop_reason=stop_reason,
        raw_contact_detections=int(total_raw),
        unique_contact_events=int(total_unique),
        final_active_contacts=int(records[-1]["active_contacts"]) if records else 0,
        max_rho_observed=float(max_rho_observed),
        max_u_observed=float(max_u_observed),
        max_agent_speed_pre_clamp_observed=float(max_agent_speed_pre_clamp_observed),
        agent_clamp_activation_count=int(agent_clamp_activation_count),
        agent_clamp_checks=int(agent_clamp_checks),
        agent_clamp_activation_frequency=float(
            agent_clamp_activation_count / agent_clamp_checks if agent_clamp_checks > 0 else 0.0
        ),
    )
    return result, pd.DataFrame.from_records(records)


def summarize(values: List[float], label: str) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {f"{label}_n": 0}
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0.0:
        ci_low, ci_high = t.interval(0.95, n - 1, loc=mean, scale=sd / np.sqrt(n))
    else:
        ci_low = ci_high = mean
    return {
        f"{label}_n": int(n),
        f"{label}_mean": mean,
        f"{label}_median": float(np.median(arr)),
        f"{label}_sd": sd,
        f"{label}_ci95_low": float(ci_low),
        f"{label}_ci95_high": float(ci_high),
        f"{label}_iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
    }




TIME_SERIES_METRICS = [
    "raw_contact_detections",
    "unique_contact_events",
    "active_contacts",
    "cumulative_raw_contact_detections",
    "cumulative_unique_contact_events",
    "max_rho",
    "mean_rho",
    "rho_safety_margin",
    "max_u",
    "mean_u",
    "velocity_safety_margin",
    "mean_agent_speed",
    "max_agent_speed",
    "max_agent_speed_pre_clamp",
    "agent_clamp_activations_step",
    "agent_clamp_checks_step",
    "agent_clamp_activation_frequency_cumulative",
]


def ci95_mean(values: np.ndarray) -> Tuple[float, float]:
    """Return a 95% t-interval for the mean, with safe handling of n<=1 or sd=0."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, np.nan
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0.0:
        low, high = t.interval(0.95, n - 1, loc=mean, scale=sd / np.sqrt(n))
        return float(low), float(high)
    return mean, mean


def summarize_time_series_by_step(df: pd.DataFrame, metrics: List[str] | None = None) -> pd.DataFrame:
    """Create long-format stepwise summaries for all selected time-series metrics."""
    if df.empty:
        return pd.DataFrame()
    metrics = metrics or [m for m in TIME_SERIES_METRICS if m in df.columns]
    rows: List[Dict[str, float]] = []
    for step, group in df.groupby("step", sort=True):
        for metric in metrics:
            if metric not in group.columns:
                continue
            arr = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if len(arr) == 0:
                continue
            ci_low, ci_high = ci95_mean(arr)
            rows.append(
                {
                    "step": int(step),
                    "metric": metric,
                    "n": int(len(arr)),
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
                }
            )
    return pd.DataFrame(rows)


def compute_run_time_series_metrics(long_df: pd.DataFrame) -> pd.DataFrame:
    """Reduce the long time-series table to one transient-diagnostics row per attempt."""
    if long_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, float]] = []
    for (attempt, seed), group in long_df.groupby(["attempt", "seed"], sort=True):
        group = group.sort_values("step")
        raw_positive = group.loc[group["raw_contact_detections"] > 0, "step"]
        unique_positive = group.loc[group["unique_contact_events"] > 0, "step"]
        rows.append(
            {
                "attempt": int(attempt),
                "seed": int(seed),
                "stable": bool(group["stable"].iloc[0]),
                "stop_reason": str(group["stop_reason"].iloc[0]),
                "steps_completed": int(group["step"].max() + 1),
                "total_raw_contact_detections": int(group["raw_contact_detections"].sum()),
                "total_unique_contact_events": int(group["unique_contact_events"].sum()),
                "first_raw_contact_step": int(raw_positive.min()) if len(raw_positive) else -1,
                "first_unique_contact_step": int(unique_positive.min()) if len(unique_positive) else -1,
                "peak_active_contacts": int(group["active_contacts"].max()),
                "peak_max_rho": float(group["max_rho"].max()),
                "minimum_rho_safety_margin": float(group["rho_safety_margin"].min()),
                "peak_max_u": float(group["max_u"].max()),
                "minimum_velocity_safety_margin": float(group["velocity_safety_margin"].min()),
                "mean_agent_speed_over_time": float(group["mean_agent_speed"].mean()),
                "peak_agent_speed": float(group["max_agent_speed"].max()),
            }
        )
    return pd.DataFrame(rows)


def plot_time_series_summary(summary_df: pd.DataFrame, output_dir: Path, prefix: str = "stable") -> None:
    """Write simple manuscript-support plots from stepwise time-series summaries."""
    if summary_df.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        with open(output_dir / "plotting_skipped.txt", "w", encoding="utf-8") as f:
            f.write(f"Plotting skipped because matplotlib could not be imported: {exc}\n")
        return

    def _plot_metric(metric: str, ylabel: str, filename: str, threshold: float | None = None) -> None:
        data = summary_df[summary_df["metric"] == metric].sort_values("step")
        if data.empty:
            return
        fig = plt.figure(figsize=(7.0, 4.5))
        plt.plot(data["step"], data["mean"], label="Mean")
        if "ci95_low" in data and "ci95_high" in data:
            plt.fill_between(data["step"], data["ci95_low"], data["ci95_high"], alpha=0.2, label="95% CI")
        if threshold is not None:
            plt.axhline(threshold, linestyle="--", label="Threshold")
        plt.xlabel("Time step")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        fig.savefig(output_dir / filename, dpi=300)
        plt.close(fig)

    _plot_metric(
        "cumulative_raw_contact_detections",
        "Cumulative raw contact detections",
        f"{prefix}_cumulative_raw_contacts.png",
    )
    _plot_metric(
        "cumulative_unique_contact_events",
        "Cumulative unique contact events",
        f"{prefix}_cumulative_unique_contacts.png",
    )
    _plot_metric("active_contacts", "Active contacts", f"{prefix}_active_contacts.png")
    _plot_metric("max_rho", "Maximum density", f"{prefix}_max_rho.png", threshold=RHO_THRESHOLD)
    _plot_metric("max_u", "Maximum fluid speed", f"{prefix}_max_u.png", threshold=MAX_VEL * 1.1)
    _plot_metric("mean_agent_speed", "Mean agent speed", f"{prefix}_mean_agent_speed.png")


def write_time_series_analysis(
    output_dir: Path,
    attempts_df: pd.DataFrame,
    time_series_frames: List[pd.DataFrame],
    make_plots: bool = True,
) -> None:
    """Write raw data and time-series analysis files for audit and manuscript figures."""
    if output_dir is None or not time_series_frames:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    long_df = pd.concat(time_series_frames, ignore_index=True)
    _safe_to_csv(long_df, output_dir / "all_timeseries_long.csv", index=False)

    stable_long = long_df[long_df["stable"] == True].copy()  # noqa: E712
    if not stable_long.empty:
        _safe_to_csv(stable_long, output_dir / "stable_timeseries_long.csv", index=False)

    all_summary = summarize_time_series_by_step(long_df)
    _safe_to_csv(all_summary, output_dir / "all_timeseries_summary_by_step.csv", index=False)

    stable_summary = summarize_time_series_by_step(stable_long)
    if not stable_summary.empty:
        _safe_to_csv(stable_summary, output_dir / "stable_timeseries_summary_by_step.csv", index=False)

    run_metrics = compute_run_time_series_metrics(long_df)
    _safe_to_csv(run_metrics, output_dir / "run_timeseries_metrics.csv", index=False)

    if not attempts_df.empty and "stop_reason" in attempts_df.columns:
        stop_summary = (
            attempts_df["stop_reason"]
            .value_counts(dropna=False)
            .rename_axis("stop_reason")
            .reset_index(name="attempt_count")
        )
        _safe_to_csv(stop_summary, output_dir / "instability_stop_reason_counts.csv", index=False)

    report_lines = [
        "# Time-series analysis report",
        "",
        f"Total attempts: {len(attempts_df)}",
        f"Stable attempts: {int(attempts_df['stable'].sum()) if 'stable' in attempts_df else 'NA'}",
        f"Unstable attempts: {int((~attempts_df['stable']).sum()) if 'stable' in attempts_df else 'NA'}",
        "",
        "Generated files:",
        "- all_timeseries_long.csv: raw per-step data for every attempt.",
        "- stable_timeseries_long.csv: raw per-step data restricted to completed stable attempts.",
        "- all_timeseries_summary_by_step.csv: stepwise summary over all attempts.",
        "- stable_timeseries_summary_by_step.csv: stepwise summary over stable attempts only.",
        "- run_timeseries_metrics.csv: one row per attempt with transient diagnostics.",
        "- instability_stop_reason_counts.csv: frequency of stopping reasons.",
        "",
        "Recommended manuscript use:",
        "- Use stable_timeseries_summary_by_step.csv for temporal-convergence and transient-contact figures.",
        "- Use run_timeseries_metrics.csv to report first-contact time, peak active contacts, and stability margins.",
        "- Use instability_stop_reason_counts.csv as a reproducibility/audit supplement, not as a performance result.",
    ]
    if not run_metrics.empty:
        stable_metrics = run_metrics[run_metrics["stable"] == True]  # noqa: E712
        if not stable_metrics.empty:
            report_lines.extend(
                [
                    "",
                    "Stable-run transient summary:",
                    f"- Mean total raw detections: {stable_metrics['total_raw_contact_detections'].mean():.4g}",
                    f"- Mean total unique events: {stable_metrics['total_unique_contact_events'].mean():.4g}",
                    f"- Mean peak active contacts: {stable_metrics['peak_active_contacts'].mean():.4g}",
                    f"- Minimum density safety margin across stable attempts: {stable_metrics['minimum_rho_safety_margin'].min():.4g}",
                    f"- Minimum velocity safety margin across stable attempts: {stable_metrics['minimum_velocity_safety_margin'].min():.4g}",
                ]
            )
    with open(output_dir / "time_series_analysis_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    if make_plots:
        plot_time_series_summary(stable_summary, output_dir, prefix="stable")
        plot_time_series_summary(all_summary, output_dir, prefix="all_attempts")

def run_multiple(
    nx: int,
    ny: int,
    nz: int,
    rho0: float,
    stable_runs: int,
    n_agents: int,
    steps: int,
    seed: int = DEFAULT_SEED,
    force_scheme: str = "nearest",
    max_attempts_factor: int = 3,
    output_dir: Path | None = None,
    agent_radius: float = 5.0,
    agent_mass: float = 5.0,
    v0_max: float = 0.2,
    max_vel: float = MAX_VEL,
    write_analysis: bool = True,
    make_plots: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    results: List[RunResult] = []
    stable_results: List[RunResult] = []
    time_series_frames: List[pd.DataFrame] = []
    max_attempts = stable_runs * max_attempts_factor

    for attempt in trange(max_attempts, desc="attempts"):
        run_seed = seed + attempt
        result, time_series = simulate_jit(
            nx=nx,
            ny=ny,
            nz=nz,
            rho0=rho0,
            n_agents=n_agents,
            steps=steps,
            seed=run_seed,
            force_scheme=force_scheme,
            agent_radius=agent_radius,
            agent_mass=agent_mass,
            v0_max=v0_max,
            max_vel=max_vel,
        )
        results.append(result)

        # Add run metadata directly to each per-step record so the CSV files are
        # self-contained raw data suitable for later pooling across conditions.
        time_series = time_series.copy()
        time_series.insert(0, "attempt", int(attempt))
        time_series.insert(1, "seed", int(run_seed))
        time_series.insert(2, "stable", bool(result.stable))
        time_series.insert(3, "stop_reason", result.stop_reason)
        time_series.insert(4, "rho0", float(rho0))
        time_series.insert(5, "n_agents", int(n_agents))
        time_series.insert(6, "grid", f"{nx}x{ny}x{nz}")
        time_series.insert(7, "force_scheme", force_scheme)
        time_series.insert(8, "agent_radius", float(agent_radius))
        time_series.insert(9, "agent_mass", float(agent_mass))
        time_series.insert(10, "v0_max", float(v0_max))
        time_series.insert(11, "max_vel", float(max_vel))
        time_series_frames.append(time_series)

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = (
                f"rho{rho0}_N{n_agents}_{nx}x{ny}x{nz}_"
                f"{force_scheme}_r{agent_radius}_m{agent_mass}_v{v0_max}_seed{run_seed}"
            )
            _safe_to_csv(time_series, output_dir / f"{stem}_timeseries.csv", index=False)
        if result.stable:
            stable_results.append(result)
        if len(stable_results) >= stable_runs:
            break

    attempts_df = pd.DataFrame([asdict(r) for r in results])

    if output_dir is not None:
        _safe_to_csv(attempts_df, output_dir / "attempt_log.csv", index=False)
        if write_analysis:
            write_time_series_analysis(output_dir, attempts_df, time_series_frames, make_plots=make_plots)

    if len(stable_results) < stable_runs:
        if output_dir is not None:
            with open(output_dir / "failed_run_config.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "nx": nx,
                        "ny": ny,
                        "nz": nz,
                        "rho0": rho0,
                        "stable_runs_requested": stable_runs,
                        "stable_runs_obtained": len(stable_results),
                        "attempts_total": len(results),
                        "n_agents": n_agents,
                        "steps": steps,
                        "seed": seed,
                        "force_scheme": force_scheme,
                        "agent_radius": agent_radius,
                        "agent_mass": agent_mass,
                        "v0_max": v0_max,
                        "rho_threshold": RHO_THRESHOLD,
                        "max_vel": max_vel,
                        "tau_bgk": TAU_BGK,
                        "analysis_outputs_written": bool(write_analysis),
                    },
                    f,
                    indent=2,
                )
        raise RuntimeError(
            f"Only {len(stable_results)} stable runs obtained out of "
            f"{stable_runs} requested after {len(results)} attempts. "
            "Do not use this configuration for manuscript tables. "
            "Diagnostics and time-series analysis were written to output_dir if provided."
        )

    summary = {
        "grid": f"{nx}x{ny}x{nz}",
        "rho0": rho0,
        "n_agents": n_agents,
        "steps": steps,
        "force_scheme": force_scheme,
        "agent_radius": agent_radius,
        "agent_mass": agent_mass,
        "v0_max": v0_max,
        "max_vel": max_vel,
        "stable_runs_requested": stable_runs,
        "attempts_total": len(results),
        "stable_runs_obtained": len(stable_results),
        "unstable_attempts": len(results) - len(stable_results),
    }
    summary.update(summarize([r.raw_contact_detections for r in stable_results], "raw_detections"))
    summary.update(summarize([r.unique_contact_events for r in stable_results], "unique_events"))
    summary.update(summarize([r.steps_completed for r in stable_results], "steps_completed"))
    summary.update(summarize([r.max_agent_speed_pre_clamp_observed for r in stable_results], "max_agent_speed_pre_clamp"))
    summary.update(summarize([r.agent_clamp_activation_frequency for r in stable_results], "agent_clamp_activation_frequency"))

    summary_df = pd.DataFrame([summary])

    if output_dir is not None:
        _safe_to_csv(attempts_df, output_dir / "attempt_log.csv", index=False)
        _safe_to_csv(summary_df, output_dir / "summary.csv", index=False)
        with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "nx": nx,
                    "ny": ny,
                    "nz": nz,
                    "rho0": rho0,
                    "stable_runs": stable_runs,
                    "n_agents": n_agents,
                    "steps": steps,
                    "seed": seed,
                    "force_scheme": force_scheme,
                    "agent_radius": agent_radius,
                    "agent_mass": agent_mass,
                    "v0_max": v0_max,
                    "rho_threshold": RHO_THRESHOLD,
                    "max_vel": max_vel,
                    "tau_bgk": TAU_BGK,
                    "analysis_outputs_written": bool(write_analysis),
                    "plots_written": bool(make_plots),
                },
                f,
                indent=2,
            )
    return summary_df, attempts_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run revised LBM drone-proxy contact simulation.")
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--runs", type=int, default=1, help="number of stable runs requested")
    parser.add_argument("--agents", type=int, default=30)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--grid", type=str, default="100,100,100", help="nx,ny,nz")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force-scheme", choices=["nearest", "trilinear"], default="nearest")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--agent-radius", type=float, default=5.0)
    parser.add_argument("--agent-mass", type=float, default=1.0)
    parser.add_argument("--v0-max", type=float, default=0.2)
    parser.add_argument("--max-vel", type=float, default=MAX_VEL, help="velocity-clamp scale used for fluid and agent speed limits")
    parser.add_argument("--max-attempts-factor", type=int, default=3)
    parser.add_argument("--no-analysis", action="store_true", help="disable pooled raw-data/time-series analysis outputs")
    parser.add_argument("--no-plots", action="store_true", help="disable PNG plot generation")
    args = parser.parse_args()

    nx, ny, nz = [int(x.strip()) for x in args.grid.split(",")]
    summary, attempts = run_multiple(
        nx=nx,
        ny=ny,
        nz=nz,
        rho0=args.rho,
        stable_runs=args.runs,
        n_agents=args.agents,
        steps=args.steps,
        seed=args.seed,
        force_scheme=args.force_scheme,
        max_attempts_factor=args.max_attempts_factor,
        output_dir=args.output_dir,
        agent_radius=args.agent_radius,
        agent_mass=args.agent_mass,
        v0_max=args.v0_max,
        max_vel=args.max_vel,
        write_analysis=not args.no_analysis,
        make_plots=not args.no_plots,
    )
    print("\nSummary")
    print(summary.to_string(index=False))
    print("\nAttempts")
    print(attempts.to_string(index=False))


if __name__ == "__main__":
    main()
