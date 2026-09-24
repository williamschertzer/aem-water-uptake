#!/usr/bin/env python3
"""Analyse an aemwater uptake run against the total water chemical potential.

Saturation with liquid water is mu_w,m = mu_w,b, i.e.

    gap = (mu_ex,m - mu_ex,b) + kT ln(rho_w,m / rho_w,b) = 0

with rho = (N + 1) / V of the cell mu_ex was sampled in (the +1 is the ghost
or test particle; see ``aemwater.widom.water_number_density``). The excess
part alone is what runs before ``gap_definition = total_mu_v1`` stored and
stopped on; it omits the density term, which is negative in any membrane and
makes the excess-only gap reach zero at too low a water content.

For checkpoints that predate the change the total gap is reconstructed from
what is on disk:

* bulk mu_ex / stderr: result.json, e2e_summary.json, fep_bulk.json, or the
  constant implied by the stored excess-only gap (mu_ex - mu_gap);
* bulk density: result.json, the bulk cache (density + cell volume), or the
  bulk box (bulk/bulk.data water count, bulk_density.dat NPT-mean volume);
* membrane cell volume: the ``mu_volume`` column; otherwise, for FEP (NVT
  windows), the box of iter_NNN/relaxed.data, and for Widom the NPT mean.

Outputs (next to the figure): a per-iteration CSV with the excess gap, the
density term and the total gap, and a JSON summary including the
interpolated zero of the total gap (``aemwater.saturation``) when the run
brackets it. Without a trustworthy crossing the final loaded state is a lower
bound and is reported as such.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from aemwater.saturation import estimate_saturation_point
    from aemwater.widom import KB_KCAL
except ImportError:  # running from a checkout without an installed package
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from aemwater.saturation import estimate_saturation_point
    from aemwater.widom import KB_KCAL

#: g/mol for the rigid 3-site models used here (SPC/E, TIP3P, ...).
WATER_MOLAR_MASS = 18.01528
#: g/cm^3 -> molecules/A^3 is rho * N_A * 1e-24 / M.
AVOGADRO_E24 = 0.602214076
TOTAL_GAP_DEFINITION = "total_mu_v1"


def _read_trajectory(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Iterations plus the checkpoint header (gap_definition etc.)."""
    state_path = run_dir / "uptake_state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    csv_path = run_dir / "uptake_trajectory.csv"
    if state.get("iterations"):
        # The checkpoint is written every iteration; the CSV only at the end,
        # so the checkpoint is the more complete record of an unfinished run.
        frame = pd.DataFrame(state["iterations"])
    elif csv_path.is_file():
        frame = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"no uptake_state.json or uptake_trajectory.csv in {run_dir}")
    required = {"index", "mu_ex", "water_uptake_pct", "n_waters_after"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"trajectory is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("uptake trajectory contains no iterations")
    header = {k: v for k, v in state.items() if k != "iterations"}
    return frame.sort_values("index").reset_index(drop=True), header


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text()) if path.is_file() else None
    except json.JSONDecodeError:
        return None


def _run_config(run_dir: Path) -> dict:
    """The run's YAML config, if it can be found (for T and the mu method)."""
    import yaml

    candidates = []
    summary = _load_json(run_dir / "e2e_summary.json")
    if summary and summary.get("config"):
        candidates.append(Path(summary["config"]))
    candidates += sorted(run_dir.glob("*.yaml")) + sorted(run_dir.glob("*.yml"))
    for path in candidates:
        if path.is_file():
            loaded = yaml.safe_load(path.read_text()) or {}
            loaded["_path"] = str(path)
            return loaded
    return {}


def _mu_method(run_dir: Path, frame: pd.DataFrame, config: dict) -> str:
    for index in frame["index"]:
        stage = run_dir / f"iter_{int(index):03d}"
        if (stage / "fep_membrane.json").is_file() or (stage / "fep").is_dir():
            return "fep"
        if (stage / "mu.dat").is_file():
            return "widom"
    return str(config.get("mu_ex_method", "fep"))


def _temperature(run_dir: Path, config: dict, override: float | None) -> float:
    if override is not None:
        return override
    temperature = (config.get("md") or {}).get("temperature")
    if temperature is not None:
        return float(temperature)
    report = _load_json(run_dir / "fep_bulk.json")
    if report and "temperature_K" in report.get("combined", {}):
        return float(report["combined"]["temperature_K"])
    raise ValueError("temperature unknown; pass --temperature")


def _box_volume(data_path: Path) -> float:
    text = data_path.read_text()
    lengths = []
    for axis in "xyz":
        match = re.search(rf"(?m)^\s*(\S+)\s+(\S+)\s+{axis}lo\s+{axis}hi", text)
        if match is None:
            raise ValueError(f"no {axis} bounds in {data_path}")
        lengths.append(float(match.group(2)) - float(match.group(1)))
    if re.search(r"(?m)^\s*\S+\s+\S+\s+\S+\s+xy\s+xz\s+yz", text):
        raise ValueError(f"{data_path} is triclinic; volume needs the tilt terms")
    return float(np.prod(lengths))


def _atom_count(data_path: Path) -> int:
    match = re.search(r"(?m)^\s*(\d+)\s+atoms\s*$", data_path.read_text())
    if match is None:
        raise ValueError(f"could not read the atom count from {data_path}")
    return int(match.group(1))


def _second_half_mean(dat_path: Path, column: int) -> float:
    rows = [line.split() for line in dat_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    values = np.array([float(r[column]) for r in rows])
    return float(values[len(values) // 2:].mean())


def _bulk_reference(run_dir: Path, frame: pd.DataFrame, header: dict,
                    mu_override: float | None, rho_override: float | None) -> dict:
    """Bulk mu_ex, its stderr and the (N_b + 1)/V_b number density."""
    result = _load_json(run_dir / "result.json") or {}
    summary = _load_json(run_dir / "e2e_summary.json") or {}
    report = _load_json(run_dir / "fep_bulk.json") or {}
    notes = []

    mu = stderr = None
    if mu_override is not None:
        mu, mu_source = mu_override, "--water-mu"
    elif "bulk_mu_ex_kcal_mol" in result:
        mu, mu_source = float(result["bulk_mu_ex_kcal_mol"]), "result.json"
    elif "mu_ex" in summary.get("bulk", {}):
        mu, mu_source = float(summary["bulk"]["mu_ex"]), "e2e_summary.json"
        stderr = summary["bulk"].get("stderr")
    elif "mu_ex_kcal_mol" in report.get("combined", {}):
        mu, mu_source = float(report["combined"]["mu_ex_kcal_mol"]), "fep_bulk.json"
    else:
        # A pre-total_mu checkpoint stores mu_gap = mu_ex - mu_bulk, so the
        # bulk value is recoverable exactly -- if it is a single constant.
        if header.get("gap_definition") == TOTAL_GAP_DEFINITION:
            raise ValueError("bulk mu_ex unavailable; pass --water-mu")
        implied = (frame["mu_ex"] - frame["mu_gap"]).dropna()
        if implied.empty or implied.max() - implied.min() > 1e-6:
            raise ValueError("bulk mu_ex unavailable; pass --water-mu")
        mu, mu_source = float(implied.iloc[0]), "implied by stored excess gap"
    if stderr is None and "stderr_kcal_mol" in report.get("combined", {}):
        stderr = report["combined"]["stderr_kcal_mol"]
    if stderr is None:
        notes.append("bulk stderr unavailable; endpoint CI excludes it")
        stderr = 0.0

    rho = None
    if rho_override is not None:
        rho, rho_source = rho_override, "--bulk-rho"
    elif result.get("bulk_rho_water_per_A3"):
        rho, rho_source = float(result["bulk_rho_water_per_A3"]), "result.json"
    if rho is None:
        # Bulk cache payloads carry the density and cell volume of the cell
        # mu_ex was sampled in; choose the one whose mu_ex matches.
        caches = []
        for folder in (run_dir / "bulk_cache", run_dir.parent / "bulk_cache"):
            caches += sorted(folder.glob("bulk*.json")) if folder.is_dir() else []
        for path in caches:
            payload = _load_json(path) or {}
            if {"density", "volume", "mu_ex"} <= payload.keys() and \
                    abs(float(payload["mu_ex"]) - mu) < 1e-6:
                volume = float(payload["volume"])
                n_bulk = round(float(payload["density"]) * volume
                               * AVOGADRO_E24 / WATER_MOLAR_MASS)
                rho = (n_bulk + 1) / volume
                rho_source = f"{path.name} (N_b={n_bulk}, V_b={volume:.1f} A^3)"
                break
    if rho is None:
        bulk_dir = run_dir / "bulk"
        data = next((p for p in (bulk_dir / "bulk.data", bulk_dir / "bulk_final.data")
                     if p.is_file()), None)
        if data is not None:
            n_bulk = _atom_count(data) // 3
            dens = bulk_dir / "bulk_density.dat"
            if dens.is_file():
                volume = _second_half_mean(dens, 2)
                how = "NPT-mean volume"
            else:
                volume = _box_volume(data)
                how = "final box"
            rho = (n_bulk + 1) / volume
            rho_source = f"bulk/ (N_b={n_bulk}, V_b={volume:.1f} A^3, {how})"
    if rho is None:
        raise ValueError("bulk water density unavailable; pass --bulk-rho "
                         "(molecules/A^3, e.g. 0.0334 for SPC/E at 1 bar)")
    return {"mu_ex": mu, "stderr": float(stderr), "rho": rho,
            "mu_source": mu_source, "rho_source": rho_source, "notes": notes}


def _membrane_volumes(run_dir: Path, frame: pd.DataFrame, method: str) -> tuple[np.ndarray, list[str]]:
    volumes, sources = [], []
    for _, row in frame.iterrows():
        stored = row.get("mu_volume")
        if stored is not None and pd.notna(stored):
            volumes.append(float(stored)); sources.append("mu_volume")
            continue
        relaxed = run_dir / f"iter_{int(row['index']):03d}" / "relaxed.data"
        if method == "fep" and relaxed.is_file():
            volumes.append(_box_volume(relaxed)); sources.append("relaxed.data (NVT cell)")
        else:
            volumes.append(float(row["volume"])); sources.append("NPT mean")
    return np.array(volumes), sources


def analyze_run(run_dir: Path, water_mu: float | None = None,
                bulk_rho: float | None = None,
                temperature: float | None = None) -> tuple[pd.DataFrame, dict]:
    """Per-iteration total-mu table and a summary with the saturation point."""
    frame, header = _read_trajectory(run_dir)
    config = _run_config(run_dir)
    method = _mu_method(run_dir, frame, config)
    kt = KB_KCAL * _temperature(run_dir, config, temperature)
    bulk = _bulk_reference(run_dir, frame, header, water_mu, bulk_rho)

    volumes, volume_sources = _membrane_volumes(run_dir, frame, method)
    table = frame.copy()
    table["mu_volume"] = volumes
    table["rho_water"] = (table["n_waters_after"].astype(float) + 1.0) / volumes
    table["mu_gap_excess"] = table["mu_ex"] - bulk["mu_ex"]
    table["density_term"] = kt * np.log(table["rho_water"] / bulk["rho"])
    table["mu_gap_total"] = table["mu_gap_excess"] + table["density_term"]
    table["gap_stderr"] = np.hypot(table.get("mu_ex_stderr", 0.0).fillna(0.0),
                                   bulk["stderr"])
    if header.get("gap_definition") == TOTAL_GAP_DEFINITION and "mu_gap" in table:
        drift = (table["mu_gap"] - table["mu_gap_total"]).abs().max()
        if drift > 1e-3:
            warnings.warn(f"reconstructed total gap differs from the stored one "
                          f"by up to {drift:.3g} kcal/mol")

    if "sampling_adequate" in table:
        adequate = table["sampling_adequate"].fillna(False).astype(bool)
        sampling_note = "per-iteration FEP/Widom gates from the checkpoint"
    else:
        adequate = pd.Series(True, index=table.index)
        sampling_note = ("not recorded (pre-gate checkpoint): every finite "
                         "estimate treated as adequate")
    adequate &= table["mu_ex"].notna()
    table["sampling_adequate"] = adequate
    table["crossed"] = adequate & (table["mu_gap_total"] >= 0)

    rows = [SimpleNamespace(index=int(r["index"]),
                            n_waters_before=int(r["n_waters_before"]),
                            n_waters_after=int(r["n_waters_after"]),
                            mu_gap=float(r["mu_gap_total"]),
                            mu_ex_stderr=(float(r["mu_ex_stderr"])
                                          if pd.notna(r.get("mu_ex_stderr")) else None),
                            sampling_adequate=bool(r["sampling_adequate"]),
                            saturated=bool(r["crossed"]))
            for _, r in table.iterrows()]
    point = estimate_saturation_point(rows, bulk_stderr=bulk["stderr"])

    n_final = int(table["n_waters_after"].iloc[-1])
    pct_per_water = _pct_per_water(table)
    lam_per_water = _lambda_per_water(table)
    endpoint = None
    if point is not None:
        endpoint = {
            **point.summary(),
            "water_uptake_wt_pct": _scale(point.n_waters, pct_per_water),
            "water_uptake_wt_pct_ci95": [_scale(point.n_waters_low, pct_per_water),
                                         _scale(point.n_waters_high, pct_per_water)],
            "lambda": _scale(point.n_waters, lam_per_water),
            "lambda_ci95": [_scale(point.n_waters_low, lam_per_water),
                            _scale(point.n_waters_high, lam_per_water)],
        }
    excess_cross = table.index[adequate & (table["mu_gap_excess"] >= 0)]
    summary = {
        "run_dir": str(run_dir),
        "config": config.get("_path"),
        "mu_ex_method": method,
        "temperature_K": kt / KB_KCAL,
        "checkpoint_gap_definition": header.get("gap_definition", "excess-only (legacy)"),
        "bulk": bulk,
        "membrane_volume_source": sorted(set(volume_sources)),
        "sampling_verdict": sampling_note,
        "n_iterations": len(table),
        "final_state": {
            "n_waters": n_final,
            "water_uptake_wt_pct": round(float(table["water_uptake_pct"].iloc[-1]), 3),
            "lambda": _finite(table["lambda_value"].iloc[-1]),
            "mu_gap_total": round(float(table["mu_gap_total"].iloc[-1]), 4),
            "mu_gap_excess": round(float(table["mu_gap_excess"].iloc[-1]), 4),
        },
        "density_term_range_kcal_mol": [round(float(table["density_term"].min()), 4),
                                        round(float(table["density_term"].max()), 4)],
        "total_gap_crossed": point is not None,
        "saturation_point": endpoint,
        "excess_only_first_crossing_n_waters": (
            int(table.loc[excess_cross[0], "n_waters_after"]) if len(excess_cross) else None),
        "interpretation": (
            "total gap brackets zero: saturation_point is the equilibrium estimate"
            if point is not None and point.bracketed else
            "first trustworthy point already at/above zero: saturation_point is an upper bound"
            if point is not None else
            "total gap never reached zero: final_state is a lower bound on uptake"),
    }
    summary["bulk"]["notes"] = bulk["notes"]
    return table, summary


def _finite(value):
    value = float(value)
    return round(value, 4) if math.isfinite(value) else None


def _pct_per_water(table: pd.DataFrame) -> float | None:
    wet = table[table["n_waters_after"] > 0]
    if wet.empty:
        return None
    return float((wet["water_uptake_pct"] / wet["n_waters_after"]).median())


def _lambda_per_water(table: pd.DataFrame) -> float | None:
    wet = table[(table["n_waters_after"] > 0) & np.isfinite(table["lambda_value"])]
    if wet.empty:
        return None
    return float((wet["lambda_value"] / wet["n_waters_after"]).median())


def _scale(n_waters: float, per_water: float | None):
    return None if per_water is None else round(float(n_waters) * per_water, 4)


def _dry_atom_counts(run_dir: Path) -> tuple[int, int]:
    data_path = run_dir / "dry" / "dry.data"
    if not data_path.is_file():
        data_path = run_dir / "dry" / "system.data"
    if not data_path.is_file():
        raise FileNotFoundError("no dry/dry.data or dry/system.data found")
    dry_atoms = _atom_count(data_path)
    composition_path = run_dir / "dry" / "composition.json"
    n_counterions = 0
    if composition_path.is_file():
        composition = json.loads(composition_path.read_text())
        n_counterions = int(composition.get("n_counterions", 0))
    polymer_atoms = dry_atoms - n_counterions
    if polymer_atoms < 0:
        raise ValueError("n_counterions exceeds the dry-system atom count")
    return polymer_atoms, n_counterions


def plot_run(run_dir: Path, output: Path, water_mu: float | None = None,
             bulk_rho: float | None = None, temperature: float | None = None) -> Path:
    """Three panels: excess mu, total gap vs water count, uptake with endpoint."""
    table, summary = analyze_run(run_dir, water_mu, bulk_rho, temperature)
    bulk = summary["bulk"]
    point = summary["saturation_point"]
    try:
        polymer_atoms, n_counterions = _dry_atom_counts(run_dir)
    except (FileNotFoundError, ValueError):
        polymer_atoms, n_counterions = None, 0

    fig, (ax_mu, ax_gap, ax_up) = plt.subplots(1, 3, figsize=(14, 4.4))
    blue, red, grey, orange = "#1f4e79", "#b04a3a", "0.55", "#c2410c"
    valid = table["mu_ex"].notna()
    yerr = table.loc[valid, "mu_ex_stderr"] if "mu_ex_stderr" in table else None
    bad = ~table["sampling_adequate"]

    # (a) excess chemical potentials -- a diagnostic, not the criterion
    ax_mu.axhspan(bulk["mu_ex"] - 1.96 * bulk["stderr"],
                  bulk["mu_ex"] + 1.96 * bulk["stderr"], color=red, alpha=0.1, lw=0)
    ax_mu.axhline(bulk["mu_ex"], color=red, ls="--", lw=1.4,
                  label=rf"bulk ({bulk['mu_ex']:.2f} kcal mol$^{{-1}}$)")
    ax_mu.errorbar(table.loc[valid, "n_waters_after"], table.loc[valid, "mu_ex"],
                   yerr=None if yerr is None else 1.96 * yerr, fmt="o-", color=blue,
                   lw=1.4, ms=4, capsize=3, label="membrane")
    ax_mu.set(xlabel="water molecules", ylabel=r"$\mu_{ex}$ of water (kcal mol$^{-1}$)",
              title="(a) Excess chemical potential")
    ax_mu.legend(frameon=False, fontsize=8)

    # (b) the criterion: total gap vs water count
    x = table["n_waters_after"]
    ax_gap.axhline(0, color="0.2", lw=0.9)
    ax_gap.plot(x, table["mu_gap_excess"], "s--", color=grey, ms=3.5, lw=1,
                label="excess only (legacy)")
    ax_gap.errorbar(x, table["mu_gap_total"], yerr=1.96 * table["gap_stderr"],
                    fmt="o-", color=orange, ms=4, lw=1.4, capsize=3,
                    label=r"total: + $kT\,\ln(\rho_m/\rho_b)$")
    if bad.any():
        ax_gap.plot(x[bad], table.loc[bad, "mu_gap_total"], "x", color="k", ms=7,
                    label="sampling gate failed")
    if point is not None:
        ax_gap.axvspan(point["n_waters_low"], point["n_waters_high"],
                       color=orange, alpha=0.12, lw=0)
        ax_gap.axvline(point["n_waters"], color=orange, lw=1)
    ax_gap.set(xlabel="water molecules",
               ylabel=r"$\mu_{w,m}-\mu_{w,b}$ (kcal mol$^{-1}$)",
               title="(b) Saturation criterion (95% bars)")
    ax_gap.legend(frameon=False, fontsize=8, loc="lower right")

    # (c) uptake, with the interpolated endpoint when there is one
    ax_up.plot(table["index"], table["water_uptake_pct"], "o-", color=blue, lw=1.4, ms=4)
    for it, up, n in zip(table["index"], table["water_uptake_pct"], x):
        ax_up.annotate(f"{n}", (it, up), xytext=(0, 7), textcoords="offset points",
                       ha="center", fontsize=7, color="0.3")
    if point is not None and point["water_uptake_wt_pct"] is not None:
        lo, hi = point["water_uptake_wt_pct_ci95"]
        ax_up.axhspan(lo, hi, color=orange, alpha=0.12, lw=0)
        ax_up.axhline(point["water_uptake_wt_pct"], color=orange, lw=1.2,
                      label=f"saturation: {point['water_uptake_wt_pct']:.1f} wt%")
        ax_up.legend(frameon=False, fontsize=8, loc="lower right")
    lines = []
    if polymer_atoms is not None:
        lines.append(f"Polymer atoms: {polymer_atoms:,}")
    if n_counterions:
        lines.append(f"Counterion atoms: {n_counterions:,}")
    lines.append(f"Water molecules: {int(x.iloc[-1]):,} (final)")
    lines.append("total gap crossed zero" if point is not None
                 else "not saturated: final state is a lower bound")
    ax_up.text(0.03, 0.97, "\n".join(lines), transform=ax_up.transAxes, ha="left",
               va="top", fontsize=8.5,
               bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                     "alpha": 0.9, "edgecolor": "0.75"})
    ax_up.set(xlabel="iteration", ylabel="water uptake (wt %)",
              title="(c) Water uptake (labels = waters)")
    ax_up.xaxis.get_major_locator().set_params(integer=True)

    for axis in (ax_mu, ax_gap, ax_up):
        axis.grid(alpha=0.25, lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    table.to_csv(output.with_suffix(".csv"), index=False)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str))
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="aemwater run directory")
    parser.add_argument("-o", "--output", type=Path,
                        help="output image (default: RUN_DIR/uptake_analysis.png); "
                             ".csv and .json are written alongside")
    parser.add_argument("--water-mu", type=float,
                        help="bulk-water excess chemical potential, kcal/mol")
    parser.add_argument("--bulk-rho", type=float,
                        help="bulk water number density (N+1)/V, molecules/A^3")
    parser.add_argument("--temperature", type=float, help="K (default: run config)")
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output = args.output or run_dir / "uptake_analysis.png"
    result = plot_run(run_dir, output.expanduser().resolve(), args.water_mu,
                      args.bulk_rho, args.temperature)
    print(result)


if __name__ == "__main__":
    main()
