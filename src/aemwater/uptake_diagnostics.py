"""Compact convergence plot for a completed or checkpointed uptake run."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


def _read_trajectory(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "uptake_trajectory.csv"
    if csv_path.is_file():
        frame = pd.read_csv(csv_path)
    else:
        state_path = run_dir / "uptake_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(
                f"no uptake_trajectory.csv or uptake_state.json in {run_dir}"
            )
        frame = pd.DataFrame(json.loads(state_path.read_text())["iterations"])
    required = {"index", "mu_ex", "water_uptake_pct", "n_waters_after"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"trajectory is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("uptake trajectory contains no iterations")
    return frame.sort_values("index")


def _bulk_water_mu(run_dir: Path, override: float | None) -> float:
    if override is not None:
        return override
    result_path = run_dir / "result.json"
    if not result_path.is_file():
        raise ValueError(
            "the bulk-water chemical potential is unavailable; pass --water-mu "
            "when analyzing an unfinished run"
        )
    result = json.loads(result_path.read_text())
    try:
        return float(result["bulk_mu_ex_kcal_mol"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{result_path} has no valid bulk_mu_ex_kcal_mol; pass --water-mu"
        ) from exc


def _dry_atom_counts(run_dir: Path) -> tuple[int, int]:
    data_path = run_dir / "dry" / "dry.data"
    if not data_path.is_file():
        data_path = run_dir / "dry" / "system.data"
    if not data_path.is_file():
        raise FileNotFoundError("no dry/dry.data or dry/system.data found")
    match = re.search(r"(?m)^\s*(\d+)\s+atoms\s*$", data_path.read_text())
    if match is None:
        raise ValueError(f"could not read the atom count from {data_path}")
    dry_atoms = int(match.group(1))
    composition_path = run_dir / "dry" / "composition.json"
    n_counterions = 0
    if composition_path.is_file():
        composition = json.loads(composition_path.read_text())
        n_counterions = int(composition.get("n_counterions", 0))
    polymer_atoms = dry_atoms - n_counterions
    if polymer_atoms < 0:
        raise ValueError("n_counterions exceeds the dry-system atom count")
    return polymer_atoms, n_counterions


def plot_run(run_dir: Path | str, output: Path | str,
             water_mu: float | None = None) -> Path:
    """Plot chemical potential and water uptake against iteration."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir, output = Path(run_dir), Path(output)
    frame = _read_trajectory(run_dir)
    reference_mu = _bulk_water_mu(run_dir, water_mu)
    polymer_atoms, n_counterions = _dry_atom_counts(run_dir)
    water_molecules = frame["n_waters_after"].astype(int)

    fig, (ax_mu, ax_uptake) = plt.subplots(1, 2, figsize=(11, 4.4))
    blue, red = "#1f4e79", "#b04a3a"
    valid = frame["mu_ex"].notna()
    yerr = frame.loc[valid, "mu_ex_stderr"] if "mu_ex_stderr" in frame else None
    ax_mu.errorbar(frame.loc[valid, "index"], frame.loc[valid, "mu_ex"],
                   yerr=yerr, fmt="o-", color=blue, lw=1.6, ms=4, capsize=3,
                   label="membrane water")
    ax_mu.axhline(reference_mu, color=red, ls="--", lw=1.5,
                  label=rf"bulk water ({reference_mu:g} kcal mol$^{{-1}}$)")
    ax_mu.set(xlabel="iteration",
              ylabel=r"excess chemical potential (kcal mol$^{-1}$)",
              title="Chemical-potential convergence")
    ax_mu.legend(frameon=False)

    ax_uptake.plot(frame["index"], frame["water_uptake_pct"], "o-",
                   color=blue, lw=1.6, ms=4)
    ax_uptake.set(xlabel="iteration", ylabel="water uptake (wt %)",
                  title="Water uptake")
    count_text = f"Polymer atoms: {polymer_atoms:,}"
    if n_counterions:
        count_text += f"\nCounterion atoms: {n_counterions:,}"
    count_text += f"\nWater molecules: {water_molecules.iloc[-1]:,} (final)"
    ax_uptake.text(0.03, 0.97, count_text, transform=ax_uptake.transAxes,
                   ha="left", va="top", fontsize=9,
                   bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                         "alpha": 0.9, "edgecolor": "0.75"})
    for iteration, uptake, count in zip(
            frame["index"], frame["water_uptake_pct"], water_molecules):
        ax_uptake.annotate(f"{count}", (iteration, uptake), xytext=(0, 7),
                           textcoords="offset points", ha="center", fontsize=7,
                           color="0.3")
    ax_uptake.text(0.98, 0.03, "labels = water molecules",
                   transform=ax_uptake.transAxes, ha="right", va="bottom",
                   fontsize=7, color="0.4")
    for axis in (ax_mu, ax_uptake):
        axis.grid(alpha=0.25, lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.xaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


__all__ = ["plot_run"]
