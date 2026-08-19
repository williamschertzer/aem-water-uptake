"""Build and equilibrate the dry membrane.

Separated from the uptake loop because it is the expensive, one-off half of the
workflow: semi-empirical charge derivation and a 100k-step anneal cost more than
several insertion iterations, and the result is reusable across water-loading
runs with different protocols. Its output is a directory the loop reads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utils import LOG


class EquilibrationError(RuntimeError):
    """Raised when the dry membrane fails its convergence criteria."""


@dataclass(frozen=True)
class DryConvergence:
    """Whether the dry membrane is actually equilibrated.

    Two independent criteria, both necessary. Density alone is not enough: a
    structure can pass through the right density while still densifying, and a
    structure still densifying will keep densifying during the water-loading
    loop -- which is what produced the negative partial molar volume of water
    (cell contracting as water is added) in the diagnosed runs. Drift alone is
    not enough either, because a structure can sit stably at the wrong density.
    """

    density: float
    expected_density: float
    density_tolerance: float
    drift_per_100ps: float
    drift_tolerance: float
    #: Standard error of the fitted drift, same units. Scatter on a short
    #: window produces an apparent slope of order sigma/(sd(t)*sqrt(n)) with no
    #: real densification behind it, so a drift smaller than its own
    #: uncertainty is not evidence of anything.
    drift_stderr_per_100ps: float
    n_samples: int
    #: Mean density over each half of the production window, g/cm^3. A
    #: split-half disagreement is drift the linear fit can miss.
    first_half: float
    second_half: float

    @property
    def density_ok(self) -> bool:
        if not np.isfinite(self.density) or self.expected_density <= 0:
            return False
        rel = abs(self.density - self.expected_density) / self.expected_density
        return rel <= self.density_tolerance

    @property
    def drift_ok(self) -> bool:
        """Drift is a failure only if it is both large and statistically real.

        Requiring |slope| <= tolerance alone makes the criterion unpassable on a
        short or noisy window, where scatter fakes a slope far above tolerance.
        A drift within 2 standard errors of zero is consistent with no
        densification, so it does not fail; a resolved drift above tolerance --
        the monotone climb of a structure still collapsing -- does.
        """
        if not np.isfinite(self.drift_per_100ps):
            return False
        if abs(self.drift_per_100ps) <= self.drift_tolerance:
            return True
        if not np.isfinite(self.drift_stderr_per_100ps):
            return False
        return abs(self.drift_per_100ps) <= 2.0 * self.drift_stderr_per_100ps

    @property
    def converged(self) -> bool:
        # Too few samples to judge is not the same as converged.
        return bool(self.density_ok and self.drift_ok and self.n_samples >= 4)

    def to_dict(self) -> dict[str, object]:
        return {
            "converged": self.converged,
            "density_g_cm3": round(self.density, 4),
            "expected_density_g_cm3": round(self.expected_density, 4),
            "relative_deviation": (
                round(abs(self.density - self.expected_density)
                      / self.expected_density, 4)
                if self.expected_density > 0 else None),
            "density_tolerance": self.density_tolerance,
            "density_ok": self.density_ok,
            "drift_g_cm3_per_100ps": round(float(self.drift_per_100ps), 5),
            "drift_stderr_g_cm3_per_100ps": round(
                float(self.drift_stderr_per_100ps), 5),
            "drift_tolerance": self.drift_tolerance,
            "drift_ok": self.drift_ok,
            "first_half_mean": round(self.first_half, 4),
            "second_half_mean": round(self.second_half, 4),
            "n_samples": self.n_samples,
        }

    def report(self) -> str:
        tick = lambda ok: "OK  " if ok else "FAIL"
        rel = (abs(self.density - self.expected_density) / self.expected_density
               if self.expected_density > 0 else float("nan"))
        return "\n".join([
            f"  [{tick(self.density_ok)}] density {self.density:.4f} g/cm3 vs "
            f"expected {self.expected_density:.4f} "
            f"({100 * rel:+.1f}%, tolerance {100 * self.density_tolerance:.0f}%)",
            f"  [{tick(self.drift_ok)}] drift {self.drift_per_100ps:+.4f} "
            f"+/- {self.drift_stderr_per_100ps:.4f} g/cm3 per 100 ps "
            f"(tolerance {self.drift_tolerance:.4f}); "
            f"halves {self.first_half:.4f} -> {self.second_half:.4f}",
            f"  [{tick(self.n_samples >= 4)}] {self.n_samples} density samples "
            f"in the production window (need >= 4)",
        ])


def check_dry_convergence(
    density_file: Path | str, config, *, thermo_every: int,
    n_averages: int, timestep_fs: float,
) -> DryConvergence:
    """Read the production-window density trace and judge convergence.

    ``dry_density.dat`` is written by ``fix ave/time`` over the production step
    only, so every row is already inside the window the density is reported
    from -- no transient to discard here.
    """
    path = Path(density_file)
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    rows.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue
    equil = config.equilibration
    expected = (equil.expected_density if equil.expected_density is not None
                else config.box.target_density)
    if len(rows) < 2:
        return DryConvergence(
            density=float("nan"), expected_density=float(expected),
            density_tolerance=equil.density_tolerance,
            drift_per_100ps=float("nan"),
            drift_stderr_per_100ps=float("nan"),
            drift_tolerance=equil.drift_tolerance,
            n_samples=len(rows), first_half=float("nan"),
            second_half=float("nan"),
        )
    arr = np.array(rows)
    steps, dens = arr[:, 0], arr[:, 1]
    # fs -> ps, so the slope is reported in the units the tolerance is stated in.
    ps = steps * float(timestep_fs) / 1000.0
    # np.ptp(), not ps.ptp(): the ndarray method was removed in NumPy 2.
    if np.ptp(ps) > 0 and len(ps) >= 3:
        coeffs, cov = np.polyfit(ps, dens, 1, cov=True)
        slope_per_ps = float(coeffs[0])
        slope_stderr_per_ps = float(np.sqrt(cov[0, 0]))
    elif np.ptp(ps) > 0:
        slope_per_ps = float(np.polyfit(ps, dens, 1)[0])
        slope_stderr_per_ps = float("inf")  # two points constrain nothing
    else:
        slope_per_ps = 0.0
        slope_stderr_per_ps = 0.0
    half = len(dens) // 2
    return DryConvergence(
        density=float(dens[half:].mean()),
        expected_density=float(expected),
        density_tolerance=equil.density_tolerance,
        drift_per_100ps=slope_per_ps * 100.0,
        drift_stderr_per_100ps=slope_stderr_per_ps * 100.0,
        drift_tolerance=equil.drift_tolerance,
        n_samples=len(dens),
        first_half=float(dens[:half].mean()),
        second_half=float(dens[half:].mean()),
    )


@dataclass
class DryMembrane:
    """An equilibrated dry cell, ready for water."""

    workdir: Path
    data_file: Path
    edge: float
    density: float
    n_atoms: int
    typed_chains: list
    composition: object
    convergence: DryConvergence | None = None

    def summary(self) -> dict[str, object]:
        out: dict[str, object] = {
            "edge_angstrom": round(self.edge, 3),
            "density_g_cm3": round(self.density, 4),
            "atoms": self.n_atoms,
            "data_file": str(self.data_file),
        }
        if self.convergence is not None:
            out["convergence"] = self.convergence.to_dict()
        return out


def prepare_dry_membrane(config, workdir: Path | str) -> DryMembrane:
    """SMILES -> typed chains -> packed cell -> annealed dry membrane."""
    from .assembly import CellContents, assemble, ion_molecules
    from .chemistry import composition_from_config
    from .forcefield.gaff2 import GAFF2Backend
    from .lammps.inputs import (
        ConstraintSpec,
        GroupSpec,
        comm_cutoff,
        context_from_config,
        minimise_spec,
        pair_coeff_lines,
        render_input,
        equilibration_schedule,
        soft_push_spec,
        stage_spec,
    )
    from .lammps.runner import run_lammps
    from .lammps.writer import write_data_file
    import numpy as np

    from .packing import pack_cell
    from .polymer import build_chain

    workdir = Path(workdir) / "dry"
    workdir.mkdir(parents=True, exist_ok=True)

    comp = composition_from_config(config)
    LOG.info(
        "composition: %d chains x %d units, %d ionic groups, M_dry = %.1f g/mol",
        config.polymer.n_chains, config.polymer.chain_length,
        comp.total_ionic_groups, comp.dry_molar_mass,
    )

    # --- one typing run, reused for every chain -----------------------------
    # Every chain is the same molecule, so charges are derived once. This is the
    # single most expensive step in the workflow.
    chain = build_chain(
        config.polymer.smiles, config.polymer.chain_length,
        terminal_group=config.polymer.terminal_group, seed=config.box.seed,
    )
    LOG.info("typing the chain with GAFF2 (semi-empirical charges)")
    backend = GAFF2Backend(charge_method=config.polymer.charge_method)
    typed, fragment_typing = backend.type_chain(chain, workdir / "typing")
    # The same ParmEd structure for every chain: identical molecule, identical
    # charges. Coordinates come from the packer, not from the structure.
    typed_chains = [typed] * config.polymer.n_chains

    # --- pack at low density, compress with MD ------------------------------
    # The packer places rigid bodies, so it takes coordinates rather than typed
    # structures: chain conformations from the builder, ions as single points.
    # It sizes the cell itself from the target density and its dilation factor.
    ions = ion_molecules(comp.n_counterions, comp.counterion)
    chain_coords = [chain.coordinates()] * config.polymer.n_chains
    ion_coords = [np.asarray(ion.coordinates, dtype=float) for ion in ions]
    packed = pack_cell(
        chain_coords, ion_coords, comp,
        target_density=config.box.target_density,
        # BoxSpec states the packing density directly; the packer wants it as a
        # linear expansion of the target edge. One knob, two conventions:
        # dilation = (rho_target / rho_initial)^(1/3).
        dilation=(config.box.target_density / config.box.initial_density) ** (1 / 3),
        seed=config.box.seed,
        min_distance=config.box.min_separation,
    )
    contents = CellContents(chains=typed_chains, ions=ions, waters=[])
    edge = packed.edge
    system = assemble(contents, packed.coordinates, edge=edge)
    write_data_file(system, workdir / "system.data")

    md = config.md
    common = context_from_config(config, system)
    # The packed cell has close contacts by construction, so minimisation runs
    # a soft-core push first to separate overlaps before the real potential.
    render_input("minimise.in.j2", workdir / "in.minimise",
                 data_file="system.data", out_data="min.data",
                 out_restart="min.restart", minim=minimise_spec(md),
                 soft=soft_push_spec(md), **common)
    run_lammps(workdir / "in.minimise", ranks=md.mpi_ranks, log_name="min.log")

    # The 21-step schedule replaces the single squeeze-and-release cycle. See
    # EquilibrationSpec and the template header for why: one squeeze plateaus
    # ~18% below experiment with density still drifting, and the missing volume
    # is void space that the uptake loop fills.
    equil = config.equilibration
    schedule = (equilibration_schedule(md, equil)
                if equil.scheme == "21step" else None)
    if schedule is not None:
        total_ps = sum(s.ps for s in schedule)
        prod_steps = schedule[-1].steps(md.timestep)
        # Average over ~10 windows within the production step, so the drift
        # check below has enough points to fit a slope.
        n_averages = max(1, prod_steps // (md.thermo_every * 10))
        # A production window shorter than a few averaging intervals writes an
        # empty density file, and the gate then fails for want of samples
        # rather than for anything physical. Catch it before the MD runs, not
        # 20 stages later.
        expected_rows = prod_steps // (md.thermo_every * n_averages)
        if expected_rows < 4:
            raise EquilibrationError(
                f"equilibration.final_npt_ps = {equil.final_npt_ps} ps gives "
                f"{prod_steps} production steps, which yields only "
                f"{expected_rows} density sample(s) at md.thermo_every = "
                f"{md.thermo_every}. The convergence check needs at least 4. "
                "Raise equilibration.final_npt_ps or lower md.thermo_every."
            )
        LOG.info(
            "dry equilibration: 21-step scheme, %.0f ps total (%.0f ps production), "
            "peak %.0f atm, excursions to %.0f K",
            total_ps, schedule[-1].ps, equil.max_pressure, equil.high_temperature,
        )
    else:
        total_ps = None
        n_averages = max(1, md.dry_npt_steps // (md.thermo_every * 10))
        LOG.info("dry equilibration: legacy single-squeeze scheme")

    render_input("equilibrate.in.j2", workdir / "in.equilibrate",
                 data_file="min.data", out_data="dry.data",
                 out_restart="dry.restart", density_file="dry_density.dat",
                 dump_file="dry.lammpstrj", stages=stage_spec(md),
                 equil=equil, equil_schedule=schedule, equil_total_ps=total_ps,
                 n_averages=n_averages,
                 **common)
    run_lammps(workdir / "in.equilibrate", ranks=md.mpi_ranks, log_name="equil.log")

    from .driver import _read_final_state

    coords, elements, final_edge = _read_final_state(workdir / "dry.data")
    density = comp.dry_molar_mass / (6.02214076e23 * (final_edge * 1e-8) ** 3)
    LOG.info("dry membrane: %.4f g/cm3 in a %.2f A cell", density, final_edge)

    convergence = check_dry_convergence(
        workdir / "dry_density.dat", config,
        thermo_every=md.thermo_every, n_averages=n_averages,
        timestep_fs=md.timestep,
    )
    (workdir / "convergence.json").write_text(
        json.dumps(convergence.to_dict(), indent=2))
    for line in convergence.report().splitlines():
        (LOG.info if convergence.converged else LOG.warning)("%s", line)
    if not convergence.converged and equil.enforce_convergence:
        raise EquilibrationError(
            "the dry membrane did not converge, so any uptake computed from it "
            "would be biased high by void filling:\n" + convergence.report()
            + "\n\nRaise equilibration.time_scale or max_pressure, or set "
              "equilibration.enforce_convergence=false to proceed anyway "
              "(the result is a lower bound at best)."
        )

    (workdir / "composition.json").write_text(
        json.dumps({"n_ionic_groups": comp.total_ionic_groups,
                    "dry_molar_mass": comp.dry_molar_mass,
                    "n_counterions": comp.n_counterions,
                    "density": density, "edge": final_edge,
                    "converged": convergence.converged}, indent=2)
    )
    return DryMembrane(workdir, workdir / "dry.data", final_edge, density,
                       len(coords), typed_chains, comp, convergence)


__all__ = [
    "DryMembrane",
    "DryConvergence",
    "EquilibrationError",
    "check_dry_convergence",
    "prepare_dry_membrane",
]
