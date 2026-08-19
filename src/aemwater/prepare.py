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

    def summary(self) -> dict[str, object]:
        return {
            "edge_angstrom": round(self.edge, 3),
            "density_g_cm3": round(self.density, 4),
            "atoms": self.n_atoms,
            "data_file": str(self.data_file),
        }


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

    render_input("equilibrate.in.j2", workdir / "in.equilibrate",
                 data_file="min.data", out_data="dry.data",
                 out_restart="dry.restart", density_file="dry_density.dat",
                 dump_file="dry.lammpstrj", stages=stage_spec(md),
                 n_averages=max(1, md.dry_npt_steps // (md.thermo_every * 10)),
                 **common)
    run_lammps(workdir / "in.equilibrate", ranks=md.mpi_ranks, log_name="equil.log")

    from .driver import _read_final_state

    coords, elements, final_edge = _read_final_state(workdir / "dry.data")
    density = comp.dry_molar_mass / (6.02214076e23 * (final_edge * 1e-8) ** 3)
    LOG.info("dry membrane: %.4f g/cm3 in a %.2f A cell", density, final_edge)

    (workdir / "composition.json").write_text(
        json.dumps({"n_ionic_groups": comp.total_ionic_groups,
                    "dry_molar_mass": comp.dry_molar_mass,
                    "n_counterions": comp.n_counterions,
                    "density": density, "edge": final_edge}, indent=2)
    )
    return DryMembrane(workdir, workdir / "dry.data", final_edge, density,
                       len(coords), typed_chains, comp)


__all__ = ["DryMembrane", "prepare_dry_membrane"]
