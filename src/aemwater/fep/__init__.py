"""Alchemical free-energy calculation of the excess chemical potential.

A ghost SPC/E water is decoupled from the system in two legs -- soft-core
Lennard-Jones first, then electrostatics -- at fixed lambda, and the free energy
is estimated by MBAR, BAR and TI on the same samples. See docs/fep_design.md.

This file is not optional decoration: `[tool.setuptools.packages.find]` uses
`find_packages`, which skips directories without an `__init__.py`, so without it
`aemwater.fep` imports fine from the source tree and is silently absent from an
installed wheel.

The exports below are resolved lazily (PEP 562). They must be: `config` imports
the default ladders from `fep.schedule`, while `fep.inputs` imports `FEPSpec`
back from `config`. Eagerly importing the submodules here closes that loop and
`import aemwater.config` fails with a partially-initialised module. Lazy access
keeps the flat `from aemwater.fep import mbar_estimate` API without the cycle,
and makes `import aemwater.config` cheaper besides -- it no longer drags in
pymbar.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    from .estimators import (
        LegEstimate,
        LegResult,
        bar_estimate,
        combine_legs,
        dudl_from_finite_differences,
        mbar_estimate,
        read_fep_columns,
        ti_estimate,
        ti_from_state_dirs,
    )
    from .ghost import (
        GhostTopology,
        add_ghost_water,
        ghost_pair_coeff_lines,
        scale_ghost_charges,
    )
    from .inputs import Perturbation, perturbations_for, render_state_input
    from .rerun import EnergyMatrix, build_energy_matrix, write_rerun_input
    from .schedule import FEPLeg, LambdaLadder, LambdaState, default_ladders

#: export name -> submodule that defines it
_EXPORTS = {
    "FEPLeg": "schedule",
    "LambdaLadder": "schedule",
    "LambdaState": "schedule",
    "default_ladders": "schedule",
    "GhostTopology": "ghost",
    "add_ghost_water": "ghost",
    "ghost_pair_coeff_lines": "ghost",
    "scale_ghost_charges": "ghost",
    "Perturbation": "inputs",
    "perturbations_for": "inputs",
    "render_state_input": "inputs",
    "EnergyMatrix": "rerun",
    "build_energy_matrix": "rerun",
    "write_rerun_input": "rerun",
    "LegEstimate": "estimators",
    "LegResult": "estimators",
    "bar_estimate": "estimators",
    "combine_legs": "estimators",
    "dudl_from_finite_differences": "estimators",
    "mbar_estimate": "estimators",
    "read_fep_columns": "estimators",
    "ti_estimate": "estimators",
    "ti_from_state_dirs": "estimators",
}


def __getattr__(name: str):
    """Resolve an export on first access (PEP 562)."""
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    from importlib import import_module

    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value          # cache, so this runs once per name
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    # schedule
    "FEPLeg",
    "LambdaLadder",
    "LambdaState",
    "default_ladders",
    # ghost topology
    "GhostTopology",
    "add_ghost_water",
    "ghost_pair_coeff_lines",
    "scale_ghost_charges",
    # input generation
    "Perturbation",
    "perturbations_for",
    "render_state_input",
    # rerun matrix
    "EnergyMatrix",
    "build_energy_matrix",
    "write_rerun_input",
    # estimators
    "LegEstimate",
    "LegResult",
    "bar_estimate",
    "combine_legs",
    "dudl_from_finite_differences",
    "mbar_estimate",
    "read_fep_columns",
    "ti_estimate",
    "ti_from_state_dirs",
]
