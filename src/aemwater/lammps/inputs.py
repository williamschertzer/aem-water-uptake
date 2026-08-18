"""Render LAMMPS input scripts from Jinja2 templates.

Every LAMMPS run in the workflow is generated from a template in
``templates/``, never assembled by string concatenation, so the exact input for
any stage of any run can be read back from the run directory and re-run by hand.
That matters here: the uptake number is the endpoint of dozens of chained runs,
and "what did stage 7 actually do" has to be answerable.

The templates receive the config dataclasses directly rather than a flattened
dict, so a template reads ``md.temperature`` and fails loudly on a typo instead
of silently substituting an empty string.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..utils import LOG
from .writer import LammpsWriteError

TEMPLATE_DIR = Path(__file__).parent / "templates"

#: Water molecule template for `fix widom` / `fix gcmc`. Written next to the run
#: so the insertion fix and the data file cannot disagree about atom types.
WATER_MOLECULE_TEMPLATE = """\
# Rigid water for Widom test insertions.
#
# The Bonds and Angles sections are load-bearing even though the molecule is
# rigid and never integrated. LAMMPS applies `special_bonds` exclusions through
# the bond topology, so a template without bonds leaves the molecule's own O-H
# and H-H Coulomb interactions in the insertion energy. For SPC/E that is about
# -202 kcal/mol of pure self-interaction added to every trial, which swamps the
# real solvation free energy completely.
3 atoms
2 bonds
1 angles

Coords

1 {ox:.6f} {oy:.6f} {oz:.6f}
2 {h1x:.6f} {h1y:.6f} {h1z:.6f}
3 {h2x:.6f} {h2y:.6f} {h2z:.6f}

Types

1 {type_o}
2 {type_h}
3 {type_h}

Charges

1 {q_o:.6f}
2 {q_h:.6f}
3 {q_h:.6f}

Masses

1 {mass_o:.6f}
2 {mass_h:.6f}
3 {mass_h:.6f}

Bonds

1 {bond_type} 1 2
2 {bond_type} 1 3

Angles

1 {angle_type} 2 1 3
"""


@dataclass(frozen=True)
class SoftPushSpec:
    """Soft-core push-off parameters.

    Derived rather than configured: these are numerical safety settings, not
    physics. A packed cell can contain contacts closer than any real minimum, and
    a 12-6 potential there produces forces large enough to eject atoms before the
    minimiser can act. The prefactor is ramped from zero so the overlap is pushed
    apart gradually.
    """

    cutoff: float = 3.0
    max_prefactor: float = 60.0
    max_displacement: float = 0.05
    timestep: float = 1.0
    steps: int = 5000


@dataclass(frozen=True)
class MinimiseSpec:
    """Energy-minimisation tolerances, taken from the MD config."""

    etol: float
    ftol: float
    max_iter: int
    max_eval: int


@dataclass(frozen=True)
class StageSpec:
    """Step counts for the dry-membrane equilibration sequence."""

    nvt_relax_steps: int
    heat_steps: int
    anneal_steps: int
    anneal_temperature: float
    squeeze_steps: int
    cool_steps: int
    npt_equil_steps: int
    npt_prod_steps: int


@dataclass(frozen=True)
class ConstraintSpec:
    """SHAKE settings.

    Water is constrained by bond and angle *type*, which requires knowing the
    type indices assigned by the data writer; the driver supplies them. Polymer
    X-H bonds are constrained by hydrogen mass instead, which catches every
    X-H bond without enumerating types.

    X-H constraints are only worth having above a 1 fs timestep -- their purpose
    is to remove the fastest vibration so the timestep can be raised -- and they
    are actively harmful during Widom sampling. Measured on the real membrane
    (1077 atoms, 22 A cell, identical inputs otherwise):

        b 8 a 14              + fix widom ->  0 SHAKE warnings
        b 8 a 14 m 1.008      + fix widom -> 98 SHAKE warnings
        b 5 6 7 8 a 14        + fix widom -> 98 SHAKE warnings

    The third line rules out the selection syntax: enumerating the X-H bond
    types explicitly reproduces it, so the trigger is constraining polymer X-H
    at all while test particles are being inserted and removed, not the mass
    keyword. `fix widom` adds and deletes atoms on every trial, and SHAKE builds
    its cluster list once at setup, so a constrained cluster can end up resolved
    against the wrong atoms -- "Shake determinant < 0.0" is that arithmetic
    failing. Both variants stayed at 298 K over 10 ps, so this is not a stability
    problem; it is a correctness one, and it is silent.

    Hence `for_timestep`: X-H constraints are requested only when the timestep
    needs them.
    """

    shake_water: bool
    shake_hydrogen: bool
    water_bond_type: int | None = None
    water_angle_type: int | None = None
    hydrogen_mass: float = 1.008
    shake_tol: float = 1.0e-4
    shake_iter: int = 20

    @property
    def shake_command(self) -> str:
        """The single ``fix shake`` line, or ``""`` when nothing is constrained.

        LAMMPS allows only ONE fix shake instance per run ("More than one fix
        shake instance", src/RIGID/fix_shake.cpp), so water and polymer X-H
        constraints must share one command rather than being emitted as two
        fixes. All constraint keywords go in that one command; the group is
        ``all`` because a bond type only exists in the molecule that defines it,
        so scoping by type is already selective.
        """
        tokens: list[str] = []
        if self.shake_water:
            if self.water_bond_type is None or self.water_angle_type is None:
                raise LammpsWriteError(
                    "shake_water needs water_bond_type and water_angle_type; "
                    "the data writer assigns them and the driver must pass them"
                )
            tokens += [f"b {self.water_bond_type}", f"a {self.water_angle_type}"]
        if self.shake_hydrogen:
            # By mass, so every X-H bond is caught without enumerating types.
            tokens.append(f"m {self.hydrogen_mass}")
        if not tokens:
            return ""
        return (
            f"fix             shake_all all shake {self.shake_tol} "
            f"{self.shake_iter} 0 " + " ".join(tokens)
        )


def constraint_spec(
    md, water_bond_type: int | None, water_angle_type: int | None,
    *, has_widom: bool = False,
) -> ConstraintSpec:
    """The constraints appropriate to this timestep, with rigid water always on.

    X-H constraints are requested only above 1 fs, where they buy the timestep
    they cost. At or below 1 fs the C-H stretch is resolved without them, and
    keeping them on during Widom sampling produces SHAKE clusters resolved
    against the wrong atoms (see ConstraintSpec). Water is always rigid: SPC/E
    and TIP3P are *defined* with fixed geometry, so relaxing it changes the model
    rather than the integration.
    """
    shake_hydrogen = float(md.timestep) > 1.0
    if shake_hydrogen and has_widom:
        LOG.warning(
            "timestep %.2f fs needs X-H constraints, but this stage runs Widom "
            "insertion; expect 'Shake determinant < 0.0' warnings. Use a 1 fs "
            "timestep for stages that measure the chemical potential.",
            float(md.timestep),
        )
    return ConstraintSpec(
        shake_water=True,
        shake_hydrogen=shake_hydrogen,
        water_bond_type=water_bond_type,
        water_angle_type=water_angle_type,
    )


@dataclass(frozen=True)
class GroupSpec:
    """Molecule-ID and atom-type ranges defining the LAMMPS groups."""

    n_polymer_molecules: int
    n_ion_molecules: int
    water_type_o: int | None = None
    water_type_h: int | None = None


def soft_push_spec(md) -> SoftPushSpec:
    return SoftPushSpec(steps=int(md.soft_push_steps))


def minimise_spec(md, max_iter: int | None = None) -> MinimiseSpec:
    iters = int(max_iter if max_iter is not None else md.min_maxiter)
    return MinimiseSpec(
        etol=float(md.min_etol),
        ftol=float(md.min_ftol),
        max_iter=iters,
        max_eval=iters * 10,
    )


def stage_spec(md) -> StageSpec:
    """Split the configured step budgets across the equilibration sequence.

    The heat ramp is given a fifth of the anneal budget: ramping is cheap, and
    what matters is the time spent *above* Tg where chains can interpenetrate.
    """
    anneal = int(md.anneal_steps)
    dry = int(md.dry_npt_steps)
    return StageSpec(
        nvt_relax_steps=max(1000, anneal // 10),
        heat_steps=max(1000, anneal // 5),
        anneal_steps=anneal,
        anneal_temperature=float(md.anneal_temperature),
        # The squeeze carries the volume collapse and the cool-down only has to
        # freeze it in, so the budget is split in favour of the squeeze.
        squeeze_steps=max(1000, int(md.compression_steps)),
        cool_steps=max(1000, int(md.compression_steps) // 2),
        npt_equil_steps=dry // 2,
        npt_prod_steps=dry - dry // 2,
    )


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        undefined=StrictUndefined,  # a missing variable is a bug, not a blank
        trim_blocks=True,
        lstrip_blocks=False,
        keep_trailing_newline=True,
    )


def render_input(template: str, path: Path, **context: Any) -> Path:
    """Render ``template`` with ``context`` and write it to ``path``."""
    env = _environment()
    tmpl = env.get_template(template)
    text = tmpl.render(**context)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    LOG.debug("rendered %s -> %s (%d lines)", template, path.name, text.count("\n"))
    return path


def write_water_molecule_template(
    path: Path,
    water_model,
    type_o: int,
    type_h: int,
    bond_type: int,
    angle_type: int,
) -> Path:
    """Write the LAMMPS molecule file used for water test insertions.

    ``bond_type`` and ``angle_type`` must be the same numeric types the data
    file uses for water, so that the exclusions LAMMPS applies to the inserted
    molecule match those applied to every resident water.
    """
    import math

    half = math.radians(water_model.angle_HOH) / 2.0
    r = water_model.r_OH
    text = WATER_MOLECULE_TEMPLATE.format(
        ox=0.0, oy=0.0, oz=0.0,
        h1x=r * math.sin(half), h1y=0.0, h1z=r * math.cos(half),
        h2x=-r * math.sin(half), h2y=0.0, h2z=r * math.cos(half),
        type_o=type_o, type_h=type_h,
        q_o=water_model.charge_O, q_h=water_model.charge_H,
        mass_o=water_model.mass_O, mass_h=water_model.mass_H,
        bond_type=bond_type, angle_type=angle_type,
    )
    path = Path(path)
    path.write_text(text)
    return path


def pair_coeff_lines(system) -> list[str]:
    """Explicit ``pair_coeff`` lines, needed after a ``pair_style`` switch.

    Changing ``pair_style`` discards all pair coefficients, so the soft-core
    push-off stage has to re-issue them rather than relying on the data file.
    """
    lines = []
    for tid, eps, sigma, name in system.pair_coeffs():
        lines.append(f"pair_coeff      {tid} {tid} {eps:.8f} {sigma:.8f}  # {name}")
    return lines


def comm_cutoff(md, longest_bond: float = 6.0) -> float:
    """Ghost-atom communication cutoff (A).

    Must exceed both the pair cutoff plus skin and the longest bonded
    interaction; LAMMPS warns and can lose atoms otherwise.
    """
    return float(max(md.cutoff + 2.0, longest_bond + 2.0))


def context_from_config(config, system=None, **extra: Any) -> dict[str, Any]:
    """Build a template context from a :class:`~aemwater.config.RunConfig`.

    When ``system`` is given, the fields every template needs from the written
    data file -- explicit pair coefficients, the SHAKE constraint types, the
    group definitions and the ghost-atom cutoff -- are derived from it rather
    than assembled by each caller. Type numbering is assigned by the writer, so
    reading it off the system is the only way these stay in step with the data
    file across iterations.
    """
    context: dict[str, Any] = {
        "md": config.md,
        "insertion": config.insertion,
        "widom": config.widom,
        "box": config.box,
        "polymer": config.polymer,
    }
    if system is not None:
        o_type, h_type = (system.water_atom_types() if system.has_water()
                          else (None, None))
        context.update(
            title=f"{config.polymer.name}: {config.polymer.n_chains} chains "
                  f"x {config.polymer.chain_length} units",
            pair_coeff_lines=pair_coeff_lines(system),
            extra_types=None,
            # The dry stages run no Widom insertion, so X-H constraints are safe
            # here; they still follow the timestep rule so every stage uses one
            # policy rather than two that can drift.
            constraints=ConstraintSpec(
                shake_water=system.has_water(),
                shake_hydrogen=float(config.md.timestep) > 1.0,
                water_bond_type=system.water_bond_type() if system.has_water() else None,
                water_angle_type=system.water_angle_type() if system.has_water() else None,
            ),
            groups=GroupSpec(
                n_polymer_molecules=system.n_polymer_molecules(),
                n_ion_molecules=system.n_ion_molecules(),
                water_type_o=o_type, water_type_h=h_type,
            ),
            comm_cutoff=comm_cutoff(config.md),
            seed=config.box.seed,
        )
    context.update(extra)
    return context


__all__ = [
    "render_input",
    "SoftPushSpec",
    "MinimiseSpec",
    "StageSpec",
    "ConstraintSpec",
    "constraint_spec",
    "GroupSpec",
    "soft_push_spec",
    "comm_cutoff",
    "minimise_spec",
    "stage_spec",
    "write_water_molecule_template",
    "pair_coeff_lines",
    "context_from_config",
    "TEMPLATE_DIR",
]
