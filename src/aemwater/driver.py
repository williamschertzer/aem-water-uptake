"""Iterative water loading to saturation.

The loop
--------
Starting from an equilibrated dry membrane, each iteration:

  1. maps the cavities in the current configuration and inserts a batch of water
  2. relaxes at constant pressure so the box swells to accommodate it
  3. measures mu_ex of water in the swollen membrane by Widom insertion
  4. compares it with the bulk reference

and stops when water is no longer thermodynamically driven into the membrane.

Why a batch loop rather than one water at a time
------------------------------------------------
Inserting one water and re-equilibrating would be the cleanest protocol and is
computationally hopeless: a membrane at lambda = 15 holds hundreds of waters, and
each insertion needs tens of picoseconds to relax. Batches of a few percent of
the current content keep the perturbation small enough that the relaxation is
short, while reaching saturation in tens of iterations rather than hundreds.

The batch size shrinks as saturation approaches: overshooting past the endpoint
wastes the most expensive iterations and blurs the answer, so once the chemical
potential gap closes the loop takes smaller steps.

Why a difference and not an absolute
------------------------------------
Direct Widom insertion into dense liquid water is badly under-converged at any
affordable sample count: measured here, 75k insertions into SPC/E at 298 K gave
-2.9 kcal/mol against a literature -6.3, a factor of 322 short in the Boltzmann
average, with individual blocks spanning four orders of magnitude. The estimator
is bounded above by the rare trials that land in a cavity, so it always
underestimates the magnitude, and it approaches the true value from the wrong
side.

The saturation test is a *difference* of two such estimates, both computed with
the same insertion count, the same water model and the same cutoffs. The bias
comes from the unsampled tail of the cavity-size distribution, so it cancels to
the extent that the two systems have similar cavity statistics -- and a membrane
approaching saturation is, by construction, converging on bulk-like water
domains. The cancellation is therefore best exactly where the criterion is
evaluated, and worst in the dry membrane where the answer is not in doubt.

This is why the reference is computed by this code rather than taken from the
literature. A published mu_ex is the converged value; subtracting it from an
under-converged membrane estimate would compare two different quantities and
report saturation several waters early. An equally under-converged reference is
the correct thing to subtract.

The consequence to keep in view: `BulkReference.mu_ex` is not a validated
measurement of the excess chemical potential of water and should not be
reported as one. It is one half of a matched pair. `sanity()` will say so.

The practical consequence is that the reference must be run at the same settings
as the membrane measurement. `BulkReference` records them and the driver refuses
a reference whose settings do not match.

Three ways the loop can stop
----------------------------
* thermodynamic  -- mu_ex(membrane) >= mu_ex(bulk) - tolerance. The real answer.
* geometric      -- no cavity can accept another water. Usually means the batch
                    size is too large for the remaining free volume, but in a
                    tightly crosslinked membrane it can be the physical answer.
* budget         -- max_iterations reached. Reported as *not converged*, because
                    a number produced by running out of iterations is not an
                    uptake measurement.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .utils import LOG
from .widom import SaturationTest, WidomEstimate, read_widom_file


def bulk_n_waters(widom_spec) -> int:
    """Waters filling the requested bulk box at liquid density.

    The user specifies a box length rather than a molecule count because the
    finite-size error in mu_ex is controlled by the box edge relative to the
    cutoff, not by N.
    """
    volume_cm3 = (widom_spec.bulk_box_length * 1e-8) ** 3
    return max(64, int(round(0.997 * 6.02214076e23 * volume_cm3 / M_WATER)))


def settle_steps(md_spec) -> int:
    """Constant-volume settling before the barostat is switched on.

    Freshly inserted waters can sit closer to the matrix than equilibrium. Under
    a barostat that local repulsion is read as pressure and the box jumps; a
    short NVT window lets it dissipate first.
    """
    return max(2000, md_spec.relax_npt_steps // 10)

#: Molar mass of water, g/mol.
M_WATER = 18.01528


class DriverError(RuntimeError):
    """Raised when the uptake loop cannot proceed."""


@dataclass
class Iteration:
    """One insert-relax-measure cycle."""

    index: int
    n_waters_before: int
    n_requested: int
    n_inserted: int
    n_waters_after: int
    density: float
    volume: float
    lambda_value: float          # waters per ionic group
    water_uptake_pct: float      # 100 * m_water / m_dry
    mu_ex: float | None = None
    mu_ex_stderr: float | None = None
    mu_gap: float | None = None  # membrane - bulk
    saturated: bool = False
    geometrically_saturated: bool = False
    free_volume_fraction: float = 0.0
    wall_seconds: float = 0.0

    def to_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class UptakeResult:
    """The endpoint of the loop, with everything needed to judge it."""

    iterations: list[Iteration]
    n_waters: int
    lambda_value: float
    water_uptake_pct: float
    hydrated_density: float
    dry_density: float
    stop_reason: str
    converged: bool
    bulk_mu_ex: float
    workdir: Path
    composition: dict[str, object] = field(default_factory=dict)

    def summary(self) -> dict[str, object]:
        return {
            "n_waters": self.n_waters,
            "lambda_waters_per_ionic_group": round(self.lambda_value, 3),
            "water_uptake_wt_pct": round(self.water_uptake_pct, 2),
            "hydrated_density_g_cm3": round(self.hydrated_density, 4),
            "dry_density_g_cm3": round(self.dry_density, 4),
            "bulk_mu_ex_kcal_mol": round(self.bulk_mu_ex, 3),
            "stop_reason": self.stop_reason,
            "converged": self.converged,
            "iterations": len(self.iterations),
        }

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame([it.to_row() for it in self.iterations])


def next_batch_size(
    n_current: int,
    n_ionic_groups: int,
    mu_gap: float | None,
    stderr: float,
    initial_fraction: float = 0.25,
    min_batch: int = 1,
    max_batch: int = 200,
) -> int:
    """How many waters to add next.

    Scales with the current content (a fixed batch is a huge perturbation when
    the membrane is nearly dry and a negligible one when it is swollen), and
    shrinks as the chemical-potential gap closes so the endpoint is approached
    rather than overshot.

    ``mu_gap`` is mu_ex(membrane) - mu_ex(bulk); negative means water is still
    driven in. The first iteration has no measurement yet and gets a batch sized
    from the ionic-group count, since roughly lambda ~ 1 is a safe first step.
    """
    if mu_gap is None:
        return max(min_batch, min(max_batch, int(round(n_ionic_groups))))

    base = max(min_batch, int(round(initial_fraction * max(n_current, n_ionic_groups))))
    # Within a few sigma of the endpoint, take small steps: the cost of
    # overshooting is a wasted expensive iteration plus a blurred answer.
    scale = 1.0
    if stderr > 0:
        sigmas = abs(mu_gap) / stderr
        if sigmas < 2.0:
            scale = 0.25
        elif sigmas < 5.0:
            scale = 0.5
    return int(max(min_batch, min(max_batch, round(base * scale))))


def update_failed_batches(previous: int, requested: int, inserted: int) -> int:
    """Count consecutive geometric shortfalls, resetting after a full batch."""
    return previous + 1 if inserted < requested else 0


def hydration_number(n_waters: int, n_ionic_groups: int) -> float:
    """lambda: waters per ionic group, the standard AEM hydration measure."""
    if n_ionic_groups <= 0:
        raise DriverError("composition has no ionic groups; lambda is undefined")
    return n_waters / n_ionic_groups


def water_uptake_percent(n_waters: int, dry_mass_g_mol: float) -> float:
    """Mass water uptake, 100 * m_water / m_dry.

    The other standard reporting convention. Both are returned because the
    literature is split between them and converting requires the IEC, which a
    reader of a single number may not have.
    """
    if dry_mass_g_mol <= 0:
        raise DriverError("dry mass must be positive")
    return 100.0 * n_waters * M_WATER / dry_mass_g_mol


def _read_final_state(data_file: Path) -> tuple[np.ndarray, list[str], float]:
    """Coordinates, elements and box edge from a LAMMPS data file.

    Read back from the file the previous stage wrote rather than kept in memory:
    the configuration that matters is the one LAMMPS produced, and reading it
    back means a crashed or restarted workflow resumes from the same state.
    """
    import re

    text = data_file.read_text()
    edge_match = re.search(r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+xlo xhi", text)
    if not edge_match:
        raise DriverError(f"{data_file} has no box definition")
    lo, hi = float(edge_match.group(1)), float(edge_match.group(2))
    edge = hi - lo

    def section(name: str) -> list[str]:
        if name not in text:
            raise DriverError(f"{data_file} has no {name} section")
        body = text.split(name, 1)[1]
        rows = []
        for line in body.splitlines()[1:]:
            s = line.strip()
            if not s:
                if rows:
                    break
                continue
            if not (s[0].isdigit() or s[0] == "-"):
                break
            rows.append(s)
        return rows

    masses = {int(l.split()[0]): float(l.split()[1]) for l in section("\nMasses\n")}
    atom_rows = [l.split() for l in section("Atoms # full")]
    # Sort by atom ID (column 1). LAMMPS writes the Atoms section in its own
    # internal order -- spatially sorted for cache efficiency, NOT by ID -- so
    # reading the lines sequentially assigns each coordinate to the wrong atom.
    # The consequence is silent: the file has the right atom count and a
    # plausible density, but 509 of 1056 atoms had a different element from the
    # molecule they were assigned to, and reassembly produced 894 bonds over
    # 2.5 A (longest 69.9 A in a 22.06 A cell) before LAMMPS aborted in SHAKE.
    atom_rows.sort(key=lambda r: int(r[0]))
    ids = [int(r[0]) for r in atom_rows]
    if ids != list(range(1, len(ids) + 1)):
        raise DriverError(
            f"{data_file} has atom IDs that are not a contiguous 1..N range "
            f"(got {len(ids)} atoms, IDs {ids[0]}..{ids[-1]}). The molecule "
            "inventory is ordered 1..N, so the mapping would be ambiguous."
        )
    coords = np.array([[float(r[4]), float(r[5]), float(r[6])] for r in atom_rows])
    # Unwrap with the image flags LAMMPS writes in columns 7-9. LAMMPS stores
    # coordinates wrapped into the primary cell, so a molecule straddling a
    # boundary has its atoms on opposite faces. Reassembling from wrapped
    # coordinates gives that molecule bonds the length of the box: the first real
    # iteration produced 894 bonds over 2.5 A, the longest 27 A in a 22 A cell,
    # and LAMMPS died in SHAKE ("Shake determinant < 0.0") after warning about
    # bond extent and inconsistent image flags. Only the difference within a
    # molecule matters here, so unwrapping into an unbounded frame is enough --
    # the NPT stage rewraps.
    ylo, yhi = (float(v) for v in re.search(
        r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+ylo yhi", text).groups())
    zlo, zhi = (float(v) for v in re.search(
        r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+zlo zhi", text).groups())
    lengths = np.array([edge, yhi - ylo, zhi - zlo])
    if len(atom_rows[0]) >= 10:
        images = np.array([[int(r[7]), int(r[8]), int(r[9])] for r in atom_rows],
                          dtype=float)
        coords = coords + images * lengths
    else:
        # No image flags: detect the damage rather than silently proceeding,
        # since the failure downstream is an opaque SHAKE abort.
        raise DriverError(
            f"{data_file} has no image flags in its Atoms section. Coordinates "
            "cannot be unwrapped, and molecules crossing a periodic boundary "
            "would be reassembled stretched across the cell."
        )
    # Element from mass: the data file carries no element symbols, and typing is
    # unambiguous at this tolerance for the elements a GAFF2 system contains.
    table = [("H", 1.008), ("C", 12.011), ("N", 14.007), ("O", 15.999),
             ("F", 18.998), ("Na", 22.990), ("S", 32.06), ("Cl", 35.453),
             ("K", 39.098), ("Br", 79.904), ("I", 126.904)]
    elements = []
    for r in atom_rows:
        m = masses[int(r[2])]
        elements.append(min(table, key=lambda kv: abs(kv[1] - m))[0])
    return coords, elements, edge


def _resume_data_file(
    workdir: Path,
    dry_data: Path,
    iterations: list[Iteration],
) -> Path:
    """Return the structure that a resumed uptake loop must continue from.

    A checkpoint records an iteration only after LAMMPS has written that
    iteration's ``relaxed.data`` and the driver has reduced its outputs. The
    next batch must therefore start from the last checkpointed relaxed
    structure, not from the dry membrane.
    """
    if not iterations:
        return dry_data

    last = iterations[-1]
    relaxed = workdir / f"iter_{last.index:03d}" / "relaxed.data"
    if not relaxed.exists():
        raise DriverError(
            f"uptake checkpoint ends at iteration {last.index}, but its relaxed "
            f"structure is missing: {relaxed}. Restore that file or restart the "
            "uptake loop from the dry membrane with --force."
        )
    return relaxed


def _write_state(path: Path, payload: dict) -> None:
    """Checkpoint the loop so an interrupted run resumes instead of restarting."""
    path.write_text(json.dumps(payload, indent=2, default=str))


def _run_iteration(
    config, stage: Path, coords, elements, edge, insertion_result, comp,
    n_waters: int, model, bulk_reference, step: int, typed_chains: list,
) -> dict:
    """Relax one water batch at constant pressure and measure mu_ex.

    Rebuilds the LAMMPS system from the previous configuration plus the new
    waters. Rebuilding rather than editing the data file in place keeps a single
    code path for "turn molecules into LAMMPS input", so the type numbering and
    coefficient blocks cannot drift between the first iteration and the tenth.
    """
    from .assembly import CellContents, assemble, ion_molecules, water_molecules
    from .lammps.inputs import (
        ConstraintSpec, GroupSpec, comm_cutoff, constraint_spec, minimise_spec,
        pair_coeff_lines,
        render_input, soft_push_spec, write_water_molecule_template,
    )
    from .lammps.runner import run_lammps
    from .lammps.writer import write_data_file
    from .widom import SaturationTest, read_widom_file

    contents = CellContents(
        chains=typed_chains,
        ions=ion_molecules(comp.n_counterions, comp.counterion),
        waters=water_molecules(n_waters, config.water_model),
    )
    all_coords = np.vstack([coords, insertion_result.coordinates])
    if all_coords.shape[0] != contents.n_atoms:
        raise DriverError(
            f"iteration {step}: {all_coords.shape[0]} coordinates for "
            f"{contents.n_atoms} atoms. The molecule inventory and the "
            "configuration have diverged."
        )
    # Cross-check the read-back element order against the molecule inventory
    # before assembling. The coordinate-count check above passes even when the
    # ordering is scrambled, and a scrambled assignment is silent: plausible
    # density, right atom count, wrong molecule for every coordinate. This is the
    # cheap check that turns it into an error.
    if elements is not None:
        expected = [a.element_name for m in contents.molecules for a in m.atoms
                    if True][:len(elements)]
        wrong = sum(1 for a, b in zip(elements, expected) if a != b)
        if wrong:
            raise DriverError(
                f"iteration {step}: {wrong} of {len(elements)} atoms read back "
                "from LAMMPS have a different element from the molecule they "
                "would be assigned. The atom ordering has diverged from the "
                "inventory; coordinates would be attached to the wrong atoms."
            )
    system = assemble(contents, all_coords, edge=edge)
    write_data_file(system, stage / "start.data")

    o_type, h_type = system.water_atom_types()
    write_water_molecule_template(
        stage / "h2o.mol", model, o_type, h_type,
        system.water_bond_type(), system.water_angle_type())
    n_poly, n_ion = len(contents.chains), len(contents.ions)
    md = config.md
    n_averages = max(1, config.md.relax_npt_steps // (md.thermo_every * 10))
    n_widom_samples = max(1, config.widom.steps_per_block // config.widom.every)

    render_input(
        "insert.in.j2", stage / "in.insert",
        md=md, widom=config.widom, title=f"iteration {step}",
        data_file="start.data", pair_coeff_lines=pair_coeff_lines(system),
        extra_types=None, comm_cutoff=comm_cutoff(md),
        minim=minimise_spec(md), soft=soft_push_spec(md),
        constraints=constraint_spec(
            md, system.water_bond_type(), system.water_angle_type(),
            has_widom=config.widom.enabled),
        groups=GroupSpec(n_polymer_molecules=n_poly, n_ion_molecules=n_ion,
                         water_type_o=o_type, water_type_h=h_type),
        out_data="relaxed.data", out_restart="relaxed.restart",
        dump_file="iter.lammpstrj", density_file="density.dat",
        mu_file="mu.dat", water_template="h2o.mol", seed=md.seed + step,
        n_averages=n_averages, n_widom_samples=n_widom_samples,
        widom_window=config.widom.every * n_widom_samples,
        settle_steps=settle_steps(config.md),
        velocity_create=(step == 0),
        npt_equil_steps=config.md.relax_npt_steps // 2,
        npt_prod_steps=config.md.relax_npt_steps - config.md.relax_npt_steps // 2,
        widom_steps=config.widom.steps_per_block * config.widom.n_blocks,
    )
    run_lammps(stage / "in.insert", ranks=md.mpi_ranks, log_name="iter.log")

    rows = [
        [float(x) for x in line.split()]
        for line in (stage / "density.dat").read_text().splitlines()
        if line.strip() and not line.startswith("#") and len(line.split()) >= 3
    ]
    if not rows:
        raise DriverError(f"iteration {step} wrote no density data")
    arr = np.array(rows)
    half = arr[len(arr) // 2:]
    density, volume = float(half[:, 1].mean()), float(half[:, 2].mean())

    est = read_widom_file(stage / "mu.dat", md.temperature,
                          n_blocks=config.widom.n_blocks)
    test = SaturationTest(est, bulk_reference.mu_ex,
                          tolerance_sigma=config.widom.sigma_tolerance)
    new_coords, new_elements, new_edge = _read_final_state(stage / "relaxed.data")
    return {
        "coords": new_coords, "elements": new_elements, "edge": new_edge,
        "density": density, "volume": volume, "mu_ex": est.mu_ex,
        "stderr": est.stderr if np.isfinite(est.stderr) else 0.0,
        "mu_gap": test.difference, "saturated": test.saturated,
        "trustworthy": test.trustworthy,
    }


#: Settings that must agree between the reference and the membrane measurement.
#: These are the ones the bias depends on: a different water model, temperature
#: or cutoff changes the cavity distribution, and a different insertion count
#: changes how far up the tail each estimate has climbed. n_waters and seed are
#: excluded -- box size and random stream affect the variance, not the bias.
_REFERENCE_CRITICAL = (
    "water_model", "temperature", "cutoff", "kspace_accuracy",
    "insertions_per_call",
)


def _check_reference_matches(reference, wanted) -> None:
    """Refuse a bulk reference computed at different settings.

    The saturation criterion is a difference of two Widom estimates and is only
    meaningful because their biases are comparable (see the module docstring).
    Comparing against a reference run at a different water model or cutoff
    silently produces a number that looks like an uptake and is not one.
    """
    mismatched = [
        f"{name}: reference {getattr(reference.settings, name)!r} "
        f"!= run {getattr(wanted, name)!r}"
        for name in _REFERENCE_CRITICAL
        if getattr(reference.settings, name) != getattr(wanted, name)
    ]
    if mismatched:
        raise ValueError(
            "bulk reference was computed at different settings, so the "
            "chemical-potential difference is not meaningful:\n  "
            + "\n  ".join(mismatched)
        )


def run_uptake(
    config,
    workdir: Path | str,
    typed_chains: list,
    bulk_reference=None,
    resume: bool = True,
) -> UptakeResult:
    """Load water into an equilibrated dry membrane until it saturates.

    ``typed_chains`` are the GAFF2-typed polymer structures from the dry-stage
    preparation. They are passed in rather than re-typed here because semi-
    empirical charge derivation is the single most expensive step in the
    workflow and its result is identical at every iteration.

    ``config`` is a :class:`~aemwater.config.RunConfig`. The dry membrane is
    built and equilibrated first (or reused from ``workdir`` if already present),
    then water is added in batches until one of the three stop conditions fires.
    """
    import time

    from .assembly import CellContents, assemble, ion_molecules, water_molecules
    from .bulk import BulkSettings, run_bulk_reference
    from .chemistry import composition_from_config
    from .insertion import insert_waters
    from .lammps.inputs import (
        ConstraintSpec,
        GroupSpec,
        pair_coeff_lines,
        render_input,
        write_water_molecule_template,
    )
    from .lammps.runner import run_lammps
    from .lammps.writer import write_data_file
    from .forcefield.water import water_model as get_water_model

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    state_file = workdir / "uptake_state.json"

    comp = composition_from_config(config)
    n_ionic = comp.total_ionic_groups
    dry_mass = comp.dry_molar_mass

    # --- the reservoir -----------------------------------------------------
    bulk_settings = BulkSettings(
            water_model=config.water_model,
            temperature=config.md.temperature,
            pressure=config.md.pressure,
            n_waters=bulk_n_waters(config.widom),
            cutoff=config.md.cutoff,
            kspace_accuracy=config.md.kspace_accuracy,
            equil_steps=config.widom.bulk_equil_steps,
            widom_steps=config.widom.n_blocks * config.widom.steps_per_block,
            insertions_per_call=config.widom.insertions_per_call,
        seed=config.widom.seed,
    )
    if bulk_reference is None:
        LOG.info("no bulk reference supplied; computing one")
        bulk_reference = run_bulk_reference(
            bulk_settings, workdir / "bulk", cache_dir=config.widom.cache_dir,
            ranks=config.md.mpi_ranks
        )
    _check_reference_matches(bulk_reference, bulk_settings)
    for issue in bulk_reference.sanity():
        LOG.warning("bulk reference: %s", issue)

    # --- the dry membrane --------------------------------------------------
    dry_data = workdir / "dry" / "dry.data"
    if not dry_data.exists():
        raise DriverError(
            f"no equilibrated dry membrane at {dry_data}. Run the dry stages "
            "first (aemwater prepare) or point --workdir at a directory that "
            "has them."
        )
    coords, elements, edge = _read_final_state(dry_data)
    dry_density = dry_mass / (6.02214076e23 * (edge * 1e-8) ** 3)

    model = get_water_model(config.water_model)
    typed_cache = workdir / "typing"
    iterations: list[Iteration] = []
    n_waters = 0
    mu_gap = None
    stderr = 0.0
    stop_reason = "max_iterations"
    converged = False
    failed_batches = 0
    start_step = 0

    if resume and state_file.exists():
        saved = json.loads(state_file.read_text())
        iterations = [Iteration(**row) for row in saved.get("iterations", [])]
        n_waters = saved.get("n_waters", 0)
        mu_gap = saved.get("mu_gap")
        stderr = saved.get("stderr", 0.0)
        if iterations and n_waters != iterations[-1].n_waters_after:
            raise DriverError(
                f"uptake checkpoint records {n_waters} waters globally but "
                f"iteration {iterations[-1].index} ended with "
                f"{iterations[-1].n_waters_after}; refusing an inconsistent resume"
            )
        resume_data = _resume_data_file(workdir, dry_data, iterations)
        coords, elements, edge = _read_final_state(resume_data)
        failed_batches = saved.get("failed_batches", 0)
        start_step = saved.get("next_step", len(iterations))
        LOG.info(
            "resuming at iteration %d with %d waters from %s "
            "(%d consecutive geometric shortfalls)",
            start_step, n_waters, resume_data, failed_batches,
        )

    for step in range(start_step, config.insertion.max_iterations):
        t0 = time.time()
        n_add = next_batch_size(
            n_waters, n_ionic, mu_gap, stderr,
            initial_fraction=config.insertion.batch_fraction,
            min_batch=config.insertion.min_batch_size,
            max_batch=config.insertion.batch_size,
        )
        stage = workdir / f"iter_{step:03d}"
        stage.mkdir(exist_ok=True)

        result = insert_waters(
            coords, elements, edge, n_add, model,
            probe_radius=config.insertion.probe_radius,
            vdw_scale=config.insertion.vdw_scale,
            water_water_min=config.insertion.water_water_min,
            seed=config.insertion.seed + step,
        )
        failed_batches = update_failed_batches(
            failed_batches, result.n_requested, result.n_inserted
        )

        if result.n_inserted == 0:
            _write_state(state_file, {
                "iterations": [i.to_row() for i in iterations],
                "n_waters": n_waters, "mu_gap": mu_gap, "stderr": stderr,
                "failed_batches": failed_batches, "next_step": step + 1,
            })
            if failed_batches >= config.insertion.max_failed_batches:
                stop_reason = "geometric_saturation"
                LOG.info(
                    "iteration %d: no cavity accepts another water; stopping after "
                    "%d consecutive geometric shortfalls",
                    step, failed_batches,
                )
                break
            LOG.warning(
                "iteration %d: no cavity accepts another water "
                "(%d/%d consecutive geometric shortfalls); retrying",
                step, failed_batches, config.insertion.max_failed_batches,
            )
            continue

        n_waters += result.n_inserted
        state = _run_iteration(
            config, stage, coords, elements, edge, result, comp, n_waters,
            model, bulk_reference, step, typed_chains,
        )
        coords, elements, edge = state["coords"], state["elements"], state["edge"]
        mu_gap, stderr = state["mu_gap"], state["stderr"]

        it = Iteration(
            index=step,
            n_waters_before=n_waters - result.n_inserted,
            n_requested=n_add,
            n_inserted=result.n_inserted,
            n_waters_after=n_waters,
            density=state["density"],
            volume=state["volume"],
            lambda_value=hydration_number(n_waters, n_ionic),
            water_uptake_pct=water_uptake_percent(n_waters, dry_mass),
            mu_ex=state["mu_ex"],
            mu_ex_stderr=state["stderr"],
            mu_gap=mu_gap,
            saturated=state["saturated"],
            geometrically_saturated=result.saturated,
            free_volume_fraction=result.void_map.free_volume_fraction,
            wall_seconds=time.time() - t0,
        )
        iterations.append(it)
        _write_state(state_file, {
            "iterations": [i.to_row() for i in iterations],
            "n_waters": n_waters, "mu_gap": mu_gap, "stderr": stderr,
            "failed_batches": failed_batches, "next_step": step + 1,
        })
        LOG.info(
            "iteration %d: +%d -> %d waters, lambda = %.2f, uptake = %.1f%%, "
            "mu_ex = %.3f (gap %.3f kcal/mol)",
            step, result.n_inserted, n_waters, it.lambda_value,
            it.water_uptake_pct, state["mu_ex"], mu_gap,
        )

        if state["saturated"]:
            stop_reason = "thermodynamic_saturation"
            converged = True
            break
        if failed_batches >= config.insertion.max_failed_batches:
            stop_reason = "geometric_saturation"
            converged = True
            LOG.info(
                "iteration %d: stopping after %d consecutive geometric "
                "shortfalls (inserted %d/%d in the latest batch)",
                step, failed_batches, result.n_inserted, result.n_requested,
            )
            break
    else:
        LOG.warning(
            "reached max_iterations (%d) without saturating; the reported uptake "
            "is a lower bound", config.insertion.max_iterations
        )

    if stop_reason == "geometric_saturation":
        converged = True

    final = iterations[-1] if iterations else None
    return UptakeResult(
        iterations=iterations,
        n_waters=n_waters,
        lambda_value=hydration_number(n_waters, n_ionic) if n_waters else 0.0,
        water_uptake_pct=water_uptake_percent(n_waters, dry_mass) if n_waters else 0.0,
        hydrated_density=final.density if final else dry_density,
        dry_density=dry_density,
        stop_reason=stop_reason,
        converged=converged,
        bulk_mu_ex=bulk_reference.mu_ex.mu_ex,
        workdir=workdir,
        composition=comp.summary() if hasattr(comp, "summary") else {},
    )


__all__ = [
    "run_uptake",
    "Iteration",
    "UptakeResult",
    "DriverError",
    "next_batch_size",
    "hydration_number",
    "water_uptake_percent",
    "M_WATER",
]
