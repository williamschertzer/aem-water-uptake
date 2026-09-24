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
* insertion stalled -- repeated attempts add zero waters. Reported as not
                       converged; geometric blockage is not a thermodynamic endpoint.
* budget         -- max_iterations reached. Reported as *not converged*, because
                    a number produced by running out of iterations is not an
                    uptake measurement.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from .utils import LOG, read_json_or_none, write_json
from .saturation import estimate_saturation_point
from .widom import (KB_KCAL, SaturationTest, WidomEstimate, read_widom_file,
                    water_number_density)

#: Definition of ``mu_gap`` written into uptake_state.json. Checkpoints
#: without it hold the excess-only gap mu_ex,m - mu_ex,b and old stop flags
#: from the ``>= -k sigma`` rule; they are migrated on resume (see
#: :func:`_migrate_gap_definition`) rather than silently mixed.
GAP_DEFINITION = "total_mu_v1"


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


def _saturation_complete(iterations, extra_iterations: int) -> bool:
    """Count measured cycles after the first crossing, including on resume."""
    for position, iteration in enumerate(iterations):
        if iteration.saturated:
            return len(iterations) - position - 1 >= extra_iterations
    return False


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
    #: Total mu_w gap, membrane - bulk: excess part + kT ln(rho_m/rho_b).
    mu_gap: float | None = None
    #: The stop flag: sampling adequate and total gap >= 0 (SaturationTest.crossed).
    saturated: bool = False
    geometrically_saturated: bool = False
    sampling_adequate: bool = False
    free_volume_fraction: float = 0.0
    wall_seconds: float = 0.0
    #: mu_ex,m - mu_ex,b (the pre-``total_mu_v1`` stop quantity), kcal/mol.
    mu_gap_excess: float | None = None
    #: kT ln(rho_w,m / rho_w,b), kcal/mol; mu_gap = mu_gap_excess + density_term.
    density_term: float | None = None
    #: (N + 1) / V of the cell mu_ex was sampled in, molecules / A^3.
    rho_water: float | None = None
    #: Volume (A^3) of the cell mu_ex was sampled in: the NVT FEP cell, or the
    #: NPT mean for Widom. ``volume`` above stays the NPT mean.
    mu_volume: float | None = None
    #: Gap within tolerance*sigma of zero -- reported, not a stop condition.
    gap_within_noise: bool = False

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
    #: The dry stage's convergence record, or None if the dry membrane predates
    #: the gate. A result built on an unconverged dry cell is a lower bound at
    #: best, and that has to be visible in the saved state, not only in a log.
    dry_convergence: dict[str, object] | None = None
    #: Interpolated zero of the total mu gap (aemwater.saturation), or None
    #: when no trustworthy crossing was recorded.
    saturation_point: object | None = None
    #: Bulk water number density (N_b + 1) / V_b used in the density term.
    bulk_rho_water: float | None = None
    #: lambda and wt% at ``saturation_point.n_waters`` (None without a crossing).
    saturation_lambda: float | None = None
    saturation_uptake_pct: float | None = None

    @property
    def has_ionic_groups(self) -> bool:
        """Whether lambda is defined for this composition.

        Read from the composition record rather than from ``lambda_value``: a
        NaN lambda and a zero IEC are the same condition here, but the
        composition is the cause and reading the cause keeps the two reporting
        paths from drifting apart again.
        """
        count = self.composition.get("total_ionic_groups")
        return bool(count) if count is not None else math.isfinite(self.lambda_value)

    def summary(self) -> dict[str, object]:
        first_crossing = next((i for i in self.iterations if i.saturated), None)
        return {
            "first_saturation_crossing": (first_crossing.to_row() if first_crossing else None),
            "n_waters": self.n_waters,
            # None, not NaN: `json.dumps` writes a bare `NaN` literal, which no
            # strict JSON reader will parse, and this dict is written to
            # result.json at the end of a multi-hour run.
            "lambda_waters_per_ionic_group": (
                round(self.lambda_value, 3)
                if math.isfinite(self.lambda_value) else None
            ),
            "lambda_undefined_reason": (
                None if math.isfinite(self.lambda_value)
                else "composition has no ionic groups (IEC = 0)"
            ),
            "water_uptake_wt_pct": round(self.water_uptake_pct, 2),
            "hydrated_density_g_cm3": round(self.hydrated_density, 4),
            "dry_density_g_cm3": round(self.dry_density, 4),
            "bulk_mu_ex_kcal_mol": round(self.bulk_mu_ex, 3),
            "bulk_rho_water_per_A3": self.bulk_rho_water,
            "gap_definition": GAP_DEFINITION,
            # The equilibrium estimate: the interpolated zero of the total
            # gap. n_waters / lambda / wt% above are the *final loaded*
            # state, which includes any post-saturation overshoot.
            "saturation_point": (self.saturation_point.summary()
                                 if self.saturation_point is not None else None),
            "saturation_lambda": (
                round(self.saturation_lambda, 3)
                if self.saturation_lambda is not None
                and math.isfinite(self.saturation_lambda) else None
            ),
            "saturation_uptake_wt_pct": (
                round(self.saturation_uptake_pct, 2)
                if self.saturation_uptake_pct is not None else None
            ),
            "stop_reason": self.stop_reason,
            "converged": self.converged,
            "convergence_scope": "single_trajectory_crossing",
            "iterations": len(self.iterations),
            "dry_converged": (None if self.dry_convergence is None
                              else bool(self.dry_convergence.get("converged"))),
            "dry_convergence": self.dry_convergence,
        }

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame([it.to_row() for it in self.iterations])


def next_batch_size(
    n_current: int,
    n_reference_sites: int,
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

    ``mu_gap`` is the total water chemical-potential gap, membrane - bulk
    (excess part plus kT ln(rho_m/rho_b), see :class:`SaturationTest`);
    negative means water is still driven in. The first iteration has no measurement yet and gets a batch sized
    from ``n_reference_sites``, since roughly one water per site is a safe first
    step.

    ``n_reference_sites`` is the ionic-group count for a charged membrane. It is
    only a step-size scale, not part of any reported quantity, so an uncharged
    composition passes a different site count (see :func:`run_uptake`) rather
    than zero -- zero would collapse every batch to ``min_batch`` and make a
    hydrophobic polymer take hundreds of iterations to reach the same endpoint.
    """
    if mu_gap is None:
        return max(min_batch, min(max_batch, int(round(n_reference_sites))))

    base = max(min_batch, int(round(initial_fraction * max(n_current, n_reference_sites))))
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
    """Count consecutive zero-insertion attempts; partial batches are progress."""
    return previous + 1 if requested > 0 and inserted == 0 else 0


def hydration_number(n_waters: int, n_ionic_groups: int) -> float:
    """lambda: waters per ionic group, the standard AEM hydration measure.

    Returns NaN for an uncharged composition rather than raising. A neutral
    polymer (polyethylene, unfunctionalised polystyrene) has IEC = 0, so lambda
    is 0/0 -- undefined, not erroneous. The *measurement* is unaffected: the
    saturation criterion is a chemical-potential difference and the mass uptake
    is 100 * m_water / m_dry, neither of which references the ionic-group count.
    Raising here discarded a completed run because one of two reporting
    conventions did not apply to it.

    NaN, not zero: zero would claim the membrane takes up no water per ionic
    group, which is a measurement, and it would average into a campaign mean
    and pull it toward a number that means nothing. NaN propagates and is
    excluded explicitly wherever lambda is reported.

    This matches :meth:`aemwater.chemistry.SystemComposition.lambda_from_n_water`,
    which has always returned NaN for the same input; the two disagreeing was
    the actual defect.
    """
    if n_ionic_groups <= 0:
        return float("nan")
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


def _migrate_gap_definition(iterations, workdir: Path, bulk_reference,
                            config) -> list[Iteration]:
    """Recompute stored gaps and stop flags of a pre-``total_mu_v1`` checkpoint.

    Older checkpoints stored the excess-only gap and a ``>= -k sigma`` stop
    flag. Every input of the total gap is on disk -- mu_ex, the sampling
    verdict (which never depended on the gap), the water count and the cell --
    so the migration is exact rather than a restart: for FEP the cell is the
    iteration's ``relaxed.data`` box, which is the box its NVT windows ran in.
    Only if that file is gone does it fall back to the NPT-mean volume, and it
    says so.
    """
    rho_bulk = _bulk_water_density(bulk_reference)
    bulk_mu = float(bulk_reference.mu_ex.mu_ex)
    temperature = float(config.md.temperature)
    migrated = []
    for it in iterations:
        if it.mu_ex is None or not math.isfinite(it.mu_ex):
            migrated.append(it)
            continue
        mu_volume = float(it.volume)
        if config.mu_ex_method == "fep":
            relaxed = workdir / f"iter_{it.index:03d}" / "relaxed.data"
            if relaxed.exists():
                mu_volume = float(_read_final_state(relaxed)[2]) ** 3
            else:
                LOG.warning("migrating iteration %d: %s missing, using the "
                            "NPT-mean volume for the density term",
                            it.index, relaxed)
        rho = water_number_density(int(it.n_waters_after), mu_volume)
        excess = float(it.mu_ex) - bulk_mu
        density_term = KB_KCAL * temperature * math.log(rho / rho_bulk)
        gap = excess + density_term
        combined = math.hypot(it.mu_ex_stderr or 0.0,
                              float(bulk_reference.mu_ex.stderr or 0.0))
        adequate = bool(it.sampling_adequate)
        migrated.append(replace(
            it, mu_gap=gap, mu_gap_excess=excess, density_term=density_term,
            rho_water=rho, mu_volume=mu_volume,
            saturated=adequate and gap >= 0.0,
            gap_within_noise=adequate and gap >= -config.widom.sigma_tolerance * combined,
        ))
    return migrated


def _write_state(path: Path, payload: dict) -> None:
    """Checkpoint the loop so an interrupted run resumes instead of restarting.

    Atomic, because this file is rewritten after *every* iteration: it is the
    checkpoint most likely to be the one in flight when a walltime kill lands,
    and a half-written copy would make the next run discard a whole hydration
    trajectory rather than the last iteration.
    """
    write_json(path, payload)


def _uptake_saturation_test(membrane_estimate, bulk, config, *,
                            rho_membrane=None, rho_bulk=None):
    """Judge a local crossing without claiming between-cell convergence.

    ``rho_membrane`` / ``rho_bulk`` are the (N + 1) / V water number densities
    of the two cells the mu_ex values were sampled in. The driver always passes
    both so ``difference`` is the total mu_w gap; they are optional only so the
    local-precision gates can be tested in isolation.
    """
    from .fep.campaign import FEPEstimate

    local_ok = None
    if isinstance(membrane_estimate, FEPEstimate):
        local_ok = (
            math.isfinite(membrane_estimate.mu_ex)
            and math.isfinite(membrane_estimate.stderr)
            and 0 <= 1.96 * membrane_estimate.stderr <= config.fep.max_stderr
            and len(membrane_estimate.per_morphology) == 1
        )
        for morphology in membrane_estimate.per_morphology:
            local_ok = local_ok and morphology.usable and set(morphology.legs) == {"lj", "coul"}
            for leg in morphology.legs.values():
                overlaps = leg.diagnostics.get("neighbour_overlap", [])
                counts = leg.diagnostics.get("N_k", [])
                local_ok = local_ok and bool(overlaps) and all(
                    math.isfinite(x) and x >= config.fep.min_overlap for x in overlaps
                ) and bool(counts) and min(counts) >= config.fep.min_effective_samples
    return SaturationTest(
        membrane_estimate, bulk, tolerance_sigma=config.widom.sigma_tolerance,
        membrane_converged=local_ok, rho_membrane=rho_membrane,
        rho_bulk=rho_bulk, temperature=config.md.temperature,
    )


def _saturation_fields(test: SaturationTest, rho_membrane: float,
                       mu_volume: float) -> dict[str, object]:
    """The per-iteration saturation record shared by every code path."""
    return {
        "mu_gap": test.difference,
        "mu_gap_excess": test.excess_difference,
        "density_term": test.density_term,
        "rho_water": rho_membrane,
        "mu_volume": mu_volume,
        "saturated": test.crossed,
        "gap_within_noise": test.saturated,
        "trustworthy": test.trustworthy,
    }


def _bulk_water_density(bulk_reference) -> float:
    """(N_b + 1) / V_b of the bulk cell, refusing a reference without a volume."""
    try:
        return float(bulk_reference.water_number_density)
    except (AttributeError, TypeError, ValueError) as exc:
        raise DriverError(
            "the bulk reference carries no usable cell volume, so the water "
            "density term of the chemical-potential gap cannot be formed; "
            f"recompute the reference ({exc})"
        ) from exc


def _membrane_mu_ex_fep(config, stage: Path, contents, coords, edge, step: int):
    """mu_ex of water in the relaxed cell, by FEP on that one cell.

    One cell, not ``fep.n_morphologies`` of them. The between-morphology term
    for the reported answer comes from replicating the whole uptake loop
    (``aemwater campaign``), because M ghost insertions into a single
    trajectory's cell share that cell's packing entirely and would report
    sampling noise as structural heterogeneity -- see docs/fep_design.md,
    "Uptake averaged over morphologies". So the per-iteration spec is forced to
    one morphology here; anything else would double-count replication and make
    ``run_membrane_campaign`` refuse the single cell it is given.
    """
    from .assembly import assemble
    from .fep.campaign import run_membrane_campaign, write_campaign_report

    # Honor the supplied sampling settings. Campaign screening, when selected,
    # is applied by the campaign caller before reaching this helper.
    spec = replace(config.fep, n_morphologies=1)
    cell = assemble(contents, coords, edge=edge,
                    water_model_name=config.water_model)
    estimate = run_membrane_campaign(
        replace(config, fep=spec), stage / "fep", systems=[cell],
        ranks=config.md.mpi_ranks,
    )
    write_campaign_report(estimate, stage / "fep_membrane.json")
    LOG.info("iteration %d: membrane mu_ex = %.3f +/- %.3f kcal/mol (FEP)",
             step, estimate.mu_ex, estimate.stderr)
    return estimate


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

    # Under FEP the insertion block's output is never read, so paying for it
    # would add the whole Widom sampling run to every iteration for nothing.
    # The template keys still up: `enabled` false drops the fix and the run.
    widom_spec = (replace(config.widom, enabled=False)
                  if config.mu_ex_method == "fep" else config.widom)

    render_input(
        "insert.in.j2", stage / "in.insert",
        md=md, widom=widom_spec, title=f"iteration {step}",
        data_file="start.data", pair_coeff_lines=pair_coeff_lines(system),
        extra_types=None, comm_cutoff=comm_cutoff(md),
        minim=minimise_spec(md), soft=soft_push_spec(md),
        constraints=constraint_spec(
            md, system.water_bond_type(), system.water_angle_type(),
            has_widom=widom_spec.enabled),
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

    # The membrane estimator must match the one that produced the reference.
    # The saturation test is a difference, and for Widom it is only meaningful
    # because both halves carry the same insertion bias (module docstring). A
    # converged FEP reference against an under-converged Widom membrane number
    # differs by that bias -- several kcal/mol -- and would report saturation
    # many waters early while looking entirely plausible.
    new_coords, new_elements, new_edge = _read_final_state(stage / "relaxed.data")
    if config.mu_ex_method == "fep":
        # The relaxed configuration and box, not the freshly-inserted ones:
        # mu_ex is a property of the equilibrated ensemble, and `all_coords`
        # still carries the insertion geometry that the NPT stage just relaxed
        # (and `edge` its pre-relaxation box, which NPT has since changed).
        est = _membrane_mu_ex_fep(config, stage, contents, new_coords,
                                  new_edge, step)
    else:
        est = read_widom_file(stage / "mu.dat", md.temperature,
                              n_blocks=config.widom.n_blocks)
    # The density term uses the cell mu_ex was sampled in. FEP windows run
    # NVT in the relaxed box handed to _membrane_mu_ex_fep, so its volume is
    # exact; Widom samples during NPT, so its natural volume is the NPT mean.
    mu_volume = float(new_edge) ** 3 if config.mu_ex_method == "fep" else volume
    rho_membrane = water_number_density(n_waters, mu_volume)
    test = _uptake_saturation_test(
        est, bulk_reference.mu_ex, config, rho_membrane=rho_membrane,
        rho_bulk=_bulk_water_density(bulk_reference))
    return {
        "coords": new_coords, "elements": new_elements, "edge": new_edge,
        "density": density, "volume": volume, "mu_ex": est.mu_ex,
        "stderr": est.stderr if np.isfinite(est.stderr) else 0.0,
        **_saturation_fields(test, rho_membrane, mu_volume),
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

#: For FEP, ``insertions_per_call`` is dropped. It is reference-critical for
#: Widom because it sets how far up the tail of the cavity distribution the
#: estimate has climbed -- i.e. it changes the bias, and the saturation
#: criterion is a difference of two comparably-biased numbers. FEP has no such
#: dependence: the parameter does not enter an alchemical calculation at all, so
#: requiring it to match would reject a perfectly valid reference.
_REFERENCE_CRITICAL_FEP = (
    "water_model", "temperature", "cutoff",
)


def _check_reference_matches(reference, wanted, method: str = "widom") -> None:
    """Refuse a bulk reference computed at different settings.

    The saturation criterion is a difference of two mu_ex estimates and is only
    meaningful because their biases are comparable (see the module docstring).
    Comparing against a reference run at a different water model or cutoff
    silently produces a number that looks like an uptake and is not one.

    ``kspace_accuracy`` is checked for Widom but not for FEP: the FEP path
    deliberately overrides it with the tighter ``fep.kspace_accuracy`` (the
    charge-leg dU carries a PPPM grid error that ordinary MD never sees), so the
    value in ``BulkSettings`` is not the one that was used.
    """
    critical = (_REFERENCE_CRITICAL_FEP if method == "fep"
                else _REFERENCE_CRITICAL)
    mismatched = [
        f"{name}: reference {getattr(reference.settings, name)!r} "
        f"!= run {getattr(wanted, name)!r}"
        for name in critical
        if getattr(reference.settings, name) != getattr(wanted, name)
    ]
    if mismatched:
        raise ValueError(
            "bulk reference was computed at different settings, so the "
            "chemical-potential difference is not meaningful:\n  "
            + "\n  ".join(mismatched)
        )


def bulk_settings_for(config):
    """The reservoir settings implied by a run config.

    Extracted so a caller that needs the reference *before* the loop (the
    multi-morphology campaign, which shares one reference across trajectories)
    derives it from the same expression the loop does. Two copies of this would
    drift and the mismatch would surface as a spurious
    ``_check_reference_matches`` failure.
    """
    from .bulk import BulkSettings

    return BulkSettings(
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


def obtain_bulk_reference(config, workdir: Path | str, ranks: int | None = None,
                          resume: bool = True):
    """Compute (or load from cache) the bulk reference for this config.

    Dispatches on ``config.mu_ex_method`` so the reference is measured by the
    same estimator as the membrane. Mixing them would compare a FEP membrane
    number against a Widom reservoir number, and since the Widom bias in bulk
    water is several kcal/mol, the saturation point would move by more than the
    effect being measured.
    """
    from .bulk import run_bulk_reference, run_bulk_reference_fep

    settings = bulk_settings_for(config)
    workdir = Path(workdir)
    ranks = config.md.mpi_ranks if ranks is None else ranks
    LOG.info("computing bulk reference by %s", config.mu_ex_method.upper())
    if config.mu_ex_method == "fep":
        return run_bulk_reference_fep(
            config, settings, workdir,
            cache_dir=config.widom.cache_dir, ranks=ranks, resume=resume,
        )
    return run_bulk_reference(
        settings, workdir, cache_dir=config.widom.cache_dir, ranks=ranks,
        resume=resume,
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

    # An uncharged composition (polyethylene, unfunctionalised polystyrene) is a
    # legitimate input -- it is the hydrophobic control that says how much of a
    # real AEM's uptake comes from the ionic groups rather than from free
    # volume. It has no IEC, so lambda is undefined and only the mass uptake is
    # reported; that is handled at the reporting sites.
    #
    # The batch-size scale still needs a number. Repeat units stand in for ionic
    # groups: it keeps the first batch and the swelling steps on the same scale
    # as a charged run of the same size, so the two are comparable in cost and
    # in how finely they approach saturation.
    n_batch_sites = n_ionic if n_ionic > 0 else comp.n_chains * comp.chain_length
    if n_ionic == 0:
        LOG.warning(
            "composition has no ionic groups (IEC = 0): lambda is undefined and "
            "will be reported as null. The saturation criterion and the mass "
            "uptake (wt %%) are unaffected. Sizing insertion batches from %d "
            "repeat units instead.",
            n_batch_sites,
        )

    # --- the reservoir -----------------------------------------------------
    bulk_settings = bulk_settings_for(config)
    if bulk_reference is None:
        bulk_reference = obtain_bulk_reference(config, workdir / "bulk",
                                               resume=resume)
    _check_reference_matches(bulk_reference, bulk_settings,
                            method=getattr(bulk_reference, "method", "widom"))
    for issue in bulk_reference.sanity():
        LOG.warning("bulk reference: %s", issue)

    if not bulk_reference.mu_ex.converged:
        raise DriverError("bulk reference is unconverged; improve its sampling or "
                          "replication before running uptake")

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

    # The uptake number inherits the dry cell's quality, so carry that
    # judgement forward instead of leaving it in the prepare-stage log. A
    # membrane that had not finished densifying keeps densifying once water is
    # in it, which shows up as water apparently shrinking the cell and a
    # lambda biased high by void filling. This is a read, not a re-check: the
    # dry stage may have run days ago or under a different config.
    dry_convergence: dict[str, object] | None = None
    conv_file = workdir / "dry" / "convergence.json"
    if conv_file.exists():
        try:
            dry_convergence = json.loads(conv_file.read_text())
        except (OSError, ValueError) as exc:  # pragma: no cover - defensive
            LOG.warning("could not read %s: %s", conv_file, exc)
    if dry_convergence is None:
        LOG.warning(
            "no dry-stage convergence record at %s -- the dry membrane predates "
            "the convergence gate, so its density is unverified", conv_file)
    elif not dry_convergence.get("converged"):
        LOG.warning(
            "dry membrane did NOT pass its convergence check (%.4f g/cm3, "
            "drift %+.4f g/cm3 per 100 ps). Uptake from this cell is likely "
            "biased high by void filling.",
            dry_convergence.get("density_g_cm3", float("nan")),
            dry_convergence.get("drift_g_cm3_per_100ps", float("nan")))

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
    baseline_measured = False

    saved = read_json_or_none(state_file, description="uptake checkpoint") \
        if resume else None
    if saved is not None:
        iterations = [Iteration(**row) for row in saved.get("iterations", [])]
        n_waters = saved.get("n_waters", 0)
        mu_gap = saved.get("mu_gap")
        stderr = saved.get("stderr", 0.0)
        if saved.get("gap_definition") != GAP_DEFINITION and iterations:
            LOG.warning(
                "uptake checkpoint predates the total-mu saturation criterion "
                "(%s); recomputing %d stored gaps with the water density term "
                "and re-judging the stop condition",
                saved.get("gap_definition", "excess-only"), len(iterations))
            iterations = _migrate_gap_definition(
                iterations, workdir, bulk_reference, config)
            measured = [it for it in iterations if it.mu_gap is not None]
            if measured:
                mu_gap = measured[-1].mu_gap
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
        baseline_measured = bool(saved.get("baseline_measured", False))
        LOG.info(
            "resuming at iteration %d with %d waters from %s "
            "(%d consecutive zero-insertion attempts)",
            start_step, n_waters, resume_data, failed_batches,
        )
        if _saturation_complete(iterations, config.insertion.post_saturation_iterations):
            stop_reason = "thermodynamic_saturation"
            converged = True

    # Measure the zero-water endpoint before changing the composition. The dry
    # data file is also this point's relaxed checkpoint, copied into iter_000 so
    # an interruption immediately after FEP resumes without special cases.
    if saved is None and config.mu_ex_method == "fep":
        t0 = time.time()
        stage = workdir / "iter_000"
        stage.mkdir(exist_ok=True)
        shutil.copy2(dry_data, stage / "relaxed.data")
        contents = CellContents(
            chains=typed_chains,
            ions=ion_molecules(comp.n_counterions, comp.counterion),
            waters=[],
        )
        est = _membrane_mu_ex_fep(config, stage, contents, coords, edge, 0)
        # N = 0: the ghost is the only water, so rho = 1/V. Finite, which is
        # why the (N + 1)/V convention matters most exactly here.
        rho_membrane = water_number_density(0, float(edge) ** 3)
        test = _uptake_saturation_test(
            est, bulk_reference.mu_ex, config, rho_membrane=rho_membrane,
            rho_bulk=_bulk_water_density(bulk_reference))
        fields = _saturation_fields(test, rho_membrane, float(edge) ** 3)
        mu_gap = fields["mu_gap"]
        stderr = est.stderr if np.isfinite(est.stderr) else 0.0
        iterations.append(Iteration(
            index=0, n_waters_before=0, n_requested=0, n_inserted=0,
            n_waters_after=0, density=dry_density, volume=edge ** 3,
            lambda_value=hydration_number(0, n_ionic), water_uptake_pct=0.0,
            mu_ex=est.mu_ex, mu_ex_stderr=stderr, mu_gap=mu_gap,
            saturated=fields["saturated"], geometrically_saturated=False,
            sampling_adequate=fields["trustworthy"],
            free_volume_fraction=0.0, wall_seconds=time.time() - t0,
            mu_gap_excess=fields["mu_gap_excess"],
            density_term=fields["density_term"], rho_water=rho_membrane,
            mu_volume=fields["mu_volume"],
            gap_within_noise=fields["gap_within_noise"],
        ))
        baseline_measured = True
        start_step = 1
        _write_state(state_file, {
            "iterations": [i.to_row() for i in iterations],
            "n_waters": 0, "mu_gap": mu_gap, "stderr": stderr,
            "failed_batches": 0, "next_step": start_step,
            "gap_definition": GAP_DEFINITION,
            "baseline_measured": True,
        })
        LOG.info("iteration 0: dry baseline, uptake = 0.0%%, mu_ex = %.3f "
                 "(gap %.3f kcal/mol)", est.mu_ex, mu_gap)
        if _saturation_complete(iterations, config.insertion.post_saturation_iterations):
            stop_reason = "thermodynamic_saturation"
            converged = True

    # Baseline FEP is additional to the configured insertion-cycle budget.
    stop_step = config.insertion.max_iterations + (1 if baseline_measured else 0)
    steps = range(start_step, stop_step) if not converged else ()
    for step in steps:
        t0 = time.time()
        n_add = next_batch_size(
            n_waters, n_batch_sites, mu_gap, stderr,
            initial_fraction=config.insertion.batch_fraction,
            min_batch=config.insertion.min_batch_size,
            max_batch=config.insertion.batch_size,
        )
        if iterations and 0 < iterations[-1].n_inserted < iterations[-1].n_requested:
            n_add = min(n_add, max(config.insertion.min_batch_size,
                                   iterations[-1].n_inserted))
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
                "gap_definition": GAP_DEFINITION,
                "baseline_measured": baseline_measured,
            })
            if failed_batches >= config.insertion.max_failed_batches:
                stop_reason = "insertion_stalled"
                LOG.info(
                    "iteration %d: no cavity accepts another water; stopping after "
                    "%d consecutive zero-insertion attempts",
                    step, failed_batches,
                )
                break
            LOG.warning(
                "iteration %d: no cavity accepts another water "
                "(%d/%d consecutive zero-insertion attempts); retrying",
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
            sampling_adequate=state.get("trustworthy", False),
            geometrically_saturated=False,
            free_volume_fraction=result.void_map.free_volume_fraction,
            wall_seconds=time.time() - t0,
            mu_gap_excess=state.get("mu_gap_excess"),
            density_term=state.get("density_term"),
            rho_water=state.get("rho_water"),
            mu_volume=state.get("mu_volume"),
            gap_within_noise=state.get("gap_within_noise", False),
        )
        iterations.append(it)
        _write_state(state_file, {
            "iterations": [i.to_row() for i in iterations],
            "n_waters": n_waters, "mu_gap": mu_gap, "stderr": stderr,
            "failed_batches": failed_batches, "next_step": step + 1,
            "gap_definition": GAP_DEFINITION,
            "baseline_measured": baseline_measured,
        })
        LOG.info(
            "iteration %d: +%d -> %d waters, lambda = %.2f, uptake = %.1f%%, "
            "mu_ex = %.3f (gap %.3f kcal/mol)",
            step, result.n_inserted, n_waters, it.lambda_value,
            it.water_uptake_pct, state["mu_ex"], mu_gap,
        )

        if _saturation_complete(iterations, config.insertion.post_saturation_iterations):
            stop_reason = "thermodynamic_saturation"
            converged = True
            break
    else:
        if not converged:
            LOG.warning(
                "reached max_iterations (%d) before completing saturation and "
                "the requested post-saturation cycles", config.insertion.max_iterations
            )

    final = iterations[-1] if iterations else None
    point = estimate_saturation_point(
        iterations, bulk_stderr=float(bulk_reference.mu_ex.stderr or 0.0))
    if point is not None:
        LOG.info(
            "total mu gap crosses zero at %.1f waters (95%% CI %.1f-%.1f, %s "
            "over %d points)", point.n_waters, point.n_waters_low,
            point.n_waters_high, point.method, point.n_points)
    return UptakeResult(
        iterations=iterations,
        n_waters=n_waters,
        # No `if n_waters` guard: hydration_number(0, n>0) is already 0.0, and
        # for an uncharged composition the guard was the one path that returned
        # a hard 0.0 for an undefined lambda.
        lambda_value=hydration_number(n_waters, n_ionic),
        water_uptake_pct=water_uptake_percent(n_waters, dry_mass),
        hydrated_density=final.density if final else dry_density,
        dry_density=dry_density,
        stop_reason=stop_reason,
        converged=converged,
        bulk_mu_ex=bulk_reference.mu_ex.mu_ex,
        workdir=workdir,
        composition=comp.summary() if hasattr(comp, "summary") else {},
        dry_convergence=dry_convergence,
        saturation_point=point,
        bulk_rho_water=_bulk_water_density(bulk_reference),
        saturation_lambda=(None if point is None
                           else hydration_number(point.n_waters, n_ionic)),
        saturation_uptake_pct=(None if point is None
                               else water_uptake_percent(point.n_waters, dry_mass)),
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
