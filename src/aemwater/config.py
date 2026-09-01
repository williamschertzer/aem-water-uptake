"""Configuration schema for an AEM water-uptake calculation.

Everything the workflow needs is expressed as frozen dataclasses so a run is
fully described by one YAML file (plus CLI overrides). ``validate()`` is called
on construction and raises on non-physical input rather than letting LAMMPS
fail 40 minutes later.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .fep.schedule import (
    DEFAULT_COUL_LAMBDAS,
    DEFAULT_LJ_LAMBDAS,
    SCREENING_COUL_LAMBDAS,
    SCREENING_LJ_LAMBDAS,
)
from .utils import LOG

SUPPORTED_COUNTERIONS = ("Cl-", "Br-", "OH-", "HCO3-")
SUPPORTED_WATER_MODELS = ("spce", "tip3p")

#: Estimators available for the excess chemical potential. ``"fep"`` is the
#: alchemical ghost-particle path (default); ``"widom"`` is test-particle
#: insertion, kept as an independent cross-check.
SUPPORTED_MU_EX_METHODS = ("fep", "widom")


class ConfigError(ValueError):
    """Raised when a configuration is internally inconsistent."""


def _is_tuple_field(annotation: Any) -> bool:
    """Whether a dataclass field annotation denotes a tuple.

    ``from __future__ import annotations`` is in force, so annotations reach us as
    strings rather than as types; matching on the text is the honest way to read
    them without importing ``typing.get_type_hints`` machinery that would have to
    resolve every name in the module.
    """
    return isinstance(annotation, str) and annotation.lstrip().startswith("tuple")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


@dataclass(frozen=True)
class PolymerSpec:
    """The chemistry of the membrane.

    Attributes
    ----------
    smiles:
        Repeat-unit SMILES. Either mark the two backbone attachment points with
        dummy atoms (``[*]``), e.g.
        ``[*]CC([*])c1ccc(C[N+](C)(C)C)cc1``, or supply a polymerisable vinyl
        monomer (``C=C...``) which is opened automatically.
    n_chains:
        Number of polymer chains in the periodic cell.
    chain_length:
        Repeat units per chain.
    counterion:
        Mobile anion balancing the fixed cationic charge.
    terminal_group:
        Capping chemistry for both chain ends (``"H"`` or ``"CH3"``).
    charge_method:
        antechamber charge model. ``"bcc"`` (AM1-BCC) is the GAFF2 default and
        what the force field was parameterised against. ``"gas"`` (Gasteiger) is
        far cheaper and appropriate only for a smoke test: it gets the
        electrostatics of a quaternary ammonium cation badly wrong, which is
        precisely the interaction that drives water uptake.
    """

    smiles: str
    n_chains: int = 4
    chain_length: int = 10
    counterion: str = "Cl-"
    terminal_group: str = "CH3"
    charge_method: str = "bcc"
    name: str = "AEM"

    def validate(self) -> None:
        _check(bool(self.smiles and self.smiles.strip()), "polymer.smiles must not be empty")
        _check(self.n_chains >= 1, f"polymer.n_chains must be >= 1 (got {self.n_chains})")
        _check(self.chain_length >= 1, f"polymer.chain_length must be >= 1 (got {self.chain_length})")
        _check(
            self.counterion in SUPPORTED_COUNTERIONS,
            f"polymer.counterion must be one of {SUPPORTED_COUNTERIONS} (got {self.counterion!r})",
        )
        _check(
            self.terminal_group.upper() in ("H", "CH3"),
            f"polymer.terminal_group must be 'H' or 'CH3' (got {self.terminal_group!r})",
        )
        _check(
            self.charge_method in ("bcc", "gas", "mul"),
            f"polymer.charge_method must be 'bcc', 'gas' or 'mul' "
            f"(got {self.charge_method!r})",
        )


@dataclass(frozen=True)
class BoxSpec:
    """Initial packing of the simulation cell."""

    #: Density of the initial loose gas-phase packing, g/cm^3. Deliberately low
    #: so chains do not overlap before the compression schedule runs.
    initial_density: float = 0.20
    #: Target dry-membrane density used by the compression schedule, g/cm^3.
    target_density: float = 1.10
    #: Minimum initial separation between atoms of different molecules, Angstrom.
    min_separation: float = 2.2
    #: RNG seed for placement.
    seed: int = 20260817
    #: Maximum random-rotation attempts per molecule before enlarging the cell.
    max_place_attempts: int = 400

    def validate(self) -> None:
        _check(0.001 < self.initial_density < 2.0, "box.initial_density must be in (0.001, 2.0) g/cm^3")
        _check(0.1 < self.target_density < 3.0, "box.target_density must be in (0.1, 3.0) g/cm^3")
        _check(
            self.initial_density < self.target_density,
            "box.initial_density must be below box.target_density (packing then compresses)",
        )
        _check(self.min_separation > 0.5, "box.min_separation must exceed 0.5 Angstrom")


@dataclass(frozen=True)
class EquilibrationSpec:
    """Dry-membrane equilibration schedule.

    The default is the 21-step compression/decompression scheme of Larsen,
    Lin and Colina (Macromolecules 2011; also the equilibration stage of
    Polymatic, Abbott/Colina 2013), which is the standard route to a
    converged amorphous glassy polymer density.

    Why it works where a single squeeze does not. Collapsing a loose packing
    to glassy density is not a barostat problem, it is a sampling problem: the
    chains have to find melt conformations, and at 300 K they cannot. The
    scheme interleaves seven NVT excursions to ``high_temperature`` -- above
    Tg, where chains interpenetrate -- with NPT compressions, and it *ramps
    the pressure up to ``max_pressure`` and then back down to
    ``md.pressure``*. The decompression half is the part that a naive
    "compress hard then release" cycle omits, and it is what removes the
    residual voids: the structure is repeatedly over-compressed, allowed to
    relax hot, and then let out in stages, so the final density is approached
    from the dense side at every scale rather than once at the end.

    The measured failure mode this replaces is recorded in
    ``equilibrate.in.j2`` and in the run logs: a single hot squeeze plus
    release plateaus around 0.95 g/cm^3 for BTMA-polystyrene, ~18% below
    experiment, with density still drifting upward when the run ends. The
    missing 18% is void space, and the water-uptake loop then fills it --
    giving a *negative* partial molar volume for water (the cell contracts as
    water is added) and an uptake number biased high without limit.

    Attributes
    ----------
    scheme:
        ``"21step"`` (default) or ``"legacy"`` for the single-squeeze cycle
        driven by ``md.anneal_steps`` / ``md.compression_steps``.
    max_pressure:
        Peak compression pressure, atm. 50 000 atm is the literature value.
    high_temperature:
        NVT excursion temperature, K. Must be above Tg.
    time_scale:
        Multiplies the duration of steps 1-20. 1.0 reproduces the published
        table; use ~0.01 for smoke tests. Does not change the pressure or
        temperature schedule, so a scaled run exercises the same code path
        rather than a different one. Step 21 is *not* scaled -- see below.
    final_npt_ps:
        Duration of step 21, the production NPT from which density is
        averaged, in ps. The published value is 800 ps and it dominates the
        cost of the dry stage. Set directly rather than scaled by
        ``time_scale``: this window has to stay longer than several
        ``md.thermo_every`` intervals or the density file comes out empty and
        the convergence gate has nothing to read.
    """

    scheme: str = "21step"
    max_pressure: float = 50_000.0
    high_temperature: float = 600.0
    time_scale: float = 1.0
    final_npt_ps: float = 800.0

    # --- convergence gate ---------------------------------------------------
    #: Refuse to hand a dry membrane to the uptake loop unless it converged.
    #: The uptake number is meaningless otherwise (see class docstring), so the
    #: default is to fail loudly rather than produce a plausible wrong answer.
    enforce_convergence: bool = True
    #: Expected dry density, g/cm^3. ``None`` falls back to
    #: ``box.target_density``. For quaternary-ammonium polystyrene with
    #: halide/hydroxide counterions the experimental range is 1.10-1.25.
    expected_density: float | None = None
    #: Accepted fractional deviation from ``expected_density``.
    density_tolerance: float = 0.05
    #: Maximum |d(rho)/dt| over the production window, g/cm^3 per 100 ps.
    #: A structure still densifying is still equilibrating.
    drift_tolerance: float = 0.002

    def validate(self) -> None:
        _check(
            self.scheme in ("21step", "legacy"),
            f"equilibration.scheme must be '21step' or 'legacy' (got {self.scheme!r})",
        )
        _check(
            self.max_pressure > 0,
            "equilibration.max_pressure must be positive",
        )
        _check(
            self.high_temperature > 0,
            "equilibration.high_temperature must be positive",
        )
        _check(
            0 < self.time_scale <= 10.0,
            "equilibration.time_scale must be in (0, 10]",
        )
        _check(
            self.final_npt_ps > 0,
            "equilibration.final_npt_ps must be positive",
        )
        _check(
            0 < self.density_tolerance < 1.0,
            "equilibration.density_tolerance must be a fraction in (0, 1)",
        )
        _check(
            self.drift_tolerance > 0,
            "equilibration.drift_tolerance must be positive",
        )
        if self.expected_density is not None:
            _check(
                0.1 < self.expected_density < 3.0,
                "equilibration.expected_density must be in (0.1, 3.0) g/cm^3",
            )
        if self.scheme == "21step" and self.max_pressure < 1000.0:
            LOG.warning(
                "equilibration.max_pressure = %.0f atm is far below the 50000 atm "
                "the 21-step scheme is defined with; the compression half will "
                "not densify the cell.",
                self.max_pressure,
            )


@dataclass(frozen=True)
class MDSpec:
    """MD engine settings shared by every LAMMPS stage."""

    temperature: float = 300.0
    pressure: float = 1.0
    timestep: float = 1.0
    #: Real-space cutoff for LJ and Coulomb, Angstrom.
    cutoff: float = 10.0
    #: PPPM relative accuracy.
    kspace_accuracy: float = 1.0e-4
    #: Steps of soft-core push-off used to remove packing overlaps.
    soft_push_steps: int = 20000
    #: Steps of high-temperature NVT annealing (dry membrane).
    anneal_steps: int = 50000
    anneal_temperature: float = 600.0
    #: Steps per stage of the compression/decompression schedule.
    compression_steps: int = 20000
    compression_pressure: float = 1000.0
    #: Steps of NPT equilibration after the dry membrane is compressed.
    dry_npt_steps: int = 100000
    #: NPT relaxation steps after each water-insertion batch.
    relax_npt_steps: int = 20000
    #: Minimisation tolerance/iterations used after every insertion.
    min_etol: float = 1.0e-6
    min_ftol: float = 1.0e-6
    min_maxiter: int = 20000
    thermo_every: int = 500
    dump_every: int = 0  # 0 disables trajectory dumps
    #: Number of MPI ranks; 1 means run the serial binary directly.
    mpi_ranks: int = 1
    lammps_binary: str | None = None
    #: Temperature/pressure damping constants, fs.
    tdamp: float = 100.0
    pdamp: float = 1000.0
    #: RNG seed for velocity generation.
    seed: int = 87231

    def validate(self) -> None:
        _check(self.temperature > 0, "md.temperature must be positive")
        _check(self.timestep > 0, "md.timestep must be positive")
        _check(self.cutoff >= 6.0, "md.cutoff below 6 Angstrom is not sensible for GAFF2/SPC-E")
        _check(0 < self.kspace_accuracy < 1e-2, "md.kspace_accuracy must be in (0, 1e-2)")
        _check(self.mpi_ranks >= 1, "md.mpi_ranks must be >= 1")
        _check(self.anneal_temperature >= self.temperature, "md.anneal_temperature must be >= md.temperature")
        _check(self.pressure > 0, f"md.pressure must be positive (got {self.pressure})")
        _check(
            self.compression_pressure >= self.pressure,
            f"md.compression_pressure ({self.compression_pressure}) must be at "
            f"least md.pressure ({self.pressure}): the squeeze stage densifies "
            f"the melt under load before releasing to the operating pressure",
        )
        for f in ("soft_push_steps", "anneal_steps", "compression_steps", "dry_npt_steps", "relax_npt_steps"):
            _check(getattr(self, f) >= 0, f"md.{f} must be non-negative")


@dataclass(frozen=True)
class InsertionSpec:
    """Void-detection water insertion settings."""

    #: Waters added per iteration at the start of the run.
    #: Fraction of the current water content added per iteration. A fixed batch
    #: is a large perturbation when nearly dry and negligible when swollen, so
    #: the batch scales with content; this sets that scale.
    batch_fraction: float = 0.25
    batch_size: int = 20
    #: Batch size is halved near saturation; never go below this.
    min_batch_size: int = 2
    #: Grid spacing for the void map, Angstrom.
    grid_spacing: float = 0.5
    #: Radius of the spherical probe representing a water molecule, Angstrom.
    probe_radius: float = 1.4
    #: Minimum centre-to-centre distance between two inserted waters, Angstrom.
    water_water_min: float = 2.6
    #: Scale factor applied to van der Waals radii when computing clearance.
    vdw_scale: float = 1.0
    #: Consecutive geometric failures tolerated before declaring saturation.
    max_failed_batches: int = 3
    #: Hard cap on iterations.
    max_iterations: int = 60
    #: Hard cap on total waters (safety net); 0 means unlimited.
    max_waters: int = 0
    seed: int = 5150

    def validate(self) -> None:
        _check(self.batch_size >= 1, "insertion.batch_size must be >= 1")
        _check(1 <= self.min_batch_size <= self.batch_size, "insertion.min_batch_size must be in [1, batch_size]")
        _check(0.1 <= self.grid_spacing <= 2.0, "insertion.grid_spacing must be in [0.1, 2.0] Angstrom")
        _check(0.5 <= self.probe_radius <= 3.0, "insertion.probe_radius must be in [0.5, 3.0] Angstrom")
        _check(self.water_water_min > 1.5, "insertion.water_water_min must exceed 1.5 Angstrom")
        _check(self.max_failed_batches >= 1, "insertion.max_failed_batches must be >= 1")
        _check(self.max_iterations >= 1, "insertion.max_iterations must be >= 1")
        _check(self.max_waters >= 0, "insertion.max_waters must be >= 0")


@dataclass(frozen=True)
class WidomSpec:
    """Widom test-particle insertion settings (saturation criterion)."""

    enabled: bool = True
    #: Independent Widom segments; each yields one mu_ex estimate for blocking.
    n_blocks: int = 5
    #: Test insertions attempted per invocation of ``fix widom``.
    insertions_per_call: int = 400
    #: MD steps between Widom insertion bursts within a segment.
    every: int = 100
    #: MD steps per segment.
    steps_per_block: int = 10000
    #: Bulk reference box edge length, Angstrom.
    bulk_box_length: float = 25.0
    #: Equilibration steps for the bulk reference box.
    bulk_equil_steps: int = 50000
    #: Saturation is declared when delta_mu >= -tolerance*sigma (see widom.py).
    sigma_tolerance: float = 1.0
    #: Cache directory for the bulk reference (reused across runs).
    cache_dir: str = "~/.cache/aemwater"
    seed: int = 31337

    def validate(self) -> None:
        _check(self.n_blocks >= 2, "widom.n_blocks must be >= 2 for an error bar")
        _check(self.insertions_per_call >= 10, "widom.insertions_per_call must be >= 10")
        _check(self.every >= 1, "widom.every must be >= 1")
        _check(self.steps_per_block >= self.every, "widom.steps_per_block must be >= widom.every")
        _check(self.bulk_box_length >= 20.0, "widom.bulk_box_length must be >= 20 Angstrom (cutoff + skin)")
        _check(self.sigma_tolerance >= 0, "widom.sigma_tolerance must be non-negative")


@dataclass(frozen=True)
class FEPSpec:
    """Alchemical FEP settings for the excess chemical potential of water.

    Replaces Widom insertion as the default estimator. Widom is biased toward
    zero in a dense medium because its Boltzmann average is carried by rare
    cavity-landing trials; the ghost-particle path here has no rare event in it.

    Defaults are sized for a production membrane run on a cluster. The bulk
    validation and the smoke test override the sampling lengths downward; see
    ``examples/`` and ``docs/fep_design.md``.
    """

    #: Independently equilibrated polymer morphologies to average over. The
    #: between-morphology spread usually dominates the total uncertainty in a
    #: glassy matrix, so this is the knob that actually buys precision -- more
    #: sampling within one badly-chosen morphology does not. 1 is permitted for
    #: bulk water (where there is only one liquid) and for smoke tests.
    n_morphologies: int = 3

    #: Soft-core LJ ladder (leg 1), charges off. Must span exactly 0 -> 1.
    #: Denser at low lambda because a soft-core dU/dlambda is largest there.
    #: Defined in :mod:`aemwater.fep.schedule` so the ladder has one definition
    #: rather than two that agree by coincidence.
    lj_lambdas: tuple[float, ...] = DEFAULT_LJ_LAMBDAS
    #: Charge ladder (leg 2), LJ core fully present. Must span exactly 0 -> 1.
    coul_lambdas: tuple[float, ...] = DEFAULT_COUL_LAMBDAS

    #: Soft-core exponent ``n`` and the two alpha parameters, passed straight to
    #: the pair style. The published Beutler defaults; changing them changes the
    #: path but not the endpoints, so results remain comparable in principle --
    #: but only if bulk and membrane use the same values, which is enforced.
    soft_core_n: int = 1
    alpha_lj: float = 0.5
    alpha_coul: float = 10.0

    #: PPPM accuracy for FEP runs, overriding ``md.kspace_accuracy``.
    #:
    #: Measured, not guessed. The charge-leg dU on a fixed configuration of 195
    #: SPC/E waters carries a PPPM grid error of ~0.016 kcal/mol per state at the
    #: workflow default of 1e-4 -- and the same absolute error appears in a dilute
    #: cell, so it is grid resolution rather than anything physical. Accumulated
    #: over a 6-interval charge ladder with a consistent sign that is ~0.1
    #: kcal/mol against a 0.30 kcal/mol error budget, and no amount of sampling
    #: removes it. At 1e-6 the error falls below 2e-4 kcal/mol for an 11% cost
    #: increase. Ordinary MD does not need this because forces converge much
    #: faster than these energy *differences* do.
    kspace_accuracy: float = 1.0e-6

    #: Equilibration discarded at each lambda before sampling begins, steps.
    equil_steps: int = 50_000
    #: Production sampling per lambda, steps.
    production_steps: int = 500_000
    #: Interval between stored frames. With the defaults this gives 500 frames
    #: per state, which is ample after decorrelation.
    sample_every: int = 1_000

    #: Estimators to evaluate. All three run on the same data at negligible extra
    #: cost, and their disagreement is the most useful diagnostic available: MBAR
    #: and BAR differing by more than their error bars means poor overlap, and TI
    #: differing from both means the ladder is too coarse to integrate.
    estimators: tuple[str, ...] = ("mbar", "bar", "ti")
    #: Central-difference half-width for the TI dU/dlambda estimate.
    ti_delta: float = 0.01

    #: Build the full K x N energy matrix with a rerun pass over stored frames.
    #: Required by MBAR; BAR and TI can run without it from inline compute fep
    #: output alone.
    rerun_matrix: bool = True
    #: Keep per-lambda trajectories after the rerun pass. They are the only way
    #: to rebuild the matrix with a different ladder, but they are large.
    keep_trajectories: bool = False

    #: Add the analytic long-range LJ correction. ``compute fep`` refuses
    #: ``tail yes`` for soft styles while the production templates run
    #: ``pair_modify tail yes``, so the term is computed in post. It largely
    #: cancels in the membrane-minus-bulk difference but is kept for absolute
    #: numbers.
    tail_correction: bool = True

    #: Minimum nearest-neighbour overlap in the MBAR overlap matrix below which
    #: the ladder is reported as inadequate rather than silently trusted.
    min_overlap: float = 0.03
    #: Refuse to report a result whose statistical error exceeds this, kcal/mol.
    max_stderr: float = 0.30

    seed: int = 90210

    def at_screening_resolution(self) -> "FEPSpec":
        """A cheaper copy for the uptake loop's per-iteration mu_ex.

        The loop needs mu_ex at every water content to find saturation, and the
        location of saturation is decided by where two curves cross, not by the
        third decimal of either. This trades 6.4x in cost for roughly a factor
        of 2 in error bar:

        ==================  ===========  ===========
        quantity            production   screening
        ==================  ===========  ===========
        states (LJ+charge)  12 + 7       7 + 7
        production steps    500k         150k
        equil steps         50k          25k
        morphologies        3            2
        relative cost       1.0          0.156
        ==================  ===========  ===========

        Use :meth:`FEPSpec` unchanged for the final answer. The intended pattern
        is screening at every iteration and one production campaign at the
        saturation point. ``aemwater.driver`` applies this preset per iteration
        (with ``n_morphologies`` forced to 1, since the loop measures the single
        cell it is carrying); ``aemwater.uptake_campaign`` replicates the loop to
        get the between-morphology spread.

        The ladders are placed at equal thermodynamic length from the bulk
        SPC/E validation run's measured fluctuation profile; see
        :mod:`aemwater.fep.schedule` for why that metric and not equal work.

        Every substitution here is a *ceiling*, never an override. A preset
        whose job is to make the loop cheaper must not make it more expensive
        than what was configured: someone who deliberately sets 2k production
        steps for a pipeline smoke test would otherwise get 150k per state at
        every iteration, silently, and conclude the end-to-end run is
        unaffordable. The same reasoning already governed ``n_morphologies``
        and ``max_stderr``; it applies to the step counts and the ladders too.
        The ladders are compared on state count rather than placement, since a
        shorter user ladder is the cheaper one whatever its spacing.
        """
        return replace(
            self,
            lj_lambdas=(self.lj_lambdas
                        if len(self.lj_lambdas) <= len(SCREENING_LJ_LAMBDAS)
                        else SCREENING_LJ_LAMBDAS),
            coul_lambdas=(self.coul_lambdas
                          if len(self.coul_lambdas) <= len(SCREENING_COUL_LAMBDAS)
                          else SCREENING_COUL_LAMBDAS),
            production_steps=min(self.production_steps, 150_000),
            equil_steps=min(self.equil_steps, 25_000),
            n_morphologies=min(self.n_morphologies, 2),
            # The screening error bar is ~2x the production one by construction,
            # so holding it to the production precision budget would mark every
            # screening point unconverged and the loop would never advance.
            max_stderr=max(self.max_stderr, 0.60),
        )

    def validate(self) -> None:
        _check(self.n_morphologies >= 1, "fep.n_morphologies must be >= 1")
        for name, lams in (("lj_lambdas", self.lj_lambdas),
                           ("coul_lambdas", self.coul_lambdas)):
            _check(len(lams) >= 2, f"fep.{name} needs at least 2 states")
            _check(
                all(b > a for a, b in zip(lams, lams[1:])),
                f"fep.{name} must be strictly increasing (got {list(lams)})",
            )
            _check(
                lams[0] == 0.0 and lams[-1] == 1.0,
                f"fep.{name} must span exactly 0 -> 1 (got {lams[0]} -> {lams[-1]}); "
                "the endpoints are the physical states, so a truncated ladder "
                "silently computes a different free energy",
            )
        _check(self.soft_core_n >= 1, "fep.soft_core_n must be >= 1")
        _check(
            0 < self.kspace_accuracy <= 1.0e-5,
            "fep.kspace_accuracy must be in (0, 1e-5]; looser grids put a "
            "systematic error of order 0.01 kcal/mol per lambda state into the "
            "charge leg that sampling cannot remove (see FEPSpec docstring)",
        )
        _check(self.alpha_lj > 0, "fep.alpha_lj must be positive")
        _check(self.alpha_coul > 0, "fep.alpha_coul must be positive")
        _check(self.equil_steps >= 0, "fep.equil_steps must be non-negative")
        _check(self.production_steps > 0, "fep.production_steps must be positive")
        _check(self.sample_every >= 1, "fep.sample_every must be >= 1")
        _check(
            self.production_steps >= self.sample_every,
            "fep.production_steps must be >= fep.sample_every, else no frames are stored",
        )
        n_frames = self.production_steps // self.sample_every
        _check(
            n_frames >= 20,
            f"fep.production_steps / fep.sample_every gives {n_frames} frames per "
            "lambda; MBAR needs at least ~20 after decorrelation to return a "
            "meaningful uncertainty",
        )
        _check(bool(self.estimators), "fep.estimators must not be empty")
        unknown = set(self.estimators) - {"mbar", "bar", "ti"}
        _check(not unknown, f"unknown fep.estimators: {sorted(unknown)}")
        if "mbar" in self.estimators:
            _check(
                self.rerun_matrix,
                "fep.estimators includes 'mbar' but fep.rerun_matrix is false; "
                "MBAR needs the full K x N energy matrix that the rerun pass builds",
            )
        _check(0 < self.ti_delta < 0.5, "fep.ti_delta must be in (0, 0.5)")
        _check(0 <= self.min_overlap < 1, "fep.min_overlap must be in [0, 1)")
        _check(self.max_stderr > 0, "fep.max_stderr must be positive")


@dataclass(frozen=True)
class RunConfig:
    """Complete description of one water-uptake calculation."""

    polymer: PolymerSpec
    box: BoxSpec = field(default_factory=BoxSpec)
    md: MDSpec = field(default_factory=MDSpec)
    equilibration: EquilibrationSpec = field(default_factory=EquilibrationSpec)
    insertion: InsertionSpec = field(default_factory=InsertionSpec)
    widom: WidomSpec = field(default_factory=WidomSpec)
    fep: FEPSpec = field(default_factory=FEPSpec)
    water_model: str = "spce"
    workdir: str = "runs/aem"
    #: Which estimator supplies mu_ex for the saturation criterion. ``"fep"``
    #: (default) uses the alchemical path; ``"widom"`` keeps the original test-
    #: particle route. Both remain available -- Widom is retained as an
    #: independent cross-check, not as dead code -- but only the selected one
    #: decides where uptake stops.
    mu_ex_method: str = "fep"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for sub in (self.polymer, self.box, self.md, self.equilibration,
                    self.insertion, self.widom, self.fep):
            sub.validate()
        _check(
            self.mu_ex_method in SUPPORTED_MU_EX_METHODS,
            f"mu_ex_method must be one of {SUPPORTED_MU_EX_METHODS} "
            f"(got {self.mu_ex_method!r})",
        )
        if self.mu_ex_method == "widom" and not self.widom.enabled:
            raise ConfigError(
                "mu_ex_method is 'widom' but widom.enabled is false; there would "
                "be no estimator to decide saturation"
            )
        _check(
            self.equilibration.high_temperature > self.md.temperature,
            "equilibration.high_temperature must exceed md.temperature; the "
            "NVT excursions have to be above Tg to let chains relax",
        )
        _check(
            self.equilibration.max_pressure > self.md.pressure,
            "equilibration.max_pressure must exceed md.pressure",
        )
        _check(
            self.water_model in SUPPORTED_WATER_MODELS,
            f"water_model must be one of {SUPPORTED_WATER_MODELS} (got {self.water_model!r})",
        )
        # Applies to whichever estimator is in use: both evaluate mu_ex on the
        # configuration handed to them, so an unrelaxed cell biases either one.
        if self.md.relax_npt_steps == 0 and (
            self.widom.enabled or self.mu_ex_method == "fep"
        ):
            LOG.warning(
                "md.relax_npt_steps is 0: %s mu_ex will be evaluated on an "
                "unrelaxed configuration and the saturation point will be biased.",
                self.mu_ex_method.upper(),
            )

    # ------------------------------------------------------------------ I/O --
    @property
    def path(self) -> Path:
        return Path(self.workdir).expanduser()

    def to_dict(self) -> dict[str, Any]:
        from .utils import _jsonable

        return _jsonable(self)

    def dump_yaml(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return p

    def with_overrides(self, **overrides: Any) -> "RunConfig":
        """Return a copy with dotted-path overrides applied, e.g.
        ``cfg.with_overrides(**{"polymer.n_chains": 8})``."""
        data = copy.deepcopy(self.to_dict())
        for key, value in overrides.items():
            if value is None:
                continue
            node = data
            parts = key.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return RunConfig.from_dict(data)

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "RunConfig":
        data = dict(data)
        section_types = {
            "polymer": PolymerSpec,
            "box": BoxSpec,
            "md": MDSpec,
            "equilibration": EquilibrationSpec,
            "insertion": InsertionSpec,
            "widom": WidomSpec,
            "fep": FEPSpec,
        }
        kwargs: dict[str, Any] = {}
        for name, cls in section_types.items():
            raw = data.pop(name, None)
            if raw is None:
                if name == "polymer":
                    raise ConfigError("configuration must contain a 'polymer' section")
                kwargs[name] = cls()
                continue
            if not isinstance(raw, Mapping):
                raise ConfigError(f"section '{name}' must be a mapping")
            known = {f.name for f in fields(cls)}
            unknown = set(raw) - known
            if unknown:
                raise ConfigError(
                    f"unknown key(s) in section '{name}': {sorted(unknown)}; valid keys: {sorted(known)}"
                )
            # YAML has no tuple type, so a tuple-typed field round-trips as a
            # list. Coerce it back: these dataclasses are frozen (hence hashable
            # and safely shareable), and a list field would break both that and
            # equality against a freshly constructed config.
            section = dict(raw)
            for f in fields(cls):
                if f.name in section and _is_tuple_field(f.type):
                    value = section[f.name]
                    if isinstance(value, (list, tuple)):
                        section[f.name] = tuple(value)
            kwargs[name] = cls(**section)
        top_known = {"water_model", "workdir", "mu_ex_method"}
        unknown_top = set(data) - top_known
        if unknown_top:
            raise ConfigError(f"unknown top-level key(s): {sorted(unknown_top)}; valid: {sorted(top_known)}")
        return RunConfig(**kwargs, **{k: data[k] for k in top_known if k in data})

    @staticmethod
    def from_yaml(path: str | Path) -> "RunConfig":
        raw = yaml.safe_load(Path(path).expanduser().read_text())
        if not isinstance(raw, Mapping):
            raise ConfigError(f"{path} does not contain a YAML mapping")
        return RunConfig.from_dict(raw)


def default_config(smiles: str, **overrides: Any) -> RunConfig:
    """Convenience constructor used by the CLI."""
    cfg = RunConfig(polymer=PolymerSpec(smiles=smiles))
    return cfg.with_overrides(**overrides) if overrides else cfg


__all__ = [
    "PolymerSpec",
    "BoxSpec",
    "MDSpec",
    "EquilibrationSpec",
    "InsertionSpec",
    "WidomSpec",
    "RunConfig",
    "ConfigError",
    "default_config",
    "SUPPORTED_COUNTERIONS",
    "SUPPORTED_WATER_MODELS",
]
