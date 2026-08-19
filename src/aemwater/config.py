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

from .utils import LOG

SUPPORTED_COUNTERIONS = ("Cl-", "Br-", "OH-", "HCO3-")
SUPPORTED_WATER_MODELS = ("spce", "tip3p")


class ConfigError(ValueError):
    """Raised when a configuration is internally inconsistent."""


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
class RunConfig:
    """Complete description of one water-uptake calculation."""

    polymer: PolymerSpec
    box: BoxSpec = field(default_factory=BoxSpec)
    md: MDSpec = field(default_factory=MDSpec)
    equilibration: EquilibrationSpec = field(default_factory=EquilibrationSpec)
    insertion: InsertionSpec = field(default_factory=InsertionSpec)
    widom: WidomSpec = field(default_factory=WidomSpec)
    water_model: str = "spce"
    workdir: str = "runs/aem"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for sub in (self.polymer, self.box, self.md, self.equilibration,
                    self.insertion, self.widom):
            sub.validate()
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
        if self.widom.enabled and self.md.relax_npt_steps == 0:
            LOG.warning(
                "md.relax_npt_steps is 0: Widom mu_ex will be evaluated on an unrelaxed "
                "configuration and the saturation point will be biased."
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
            kwargs[name] = cls(**raw)
        top_known = {"water_model", "workdir"}
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
