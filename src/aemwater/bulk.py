"""Bulk liquid water reference: the reservoir the membrane equilibrates against.

Why this run exists
-------------------
The saturation criterion compares mu_ex(water in membrane) with mu_ex(water in
bulk). Both must be computed the *same way* -- same water model, same cutoffs,
same long-range treatment, same Widom protocol -- because each of those choices
shifts mu_ex by more than the difference being resolved. A literature value for
SPC/E (about -6.5 kcal/mol at 298 K) is a sanity check, not a substitute: it was
computed with someone else's cutoff and tail correction.

The reference is cached on disk keyed by the settings that affect it, since it is
identical for every membrane composition at the same temperature and model, and
re-running it per uptake iteration would dominate the cost.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .utils import LOG
from .widom import WidomEstimate, read_widom_file

#: Reference values from the literature, used only to flag a suspicious run.
#: Excess chemical potential of the pure liquid at ~298 K, kcal/mol.
LITERATURE_MU_EX = {"spce": -6.5, "tip3p": -6.1, "tip4p": -6.3}
LITERATURE_DENSITY = {"spce": 0.998, "tip3p": 0.982, "tip4p": 1.001}

# Increment when the sampling/estimation protocol changes.  Results produced
# before burst-level output was preserved are not statistically compatible with
# the corrected blocking estimator and must not be reused from the cache.
WIDOM_ESTIMATOR_VERSION = 2


@dataclass(frozen=True)
class BulkSettings:
    """Everything that changes the answer. The cache key is a hash of this."""

    water_model: str
    temperature: float
    pressure: float
    n_waters: int
    cutoff: float
    kspace_accuracy: float
    equil_steps: int
    widom_steps: int
    insertions_per_call: int
    seed: int

    def key(self) -> str:
        payload = {
            "settings": asdict(self),
            "widom_estimator_version": WIDOM_ESTIMATOR_VERSION,
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class BulkReference:
    """A completed bulk-water reference calculation."""

    settings: BulkSettings
    mu_ex: WidomEstimate
    density: float
    volume: float
    workdir: Path

    def sanity(self) -> list[str]:
        """Warnings about a reference that disagrees with known behaviour.

        Returned rather than raised: an unusual water model or state point can
        legitimately differ, and the caller decides whether to proceed.
        """
        issues = []
        model = self.settings.water_model.lower()
        ref_mu = LITERATURE_MU_EX.get(model)
        if ref_mu is not None and abs(self.mu_ex.mu_ex - ref_mu) > 1.5:
            issues.append(
                f"mu_ex = {self.mu_ex.mu_ex:.2f} kcal/mol differs from the "
                f"published {model} value ({ref_mu:.1f}) by more than 1.5 "
                f"kcal/mol. This is expected at affordable insertion counts: "
                f"direct Widom sampling underestimates the magnitude until the "
                f"tail of the cavity distribution is sampled. The saturation "
                f"criterion uses the difference against an equally "
                f"under-converged membrane estimate, so it does not invalidate "
                f"the run -- but this number is not a measurement of mu_ex."
            )
        ref_rho = LITERATURE_DENSITY.get(model)
        if ref_rho is not None and abs(self.density - ref_rho) > 0.05:
            issues.append(
                f"density = {self.density:.3f} g/cm3 differs from the published "
                f"{model} value ({ref_rho:.3f}) by more than 0.05"
            )
        if not self.mu_ex.converged:
            issues.append(
                f"Widom average is carried by only {self.mu_ex.effective_samples:.1f} "
                "effective samples"
            )
        return issues

    def summary(self) -> dict[str, object]:
        return {
            "water_model": self.settings.water_model,
            "temperature_K": self.settings.temperature,
            "n_waters": self.settings.n_waters,
            "density_g_cm3": round(self.density, 4),
            **self.mu_ex.summary(),
            "sanity_warnings": self.sanity(),
        }


#: Closest O-O separation permitted on the starting lattice (A). Liquid water
#: peaks at 2.8 A; 2.4 A is strained but recoverable by minimisation, whereas a
#: 1.5 A contact is on the repulsive wall and blows the first dynamics step.
MIN_OO = 2.4

#: Molar mass of water, g/mol.
M_WATER = 18.01528
AVOGADRO = 6.02214076e23


def water_box_edge(n_waters: int, density: float = 0.997) -> float:
    """Cubic edge (A) holding ``n_waters`` at ``density`` g/cm3."""
    volume_cm3 = n_waters * M_WATER / (AVOGADRO * density)
    return float((volume_cm3 * 1.0e24) ** (1.0 / 3.0))


def build_bulk_coordinates(
    n_waters: int,
    water_model,
    density: float = 0.997,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Waters on a jittered lattice at the target density.

    A lattice rather than random placement: at liquid density random insertion
    of rigid molecules stalls, whereas a lattice is overlap-free by construction
    and melts within a few picoseconds. The jitter breaks the crystalline
    symmetry so the melt does not persist as an artificial ice.
    """
    from .insertion import water_orientations

    edge = water_box_edge(n_waters, density)
    n_side = int(np.ceil(n_waters ** (1 / 3)))
    spacing = edge / n_side
    rng = np.random.default_rng(seed)

    sites = np.stack(
        np.meshgrid(*[(np.arange(n_side) + 0.5) * spacing] * 3, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)[:n_waters]

    # Jitter has to break the lattice symmetry without creating contacts the
    # minimiser cannot recover from. An unbounded Gaussian displacement does
    # exactly that: two neighbours can each move half a spacing toward each
    # other. The displacement is therefore bounded so that no O-O pair can fall
    # below MIN_OO even in the worst case.
    max_shift = max(0.0, (spacing - MIN_OO) / 2.0)
    shift = rng.normal(scale=0.35 * max_shift, size=sites.shape)
    shift = np.clip(shift, -max_shift / np.sqrt(3), max_shift / np.sqrt(3))
    sites = sites + shift

    geometries = water_orientations(n_waters, water_model, rng)
    coords = (geometries + sites[:, None, :]).reshape(-1, 3)
    return np.mod(coords, edge), edge


def run_bulk_reference(
    settings: BulkSettings,
    workdir: Path | str,
    cache_dir: Path | str | None = None,
    ranks: int = 1,
    lammps_binary: str = "lmp",
) -> BulkReference:
    """Equilibrate bulk water and measure its excess chemical potential.

    Cached on ``settings.key()``: the reference does not depend on the membrane,
    so recomputing it for every uptake iteration would be the dominant cost of a
    workflow whose answer it does not change.
    """
    from .assembly import assemble, CellContents, water_molecules
    from .lammps.inputs import (
        ConstraintSpec,
        GroupSpec,
        comm_cutoff,
        pair_coeff_lines,
        render_input,
        write_water_molecule_template,
    )
    from .lammps.runner import run_lammps
    from .lammps.writer import write_data_file
    from .forcefield.water import water_model as get_water_model

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir) if cache_dir else workdir.parent / "bulk_cache"
    cache.mkdir(parents=True, exist_ok=True)
    cache_file = cache / f"bulk_{settings.key()}.json"

    if cache_file.exists():
        LOG.info("bulk reference: reusing cached result %s", cache_file.name)
        payload = json.loads(cache_file.read_text())
        est = WidomEstimate(
            mu_ex=payload["mu_ex"],
            stderr=payload["stderr"],
            temperature=settings.temperature,
            n_blocks=payload["n_blocks"],
            block_values=np.array(payload["block_values"]),
            mean_boltzmann=payload["mean_boltzmann"],
            effective_samples=payload["effective_samples"],
            volume=payload["volume"],
        )
        return BulkReference(settings, est, payload["density"], payload["volume"],
                             workdir)

    model = get_water_model(settings.water_model)
    coords, edge = build_bulk_coordinates(settings.n_waters, model, seed=settings.seed)
    contents = CellContents(chains=[], ions=[],
                            waters=water_molecules(settings.n_waters,
                                                   settings.water_model))
    system = assemble(contents, coords, edge=edge)
    write_data_file(system, workdir / "bulk.data")
    o_type, h_type = system.water_atom_types()
    write_water_molecule_template(
        workdir / "h2o.mol", model, o_type, h_type,
        system.water_bond_type(), system.water_angle_type())

    LOG.info(
        "bulk reference: %d %s waters in a %.2f A box (%s)",
        settings.n_waters,
        settings.water_model,
        edge,
        settings.key(),
    )

    common = dict(
        title=f"Bulk {settings.water_model} reference",
        data_file="bulk.data",
        pair_coeff_lines=pair_coeff_lines(system),
        extra_types=None,
        constraints=ConstraintSpec(shake_water=True, shake_hydrogen=False,
                                   water_bond_type=system.water_bond_type(),
                                   water_angle_type=system.water_angle_type()),
        groups=GroupSpec(n_polymer_molecules=0, n_ion_molecules=0,
                         water_type_o=o_type, water_type_h=h_type),
        comm_cutoff=settings.cutoff + 2.0,
        seed=settings.seed,
    )
    return _run_bulk_stages(settings, workdir, cache_file, system, common, ranks)


def _run_bulk_stages(settings, workdir, cache_file, system, common, ranks):
    """Render, run and reduce the bulk reference; cache the result."""
    from .lammps.inputs import render_input
    from .lammps.runner import run_lammps
    from .config import MDSpec, WidomSpec

    md = MDSpec(
        temperature=settings.temperature,
        pressure=settings.pressure,
        cutoff=settings.cutoff,
        kspace_accuracy=settings.kspace_accuracy,
    )
    widom = WidomSpec(insertions_per_call=settings.insertions_per_call)
    n_averages = max(1, settings.equil_steps // (md.thermo_every * 10))
    # Widom samples are accumulated over the whole production run and reduced to
    # one row per averaging window. Computed here rather than in the template so
    # the actual number appears in the run directory, not just its recipe.
    n_widom_samples = max(1, settings.widom_steps // (widom.every * 20))

    render_input(
        "bulk.in.j2",
        workdir / "in.bulk",
        md=md,
        widom=widom,
        equil_steps=settings.equil_steps,
        widom_steps=settings.widom_steps,
        n_averages=n_averages,
        n_widom_samples=n_widom_samples,
        widom_window=widom.every * n_widom_samples,
        density_file="bulk_density.dat",
        mu_file="bulk_mu.dat",
        water_template="h2o.mol",
        out_data="bulk_final.data",
        **common,
    )
    run = run_lammps(workdir / "in.bulk", ranks=ranks, log_name="bulk.log")

    density_rows = [
        [float(x) for x in line.split()]
        for line in (workdir / "bulk_density.dat").read_text().splitlines()
        if line.strip() and not line.startswith("#") and len(line.split()) >= 3
    ]
    if not density_rows:
        raise RuntimeError(f"bulk run wrote no density data; see {run.log_file}")
    arr = np.array(density_rows)
    # Second half only: the first half contains the lattice melting.
    half = arr[len(arr) // 2 :]
    density = float(half[:, 1].mean())
    volume = float(half[:, 2].mean())

    est = read_widom_file(workdir / "bulk_mu.dat", settings.temperature)
    ref = BulkReference(settings, est, density, volume, workdir)

    cache_file.write_text(
        json.dumps(
            {
                "mu_ex": est.mu_ex,
                "stderr": est.stderr,
                "n_blocks": est.n_blocks,
                "block_values": est.block_values.tolist(),
                "mean_boltzmann": est.mean_boltzmann,
                "effective_samples": est.effective_samples,
                "volume": volume,
                "density": density,
                "settings": asdict(settings),
            },
            indent=2,
        )
    )
    for issue in ref.sanity():
        LOG.warning("bulk reference: %s", issue)
    LOG.info(
        "bulk reference: mu_ex = %.3f +/- %.3f kcal/mol, rho = %.4f g/cm3",
        est.mu_ex,
        est.stderr,
        density,
    )
    return ref


__all__ = [
    "BulkSettings",
    "BulkReference",
    "run_bulk_reference",
    "build_bulk_coordinates",
    "water_box_edge",
    "LITERATURE_MU_EX",
    "M_WATER",
]
