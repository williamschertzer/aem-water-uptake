"""Water uptake averaged over independently equilibrated morphologies.

A single hydrated cell gives one saturation point, and there is no way to know
from that number alone whether it is a property of the polymer or of the one
packing that happened to be built. This module runs the whole uptake loop once
per morphology -- independent packing seed, independent dry equilibration,
independent hydration trajectory -- and combines the endpoints.

**Why the loop is replicated rather than the mu_ex measurement inside it.**
The obvious cheaper design is one hydration trajectory with an M-morphology FEP
campaign at each water content. It does not work: the campaign layer's
between-morphology variance requires cells that are independent *as cells*, and
M ghost insertions into one trajectory's cell share that cell's packing
entirely. They would measure sampling noise and report it as morphology spread,
which is worse than not measuring it, because the error bar would look
respectable while resting on one packing.

Replicating the loop costs M times the wall clock and buys a between-morphology
error bar on the quantity actually being reported -- the saturation point --
plus a diagnostic that no single trajectory can provide: if two morphologies
saturate at different water contents, that spread *is* the uncertainty of the
prediction, and a single-cell run would have reported the tighter of the two
with a confident-looking error bar.

Screening resolution makes this affordable. Each trajectory's per-iteration
mu_ex runs at ``FEPSpec.at_screening_resolution()`` (7+7 states, 150k steps),
which is 6.4x cheaper than production, so M=2 or 3 trajectories cost less than
one production-resolution trajectory would.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .fep.campaign import t95
from .utils import LOG


class UptakeCampaignError(RuntimeError):
    """Raised when the campaign cannot produce a defensible average."""


@dataclass
class MorphologyUptake:
    """One trajectory's endpoint, plus whether it is fit to average."""

    index: int
    seed: int
    workdir: Path
    n_waters: int
    lambda_value: float
    water_uptake_pct: float
    hydrated_density: float
    stop_reason: str
    converged: bool
    n_iterations: int
    #: None when the trajectory completed; the exception text when it did not.
    failure: str | None = None

    @property
    def usable(self) -> bool:
        """Whether this trajectory may enter the average.

        A trajectory that stopped at ``max_iterations`` never reached
        saturation, so its water content is a lower bound rather than a
        measurement, and averaging it in would bias the campaign low without
        any indication in the error bar.
        """
        if self.failure is not None:
            return False
        if not math.isfinite(self.lambda_value):
            return False
        return self.stop_reason != "max_iterations"

    def summary(self) -> dict[str, object]:
        return {
            "index": self.index,
            "seed": self.seed,
            "n_waters": self.n_waters,
            "lambda_waters_per_ionic_group": round(self.lambda_value, 3),
            "water_uptake_wt_pct": round(self.water_uptake_pct, 2),
            "hydrated_density_g_cm3": round(self.hydrated_density, 4),
            "stop_reason": self.stop_reason,
            "converged": self.converged,
            "iterations": self.n_iterations,
            "usable": self.usable,
            "failure": self.failure,
        }


@dataclass
class UptakeCampaign:
    """Uptake averaged over morphologies, with the spread as the error bar."""

    #: Mean water uptake, weight percent.
    water_uptake_pct: float
    #: Standard error of that mean, from the between-morphology scatter.
    stderr: float
    #: Mean hydration number lambda, waters per ionic group.
    lambda_value: float
    lambda_stderr: float
    per_morphology: list[MorphologyUptake]
    bulk_mu_ex: float
    workdir: Path
    diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def n_usable(self) -> int:
        return sum(1 for m in self.per_morphology if m.usable)

    @property
    def dof(self) -> int:
        return max(0, self.n_usable - 1)

    @property
    def ci95(self) -> tuple[float, float]:
        """Student-t 95% interval on the uptake, honest about small M.

        Same reasoning as the mu_ex campaign: at two or three morphologies the
        normal quantile undercovers badly, because the scatter itself is
        estimated from one or two degrees of freedom.
        """
        if self.dof == 0:
            return (float("-inf"), float("inf"))
        half = t95(self.dof) * self.stderr
        return (self.water_uptake_pct - half, self.water_uptake_pct + half)

    def summary(self) -> dict[str, object]:
        lo, hi = self.ci95
        return {
            "water_uptake_wt_pct": round(self.water_uptake_pct, 2),
            "water_uptake_stderr": round(self.stderr, 3),
            "water_uptake_ci95": [
                None if not math.isfinite(lo) else round(lo, 2),
                None if not math.isfinite(hi) else round(hi, 2),
            ],
            "lambda_waters_per_ionic_group": round(self.lambda_value, 3),
            "lambda_stderr": round(self.lambda_stderr, 3),
            "bulk_mu_ex_kcal_mol": round(self.bulk_mu_ex, 3),
            "n_morphologies_usable": self.n_usable,
            "n_morphologies_run": len(self.per_morphology),
            "per_morphology": [m.summary() for m in self.per_morphology],
            "diagnostics": self.diagnostics,
        }


def morphology_box_seed(base_seed: int, index: int) -> int:
    """A distinct packing seed per morphology.

    Delegates to :func:`aemwater.fep.campaign.morphology_seed` rather than
    reimplementing it. A first version of this function did reimplement it, and
    reintroduced exactly the affine-stride defect that docstring warns about
    (consecutive seeds differing by a constant 40503) -- which matters more here
    than there, since this seed drives the chain packing that *defines* the
    morphology.
    """
    from .fep.campaign import morphology_seed

    return morphology_seed(base_seed, index)


def run_uptake_campaign(
    config,
    workdir: Path | str,
    n_morphologies: int | None = None,
    bulk_reference=None,
    resume: bool = True,
    screening: bool = True,
) -> UptakeCampaign:
    """Run the uptake loop once per morphology and average the endpoints.

    ``n_morphologies`` defaults to ``config.fep.n_morphologies``, so the same
    knob that sets replication for a mu_ex campaign sets it here.

    The bulk reference is computed once and shared. It is a property of the
    water model and the state point, not of the membrane packing, so computing
    it per morphology would spend M times the CPU on M estimates of the same
    number -- and worse, would let the M trajectories be judged against
    slightly different reference values, putting reference noise into the
    between-morphology spread that is supposed to measure packing.

    ``screening=True`` runs each trajectory's per-iteration mu_ex at
    :meth:`FEPSpec.at_screening_resolution`. One trajectory at production
    resolution costs more than three at screening resolution, and the
    saturation point is decided by a curve crossing rather than by the third
    decimal of any single mu_ex.

    Each morphology gets its own subdirectory and its own checkpoint, so a
    campaign interrupted after two of three trajectories resumes into the third
    rather than restarting.
    """
    from .driver import obtain_bulk_reference, run_uptake
    from .prepare import obtain_dry_membrane

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    m_count = n_morphologies if n_morphologies is not None else config.fep.n_morphologies
    if m_count < 1:
        raise UptakeCampaignError(
            f"n_morphologies must be >= 1, got {m_count}"
        )
    if m_count == 1:
        LOG.warning(
            "uptake campaign with one morphology: the result will carry no "
            "uncertainty estimate, because a single packing measures no spread. "
            "Use this for smoke tests, not for a reported number."
        )

    loop_config = config
    if screening:
        scr = config.fep.at_screening_resolution()
        # Dotted overrides rather than dataclasses.replace on the section: this
        # is the path that round-trips through RunConfig.from_dict and so
        # re-runs validation, which is what should reject an inconsistent
        # screening preset rather than letting it reach LAMMPS.
        loop_config = config.with_overrides(**{
            "fep.lj_lambdas": list(scr.lj_lambdas),
            "fep.coul_lambdas": list(scr.coul_lambdas),
            "fep.production_steps": scr.production_steps,
            "fep.equil_steps": scr.equil_steps,
            "fep.n_morphologies": scr.n_morphologies,
            "fep.max_stderr": scr.max_stderr,
        })
        LOG.info(
            "uptake campaign: %d morphologies at screening resolution "
            "(%d+%d lambda states, %d steps/state)",
            m_count, len(loop_config.fep.lj_lambdas),
            len(loop_config.fep.coul_lambdas), loop_config.fep.production_steps,
        )

    # Computed once, up front, and passed to every trajectory. Deliberately not
    # "computed by the first trajectory and reused": UptakeResult carries only
    # the reference's mu_ex as a float, not the BulkReference object, and
    # run_uptake needs the object (it calls .sanity() and checks .settings).
    # Reconstructing a stand-in from the float would pass the type check and
    # fail at the first attribute access.
    shared_reference = bulk_reference
    if shared_reference is None:
        shared_reference = obtain_bulk_reference(loop_config, workdir / "bulk",
                                                 resume=resume)
        for issue in shared_reference.sanity():
            LOG.warning("bulk reference: %s", issue)

    results: list[MorphologyUptake] = []

    for index in range(m_count):
        seed = morphology_box_seed(config.box.seed, index)
        mdir = workdir / f"morph{index:02d}"
        mdir.mkdir(parents=True, exist_ok=True)
        # Independent packing: the box seed is what decides where the chains go,
        # so this is the line that makes the morphologies genuinely different
        # rather than differently-perturbed copies of one packing.
        mconfig = loop_config.with_overrides(**{"box.seed": seed})

        LOG.info("uptake campaign: morphology %d/%d (box seed %d) in %s",
                 index + 1, m_count, seed, mdir)
        try:
            # Reuse an already-equilibrated cell rather than rebuilding it. The
            # anneal and GAFF2 charge derivation dominate per-morphology cost,
            # so an unconditional rebuild here would make a requeued campaign on
            # a preemptible queue unable to make forward progress.
            typed_chains, reused = obtain_dry_membrane(
                mconfig, mdir, resume=resume)
            if reused:
                LOG.info("morphology %d: reused its dry membrane", index)
            # Named "uptake", not "result": driver.py binds "result" to an
            # InsertionResult, and the attribute guard in tests/test_cli.py
            # holds one type per variable name across the audited modules.
            uptake = run_uptake(
                mconfig, mdir, typed_chains,
                bulk_reference=shared_reference, resume=resume,
            )
            results.append(MorphologyUptake(
                index=index, seed=seed, workdir=mdir,
                n_waters=uptake.n_waters,
                lambda_value=uptake.lambda_value,
                water_uptake_pct=uptake.water_uptake_pct,
                hydrated_density=uptake.hydrated_density,
                stop_reason=uptake.stop_reason,
                converged=uptake.converged,
                n_iterations=len(uptake.iterations),
            ))
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            # One packing that fails to equilibrate or crashes LAMMPS must not
            # discard the trajectories that succeeded. The failure is recorded
            # per morphology, excluded from the average, and reported.
            LOG.error("uptake campaign: morphology %d failed: %s", index, exc)
            results.append(MorphologyUptake(
                index=index, seed=seed, workdir=mdir, n_waters=0,
                lambda_value=float("nan"), water_uptake_pct=float("nan"),
                hydrated_density=float("nan"), stop_reason="failed",
                converged=False, n_iterations=0, failure=f"{type(exc).__name__}: {exc}",
            ))

    # BulkReference.mu_ex is an estimate object, not a float; the scalar lives
    # one level in. Same convention as UptakeResult.bulk_mu_ex.
    campaign = combine_uptake(
        results,
        bulk_mu_ex=(float(shared_reference.mu_ex.mu_ex)
                    if shared_reference is not None else float("nan")),
        workdir=workdir,
    )
    report = workdir / "uptake_campaign.json"
    report.write_text(json.dumps(campaign.summary(), indent=2, default=float))
    LOG.info("uptake campaign: %.2f +/- %.2f wt%% over %d morphologies -> %s",
             campaign.water_uptake_pct, campaign.stderr, campaign.n_usable, report)
    return campaign


def combine_uptake(
    morphologies: Sequence[MorphologyUptake],
    bulk_mu_ex: float,
    workdir: Path | str = "",
) -> UptakeCampaign:
    """Average the per-morphology saturation points with equal weights.

    Equal weights, and the error bar from the between-morphology scatter --
    the same two choices as :func:`aemwater.fep.campaign.combine_morphologies`,
    for the same reasons. There is no within-morphology error to propagate here
    anyway: a trajectory's saturation point is a single number, not a mean over
    samples, so the scatter across trajectories is the only uncertainty
    estimate available.
    """
    import numpy as np

    usable = [m for m in morphologies if m.usable]
    if not usable:
        reasons = "; ".join(
            f"morph {m.index}: {m.failure or m.stop_reason}" for m in morphologies
        )
        raise UptakeCampaignError(
            f"no usable morphology among {len(morphologies)}. Every trajectory "
            f"either failed or never saturated ({reasons}). A trajectory that "
            "stopped at max_iterations gives a lower bound, not an uptake."
        )
    if len(usable) < len(morphologies):
        LOG.warning(
            "uptake campaign: %d of %d morphologies unusable (failed or never "
            "saturated); averaging over %d",
            len(morphologies) - len(usable), len(morphologies), len(usable),
        )

    pct = np.array([m.water_uptake_pct for m in usable], dtype=float)
    lam = np.array([m.lambda_value for m in usable], dtype=float)
    m_count = pct.size

    if m_count == 1:
        return UptakeCampaign(
            water_uptake_pct=float(pct[0]), stderr=float("nan"),
            lambda_value=float(lam[0]), lambda_stderr=float("nan"),
            per_morphology=list(morphologies), bulk_mu_ex=bulk_mu_ex,
            workdir=Path(workdir),
            diagnostics={
                "note": "one usable morphology: no spread measured, so the "
                        "uptake carries no uncertainty estimate at all. This is "
                        "a single sample, not a prediction with an error bar.",
                "n_unusable": len(morphologies) - 1,
            },
        )

    return UptakeCampaign(
        water_uptake_pct=float(pct.mean()),
        stderr=float(pct.std(ddof=1) / math.sqrt(m_count)),
        lambda_value=float(lam.mean()),
        lambda_stderr=float(lam.std(ddof=1) / math.sqrt(m_count)),
        per_morphology=list(morphologies), bulk_mu_ex=bulk_mu_ex,
        workdir=Path(workdir),
        diagnostics={
            "water_uptake_per_morphology": [float(x) for x in pct],
            "lambda_per_morphology": [float(x) for x in lam],
            "spread_wt_pct": float(pct.max() - pct.min()),
            "n_unusable": len(morphologies) - m_count,
            "stop_reasons": sorted({m.stop_reason for m in usable}),
        },
    )
