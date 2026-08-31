"""Orchestrate an FEP campaign over independently equilibrated morphologies.

One *campaign* is: for each of ``fep.n_morphologies`` independently equilibrated
cells, run both legs (soft-core LJ, then electrostatics) at every lambda, build
the rerun matrix, estimate the leg free energies, and sum them. The per-morphology
results are then combined into a single excess chemical potential.

The combination is the statistically load-bearing part of this module, and it is
where the two mistakes that matter live:

**Equal weights, not inverse-variance weights.** Each morphology is a *draw from
the ensemble of morphologies*, so the quantity wanted is the ensemble mean.
Inverse-variance weighting would tilt the answer toward whichever cell happened
to get a tight within-morphology error bar, which says nothing about how
representative that cell is. In a glassy matrix a single well-sampled but
atypical pocket structure can carry a very small ``s_m`` while sitting far from
the ensemble mean; weighting by ``1/s_m^2`` would pull the answer toward it.
Equal weighting is correct for the ensemble mean and is what is used.

**The error bar comes from the scatter *between* morphologies**, not from
propagating the within-morphology errors. ``stderr = sqrt(var(mu_m, ddof=1)/M)``
already contains both contributions, because each ``mu_m`` carries its own
sampling noise. Propagating only the within-morphology errors
(``sqrt(sum(s_m^2))/M``) would omit the morphology-to-morphology spread, which in
a glassy matrix is usually the *larger* of the two -- that is the error that makes
a converged-looking calculation confidently wrong.

The variance decomposition is reported because it says what to do next:
``v_between >> v_within`` means run more morphologies, and no amount of extra
sampling per cell will help; ``v_within >> v_between`` means sample each cell
longer, and more cells would be wasted effort.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from ..utils import LOG
from .estimators import LegEstimate
from .schedule import FEPLeg

#: Two-sided 95% Student-t quantiles by degrees of freedom (M-1).
#:
#: With three morphologies the standard error is itself estimated from two
#: degrees of freedom, so a 1.96-sigma interval understates the uncertainty
#: badly: t(0.975, 2) = 4.303, not 1.96. Tabulated rather than pulled from
#: scipy.stats so this module has no scipy import for four numbers.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131,
        20: 2.086, 30: 2.042}


def t95(dof: int) -> float:
    """Two-sided 95% Student-t quantile, interpolated conservatively."""
    if dof <= 0:
        return float("inf")
    if dof in _T95:
        return _T95[dof]
    if dof > 30:
        return 1.96
    # Between tabulated points take the larger (wider interval) of the
    # neighbours rather than interpolating: erring wide is the safe direction.
    below = max(k for k in _T95 if k < dof)
    return _T95[below]


class CampaignError(RuntimeError):
    """Raised when a campaign cannot produce a trustworthy estimate."""


@dataclass
class MorphologyEstimate:
    """One morphology's excess chemical potential, summed over both legs."""

    index: int
    mu_ex: float                          # kcal/mol
    stderr: float                         # kcal/mol, within-morphology only
    legs: dict[str, LegEstimate] = field(default_factory=dict)
    workdir: str = ""
    diagnostics: dict = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """Whether every leg produced a finite estimate with an error bar."""
        if not math.isfinite(self.mu_ex) or not math.isfinite(self.stderr):
            return False
        return all(
            math.isfinite(e.delta_f) and math.isfinite(e.stderr)
            for e in self.legs.values()
        )

    def summary(self) -> dict[str, object]:
        return {
            "morphology": self.index,
            "mu_ex_kcal_mol": round(self.mu_ex, 4),
            "stderr_kcal_mol": round(self.stderr, 4),
            "usable": self.usable,
            **{f"{name}_kcal_mol": round(e.delta_f, 4)
               for name, e in sorted(self.legs.items())},
        }


@dataclass
class FEPEstimate:
    """Campaign result: mu_ex averaged over morphologies.

    Deliberately field-compatible with :class:`aemwater.widom.WidomEstimate` on
    ``mu_ex``, ``stderr``, ``temperature``, ``converged`` and ``summary()`` so
    the driver can consume either estimator without branching.
    """

    mu_ex: float                          # kcal/mol
    stderr: float                         # kcal/mol, total
    temperature: float
    n_morphologies: int
    per_morphology: list[MorphologyEstimate] = field(default_factory=list)
    #: Method-of-moments variance decomposition, (kcal/mol)^2.
    var_between: float = 0.0
    var_within: float = 0.0
    #: True when only one morphology was run, so var_between is *unmeasured*
    #: rather than measured to be zero.
    between_unmeasured: bool = False
    #: True when the moment estimator returned a negative between-variance and
    #: it was clamped to zero. Expected when morphologies genuinely agree.
    between_clamped: bool = False
    max_stderr: float = 0.30
    diagnostics: dict = field(default_factory=dict)

    @property
    def dof(self) -> int:
        return max(0, self.n_morphologies - 1)

    @property
    def ci95(self) -> tuple[float, float]:
        """Student-t 95% interval, honest about small M."""
        if self.dof == 0:
            return (float("-inf"), float("inf"))
        half = t95(self.dof) * self.stderr
        return (self.mu_ex - half, self.mu_ex + half)

    @property
    def converged(self) -> bool:
        """Whether the estimate is precise enough and rests on real replication.

        Tested against the 95% interval half-width, not the bare standard error.
        The two differ by the t-quantile, which is 12.7 at two morphologies and
        4.3 at three -- so comparing ``stderr`` against the budget would call a
        result converged while its interval was several times wider than the
        budget it claimed to meet. The bulk SPC/E validation did exactly that:
        stderr 0.030 against a 0.30 budget reported ``converged``, while the
        interval spanned +/-0.385. The half-width is what a reader means by "we
        know mu_ex to 0.3 kcal/mol", so that is what is checked.

        A single morphology can never satisfy this: its between-morphology
        variance is unmeasured, so the reported error bar is a lower bound of
        unknown tightness. That is a fine smoke test and not a result.
        """
        if self.between_unmeasured:
            return False
        if not all(m.usable for m in self.per_morphology):
            return False
        lo, hi = self.ci95
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return False
        return (hi - lo) / 2.0 <= self.max_stderr

    @property
    def limiting_factor(self) -> str:
        """What to spend the next CPU-hour on."""
        if self.between_unmeasured:
            return "replication: only one morphology, between-cell spread unknown"
        if self.var_between > 2.0 * self.var_within:
            return "morphologies: between-cell spread dominates, add cells"
        if self.var_within > 2.0 * self.var_between:
            return "sampling: within-cell noise dominates, sample longer"
        return "balanced: both contributions comparable"

    def summary(self) -> dict[str, object]:
        lo, hi = self.ci95
        return {
            "mu_ex_kcal_mol": round(self.mu_ex, 4),
            "stderr_kcal_mol": round(self.stderr, 4),
            "ci95_kcal_mol": [round(lo, 4), round(hi, 4)],
            "n_morphologies": self.n_morphologies,
            "var_between": round(self.var_between, 6),
            "var_within": round(self.var_within, 6),
            "between_unmeasured": self.between_unmeasured,
            "converged": self.converged,
            "limiting_factor": self.limiting_factor,
            "temperature_K": self.temperature,
        }


def combine_morphologies(
    estimates: Sequence[MorphologyEstimate],
    temperature: float,
    max_stderr: float = 0.30,
) -> FEPEstimate:
    """Average per-morphology mu_ex with equal weights and decompose the variance.

    See the module docstring for why the weights are equal and why the error bar
    comes from the between-morphology scatter rather than from propagating the
    within-morphology errors.
    """
    usable = [m for m in estimates if m.usable]
    if not usable:
        raise CampaignError(
            f"no usable morphology among {len(estimates)}: every one produced a "
            "non-finite free energy or error bar. Check the per-state logs and "
            "the BAR overlap diagnostics before rerunning."
        )
    if len(usable) < len(estimates):
        LOG.warning(
            "fep campaign: dropping %d of %d morphologies with non-finite "
            "estimates; averaging over %d",
            len(estimates) - len(usable), len(estimates), len(usable),
        )

    mus = np.array([m.mu_ex for m in usable], dtype=float)
    within = np.array([m.stderr for m in usable], dtype=float)
    m_count = mus.size

    mu_bar = float(mus.mean())
    v_within = float(np.mean(within ** 2))

    if m_count == 1:
        # One draw measures no spread. Report the within-morphology error and
        # say plainly that the between-morphology term is missing, rather than
        # letting a zero pass for a measurement.
        return FEPEstimate(
            mu_ex=mu_bar, stderr=float(within[0]), temperature=temperature,
            n_morphologies=1, per_morphology=list(usable),
            var_between=0.0, var_within=v_within,
            between_unmeasured=True, max_stderr=max_stderr,
            diagnostics={"note": "single morphology: total uncertainty is a "
                                 "lower bound of unknown tightness"},
        )

    v_obs = float(mus.var(ddof=1))
    # Method of moments: the observed scatter contains both the real
    # morphology-to-morphology variance and the within-morphology sampling noise.
    v_between_raw = v_obs - v_within
    clamped = v_between_raw < 0.0
    v_between = max(0.0, v_between_raw)

    # Standard error of the equally-weighted mean. v_obs estimates
    # sigma_between^2 + sigma_within^2, so it is already the total -- adding
    # v_within again would double-count it.
    #
    # A floor at v_within was tried here and removed. The motivating observation
    # looked damning: on the bulk SPC/E validation two morphologies landed 0.06
    # kcal/mol apart while each carried a 0.5-0.6 kcal/mol internal error, so
    # sqrt(v_obs/M) came out 13x smaller than the sampling noise measured inside
    # those same cells, which reads as luck being reported as precision. But
    # Monte Carlo against a known truth in exactly that regime (sigma_between
    # 0.05, sigma_within 0.55) showed the unfloored interval covering at 0.949,
    # 0.943 and 0.944 for M = 2, 3, 5 -- nominal -- while the floored version
    # covered at 1.000, i.e. the floor produces intervals that are too wide and
    # a "precision" that is unearned in the other direction.
    #
    # The reason is that the t-quantile already does this job: a v_obs that comes
    # out small by luck is paired with dof = M - 1 and hence a very wide
    # multiplier (12.7 at M = 2), and the two effects cancel to give correct
    # coverage. That cancellation is the entire point of the t distribution, and
    # flooring the variance breaks it. What the small v_obs does mean is that the
    # *interval*, not the standard error, is the quantity to compare against a
    # precision budget -- see FEPEstimate.converged.
    stderr = math.sqrt(v_obs / m_count)

    return FEPEstimate(
        mu_ex=mu_bar, stderr=stderr, temperature=temperature,
        n_morphologies=m_count, per_morphology=list(usable),
        var_between=v_between, var_within=v_within,
        between_clamped=clamped, max_stderr=max_stderr,
        diagnostics={
            "var_observed": v_obs,
            "var_between_raw": v_between_raw,
            "mu_ex_per_morphology": [float(x) for x in mus],
            "n_dropped": len(estimates) - m_count,
        },
    )


def morphology_seed(base_seed: int, index: int) -> int:
    """A distinct, reproducible seed per morphology.

    Independence between morphologies is the entire point of running several, so
    the seeds must not collide and must not be sequentially correlated in a way
    the placement RNG could pick up.

    A single multiply-add (``base*A + index*B``) is *not* good enough here even
    though it looks like hashing: it is affine in ``index``, so consecutive
    morphologies get seeds differing by exactly ``B``. A first version of this
    function did that and produced 24740026, 24780529, 24821032, ... -- a
    constant stride of 40503. Some RNGs seeded from nearby integers produce
    correlated early output, which would quietly undermine the independence the
    whole campaign is built on. SplitMix64 finalisation avalanches the bits
    instead, so a one-step change in ``index`` changes about half the seed bits.
    """
    # SplitMix64 finalisation of (base_seed, index), then truncated to LAMMPS's
    # positive 32-bit seed range.
    x = (base_seed * 0x9E3779B97F4A7C15 + index * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 31
    return int(x & 0x7FFFFFFF) or 1


def combine_legs_for_morphology(
    index: int,
    legs: dict[str, LegEstimate],
    workdir: Path | str = "",
) -> MorphologyEstimate:
    """Sum leg free energies into one mu_ex for a single morphology.

    Both ladders run lambda = 0 -> 1 in the direction of *growth*, so the legs
    sum directly to the free energy of introducing the water -- which is mu_ex,
    with no sign flip. For SPC/E water in bulk SPC/E the LJ leg is positive
    (order +2 kcal/mol, the cost of opening a cavity) and the charge leg is
    strongly negative (order -10 kcal/mol, electrostatic solvation), summing to
    roughly -8 kcal/mol. A result near +8 means a sign error, not a hydrophobic
    water.
    """
    total = float(sum(e.delta_f for e in legs.values()))
    stderr = float(np.sqrt(sum(e.stderr ** 2 for e in legs.values())))
    return MorphologyEstimate(
        index=index,
        mu_ex=total,
        stderr=stderr,
        legs=dict(legs),
        workdir=str(workdir),
        diagnostics={
            "per_leg": {k: {"delta_f": v.delta_f, "stderr": v.stderr,
                            "estimator": v.estimator}
                        for k, v in legs.items()},
        },
    )


def run_leg(
    leg: FEPLeg,
    *,
    ladder,
    system,
    ghost,
    config,
    workdir: Path,
    groups,
    constraints,
    comm_cutoff: float,
    seed: int,
    lammps_args: Sequence[str] = (),
    ranks: int = 1,
) -> dict:
    """Sample every state of one leg, then build the matrix and estimate it.

    Returns ``{"estimates": {name: LegEstimate}, "state_dirs": [...],
    "matrix": EnergyMatrix | None}``. Sampling runs are sequential: each state is
    an independent fixed-lambda MD run, so they parallelise trivially, but LAMMPS
    already uses the available cores through ``ranks`` and oversubscribing makes
    every state slower.

    ``system`` is the fully-coupled topology. There is deliberately no
    ``data_file`` parameter: each state's topology is written here with that
    state's ghost charges, because leg 2 is applied through the data file.
    """
    from ..lammps.runner import run_lammps
    from ..lammps.writer import write_data_file
    from .ghost import scale_ghost_charges
    from .inputs import render_state_input
    from .rerun import build_energy_matrix
    from .estimators import bar_estimate, mbar_estimate, ti_from_state_dirs

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    states = ladder.states
    lambdas = tuple(s.lam for s in states)

    state_dirs: list[Path] = []
    state_systems: list = []
    for state in states:
        sdir = workdir / f"lam_{state.index:02d}"
        sdir.mkdir(parents=True, exist_ok=True)

        # Each state gets its own data file carrying that state's ghost charges.
        # This is not tidiness: leg 2 is applied *through the data file* (see
        # scale_ghost_charges -- PPPM builds its grid at setup), so a single
        # shared topology would sample every charge state with the same charges.
        # Passing one shared file here silently sampled the whole Coulomb leg at
        # lambda_q = 0; the rerun diagonal check caught it at 1.5 kcal/mol.
        state_system = (
            scale_ghost_charges(system, ghost, state.lambda_q)
            if state.lambda_q != 0.0 else system
        )
        state_data = sdir / "state.data"
        write_data_file(state_system, state_data, include_pair_coeffs=False)
        state_systems.append(state_system)

        render_state_input(
            state, directory=sdir, system=state_system, ghost=ghost,
            ladder_lambdas=lambdas, config=config, groups=groups,
            constraints=constraints, comm_cutoff=comm_cutoff,
            data_file=str(state_data.resolve()), seed=seed + state.index,
        )
        LOG.info("fep %s leg: sampling lambda=%.3f (%d/%d)",
                 leg.value, state.lam, state.index + 1, len(states))
        run_lammps(sdir / "in.fep", ranks=ranks, log_name="state.log",
                   extra_args=lammps_args)
        state_dirs.append(sdir)

    estimates: dict[str, LegEstimate] = {}
    matrix = None
    wanted = tuple(config.fep.estimators)

    if config.fep.rerun_matrix and ({"mbar", "bar"} & set(wanted)):
        # data_file left to default: each rerun reads its own state's topology
        # from state_dirs[j]/state.data, which run_leg just wrote.
        matrix = build_energy_matrix(
            ladder, state_dirs=state_dirs, systems=state_systems,
            ghost=ghost, config=config, workdir=workdir / "rerun",
            lammps_args=lammps_args,
        )
        if "mbar" in wanted:
            estimates["mbar"] = mbar_estimate(matrix)
        if "bar" in wanted:
            estimates["bar"] = bar_estimate(matrix)
    if "ti" in wanted:
        estimates["ti"] = ti_from_state_dirs(
            lambdas,
            [d / "fep.dat" for d in state_dirs],
            delta=config.fep.ti_delta,
            leg=leg,
        )

    if not estimates:
        raise CampaignError(
            f"{leg.value} leg produced no estimate: fep.estimators="
            f"{wanted!r} with rerun_matrix={config.fep.rerun_matrix}. "
            "MBAR and BAR both need the rerun matrix."
        )
    return {"estimates": estimates, "state_dirs": state_dirs, "matrix": matrix}


def select_reported(
    estimates: dict[str, LegEstimate], degeneracy_factor: float = 5.0
) -> LegEstimate:
    """Pick the estimate to report from one leg's estimators.

    MBAR is preferred on principle -- it uses every sample against every state
    and is the minimum-variance estimator *when its covariance is well
    conditioned*. That last clause is not decoration. On a three-state LJ ladder
    with 20 frames per state this code reported MBAR at +6.79 +/- 73.02 kcal/mol
    against BAR at +8.15 +/- 0.78: with almost no overlap between neighbours the
    MBAR covariance matrix is near-singular and its uncertainty explodes, so
    unconditional preference would hand back a number with an error bar twenty
    times the quantity being measured.

    So MBAR wins unless its error bar is non-finite or more than
    ``degeneracy_factor`` times the best alternative's, in which case the
    tighter estimator is reported and the reason is recorded in
    ``diagnostics["selection"]``. This is a symptom of an under-resolved ladder
    either way -- :func:`estimator_disagreement` will say so -- but reporting
    the usable number beats reporting the principled one.
    """
    order = [n for n in ("mbar", "bar", "ti") if n in estimates]
    if not order:
        raise CampaignError(f"no estimator among {sorted(estimates)}")

    finite = [n for n in order if math.isfinite(estimates[n].stderr)]
    if not finite:
        LOG.warning(
            "fep: no estimator produced a finite uncertainty; reporting %s",
            order[0],
        )
        return estimates[order[0]]

    preferred = next((n for n in order if n in finite), finite[0])
    best = min(finite, key=lambda n: estimates[n].stderr)
    chosen = preferred
    if (preferred != best
            and estimates[preferred].stderr
            > degeneracy_factor * estimates[best].stderr):
        LOG.warning(
            "fep: %s uncertainty %.3f exceeds %s's %.3f by more than %.0fx "
            "(near-degenerate covariance, likely poor state overlap); "
            "reporting %s",
            preferred, estimates[preferred].stderr, best,
            estimates[best].stderr, degeneracy_factor, best,
        )
        chosen = best

    est = estimates[chosen]

    # Carry forward the diagnostics only the *unreported* estimators measured.
    # The reported estimate is the only one kept past this point, and the two
    # estimators measure complementary things: MBAR records per-pair overlap,
    # TI records the per-state dU/dlambda mean and fluctuation. Dropping the
    # loser's keys would discard measurements that cost the whole leg to make
    # and cannot be recovered without re-running it. Existing keys always win,
    # so the reported estimator's own numbers are never overwritten.
    carried: dict = {}
    for name in ("ti", "mbar", "bar"):
        other = estimates.get(name)
        if other is None or name == chosen:
            continue
        for key in ("lambdas", "dudl_mean", "dudl_sd", "neighbour_overlap"):
            if key in other.diagnostics and key not in est.diagnostics:
                carried[key] = other.diagnostics[key]

    extra: dict = dict(carried)
    if chosen != preferred:
        extra["selection"] = (
            f"reported instead of {preferred}: "
            f"{preferred} stderr {estimates[preferred].stderr:.3f} "
            f"vs {est.stderr:.3f}"
        )
    if extra:
        est = replace(est, diagnostics={**est.diagnostics, **extra})
    return est


def estimator_disagreement(
    estimates: dict[str, LegEstimate], n_sigma: float = 3.0
) -> list[str]:
    """Flag estimator pairs disagreeing by more than their combined error.

    This is the diagnostic the three-estimator design exists to provide, and each
    pair means something specific:

    * MBAR vs BAR -- poor overlap between neighbouring states.
    * TI vs either -- the ladder is too coarse to integrate.

    Returned as warnings rather than raised: a campaign that has already spent
    the CPU time should hand back its numbers plus the caveat, not nothing.
    """
    warnings: list[str] = []
    names = [n for n in ("mbar", "bar", "ti") if n in estimates]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ea, eb = estimates[a], estimates[b]
            if not (math.isfinite(ea.stderr) and math.isfinite(eb.stderr)):
                continue
            combined = math.sqrt(ea.stderr ** 2 + eb.stderr ** 2)
            gap = abs(ea.delta_f - eb.delta_f)
            if combined > 0 and gap > n_sigma * combined:
                meaning = ("poor overlap between neighbouring states"
                           if {a, b} == {"mbar", "bar"}
                           else "ladder too coarse to integrate")
                warnings.append(
                    f"{a} and {b} differ by {gap:.3f} kcal/mol "
                    f"({gap / combined:.1f} sigma): {meaning}"
                )
    return warnings


#: Bumped when a change to this module would alter a cached FEP number.
FEP_ESTIMATOR_VERSION = 1


def fep_cache_key(bulk_settings, fep_spec) -> str:
    """Cache key for an FEP bulk reference.

    Deliberately *not* ``BulkSettings.key()``. That key hashes the bulk settings
    plus ``WIDOM_ESTIMATOR_VERSION`` and is written to ``bulk_<key>.json``, so an
    FEP reference sharing it would be served a Widom number for an FEP request
    (and vice versa) whenever the thermodynamic settings matched -- which they
    normally do, since both measure the same state point. The key therefore
    includes the alchemical protocol: changing a lambda ladder, the sampling
    length, or the morphology count changes the answer, so it must change the
    key.
    """
    import hashlib
    from dataclasses import asdict

    payload = {
        "bulk": asdict(bulk_settings),
        "fep": {
            "lj_lambdas": list(fep_spec.lj_lambdas),
            "coul_lambdas": list(fep_spec.coul_lambdas),
            "equil_steps": fep_spec.equil_steps,
            "production_steps": fep_spec.production_steps,
            "sample_every": fep_spec.sample_every,
            "n_morphologies": fep_spec.n_morphologies,
            "estimators": sorted(fep_spec.estimators),
            "soft_core_n": fep_spec.soft_core_n,
            "alpha_lj": fep_spec.alpha_lj,
            "alpha_coul": fep_spec.alpha_coul,
            "ti_delta": fep_spec.ti_delta,
        },
        "fep_estimator_version": FEP_ESTIMATOR_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def run_bulk_campaign(
    config,
    workdir: Path | str,
    n_waters: int,
    ranks: int = 1,
    lammps_args: Sequence[str] = (),
) -> FEPEstimate:
    """Measure mu_ex of bulk water by FEP over independent morphologies.

    The bulk analogue of a polymer campaign, and the calculation that validates
    the whole chain: SPC/E has a published mu_ex near -6.5 kcal/mol, so a result
    that misses it by more than a few tenths indicts the protocol rather than the
    sample. Each "morphology" here is an independently seeded and equilibrated
    water box.
    """
    from ..assembly import CellContents, assemble, water_molecules
    from ..bulk import build_bulk_coordinates
    from ..forcefield.water import water_model as get_water_model
    from ..lammps.inputs import GroupSpec, comm_cutoff, constraint_spec
    from ..lammps.writer import write_data_file
    from .ghost import add_ghost_water
    from .schedule import LambdaLadder

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    spec = config.fep
    model = get_water_model(config.water_model)

    morphologies: list[MorphologyEstimate] = []
    for index in range(spec.n_morphologies):
        seed = morphology_seed(spec.seed, index)
        mdir = workdir / f"morph{index:02d}"
        mdir.mkdir(parents=True, exist_ok=True)

        coords, edge = build_bulk_coordinates(n_waters, model, seed=seed)
        contents = CellContents(
            chains=[], ions=[],
            waters=water_molecules(n_waters, config.water_model),
        )
        system, ghost = add_ghost_water(
            assemble(contents, coords, edge=edge), model=model, seed=seed,
        )
        o_type, h_type = system.water_atom_types()
        shared = dict(
            groups=GroupSpec(n_polymer_molecules=0, n_ion_molecules=0,
                             water_type_o=o_type, water_type_h=h_type),
            constraints=constraint_spec(config.md, system.water_bond_type(),
                                        system.water_angle_type()),
            comm_cutoff=comm_cutoff(config.md),
        )

        legs: dict[str, LegEstimate] = {}
        for leg, lambdas in ((FEPLeg.LJ, spec.lj_lambdas),
                             (FEPLeg.COUL, spec.coul_lambdas)):
            result = run_leg(
                leg, ladder=LambdaLadder(leg=leg, lambdas=lambdas),
                system=system, ghost=ghost, config=config,
                workdir=mdir / leg.value, seed=seed, ranks=ranks,
                lammps_args=lammps_args, **shared,
            )
            legs[leg.value] = select_reported(result["estimates"])
            for warning in estimator_disagreement(result["estimates"]):
                LOG.warning("bulk morphology %d, %s leg: %s",
                            index, leg.value, warning)

        estimate = combine_legs_for_morphology(index, legs, workdir=mdir)
        LOG.info("bulk morphology %d: mu_ex = %.3f +/- %.3f kcal/mol",
                 index, estimate.mu_ex, estimate.stderr)
        morphologies.append(estimate)

    return combine_morphologies(
        morphologies, config.md.temperature, max_stderr=spec.max_stderr,
    )


def run_membrane_campaign(
    config,
    workdir: Path | str,
    systems: Sequence,
    ranks: int = 1,
    lammps_args: Sequence[str] = (),
) -> FEPEstimate:
    """Measure mu_ex of water inside the membrane by FEP over morphologies.

    ``systems`` are already-equilibrated hydrated membrane cells, one per
    morphology -- typically the relaxed cells the driver carries in memory at a
    given water content. Unlike :func:`run_bulk_campaign`, this function does
    *not* build or equilibrate anything: a polymer morphology takes far longer
    to equilibrate than the FEP itself, so the cells are the caller's
    responsibility and their independence is the caller's guarantee.

    ``spec.n_morphologies`` is therefore a *check* here rather than a loop
    bound. Silently averaging over fewer cells than configured would report an
    error bar whose between-morphology term rests on less replication than the
    config claims, which is the specific failure this framework exists to
    avoid.
    """
    from ..lammps.inputs import GroupSpec, comm_cutoff, constraint_spec
    from .ghost import add_ghost_water
    from .schedule import LambdaLadder

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    spec = config.fep

    if not systems:
        raise CampaignError(
            "run_membrane_campaign got no cells. The caller must supply at "
            "least one equilibrated morphology; this function does not build "
            "them."
        )
    if len(systems) < spec.n_morphologies:
        raise CampaignError(
            f"fep.n_morphologies is {spec.n_morphologies} but only "
            f"{len(systems)} equilibrated cell(s) were supplied. Averaging "
            "over fewer cells than configured would report a between-morphology "
            "error bar that rests on less replication than the config claims. "
            "Either equilibrate more morphologies or lower fep.n_morphologies "
            "so the number in the config is the number that ran."
        )
    if len(systems) > spec.n_morphologies:
        LOG.info("membrane campaign: %d cells supplied, using the first %d "
                 "per fep.n_morphologies", len(systems), spec.n_morphologies)
    systems = list(systems)[:spec.n_morphologies]

    from ..forcefield.water import water_model as get_water_model
    model = get_water_model(config.water_model)

    morphologies: list[MorphologyEstimate] = []
    for index, cell in enumerate(systems):
        seed = morphology_seed(spec.seed, index)
        mdir = workdir / f"morph{index:02d}"
        mdir.mkdir(parents=True, exist_ok=True)

        # The ghost is added per morphology rather than once, because it takes
        # new atom types and those depend on the cell's own type table.
        system, ghost = add_ghost_water(cell, model=model, seed=seed)
        o_type, h_type = system.water_atom_types()
        # Counted off the *input* cell, before the ghost is added.
        # ``n_polymer_molecules`` counts residues that are neither water nor
        # ion, and the ghost has its own residue name (GHO), so counting the
        # ghosted system reports one phantom chain and would put the ghost in
        # the polymer group. Verified: a 30-water box reports 0 chains before
        # add_ghost_water and 1 after.
        #
        # fep_state.in.j2 does not currently reference the group block at all,
        # so this is latent rather than active -- which is exactly why it is
        # worth getting right here instead of discovering it the first time a
        # group-dependent fix is added to that template.
        n_poly = cell.n_polymer_molecules()
        n_ion = cell.n_ion_molecules()
        shared = dict(
            groups=GroupSpec(n_polymer_molecules=n_poly, n_ion_molecules=n_ion,
                             water_type_o=o_type, water_type_h=h_type),
            constraints=constraint_spec(config.md, system.water_bond_type(),
                                        system.water_angle_type()),
            comm_cutoff=comm_cutoff(config.md),
        )

        legs: dict[str, LegEstimate] = {}
        for leg, lambdas in ((FEPLeg.LJ, spec.lj_lambdas),
                             (FEPLeg.COUL, spec.coul_lambdas)):
            result = run_leg(
                leg, ladder=LambdaLadder(leg=leg, lambdas=lambdas),
                system=system, ghost=ghost, config=config,
                workdir=mdir / leg.value, seed=seed, ranks=ranks,
                lammps_args=lammps_args, **shared,
            )
            legs[leg.value] = select_reported(result["estimates"])
            for warning in estimator_disagreement(result["estimates"]):
                LOG.warning("membrane morphology %d, %s leg: %s",
                            index, leg.value, warning)

        estimate = combine_legs_for_morphology(index, legs, workdir=mdir)
        LOG.info("membrane morphology %d: mu_ex = %.3f +/- %.3f kcal/mol",
                 index, estimate.mu_ex, estimate.stderr)
        morphologies.append(estimate)

    return combine_morphologies(
        morphologies, config.md.temperature, max_stderr=spec.max_stderr,
    )


def write_campaign_report(estimate: FEPEstimate, path: Path | str) -> Path:
    """Serialise a campaign result to JSON, per-morphology rows included."""
    path = Path(path)
    payload = {
        "combined": estimate.summary(),
        "per_morphology": [m.summary() for m in estimate.per_morphology],
        "diagnostics": estimate.diagnostics,
    }
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path


__all__ = [
    "CampaignError",
    "FEPEstimate",
    "MorphologyEstimate",
    "combine_legs_for_morphology",
    "combine_morphologies",
    "estimator_disagreement",
    "morphology_seed",
    "run_leg",
    "run_membrane_campaign",
    "select_reported",
    "t95",
    "write_campaign_report",
]
