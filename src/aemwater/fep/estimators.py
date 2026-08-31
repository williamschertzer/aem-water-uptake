"""MBAR, BAR and TI on the same samples.

All three estimators consume one sampling campaign. That is deliberate: they are
not alternatives to choose between but three views of the same data, and their
*disagreement* is the diagnostic. MBAR uses every sample against every state and
is the reported value; BAR uses only neighbouring pairs and needs no rerun
matrix; TI integrates <dU/dlambda> over the ladder and is sensitive to ladder
resolution in a way the other two are not.

When they agree the ladder is adequate. When TI disagrees with MBAR the ladder is
too coarse where the integrand curves -- a statement about the schedule, not the
sampling. When BAR disagrees with MBAR on a *neighbour* pair, that pair's overlap
is marginal.

Correlation is handled once, up front. Fixed-lambda MD produces correlated
frames; feeding them to MBAR as independent samples gives a free energy that is
fine and an uncertainty that is optimistic by roughly sqrt(g). Every estimator
here therefore subsamples to the statistical inefficiency computed from the
state's own dU trace before doing anything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, NamedTuple, Sequence

import numpy as np

from ..utils import LOG
from .rerun import EnergyMatrix
from .schedule import FEPLeg

#: kcal/mol per kT is temperature-dependent; this is the gas constant in
#: kcal/mol/K, matching the rest of the workflow.
R_KCAL = 0.0019872041

#: exp(-w) underflows to exactly zero in float64 above this, taking BAR's
#: variance estimate with it. np.log(np.finfo(float).max) is 709.78.
_EXP_LIMIT = 709.0


@dataclass(frozen=True)
class LegEstimate:
    """One leg's free energy from one estimator.

    ``delta_f`` is dG for the leg in kcal/mol, signed as
    ``G(lambda=1) - G(lambda=0)``: the cost of *introducing* the interaction.
    """

    estimator: str
    leg: FEPLeg
    delta_f: float
    stderr: float
    n_effective: float
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.estimator} {self.leg.value}: "
            f"{self.delta_f:+.3f} +/- {self.stderr:.3f} kcal/mol"
        )


def _subsample(series: np.ndarray) -> tuple[np.ndarray, float]:
    """Indices of effectively independent frames, and the inefficiency g.

    Falls back to keeping every frame when the trace is too short or too flat
    for ``statistical_inefficiency`` to estimate g -- with a warning, because an
    unmeasurable g means the reported uncertainty is a lower bound.
    """
    from pymbar import timeseries

    if series.size < 3:
        return np.arange(series.size), 1.0
    try:
        g = float(timeseries.statistical_inefficiency(series))
    except Exception as exc:  # pragma: no cover - pymbar raises several types
        LOG.warning(
            "could not estimate statistical inefficiency (%s); treating %d frames "
            "as independent, so the reported uncertainty is a lower bound",
            exc, series.size,
        )
        return np.arange(series.size), 1.0
    if not np.isfinite(g) or g < 1.0:
        g = 1.0
    idx = timeseries.subsample_correlated_data(series, g=g)
    return np.asarray(idx, dtype=int), g


def mbar_estimate(matrix: EnergyMatrix, *, subsample: bool = True) -> LegEstimate:
    """MBAR over the full K x N matrix; the reported estimator.

    Uses every sample from every state to estimate all K free energies at once,
    which is why it beats BAR when more than two states are involved: a sample
    from state 3 informs the 0->1 difference too.
    """
    from pymbar import MBAR

    u_kn, N_k = matrix.u_kn, matrix.N_k.copy()
    if subsample:
        keep, gs = [], []
        start = 0
        for k, n in enumerate(matrix.N_k):
            block = slice(start, start + n)
            # Decorrelate on this state's own energy under its own Hamiltonian,
            # which is the series whose autocorrelation actually limits it.
            idx, g = _subsample(u_kn[k, block])
            keep.append(idx + start)
            gs.append(g)
            start += n
        cols = np.concatenate(keep)
        N_k = np.array([len(i) for i in keep], dtype=int)
        u_kn = u_kn[:, cols]
    else:
        gs = [1.0] * len(N_k)

    mbar = MBAR(u_kn, N_k)
    res = mbar.compute_free_energy_differences()
    df = float(res["Delta_f"][0, -1]) * matrix.kT
    ddf = float(res["dDelta_f"][0, -1]) * matrix.kT
    overlap = mbar.compute_overlap()["matrix"]
    neighbours = [float(overlap[i, i + 1]) for i in range(matrix.n_states - 1)]

    return LegEstimate(
        estimator="mbar",
        leg=matrix.leg,
        delta_f=df,
        stderr=ddf,
        n_effective=float(N_k.sum()),
        diagnostics={
            # The ladder is recorded because `neighbour_overlap` is otherwise
            # unattributable: a value below threshold says a pair is too thin,
            # but not *which* pair, and that is the actionable part.
            "lambdas": [float(x) for x in matrix.lambdas],
            "N_k": N_k.tolist(),
            "statistical_inefficiency": [round(g, 2) for g in gs],
            "neighbour_overlap": [round(o, 4) for o in neighbours],
            "min_overlap": min(neighbours) if neighbours else float("nan"),
            "per_state_delta_f": (
                np.asarray(res["Delta_f"][0]) * matrix.kT
            ).tolist(),
        },
    )


def bar_estimate(
    matrix: EnergyMatrix, *, subsample: bool = True
) -> LegEstimate:
    """BAR on each neighbouring pair, summed along the ladder.

    Independent of MBAR's self-consistent solve, and computable from the rerun
    matrix or from the inline sampling perturbations alone.

    Caveat on the uncertainty: the per-pair errors are combined in quadrature,
    which assumes they are independent. They are not -- state k's samples enter
    both the (k-1, k) and the (k, k+1) pair -- so the quadrature sum *under*
    counts. On the harmonic reference ladder BAR reports 0.0104 against MBAR's
    0.0117 on identical data, despite using strictly less information. Treat
    BAR's error bar as a lower bound and MBAR's as the reported one; BAR's value
    here is as an independent check on the central estimate, not on its
    precision.
    """
    from pymbar import bar as bar_fn

    total = 0.0
    var = 0.0
    per_pair = []
    offsets = np.concatenate([[0], np.cumsum(matrix.N_k)])

    for i in range(matrix.n_states - 1):
        j = i + 1
        fwd_block = slice(offsets[i], offsets[i + 1])
        rev_block = slice(offsets[j], offsets[j + 1])
        # w_F: work to go i->j measured on samples from i; w_R: j->i on samples
        # from j. Both are already reduced (kT units) in the matrix.
        w_f = matrix.u_kn[j, fwd_block] - matrix.u_kn[i, fwd_block]
        w_r = matrix.u_kn[i, rev_block] - matrix.u_kn[j, rev_block]
        if subsample:
            w_f = w_f[_subsample(w_f)[0]]
            w_r = w_r[_subsample(w_r)[0]]
        res = bar_fn(w_f, w_r)
        d = float(res["Delta_f"]) * matrix.kT
        dd = float(res["dDelta_f"]) * matrix.kT

        # A single frame with |w| above ~709 kT makes exp(-w) underflow to
        # exactly zero in double precision, and BAR's variance formula then
        # divides by it and returns nan. That is not a numerical accident to
        # paper over: it means one sampled configuration is astronomically
        # improbable in the neighbouring state, i.e. that pair has no usable
        # overlap. Left alone the nan propagates silently through the quadrature
        # sum into the reported total, so it is surfaced here instead.
        n_extreme = int((np.abs(w_f) > _EXP_LIMIT).sum() + (np.abs(w_r) > _EXP_LIMIT).sum())
        if not np.isfinite(dd):
            LOG.warning(
                "BAR uncertainty is not finite for pair %d->%d "
                "(lambda %.3g -> %.3g): %d of %d work values exceed %.0f kT, "
                "where exp(-w) underflows. This pair needs intermediate states, "
                "not more sampling.",
                i, j, matrix.lambdas[i], matrix.lambdas[j],
                n_extreme, w_f.size + w_r.size, _EXP_LIMIT,
            )
        total += d
        var += dd * dd
        per_pair.append(
            {
                "pair": (i, j),
                "lambdas": (matrix.lambdas[i], matrix.lambdas[j]),
                "delta_f": round(d, 4),
                "stderr": dd if not np.isfinite(dd) else round(dd, 4),
                "n_forward": int(w_f.size),
                "n_reverse": int(w_r.size),
                "n_extreme_work": n_extreme,
                "max_abs_work": round(float(max(np.abs(w_f).max(), np.abs(w_r).max())), 2),
            }
        )

    bad = [p["pair"] for p in per_pair if not np.isfinite(p["stderr"])]
    return LegEstimate(
        estimator="bar",
        leg=matrix.leg,
        delta_f=total,
        stderr=float(np.sqrt(var)),
        n_effective=float(sum(p["n_forward"] + p["n_reverse"] for p in per_pair)),
        diagnostics={
            "per_pair": per_pair,
            # Explicit rather than inferred from a nan stderr, so the orchestrator
            # can gate on it without re-deriving why.
            "pairs_without_uncertainty": bad,
            "usable": not bad,
        },
    )


def ti_estimate(
    lambdas: Sequence[float],
    dudl_means: Sequence[float],
    dudl_stderrs: Sequence[float],
    *,
    leg: FEPLeg,
) -> LegEstimate:
    """Trapezoid integration of <dU/dlambda> over the ladder.

    The ladder is deliberately *not* uniformly spaced -- it is clustered where the
    soft-core derivative peaks -- so the quadrature must be spacing-aware.
    Integrating 3x^2 over the default LJ ladder: ``np.trapezoid`` with explicit
    lambdas gives 1.0046 against an exact 1.0, while a naive mean of the
    integrand gives 0.9631, an error eight times larger. Simpson's rule is not
    applicable at all on a non-uniform grid.

    TI's error is dominated by ladder resolution rather than sampling, which is
    exactly why it is kept: a TI-vs-MBAR gap indicts the schedule, and no amount
    of extra sampling closes it.
    """
    lam = np.asarray(lambdas, dtype=float)
    mean = np.asarray(dudl_means, dtype=float)
    err = np.asarray(dudl_stderrs, dtype=float)
    if lam.shape != mean.shape or lam.shape != err.shape:
        raise ValueError(
            f"lambdas ({lam.shape}), means ({mean.shape}) and stderrs "
            f"({err.shape}) must have matching shapes"
        )
    if lam.size < 2:
        raise ValueError("TI needs at least two lambda points")

    delta_f = float(np.trapezoid(mean, lam))
    # Trapezoid weights: half-intervals at the ends, full at the interior. Errors
    # are assumed independent between states, which they are -- separate runs.
    w = np.zeros_like(lam)
    w[0] = (lam[1] - lam[0]) / 2.0
    w[-1] = (lam[-1] - lam[-2]) / 2.0
    w[1:-1] = (lam[2:] - lam[:-2]) / 2.0
    stderr = float(np.sqrt(np.sum((w * err) ** 2)))

    # Curvature is what trapezoid gets wrong; report it so a coarse ladder is
    # visible rather than inferred from a TI-MBAR gap.
    curvature = (
        float(np.abs(np.diff(mean, 2)).max()) if lam.size > 2 else float("nan")
    )
    return LegEstimate(
        estimator="ti",
        leg=leg,
        delta_f=delta_f,
        stderr=stderr,
        n_effective=float(lam.size),
        diagnostics={
            "lambdas": lam.tolist(),
            "dudl_mean": np.round(mean, 4).tolist(),
            "dudl_stderr": np.round(err, 4).tolist(),
            "max_abs_second_difference": curvature,
            "uniform_spacing": bool(
                np.allclose(np.diff(lam), np.diff(lam)[0])
            ),
        },
    )


def read_fep_columns(path) -> dict[str, np.ndarray]:
    """Named dU columns from one state's ``fep.dat``, frame 0 dropped.

    The header's third line names the columns, so the file is self-describing and
    the estimators never have to assume a column order that
    ``perturbations_for`` might change.
    """
    from pathlib import Path

    lines = Path(path).read_text().splitlines()
    header = [l for l in lines if l.startswith("#")]
    if len(header) < 2:
        raise ValueError(f"{path}: expected a commented column header")
    # LAMMPS writes `v_dU_fwd` when it falls back to its own default header (see
    # the title2/title3 note in fep_state.in.j2); strip the prefix so a header
    # regression degrades to a working read rather than a missing-column crash.
    names = [n[2:] if n.startswith("v_") else n for n in header[-1].lstrip("# ").split()]
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[0] < 2:
        raise ValueError(
            f"{path}: only {data.shape[0]} frames; frame 0 is dropped, so at "
            "least 2 are needed"
        )
    # Frame 0 is the equilibration endpoint, evaluated at run setup rather than
    # during dynamics, and is not an independent sample. See docs/fep_design.md.
    data = data[1:]
    if data.shape[1] != len(names):
        raise ValueError(
            f"{path}: header names {len(names)} columns but the file has "
            f"{data.shape[1]}"
        )
    return {n: data[:, i] for i, n in enumerate(names)}


class DudlPoint(NamedTuple):
    """One state's dU/dlambda: its mean, the error on that mean, and its sd.

    Named rather than a bare tuple because ``stderr`` and ``sd`` are easy to
    swap at a call site and the consequences differ: ``stderr`` propagates into
    the TI error bar, ``sd`` decides where states belong on the ladder.
    """

    mean: float
    stderr: float
    sd: float


def dudl_from_finite_differences(
    plus: np.ndarray | None, minus: np.ndarray | None, delta: float
) -> DudlPoint:
    """dU/dlambda from a state's TI columns.

    ``plus`` and ``minus`` are the per-frame dU to lambda +/- delta. A central
    difference is used when both are present: it cancels the leading error term
    that a one-sided difference leaves at O(delta).

    At lambda = 0 and lambda = 1 one probe would fall outside the path, so
    ``perturbations_for`` emits only the inward one and the corresponding
    argument is ``None`` here. The one-sided difference that results is less
    accurate, which matters most at lambda = 0 where the soft-core derivative
    curves hardest -- a further reason TI is a cross-check rather than the
    reported value.
    """
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta}")
    if plus is not None and minus is not None:
        if plus.shape != minus.shape:
            raise ValueError(
                f"plus ({plus.shape}) and minus ({minus.shape}) must have the "
                "same shape"
            )
        series = (plus - minus) / (2.0 * delta)
    elif plus is not None:
        series = plus / delta
    elif minus is not None:
        # dU to lambda - delta, so the forward derivative is its negation.
        series = -minus / delta
    else:
        raise ValueError("need at least one of plus/minus to form a derivative")
    idx, _ = _subsample(series)
    kept = series[idx]
    sd = float(kept.std(ddof=1))
    # sd is returned as well as the standard error because it is a different
    # diagnostic, not a scaled copy: neighbouring-state overlap is governed by
    # the *fluctuation* of dU/dlambda, which is what places the ladder (see
    # fep.schedule), while the stderr only says how well this state's mean is
    # known. Reconstructing one from the other needs the retained sample count,
    # which is dropped here, so a caller that wants the fluctuation cannot
    # recover it downstream.
    return DudlPoint(float(kept.mean()), sd / np.sqrt(kept.size), sd)


def ti_from_state_dirs(
    lambdas: Sequence[float],
    fep_files: Sequence[object],
    *,
    delta: float,
    leg: FEPLeg,
) -> LegEstimate:
    """TI over a ladder, reading each state's finite-difference columns."""
    means, errs, sds = [], [], []
    for path in fep_files:
        cols = read_fep_columns(path)
        point = dudl_from_finite_differences(
            cols.get("dU_ti_plus"), cols.get("dU_ti_minus"), delta
        )
        means.append(point.mean)
        errs.append(point.stderr)
        sds.append(point.sd)
    est = ti_estimate(lambdas, means, errs, leg=leg)
    # The per-state fluctuation is carried forward for the diagnostic figures
    # and for any future ladder re-placement: it is measured here and nowhere
    # else, and re-running the states to recover it would cost the whole leg.
    return replace(est, diagnostics={**est.diagnostics,
                                     "dudl_sd": [round(s, 4) for s in sds]})


@dataclass(frozen=True)
class LegResult:
    """Every estimator's view of one leg, plus whether they agree."""

    leg: FEPLeg
    estimates: tuple[LegEstimate, ...]
    reported: LegEstimate

    def by_name(self, name: str) -> LegEstimate | None:
        for e in self.estimates:
            if e.estimator == name:
                return e
        return None

    @property
    def spread(self) -> float:
        """Largest pairwise disagreement between estimators, kcal/mol."""
        vals = [e.delta_f for e in self.estimates]
        return float(max(vals) - min(vals)) if len(vals) > 1 else 0.0


def combine_legs(legs: Sequence[LegResult]) -> tuple[float, float]:
    """Total dG and its uncertainty across the legs.

    The legs are sequential stages of one thermodynamic path, so their free
    energies add and their errors add in quadrature.

    Sign convention: both ladders run lambda = 0 -> 1 in the direction of
    *growth* (see ``DEFAULT_LJ_LAMBDAS``), i.e. from a fully decoupled ghost to a
    fully interacting water. The sum is therefore the free energy of introducing
    the water, which *is* the excess chemical potential -- no negation. An
    earlier version of this docstring claimed mu_ex was the negation of this sum,
    which would have reported bulk SPC/E water at +8 kcal/mol instead of -8: the
    sign that says water does not condense.
    """
    total = sum(l.reported.delta_f for l in legs)
    err = float(np.sqrt(sum(l.reported.stderr ** 2 for l in legs)))
    return float(total), err


__all__ = [
    "LegEstimate",
    "LegResult",
    "R_KCAL",
    "bar_estimate",
    "combine_legs",
    "dudl_from_finite_differences",
    "mbar_estimate",
    "read_fep_columns",
    "ti_estimate",
    "ti_from_state_dirs",
]
