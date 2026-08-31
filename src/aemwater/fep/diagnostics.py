"""Diagnostic figures for a FEP campaign.

These are the plots that decide whether a mu_ex is trustworthy, so each one is
tied to a specific way the calculation can be wrong rather than being a general
illustration:

``dudl``
    <dU/dlambda> with its per-state fluctuation. A ladder is adequate when the
    curve is resolved everywhere it is steep; a spike between two widely spaced
    states is the signature of a ladder too coarse to integrate.
``overlap``
    Per-neighbour-pair phase-space overlap. BAR and MBAR are only defined when
    neighbouring states share configurations; a pair with near-zero overlap
    makes the free energy across it unidentifiable no matter how long the run.
``morphologies``
    Per-morphology mu_ex against the pooled estimate. This is the only plot that
    shows whether the reported error bar is dominated by sampling noise inside
    each cell or by real structural heterogeneity between cells -- which is what
    decides whether to run longer or to run more cells.

Matplotlib is imported inside the functions. It is an optional dependency for a
package whose main job is running LAMMPS on a cluster, and importing it at
module scope would make ``import aemwater.fep`` fail on a headless node that
never intends to plot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ..utils import LOG

__all__ = [
    "plot_dudl",
    "plot_overlap",
    "plot_morphologies",
    "write_campaign_figures",
]

# Bound once so the same physical quantity keeps its colour across every figure
# in the set: a reader should not have to re-read a legend to know which leg is
# which.
_LEG_COLOUR = {"lj": "#1f4e79", "coul": "#b45f06"}
_LEG_LABEL = {"lj": "Lennard-Jones", "coul": "electrostatic"}


def _style():
    """Apply the shared look, tolerating a bare matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    })
    return plt


def plot_dudl(
    curves: dict[str, tuple[Sequence[float], Sequence[float], Sequence[float]]],
    path: Path | str,
):
    """<dU/dlambda> per leg, with the fluctuation as a band.

    ``curves`` maps leg name -> (lambdas, mean, sd), all in kT. The band is the
    per-state standard deviation of dU/dlambda, not a standard error: the point
    of showing it is that the *fluctuation* governs neighbouring-state overlap,
    so a region where the mean is flat but the band is wide still needs states.
    """
    plt = _style()

    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 2.5), squeeze=False)
    axes = axes[0]

    for ax, (leg, (lam, mean, sd)) in zip(axes, curves.items()):
        lam = np.asarray(lam, float)
        mean = np.asarray(mean, float)
        sd = np.asarray(sd, float)
        colour = _LEG_COLOUR.get(leg, "#333333")

        ax.axhline(0.0, color="#999999", lw=0.6, zorder=1)
        ax.fill_between(lam, mean - sd, mean + sd, color=colour, alpha=0.18,
                        lw=0, zorder=2, label="+/- 1 sd per state")
        ax.plot(lam, mean, color=colour, lw=1.4, marker="o", ms=3.5,
                zorder=3, label="<dU/dlambda>")

        # The integral is the leg's free energy, so state it: it is the number
        # the rest of the pipeline consumes, and seeing it beside the curve is
        # how a reader checks the curve against the reported result.
        area = float(np.trapezoid(mean, lam)) if hasattr(np, "trapezoid") \
            else float(np.trapz(mean, lam))
        ax.set_title(f"{_LEG_LABEL.get(leg, leg)} leg: integral = {area:+.2f} kT")
        ax.set_xlabel("coupling parameter lambda")
        ax.set_xlim(-0.03, 1.03)
        ax.margins(y=0.12)
        ax.legend(frameon=False, loc="best")

    axes[0].set_ylabel("dU/dlambda  (kT)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def plot_overlap(
    overlaps: dict[str, tuple[Sequence[float], Sequence[float]]],
    path: Path | str,
    *,
    min_overlap: float = 0.03,
):
    """Neighbour-pair overlap per leg, against the acceptance threshold.

    A pair below ``min_overlap`` shares almost no configurations, so the free
    energy across it is not identifiable from the samples -- more sampling does
    not fix it, only an extra state does. Drawn as a lollipop rather than bars
    because these are single values per pair, not distributions.
    """
    plt = _style()

    n = len(overlaps)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 2.5), squeeze=False)
    axes = axes[0]

    for ax, (leg, (midpoints, values)) in zip(axes, overlaps.items()):
        x = np.arange(len(values), dtype=float)
        values = np.asarray(values, float)
        colour = _LEG_COLOUR.get(leg, "#333333")
        # One alarm colour, reserved for failures and used for nothing else.
        bad = values < min_overlap

        ax.axhline(min_overlap, color="#999999", lw=0.8, ls="--", zorder=1)
        ax.vlines(x, 0.0, values, color=colour, lw=1.0, alpha=0.55, zorder=2)
        ax.scatter(x[~bad], values[~bad], s=22, color=colour, zorder=3,
                   label="pair overlap")
        if bad.any():
            ax.scatter(x[bad], values[bad], s=34, color="#c00000",
                       marker="D", zorder=4,
                       label=f"below {min_overlap:g} (add a state)")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{m:.2f}" for m in midpoints], rotation=90)
        ax.set_title(f"{_LEG_LABEL.get(leg, leg)} leg")
        ax.set_xlabel("lambda at pair midpoint")
        ax.set_ylim(0.0, max(0.06, float(values.max()) * 1.18))
        ax.legend(frameon=False, loc="best")

    # The direction cue goes inside the axes, not in the figure margin: a
    # rotated fig.text at x=0.005 collides with the y-axis label once
    # tight_layout pulls the axes leftward.
    axes[0].set_ylabel("neighbour overlap  (higher = better)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def plot_morphologies(estimate, path: Path | str):
    """Per-morphology mu_ex against the pooled estimate.

    Answers the question a single number cannot: is the uncertainty sampling
    noise inside each cell, or real heterogeneity between cells? The first is
    reducible by longer runs, the second only by more cells, and the honest
    response differs.
    """
    plt = _style()

    per = list(estimate.per_morphology)
    mus = np.array([m.mu_ex for m in per], float)
    errs = np.array([m.stderr for m in per], float)
    y = np.arange(len(per), dtype=float)

    # Height tracks the number of rows with a small floor. A fixed 1.7in of
    # non-data height left the 2-morphology case (the common one) mostly empty.
    fig, ax = plt.subplots(figsize=(3.6, max(1.6, 0.34 * len(per) + 1.0)))

    lo, hi = estimate.diagnostics.get("ci95", (np.nan, np.nan))
    if np.isfinite(lo) and np.isfinite(hi):
        ax.axvspan(lo, hi, color="#1f4e79", alpha=0.12, lw=0,
                   label="pooled 95% CI")
    ax.axvline(estimate.mu_ex, color="#1f4e79", lw=1.3,
               label=f"pooled  {estimate.mu_ex:+.2f} kcal/mol")

    ax.errorbar(mus, y, xerr=errs, fmt="o", ms=4, color="#333333",
                ecolor="#333333", elinewidth=1.0, capsize=2.5,
                label="morphology +/- within-cell sd", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([f"morph {m.index:02d}" for m in per])
    ax.invert_yaxis()
    ax.set_xlabel("excess chemical potential  (kcal/mol)")
    fig.subplots_adjust(left=0.26, right=0.97, top=0.86, bottom=0.30)

    # Name which variance dominates -- the actionable content of the figure.
    v_b = estimate.var_between
    v_w = estimate.var_within
    if np.isfinite(v_b) and np.isfinite(v_w) and max(v_b, v_w) > 0:
        which = ("between-morphology (run more cells)" if v_b > v_w
                 else "within-morphology (run longer)")
        ax.set_title(f"dominant variance: {which}")
    # Below the axes rather than loc="best": with only 2-4 rows there is no
    # interior whitespace, and "best" put the legend on top of the error bars.
    # tight_layout ignores an out-of-axes legend, so the axes rect is set
    # explicitly and savefig's tight bbox grows the canvas to include it.
    ax.margins(y=0.25)
    # bbox_to_anchor is in axes fractions, so a fixed offset shrinks in
    # absolute terms as the axes get shorter -- at M=2 the first legend row
    # then touched the x-label. Convert a fixed clearance (inches) instead.
    axes_h_in = fig.get_size_inches()[1] * (0.86 - 0.30)
    leg = ax.legend(frameon=False, loc="upper center", ncol=1,
                    handletextpad=0.5, borderaxespad=0.0,
                    bbox_to_anchor=(0.5, -0.45 / axes_h_in))

    fig.savefig(path, bbox_extra_artists=(leg,), bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def _figure_data(estimate):
    """Pull the per-state curves and overlaps out of a real ``FEPEstimate``.

    The data lives on the *leg* estimates of each morphology, not on the
    campaign's own ``diagnostics`` dict (which carries the variance
    decomposition). Written against those structures rather than a flattened
    convenience dict because a flattening step is another thing that can drift
    silently out of agreement with the estimators.

    The first usable morphology is used rather than an average across
    morphologies: these are diagnostics of the *ladder*, and averaging curves
    from different cells would blur exactly the per-state features -- a wide
    fluctuation region, a thin overlap pair -- that the figures exist to expose.
    Two accepted key spellings are tolerated for the fluctuation because TI
    records ``dudl_sd`` while a pre-computed profile may supply ``sd``.
    ``dudl_stderr`` is deliberately *not* accepted as a substitute: the band is
    documented as the per-state fluctuation, and quietly filling it with the
    error on the mean would draw a band a factor of sqrt(N) too narrow while
    still looking like the thing that governs overlap. A leg with no sd simply
    gets no curve.
    """
    curves: dict = {}
    overlaps: dict = {}

    per = [m for m in getattr(estimate, "per_morphology", ()) or ()
           if getattr(m, "legs", None)]
    if not per:
        return curves, overlaps
    legs = per[0].legs

    for leg_name, leg_est in legs.items():
        d = getattr(leg_est, "diagnostics", None) or {}
        lam = d.get("lambdas")
        mean = d.get("dudl_mean")
        sd = d.get("dudl_sd") or d.get("sd")
        if lam and mean and sd and len(lam) == len(mean) == len(sd):
            curves[leg_name] = (lam, mean, sd)

        ov = d.get("neighbour_overlap")
        if ov and lam and len(lam) == len(ov) + 1:
            mids = [0.5 * (lam[i] + lam[i + 1]) for i in range(len(ov))]
            overlaps[leg_name] = (mids, ov)

    return curves, overlaps


def write_campaign_figures(
    estimate,
    outdir: Path | str,
    *,
    min_overlap: float | None = None,
) -> list[Path]:
    """Every figure that can be built from what ``estimate`` actually carries.

    ``min_overlap`` should be the value the campaign actually screened against
    (``config.fep.min_overlap``). It is forwarded to the overlap figure so the
    drawn threshold is the same line the accept/reject decision used; a figure
    showing the module default while the run used a different value would
    misreport which pairs were considered adequate.

    Deliberately partial rather than all-or-nothing: a campaign run without the
    rerun matrix has no overlap data, and refusing to draw the morphology plot
    in that case would withhold the diagnostic that is available. Each figure is
    attempted independently and a failure is logged, never raised -- a plotting
    problem must not destroy a completed campaign's numbers.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    curves, overlaps = _figure_data(estimate)

    overlap_kw = {} if min_overlap is None else {"min_overlap": min_overlap}
    for name, fn, arg, kw in (
        ("fep_dudl.png", plot_dudl, curves, {}),
        ("fep_overlap.png", plot_overlap, overlaps, overlap_kw),
        ("fep_morphologies.png", plot_morphologies, estimate, {}),
    ):
        if not arg:  # empty dict -> nothing measured for this figure
            LOG.info("skipping %s: no data in the campaign diagnostics", name)
            continue
        try:
            written.append(fn(arg, outdir / name, **kw))
        except Exception as exc:  # noqa: BLE001 - a plot must not kill a result
            LOG.warning("could not write %s: %s", name, exc)

    return written
