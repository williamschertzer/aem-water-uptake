"""Where the total water chemical-potential gap crosses zero.

The uptake loop adds water in discrete batches, so the equilibrium water
content almost never coincides with a loaded state. Two things follow.

* **Stopping.** An iteration counts as a crossing when its sampling is
  adequate and its *total* gap (excess + density term, see
  :class:`aemwater.widom.SaturationTest`) is >= 0. It is deliberately not
  ``gap >= -k*sigma``: a tolerance on the stop condition stops the loop a
  systematic ``k*sigma/slope`` waters early, and does so more severely the
  noisier the estimate. Noise is instead carried into the endpoint's error
  bar, and the loop keeps loading until the gap brackets zero -- one or
  more trustworthy points below, one at or above. ``insertion.
  post_saturation_iterations`` adds points above the crossing to the fit.

* **Reporting.** The endpoint is the zero of a weighted linear fit of the
  gap against the water count over the points nearest the crossing, not the
  final loaded count (which includes the post-saturation overshoot). Its 95%
  interval propagates the fit covariance (delta method) plus the bulk
  reference's error, which is common to every point: it shifts the whole
  curve and therefore the root by ``sigma_bulk / slope``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

#: Two-sided 95% normal quantile, the same local decision the driver uses.
Z95 = 1.96


@dataclass(frozen=True)
class SaturationPoint:
    """Interpolated saturation water content for one trajectory."""

    n_waters: float
    n_waters_low: float
    n_waters_high: float
    #: True when trustworthy points exist on both sides of zero.
    bracketed: bool
    #: "fit" (weighted linear fit), "bracket" (two-point interpolation after a
    #: non-increasing fit), or "first_crossing" (no usable lower point).
    method: str
    n_points: int
    slope_kcal_per_water: float | None
    fit_indices: tuple[int, ...]

    def summary(self) -> dict[str, object]:
        row = asdict(self)
        row["fit_indices"] = list(self.fit_indices)
        return row


def _weighted_line(x, y, sigma):
    """Intercept, slope and their covariance for y = a + b x."""
    x, y, sigma = map(np.asarray, (x, y, sigma))
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)
    if np.isnan(sigma).any():
        # No usable per-point error: equal weights, residual-scaled below.
        sigma = np.ones_like(y)
        scale_by_residual = True
    else:
        scale_by_residual = False
    w = 1.0 / sigma ** 2
    design = np.column_stack([np.ones_like(x), x])
    normal = design.T @ (design * w[:, None])
    cov = np.linalg.inv(normal)
    a, b = cov @ (design.T @ (w * y))
    dof = len(x) - 2
    if dof > 0:
        chi2 = float(np.sum(w * (y - a - b * x) ** 2)) / dof
        # Never shrink the error bar below the per-point errors; inflate it
        # when the points scatter more than they claim (curvature, bad sigma).
        cov = cov * (chi2 if scale_by_residual else max(1.0, chi2))
    return float(a), float(b), cov


def estimate_saturation_point(
    iterations,
    bulk_stderr: float = 0.0,
    *,
    n_below: int = 2,
    n_above: int = 2,
) -> SaturationPoint | None:
    """Zero of the total gap near the first trustworthy crossing.

    ``iterations`` are :class:`aemwater.driver.Iteration` rows (anything with
    ``n_waters_after``, ``mu_gap``, ``mu_ex_stderr``, ``sampling_adequate``
    and ``saturated``). Returns None when no crossing has been recorded.
    """
    rows = list(iterations)
    first = next((k for k, it in enumerate(rows) if it.saturated), None)
    if first is None:
        return None

    def usable(it) -> bool:
        return (bool(it.sampling_adequate) and it.mu_gap is not None
                and math.isfinite(it.mu_gap))

    below = [k for k in range(first) if usable(rows[k]) and rows[k].mu_gap < 0]
    above = [k for k in range(first, len(rows)) if usable(rows[k])]
    picked = below[-n_below:] + above[:n_above]
    crossing = rows[first]
    n_cross = float(crossing.n_waters_after)

    if not below:
        # Already at or above zero at the first trustworthy point: the true
        # endpoint lies at or below it. Report it as an upper bound.
        low = float(rows[0].n_waters_before) if rows else 0.0
        return SaturationPoint(n_cross, low, n_cross, False, "first_crossing",
                               1, None, (first,))

    x = np.array([rows[k].n_waters_after for k in picked], dtype=float)
    y = np.array([rows[k].mu_gap for k in picked], dtype=float)
    s = np.array([rows[k].mu_ex_stderr if rows[k].mu_ex_stderr is not None
                  else float("nan") for k in picked], dtype=float)
    bulk_var = bulk_stderr ** 2 if math.isfinite(bulk_stderr) else 0.0

    method = "fit"
    if len(np.unique(x)) >= 2:
        a, b, cov = _weighted_line(x, y, s)
    else:
        b = float("nan")
    if not (math.isfinite(b) and b > 0):
        # A flat or decreasing local fit has no meaningful root. Fall back to
        # straight-line interpolation across the bracketing pair.
        method = "bracket"
        k0, k1 = below[-1], first
        x = np.array([rows[k0].n_waters_after, rows[k1].n_waters_after], float)
        y = np.array([rows[k0].mu_gap, rows[k1].mu_gap], float)
        s = np.array([rows[k0].mu_ex_stderr or 0.0, rows[k1].mu_ex_stderr or 0.0])
        picked = [k0, k1]
        if x[1] <= x[0] or y[1] <= y[0]:
            return SaturationPoint(n_cross, float(x[0]), n_cross, True,
                                   "first_crossing", 2, None, tuple(picked))
        b = (y[1] - y[0]) / (x[1] - x[0])
        a = y[0] - b * x[0]
        # Linear interpolation weights; the two points are independent.
        t = -y[0] / (y[1] - y[0])
        var_root = ((1 - t) ** 2 * s[0] ** 2 + t ** 2 * s[1] ** 2) / b ** 2
    else:
        root = -a / b
        var_root = (cov[0, 0] + root ** 2 * cov[1, 1]
                    + 2 * root * cov[0, 1]) / b ** 2
    root = -a / b
    half = Z95 * math.sqrt(max(var_root, 0.0) + bulk_var / b ** 2)
    # The root cannot be below zero waters; keep it inside the loaded range
    # if a noisy fit throws it outside, and say so through the interval.
    root_clamped = min(max(root, 0.0), float(max(r.n_waters_after for r in rows)))
    return SaturationPoint(
        n_waters=root_clamped,
        n_waters_low=max(0.0, root - half),
        n_waters_high=root + half,
        bracketed=True,
        method=method,
        n_points=len(picked),
        slope_kcal_per_water=float(b),
        fit_indices=tuple(rows[k].index if hasattr(rows[k], "index") else k
                          for k in picked),
    )


__all__ = ["SaturationPoint", "estimate_saturation_point", "Z95"]
