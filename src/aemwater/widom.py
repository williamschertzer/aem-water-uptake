"""Excess chemical potential of water by Widom test-particle insertion.

The saturation criterion
------------------------
"No more water can be inserted" is not a geometric statement -- a membrane always
has *some* cavity somewhere. It is a thermodynamic one: water stops entering when
its chemical potential inside the membrane equals its chemical potential in the
reservoir it is equilibrating with (liquid water at the same T, p).

Widom's identity gives the excess part directly,

    mu_ex = -kT ln <exp(-beta dU)>

where dU is the energy of inserting a ghost molecule at a random position and the
average is over configurations and insertion attempts. The ideal-gas part depends
only on temperature and the number density, so for two systems at the same
temperature the *difference* in mu_ex, plus the density term, sets the driving
force for transfer.

Saturation is therefore declared when

    mu_ex(membrane at N waters) >= mu_ex(bulk water) - tolerance

with the tolerance set from the statistical uncertainty of both estimates rather
than picked arbitrarily.

Statistics
----------
The Boltzmann average is dominated by rare favourable insertions, so its
distribution is heavily skewed and a naive standard error understates the true
uncertainty. Two things are done about it:

* mu_ex is block-averaged. Each block is an independent estimate, and the
  standard error over blocks accounts for correlation within a block.
* The number of *effective* samples is reported. When a handful of insertions
  dominate the average, the estimate is flagged as unconverged rather than
  silently returned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .utils import LOG

#: Boltzmann constant in kcal/mol/K, matching LAMMPS 'real' units.
KB_KCAL = 0.0019872041


class WidomError(RuntimeError):
    """Raised when a chemical-potential estimate cannot be formed or trusted."""


@dataclass
class WidomEstimate:
    """A block-averaged excess chemical potential with its uncertainty."""

    mu_ex: float                 # kcal/mol
    stderr: float                # kcal/mol, standard error over blocks
    temperature: float
    n_blocks: int
    block_values: np.ndarray
    mean_boltzmann: float
    effective_samples: float
    volume: float = 0.0

    @property
    def converged(self) -> bool:
        """Whether the estimate rests on enough independent favourable insertions.

        A Boltzmann average carried by fewer than ~10 effective samples is not a
        chemical potential, it is a lucky insertion.
        """
        return (self.effective_samples >= MIN_EFFECTIVE_SAMPLES
                and self.n_blocks >= 3)

    def summary(self) -> dict[str, object]:
        return {
            "mu_ex_kcal_mol": round(self.mu_ex, 4),
            "stderr_kcal_mol": round(self.stderr, 4),
            "n_blocks": self.n_blocks,
            "effective_samples": round(self.effective_samples, 1),
            "converged": self.converged,
            "temperature_K": self.temperature,
        }


def mu_ex_from_boltzmann(mean_exp: float, temperature: float) -> float:
    """mu_ex = -kT ln <exp(-beta dU)>, guarding the log."""
    if mean_exp <= 0.0 or not math.isfinite(mean_exp):
        raise WidomError(
            f"Boltzmann average is {mean_exp!r}: no successful test insertion. "
            "The configuration may be too dense, or widom.insertions_per_call too small."
        )
    return -KB_KCAL * temperature * math.log(mean_exp)


#: Insertions needed for a converged bulk-water estimate at ambient density.
#:
#: Measured on this implementation: the block estimates of <exp(-dU/kT)> for
#: SPC/E at 298 K span four orders of magnitude, because the average is carried
#: by the rare trials that land in a cavity. The estimator is biased *upward*
#: (mu_ex too small in magnitude) until the tail is sampled, so an
#: under-converged run looks like a membrane that is easier to hydrate than it
#: is. 10^5 insertions gave -3.1 kcal/mol against a literature -6.3; the tail
#: needs of order 10^6.
#:
#: This is the single most expensive parameter in the workflow. Insertions cost
#: about 30 per second per rank with `full_energy` and PPPM, so 10^6 is roughly
#: nine core-hours for the reference alone -- computed once and cached across
#: every membrane sharing the same state point.
CONVERGED_INSERTIONS = 1_000_000

#: Kish effective sample size below which an estimate is not reported as a
#: measurement. Ten independent contributions is already generous for a
#: quantity whose distribution is this heavy-tailed.
MIN_EFFECTIVE_SAMPLES = 10


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size of a set of Boltzmann weights.

    (sum w)^2 / sum(w^2). Equals n for equal weights and drops toward 1 when a
    single insertion dominates -- which is the failure mode that makes a Widom
    estimate look precise while being wrong.
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights[np.isfinite(weights)]
    if weights.size == 0 or weights.sum() <= 0:
        return 0.0
    return float(weights.sum() ** 2 / np.square(weights).sum())


def estimate_from_series(
    boltzmann: np.ndarray,
    temperature: float,
    n_blocks: int = 5,
    volumes: np.ndarray | None = None,
) -> WidomEstimate:
    """Block-average a time series of <exp(-beta dU)> into mu_ex.

    ``boltzmann`` is the per-sample Boltzmann factor average that LAMMPS's
    ``fix widom`` reports (its output slot 2). Blocks are contiguous in time,
    which is what makes them approximately independent -- shuffling would
    destroy the correlation structure the block average exists to handle.
    """
    boltzmann = np.asarray(boltzmann, dtype=float)
    boltzmann = boltzmann[np.isfinite(boltzmann)]
    if boltzmann.size == 0:
        raise WidomError("no finite Widom samples in the series")
    n_blocks = max(1, min(n_blocks, boltzmann.size))
    blocks = np.array_split(boltzmann, n_blocks)

    block_mu = []
    for b in blocks:
        m = float(b.mean())
        if m <= 0:
            continue
        block_mu.append(mu_ex_from_boltzmann(m, temperature))
    if not block_mu:
        raise WidomError(
            "every block had a zero Boltzmann average: no insertion was ever "
            "accepted. Increase widom.insertions_per_call or widom.steps_per_block."
        )
    block_mu_arr = np.array(block_mu)

    mean_exp = float(boltzmann.mean())
    estimate = WidomEstimate(
        mu_ex=mu_ex_from_boltzmann(mean_exp, temperature),
        stderr=float(block_mu_arr.std(ddof=1) / math.sqrt(len(block_mu_arr)))
        if len(block_mu_arr) > 1
        else float("nan"),
        temperature=temperature,
        n_blocks=len(block_mu_arr),
        block_values=block_mu_arr,
        mean_boltzmann=mean_exp,
        effective_samples=effective_sample_size(boltzmann),
        volume=float(np.mean(volumes)) if volumes is not None and len(volumes) else 0.0,
    )
    LOG.info(
        "Widom: mu_ex = %.3f +/- %.3f kcal/mol over %d blocks (N_eff = %.1f)%s",
        estimate.mu_ex,
        estimate.stderr,
        estimate.n_blocks,
        estimate.effective_samples,
        "" if estimate.converged else "  [UNCONVERGED]",
    )
    return estimate


def read_widom_file(path, temperature: float, n_blocks: int = 5) -> WidomEstimate:
    """Parse the ``fix ave/time`` output written by the Widom template.

    Columns are step, mu_ex, <exp(-beta dU)>, volume -- matching the order the
    template writes. mu_ex is recomputed here from the Boltzmann factors rather
    than averaged directly, because the average of a logarithm is not the
    logarithm of an average and only the latter is the estimator.
    """
    from pathlib import Path

    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    if not rows:
        raise WidomError(f"no data rows in {path}")
    arr = np.array(rows)
    volumes = arr[:, 3] if arr.shape[1] > 3 else None
    return estimate_from_series(arr[:, 2], temperature, n_blocks=n_blocks,
                                volumes=volumes)


@dataclass
class SaturationTest:
    """Comparison of membrane and bulk chemical potentials."""

    membrane: WidomEstimate
    bulk: WidomEstimate
    tolerance_sigma: float = 2.0

    @property
    def difference(self) -> float:
        """mu_ex(membrane) - mu_ex(bulk). Negative means water still wants in."""
        return self.membrane.mu_ex - self.bulk.mu_ex

    @property
    def combined_stderr(self) -> float:
        a = self.membrane.stderr if math.isfinite(self.membrane.stderr) else 0.0
        b = self.bulk.stderr if math.isfinite(self.bulk.stderr) else 0.0
        return math.hypot(a, b)

    @property
    def saturated(self) -> bool:
        """Saturated once the membrane is no longer more favourable than bulk.

        The threshold is the combined statistical uncertainty, so a run stops on a
        difference it can actually resolve rather than on noise.
        """
        threshold = self.tolerance_sigma * self.combined_stderr
        return self.difference >= -threshold

    @property
    def trustworthy(self) -> bool:
        return self.membrane.converged and self.bulk.converged

    def summary(self) -> dict[str, object]:
        return {
            "mu_ex_membrane": round(self.membrane.mu_ex, 4),
            "mu_ex_bulk": round(self.bulk.mu_ex, 4),
            "difference_kcal_mol": round(self.difference, 4),
            "combined_stderr": round(self.combined_stderr, 4),
            "threshold_kcal_mol": round(self.tolerance_sigma * self.combined_stderr, 4),
            "saturated": self.saturated,
            "trustworthy": self.trustworthy,
        }


__all__ = [
    "WidomEstimate",
    "WidomError",
    "SaturationTest",
    "estimate_from_series",
    "read_widom_file",
    "mu_ex_from_boltzmann",
    "effective_sample_size",
    "CONVERGED_INSERTIONS",
    "MIN_EFFECTIVE_SAMPLES",
    "KB_KCAL",
]
