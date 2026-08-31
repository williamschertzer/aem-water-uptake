"""Lambda ladders for the two-leg alchemical protocol.

The two legs are separate objects rather than one list with a flag, because they
differ in what is being scaled (pair lambda vs. atomic charge), in which LAMMPS
mechanism applies it (``pair_coeff`` vs. the ``Atoms`` section of the data file),
and in which endpoint is the physical one. Conflating them is how a charge ends
up scaled twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Sequence


class FEPLeg(str, Enum):
    """Which coupling a state belongs to."""

    LJ = "lj"
    COUL = "coul"


#: Leg 1 -- soft-core LJ growth, charges off. Denser at low lambda, where a
#: soft-core dU/dlambda is largest: the core is still permeable, so the ghost
#: samples configurations whose energy changes fastest with lambda. An evenly
#: spaced ladder wastes states at the top where nothing happens.
#:
#: Measured on the bulk SPC/E validation run, <dU/dlambda> in kT peaks at
#: lambda = 0.2 (+14.4, against +0.7 at lambda = 0) and its standard deviation
#: peaks later still, at lambda = 0.35 (11.8 kT). So the busy region is the
#: shoulder around 0.2-0.5, not the immediate neighbourhood of zero, and this
#: ladder's two extra states at 0.05 and 0.1 buy less than the same two states
#: would buy at 0.25 and 0.45. It is kept as the production default because it
#: is the ladder the validation was run on and it is dense enough (12 states)
#: that the misallocation costs CPU time rather than accuracy. See
#: SCREENING_LJ_LAMBDAS for placement that follows the measurement.
DEFAULT_LJ_LAMBDAS: tuple[float, ...] = (
    0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
)

#: Leg 2 -- linear charge scaling with the LJ core fully present. Fewer states
#: suffice: U is quadratic in lambda_Q (the ghost's own periodic images through
#: PPPM scale as lambda_Q^2), which is smooth and well covered by 7 points.
DEFAULT_COUL_LAMBDAS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)

#: Screening ladders: 7 states per leg instead of 12 + 7, for the uptake loop's
#: per-iteration mu_ex where a 0.5 kcal/mol answer now beats a 0.1 kcal/mol
#: answer after the run is over.
#:
#: Placed at equal *thermodynamic length* -- equal integral of
#: sd(dU/dlambda) d(lambda) -- rather than equal lambda spacing or equal work.
#: That is the criterion that equalises the variance each interval contributes
#: to the free energy, and it is what makes a short ladder safe: the states go
#: where the fluctuation is, so no single interval dominates the error.
#:
#: Two alternatives were tried on the validation run's measured profile and
#: rejected. Equal lambda spacing starves the 0.2-0.5 shoulder on the LJ leg.
#: Equal *work* (equal integral of |<dU/dlambda>|) clusters four of six states
#: into lambda < 0.5 and then leaves a single 0.53-wide interval to the
#: endpoint, because the work density collapses above 0.65 while the
#: fluctuation does not: sd is still 4.5 kT at lambda = 0.65 where the mean is
#: 1.0 kT. Thermodynamic length is the metric that sees this.
#:
#: The measured segment lengths are 0.87-0.94 kT (LJ) and 1.01-1.07 kT (charge),
#: i.e. even to within 8%, with no interval wider than 0.32 in lambda.
SCREENING_LJ_LAMBDAS: tuple[float, ...] = (
    0.0, 0.23, 0.33, 0.41, 0.52, 0.68, 1.0,
)
SCREENING_COUL_LAMBDAS: tuple[float, ...] = (
    0.0, 0.24, 0.43, 0.58, 0.73, 0.87, 1.0,
)


@dataclass(frozen=True)
class LambdaState:
    """One fixed-lambda simulation.

    ``index`` is position within the leg, used for directory naming and as the
    row index of the MBAR energy matrix, so it must be stable across a run.
    """

    leg: FEPLeg
    index: int
    lam: float

    @property
    def label(self) -> str:
        """Directory-safe name, e.g. ``lj_03_l0.200``."""
        return f"{self.leg.value}_{self.index:02d}_l{self.lam:.3f}"

    @property
    def lambda_lj(self) -> float:
        """Pair-coupling lambda in force at this state.

        On the Coulomb leg the LJ core is fully present -- that is the whole
        point of running it second -- so this is 1.0 there.
        """
        return self.lam if self.leg is FEPLeg.LJ else 1.0

    @property
    def lambda_q(self) -> float:
        """Charge-scaling factor at this state. Zero throughout leg 1."""
        return self.lam if self.leg is FEPLeg.COUL else 0.0


@dataclass(frozen=True)
class LambdaLadder:
    """The ordered states of one leg."""

    leg: FEPLeg
    lambdas: tuple[float, ...]

    def __post_init__(self) -> None:
        lams = self.lambdas
        if len(lams) < 2:
            raise ValueError(
                f"{self.leg.value} ladder needs at least 2 states, got {len(lams)}"
            )
        if any(b <= a for a, b in zip(lams, lams[1:])):
            raise ValueError(
                f"{self.leg.value} lambdas must be strictly increasing, got {lams}"
            )
        if lams[0] != 0.0 or lams[-1] != 1.0:
            raise ValueError(
                f"{self.leg.value} ladder must span exactly 0 -> 1, got "
                f"{lams[0]} -> {lams[-1]}. The endpoints are the physical states; "
                "a truncated ladder silently computes a different free energy."
            )

    def __len__(self) -> int:
        return len(self.lambdas)

    def __iter__(self) -> Iterator[LambdaState]:
        for i, lam in enumerate(self.lambdas):
            yield LambdaState(leg=self.leg, index=i, lam=float(lam))

    @property
    def states(self) -> tuple[LambdaState, ...]:
        return tuple(self)

    def neighbour_deltas(self) -> list[tuple[int, int, float]]:
        """``(i, j, lambda_j - lambda_i)`` for each adjacent pair.

        These feed the inline ``compute fep`` perturbations, which measure the
        energy difference to the neighbouring state at no extra sampling cost and
        provide the BAR cross-check on the rerun matrix.
        """
        lams = self.lambdas
        return [(i, i + 1, float(lams[i + 1] - lams[i])) for i in range(len(lams) - 1)]

    def max_gap(self) -> float:
        """Largest spacing, the first thing to look at when overlap is poor."""
        return max(b - a for a, b in zip(self.lambdas, self.lambdas[1:]))


def default_ladders(
    lj_lambdas: Sequence[float] | None = None,
    coul_lambdas: Sequence[float] | None = None,
) -> tuple[LambdaLadder, LambdaLadder]:
    """The (LJ, Coulomb) ladders, with optional overrides from config."""
    lj = tuple(
        float(x) for x in (lj_lambdas if lj_lambdas is not None else DEFAULT_LJ_LAMBDAS)
    )
    coul = tuple(
        float(x) for x in (coul_lambdas if coul_lambdas is not None else DEFAULT_COUL_LAMBDAS)
    )
    return LambdaLadder(FEPLeg.LJ, lj), LambdaLadder(FEPLeg.COUL, coul)


__all__ = [
    "FEPLeg",
    "LambdaState",
    "LambdaLadder",
    "default_ladders",
    "DEFAULT_LJ_LAMBDAS",
    "DEFAULT_COUL_LAMBDAS",
]
