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


#: Leg 1 -- soft-core LJ growth, charges off. Clustered near zero because a
#: soft-core dU/dlambda peaks there: the core is still permeable, so the ghost
#: samples configurations whose energy changes fastest with lambda. An evenly
#: spaced ladder wastes states at the top where nothing happens and starves the
#: bottom where the variance lives.
DEFAULT_LJ_LAMBDAS: tuple[float, ...] = (
    0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
)

#: Leg 2 -- linear charge scaling with the LJ core fully present. Fewer states
#: suffice: U is quadratic in lambda_Q (the ghost's own periodic images through
#: PPPM scale as lambda_Q^2), which is smooth and well covered by 7 points.
DEFAULT_COUL_LAMBDAS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)


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
