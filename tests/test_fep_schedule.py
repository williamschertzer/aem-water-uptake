"""Lambda ladders and the two-leg state bookkeeping.

The invariant worth stating plainly: on the LJ leg the charges are off and the
pair lambda varies; on the Coulomb leg the LJ core is fully present and the
charge scaling varies. Getting that backwards, or letting a charge scale on both
legs, double-counts the electrostatic work -- which is the largest term in the
whole calculation for water.
"""

import pytest

from aemwater.fep.schedule import (
    DEFAULT_COUL_LAMBDAS,
    DEFAULT_LJ_LAMBDAS,
    FEPLeg,
    LambdaLadder,
    LambdaState,
    default_ladders,
)


def test_default_ladders_are_the_configured_defaults():
    lj, coul = default_ladders()
    assert lj.lambdas == DEFAULT_LJ_LAMBDAS
    assert coul.lambdas == DEFAULT_COUL_LAMBDAS
    assert lj.leg is FEPLeg.LJ
    assert coul.leg is FEPLeg.COUL


def test_overrides_are_honoured():
    lj, coul = default_ladders([0.0, 0.5, 1.0], [0.0, 1.0])
    assert lj.lambdas == (0.0, 0.5, 1.0)
    assert coul.lambdas == (0.0, 1.0)


# ------------------------------------------------------------- leg semantics --


def test_lj_leg_keeps_charges_off():
    """Leg 1 grows the core uncharged; any charge here belongs to leg 2."""
    for state in LambdaLadder(FEPLeg.LJ, DEFAULT_LJ_LAMBDAS):
        assert state.lambda_q == 0.0
        assert state.lambda_lj == state.lam


def test_coul_leg_keeps_the_lj_core_fully_present():
    """The whole reason electrostatics runs second."""
    for state in LambdaLadder(FEPLeg.COUL, DEFAULT_COUL_LAMBDAS):
        assert state.lambda_lj == 1.0
        assert state.lambda_q == state.lam


def test_legs_join_at_a_shared_physical_state():
    """The end of leg 1 and the start of leg 2 must be the same Hamiltonian."""
    lj, coul = default_ladders()
    end_of_lj = lj.states[-1]
    start_of_coul = coul.states[0]
    assert (end_of_lj.lambda_lj, end_of_lj.lambda_q) == (1.0, 0.0)
    assert (start_of_coul.lambda_lj, start_of_coul.lambda_q) == (1.0, 0.0)


def test_full_path_ends_at_a_real_water():
    lj, coul = default_ladders()
    final = coul.states[-1]
    assert (final.lambda_lj, final.lambda_q) == (1.0, 1.0)


def test_path_starts_from_nothing():
    lj, _ = default_ladders()
    first = lj.states[0]
    assert (first.lambda_lj, first.lambda_q) == (0.0, 0.0)


# ------------------------------------------------------------------- labels ---


def test_labels_are_unique_and_directory_safe():
    lj, coul = default_ladders()
    labels = [s.label for s in lj.states] + [s.label for s in coul.states]
    assert len(set(labels)) == len(labels)
    for label in labels:
        assert all(c.isalnum() or c in "._-" for c in label), label


def test_labels_sort_in_ladder_order():
    """Directory listings are read by humans; lexical order must match lambda."""
    lj, _ = default_ladders()
    labels = [s.label for s in lj.states]
    assert labels == sorted(labels)


def test_state_index_matches_position():
    lj, _ = default_ladders()
    for i, state in enumerate(lj.states):
        assert state.index == i


# --------------------------------------------------------------- neighbours ---


def test_neighbour_deltas_cover_every_adjacent_pair():
    lj, _ = default_ladders()
    deltas = lj.neighbour_deltas()
    assert len(deltas) == len(lj) - 1
    for (i, j, d), (a, b) in zip(deltas, zip(lj.lambdas, lj.lambdas[1:])):
        assert (i, j) == (lj.states[i].index, lj.states[j].index)
        assert d == pytest.approx(b - a)
        assert d > 0


def test_neighbour_deltas_sum_to_the_full_path():
    lj, _ = default_ladders()
    assert sum(d for _, _, d in lj.neighbour_deltas()) == pytest.approx(1.0)


def test_max_gap_reports_the_widest_spacing():
    ladder = LambdaLadder(FEPLeg.LJ, (0.0, 0.1, 0.7, 1.0))
    assert ladder.max_gap() == pytest.approx(0.6)


# --------------------------------------------------------------- validation ---


@pytest.mark.parametrize(
    "lams, match",
    [
        ((0.0,), "at least 2"),
        ((), "at least 2"),
        ((0.0, 0.5, 0.4, 1.0), "strictly increasing"),
        ((0.0, 0.5, 0.5, 1.0), "strictly increasing"),
        ((0.1, 0.5, 1.0), "span exactly"),
        ((0.0, 0.5, 0.9), "span exactly"),
    ],
)
def test_invalid_ladders_are_rejected(lams, match):
    with pytest.raises(ValueError, match=match):
        LambdaLadder(FEPLeg.LJ, lams)


def test_two_state_ladder_is_permitted():
    """Plain FEP between the endpoints -- poor overlap, but not invalid."""
    ladder = LambdaLadder(FEPLeg.COUL, (0.0, 1.0))
    assert len(ladder) == 2
    assert ladder.max_gap() == pytest.approx(1.0)


def test_state_is_hashable_and_frozen():
    state = LambdaState(leg=FEPLeg.LJ, index=0, lam=0.0)
    assert hash(state) is not None
    with pytest.raises(Exception):
        state.lam = 0.5
