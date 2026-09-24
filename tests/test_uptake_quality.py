"""Local FEP precision and campaign replication are separate decisions."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from aemwater.config import PolymerSpec, RunConfig
from aemwater.driver import _uptake_saturation_test
from aemwater.fep.campaign import FEPEstimate, MorphologyEstimate
from aemwater.uptake_campaign import MorphologyUptake


def estimate(stderr=0.05, overlap=0.2, samples=100):
    leg = SimpleNamespace(delta_f=-3.0, stderr=stderr,
                          diagnostics={"neighbour_overlap": [overlap], "N_k": [samples, samples]})
    return FEPEstimate(-6.4, stderr, 300, 1,
                       per_morphology=[MorphologyEstimate(0, -6.4, stderr,
                                                         legs={"lj": leg, "coul": leg})],
                       between_unmeasured=True)


def test_precise_local_crossing_does_not_claim_replicated_fep_convergence():
    cfg = RunConfig(polymer=PolymerSpec(smiles="CC"))
    membrane = estimate()
    bulk = SimpleNamespace(mu_ex=-6.5, stderr=0.05, converged=True)
    assert not membrane.converged
    assert _uptake_saturation_test(membrane, bulk, cfg).saturated
    assert not membrane.converged
    bulk.converged = False
    assert not _uptake_saturation_test(membrane, bulk, cfg).saturated


@pytest.mark.parametrize("kwargs", [
    {"stderr": 0.5}, {"stderr": float("nan")},
    {"overlap": 0.001}, {"overlap": float("nan")}, {"samples": 5},
])
def test_poor_local_sampling_cannot_stop_uptake(kwargs):
    cfg = RunConfig(polymer=PolymerSpec(smiles="CC"))
    bulk = SimpleNamespace(mu_ex=-6.5, stderr=0.05, converged=True)
    assert not _uptake_saturation_test(estimate(**kwargs), bulk, cfg).saturated


@pytest.mark.parametrize("reason", ["insertion_stalled", "geometric_saturation", "max_iterations"])
def test_censored_endpoints_never_enter_campaign_average(reason):
    m = MorphologyUptake(0, 1, ".", 100, 2.0, 10.0, 1.2, reason, True, 10)
    assert not m.usable
    assert replace(m, stop_reason="thermodynamic_saturation").usable
    assert not replace(m, stop_reason="thermodynamic_saturation", converged=False).usable


@pytest.mark.parametrize("inserted,expected_reason,extra,crossing", [
    (0, "insertion_stalled", 0, False),
    (1, "max_iterations", 0, False),
    (1, "thermodynamic_saturation", 2, True),
])
def test_loading_loop_never_calls_insertion_blockage_converged(tmp_path, monkeypatch,
                                                              inserted, expected_reason,
                                                              extra, crossing):
    from aemwater import driver
    import aemwater.insertion
    cfg = RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]", n_chains=1, chain_length=1))
    cfg = cfg.with_overrides(**{"insertion.max_iterations": 3,
                               "insertion.post_saturation_iterations": extra})
    dry = tmp_path / "dry" / "dry.data"
    dry.parent.mkdir()
    dry.write_text("placeholder")
    monkeypatch.setattr(driver, "_read_final_state", lambda p: (None, [], 30.0))
    monkeypatch.setattr(driver, "_check_reference_matches", lambda *a, **k: None)
    monkeypatch.setattr(driver, "_membrane_mu_ex_fep", lambda *a: SimpleNamespace(
        mu_ex=-10., stderr=.05, converged=True))
    monkeypatch.setattr(aemwater.insertion, "insert_waters", lambda c,e,box,n,*a,**k:
                        SimpleNamespace(n_requested=n, n_inserted=inserted,
                                        saturated=True,
                                        void_map=SimpleNamespace(free_volume_fraction=0.01)))
    monkeypatch.setattr(driver, "_run_iteration", lambda *a: {
        "coords": None, "elements": [], "edge": 30., "density": 1.2,
        "volume": 27000., "mu_ex": -10., "stderr": .05, "mu_gap": -3.5,
        "saturated": crossing, "trustworthy": True,
    })
    bulk = SimpleNamespace(mu_ex=SimpleNamespace(mu_ex=-6.5, stderr=.05, converged=True),
                           sanity=lambda: [], method="fep", settings=None)
    outcome = driver.run_uptake(cfg, tmp_path, [object()], bulk_reference=bulk)
    assert outcome.stop_reason == expected_reason
    assert outcome.converged is crossing
    assert outcome.n_waters == 3 * inserted


@pytest.mark.parametrize("flags,extra,expected", [
    ([], 2, False),
    ([False, True], 0, True),
    ([False, True], 2, False),
    ([False, True, True], 2, False),
    ([False, True, False, False], 2, True),
    ([True, False, True], 2, True),
    ([False, False, False], 2, False),
])
def test_post_saturation_cycles_recover_from_checkpoint(flags, extra, expected):
    from aemwater.driver import _saturation_complete
    rows = [SimpleNamespace(saturated=value) for value in flags]
    assert _saturation_complete(rows, extra) is expected


def test_negative_post_saturation_cycles_rejected():
    from aemwater.config import InsertionSpec
    with pytest.raises(ValueError, match="post_saturation_iterations"):
        InsertionSpec(post_saturation_iterations=-1).validate()
