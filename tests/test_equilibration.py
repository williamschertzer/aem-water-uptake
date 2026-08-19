"""Dry-membrane equilibration schedule and convergence gate.

The gate is the thing that stops an unconverged dry cell reaching the uptake
loop, so it needs to discriminate all four regimes: right density and stable
(pass), wrong density, still densifying, and too few samples to judge.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from aemwater.config import PolymerSpec, RunConfig
from aemwater.prepare import check_dry_convergence


def _config(expected=1.10, **equil_overrides):
    cfg = RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]"))
    equil = dataclasses.replace(cfg.equilibration, expected_density=expected,
                                **equil_overrides)
    return dataclasses.replace(cfg, equilibration=equil)


def _trace(tmp_path, densities, name="dry_density.dat", every=500):
    """Write a file in the format `fix ave/time ... file` produces."""
    lines = ["# Time-averaged data for fix avg_dens", "# TimeStep v_dens v_vol_"]
    for i, d in enumerate(densities, start=1):
        lines.append(f"{i * every} {d:.6f} 250000")
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


def test_stable_trace_at_target_density_converges(tmp_path):
    rng = np.random.default_rng(0)
    path = _trace(tmp_path, 1.10 + rng.normal(0, 0.002, 20))
    conv = check_dry_convergence(path, _config(), thermo_every=500,
                                 n_averages=1, timestep_fs=1.0)
    assert conv.converged, conv.report()
    assert conv.density_ok and conv.drift_ok


def test_stable_trace_at_the_wrong_density_fails_on_density(tmp_path):
    """0.95 g/cm3 is the measured plateau of the old single-squeeze scheme."""
    rng = np.random.default_rng(1)
    path = _trace(tmp_path, 0.95 + rng.normal(0, 0.002, 20))
    conv = check_dry_convergence(path, _config(), thermo_every=500,
                                 n_averages=1, timestep_fs=1.0)
    assert not conv.converged
    assert not conv.density_ok
    assert conv.drift_ok, "a flat trace must not be reported as drifting"


def test_densifying_trace_fails_on_drift(tmp_path):
    """A cell still collapsing keeps collapsing once water is added.

    That is what produced the negative partial molar volume of water in the
    diagnosed runs, so it has to fail even when the density passes through the
    accepted window.
    """
    rng = np.random.default_rng(2)
    path = _trace(tmp_path, 1.08 + 0.004 * np.arange(20) + rng.normal(0, 0.001, 20))
    conv = check_dry_convergence(path, _config(), thermo_every=500,
                                 n_averages=1, timestep_fs=1.0)
    assert not conv.converged
    assert not conv.drift_ok
    assert conv.second_half > conv.first_half


def test_noise_alone_does_not_count_as_drift(tmp_path):
    """A slope inside its own uncertainty is not evidence of densification.

    Without the standard-error test the criterion is unpassable on a short
    window: scatter fakes a slope well above a 0.002 tolerance.
    """
    rng = np.random.default_rng(3)
    path = _trace(tmp_path, 1.10 + rng.normal(0, 0.004, 8))
    conv = check_dry_convergence(path, _config(), thermo_every=500,
                                 n_averages=1, timestep_fs=1.0)
    assert conv.drift_ok, conv.report()
    assert abs(conv.drift_per_100ps) <= 2.0 * conv.drift_stderr_per_100ps


def test_too_few_samples_is_not_convergence(tmp_path):
    """An empty or near-empty density file must not read as converged."""
    path = _trace(tmp_path, [1.10, 1.10])
    conv = check_dry_convergence(path, _config(), thermo_every=500,
                                 n_averages=1, timestep_fs=1.0)
    assert not conv.converged
    assert conv.n_samples < 4

    missing = check_dry_convergence(tmp_path / "absent.dat", _config(),
                                    thermo_every=500, n_averages=1,
                                    timestep_fs=1.0)
    assert not missing.converged
    assert not np.isfinite(missing.density)


def test_expected_density_falls_back_to_the_packing_target(tmp_path):
    cfg = RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]"))
    assert cfg.equilibration.expected_density is None
    rng = np.random.default_rng(4)
    path = _trace(tmp_path, cfg.box.target_density + rng.normal(0, 0.002, 20))
    conv = check_dry_convergence(path, cfg, thermo_every=500, n_averages=1,
                                 timestep_fs=1.0)
    assert conv.expected_density == pytest.approx(cfg.box.target_density)
    assert conv.converged, conv.report()


def test_drift_is_reported_per_100ps_not_per_step(tmp_path):
    """The slope must be in the units the tolerance is stated in.

    Fitting against step number instead of ps understates the drift by the
    timestep factor, which would let a densifying cell through.
    """
    # +0.01 g/cm3 over 10 samples spaced 500 steps at 1 fs = 5 ps -> 50 ps total.
    path = _trace(tmp_path, 1.10 + 0.01 * np.arange(10) / 9.0, every=5000)
    conv = check_dry_convergence(path, _config(), thermo_every=5000,
                                 n_averages=1, timestep_fs=1.0)
    # 5000 steps * 1 fs = 5 ps per sample; 0.01 g/cm3 over 45 ps.
    assert conv.drift_per_100ps == pytest.approx(0.01 / 45.0 * 100.0, rel=0.05)


def test_convergence_serialises_every_criterion(tmp_path):
    """`convergence.json` is the record of why a run was accepted or refused."""
    rng = np.random.default_rng(5)
    path = _trace(tmp_path, 1.10 + rng.normal(0, 0.002, 20))
    d = check_dry_convergence(path, _config(), thermo_every=500,
                              n_averages=1, timestep_fs=1.0).to_dict()
    for key in ("converged", "density_g_cm3", "expected_density_g_cm3",
                "density_ok", "drift_g_cm3_per_100ps", "drift_ok",
                "drift_stderr_g_cm3_per_100ps", "n_samples",
                "first_half_mean", "second_half_mean"):
        assert key in d, key


def test_final_stage_is_not_scaled_by_time_scale():
    """`final_npt_ps` is absolute; scaling it twice emptied the density file.

    With `time_scale` applied to step 21 as well, the smoke config's production
    window fell to 0.15 ps -- below one `thermo_every` interval -- so `fix
    ave/time` wrote no rows and the gate failed for want of samples rather than
    for anything physical.
    """
    from aemwater.lammps.inputs import equilibration_schedule

    cfg = RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]"))
    equil = dataclasses.replace(cfg.equilibration, time_scale=0.01,
                                final_npt_ps=15.0)
    sched = equilibration_schedule(cfg.md, equil)

    assert sched[-1].ps == pytest.approx(15.0), "step 21 must not be scaled"
    # The other 20 stages ARE scaled.
    assert sum(s.ps for s in sched[:-1]) == pytest.approx(760.0 * 0.01)


def test_every_example_config_yields_enough_density_samples():
    """A shipped config must not render a deck the gate cannot evaluate."""
    from pathlib import Path

    from aemwater.lammps.inputs import equilibration_schedule

    examples = sorted(Path(__file__).resolve().parent.parent
                      .joinpath("examples").glob("*.yaml"))
    assert examples, "no example configs found"
    for path in examples:
        cfg = RunConfig.from_yaml(path)
        if cfg.equilibration.scheme != "21step":
            continue
        sched = equilibration_schedule(cfg.md, cfg.equilibration)
        prod = sched[-1].steps(cfg.md.timestep)
        n_avg = max(1, prod // (cfg.md.thermo_every * 10))
        rows = prod // (cfg.md.thermo_every * n_avg)
        assert rows >= 4, (
            f"{path.name}: production window yields {rows} density sample(s); "
            "the convergence check needs at least 4")


def test_result_summary_carries_the_dry_convergence_verdict():
    """An uptake number built on an unconverged dry cell must say so.

    Leaving the verdict in the prepare-stage log means a resumed run, or anyone
    reading `uptake_state.json` later, cannot tell whether the dry membrane the
    number rests on was trustworthy.
    """
    from pathlib import Path

    from aemwater.driver import UptakeResult

    def _result(dry_convergence):
        return UptakeResult(
            iterations=[], n_waters=120, lambda_value=8.0,
            water_uptake_pct=22.0, hydrated_density=1.08, dry_density=1.15,
            stop_reason="chemical_potential", converged=True, bulk_mu_ex=-6.1,
            workdir=Path("."), dry_convergence=dry_convergence,
        )

    passed = _result({"converged": True, "density_g_cm3": 1.15})
    assert passed.summary()["dry_converged"] is True

    failed = _result({"converged": False, "density_g_cm3": 0.95})
    assert failed.summary()["dry_converged"] is False
    assert failed.summary()["dry_convergence"]["density_g_cm3"] == 0.95

    # Pre-gate runs have no record; that is distinct from "failed".
    legacy = _result(None)
    assert legacy.summary()["dry_converged"] is None


# --------------------------------------------------------- config validation --
def test_high_temperature_must_exceed_the_operating_temperature():
    from aemwater.config import ConfigError

    cfg = RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]"))
    with pytest.raises(ConfigError, match="high_temperature"):
        dataclasses.replace(
            cfg, equilibration=dataclasses.replace(
                cfg.equilibration, high_temperature=250.0))


def test_max_pressure_must_exceed_the_operating_pressure():
    from aemwater.config import ConfigError

    cfg = RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]"))
    with pytest.raises(ConfigError, match="max_pressure"):
        dataclasses.replace(
            cfg, equilibration=dataclasses.replace(
                cfg.equilibration, max_pressure=0.5))


def test_scheme_must_be_recognised():
    from aemwater.config import ConfigError

    cfg = RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]"))
    with pytest.raises(ConfigError, match="scheme"):
        dataclasses.replace(
            cfg, equilibration=dataclasses.replace(cfg.equilibration,
                                                   scheme="squeeze"))


def test_equilibration_round_trips_through_yaml(tmp_path):
    """The new section must survive dump/load, or configs silently lose it."""
    cfg = RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]"))
    cfg = dataclasses.replace(
        cfg, equilibration=dataclasses.replace(
            cfg.equilibration, max_pressure=40000.0, time_scale=0.5,
            expected_density=1.18))
    path = cfg.dump_yaml(tmp_path / "cfg.yaml")
    back = RunConfig.from_yaml(path)
    assert back.equilibration.max_pressure == pytest.approx(40000.0)
    assert back.equilibration.time_scale == pytest.approx(0.5)
    assert back.equilibration.expected_density == pytest.approx(1.18)
    assert back.equilibration.scheme == "21step"
