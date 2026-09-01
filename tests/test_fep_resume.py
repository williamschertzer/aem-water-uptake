"""Resume logic for FEP campaigns.

The tests that matter here are the negative ones. A resume that redoes work it
did not need to is a waste of CPU; a resume that *reuses* work it should not have
is a wrong free energy with no symptom, because every individual window is
internally consistent and every downstream check passes.
"""

from __future__ import annotations

import json

import pytest

from aemwater.config import PolymerSpec, RunConfig
from aemwater.fep.campaign import MorphologyEstimate
from aemwater.fep.estimators import LegEstimate
from aemwater.fep.resume import (
    StampMismatch,
    campaign_stamp,
    check_stamp,
    load_morphology,
    rerun_complete,
    save_morphology,
    state_complete,
)
from aemwater.fep.schedule import FEPLeg

CLEAN_LOG = "...thermo output...\nTotal wall time: 0:12:31\n"
KILLED_LOG = "...thermo output...\nStep Temp E_pair\n 1000 298.1 -4200.0\n"


def _finished_state(directory, log=CLEAN_LOG):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "state.log").write_text(log)
    for name in ("fep.dat", "pe.dat", "traj.lammpstrj"):
        (directory / name).write_text("1000 -1.0\n")
    return directory


def _config():
    return RunConfig(
        polymer=PolymerSpec(smiles="[*]CC[*]", n_chains=1, chain_length=1),
    )


# ---------------------------------------------------------------- windows ---
def test_finished_window_is_complete(tmp_path):
    assert state_complete(_finished_state(tmp_path / "lam_00"))


def test_window_killed_before_lammps_exited_is_not_complete(tmp_path):
    """The whole point: outputs exist but the run never finished.

    A killed run leaves partial fep.dat/pe.dat/traj files behind, so presence of
    the outputs cannot be the test. Only LAMMPS's terminal line proves the
    trajectory is the length the ladder expects.
    """
    sdir = _finished_state(tmp_path / "lam_00", log=KILLED_LOG)
    assert not state_complete(sdir)


@pytest.mark.parametrize("missing", ["fep.dat", "pe.dat", "traj.lammpstrj"])
def test_window_missing_a_downstream_input_is_not_complete(tmp_path, missing):
    sdir = _finished_state(tmp_path / "lam_00")
    (sdir / missing).unlink()
    assert not state_complete(sdir)


def test_empty_output_is_not_complete(tmp_path):
    """Zero-length files are the signature of a run killed at setup."""
    sdir = _finished_state(tmp_path / "lam_00")
    (sdir / "pe.dat").write_text("")
    assert not state_complete(sdir)


def test_absent_directory_is_not_complete(tmp_path):
    assert not state_complete(tmp_path / "never_ran")


def test_clean_exit_is_found_in_a_long_log(tmp_path):
    """The tail read must not miss the marker in a realistically large log."""
    sdir = _finished_state(tmp_path / "lam_00")
    padding = "".join(f"{i} 298.0 -4200.0\n" for i in range(200_000))
    (sdir / "state.log").write_text(padding + "Total wall time: 1:02:03\n")
    assert state_complete(sdir)


# ----------------------------------------------------------------- reruns ---
def test_rerun_completion_tracks_its_own_index(tmp_path):
    d = tmp_path / "rerun"
    d.mkdir()
    (d / "rerun_0.log").write_text(CLEAN_LOG)
    (d / "rerun_0.dat").write_text("1000 -1.0\n")
    assert rerun_complete(d, 0)
    # Index 1 shares the directory and must not be judged by index 0's files.
    assert not rerun_complete(d, 1)


# ------------------------------------------------------------------ stamps ---
def test_stamp_is_written_then_accepted(tmp_path):
    config = _config()
    stamp = campaign_stamp(config, kind="bulk", n_waters=1000)
    path = tmp_path / "campaign_stamp.json"
    check_stamp(path, stamp, resume=True)
    assert path.is_file()
    check_stamp(path, stamp, resume=True)  # must not raise


def test_changed_sampling_length_refuses_to_resume(tmp_path):
    """Averaging windows from two protocols would pass every internal check."""
    path = tmp_path / "campaign_stamp.json"
    check_stamp(path, campaign_stamp(_config(), kind="bulk", n_waters=1000),
                resume=True)

    longer = _config().with_overrides(**{"fep.production_steps": 200_000})
    with pytest.raises(StampMismatch, match="production_steps"):
        check_stamp(path, campaign_stamp(longer, kind="bulk", n_waters=1000),
                    resume=True)


def test_changed_ladder_refuses_to_resume(tmp_path):
    path = tmp_path / "campaign_stamp.json"
    check_stamp(path, campaign_stamp(_config(), kind="bulk", n_waters=1000),
                resume=True)

    denser = _config().with_overrides(
        **{"fep.lj_lambdas": [0.0, 0.25, 0.5, 0.75, 1.0]}
    )
    with pytest.raises(StampMismatch, match="lj_lambdas"):
        check_stamp(path, campaign_stamp(denser, kind="bulk", n_waters=1000),
                    resume=True)


def test_changed_box_size_refuses_to_resume(tmp_path):
    """n_waters lives outside FEPSpec, so it must be carried in explicitly."""
    path = tmp_path / "campaign_stamp.json"
    config = _config()
    check_stamp(path, campaign_stamp(config, kind="bulk", n_waters=1000),
                resume=True)
    with pytest.raises(StampMismatch, match="n_waters"):
        check_stamp(path, campaign_stamp(config, kind="bulk", n_waters=500),
                    resume=True)


def test_no_resume_overwrites_a_mismatched_stamp(tmp_path):
    """--no-resume must work in exactly the case it exists for."""
    path = tmp_path / "campaign_stamp.json"
    check_stamp(path, campaign_stamp(_config(), kind="bulk", n_waters=1000),
                resume=True)
    longer = _config().with_overrides(**{"fep.production_steps": 200_000})
    new = campaign_stamp(longer, kind="bulk", n_waters=1000)
    check_stamp(path, new, resume=False)
    assert json.loads(path.read_text())["production_steps"] == 200_000


def test_unreadable_stamp_refuses_rather_than_guessing(tmp_path):
    path = tmp_path / "campaign_stamp.json"
    path.write_text("{ truncated")
    with pytest.raises(StampMismatch, match="unreadable"):
        check_stamp(path, campaign_stamp(_config(), kind="bulk", n_waters=1000),
                    resume=True)


def test_bulk_and_membrane_stamps_differ(tmp_path):
    config = _config()
    assert (campaign_stamp(config, kind="bulk")["digest"]
            != campaign_stamp(config, kind="membrane")["digest"])


# ----------------------------------------------------- morphology results ---
def _estimate(index=0, mu_ex=-6.4):
    return MorphologyEstimate(
        index=index,
        mu_ex=mu_ex,
        stderr=0.12,
        legs={
            "lj": LegEstimate(estimator="mbar", leg=FEPLeg.LJ, delta_f=2.3,
                              stderr=0.05, n_effective=140.0),
            "coul": LegEstimate(estimator="mbar", leg=FEPLeg.COUL,
                                delta_f=mu_ex - 2.3, stderr=0.11,
                                n_effective=120.0),
        },
        workdir="morph00",
    )


def test_morphology_round_trips(tmp_path):
    path = save_morphology(_estimate(), tmp_path / "morphology.json")
    back = load_morphology(path)
    assert back.index == 0
    assert back.mu_ex == pytest.approx(-6.4)
    assert back.stderr == pytest.approx(0.12)
    assert back.usable
    # The leg enum must survive the JSON round trip as an enum, not a string:
    # combine_legs_for_morphology and the estimators branch on it.
    assert back.legs["lj"].leg is FEPLeg.LJ
    assert back.legs["coul"].delta_f == pytest.approx(-8.7)


def test_truncated_checkpoint_is_ignored_not_fatal(tmp_path):
    """Redoing a morphology beats refusing to start."""
    path = tmp_path / "morphology.json"
    path.write_text('{"index": 0, "mu_ex":')
    assert load_morphology(path) is None


def test_non_finite_checkpoint_is_ignored(tmp_path):
    bad = _estimate(mu_ex=float("nan"))
    path = save_morphology(bad, tmp_path / "morphology.json")
    assert load_morphology(path) is None


def test_absent_checkpoint_returns_none(tmp_path):
    assert load_morphology(tmp_path / "morphology.json") is None


# ------------------------------------------------- campaign actually skips ---
def _fake_lammps(calls):
    """A run_lammps that writes plausible outputs and records its calls.

    The point of the integration test is the *count* of LAMMPS invocations
    across a kill and a resume, which no unit test of the helpers can show.
    """
    def run_lammps(input_file, workdir=None, ranks=1, log_name=None,
                   extra_args=None, **kwargs):
        from pathlib import Path

        input_file = Path(input_file)
        directory = Path(workdir) if workdir else input_file.parent
        calls.append(str(input_file))
        (directory / (log_name or f"{input_file.stem}.log")).write_text(CLEAN_LOG)
        frames = "\n".join(
            f"{(i + 1) * 100} {-4200.0 - i * 0.01} {0.4 + i * 0.001} "
            f"{-0.4 - i * 0.001}"
            for i in range(6)
        )
        (directory / "fep.dat").write_text(
            "# FEP\n# comment\n# TimeStep pe dU_ti_plus dU_ti_minus\n"
            + frames + "\n"
        )
        (directory / "pe.dat").write_text(
            "\n".join(f"{(i + 1) * 100} {-4200.0 - i * 0.01}" for i in range(6))
            + "\n"
        )
        (directory / "traj.lammpstrj").write_text("ITEM: TIMESTEP\n100\n")
        return None
    return run_lammps


@pytest.fixture
def ti_config():
    """TI only, so no rerun pass, and a short ladder to keep the count legible."""
    return _config().with_overrides(**{
        "fep.n_morphologies": 2,
        "fep.lj_lambdas": [0.0, 0.5, 1.0],
        "fep.coul_lambdas": [0.0, 1.0],
        "fep.estimators": ["ti"],
        "fep.rerun_matrix": False,
        # The validator enforces a floor of ~20 frames per lambda, so these are
        # the smallest settings it accepts rather than the smallest that would
        # exercise the plumbing.
        "fep.production_steps": 4000,
        "fep.equil_steps": 100,
        "fep.sample_every": 100,
    })


def test_resumed_campaign_reruns_nothing_and_reports_the_same_number(
    tmp_path, monkeypatch, ti_config,
):
    """The behaviour the user asked for, asserted end to end.

    Five windows per morphology across two morphologies: a completed campaign is
    10 invocations, and re-running it must be 0 while returning the same mu_ex.
    """
    from aemwater.fep import campaign as campaign_module

    calls: list[str] = []
    monkeypatch.setattr(campaign_module, "run_lammps", _fake_lammps(calls),
                        raising=False)
    monkeypatch.setattr("aemwater.lammps.runner.run_lammps",
                        _fake_lammps(calls))

    first = campaign_module.run_bulk_campaign(
        config=ti_config, workdir=tmp_path / "run", n_waters=64,
    )
    assert len(calls) == 10, calls

    calls.clear()
    second = campaign_module.run_bulk_campaign(
        config=ti_config, workdir=tmp_path / "run", n_waters=64,
    )
    assert calls == []
    assert second.mu_ex == pytest.approx(first.mu_ex)
    assert second.stderr == pytest.approx(first.stderr)


def test_campaign_killed_mid_morphology_redoes_only_what_was_unfinished(
    tmp_path, monkeypatch, ti_config,
):
    """A kill inside morphology 1 must not cost morphology 0.

    This is the failure the feature exists for: before it, the exception
    discarded every finished window in the run.
    """
    from aemwater.fep import campaign as campaign_module

    calls: list[str] = []
    real = _fake_lammps(calls)

    def dies_on_the_eighth(*args, **kwargs):
        if len(calls) == 7:
            raise KeyboardInterrupt("walltime")
        return real(*args, **kwargs)

    monkeypatch.setattr("aemwater.lammps.runner.run_lammps", dies_on_the_eighth)
    with pytest.raises(KeyboardInterrupt):
        campaign_module.run_bulk_campaign(
            config=ti_config, workdir=tmp_path / "run", n_waters=64,
        )
    assert len(calls) == 7

    # Morphology 0 (5 windows) is checkpointed; morphology 1 had 2 of its 5
    # windows finished, so the resume owes exactly the remaining 3.
    calls.clear()
    monkeypatch.setattr("aemwater.lammps.runner.run_lammps", real)
    estimate = campaign_module.run_bulk_campaign(
        config=ti_config, workdir=tmp_path / "run", n_waters=64,
    )
    assert len(calls) == 3, calls
    assert estimate.n_morphologies == 2


def test_no_resume_recomputes_everything(tmp_path, monkeypatch, ti_config):
    from aemwater.fep import campaign as campaign_module

    calls: list[str] = []
    monkeypatch.setattr("aemwater.lammps.runner.run_lammps", _fake_lammps(calls))

    campaign_module.run_bulk_campaign(
        config=ti_config, workdir=tmp_path / "run", n_waters=64,
    )
    calls.clear()
    campaign_module.run_bulk_campaign(
        config=ti_config, workdir=tmp_path / "run", n_waters=64, resume=False,
    )
    assert len(calls) == 10


def test_resume_into_a_directory_with_different_settings_refuses(
    tmp_path, monkeypatch, ti_config,
):
    from aemwater.fep import campaign as campaign_module

    calls: list[str] = []
    monkeypatch.setattr("aemwater.lammps.runner.run_lammps", _fake_lammps(calls))
    campaign_module.run_bulk_campaign(
        config=ti_config, workdir=tmp_path / "run", n_waters=64,
    )

    longer = ti_config.with_overrides(**{"fep.production_steps": 5000})
    with pytest.raises(StampMismatch, match="production_steps"):
        campaign_module.run_bulk_campaign(
            config=longer, workdir=tmp_path / "run", n_waters=64,
        )
