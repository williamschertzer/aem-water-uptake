import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aemwater.config import RunConfig, PolymerSpec
from aemwater.dry_snapshots import snapshot_offsets, prepare_snapshot_seeds
from aemwater.uptake_campaign import UptakeCampaignError


def config(**overrides):
    return RunConfig(polymer=PolymerSpec(smiles='[*]CC[*]')).with_overrides(**overrides)


def test_offsets_cover_final_two_ns():
    c = config(**{'equilibration.final_npt_ps': 4000,
                  'equilibration.snapshot_count': 3, 'md.timestep': 2})
    assert snapshot_offsets(c) == [1000000, 1500000, 2000000]
    assert snapshot_offsets(c.with_overrides(**{'equilibration.snapshot_count': 1})) == [2000000]


def test_shared_preparation_resume_and_tamper_detection(tmp_path, monkeypatch):
    import aemwater.prepare as prep
    calls = []
    def obtain(c, folder, resume=True):
        calls.append(c)
        dry = folder / 'dry'
        dry.mkdir(exist_ok=True)
        (dry / 'convergence.json').write_text('{"converged": true}')
        (dry / prep.TYPED_CHAIN_CHECKPOINT).write_bytes(b'typed')
        for i in range(c.equilibration.snapshot_count):
            (dry / f'snapshot_{i:03d}.data').write_text(f'state {i}')
        return ['typed'], False
    monkeypatch.setattr(prep, 'obtain_dry_membrane', obtain)
    c, chains = prepare_snapshot_seeds(config(), tmp_path, 3)
    assert len(calls) == 1 and chains == ['typed']
    assert c.equilibration.final_npt_ps == 3000
    assert [(tmp_path / f'morph{i:02d}/dry/dry.data').read_text() for i in range(3)] == ['state 0', 'state 1', 'state 2']
    prepare_snapshot_seeds(config(), tmp_path, 3)
    with pytest.raises(UptakeCampaignError, match='settings changed'):
        prepare_snapshot_seeds(config(), tmp_path, 2)
    (tmp_path / 'shared_dry/dry/snapshot_000.data').write_text('changed')
    with pytest.raises(UptakeCampaignError, match='missing or changed'):
        prepare_snapshot_seeds(config(), tmp_path, 3)


def test_rejects_old_campaign(tmp_path):
    (tmp_path / 'morph00').mkdir()
    with pytest.raises(UptakeCampaignError, match='no snapshot provenance'):
        prepare_snapshot_seeds(config(), tmp_path, 3)


def test_does_not_release_unconverged_snapshots(tmp_path, monkeypatch):
    import aemwater.prepare as prep
    def obtain(c, folder, resume=True):
        (folder / 'dry').mkdir()
        (folder / 'dry/convergence.json').write_text('{"converged": false}')
        return [], True
    monkeypatch.setattr(prep, 'obtain_dry_membrane', obtain)
    with pytest.raises(UptakeCampaignError, match='not passed convergence'):
        prepare_snapshot_seeds(config(), tmp_path, 3)
    assert not (tmp_path / 'morph00').exists()


def test_template_writes_snapshots_without_restarting_barostat():
    from aemwater.lammps.inputs import _environment, equilibration_schedule
    from jinja2 import DictLoader, ChoiceLoader
    c = config(**{'equilibration.final_npt_ps': 3000,
                  'equilibration.snapshot_count': 3})
    env = _environment()
    env.loader = ChoiceLoader([DictLoader({'common.in.j2': ''}), env.loader])
    text = env.get_template('equilibrate.in.j2').render(
        constraints=SimpleNamespace(shake_command=''), md=c.md, seed=1,
        equil=c.equilibration, equil_schedule=equilibration_schedule(c.md, c.equilibration),
        equil_total_ps=4000, n_averages=10, density_file='density.dat',
        snapshot_steps=snapshot_offsets(c), out_data='dry.data', out_restart='dry.restart')
    tail = text.split('# -- step 21/21:')[1]
    assert tail.count('fix             eq21 ') == 1
    assert tail.count('run             1000000') == 3
    assert tail.index('snapshot_002.data') < tail.index('unfix           eq21')


def test_campaign_prepares_once_and_reports_conditional_spread(tmp_path, monkeypatch):
    import pandas as pd
    import aemwater.prepare as prep
    import aemwater.driver as driver
    from aemwater.uptake_campaign import run_uptake_campaign
    prepared, hydrated = [], []
    def obtain(c, folder, resume=True):
        prepared.append(folder)
        dry = folder / 'dry'
        dry.mkdir(exist_ok=True)
        (dry / 'convergence.json').write_text('{"converged": true}')
        (dry / prep.TYPED_CHAIN_CHECKPOINT).write_bytes(b'typed')
        for i in range(3):
            (dry / f'snapshot_{i:03d}.data').write_text(f'state {i}')
        return ['typed'], False
    def uptake(c, folder, chains, **kwargs):
        hydrated.append((c.box.seed, c.insertion.seed,
                         (folder / 'dry/dry.data').read_text()))
        return SimpleNamespace(n_waters=5, lambda_value=.5,
            water_uptake_pct=2., hydrated_density=1.1, stop_reason='saturated',
            converged=True, iterations=[1], to_dataframe=lambda: pd.DataFrame(),
            summary=lambda: {})
    monkeypatch.setattr(prep, 'obtain_dry_membrane', obtain)
    monkeypatch.setattr(driver, 'run_uptake', uptake)
    ref = SimpleNamespace(mu_ex=SimpleNamespace(mu_ex=-6.8, converged=True))
    result = run_uptake_campaign(config(), tmp_path, n_morphologies=3,
                                bulk_reference=ref, parallel_morphologies=2)
    assert len(prepared) == 1
    assert len({row[0] for row in hydrated}) == 1
    assert len({row[1] for row in hydrated}) == 3
    assert {row[2] for row in hydrated} == {'state 0', 'state 1', 'state 2'}
    assert result.summary()['water_uptake_stderr'] is None
    assert result.summary()['water_uptake_ci95'] == [None, None]
    assert result.diagnostics['independence_verified'] is False
