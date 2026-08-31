import json
from pathlib import Path

from aemwater.config import PolymerSpec, RunConfig
from aemwater.fep.campaign import run_bulk_campaign, write_campaign_report

config = RunConfig(
    polymer=PolymerSpec(
        smiles="[*]CC[*]",
        n_chains=1,
        chain_length=1,
    ),
).with_overrides(**{
    "water_model": "spce",
    "md.temperature": 298.15,
    "md.cutoff": 10.0,
    "fep.n_morphologies": 10,

    # Published validation-resolution ladders
    "fep.lj_lambdas": [
        0.0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0
    ],
    "fep.coul_lambdas": [
        0.0, 0.25, 0.50, 0.75, 1.0
    ],

    # Cheap validation settings from docs/fep_design.md
    "fep.equil_steps": 10000,
    "fep.production_steps": 100000,
    "fep.sample_every": 1000,
    "fep.max_stderr": 0.60,
})

workdir = Path("runs/fep_water_validation")
workdir.mkdir(parents=True, exist_ok=True)
config.dump_yaml(workdir / "config.yaml")

estimate = run_bulk_campaign(
    config=config,
    workdir=workdir,
    n_waters=1000,
    ranks=1,
)

write_campaign_report(estimate, workdir / "fep_bulk.json")

print(json.dumps({
    **estimate.summary(),
    "per_morphology": [
        result.summary() for result in estimate.per_morphology
    ],
}, indent=2))