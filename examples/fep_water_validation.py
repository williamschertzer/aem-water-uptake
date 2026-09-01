"""Bulk SPC/E mu_ex by FEP -- the calculation that validates the protocol.

Restartable. This is 10 morphologies x (8 + 5) lambda windows plus a rerun pass
of the same order, so an exit part-way through used to discard everything: the
result only existed once the final JSON was written. Now every finished lambda
window and every finished morphology is left on disk and skipped on the next
invocation, so re-running the script picks up where it stopped.

    python examples/fep_water_validation.py                 # start or resume
    python examples/fep_water_validation.py --no-resume     # discard and restart
    python examples/fep_water_validation.py --ranks 8       # MPI ranks per window

Completeness is judged from LAMMPS's own terminal wall-time line plus the output
files the next stage reads, not from a marker this package writes -- a marker can
outlive what it describes. A window killed mid-trajectory is redone from its
start rather than restarted mid-way, because a half-length trace would carry
less weight in the estimators than its neighbours and quietly reweight the
ladder.
"""

import argparse
import json
import logging
from pathlib import Path

from aemwater.config import PolymerSpec, RunConfig
from aemwater.fep.campaign import run_bulk_campaign, write_campaign_report

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--no-resume", dest="resume", action="store_false",
    help="ignore finished windows on disk and recompute from scratch",
)
parser.add_argument(
    "--ranks", type=int, default=1,
    help="MPI ranks per lambda window (default 1)",
)
parser.add_argument(
    "--workdir", type=Path, default=Path("runs/fep_water_validation"),
    help="run directory; resuming reads and writes here",
)
args = parser.parse_args()

# Progress goes to the console: the campaign logs each window as it starts,
# which is the only way to see that a multi-hour run is alive and, on a resume,
# which windows are being skipped.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

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

workdir = args.workdir
workdir.mkdir(parents=True, exist_ok=True)
config.dump_yaml(workdir / "config.yaml")

estimate = run_bulk_campaign(
    config=config,
    workdir=workdir,
    n_waters=1000,
    ranks=args.ranks,
    resume=args.resume,
)

write_campaign_report(estimate, workdir / "fep_bulk.json")

print(json.dumps({
    **estimate.summary(),
    "per_morphology": [
        result.summary() for result in estimate.per_morphology
    ],
}, indent=2))