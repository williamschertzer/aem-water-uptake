"""End-to-end validation on poly(vinylbenzyl trimethylammonium chloride).

The reference AEM chemistry: a polystyrene backbone with a benzyltrimethyl-
ammonium cation and a chloride counterion. Experimental hydration numbers for
this family sit around lambda = 10-20 at full hydration depending on IEC and
processing, which is the range the calculation should reproduce.

Deliberately small (4 chains x 8 units) so it finishes on a laptop. A production
run needs a cell several times larger: the Widom estimate is noisier in a small
box, and a cell whose edge is comparable to the water cluster size biases the
percolation analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from aemwater.analysis import hydration_structure, plot_uptake, write_report
from aemwater.config import (
    BoxSpec,
    InsertionSpec,
    MDSpec,
    PolymerSpec,
    RunConfig,
    WidomSpec,
)
from aemwater.driver import _read_final_state, run_uptake
from aemwater.prepare import prepare_dry_membrane

#: Benzyltrimethylammonium-functionalised styrene, the standard AEM model unit.
PVBTMA = "[*]CC([*])c1ccc(C[N+](C)(C)C)cc1"


def build_config(quick: bool) -> RunConfig:
    """Short protocol for a smoke test, longer one for a real number."""
    scale = 1 if quick else 5
    return RunConfig(
        polymer=PolymerSpec(smiles=PVBTMA, n_chains=4, chain_length=8,
                            counterion="Cl-", name="pVBTMA-Cl"),
        water_model="spce",
        md=MDSpec(temperature=298.15, pressure=1.0, cutoff=10.0,
                  anneal_steps=20_000 * scale, compression_steps=30_000 * scale,
                  dry_npt_steps=20_000 * scale, relax_npt_steps=20_000 * scale,
                  mpi_ranks=1),
        box=BoxSpec(initial_density=0.35, target_density=1.10, seed=7),
        insertion=InsertionSpec(batch_fraction=0.25, max_iterations=25, seed=7),
        widom=WidomSpec(n_blocks=5, steps_per_block=20_000 * scale,
                        insertions_per_call=100, bulk_box_length=22.0,
                        bulk_equil_steps=40_000, sigma_tolerance=2.0, seed=7),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", type=Path, default=Path("validation_run"))
    ap.add_argument("--quick", action="store_true",
                    help="short protocol: checks the machinery, not the physics")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    config = build_config(args.quick)
    args.workdir.mkdir(parents=True, exist_ok=True)
    config.dump_yaml(args.workdir / "config.yaml")

    dry = prepare_dry_membrane(config, args.workdir)
    print(json.dumps(dry.summary(), indent=2))

    result = run_uptake(config, args.workdir, dry.typed_chains)

    coords, elements, edge = _read_final_state(
        args.workdir / f"iter_{result.iterations[-1].index:03d}" / "relaxed.data"
    )
    import numpy as np

    cations = np.array([i for i, e in enumerate(elements) if e == "N"])
    structure = hydration_structure(coords, elements, edge, result.n_waters, cations)

    result.to_dataframe().to_csv(args.workdir / "uptake_trajectory.csv", index=False)
    plot_uptake(result, args.workdir / "uptake.png", bulk_mu_ex=result.bulk_mu_ex)
    write_report(result, structure, args.workdir / "report.md")
    (args.workdir / "result.json").write_text(json.dumps(
        {**result.summary(), **structure.summary()}, indent=2))

    print(json.dumps({**result.summary(), **structure.summary()}, indent=2))
    if 5.0 <= result.lambda_value <= 30.0:
        print("\nlambda is in the range reported for this chemistry.")
    else:
        print(f"\nWARNING: lambda = {result.lambda_value:.1f} is outside the "
              "range reported for pVBTMA-type membranes (roughly 5-30).")
    return 0 if result.converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
