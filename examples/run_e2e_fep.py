"""End-to-end pipeline test: bulk mu_ex -> dry membrane -> hydrate -> compare.

Runs the same three stages the CLI does, in the same order, but prints the
saturation comparison side by side at the end so the whole chain can be checked
in one screenful.

    python examples/run_e2e_fep.py                    # ~1-2 h, 4 ranks
    python examples/run_e2e_fep.py --config other.yaml

Every number printed is physically meaningless -- see the header of
examples/e2e_fep_smoke.yaml. This asks "did the pipeline run and did the
comparison logic fire", not "how much water does this membrane hold".

Stages are resumable: rerun the same command after an interruption and the
equilibrated cell and the loop's checkpointed iterations are reused.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from aemwater.config import RunConfig
from aemwater.driver import obtain_bulk_reference, run_uptake
from aemwater.prepare import obtain_dry_membrane

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "examples" / "e2e_fep_smoke.yaml"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--ranks", type=int, default=None,
                    help="MPI ranks; default is md.mpi_ranks from the config")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    config = RunConfig.from_yaml(args.config)
    config.validate()
    ranks = args.ranks if args.ranks is not None else config.md.mpi_ranks
    workdir = Path(config.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"config   : {args.config}")
    print(f"backend  : mu_ex_method = {config.mu_ex_method}")
    print(f"workdir  : {workdir}")
    print(f"ranks    : {ranks}")

    marks: list[tuple[str, float]] = []

    def mark(label: str, t0: float) -> None:
        dt = time.time() - t0
        marks.append((label, dt))
        print(f"--- {label}: {dt / 60:.1f} min")

    # Stage 1 -- the reservoir. Computed by this code at these settings rather
    # than taken from the literature: the saturation criterion is a difference
    # between two numbers this protocol produced, and a converged published
    # value subtracted from an under-converged membrane estimate would compare
    # two different quantities. obtain_bulk_reference dispatches on
    # mu_ex_method, so the reservoir is measured by the same estimator as the
    # membrane.
    #
    # Run explicitly here (the driver would do it internally) so the stage is
    # timed and printed separately -- the point of this script.
    print("\n=== stage 1/3: bulk water reference ===")
    t0 = time.time()
    bulk = obtain_bulk_reference(config, workdir, ranks=ranks)
    mark("bulk reference", t0)
    estimate = bulk.mu_ex          # a WidomEstimate-shaped record either way
    print(f"mu_ex(bulk) = {estimate.mu_ex:+.3f} +/- {estimate.stderr:.3f} "
          f"kcal/mol  [method={bulk.method}]")

    # Stage 2 -- the dry cell, through the full 21-step scheme.
    print("\n=== stage 2/3: dry membrane ===")
    t0 = time.time()
    typed_chains, reused = obtain_dry_membrane(config, workdir, resume=True)
    mark("dry membrane", t0)
    print(f"chains: {len(typed_chains)}  (reused cached cell: {reused})")

    # Stage 3 -- hydrate, re-measure, compare, repeat.
    print("\n=== stage 3/3: hydration loop ===")
    t0 = time.time()
    result = run_uptake(config, workdir, typed_chains,
                        bulk_reference=bulk, resume=True)
    mark("hydration loop", t0)

    print("\n=== saturation comparison ===")
    print(f"mu_ex(bulk)     = {estimate.mu_ex:+.3f} +/- {estimate.stderr:.3f} "
          f"kcal/mol")
    final = result.iterations[-1] if result.iterations else None
    if final is not None:
        print(f"mu_ex(membrane) = {final.mu_ex:+.3f} +/- {final.mu_ex_stderr:.3f} "
              f"kcal/mol at lambda = {final.lambda_value:.2f}")
        print(f"gap             = {final.mu_gap:+.3f} kcal/mol "
              f"(saturated at this iteration: {final.saturated})")
    print(f"stop reason     = {result.stop_reason}")
    print(f"converged       = {result.converged}")
    print(f"lambda (H2O/ion)= {result.lambda_value:.2f}")
    print(f"uptake (wt%)    = {result.water_uptake_pct:.1f}")
    print(f"density dry/wet = {result.dry_density:.3f} / "
          f"{result.hydrated_density:.3f} g/cm3")

    # The per-iteration trajectory is what shows the two curves approaching,
    # so write it beside the summary as the CLI does.
    result.to_dataframe().to_csv(workdir / "uptake_trajectory.csv", index=False)

    summary = {
        "config": str(args.config),
        "mu_ex_method": config.mu_ex_method,
        "bulk": {"mu_ex": estimate.mu_ex, "stderr": estimate.stderr,
                 "method": bulk.method},
        "result": result.summary(),
        "timings_min": {label: round(dt / 60, 2) for label, dt in marks},
    }
    out = workdir / "e2e_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    print(f"wrote {workdir / 'uptake_trajectory.csv'}")
    print("Reminder: smoke-scale numbers. Not an uptake prediction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
