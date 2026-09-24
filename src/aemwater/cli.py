"""Command line interface.

Three subcommands mirroring the three phases of the workflow:

    aemwater prepare  -- SMILES to an equilibrated dry membrane
    aemwater bulk     -- the bulk-water reference (cached, shared across runs)
    aemwater run      -- the water-loading loop, and the answer

``run`` will invoke the other two if their outputs are missing, so a single
command is enough for a first calculation. They are exposed separately because
both are expensive, reusable, and worth inspecting before committing to a loop.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import RunConfig
from .utils import LOG


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workdir", type=Path, default=Path("aem_run"),
                   help="directory for inputs, logs and results")
    p.add_argument("--config", type=Path,
                   help="YAML config; command-line options override it")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--ranks", type=int, help="MPI ranks for LAMMPS")


def _add_polymer(p: argparse.ArgumentParser) -> None:
    p.add_argument("--smiles", help="repeat-unit SMILES, [*] marking the two "
                                    "polymerisation points")
    p.add_argument("--n-chains", type=int, help="number of chains in the cell")
    p.add_argument("--chain-length", type=int, help="repeat units per chain")
    p.add_argument("--counterion", help="e.g. Cl, Br, OH")
    p.add_argument("--water-model", choices=["spce", "tip3p", "tip4p"])
    p.add_argument("--temperature", type=float, help="K")


def _load_config(args) -> RunConfig:
    """Config file if given, otherwise built from the command line.

    RunConfig requires a polymer, so there is no empty default to override:
    without a config file the SMILES has to come from the arguments.
    """
    from .config import PolymerSpec

    if args.config:
        config = RunConfig.from_yaml(args.config)
    else:
        if not getattr(args, "smiles", None):
            raise SystemExit(
                "no polymer specified. Pass --smiles (with --n-chains and "
                "--chain-length), or --config pointing at a YAML file."
            )
        config = RunConfig(polymer=PolymerSpec(
            smiles=args.smiles,
            n_chains=args.n_chains or 4,
            chain_length=args.chain_length or 10,
            counterion=args.counterion or "Cl-",
        ))
    overrides = {
        "polymer.smiles": getattr(args, "smiles", None),
        "polymer.n_chains": getattr(args, "n_chains", None),
        "polymer.chain_length": getattr(args, "chain_length", None),
        "polymer.counterion": getattr(args, "counterion", None),
        "water_model": getattr(args, "water_model", None),
        "md.temperature": getattr(args, "temperature", None),
        "md.mpi_ranks": getattr(args, "ranks", None),
        # Keep the persisted, resolved configuration honest when --workdir
        # overrides the (usually stale) value embedded in the input YAML.
        "workdir": str(args.workdir) if getattr(args, "workdir", None) is not None else None,
    }
    return config.with_overrides(**{k: v for k, v in overrides.items() if v is not None})


def _save_run_config(config: RunConfig, workdir: Path | str) -> Path:
    """Persist the fully resolved configuration before expensive work starts.

    ``run_config.yaml`` is deliberately distinct from a user's input
    ``config.yaml``: the latter may live in the work directory and must not be
    overwritten merely because defaults and command-line overrides were
    resolved.
    """
    return config.dump_yaml(Path(workdir) / "run_config.yaml")


def cmd_prepare(args) -> int:
    from .prepare import prepare_dry_membrane

    config = _load_config(args)
    _save_run_config(config, args.workdir)
    dry = prepare_dry_membrane(config, args.workdir)
    print(json.dumps(dry.summary(), indent=2))
    return 0


def _bulk_only_config(args) -> RunConfig:
    """A config good enough to describe the reservoir, with a placeholder polymer."""
    from .config import PolymerSpec

    config = RunConfig(polymer=PolymerSpec(smiles="[*]CC[*]", n_chains=1,
                                           chain_length=1))
    overrides = {"water_model": getattr(args, "water_model", None),
                 "md.temperature": getattr(args, "temperature", None),
                 "md.mpi_ranks": getattr(args, "ranks", None)}
    return config.with_overrides(**{k: v for k, v in overrides.items() if v is not None})


def cmd_bulk(args) -> int:
    from .bulk import BulkSettings, run_bulk_reference
    from .driver import bulk_n_waters

    # The bulk reference does not depend on the polymer at all, so `bulk` runs
    # without one; only the state point and water model matter.
    config = _load_config(args) if (args.config or args.smiles) else _bulk_only_config(args)
    config = config.with_overrides(**{"workdir": str(args.workdir)})
    _save_run_config(config, args.workdir)
    settings = BulkSettings(
        water_model=config.water_model, temperature=config.md.temperature,
        pressure=config.md.pressure, n_waters=bulk_n_waters(config.widom),
        cutoff=config.md.cutoff, kspace_accuracy=config.md.kspace_accuracy,
        equil_steps=config.widom.bulk_equil_steps,
        widom_steps=config.widom.n_blocks * config.widom.steps_per_block,
        insertions_per_call=config.widom.insertions_per_call,
        seed=config.widom.seed,
    )
    ref = run_bulk_reference(settings, args.workdir / "bulk",
                             cache_dir=config.widom.cache_dir,
                             ranks=config.md.mpi_ranks)
    print(json.dumps(ref.summary(), indent=2))
    return 0


def cmd_campaign(args) -> int:
    """Run the uptake loop over several morphologies and average the endpoints.

    Separate from ``run`` rather than a flag on it, because the two return
    different things: ``run`` reports one packing's saturation point, this
    reports a mean with a between-morphology error bar. Conflating them would
    make the meaning of ``result.json`` depend on a flag.
    """
    from .uptake_campaign import UptakeCampaignError, run_uptake_campaign

    config = _load_config(args)
    workdir = args.workdir
    _save_run_config(config, workdir)
    bulk_mu = getattr(args, "bulk_mu_ex", None)
    bulk_err = getattr(args, "bulk_stderr", None)
    if (bulk_mu is None) != (bulk_err is None):
        raise SystemExit("--bulk-mu-ex and --bulk-stderr must be supplied together")
    if bulk_err is not None and bulk_err < 0:
        raise SystemExit("--bulk-stderr must be non-negative")

    bulk_reference = None
    if bulk_mu is not None:
        bulk_reference = _expert_bulk_reference(config, workdir, bulk_mu, bulk_err)
        LOG.warning(
            "EXPERT OVERRIDE: using user-specified bulk mu_ex = %.4f +/- %.4f "
            "kcal/mol for every morphology; no bulk simulation will be run",
            bulk_mu, bulk_err,
        )

    try:
        campaign = run_uptake_campaign(
            config, workdir,
            n_morphologies=args.morphologies,
            bulk_reference=bulk_reference,
            resume=not args.force,
            screening=not args.production_resolution,
            parallel_morphologies=getattr(args, "parallel_morphologies", 1),
            first_morphology_seed=getattr(args, "first_morphology_seed", None),
        )
    except UptakeCampaignError as exc:
        raise SystemExit(str(exc)) from exc

    summary = campaign.summary()
    (workdir / "campaign_result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if campaign.n_usable != len(campaign.per_morphology):
        print("\nCampaign incomplete: some trajectories did not reach thermodynamic "
              "saturation; the reported mean covers only the completed subset.",
              file=sys.stderr)
        return 2
    if campaign.n_usable < 2:
        print("\nWARNING: fewer than two usable morphologies, so the uptake "
              "carries no uncertainty estimate. This is a single sample.",
              file=sys.stderr)
        return 2
    return 0


def cmd_run(args) -> int:
    from .driver import run_uptake
    from .prepare import obtain_dry_membrane

    config = _load_config(args)
    workdir = args.workdir
    resume_minimized = getattr(args, "resume_minimized", False)
    if resume_minimized:
        if args.force:
            raise SystemExit("--resume-minimized cannot be combined with --force")
        saved_path = Path(workdir) / "run_config.yaml"
        if not saved_path.is_file():
            raise SystemExit("--resume-minimized requires the previous run_config.yaml")
        saved = RunConfig.from_yaml(saved_path)
        if saved.polymer != config.polymer:
            raise SystemExit("--resume-minimized requires unchanged polymer settings")
        if (Path(workdir) / "uptake_state.json").exists():
            raise SystemExit("--resume-minimized cannot replace a membrane with uptake checkpoints")
        if (Path(workdir) / "dry/dry.data").exists():
            raise SystemExit("--resume-minimized requires an unfinished dry equilibration; "
                             "dry/dry.data already exists")
        from .prepare import _load_typed_chain
        if (not (Path(workdir) / "dry/min.data").is_file()
                or _load_typed_chain(Path(workdir) / "dry/typed_chain.pkl") is None):
            raise SystemExit("--resume-minimized requires min.data and a usable typed_chain.pkl")
    _save_run_config(config, workdir)
    bulk_mu = getattr(args, "bulk_mu_ex", None)
    bulk_err = getattr(args, "bulk_stderr", None)
    if (bulk_mu is None) != (bulk_err is None):
        raise SystemExit("--bulk-mu-ex and --bulk-stderr must be supplied together")
    if bulk_err is not None and bulk_err < 0:
        raise SystemExit("--bulk-stderr must be non-negative")

    bulk_reference = None
    if bulk_mu is not None:
        bulk_reference = _expert_bulk_reference(config, workdir, bulk_mu, bulk_err)
        LOG.warning(
            "EXPERT OVERRIDE: using user-specified bulk mu_ex = %.4f +/- %.4f "
            "kcal/mol; no bulk simulation or cache validation will be performed",
            bulk_mu, bulk_err,
        )
    if resume_minimized:
        from .prepare import prepare_dry_membrane

        dry = prepare_dry_membrane(config, workdir, resume_minimized=True)
        typed_chains = dry.typed_chains
    else:
        typed_chains, _reused = obtain_dry_membrane(
            config, workdir, resume=not args.force)

    result = run_uptake(
        config, workdir, typed_chains, bulk_reference=bulk_reference,
        resume=not args.force,
    )

    result.to_dataframe().to_csv(workdir / "uptake_trajectory.csv", index=False)
    (workdir / "result.json").write_text(json.dumps(result.summary(), indent=2))
    print(json.dumps(result.summary(), indent=2))
    if not result.converged:
        print("\nWARNING: the loop did not reach saturation. The reported uptake "
              "is a lower bound.", file=sys.stderr)
        return 2
    return 0


def cmd_diagnostics(args) -> int:
    """Regenerate FEP diagnostic figures from completed checkpoints."""
    from .fep.campaign import combine_morphologies
    from .fep.diagnostics import write_campaign_figures
    from .fep.resume import load_morphology
    from .uptake_diagnostics import plot_run

    workdir = Path(args.workdir)
    config_path = workdir / "run_config.yaml"
    if not config_path.is_file():
        raise SystemExit(
            f"no saved configuration at {config_path}; pass the top-level run "
            "directory containing run_config.yaml"
        )
    config = RunConfig.from_yaml(config_path)

    # A single-run uptake iteration is ``iter_NNN/fep/morph00``; bulk and
    # multi-morphology campaigns use the same morphology checkpoint one level
    # below their campaign directory. Grouping by that parent makes this
    # command work for all of them without reconstructing raw estimator files.
    written: list[Path] = []
    try:
        written.append(plot_run(
            workdir, workdir / "uptake_analysis.png", args.water_mu,
        ))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        LOG.info("skipping uptake_analysis.png: %s", exc)

    groups: dict[Path, list] = {}
    for checkpoint in sorted(workdir.rglob("morphology.json")):
        morphology = load_morphology(checkpoint)
        if morphology is not None:
            groups.setdefault(checkpoint.parent.parent, []).append(morphology)

    if not groups and not written:
        raise SystemExit(
            f"no plottable uptake trajectory or completed FEP morphology "
            f"checkpoints found below {workdir}"
        )

    for campaign_dir, morphologies in groups.items():
        estimate = combine_morphologies(
            morphologies, config.md.temperature,
            max_stderr=config.fep.max_stderr,
        )
        written.extend(write_campaign_figures(
            estimate, campaign_dir, min_overlap=config.fep.min_overlap,
        ))

    if not written:
        raise SystemExit(
            "completed checkpoints were found, but they contain no plottable "
            "FEP diagnostics"
        )
    for path in written:
        print(path)
    return 0


def _expert_bulk_reference(config, workdir: Path, mu_ex: float, stderr: float):
    """Construct an explicitly trusted bulk reference supplied by the user."""
    import math

    import numpy as np

    from .bulk import BulkReference, BulkSettings, LITERATURE_DENSITY
    from .driver import bulk_n_waters
    from .widom import KB_KCAL, MIN_EFFECTIVE_SAMPLES, WidomEstimate

    settings = BulkSettings(
        water_model=config.water_model,
        temperature=config.md.temperature,
        pressure=config.md.pressure,
        n_waters=bulk_n_waters(config.widom),
        cutoff=config.md.cutoff,
        kspace_accuracy=config.md.kspace_accuracy,
        equil_steps=config.widom.bulk_equil_steps,
        widom_steps=config.widom.n_blocks * config.widom.steps_per_block,
        insertions_per_call=config.widom.insertions_per_call,
        seed=config.widom.seed,
    )
    n_blocks = max(3, config.widom.n_blocks)
    estimate = WidomEstimate(
        mu_ex=float(mu_ex),
        stderr=float(stderr),
        temperature=config.md.temperature,
        n_blocks=n_blocks,
        block_values=np.full(n_blocks, float(mu_ex)),
        mean_boltzmann=math.exp(-float(mu_ex) / (KB_KCAL * config.md.temperature)),
        effective_samples=float(max(MIN_EFFECTIVE_SAMPLES, n_blocks)),
        volume=config.widom.bulk_box_length ** 3,
    )
    density = LITERATURE_DENSITY.get(config.water_model.lower(), float("nan"))
    return BulkReference(settings, estimate, density, estimate.volume,
                         Path(workdir) / "bulk_override")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aemwater",
        description="Maximum water uptake of anion exchange membranes by MD.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="build and equilibrate the dry membrane")
    _add_common(p); _add_polymer(p)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("bulk", help="compute the bulk water reference")
    _add_common(p); _add_polymer(p)
    p.set_defaults(func=cmd_bulk)

    p = sub.add_parser("run", help="load water to saturation")
    _add_common(p); _add_polymer(p)
    p.add_argument("--resume-minimized", action="store_true",
                   help="reuse dry/min.data and typed chain; repeat equilibration only")
    p.add_argument("--force", action="store_true",
                   help="rebuild the dry membrane and restart the loop")
    p.add_argument("--bulk-mu-ex", type=float, metavar="KCAL_PER_MOL",
                   help="expert override for bulk-water excess chemical potential; "
                        "requires --bulk-stderr and bypasses the bulk simulation")
    p.add_argument("--bulk-stderr", type=float, metavar="KCAL_PER_MOL",
                   help="uncertainty for --bulk-mu-ex")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser(
        "diagnostics", help="plot diagnostics from a completed FEP run")
    p.add_argument(
        "--workdir", type=Path, required=True,
        help="top-level run directory containing run_config.yaml",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--water-mu", type=float,
        help="bulk-water excess chemical potential for an unfinished run",
    )
    p.set_defaults(func=cmd_diagnostics)

    p = sub.add_parser(
        "campaign",
        help="uptake averaged over independently equilibrated morphologies")
    _add_common(p); _add_polymer(p)
    p.add_argument("--morphologies", type=int, default=None, metavar="M",
                   help="independent packings to run (default: fep.n_morphologies). "
                        "M=1 gives no error bar and is for smoke tests only")
    p.add_argument("--production-resolution", action="store_true",
                   help="run every iteration at full FEP resolution instead of "
                        "the screening ladder; roughly 6.4x the cost per "
                        "morphology")
    p.add_argument("--parallel-morphologies", type=int, default=1, metavar="N",
                   help="maximum simultaneous uptake trajectories (default: 1)")
    p.add_argument("--first-morphology-seed", type=int, default=None,
                   help="original packing seed when reusing a saved membrane in morph00/dry")
    p.add_argument("--force", action="store_true",
                   help="rebuild every morphology and restart its loop")
    p.add_argument("--bulk-mu-ex", type=float, metavar="KCAL_PER_MOL",
                   help="expert override for bulk-water excess chemical potential; "
                        "requires --bulk-stderr and bypasses the bulk simulation")
    p.add_argument("--bulk-stderr", type=float, metavar="KCAL_PER_MOL",
                   help="uncertainty for --bulk-mu-ex")
    p.set_defaults(func=cmd_campaign)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("interrupted; the loop checkpoints each iteration and "
              "'aemwater run' will resume", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
