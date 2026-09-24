"""Save self-contained step-21 states and seed hydration without repacking."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


def snapshot_offsets(config) -> list[int]:
    """Step-21-relative steps, including both ends of the requested window.

    One seed uses the final state. Integer rounding is at most half a timestep.
    """
    spec = config.equilibration
    if not spec.snapshot_count:
        return []
    end = round(spec.final_npt_ps * 1000 / config.md.timestep)
    window = round(spec.snapshot_window_ps * 1000 / config.md.timestep)
    if spec.scheme != "21step" or window <= 0 or window >= end:
        raise ValueError("snapshot window must fit strictly inside step 21")
    count = spec.snapshot_count
    offsets = ([end] if count == 1 else
               [end - window + round(i * window / (count - 1)) for i in range(count)])
    if len(set(offsets)) != count:
        raise ValueError("snapshot spacing is shorter than one MD timestep")
    return offsets


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_snapshot_seeds(config, workdir, count, *, resume=True, packing_seed=None):
    """Prepare one dry cell, validate provenance, then copy snapshots to branches.

    Never import old independent campaigns implicitly or reconstruct missing
    historical snapshots from a final state. A new directory is required when
    changing an established campaign's preparation settings.
    """
    from .prepare import obtain_dry_membrane, TYPED_CHAIN_CHECKPOINT
    from .uptake_campaign import UptakeCampaignError

    workdir = Path(workdir)
    window = config.equilibration.snapshot_window_ps
    config = config.with_overrides(**{
        "equilibration.snapshot_count": count,
        "equilibration.final_npt_ps": max(config.equilibration.final_npt_ps, window + 1000),
        "box.seed": config.box.seed if packing_seed is None else packing_seed,
    })
    offsets = snapshot_offsets(config)
    shared = workdir / "shared_dry"
    manifest = workdir / "dry_snapshot_manifest.json"
    # YAML is the canonical, validated configuration already used by the CLI.
    shared.mkdir(parents=True, exist_ok=True)
    candidate = shared / "requested_config.yaml"
    config.dump_yaml(candidate)
    signature = _digest(candidate)
    request = {"config_sha256": signature, "step21_offsets": offsets,
               "window_ps": window, "packing_seed": config.box.seed}
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous["request"] != request or not resume:
            raise UptakeCampaignError("snapshot campaign settings changed or --force requested; use a new workdir")
        for row in previous.get("snapshots", []):
            source = shared / "dry" / row["file"]
            if not source.exists() or _digest(source) != row["sha256"]:
                raise UptakeCampaignError("dry snapshot is missing or changed; use a new workdir")
    else:
        if any(workdir.glob("morph*")) or (shared / "dry").exists():
            raise UptakeCampaignError("existing campaign has no snapshot provenance; use a new workdir")
        manifest.write_text(json.dumps({"request": request}, indent=2))
    config.dump_yaml(shared / "run_config.yaml")
    chains, _ = obtain_dry_membrane(config, shared, resume=resume)
    dry = shared / "dry"
    convergence = dry / "convergence.json"
    if not convergence.exists() or not json.loads(convergence.read_text()).get("converged"):
        raise UptakeCampaignError("shared dry morphology has not passed convergence; no hydration seeds released")
    rows = []
    for index, offset in enumerate(offsets):
        source = dry / f"snapshot_{index:03d}.data"
        if not source.exists():
            raise UptakeCampaignError("step-21 snapshots missing; use a new workdir to equilibrate with snapshot output")
        rows.append({"file": source.name, "sha256": _digest(source),
                     "step21_step": offset, "step21_ps": offset * config.md.timestep / 1000})
    previous = json.loads(manifest.read_text())
    if "snapshots" in previous and previous["snapshots"] != rows:
        raise UptakeCampaignError("snapshot provenance changed; use a new workdir")
    manifest.write_text(json.dumps({"request": request, "snapshots": rows}, indent=2))
    for index, row in enumerate(rows):
        target = workdir / f"morph{index:02d}" / "dry"
        target.mkdir(parents=True, exist_ok=True)
        dest = target / "dry.data"
        if dest.exists() and _digest(dest) != row["sha256"]:
            raise UptakeCampaignError(f"hydration seed changed in {target}; use a new workdir")
        shutil.copy2(dry / row["file"], dest)
        shutil.copy2(dry / TYPED_CHAIN_CHECKPOINT, target / TYPED_CHAIN_CHECKPOINT)
        (target / "snapshot_source.json").write_text(json.dumps(row, indent=2))
    return config, chains
