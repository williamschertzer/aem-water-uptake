"""Resume support for FEP campaigns.

A bulk validation campaign is ``n_morphologies`` cells x 2 legs x (8 + 5)
lambda states, each an independent ``lmp`` invocation, plus a rerun pass of the
same order again. The campaign already puts every unit of work in its own
directory; what was missing was any decision about whether a directory is
*finished*, so the whole thing lived in memory until the final JSON and any exit
-- a queue walltime kill, a closed laptop, ^C -- discarded every window that had
already been paid for.

**Completeness is read from what LAMMPS wrote, not from a marker this package
controls.** A marker of our own can outlive the thing it describes: killed
between the run and the marker write, or left behind when a later change alters
what the directory ought to contain. LAMMPS emits its terminal wall-time line
only on a clean exit, so that line is evidence. The rule here is that line plus
the presence of every output file the next stage will read, each non-empty.

What this module deliberately does *not* do is resume a partially sampled state.
LAMMPS can restart mid-trajectory, but a state whose production run was cut in
half has a shorter correlated trace than the ladder's other states, and the
estimators weight states by their effective sample count. Stitching a restart
onto a killed trace would quietly change that weighting. An incomplete state is
therefore rerun from its start; the granularity of the checkpoint is one lambda
window, which on the validation settings is minutes, not hours.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from ..utils import write_json

LOG = logging.getLogger(__name__)

#: Files a finished sampling state must have left behind. ``fep.dat`` feeds TI,
#: ``pe.dat`` the rerun diagonal check, ``traj.lammpstrj`` the rerun itself.
STATE_OUTPUTS = ("fep.dat", "pe.dat", "traj.lammpstrj")

#: LAMMPS writes this once, last, on a clean exit.
_CLEAN_EXIT = "Total wall time:"


def _log_is_clean(log_file: Path) -> bool:
    """Whether LAMMPS reached its own normal termination.

    Only the tail is read: these logs carry a thermo line per sampling interval
    and there is no reason to pull megabytes into memory to look at the last
    line of it.
    """
    if not log_file.is_file():
        return False
    try:
        with log_file.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 4096))
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    return _CLEAN_EXIT in tail


def state_complete(
    sdir: Path | str,
    *,
    log_name: str = "state.log",
    outputs: tuple[str, ...] = STATE_OUTPUTS,
) -> bool:
    """Whether one lambda window finished and left everything downstream needs."""
    sdir = Path(sdir)
    if not _log_is_clean(sdir / log_name):
        return False
    return all(
        (sdir / name).is_file() and (sdir / name).stat().st_size > 0
        for name in outputs
    )


def rerun_complete(rerun_dir: Path | str, index: int) -> bool:
    """Whether the rerun pass for source state ``index`` finished.

    The rerun's own diagonal check re-validates the numbers against the sampling
    run's ``pe.dat`` every time the matrix is built, so a reused ``rerun_j.dat``
    is checked rather than trusted.
    """
    rerun_dir = Path(rerun_dir)
    out = rerun_dir / f"rerun_{index}.dat"
    return (
        _log_is_clean(rerun_dir / f"rerun_{index}.log")
        and out.is_file()
        and out.stat().st_size > 0
    )


def campaign_stamp(config, *, kind: str, **extra) -> dict:
    """Identity of the calculation a run directory holds.

    Resuming into a directory written under different settings would average
    windows from two different calculations into one free energy, which no
    downstream diagnostic would flag: every individual window is internally
    consistent.

    The fields are spelled out here rather than delegated to
    :func:`aemwater.fep.campaign.fep_cache_key`, which needs a ``BulkSettings``
    the campaign functions do not have -- and which the membrane campaign has no
    analogue of at all. ``extra`` carries whatever else identifies the specific
    call, such as ``n_waters`` for a bulk box.
    """
    import hashlib

    spec = config.fep
    payload = {
        "kind": kind,
        "water_model": str(config.water_model),
        "temperature": float(config.md.temperature),
        "cutoff": float(config.md.cutoff),
        "timestep": float(config.md.timestep),
        "lj_lambdas": [float(x) for x in spec.lj_lambdas],
        "coul_lambdas": [float(x) for x in spec.coul_lambdas],
        "equil_steps": int(spec.equil_steps),
        "production_steps": int(spec.production_steps),
        "sample_every": int(spec.sample_every),
        "n_morphologies": int(spec.n_morphologies),
        "estimators": sorted(spec.estimators),
        "soft_core_n": int(spec.soft_core_n),
        "alpha_lj": float(spec.alpha_lj),
        "alpha_coul": float(spec.alpha_coul),
        "ti_delta": float(spec.ti_delta),
        "kspace_accuracy": float(spec.kspace_accuracy),
        "seed": int(spec.seed),
        **extra,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    payload["digest"] = hashlib.sha256(blob).hexdigest()[:16]
    return payload


def cell_fingerprint(system) -> str:
    """Identify the specific configuration a membrane morphology sampled.

    The bulk campaign builds its own cells from a seed, so its stamp pins them.
    ``run_membrane_campaign`` takes cells from the caller, and a resumed run has
    no way to know whether the cell now in hand is the one whose windows are on
    disk -- an uptake loop one iteration further along supplies a *different*
    cell with the same atom count and the same type table. Reusing windows across
    that boundary would sum legs sampled in two different configurations into one
    mu_ex, and every internal check would pass.

    Coordinates are rounded before hashing so that a re-read of the same data
    file fingerprints identically despite the text format's finite precision.
    """
    import hashlib

    import numpy as np

    coords = np.asarray(system.structure.coordinates, dtype=float)
    box = np.asarray(system.structure.box, dtype=float)
    digest = hashlib.sha256()
    digest.update(np.round(coords, 3).tobytes())
    digest.update(np.round(box, 4).tobytes())
    digest.update(str(len(system.atom_types)).encode())
    return digest.hexdigest()[:16]


class StampMismatch(RuntimeError):
    """Raised when a run directory was written by a different calculation."""


def check_stamp(path: Path | str, stamp: dict, *, resume: bool) -> None:
    """Compare ``stamp`` against the one on disk, writing it if absent.

    With ``resume=False`` the stamp is overwritten: the caller has asked to
    start over, and refusing on a mismatch would make ``--no-resume`` unusable
    in exactly the case it is for.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file() and resume:
        try:
            saved = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise StampMismatch(
                f"{path} exists but is unreadable ({exc}); it cannot be checked "
                "against the requested settings. Move the run directory aside "
                "or pass resume=False to start over."
            ) from exc
        differing = {
            key: (saved.get(key), value)
            for key, value in stamp.items()
            if saved.get(key) != value
        }
        if differing:
            detail = "; ".join(
                f"{key}: on disk {was!r}, requested {now!r}"
                for key, (was, now) in sorted(differing.items())
            )
            raise StampMismatch(
                f"{path.parent} holds a different calculation -- {detail}. "
                "Resuming would average windows from two calculations into one "
                "free energy, which no downstream check would catch. Use a new "
                "workdir, or pass resume=False to discard what is there."
            )
        return

    write_json(path, stamp)


def save_morphology(estimate, path: Path | str) -> Path:
    """Checkpoint one morphology's finished result.

    Written once a morphology's legs are both estimated, so a resumed campaign
    skips its ~13 LAMMPS runs entirely rather than re-deriving a number that is
    already known.
    """
    path = Path(path)
    payload = asdict(estimate)
    # LegEstimate.leg is a FEPLeg (a str Enum), which asdict leaves as the enum;
    # json would serialise it via its str base and reload it as a bare string,
    # so it is normalised here and restored by load_morphology.
    for leg in payload.get("legs", {}).values():
        leg["leg"] = str(leg["leg"].value if hasattr(leg["leg"], "value") else leg["leg"])
    return write_json(path, payload)


def load_morphology(path: Path | str):
    """Reload a checkpointed morphology, or ``None`` if it is unusable.

    A truncated or stale checkpoint returns ``None`` rather than raising: the
    work it stands for can always be redone, and refusing to start is a worse
    outcome than repeating a morphology.
    """
    from .campaign import MorphologyEstimate
    from .estimators import LegEstimate
    from .schedule import FEPLeg

    path = Path(path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
        legs = {
            name: LegEstimate(
                estimator=row["estimator"],
                leg=FEPLeg(row["leg"]),
                delta_f=float(row["delta_f"]),
                stderr=float(row["stderr"]),
                n_effective=float(row["n_effective"]),
                diagnostics=row.get("diagnostics", {}),
            )
            for name, row in payload.get("legs", {}).items()
        }
        estimate = MorphologyEstimate(
            index=int(payload["index"]),
            mu_ex=float(payload["mu_ex"]),
            stderr=float(payload["stderr"]),
            legs=legs,
            workdir=payload.get("workdir", ""),
            diagnostics=payload.get("diagnostics", {}),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        LOG.warning("ignoring unusable morphology checkpoint %s: %s", path, exc)
        return None

    if not estimate.usable:
        LOG.warning(
            "morphology checkpoint %s holds a non-finite estimate; rerunning", path
        )
        return None
    return estimate


__all__ = [
    "STATE_OUTPUTS",
    "StampMismatch",
    "campaign_stamp",
    "cell_fingerprint",
    "check_stamp",
    "load_morphology",
    "rerun_complete",
    "save_morphology",
    "state_complete",
]
