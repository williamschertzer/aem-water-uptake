#!/usr/bin/env python3
"""Compare matched independent trajectories after changing sampling settings.

A pass means the paired 95% interval lies inside the requested uptake tolerance.
It is evidence of sensitivity convergence, not proof of force-field accuracy.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from aemwater.fep.campaign import t95


def compare(reference, candidate, tolerance=1.0):
    def endpoints(folder):
        path = Path(folder) / "morphology_results.json"
        if not path.exists():
            raise ValueError(f"missing completed campaign record: {path}")
        rows = json.loads(path.read_text())
        if len(rows) < 3 or any(not r["usable"] for r in rows):
            raise ValueError(f"{folder}: need at least three trajectories, all thermodynamically saturated")
        values = {r["seed"]: r["water_uptake_wt_pct"] for r in rows}
        if len(values) != len(rows) or not all(math.isfinite(x) for x in values.values()):
            raise ValueError(f"{folder}: duplicate seeds or non-finite uptake")
        return values

    a, b = endpoints(reference), endpoints(candidate)
    if a.keys() != b.keys():
        raise ValueError("campaigns must use the same set of independent packing seeds")
    delta = np.array([b[seed] - a[seed] for seed in sorted(a)])
    mean = float(delta.mean())
    half = t95(len(delta) - 1) * float(delta.std(ddof=1)) / math.sqrt(len(delta))
    return {"reference": str(reference), "candidate": str(candidate),
            "n_pairs": len(delta), "paired_differences_wt_pct": delta.tolist(),
            "mean_change_wt_pct": mean, "ci95_change_wt_pct": [mean-half, mean+half],
            "tolerance_wt_pct": tolerance,
            "passed": abs(mean) + half <= tolerance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    parser.add_argument("--tolerance", type=float, default=1.0,
                        help="allowed absolute change in uptake percentage points (default: 1)")
    args = parser.parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("tolerance must be finite and positive")
    try:
        reports = [compare(args.reference, p, args.tolerance) for p in args.candidates]
    except ValueError as exc:
        parser.exit(2, f"Not assessable: {exc}\n")
    print(json.dumps(reports, indent=2, allow_nan=False))
    return 0 if all(r["passed"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
