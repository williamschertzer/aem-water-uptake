"""Refresh intermediate plots without modifying either simulation directory."""
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aem-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
diag, dax = plt.subplots(1, 2, figsize=(11, 4))
summary = {"snapshot_utc": datetime.now(timezone.utc).isoformat(), "runs": {}}
rows = []
for name, label, color in [
    ("im_peek_saturation", "200 ps relaxation", "#007c91"),
    ("im_peek_saturation_longer_relax", "400 ps relaxation", "#c04470"),
]:
    run = ROOT / "runs" / name
    state = json.loads((run / "uptake_state.json").read_text())
    data = state["iterations"]
    x = [r["n_waters_after"] for r in data]
    mu = [r["mu_ex"] for r in data]
    ref = data[0]["mu_ex"] - data[0]["mu_gap"]
    axes[0, 0].errorbar(x, mu, yerr=[r["mu_ex_stderr"] for r in data],
                        color=color, marker="o", capsize=3, label=label)
    bad = [r for r in data if not r["sampling_adequate"]]
    axes[0, 0].scatter([r["n_waters_after"] for r in bad],
                        [r["mu_ex"] for r in bad], marker="x", s=110,
                        color="black", zorder=5)
    axes[0, 1].plot([r["index"] for r in data],
                    [r["water_uptake_pct"] for r in data], "o-", color=color, label=label)
    axes[1, 0].plot(x, [r["density"] for r in data], "o-", color=color, label=label)
    axes[1, 1].plot(x, [100 * (r["volume"] / data[0]["volume"] - 1) for r in data],
                    "o-", color=color, label=label)
    checks = []
    for r in data:
        rows.append({"run": name, **r})
        path = run / f"iter_{r['index']:03d}/fep/morph00/morphology.json"
        morphology = json.loads(path.read_text())
        for leg, value in morphology["legs"].items():
            d = value["diagnostics"]
            checks.append({"iteration": r["index"], "leg": leg,
                           "min_samples": min(d["N_k"]),
                           "min_overlap": min(d["neighbour_overlap"])})
    for leg, style in [("lj", "o-"), ("coul", "s--")]:
        points = [c for c in checks if c["leg"] == leg]
        for ax, key in zip(dax, ["min_overlap", "min_samples"]):
            ax.plot([c["iteration"] for c in points], [c[key] for c in points],
                    style, color=color, label=f"{label}, {leg}")
    pending = run / f"iter_{state['next_step']:03d}"
    progress = {}
    for leg in ["lj", "coul"]:
        steps = []
        for p in sorted((pending / "fep/morph00" / leg).glob("lam_*/pe.dat")):
            # Ignore incomplete trailing writes in a running simulation.
            lines = p.read_bytes().split(b"\n")[:-1]
            values = [line.split() for line in lines if line.strip() and not line.startswith(b"#")]
            if values:
                steps.append(int(values[-1][0]))
        if steps:
            progress[leg] = {"windows": len(steps), "last_step_range": [min(steps), max(steps)]}
    summary["runs"][name] = {"checkpoint": state, "inferred_bulk_mu": ref,
                             "quality_checks": checks, "pending_fep": progress}

axes[0, 0].axhline(ref, color="0.4", linestyle="--", label=f"Recorded reference: {ref:g}")
axes[0, 0].set(xlabel="Water molecules", ylabel="Excess chemical potential (kcal/mol)",
               title="Error bars: within-cell SE; x: sampling check failed")
axes[0, 1].set(xlabel="Completed iteration", ylabel="Water uptake (wt %)", title="Imposed insertion sequence")
axes[1, 0].set(xlabel="Water molecules", ylabel="Density (g/cm3)", title="Recorded density")
axes[1, 1].set(xlabel="Water molecules", ylabel="Volume change from dry checkpoint (%)",
               title="Descriptive volume change, not equilibrium swelling")
for ax, threshold, title in zip(dax, [.03, 50], ["Minimum neighboring overlap", "Minimum decorrelated samples"]):
    ax.axhline(threshold, color="0.4", linestyle=":", label=f"Required: {threshold:g}")
    ax.set(xlabel="Completed iteration", ylabel=title, yscale="log")
for ax in [*axes.flat, *dax]:
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
for f, stem in [(fig, "comparison"), (diag, "sampling")]:
    f.tight_layout()
    f.savefig(OUT / f"{stem}.png", dpi=180)
    f.savefig(OUT / f"{stem}.pdf")
    plt.close(f)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with (OUT / "trajectory.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"snapshot_utc": summary["snapshot_utc"], "pending_fep": {
    name: value["pending_fep"] for name, value in summary["runs"].items()}}, indent=2))
