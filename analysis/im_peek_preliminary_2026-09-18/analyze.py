"""Read-only analysis of the two preliminary runs; outputs are written here."""
from pathlib import Path
import json, os
from datetime import datetime, timezone
os.environ.setdefault('MPLCONFIGDIR', '/tmp/aem-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out = Path(__file__).resolve().parent
root = out.parents[1] / 'runs'
fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
summary = {'snapshot_utc': datetime.now(timezone.utc).isoformat(), 'runs': {}}
for name, label in [('im_peek_saturation','200 ps'), ('im_peek_saturation_longer_relax','400 ps')]:
    p = root / name
    state = json.loads((p/'uptake_state.json').read_text())
    d = np.loadtxt(p/'iter_001/density.dat')
    # NVE settling takes 20 ps; minimization adds a small step offset.
    axes[0].plot(d[:,0]/1000, d[:,1], 'o-', label=label)
    steps = [int(f.read_text().strip().splitlines()[-1].split()[0]) for f in sorted((p/'iter_001/fep/morph00/lj').glob('lam_*/pe.dat'))]
    summary['runs'][name] = {'completed_iterations':len(state['iterations']), 'dry_mu_ex':state['iterations'][0]['mu_ex'], 'inferred_bulk_mu_ex':state['iterations'][0]['mu_ex']-state['iterations'][0]['mu_gap'], 'hydrated_mean_density':float(d[:,1].mean()), 'hydrated_mean_volume':float(d[:,2].mean()), 'lj_production_step_range':[min(steps),max(steps)], 'hydrated_result_exists':(p/'iter_001/fep_membrane.json').exists()}
m = json.loads((root/'im_peek_saturation/iter_000/fep/morph00/morphology.json').read_text())
for leg, val in m['legs'].items():
    d = val['diagnostics']; lam = np.array(d['lambdas'])
    axes[1].plot((lam[:-1]+lam[1:])/2, d['neighbour_overlap'], 'o-', label=leg)
    axes[2].plot(lam, d['N_k'], 'o-', label=leg)
axes[0].set(xlabel='LAMMPS timestep / 1000 (ps)', ylabel='Density (g/cm³)', title='First hydration: recorded block means')
axes[1].set(xlabel='Midpoint of neighboring coupling windows', ylabel='Overlap', title='Shared dry-state FEP')
axes[1].axhline(.03, color='black', linestyle='--', linewidth=1, label='Required: 0.03')
axes[2].set(xlabel='FEP coupling parameter', ylabel='Retained decorrelated samples', title='Shared dry-state sampling')
axes[2].axhline(50, color='black', linestyle='--', linewidth=1, label='Required: 50')
for ax in axes:
    ax.legend(fontsize=8); ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(out/'comparison.png', dpi=180)
(out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
