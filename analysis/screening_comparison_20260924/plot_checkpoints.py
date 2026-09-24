import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]
out=root/'analysis/screening_comparison_20260924'
paths=[root/'runs/im_peek_small_screening',*sorted((root/'runs/im_peek_small_screening_9_23').glob('morph*'))]
fig,ax=plt.subplots(figsize=(8,5)); captured={}
for p in paths:
 state=json.loads((p/'uptake_state.json').read_text()); captured[p.name]=state
 rows=state['iterations']; label='original' if p==paths[0] else '9/23 '+p.name
 ax.errorbar([r['n_waters_after'] for r in rows],[r['mu_ex'] for r in rows],yerr=[r['mu_ex_stderr'] for r in rows],marker='o',label=label,capsize=3)
 for r in rows:
  if not r['sampling_adequate']: ax.scatter(r['n_waters_after'],r['mu_ex'],s=100,facecolors='none',edgecolors='red',zorder=5)
ax.axhline(-6.8,color='k',ls='--',label='9/23 bulk −6.8'); ax.axhline(-6.5,color='gray',ls=':',label='original bulk −6.5')
ax.set(xlabel='Water molecules',ylabel='Excess chemical potential (kcal/mol)',title='Checkpoint data captured 2026-09-24\nRed rings: sampling flagged inadequate')
ax.legend(fontsize=8,ncol=2); fig.tight_layout(); fig.savefig(out/'chemical_potential.png',dpi=180)
(out/'checkpoint_capture.json').write_text(json.dumps(captured,indent=2))
