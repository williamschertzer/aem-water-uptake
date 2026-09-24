# Im-PEEK saturation and sampling validation

These are starting settings for a convergence study, not validated accuracy claims.
Run from the repository root with the environment containing the patched package.
No expert bulk override is used: each campaign computes (or reuses an exactly
keyed cache of) a matched SPC/E FEP reference. An unconverged reference stops the
campaign before membrane preparation; improve bulk sampling/replication first.

## Independent trajectories

```bash
aemwater campaign --config examples/im_peek_saturation.yaml \
  --workdir runs/im_peek_saturation --production-resolution
```

This runs three independent packings, 11 windows per leg, 50 ps FEP equilibration,
500 ps production per window, and 200 ps swelling relaxation per insertion cycle.
The maximum batch is 10 waters, with up to 60 insertion cycles. Chemical-potential
precision is limited to a 0.3 kcal/mol 95% half-width; local membrane decisions use
1.96 times the within-cell standard error, while the replicated bulk uses Student-t.
Each local window must retain at least 50 decorrelated samples and neighboring
window overlap must be at least 0.03. These diagnostics are necessary checks,
not proof that all slow modes have equilibrated.

The local crossing can stop one trajectory without claiming that its FEP estimate
has measured between-cell uncertainty. Independent trajectory endpoints supply
that uncertainty. A single run is not a replicated saturation prediction.

Partial insertion batches count as progress and reduce the next requested batch.
Three consecutive zero-insertion attempts stop with `insertion_stalled` and
`converged: false`. Stalled and budget-exhausted trajectories are excluded from
endpoint averages and retained in `morphology_results.json`. A mean of a completed
subset can be biased by missing endpoints; do not report it as the full campaign.

## Sensitivity checks

Run each variant in its own directory. Packing seeds match the baseline so endpoint
changes can be compared in pairs; different packings within each campaign remain
independent. These commands perform whole trajectories, which also tests the
influence of early hydration history rather than only the final snapshot.

```bash
aemwater campaign --config examples/im_peek_saturation_longer_fep.yaml \
  --workdir runs/im_peek_saturation_longer_fep --production-resolution
aemwater campaign --config examples/im_peek_saturation_longer_relax.yaml \
  --workdir runs/im_peek_saturation_longer_relax --production-resolution
aemwater campaign --config examples/im_peek_saturation_smaller_batches.yaml \
  --workdir runs/im_peek_saturation_smaller_batches --production-resolution
```

The variants respectively double FEP equilibration/production, double swelling
relaxation, and halve batch sizes (doubling the iteration budget). They retain
all other configuration settings and use matching bulk-reference settings.

```bash
python scripts/check_uptake_convergence.py runs/im_peek_saturation \
  runs/im_peek_saturation_longer_fep \
  runs/im_peek_saturation_longer_relax \
  runs/im_peek_saturation_smaller_batches --tolerance 1.0
```

A comparison passes only when all trajectories reach thermodynamic saturation,
at least three paired seeds are available, and the entire paired Student-t 95%
interval is within +/-1.0 uptake percentage point. Choose that tolerance based on
the precision your application needs. A wide interval is inconclusive, not proof
of equivalence: add independent trajectories or improve sampling. Three is a
starting replication count, not a guarantee of a tight interval. Checks share the
baseline and are exploratory, not a simultaneous-confidence certification.

Inspect per-window FEP diagnostics (including estimator disagreement), the
hydration-dependent chemical potential, and equilibration histories alongside
this endpoint check. Passing does not test force-field accuracy, cell-size effects,
or the excess-chemical-potential model's relation to experiment. Do not reuse old
uptake checkpoints for this changed stopping protocol; start fresh directories.
