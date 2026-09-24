# Preliminary Im-PEEK comparison

Snapshot: 2026-09-18, approximately 20:19 UTC. These are changing run directories; the accompanying summary.json records the observed FEP progress. No simulation inputs or checkpoints were modified.

Neither run has reached saturation or completed its first hydrated chemical-potential measurement. Both have completed only iter_000 (the dry baseline). iter_001 contains five inserted waters and completed NPT relaxation; its FEP remains incomplete. The checkpoint's zero waters describes the last completed iteration, not the current hydrated structure.

| Quantity | im_peek_saturation | im_peek_saturation_longer_relax |
|---|---:|---:|
| NPT relaxation | 200 ps | 400 ps |
| Waters in first hydrated structure | 5 | 5 |
| Waters per ionic group | 0.10 | 0.10 |
| Water/dry mass × 100 | 0.447% | 0.447% |
| Recorded hydrated density mean (g/cm³) | 1.173866 | 1.164212 |
| Recorded hydrated volume mean (Å³) | 28662.64 | 28900.18 |
| Completed dry μ_ex (kcal/mol) | −15.39809 | −15.39809 |
| Within-cell standard error (kcal/mol) | 0.05545 | 0.05545 |
| Saved gap to bulk (kcal/mol) | −8.89809 | −8.89809 |

Density/volume means use the five blocks in each iter_001/density.dat. The blocks span different durations: the final 100 ps versus final 200 ps of NPT. Uptake uses five waters, 50 ionic groups and dry molar mass 20172.045 from dry/composition.json; the 15 water atoms are recorded in iter_001/iter.log. This loading is imposed by the insertion cycle, not an equilibrium uptake prediction.

## What the results support

The dry membrane favors insertion relative to the reference used by the driver. The free energy is dominated by electrostatics: Coulomb −15.56625 and Lennard–Jones +0.16816 kcal/mol. This is evidence for favorable initial hydration within this calculation, not an estimate of saturation capacity.

The two dry.data files are byte-identical, as are the two iter_001/start.data files. Configuration differences are only relaxation length and work directory; all seeds match. Consequently, identical dry free energies are expected and provide no independent replication or test of relaxation sensitivity. The calculation of iter_000 copies the dry structure without applying the insertion relaxation.

Longer relaxation produces a 0.009654 g/cm³ lower recorded mean density (0.822%) and a 237.54 Å³ larger mean volume (0.829%). Relative to the dry checkpoint volume of 28845.73 Å³, the recorded hydrated means correspond to −0.635% and +0.189%. These comparisons mix a dry endpoint with hydrated time averages and are descriptive only. They do not establish equilibrium swelling. With only one paired trajectory and five blocks per run, there is no defensible confidence interval for the relaxation effect. Both density records still change over their recorded intervals.

## Sampling and provenance limitations

- Dry LJ overlap is comfortable: minimum 0.1311 against a required 0.03; minimum retained samples 817 against 50.
- The dry Coulomb 0.9–1.0 pair has overlap 0.03239, only about 8% above the required threshold. The fully coupled window retains only 67 decorrelated samples, with statistical inefficiency 14.95. This is the main observed sampling bottleneck.
- The local 95% half-width is approximately 0.109 kcal/mol. It excludes unmeasured between-cell variability. fep_membrane.json explicitly reports one morphology, converged=false and an unbounded combined interval; sampling_adequate=true in uptake_state.json refers to the local decision checks and is not contradictory.
- Despite n_morphologies: 3 in the configuration, these directories contain only one morphology per membrane estimate and no completed independent-trajectory campaign. Three independent endpoints are not available.
- The saved gap implies μ_bulk = −6.50000 kcal/mol. No matched bulk calculation or reference uncertainty is present in these run directories, and the configured im_peek_saturation_bulk_cache directory was not found. The value is consistent with an explicit reference override, but its origin cannot be proved from these artifacts. Verify the launch command/reference provenance before treating the gap as calibrated to matched SPC/E FEP. The repository's saturation-study documentation calls for a matched bulk reference.
- Dry preparation passes its configured check, but density tolerance is broad (±50% of 1 g/cm³), and the convergence summary has only nine samples. Its reported drift is −0.00223 ± 0.00133 g/cm³ per 100 ps, with first-/second-half densities 1.1653/1.1529. Passing this check does not establish that polymer structure is fully equilibrated.
- A trapezoidal integral of the saved rounded derivative means gives approximately −15.0633 kcal/mol total, versus MBAR −15.3981 (difference 0.3348). This is an indicative TI cross-check, not a separately uncertainty-qualified result; quadrature and sampling can both contribute. The saved morphology JSON does not preserve the full BAR/TI estimates needed to assess estimator agreement formally.
- No ERROR lines were found in the inspected logs. This does not prove that the live jobs are healthy or complete.

## Next analysis priorities

1. Finish both legs and rerun/estimation for iter_001; compare hydrated μ_ex and its diagnostics at the same loading, especially the fully charged endpoint.
2. Verify and retain the bulk reference and its uncertainty; the present −6.5 value alone is insufficient provenance.
3. Follow density and chemical potential across subsequent loadings. A difference in early density is a reason to test relaxation sensitivity, not evidence that either duration is sufficient.
4. Obtain independent packing trajectories and compare paired saturation endpoints before drawing conclusions about capacity or relaxation convergence. Consider additional sampling near full Coulomb coupling if its overlap/sample bottleneck persists.

Sources: run_config.yaml, uptake_state.json, dry/{composition,convergence}.json, iter_000/fep_membrane.json, iter_000/fep/morph00/morphology.json, iter_001/{in.insert,iter.log,density.dat,start.data}, and current iter_001 FEP logs. See comparison.png for density and dry-state sampling diagnostics; analyze.py reproduces the plot and progress summary.
