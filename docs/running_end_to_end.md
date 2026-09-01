# Running the pipeline end to end

Three ways, in increasing cost. All three measure the reservoir and the membrane
with the *same* estimator -- `obtain_bulk_reference` dispatches on
`mu_ex_method` precisely so the two can never diverge. Mixing them would be a
serious error: the saturation criterion is the *difference* between the two
numbers, and the Widom bias in bulk water is several kcal/mol, larger than the
effect being measured.

## 1. Laptop smoke test (~11 min, one command)

```bash
PYTHONPATH=src python examples/run_e2e_fep.py
```

Chains all three stages against `examples/e2e_fep_smoke.yaml` and prints the
comparison. This is a wiring test, not a measurement -- see the caveats below.

Measured on a 10-core laptop, serial LAMMPS:

| stage | wall time |
|---|---|
| bulk water reference (FEP, 8 states, 1 morphology) | 2.8 min |
| dry membrane (2 chains, 21-step compression) | 0.7 min |
| hydration loop (6 iterations) | 7.4 min |

Outputs land in `runs/e2e_fep_smoke/`:

* `e2e_summary.json` -- the three stage results plus per-stage timings
* `uptake_trajectory.csv` -- one row per iteration
* `fep_overlap.png`, `fep_dudl.png`, `fep_morphologies.png` -- bulk diagnostics
* `bulk_cache/` -- the cached reference, keyed on the full protocol

The cache key covers the ladders, sampling lengths, morphology count, soft-core
parameters and the state point, so a smoke-scale reservoir can never be served
to a production run. `widom.cache_dir` is set inside the workdir here, so
deleting `runs/e2e_fep_smoke` resets the whole test.

### What the smoke run actually reports

```
mu_ex(bulk)     = -4.960 +/- 1.052 kcal/mol
mu_ex(membrane) = -5.566 +/- 0.708 kcal/mol at lambda = 1.90
gap             = -0.606 kcal/mol
stop reason     = max_iterations
```

Every one of these numbers is wrong, and the framework says so. The run emits
five warnings, and reading them is the actual point of the exercise:

1. **Bulk is 1.5 kcal/mol off the published SPC/E value** (-4.96 vs -6.5). The
   overlap figure localises it: the LJ leg pair at lambda ~ 0.38 falls below
   threshold, so the 5-state smoke ladder has a gap. The 12-state production
   ladder exists for this reason.
2. **MBAR and BAR disagree by more than 5x** on the same leg -- near-degenerate
   covariance from that same gap. The selection logic reports BAR and says why.
3. **One morphology, so the between-cell spread is unmeasured.** The reported
   stderr is a within-cell number and understates the truth.
4. **The dry membrane failed its convergence check** (0.944 vs 1.05 g/cm3, with
   density still climbing). Uptake from a cell with leftover void space is
   biased high, because the loop fills voids instead of swelling the cell.
5. **The loop hit `max_iterations` without saturating**, so 15.7 wt% is a lower
   bound, not a saturation point.

A run that printed three numbers and no warnings would be worse, not better.

## 2. Single morphology at production resolution

```bash
PYTHONPATH=src aemwater run --config examples/qa_polystyrene.yaml \
    --workdir runs/btma_ps
```

One packing's saturation point. `qa_polystyrene.yaml` does not set
`mu_ex_method`, so it inherits the default (`fep`). At `mpi_ranks: 1` this is
~178M MD steps; raise it to your allocation before launching.

## 3. Campaign over independent morphologies

```bash
PYTHONPATH=src aemwater campaign --config examples/qa_polystyrene.yaml \
    --workdir runs/btma_ps_campaign --morphologies 5
```

Separate subcommand rather than a flag, because it returns a different thing: a
mean with a between-morphology error bar rather than one cell's endpoint. Exits
2 with a warning if fewer than two morphologies survive, since a single sample
carries no uncertainty. `examples/submit_phoenix_campaign.slurm` wraps this and
preflights the FEP package so a bad build fails in one second rather than after
the first dry membrane.

## Resuming a run that was killed

Every command above resumes by default. Rerun the identical command in the same
workdir and finished work is reused; pass `--force` to discard it and start
over. Nothing needs to be cleaned up by hand between the two.

What "finished" means differs by layer, and the granularity is what determines
how much a walltime kill costs you:

| layer | unit reused | how completeness is decided |
|---|---|---|
| bulk reference | the whole reference | cache file keyed on the full protocol |
| morphology | one cell's two legs | `morphology_NN.json` checkpoint |
| lambda window | one `lmp` invocation | LAMMPS's terminal line + non-empty outputs |
| rerun row | one re-evaluated state | its output file, per index |
| uptake loop | one hydration iteration | `relaxed.data` plus the loop checkpoint |

So the worst case for a kill is one lambda window, which is minutes on
screening settings -- not the campaign.

**Completeness is read from what LAMMPS wrote, not from a marker this package
controls.** A marker of our own can outlive the thing it describes: written
before a crash, or left behind when a later change alters what the directory
should contain. LAMMPS emits its terminal wall-time line only on a clean exit,
so that line plus every non-empty output file the next stage reads is the test.
A window killed mid-trajectory therefore looks incomplete and is redone from
its start.

That last point is deliberate rather than a limitation. LAMMPS *can* restart
mid-trajectory, but a state whose production run was cut in half has a shorter
correlated trace than the ladder's other states, and the estimators weight
states by effective sample count -- stitching a restart onto a killed trace
would quietly change that weighting. Repeating a window is cheap; a silently
reweighted free energy is not.

### Resuming into a directory that has moved on

A workdir carries a stamp of the calculation that wrote it: the ladders, the
sampling lengths, the state point, the cell. Resuming with different settings
**refuses** rather than proceeding, because the failure it prevents is silent:
averaging windows from two different Hamiltonians into one free energy produces
a number, not an error, and no downstream check would catch it. Either use a
fresh workdir or pass `--force`.

Two things a resumed run is *not* allowed to reuse:

* **A stale bulk reference under `--force`.** `--force` reaches the reference
  itself, not just the membrane. Recomputing the membrane against a cached
  reservoir measured under the old settings would compare two different
  protocols -- and the saturation criterion is the *difference* between them.
* **A rerun row whose window was resampled.** Re-sampling a window invalidates
  the rerun rows derived from it; they are recomputed rather than read back.

### If a cache file is corrupt

A kill during a cache write can leave truncated JSON. Such a file is logged at
WARNING and ignored -- the run recomputes rather than raising. All checkpoint
writes go through `utils.write_json` (temp file plus rename, sibling to the
target so it stays on one filesystem), so this should not arise; the tolerant
read exists because the alternative is a pipeline that stays wedged until
someone deletes a file by hand, with a traceback pointing at the reader rather
than at the kill that caused it.
