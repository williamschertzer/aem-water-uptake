# Dry-membrane equilibration: the 21-step scheme

## Why this exists

The water-uptake number this pipeline computes is only as good as the dry
membrane it starts from. Diagnosis of the example runs in `runs/ex_cpu/` and
`aem_run/` found the dry state to be the blocking problem, ahead of any issue
with the Widom estimator:

- The dry cell plateaued near **0.95 g/cm3**, roughly **18% below** the
  experimental range for benzyltrimethylammonium polystyrene (1.10-1.25).
- Density was **still rising when the run ended** — the structure had not
  finished collapsing, so no part of the trace was an equilibrium value.

Both facts point the same way: the cell contained void space that should not
have been there. That matters because of what the uptake loop does next. Water
inserted into a voided cell fills the voids instead of swelling the membrane,
which shows up as a **negative partial molar volume of water** — the cell
contracting as water is added — and an uptake number biased high with no
internal signal that anything is wrong. A converged dry density is a
precondition for the rest of the calculation, not a nicety.

The old schedule was one hot anneal, one compression, one release. That is not
enough. Collapsing a loose packing to glassy density is a **sampling** problem,
not a barostat problem: the chains have to find melt conformations, and at
300 K they cannot move enough to do it.

## The scheme

Seven cycles of *(NVT hot / NVT quench / NPT compress)*, 21 stages, 1560 ps at
a 1 fs timestep. Pressure is ramped **up** to 50 000 atm over cycles 1-3 and
**released** back to 1 atm over cycles 4-7.

| step | ensemble | T (K) | P (atm) | ps | role |
|-----:|:---------|------:|--------:|-----:|:-----|
|  1 | NVT | 600 |     — |  50 | cycle 1 relax hot |
|  2 | NVT | 300 |     — |  50 | quench |
|  3 | NPT | 300 |    10 |  50 | compress (0.02% of peak) |
|  4 | NVT | 600 |     — |  50 | cycle 2 relax hot |
|  5 | NVT | 300 |     — | 100 | quench |
|  6 | NPT | 300 |   300 |  50 | compress (0.6%) |
|  7 | NVT | 600 |     — |  50 | cycle 3 relax hot |
|  8 | NVT | 300 |     — | 100 | quench |
|  9 | NPT | 300 | 50000 |  50 | **peak compression** |
| 10 | NVT | 600 |     — |  50 | cycle 4 relax hot |
| 11 | NVT | 300 |     — | 100 | quench |
| 12 | NPT | 300 | 25000 |   5 | decompress to 50% |
| 13 | NVT | 600 |     — |   5 | cycle 5 relax hot |
| 14 | NVT | 300 |     — |  10 | quench |
| 15 | NPT | 300 |  5000 |   5 | decompress to 10% |
| 16 | NVT | 600 |     — |   5 | cycle 6 relax hot |
| 17 | NVT | 300 |     — |  10 | quench |
| 18 | NPT | 300 |   500 |   5 | decompress to 1% |
| 19 | NVT | 600 |     — |   5 | cycle 7 relax hot |
| 20 | NVT | 300 |     — |  10 | quench |
| 21 | NPT | 300 |     1 | 800 | **production — density averaged here** |

14 NVT stages, 7 NPT stages.

**The decompression half is the load-bearing part.** A single squeeze-and-release
compresses once and lets go once; the residual voids are simply frozen in. Here
the structure is repeatedly over-compressed, allowed to relax hot, and then let
out in stages — so the final density is approached from the dense side at every
scale rather than once at the end. Cycles 4-7 are what the old schedule was
missing.

Reference: Larsen, Lin & Colina, *Macromolecules* **44** (2011) 6944; the
equilibration stage of Polymatic (Abbott, Hart & Colina, *Theor. Chem. Acc.*
**132** (2013) 1334).

## Configuration

```yaml
equilibration:
  scheme: "21step"          # or "legacy" for the old single-squeeze cycle
  max_pressure: 50000.0     # peak, atm
  high_temperature: 600.0   # NVT excursion temperature, must be above Tg
  time_scale: 1.0           # scales steps 1-20 only
  final_npt_ps: 800.0       # step 21; set directly, NOT scaled
  enforce_convergence: true
  expected_density: 1.15    # g/cm3; falls back to box.target_density
  density_tolerance: 0.08   # fractional
  drift_tolerance: 0.002    # g/cm3 per 100 ps
```

`max_pressure` is configurable but the *shape* of the ramp is stored as
fractions of it, so the schedule stays as published while the peak can be
tuned. `time_scale` shortens stages 1-20 for smoke tests while keeping all 21
present — a scaled run exercises the same code path as production, not a
different one. Step 21 is deliberately exempt: it has to stay longer than
several `md.thermo_every` intervals or the density file comes out empty.

## The convergence gate

The schedule alone is not a guarantee, so `prepare.py` now judges the result and
**refuses to hand an unconverged cell to the uptake loop** (unless
`enforce_convergence: false`). Written to `dry/convergence.json`, with a
per-criterion line in the log.

Three criteria, all necessary:

1. **Density** within `density_tolerance` of `expected_density`. A cell can sit
   stably at the wrong density — the old scheme did exactly that.
2. **Drift** — `|d(rho)/dt|` over the production window within
   `drift_tolerance`, *and* the drift must be statistically resolved. A slope
   within two standard errors of zero is consistent with no densification and
   does not fail; without this, scatter on a short window fakes a slope far
   above tolerance and the criterion becomes unpassable. A structure still
   densifying is still equilibrating.
3. **At least 4 samples** in the production window. Too few points to judge is
   not the same as converged.

Density is averaged over **step 21 only**, so no part of the compression
schedule contaminates the reported value.

## Cost

1 560 000 MD steps for the dry stage, of which step 21 is 800 000. This is now
the dominant cost of preparation and it is the intended trade: the previous
schedule was cheaper and gave an answer that could not be used.

For a quick check of the whole pipeline use `examples/smoke_test.yaml`
(`time_scale: 0.01`, gate off) — all 21 stages, ~23 ps.

## Validation performed

- Resolved schedule matches the published protocol on every checkable property:
  21 stages, 1560 ps, 14 NVT / 7 NPT, peak 50 000 atm at step 9, monotone ramp
  up and down, final stage at the operating pressure.
- The rendered deck was **executed in LAMMPS on the real BTMA-PS membrane**
  (`runs/ex_cpu/dry/min.data`, 5183 atoms): all 21 stages ran, no errors, no
  dangerous neighbour builds.
- **The scheme does what it is supposed to do.** Run at 10% of the published
  stage durations (136 000 steps, ~40 min on one CPU core) on that membrane, the
  cell went 0.200 -> peak **1.405** g/cm3 at 50 000 atm -> **1.017** g/cm3 after
  staged decompression, and the production window was flat (drift within its own
  uncertainty). The old single-squeeze scheme plateaued at **0.947** and was
  still climbing. That is +7% density from the same starting structure at 10% of
  the intended schedule length; the full-length run should land closer to the
  1.10-1.25 experimental band. **The full-length schedule has not yet been run
  — that is the next thing to do, and on the GPU machine rather than here.**
- The gate was checked against four regimes — stable at target (passes), stable
  at the wrong density (fails on density), genuine slow densification (fails on
  drift), and pure noise (does not fail) — plus the real still-densifying trace
  from the test run, which it correctly rejects.
- `tests/test_equilibration.py` pins all of the above; full suite 261 passed.

Two real bugs were caught by this validation rather than by a production run:
the template rendered `run <bound method>` instead of a step count (LAMMPS would
have died on stage 1), and `final_npt_ps` was being scaled twice, which emptied
the density file.
