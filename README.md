# aem-water-uptake

Predict the **maximum water uptake** of an anion exchange membrane from its
repeat-unit SMILES, using molecular dynamics in LAMMPS.

```bash
aemwater run --smiles "[*]CC([*])c1ccc(C[N+](C)(C)C)cc1" \
             --n-chains 8 --chain-length 20 --counterion Cl- \
             --workdir my_membrane
```

```json
{
  "n_waters": 214,
  "lambda_waters_per_ionic_group": 13.4,
  "water_uptake_wt_pct": 34.1,
  "hydrated_density_g_cm3": 1.0834,
  "stop_reason": "thermodynamic_saturation",
  "converged": true
}
```

---

## What "maximum uptake" means here

A membrane in contact with liquid water absorbs water until the water inside is
as stable as the water outside. The endpoint is thermodynamic, not geometric:
the membrane does not fill until it runs out of room, it fills until the free
energy of transferring one more water from the reservoir into the membrane
reaches zero.

This package measures that directly. Each iteration inserts a batch of water,
lets the cell swell at constant pressure, and measures the excess chemical
potential of water in the swollen membrane by alchemically coupling a single
ghost SPC/E water (`mu_ex.method: fep`, the default). Widom test-particle
insertion remains available as `mu_ex.method: widom` but is not recommended --
see "Limitations" for the measured reason.
The loop stops when

    mu_ex(membrane) >= mu_ex(bulk water) - tolerance

with the tolerance expressed in units of the combined statistical uncertainty
of the two estimates, so the criterion is "indistinguishable from bulk" rather
than an arbitrary absolute cut.

Both sides of that comparison are computed with **identical** cutoff, long-range
accuracy, water model and insertion protocol. The difference being resolved is
a fraction of a kcal/mol, which is smaller than the systematic error any one of
those settings introduces on its own; only the difference is meaningful, and
only if the systematics cancel.

### Two other ways the loop can stop

* **Geometric saturation** — no cavity in the current configuration can accept
  another water. In a tightly crosslinked membrane this can be the physical
  answer; more often it means the batch is too large for the free volume that
  remains.
* **Iteration budget exhausted** — reported as `converged: false`. The uptake
  is then a *lower bound*, not a measurement. A number produced by running out
  of iterations is not an answer, and the package does not present it as one.

---

## Method

```
SMILES ──▶ chain builder ──▶ GAFF2 typing ──▶ packed cell ──▶ dry membrane
                                                                  │
                        ┌─────────────────────────────────────────┘
                        ▼
              ┌──▶ void detection ──▶ insert batch ──▶ NPT relax ──▶ mu_ex
              │                                              (FEP or Widom)
              │                                                        │
              └──────────── not saturated ◀────── compare to bulk ◀────┘
                                                         │
                                                    saturated
                                                         ▼
                                              lambda, wt %, structure
```

**Chain building.** Repeat units are joined at the two `[*]` sites into a
self-avoiding conformation, built segment-wise so that a dead end backtracks a
few units rather than restarting the chain. The builder reproduces the expected
scaling of radius of gyration with chain length.

**Typing.** `antechamber` / `parmchk2` / `tleap` assign GAFF2 atom types and
AM1-BCC charges. Energies of the generated LAMMPS input agree with Amber
term-by-term to better than 0.01%, which is checked by a regression test in the
suite rather than asserted here. Charges are derived once and reused for every
chain: they are the dominant cost of the whole workflow.

**Packing and equilibration.** Chains are placed at low density, pushed apart
with a soft-core potential, annealed above the glass transition, then densified
by a squeeze-and-release cycle: NPT at `md.compression_pressure` (1000 atm by
default) while still hot, cooled to the operating temperature under that load,
then released to `md.pressure` so the density relaxes to equilibrium from the
dense side.

The staged protocol matters in both directions. Compressing a freshly packed
cell directly traps voids that never relax; but compressing at ambient pressure
does not densify at all on an affordable schedule — the driving force is the
1 atm itself and the cell creeps. On the validation system, 30k steps at 1 atm
plateaued at 0.70 g/cm³, about 35% short of a real dry AEM. That failure is
quiet and it biases the answer upward: the missing density is void space, and
void space is exactly what the insertion loop fills.

**Insertion.** A grid solvent-accessible-volume test finds cavities that clear
each atom's own van der Waals radius plus a water probe. Sites are thinned to a
minimum separation so a single large cavity accepts several waters, and filled
deepest-first. Geometry proposes; the subsequent NPT relaxation accepts or
rejects. This beats random insertion (which fails almost always at a few percent
free volume) and grand-canonical Monte Carlo (which converges slowly in a glassy
matrix).

**Measurement (default: `fep`).** One ghost SPC/E water with its own atom
types is coupled to the cell over a ladder of fixed-lambda NVT simulations:
soft-core Lennard-Jones first at zero charge, then electrostatics at full LJ.
Neighbouring states are combined with MBAR/BAR, cross-checked against
finite-difference thermodynamic integration on the same trajectories, and
averaged over several independently equilibrated morphologies. Because the
ghost is a distinct set of types, scaling its interactions leaves every real
water untouched. See `docs/fep_design.md` for the soft-core form, the exact
LAMMPS commands, and the bulk validation.

**Measurement (`widom`).** Widom insertion with `full_energy`, block-averaged.
The Boltzmann average is dominated by rare favourable insertions, so the Kish
effective sample size is reported alongside the standard error and an estimate
carried by fewer than ten effective samples is flagged rather than returned
silently. Retained for comparison; at affordable insertion counts it does not
converge (see "Limitations").

---

## Installation

Requires a conda environment: the toolchain is not pip-installable.

```bash
conda create -n aem -c conda-forge python=3.11 rdkit ambertools lammps parmed \
                                   mdanalysis numpy scipy matplotlib pandas \
                                   pyyaml jinja2 pytest
conda activate aem
pip install -e .
```

LAMMPS must be built with the `KSPACE`, `MOLECULE`, `RIGID` and `MC` packages
(`fix widom` lives in `MC`). The conda-forge build has all four.

```bash
aemwater --help
pytest                     # 179 tests, no LAMMPS required
```

---

## Usage

### Phoenix GPU job

The bundled Slurm script requests one GPU on the Phoenix `embers` QOS and
loads the cluster's CUDA-enabled LAMMPS build:

```bash
sbatch examples/submit_phoenix_gpu.slurm
```

GPU switches are passed to every LAMMPS stage through
`AEMWATER_LAMMPS_ARGS`. The script deliberately uses one MPI rank per GPU;
increase both together only when requesting additional GPUs.

### Phases, separately invocable

```bash
aemwater prepare --config membrane.yaml    # dry membrane (expensive, reusable)
aemwater bulk                              # reservoir reference (cached globally)
aemwater run --config membrane.yaml        # the loading loop
```

`run` invokes the other two if their outputs are missing, so a single command is
enough. They are exposed separately because both are expensive and worth
inspecting before committing to a loop.

For a laptop-scale run of all three stages in ~11 minutes, plus a reading of the
warnings such a run emits, see `docs/running_end_to_end.md`:

```bash
PYTHONPATH=src python examples/run_e2e_fep.py
```

The bulk reference is cached on a hash of every setting that shifts `mu_ex`
(water model, temperature, pressure, cutoff, kspace accuracy, box size, sampling
lengths). It is computed once and reused by every membrane sharing those
settings.

### Reporting a number: `aemwater campaign`

`run` gives one packing's saturation point. It has no error bar, because a
single packing measures no spread — and in a glassy matrix the spread between
packings is usually the dominant uncertainty.

```bash
aemwater campaign --config membrane.yaml --morphologies 3
```

This runs the whole loop three times from three independent packings and reports
the mean uptake with a between-morphology standard error and a Student-t 95%
interval. Use it for anything you intend to quote; use `run` to inspect a single
trajectory.

Per-iteration `mu_ex` runs at screening resolution (7+7 lambda states, 150k
steps per state) which is 6.4x cheaper than production resolution, so three
screening trajectories cost less than one production trajectory. The saturation
point is where two curves cross, and a crossing does not move with the third
decimal of either curve. Pass `--production-resolution` to disable this.

Trajectories that never saturated are excluded from the average — their water
content is a lower bound, and averaging it in would bias the result low without
showing up in the error bar. A morphology that crashes is recorded and excluded
rather than killing the campaign. With only one usable morphology the campaign
reports `nan` for the error, not `0.0`.

Campaigns resume by default, at both levels: a morphology that already has an
equilibrated dry cell reuses it instead of re-annealing, and its uptake loop
continues from its last checkpointed iteration. That makes the campaign safe to
run on a preemptible queue — requeue the same job and it continues rather than
restarting. `examples/submit_phoenix_campaign.slurm` is set up this way, and
preflights the LAMMPS FEP package so a build missing it fails in one second
rather than after the first dry membrane has been built.

### Configuration

```yaml
polymer:
  smiles: "[*]CC([*])c1ccc(C[N+](C)(C)C)cc1"
  n_chains: 8
  chain_length: 20
  counterion: Cl-

water_model: spce

md:
  temperature: 298.15
  pressure: 1.0
  cutoff: 10.0
  relax_npt_steps: 100000
  mpi_ranks: 8

insertion:
  batch_fraction: 0.25      # waters added per iteration, as a fraction of content
  max_iterations: 40

mu_ex_method: fep           # fep (default) | widom

fep:
  n_morphologies: 3         # independently equilibrated cells to average over
  lj_lambdas: [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  coul_lambdas: [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
  soft_core_n: 1            # soft-core exponent; with alpha_lj below
  alpha_lj: 0.5
  alpha_coul: 10.0
  equil_steps: 50000
  production_steps: 500000
  sample_every: 1000        # must leave >= ~20 frames per state
  estimators: [mbar, bar, ti]
  ti_delta: 0.01            # finite-difference step for dU/dlambda
  min_overlap: 0.03         # flag neighbouring states below this
  max_stderr: 0.3           # kcal/mol; convergence budget on the CI half-width
  seed: 90210

widom:                      # only used when mu_ex_method: widom
  n_blocks: 5
  steps_per_block: 100000
  sigma_tolerance: 2.0      # saturation when the gap is within 2 sigma
  bulk_box_length: 25.0
```

Write a starting point with `RunConfig().dump_yaml("membrane.yaml")`.

**`fep.n_morphologies` is a requirement, not a hint.** The membrane campaign
refuses to average over fewer equilibrated cells than configured rather than
reporting a between-morphology error bar that rests on less replication than
the config claims.

**`md.timestep` (default 1.0 fs) is coupled to the constraints, and raising it
degrades the measurement.** Water is always rigid -- SPC/E and TIP3P are defined
with fixed geometry. Polymer X-H bonds are a separate matter: constraining them
is what permits a timestep above 1 fs, but doing so *while* Widom test particles
are being inserted and deleted produces SHAKE clusters resolved against the wrong
atoms. On the validation membrane, rigid water alone gave zero
`Shake determinant < 0.0` warnings under `fix widom`; adding the X-H constraint
gave 98, and enumerating the X-H bond types explicitly gave the same 98, which
rules out the selection syntax rather than the constraint. Both variants held
298 K over 10 ps, so this is not a stability failure -- it is a quiet corruption
of the stage that produces the number the whole method rests on. At the 1 fs
default, X-H constraints are therefore not requested at all; above it, the run
warns.

### Resuming

Every iteration is checkpointed. An interrupted run continues from the last
completed iteration:

```bash
aemwater run --config membrane.yaml      # resumes
aemwater run --config membrane.yaml --force   # starts over
```

FEP campaigns resume at three granularities, since one campaign is
`n_morphologies` x (`len(lj_lambdas)` + `len(coul_lambdas)`) independent LAMMPS
runs plus a rerun pass of the same order:

| unit | reused when |
| --- | --- |
| morphology | `morph<NN>/morphology.json` holds a finite estimate |
| lambda window | the state log ends in LAMMPS's wall-time line **and** `fep.dat`, `pe.dat`, `traj.lammpstrj` are all non-empty |
| rerun pass | `rerun_<j>.log` ended cleanly, `rerun_<j>.dat` is non-empty, and window `j` was not re-sampled this invocation |

```bash
python examples/fep_water_validation.py               # start or resume
python examples/fep_water_validation.py --no-resume   # discard and restart
python examples/fep_water_validation.py --ranks 8
```

Completeness is judged from what LAMMPS wrote, not from a marker this package
writes: a marker can outlive the thing it describes. A window killed
mid-trajectory is redone from its start rather than restarted mid-way, because a
half-length trace carries less weight in the estimators than its neighbours and
would quietly reweight the ladder.

Each run directory holds a `campaign_stamp.json` naming the protocol that wrote
it — ladders, sampling lengths, state point, box size. Resuming into a directory
written under different settings raises rather than proceeding, because
averaging windows from two protocols into one `mu_ex` passes every downstream
check: each individual window is internally consistent. Membrane morphologies
additionally carry a `cell_stamp.json` fingerprinting the supplied
configuration, since those cells come from the caller and a later uptake
iteration would otherwise supply a *different* cell with the same atom count and
type table.

---

## Output

| file | contents |
|---|---|
| `result.json` | the answer: lambda, wt %, densities, stop reason, convergence |
| `uptake_trajectory.csv` | per-iteration water count, density, volume, mu_ex, gap |
| `uptake.png` | loading curve, saturation criterion, swelling, cavity filling |
| `report.md` | narrative summary including hydration structure |
| `iter_NNN/` | the LAMMPS inputs, logs and data files for each iteration |

Structural analysis reports whether the absorbed water forms a **percolating**
network or isolated pockets — two membranes with the same uptake can differ
entirely in this respect, and hydroxide conduction needs a connected path.

---

## Limitations

**GAFF2 is a fixed-charge force field.** Quaternary ammonium cations polarise
the water around them, and the ion pairing between the cation and its
counterion is approximate. Absolute uptakes from any non-polarisable model
carry a systematic error; the ranking between chemistries is more reliable than
the absolute number.

**Widom insertion is far from converged at affordable insertion counts. This
was the dominant caveat of the method and is why `fep` is now the default.**
It was not specific to the membrane: 90k insertions into *bulk* SPC/E water at
298 K gave mu_ex = -2.74 +/- 0.64 kcal/mol against a published -6.5, with the
Boltzmann average spanning four orders of magnitude across blocks and an
effective sample size of 3.2. The average is carried by the rare trials that
land in a cavity, so it approaches the true value from above and reaching it
takes of order 10^6 insertions -- hours of serial CPU for the reference alone.

The alchemical path removes this error rather than working around it: on the
same water model, two morphologies at deliberately cheap settings give
mu_ex = -6.825 kcal/mol, 95% CI [-7.206, -6.444], which brackets the published
value (`docs/fep_design.md`). The remaining FEP caveats are different in kind
-- ladder adequacy, between-morphology replication at small M, and the fact
that the fixed-lambda states run NVT so density is an input rather than a
result -- and are documented there.

The rest of this subsection applies only to `mu_ex.method: widom`. Two
consequences follow, and the second is why that backend was usable at all.

First, `BulkReference.mu_ex` is not a measurement of the excess chemical
potential of water and must not be reported as one. `sanity()` says so, and
`converged` is `False` below `MIN_EFFECTIVE_SAMPLES`.

Second, the saturation criterion is a *difference* between the membrane and the
reservoir, both computed by this code at the same settings, and the bias largely
cancels. It comes from the unsampled tail of the cavity-size distribution, so it
cancels to the extent that the two systems have similar cavity statistics -- and
a membrane approaching saturation is converging on bulk-like water domains. The
cancellation is therefore best exactly where the criterion is evaluated, and
worst in the dry membrane where the answer is not in doubt. This is why the
reference is recomputed rather than taken from the literature: subtracting a
*converged* published value from an under-converged membrane estimate would
compare two different quantities and report saturation several waters early.

The run is still least reliable at low hydration, where insertions almost always
overlap; the endpoint, measured on a swollen cell, is the best-sampled point.

**Cell size.** A cell whose edge is comparable to the water cluster size biases
the mu_ex estimate under either backend and the percolation analysis with it.
The example runs are small enough for a laptop and are not production settings.

**The measured quantity is equilibrium uptake in contact with liquid water**
(activity 1). Uptake from vapour at lower activity requires a different
reference and is not what this computes.

---

## Layout

```
src/aemwater/
  polymer.py      chain construction from SMILES
  chemistry.py    composition, ionic groups, counterions
  forcefield/     GAFF2 typing, water models
  packing.py      rigid-body cell packing
  assembly.py     molecules + coordinates -> one system
  lammps/         data-file writer, input templates, runner, log parser
  insertion.py    void detection and water placement
  fep/            alchemical mu_ex: ghost water, lambda schedules, state
                  inputs, rerun/energy matrix, MBAR/BAR/TI, campaign,
                  diagnostic figures
  widom.py        chemical potential estimation (alternative backend)
  bulk.py         bulk water reference
  prepare.py      dry membrane construction
  driver.py       the uptake loop
  analysis.py     clustering, percolation, figures, report
  cli.py          prepare / bulk / run
```
