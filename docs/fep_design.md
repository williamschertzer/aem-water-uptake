# Alchemical FEP for the excess chemical potential of water

This document fixes the protocol before any code is written. It records the
soft-core form, the two-leg schedule, the exact LAMMPS commands to be emitted,
and — importantly — the systematic terms that do *not* cancel and therefore
have to be handled explicitly.

## Why replace Widom

The README already records the failure honestly: 90k insertions into bulk SPC/E
gave `mu_ex = -2.74 +/- 0.64 kcal/mol` against a published `-6.5`, with a Kish
effective sample size of 3.2. The estimator is biased toward zero and approaches
the truth from above, because the Boltzmann average is carried by the rare trial
that lands in a cavity.

The current method survives that bias only by subtracting two equally
under-converged numbers and trusting the cancellation. That trust is the weakest
link in the whole workflow: the cancellation is an argument, not a measurement,
and it is least reliable exactly where the answer is least certain.

Staged FEP removes the problem rather than cancelling it. Instead of asking "what
is the energy of inserting a whole water into a random point", it grows the water
in from nothing through a ladder of intermediate states, each of which overlaps
its neighbours. No single step relies on a rare event, so the estimator converges
at a rate set by ordinary MD sampling.

## What is being computed

One **ghost** SPC/E water is present in the cell at every lambda. Only its
interactions with the rest of the system change; the molecule itself is never
created or destroyed, so `N` is constant and the ensemble is the same at both
endpoints.

At `lambda = 0` the ghost is an ideal-gas molecule sampling the box volume
uniformly. At `lambda = 1` it is an ordinary SPC/E water. The free-energy
difference between those two states is, by Widom's identity read backwards,
exactly the excess chemical potential:

    mu_ex = A(lambda=1) - A(lambda=0)

Because the ghost's mass, internal geometry and kinetic terms are identical at
both ends, they cancel exactly — no ideal-gas or de Broglie term enters, and
there is no `kT ln <V>` volume correction of the kind an NPT Widom estimate
needs. This is a genuine advantage of the ghost formulation over test-particle
insertion, not merely a variance reduction.

**The ghost is one molecule in a finite cell.** It is a `1/N` perturbation of the
composition, so the quantity converged to is the chemical potential at the
composition of the *host* cell, not of the cell plus one water. For the uptake
loop this is the right quantity: it is the free energy of transferring the next
water in.

## Soft-core form

LAMMPS' FEP package implements the Beutler soft-core. For a pair `i-j` carrying
coupling parameter `lambda`:

    V_LJ = lambda^n * 4 eps [ 1/(alpha_LJ (1-lambda)^2 + (r/sigma)^6)^2
                            - 1/(alpha_LJ (1-lambda)^2 + (r/sigma)^6) ]

    V_C  = lambda^n * (q_i q_j / 4 pi eps_0) / sqrt(alpha_C (1-lambda)^2 + r^2)

Set by `pair_style lj/cut/coul/long/soft n alpha_LJ alpha_C cutoff`, with
`pair_coeff i j eps sigma lambda`.

The `(1-lambda)^2` inside the denominator is what removes the `r -> 0`
singularity: at `lambda = 0` the potential is finite everywhere, so a ghost can
overlap a real atom without producing an infinite force. This is why the
insertion never fails the way random Widom insertion does.

Defaults adopted: `n = 1`, `alpha_LJ = 0.5`, `alpha_C = 10.0` — the values the
FEP package documents and the Beutler paper uses.

**Verified on the binary the runner actually uses** — `lmp` from the `aemmd`
conda environment, LAMMPS 22 Jul 2025 Update 5. At `lambda = 1` the soft style
reproduces plain `lj/cut/coul/long` to 2.4e-7 relative on a 30-water box
(136.8804821 vs 136.8804498 kcal/mol, the residual being PPPM noise). The
alchemical endpoint is therefore the same physics the rest of the workflow uses,
which is the property the whole comparison rests on.

The check was repeated on the Homebrew build (23 Jun 2022) with the same result
to 7.5e-7. Both builds carry the FEP package; the probe in `fep/capability.py`
tests whichever binary is configured rather than trusting either.

## Two legs, in this order

A single lambda scaling both LJ and charge at once would drag a fully charged
oxygen through a half-repulsive shell, where the electrostatic attraction is
unscreened and the sampling collapses. Hence the standard decomposition.

### Leg 1 — soft-core Lennard-Jones, charges off

Ghost charges are **zero in the data file**. `lambda_LJ` runs `0 -> 1` on every
ghost-host type pair. Because `q_ghost = 0`, the `coul/long/soft` term
contributes nothing at any lambda, so this leg is pure LJ growth.

Default ladder, clustered near zero where `dU/dlambda` of a soft-core peaks:

    0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

### Leg 2 — electrostatics, LJ fully on

`lambda_LJ = 1` (the ghost is now a real, neutral LJ water). Ghost charges are
scaled `q(lambda_Q) = lambda_Q * q_full` and written directly into each state's
data file. With the repulsive core fully present there is no singularity, so
linear charge scaling is safe and no soft-core Coulomb is needed.

    0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0

Note that `U(lambda_Q)` is **not** linear even though the charges are: PPPM
includes the ghost's interaction with its own periodic images, which scales as
`lambda_Q^2`. This is not a problem — MBAR and BAR use the actual evaluated
energies, and TI uses a finite difference — but it does mean `dU/dlambda` must
never be assumed analytic here.

    mu_ex = dA(leg 1) + dA(leg 2)

## Endpoint validation (measured, not asserted)

Both alchemical endpoints were checked against systems built without any ghost at
all, on a 15 A cell of SPC/E water, PPPM accuracy 1e-6, `rc = 10 A`. The ghost
was placed at exactly the coordinates the 31st water occupies, so the comparison
is between two descriptions of the *same* configuration.

**`lambda_LJ = 1`, `lambda_Q = 1` vs. 31 real waters** — the coupled ghost must be
indistinguishable from a genuine water:

    quantity   31 real waters     30 + coupled ghost    rel. diff
    EVDWL          -15.134428            -15.134428      0
    ELONG        -2084.968487          -2084.968487      0
    ECOUL         2032.153319           2032.153344      1.2e-08
    PE             -67.949595            -67.949571      3.6e-07

**`lambda_LJ = 0`, `lambda_Q = 0` vs. 30 real waters** — the decoupled ghost must
contribute nothing:

    quantity   30 real waters     30 + ghost off        abs. diff
    EVDWL          -14.268960            -14.268960      0
    PE             -66.171366            -66.171364      1.8e-06
    ECOUL         1965.366815           1965.293607      7.3e-02
    ELONG        -2017.269221          -2017.196012      7.3e-02

The last two lines deserve a note, because they look alarming in a log and are
not. `ECOUL` and `ELONG` shift by equal and opposite amounts, so the total is
unaffected: with the ghost's charges at exactly zero, the shift is PPPM's own
real-space/k-space split re-partitioning over a changed atom count (93 vs. 90),
not a physical interaction. The sum is what enters the free energy, and it is
conserved to 1.8e-6 kcal/mol.

Together these fix both ends of the thermodynamic path against ground truth. A
regression in the ghost machinery — a shared type, a mis-scaled charge, a dropped
cross term — breaks one of these two tests.

## PPPM accuracy is not a free parameter here

The workflow default `kspace_style pppm 1.0e-4` is adequate for forces but not
for the energy *differences* FEP is built from. Measured on one fixed
configuration of 195 SPC/E waters plus a ghost, charge leg, full
`U(lambda_q=1) - U(lambda_q=0)`:

| pppm accuracy | dU (kcal/mol) | error vs 1e-8 | relative |
|---|---|---|---|
| 1.0e-4 | -3.03044 | -1.58e-2 | 0.52% |
| 1.0e-5 | -3.01435 | +3.14e-4 | 0.01% |
| 1.0e-6 | -3.01485 | -1.82e-4 | 0.01% |
| 1.0e-7 | -3.01469 | -2.36e-5 | 0.00% |
| 1.0e-8 | -3.01466 | 0 | — |

The same absolute error (~0.016 kcal/mol) appears in a dilute 40-water cell,
which identifies it as grid resolution rather than anything physical. It carries
a consistent sign, so it accumulates over the ladder rather than averaging out:
across a 6-interval charge ladder that is ~0.1 kcal/mol against the 0.30
kcal/mol budget in `fep.max_stderr`. **No amount of sampling removes it** — it is
not noise.

Cost of the fix, 200 steps of NVT on the same cell: 1e-5 is 1.01x, 1e-6 is
1.11x, 1e-7 is 1.84x. `FEPSpec.kspace_accuracy` therefore defaults to 1e-6 and
`validate()` refuses anything looser than 1e-5.

A related note on `compute fep` itself: its reported dU agrees with an explicit
`U(lambda_b) - U(lambda_a)` to 2.6e-11 kcal/mol on the LJ leg, where no charges
move. On the charge leg the agreement is limited by exactly the PPPM noise above
(3e-6 at 1e-6 accuracy, falling to 2e-7 at 1e-8), which is how the grid error was
first identified — the discrepancy did not shrink with a finer ladder, only with
a finer grid.

## SHAKE is safe on the ghost, unlike under `fix widom`

The ghost shares bond and angle *types* with resident water (it is a renamed
copy), so a single `fix shake` command constrains both, and constraining them
identically is what makes the coupled endpoint equal to a real water.

This is worth stating explicitly because the repo's `ConstraintSpec` documents
the opposite conclusion for Widom: constraining polymer X-H while `fix widom`
runs produces 98 "Shake determinant < 0.0" warnings. The mechanism is that SHAKE
builds its cluster list once at setup and `fix widom` invalidates it by inserting
and deleting atoms on every trial. **The ghost is a permanent atom**, so no
cluster list is ever invalidated — the failure mode does not apply.

Verified over the whole ladder (`lambda_lj` = 0, 0.05, 0.1, 0.5, 1.0 and
`lambda_q` = 0, 0.5, 1.0), 2000 steps each at a 2 fs timestep: zero SHAKE
determinant warnings, zero lost atoms, temperature stable at 283-318 K. The
concern that a permeable soft core at low lambda would break the constraint
solver is not borne out — the constraint acts on the ghost's internal geometry,
which the coupling parameter never touches.

## The rerun pass: K reruns, not K squared

MBAR needs `u_kn[k, n]` — sample `n` evaluated under state `k`'s Hamiltonian.
The sampling runs only record dU to *neighbours*, which suffices for BAR and TI
but not MBAR, so the matrix is assembled in a second pass.

The obvious implementation re-evaluates every trajectory at every state: K^2
reruns. That is not necessary. `compute fep` accepts an *arbitrary* perturbation,
not just a neighbouring one, so a single rerun of state j's trajectory carrying
K-1 clauses produces the entire matrix row in one pass over the frames. For the
default 9-state LJ ladder that is 9 reruns rather than 81.

Verified: `test_only_one_rerun_per_state` asserts the input count equals K.

### Output precision is not cosmetic

Two LAMMPS defaults silently corrupt the matrix, and both were found by the
diagonal check failing.

| what | default | error it injects | fix |
|---|---|---|---|
| `fix ave/time` file | `%g`, 6 sig figs | PE of -1545.39 quantised at 0.01 kcal/mol | `format " %.14g"` |
| `dump` coordinates | `%g`, 6 sig figs | ~3e-2 kcal/mol per re-evaluated energy | `dump_modify ... format float %.16e` |

Both are two orders of magnitude above the 1e-4 diagonal tolerance and larger
than the PPPM error the ladder is already tightened to avoid. The leading space
in `" %.14g"` is required — without it `ave/time` runs the timestep and the first
value together (`043.149958107408`), which `np.loadtxt` then reads as one column.

### Frame 0 is discarded

With both precisions fixed, a rerun of a state against its own trajectory
reproduces every frame to **2e-10 kcal/mol** — except frame 0, which agrees only
to 1.7e-2:

```
   step        sampled          rerun         diff
      0   -1545.386887   -1545.403993   -1.71e-02
    100   -1736.633588   -1736.633588   -1.00e-10
    200   -1872.492952   -1872.492952   +1.00e-10
    ...            ...            ...         ...
```

The gap does not shrink with grid accuracy (2.1e-1 at pppm 1e-4, 1.7e-2 at 1e-6,
1.5e-2 at 1e-8), which rules out the solver, and it is unchanged with SHAKE off,
which rules out constraints. Frame 0 is the configuration inherited from
equilibration, whose energy LAMMPS evaluates during run setup rather than during
dynamics. It is also not an independent sample. `_read_two_column` drops it.

### The diagonal check

Every rerun's recomputed `u_kk` is compared against the `pe.dat` the sampling run
wrote, tolerance 1e-4 kcal/mol (measured agreement 2e-10, acceptance budget
0.30). A mismatch raises rather than warns: a rerun whose force field differs
from the sampling run's — a dropped pair coefficient, a different cutoff, charges
left unset after `read_data` — yields a *plausible* matrix and a confidently
wrong free energy that no downstream diagnostic exposes. This guard caught both
precision defects above during development, and
`test_diagonal_check_catches_a_wrong_hamiltonian` corrupts `pe.dat` by
1 kcal/mol to confirm it still fires.

Note for the charge leg: ghost charges live in the data file, so a rerun at a
different `lambda_q` must reissue `set type <t> charge`. Verified working under
`rerun` (`2 settings made for charge`), and both site types scale by the same
factor so the ghost stays neutral.

## Estimators: three views, not three options

All three run on one sampling campaign. They are not alternatives to pick
between — their *disagreement* is the diagnostic:

| estimator | uses | reported? | what a disagreement means |
|---|---|---|---|
| MBAR | every sample vs every state | **yes** | — |
| BAR | neighbouring pairs only | cross-check | that pair's overlap is marginal |
| TI | `<dU/dl>` integrated over the ladder | cross-check | the *ladder* is too coarse |

A TI-vs-MBAR gap indicts the schedule, not the sampling, and no amount of extra
sampling closes it. That is the whole reason TI is retained.

### Validated against an analytic answer

Coupled harmonic oscillators have a closed-form free energy
(`F = -0.5 ln(2π/K)`), so the estimators are tested against truth rather than
against each other — a mutual-consistency test would miss a shared kT factor or
sign error. On a 5-state ladder, 4000 samples per state, exact dF = 1.3863 kT:

| estimator | value | error |
|---|---|---|
| MBAR | +1.3868 ± 0.0117 | 5.0e-4 |
| BAR (summed) | +1.3736 ± 0.0105 | 1.3e-2 |

TI's error is *resolution*, and falls with ladder density at fixed sampling —
0.376 → 0.114 → 0.038 → 0.022 kT for 5, 9, 17, 33 states. This is asserted as a
monotonic sequence, so a quadrature regression shows up as a test failure.

### BAR's error bar is a lower bound, not a conservative one

Neighbouring pairs share samples: state k enters both the (k-1, k) and the
(k, k+1) pair. Adding per-pair errors in quadrature assumes an independence that
does not hold, so the sum **under**counts. On the harmonic reference BAR reports
0.0104 against MBAR's 0.0117 *while using strictly less information*. This was
initially documented backwards; the test now pins the true direction so the
tighter number is not mistaken for the better one.

### TI quadrature must be spacing-aware

The ladder is deliberately non-uniform (clustered at small λ where the soft-core
derivative peaks). Integrating `3x²` over the default LJ spacing:

| method | result | error |
|---|---|---|
| `np.trapezoid(f, lambdas)` | 1.00463 | 4.6e-3 |
| naive mean of the integrand | 0.96313 | 3.7e-2 |
| Simpson | — | not applicable on a non-uniform grid |

Eight times worse for dropping the λ argument. Uncertainty propagates with the
explicit trapezoid weights (half-intervals at the ends), and the maximum absolute
second difference of `<dU/dλ>` is reported so a coarse ladder is visible directly
rather than inferred from a TI-MBAR gap.

### `fix ave/time` column headers use title2, not title3

In scalar mode `ave/time` writes two header lines, and the column names are the
second. `title3` is **silently ignored** — LAMMPS substitutes its own default
header with `v_`-prefixed names, and a reader looking for `dU_ti_plus` finds
`v_dU_ti_plus` and reports a missing column instead. Fixed in `fep_state.in.j2`
and `rerun.py`; `read_fep_columns` additionally strips a leading `v_` so a
regression degrades to a working read rather than a crash.

### Decorrelation happens once, up front

Fixed-λ MD gives correlated frames. Feeding them to MBAR as independent yields a
free energy that is fine and an uncertainty optimistic by roughly `sqrt(g)`, so
every estimator subsamples to the statistical inefficiency of the state's own
trace first. A test duplicates every frame — information unchanged, naive N
doubled — and asserts the honest error bar is the wider one.

### One extreme work value makes BAR's variance nan

`exp(-w)` underflows to exactly zero in float64 above ~709 kT, and BAR's variance
formula divides by it. A single frame is enough: during validation one λ=0.4
frame had w_F = 870 kT and turned the whole leg's uncertainty into `nan`.

This is not a numerical accident to paper over — it means one sampled
configuration is astronomically improbable in the neighbouring state, i.e. that
pair has no usable overlap and needs *intermediate states*, not more sampling.
`bar_estimate` warns, records `n_extreme_work` and `max_abs_work` per pair, and
exposes `usable` / `pairs_without_uncertainty` so the orchestrator can gate on it
without re-deriving why. Left alone the `nan` propagates silently into the
reported total.

## Bulk SPC/E validation

The calculation that tests the whole chain: SPC/E has a published
mu_ex near -6.5 kcal/mol, so a result that misses it indicts the
protocol rather than the sample.

Two independently seeded morphologies, 100 waters, 8 LJ + 5 charge
states, 6k production steps per state (a deliberately cheap run --
9 minutes on a laptop, 4 ranks):

| | LJ leg | charge leg | mu_ex |
|---|---|---|---|
| morphology 0 | +2.513 | -9.308 | -6.795 +/- 0.517 |
| morphology 1 | +2.955 | -9.810 | -6.855 +/- 0.619 |
| combined | | | **-6.825**, 95% CI [-7.206, -6.444] |

The published -6.5 lies inside the interval. For comparison, direct
Widom insertion on the same water model gives -2.74 +/- 0.64 (README),
wrong by 3.8 kcal/mol -- so the alchemical path removes the dominant
error of the previous method, which was the point of building it.

The leg decomposition is physically sensible on its own terms: forming
the cavity costs about +2.7 kcal/mol and charging the inserted water in
it gains about -9.6.

This run reports `converged = False`, correctly: at two morphologies
the interval half-width is 0.38 against a 0.30 budget. It is a
validation, not a production number.

### What this run does *not* establish

* The between-morphology variance is unmeasured at M=2 in any useful
  sense -- one degree of freedom. The `var_between` reported is 0
  after clamping, which means "not resolved", not "zero".
* Density is not validated by the FEP path at all. The fixed-lambda
  states run NVT in a box built to a target density, so the volume is
  an input (see `_fep_cell_density`). Use the Widom bulk stage or an
  ordinary NPT run to check the water model's density.
* Ladder adequacy was checked only through estimator agreement at this
  resolution, not by refining the ladder and confirming the answer is
  unchanged.

### Two statistics defects this run exposed

Both were found because the run produced a number that could be
checked against a known answer, and neither was visible in the unit
tests as originally written.

**Convergence was tested on the wrong quantity.** The run reported
`converged = True` with `stderr = 0.030` against a 0.30 budget, while
its 95% interval spanned +/-0.38 -- wider than the budget it claimed to
meet. The gap is the t-quantile, 12.7 at one degree of freedom.
`FEPEstimate.converged` now tests the interval half-width, which is
what a reader means by "known to 0.3 kcal/mol".

**A variance floor was tried and rejected.** The two morphologies
landed 0.06 kcal/mol apart while each carried a 0.5-0.6 kcal/mol
internal error, so `sqrt(v_obs/M)` came out 13x below the sampling
noise measured inside those same cells. Flooring the total variance at
`v_within` looked obviously right. Monte Carlo against a known truth in
exactly that regime (sigma_between 0.05, sigma_within 0.55) refuted it:
the unfloored interval covers at 0.949 / 0.943 / 0.944 for M = 2, 3, 5
-- nominal -- while the floored version covers at 1.000. The
t-quantile already compensates for a variance estimate that came out
small by luck, and flooring the variance breaks that cancellation. The
lesson recorded in the code: the small `v_obs` was a signal about
which *interval* to trust, not about the variance being wrong.

The unit tests had covered only the between-dominated regime, which is
why they passed throughout. A within-dominated coverage test now runs
alongside them.

## Estimator implementation notes

All three run on the same data, which is what makes their agreement meaningful.

* **MBAR** (default). The `rerun` pass evaluates every stored frame at every
  lambda of its leg, giving the full `K x N` reduced-potential matrix. Lowest
  variance, and it yields the state-overlap matrix that diagnoses a bad ladder.
* **BAR** on neighbouring pairs, from the same matrix and cross-checked against
  the inline `compute fep` output.
* **TI** by central difference, `dU/dlambda ~ [U(l+d) - U(l-d)] / 2d`,
  integrated by Gauss-Legendre quadrature.

MBAR and BAR are statistically consistent estimators of the same quantity; TI
carries an additional quadrature error. Disagreement beyond the error bars is
therefore a ladder problem, and is reported as one.

Decorrelation uses `pymbar.timeseries` statistical inefficiency; the effective
sample count per state is reported, mirroring the existing `N_eff` discipline.

## Exact LAMMPS commands

Verified accepted by this build:

    pair_style   lj/cut/coul/long/soft 1 0.5 10.0 10.0
    kspace_style pppm 1.0e-4
    pair_coeff   i j eps sigma lambda        # every i<=j written explicitly

    variable dlam equal <lambda_next - lambda_this>
    compute  fepF all fep ${T} pair lj/cut/coul/long/soft lambda 1*2 3*4 v_dlam volume yes
    compute  fepQ all fep ${T} atom charge 3 v_dq
    fix      adap all adapt/fep 0 pair lj/cut/coul/long/soft lambda 1*2 3*4 v_lam scale yes

Two syntax constraints found by probing rather than by reading:

* `compute fep` requires a **`v_` variable**, not a numeric literal
  (`ERROR: Illegal variable in compute fep`, compute_fep.cpp:97).
* `tail yes` is **rejected** for soft styles: `ERROR: Compute fep tail when pair
  style does not compute tail corrections`. See the next section — this one has
  physical consequences.

### Mixing is written out, never inferred

`pair_modify mix arithmetic` cannot express a per-pair lambda. Every `i <= j`
pair coefficient is emitted explicitly, with ghost-host pairs carrying the
current lambda and host-host pairs pinned at `lambda = 1`.

Probed rather than assumed, and the news is good: the soft styles **refuse** to
mix two different lambdas — `ERROR: Pair lj/cut/soft different lambda values in
mix` (pair_lj_cut_soft.cpp:541). So a missing cross term is a hard failure at
setup, not a silently wrong coupling. Explicit emission is still required (that
error would otherwise stop every run), but it is backed by an engine-level check
rather than by our diligence alone.

Measured on a two-atom probe at `r = 3.5 A`, `eps = 0.2`, `sigma = 3.0`,
`n = 1`, `alpha_LJ = 0.5`:

    lambda = 0.0  ->  E = 0            (exactly, as required at the ghost end)
    lambda = 0.5  ->  E = -0.0940307
    lambda = 1.0  ->  E = -0.1914417

The `lambda = 0.5` value is not half of the `lambda = 1.0` value, confirming the
`alpha_LJ (1-lambda)^2` term is active in the denominator and the coupling is a
genuine soft core rather than a linear prefactor.

## The terms that do not cancel

**Tail correction.** The production templates run `pair_modify tail yes`; the
soft styles cannot. Dropping the tail is not acceptable silently, because it
shifts `mu_ex` by a fraction of a kcal/mol — the same order as the quantity the
saturation criterion resolves. The analytic isotropic correction for adding one
molecule to a cell of `N_j` atoms of each type `j` is

    dU_tail = (8 pi / 3V) * sum_j N_j eps_gj sigma_gj^3
                            [ (1/3)(sigma_gj/rc)^9 - (sigma_gj/rc)^3 ]

which is computed from the composition and added to leg 1. It is applied
identically to bulk and membrane so that whatever residual error it carries
cancels in the difference, exactly as the cutoff and kspace settings already do.

**Finite size.** One ghost in a small cell polarises its own periodic images
through PPPM. The effect largely cancels between membrane and bulk when both use
the same box size, which is enforced by the cache key.

**Ensemble.** Legs run in NPT at the production state point, so `dA` is strictly
`dG`. The distinction is `O(p dV)` for a single water — around 1e-4 kcal/mol at
1 atm, far below the error bars — and it cancels in the difference regardless.

## Morphology averaging

`n_morphologies` independently equilibrated cells, each from a distinct RNG seed
threaded through packing, annealing and the 21-step densification, so they are
genuinely independent structures rather than frames of one trajectory.

Two variances are reported separately, because they mean different things:

* **within-morphology** — sampling error of one cell's FEP, reducible by longer
  runs;
* **between-morphology** — real structural heterogeneity of the glassy matrix,
  reducible only by more cells.

In a glassy membrane the second usually dominates, in which case a longer run per
cell buys nothing and the honest response is more cells. The pooled estimate is a
mean over morphologies with the standard error over morphologies, and the report
states which term dominates rather than quoting a single number.

## Fallback if the FEP package is absent

Not needed for this machine — `lj/cut/soft`, `lj/cut/coul/long/soft`,
`coul/long/soft`, `compute fep`, `fix adapt/fep` and `fix adapt` are all present
in the Homebrew build. On a cluster build lacking the FEP package, leg 2 (linear
charge scaling with explicit per-state charges plus a `rerun` matrix) needs no
FEP package at all; only leg 1's soft core does. The runner probes for the styles
and fails with that message rather than at the first `pair_coeff`.
