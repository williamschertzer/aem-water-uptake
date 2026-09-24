# Restart after interrupted dry equilibration

For `aemwater run`, `--resume-minimized` reuses `dry/min.data` and
`dry/typed_chain.pkl`, skips chain generation, typing, packing and minimization,
and regenerates the equilibration input with the requested configuration.
It restarts equilibration from its beginning, not the failed timestep.

The prior `run_config.yaml` must exist and its polymer settings must match.
The option rejects `--force`, existing uptake checkpoints, completed `dry.data`,
and missing or unreadable minimized-system checkpoints. Existing equilibration
inputs, logs, density records and convergence reports are copied into a dated
`dry/equilibration_attempts/` directory before equilibration is rerun.
Normal convergence checks still run before uptake begins.

From the repository root, with the aem environment active:

```bash
python -m aemwater.cli run \
  --config examples/im_peek_small_screening.yaml \
  --workdir runs/im_peek_small_screening \
  --bulk-mu-ex -6.5 --bulk-stderr 0 \
  --resume-minimized
```

Here the reference is a fixed user-selected value; `--bulk-stderr 0` is not
a measured uncertainty. Edit `md.pdamp` in the supplied configuration to change
pressure coupling. This option does not change that setting automatically.
After successful dry equilibration, omit `--resume-minimized` for ordinary
uptake resumes. The `prepare` and `campaign` commands do not expose this option.
