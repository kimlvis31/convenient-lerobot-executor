# Verification

Automated checks live here so `executor/` contains runtime tools only. From the
project root, run:

```bash
python -B -m unittest discover -v
```

From another directory, provide both roots:

```bash
python -B -m unittest discover -s /path/to/convLexecutor/tests -t /path/to/convLexecutor -v
```

The standard-library checks cover configuration validation, question navigation,
output layout, environment selection, dependency provisioning and task isolation.
Tests requiring PyTorch, safetensors or an installed LeRobot skip when those
dependencies are absent. Run the complete suite in a compatible native environment
to exercise those checks as well. Mocked provisioning checks do not install a
simulator or establish that a model fits the machine.

Model integration reports belong in `reports/`. Each report distinguishes command
and parser coverage from actual training, checkpoint resume and simulator runs.
Short functional runs do not establish benchmark performance or every possible
combination of settings.

Keep machine-specific drivers, raw logs, temporary datasets, downloaded weights
and disposable runs in a separate verification workspace outside the checkout.
Retain sanitized results and model revisions before deleting owned temporary
assets. Preserve user datasets, checkpoints, original environments and sources.
