# Stage interruption cleanup verification — 2026-11-28

The old stop function sent an interrupt to a process group, then waited only
for its launcher. A launcher exiting promptly could leave a signal-ignoring
child alive. A Linux reproduction of the old implementation returned in
0.008 seconds with the child still running.

Stages and checkpoint conversion now use `executor/process_lifecycle.py`.
On Linux, an isolated supervisor adopts orphaned descendants, including
workers in separate sessions, and reaps them before returning. Shutdown
escalates from interruption to termination and then forced termination.
Windows uses a private kill-on-close Job Object; the supervisor joins before
launching the command. Repeated interrupts cannot abandon cleanup or the
final job-status write. Ownership never relies on process names.

## Regression checks

- Linux with the existing LeRobot environment: 382 tests passed, none skipped.
- Windows with Python 3.13: 354 passed and 28 skipped for platform/dependency
  conditions. A temporary copy of `packaging` supplied a missing test dependency;
  the installed Python environments were not changed.
- New real-process regressions cover early launcher success/failure with an
  output-pipe-holding detached child, interrupted detached children, workers
  ignoring interrupt and termination, unrelated process survival, interruption
  during process creation, stage exceptions, repeated manager interrupts, and
  abrupt manager death. The manager-death case also passed on Windows.
- Other POSIX platforms retain process-group handling; Linux orphan adoption
  and Windows Job Objects are the verified ownership implementations.

## Actual cached-model checks on kbeeshome

SmolVLA used the existing `HuggingFaceVLA/smolvla_libero` snapshot
`6721902bc4d61e50a3bfdb11dfb4cb626f05d102`, LeRobot revision
`145cc7ea53e5864bd5efec4e9bb70ee799e47bd8`, and the existing `lerobot`
environment. No model downloads or environment installations were performed.
Two tasks in `libero_spatial` used batch 2 and asynchronous simulator workers.

| Check | Result |
| --- | --- |
| Repeated interrupt after actual nonzero control progress | Exit 130, status `interrupted`, shutdown 1.444 s, no tracked process remained |
| Separate late-rollout repeated interrupt | Status `interrupted`, shutdown 1.354 s, no tracked process remained |
| Normal completion, maximum 2 control steps | Exit 0, status `completed`, 4 requested episodes and 4 videos, no tracked process remained |

The automated runs tracked 90 distinct descendant identities over each whole
manager invocation, including short-lived environment/catalog probes; this is
not a count of simultaneously loaded policies. Process identities included
their creation times when checking for survivors.

These checks verify execution and cleanup, not policy performance. Verification
used a dedicated temporary source checkout and output directory. Existing
LeRobot/model sources, datasets, weights, and user checkpoints were preserved.
