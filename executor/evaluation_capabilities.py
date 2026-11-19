"""Read-only checks for known evaluation contracts, without importing simulators."""

from pathlib import Path


UNITREE_REPOSITORY = "lerobot/unitree-g1-mujoco"
UNITREE_REVISION = "b00f8b114c3d5d98b44b7ad0285d631af73f4bc5"
UNITREE_SOURCE = (
    "https://huggingface.co/lerobot/unitree-g1-mujoco/blob/"
    f"{UNITREE_REVISION}/env.py"
)
FINEART_REPOSITORY = "lerobot/FineART-VLA-subtask-midtrained"


def hub_repository(reference):
    """Read LeRobot's repository@revision:file reference without fetching it."""
    if not isinstance(reference, str):
        return None
    return reference.strip().partition("@")[0].partition(":")[0]


def _unitree_revision(benchmark, extra):
    """Return the requested stock module's revision; None means a custom module."""
    if benchmark != "unitree_g1_mujoco":
        return None
    reference = UNITREE_REPOSITORY
    for value in extra:
        if isinstance(value, str) and value.startswith("--env.hub_path="):
            reference = value.partition("=")[2]
    reference = reference.strip()
    filename = reference.partition(":")[2] or "env.py"
    if hub_repository(reference) != UNITREE_REPOSITORY or filename != "env.py":
        return None
    return reference.partition("@")[2].partition(":")[0]


def _inspected_unitree_revision(revision):
    return revision in ("", "main") or (
        7 <= len(revision) <= len(UNITREE_REVISION) and UNITREE_REVISION.startswith(revision)
    )


def stock_evaluation_issue(benchmark, extra=()):
    """Identify the inspected Unitree transport, leaving custom task modules usable.

    The inspected upstream environment intentionally receives motor commands over
    an external transport. Installing packages cannot give its step(action) a
    task controller, reward, or episode termination contract. A real replacement
    can be supplied with LeRobot's existing env.hub_path option or a plugin type.
    Explicit revisions outside the inspected source receive a notice, not a ban.
    """
    revision = _unitree_revision(benchmark, extra)
    if revision is None or not _inspected_unitree_revision(revision):
        return None
    return (
        "Task Adapter Required: the stock Unitree G1 MuJoCo environment receives motor commands "
        "through external DDS (Data Distribution Service); its step(action) does not apply the "
        "policy action and does not report task success or episode termination. Automatic package "
        "installation cannot supply this task contract. Use a task-specific environment adapter "
        "with matching observations, action control, success and termination, supplied as a "
        "LeRobot environment plugin or --env.hub_path=owner/task-environment. "
        f"Inspected source: {UNITREE_SOURCE}"
    )


def evaluation_notices(model, evaluation):
    """Explain a known public-checkpoint mismatch without banning adapted models."""
    extra = evaluation.get("extra", ())
    revision = _unitree_revision(evaluation.get("env"), extra)
    if revision is not None and not _inspected_unitree_revision(revision):
        return (
            f"Evaluation capability unverified: Unitree revision {revision} differs from the inspected "
            "simulator transport. Confirm that this module applies policy actions and defines task "
            f"success and episode termination. Inspected source: {UNITREE_SOURCE}",
        )
    if model != "fineart_vla" or evaluation.get("env") != "aloha":
        return ()
    if any(isinstance(value, str) and value.startswith("--env.hub_path=") and value.partition("=")[2].strip()
           for value in extra):
        return ()
    policy = evaluation.get("policy")
    cached = (isinstance(policy, str) and
              "models--lerobot--FineART-VLA-subtask-midtrained" in Path(policy).parts)
    if hub_repository(policy) != FINEART_REPOSITORY and not cached:
        return ()
    return (
        "Checkpoint compatibility: the published FineART checkpoint uses five arm joints and "
        "two gripper values per arm; stock ALOHA uses six arm joints and one normalized gripper. "
        "Equal 14-value shapes do not establish compatible joint order, units or control. "
        "Use a checkpoint fine-tuned for this simulator or a matching robot environment; "
        "camera renaming alone cannot fix joint semantics. "
        f"Checkpoint: https://huggingface.co/{FINEART_REPOSITORY}",
    )
