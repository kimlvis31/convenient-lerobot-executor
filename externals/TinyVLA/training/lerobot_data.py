"""Select LIBERO demonstrations, stream training batches, or export episodes."""

import argparse
from collections import defaultdict
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import pickle
import re
import sys

import numpy as np


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def normalize_instruction(text):
    return " ".join(text.lower().split())


def source_selection(selection_path, dataset_root):
    if selection_path.is_file():
        return json.loads(selection_path.read_text(encoding="utf-8"))

    repo = "lerobot/libero"
    metadata = dataset_root / ".cache/huggingface/download/meta/info.json.metadata"
    revision = None
    if metadata.is_file() and (dataset_root / "meta/info.json").is_file():
        lines = metadata.read_text(encoding="utf-8").splitlines()
        if lines and re.fullmatch(r"[0-9a-f]{40}", lines[0]):
            revision = lines[0]
    if revision is None:
        from huggingface_hub import HfApi

        revision = HfApi().dataset_info(repo).sha
        if not revision:
            raise RuntimeError(f"Could not resolve the dataset revision for {repo}.")
    print(f"Using {repo} at {revision}; selecting a new holdout split.", flush=True)
    return {"dataset_repo": repo, "dataset_revision": revision}


def episode_labels(meta, repo_id, revision, dataset_root):
    task_names = {
        int(index): normalize_instruction(name)
        for name, index in meta.tasks["task_index"].items()
    }
    rows = list(meta.episodes)
    if all("tasks" in row or "task_index" in row for row in rows):
        for row in rows:
            labels = row.get("tasks")
            if labels is None:
                indices = row["task_index"]
                if not isinstance(indices, (list, tuple)):
                    indices = [indices]
                labels = [task_names[int(index)] for index in indices]
            yield int(row["episode_index"]), {normalize_instruction(label) for label in labels}
        return

    # Some dataset revisions store the task ID only on frame rows.
    from huggingface_hub import snapshot_download
    import pyarrow.dataset as arrow_dataset

    snapshot_download(
        repo_id, repo_type="dataset", revision=revision,
        local_dir=dataset_root, allow_patterns=["data/**/*.parquet"],
    )
    grouped = defaultdict(set)
    frames = arrow_dataset.dataset(dataset_root / "data", format="parquet")
    for batch in frames.to_batches(columns=["episode_index", "task_index"]):
        columns = batch.to_pydict()
        for episode, index in zip(columns["episode_index"], columns["task_index"]):
            grouped[int(episode)].add(task_names[int(index)])
    for row in rows:
        episode = int(row["episode_index"])
        yield episode, grouped[episode]


def select_four_suites(selection_path, destination, dataset_root, target=None):
    from libero.libero import benchmark
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    selection = source_selection(selection_path, dataset_root)
    meta = LeRobotDatasetMetadata(
        selection["dataset_repo"], root=dataset_root, revision=selection["dataset_revision"],
    )
    tasks = {}
    if target is not None and target[0] not in SUITES:
        raise ValueError(f"Unsupported LIBERO suite: {target[0]}")
    for suite in (SUITES if target is None else (target[0],)):
        env = benchmark.get_benchmark_dict()[suite]()
        if env.n_tasks != 10:
            raise ValueError(f"Expected ten tasks in {suite}.")
        task_ids = range(env.n_tasks)
        if target is not None:
            if target[1] not in task_ids:
                raise ValueError(f"Invalid task ID for {suite}: {target[1]}")
            task_ids = (target[1],)
        for task_id in task_ids:
            instruction = normalize_instruction(env.get_task(task_id).language)
            if instruction in tasks:
                raise ValueError(f"Duplicate instruction: {instruction}")
            tasks[instruction] = {"suite": suite, "task_id": task_id, "episodes": []}
    for episode, labels in episode_labels(
        meta, selection["dataset_repo"], selection["dataset_revision"], dataset_root,
    ):
        matches = labels & tasks.keys()
        if matches:
            if len(labels) != 1:
                raise ValueError(f"Expected a single task in episode {episode}.")
            tasks[matches.pop()]["episodes"].append(episode)
    train, validation = [], []
    for instruction, task in tasks.items():
        episodes = sorted(task["episodes"])
        if len(episodes) <= 5:
            raise ValueError(f"Insufficient demonstrations: {instruction}")
        held_out = episodes[-5:]
        # Preserve the earlier holdout for the cream-cheese task.
        if instruction == normalize_instruction(selection.get("task", "")):
            held_out = sorted(map(int, selection["validation_episodes"]))
            if len(set(held_out)) != 5 or not set(held_out) <= set(episodes):
                raise ValueError("The earlier validation split does not match this dataset.")
        task["episodes"] = sorted(set(episodes) - set(held_out))
        train.extend(task["episodes"])
        validation.extend(held_out)
    result = {key: selection[key] for key in ("dataset_repo", "dataset_revision")}
    result.update(tasks=tasks, train_episodes=train, validation_episodes=validation)
    with destination.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)


def read_dataset(selection, dataset_root, *, with_actions=False):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    train = [int(ep) for ep in selection["train_episodes"]]
    validation = {int(ep) for ep in selection["validation_episodes"]}
    if not train or len(set(train)) != len(train) or set(train) & validation:
        raise ValueError("Expected distinct training episodes, separate from validation.")
    dataset = LeRobotDataset(
        repo_id=selection["dataset_repo"], root=dataset_root,
        revision=selection["dataset_revision"], episodes=train,
        video_backend="pyav",
    )
    # Read the plain columns without decoding images or assuming episode ordering.
    fields = ["episode_index", "frame_index"]
    if with_actions:
        fields.append("action")
    columns = dataset.hf_dataset.data.select(fields).to_pydict()
    rows = defaultdict(list)
    for index, (episode, frame) in enumerate(zip(columns["episode_index"], columns["frame_index"])):
        rows[int(episode)].append((int(frame), index))
    if set(rows) != set(train):
        raise ValueError("The loaded episodes differ from the selected training split.")
    for episode, frames in rows.items():
        frames.sort()
        if [frame for frame, _ in frames] != list(range(len(frames))):
            raise ValueError(f"Episode {episode} has missing or duplicated frames.")
    return dataset, rows, columns.get("action")


def episode_instructions(selection):
    tasks = selection.get("tasks")
    if tasks is None:
        return {int(episode): selection["task"] for episode in selection["train_episodes"]}
    pairs = [(int(episode), instruction) for instruction, task in tasks.items() for episode in task["episodes"]]
    instructions = dict(pairs)
    if len(instructions) != len(pairs) or set(instructions) != set(map(int, selection["train_episodes"])):
        raise ValueError("Each training episode must belong to exactly one selected task.")
    return instructions


def camera_array(value):
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB image, got {image.shape}")
    if image.dtype != np.uint8:
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Expected finite camera values in [0, 1].")
        image = np.rint(image * 255).astype(np.uint8)
    return image.transpose(1, 2, 0)


def action_chunk(actions, frame):
    if not 0 <= frame < len(actions):
        raise IndexError(frame)
    positions = frame + np.arange(16)
    return actions[np.minimum(positions, len(actions) - 1)], positions >= len(actions)


def export(selection_path, destination, dataset_root=None):
    import torch

    selection_path = Path(selection_path).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    instructions = episode_instructions(selection)
    data_root = dataset_root or selection_path.parent / "dataset"
    dataset, rows, _ = read_dataset(selection, data_root)
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for number, (episode, ordered) in enumerate(sorted(rows.items()), 1):
        external, wrist, states, actions = [], [], [], []
        instruction = instructions[episode]
        for _, index in ordered:
            item = dataset[index]
            if int(torch.as_tensor(item["episode_index"]).item()) != episode:
                raise ValueError("Dataset row and episode columns disagree.")
            if normalize_instruction(item["task"]) != normalize_instruction(instruction):
                raise ValueError("Unexpected task in training split.")
            external.append(camera_array(torch.as_tensor(item["observation.images.image"]).detach().cpu()))
            wrist.append(camera_array(torch.as_tensor(item["observation.images.image2"]).detach().cpu()))
            state = torch.as_tensor(item["observation.state"]).float().cpu().numpy()
            action = torch.as_tensor(item["action"]).float().cpu().numpy()
            if state.shape != (8,) or action.shape != (7,):
                raise ValueError("Expected one 8D state and 7D action per frame.")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError("Nonfinite training target or state.")
            states.append(state)
            actions.append(action)
        path = destination / f"episode_{episode:06d}.npz"
        np.savez_compressed(
            path, external=np.stack(external), wrist=np.stack(wrist),
            states=np.stack(states), actions=np.stack(actions),
        )
        records.append({"episode": episode, "instruction": instruction,
                        "frames": len(ordered), "file": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print(f"export:{number}/{len(rows)} episode:{episode} frames:{len(ordered)}", flush=True)
    manifest = {
        "status": "TINYVLA_TRAIN_DATA_READY", "instruction": selection.get("task"), "tasks": selection.get("tasks"),
        "dataset_repo": selection["dataset_repo"], "dataset_revision": selection["dataset_revision"],
        "dataset_fps": dataset.fps, "train_episodes": list(map(int, selection["train_episodes"])),
        "validation_episodes_excluded": sorted(map(int, selection["validation_episodes"])), "episodes": records,
        "total_frames": sum(item["frames"] for item in records),
        "state_action_normalization": "identity", "gripper_sign_changed": False,
        "camera_rotation_added": False,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("TINYVLA_TRAIN_DATA_READY", "frames:", manifest["total_frames"], flush=True)



def serve(selection_source, dataset_root, selection_output, output, target=None):
    previous = json.loads(selection_source.read_text(encoding="utf-8")) if selection_source.is_file() else {}
    if previous.get("tasks"):
        selected_tasks = {(task["suite"], task["task_id"]) for task in previous["tasks"].values()}
        expected_tasks = {target} if target else {(suite, task) for suite in SUITES for task in range(10)}
        if selected_tasks != expected_tasks:
            raise ValueError("The previous training split does not match the requested tasks.")
        with selection_output.open("x", encoding="utf-8") as file:
            json.dump(previous, file, indent=2, ensure_ascii=False)
    else:
        select_four_suites(selection_source, selection_output, dataset_root, target=target)
    selection = json.loads(selection_output.read_text(encoding="utf-8"))
    episode_tasks = episode_instructions(selection)
    dataset, rows, actions = read_dataset(selection, dataset_root, with_actions=True)
    episodes, indices, targets = [], [], []
    for episode, frame_rows in sorted(rows.items()):
        episodes.append({
            "episode": episode, "frames": len(frame_rows), "instruction": episode_tasks[episode],
        })
        indices.append([row for _, row in frame_rows])
        targets.append(np.asarray([actions[row] for _, row in frame_rows], dtype=np.float32))

    manifest = {**selection, "total_frames": sum(record["frames"] for record in episodes)}
    pickle.dump((manifest, episodes), output)
    output.flush()
    while True:
        try:
            samples = pickle.load(sys.stdin.buffer)
        except EOFError:
            return
        batch = []
        for episode_index, frame in samples:
            item = dataset[indices[episode_index][frame]]
            record = episodes[episode_index]
            if int(item["episode_index"]) != record["episode"] or normalize_instruction(item["task"]) != record["instruction"]:
                raise ValueError("Dataset row and selected task disagree.")
            cameras = [camera_array(item[key]) for key in ("observation.images.image", "observation.images.image2")]
            chunk, padding = action_chunk(targets[episode_index], frame)
            state = np.asarray(item["observation.state"], dtype=np.float32)
            if state.shape != (8,) or chunk.shape != (16, 7):
                raise ValueError("Unexpected state or action dimensions.")
            if not np.isfinite(state).all() or not np.isfinite(chunk).all():
                raise ValueError("Nonfinite state or action values.")
            batch.append((*cameras, state, chunk, padding))
        buffer = BytesIO()
        np.savez(buffer, *map(np.stack, zip(*batch)))
        pickle.dump(buffer.getvalue(), output, protocol=4)
        output.flush()



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for operation in ("select", "export", "serve"):
        command = commands.add_parser(operation)
        command.add_argument("--selection", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--dataset-root", type=Path, required=operation != "export")
        if operation != "export":
            command.add_argument("--libero-task", nargs=2, metavar=("SUITE", "TASK_ID"))
    args = parser.parse_args()
    if args.command == "export":
        export(args.selection, args.output, args.dataset_root)
        return
    target = (args.libero_task[0], int(args.libero_task[1])) if args.libero_task else None
    if args.command == "select":
        select_four_suites(args.selection, args.output, args.dataset_root, target=target)
        return
    # Reserve stdout for the private pipe. Library progress goes to stderr.
    with os.fdopen(os.dup(sys.stdout.fileno()), "wb") as output:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        serve(args.selection, args.dataset_root, args.output, output, target=target)


if __name__ == "__main__":
    main()
