"""Train the existing TinyVLA action head on exported demonstrations."""
import argparse
import ast
from collections import defaultdict
from functools import lru_cache
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import pickle
import random
import subprocess
from types import SimpleNamespace

import numpy as np

from lerobot_data import action_chunk


def loader_program(source):
    end = source.index("    print('[3/5]")
    start = source.index("    tokenizer = AutoTokenizer.from_pretrained", end)
    processor_start = source.index("    processor = processor_type.from_pretrained", start)
    stop = source.index("\n", processor_start) + 1
    if "noise_scheduler" not in source[:end] or "torch.float32" not in source[:end]:
        raise ValueError("The verified float32 scheduler fix is required.")
    program = source[:end] + source[start:stop] + "\n    return locals()\n"
    ast.parse(program)
    return program


def load_data(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    train = {int(ep) for ep in manifest["train_episodes"]}
    validation = {int(ep) for ep in manifest["validation_episodes_excluded"]}
    if not train or train & validation:
        raise ValueError("Expected train episodes disjoint from validation.")
    episodes, seen = [], set()
    for record in manifest["episodes"]:
        episode = int(record["episode"])
        if episode not in train or episode in seen:
            raise ValueError("Unexpected or duplicated episode.")
        seen.add(episode)
        path = (directory / record["file"]).resolve()
        if path.parent != directory.resolve():
            raise ValueError("Episode file must be inside the exported data directory.")
        if not path.is_file():
            raise FileNotFoundError(path)
        episodes.append({**record, "path": path, "instruction": record.get("instruction", manifest["instruction"])})
    if seen != train or sum(ep["frames"] for ep in episodes) != manifest["total_frames"]:
        raise ValueError("Exported episode coverage or frame count differs from manifest.")
    return manifest, episodes


def episode_reader():
    verified = set()

    @lru_cache(maxsize=8)
    def read(path, digest, count):
        if path not in verified and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Exported data changed: {path}")
        with np.load(path, allow_pickle=False) as file:
            data = {key: file[key].copy() for key in ("external", "wrist", "states", "actions")}
        if count < 1 or data["states"].shape != (count, 8) or data["actions"].shape != (count, 7):
            raise ValueError("Invalid state/action dimensions.")
        if not np.isfinite(data["states"]).all() or not np.isfinite(data["actions"]).all():
            raise ValueError("Nonfinite state/action data.")
        for key in ("external", "wrist"):
            images = data[key]
            if images.dtype != np.uint8 or images.ndim != 4 or images.shape[0] != count or images.shape[-1] != 3:
                raise ValueError("Expected a sequence of HWC uint8 RGB images.")
        verified.add(path)
        return data

    return read


def fingerprint(manifest, episodes):
    records = [{key: value for key, value in ep.items() if key != "path"} for ep in episodes]
    data = json.dumps([manifest, records], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()


def model_fingerprint(directory):
    files = sorted([*directory.glob("*.safetensors"), *directory.glob("*.json")])
    # Include file sizes and timestamps to detect replacement of the model snapshot.
    return [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in files]


def run(args):
    if args.dataset_python:
        if not args.dataset_root:
            raise ValueError("--dataset-root is required with --dataset-python.")
        worker_path = Path(__file__).with_name("lerobot_data.py")
        command = [
            str(args.dataset_python), "-u", str(worker_path), "serve", "--selection", str(args.data),
            "--dataset-root", str(args.dataset_root), "--output", str(args.output.parent / "selection.json"),
        ]
        if args.libero_task:
            command.extend(("--libero-task", *args.libero_task))
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as worker:
            try:
                manifest, episodes = pickle.load(worker.stdout)

                def read_batch(samples):
                    pickle.dump(samples, worker.stdin, protocol=4)
                    worker.stdin.flush()
                    with np.load(BytesIO(pickle.load(worker.stdout)), allow_pickle=False) as batch:
                        return zip(*(batch[f"arr_{i}"] for i in range(5)))

                train(args, manifest, episodes, read_batch)
            except (EOFError, BrokenPipeError) as exc:
                raise RuntimeError("LeRobot data reader stopped. See its error in the training log.") from exc
            finally:
                worker.stdin.close()
        return

    if args.libero_task:
        raise ValueError("--libero-task requires --dataset-python.")
    manifest, episodes = load_data(args.data.expanduser().resolve())
    read_episode = episode_reader()

    def read_batch(samples):
        for episode_index, frame in samples:
            record = episodes[episode_index]
            episode = read_episode(record["path"], record["sha256"], record["frames"])
            chunk, mask = action_chunk(episode["actions"], frame)
            yield episode["external"][frame], episode["wrist"][frame], episode["states"][frame], chunk, mask

    train(args, manifest, episodes, read_batch)


def train(args, manifest, episodes, read_batch):
    import torch
    from PIL import Image
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a new training output directory: {output}")
    by_task = defaultdict(list)
    for index, episode in enumerate(episodes):
        by_task[episode["instruction"]].append(index)
    tasks = list(by_task)
    if manifest.get("tasks") and set(tasks) != set(manifest["tasks"]):
        raise ValueError("Expected data for every selected LIBERO task.")
    steps = args.steps
    if args.updates_per_task is not None:
        steps = args.updates_per_task * len(tasks)
    if (not tasks or args.batch_size < 1 or steps < len(tasks) or
            not math.isfinite(args.lr) or args.lr <= 0 or args.save_freq < 1):
        raise ValueError("Use a positive batch size and learning rate and at least one update per task.")
    total = manifest["total_frames"]
    if total < args.batch_size:
        raise ValueError("The dataset is smaller than the requested batch.")
    output.mkdir(parents=True)
    workspace = args.workspace.expanduser().resolve()
    checkpoint_source = args.init_checkpoint.expanduser().resolve() if args.init_checkpoint else None
    resumed = torch.load(args.resume_state, map_location="cpu", weights_only=True) if args.resume_state else None
    model_source = checkpoint_source or args.model or workspace / "models/Llava-Pythia-700M"
    if resumed:
        model_source = Path(resumed["metadata"]["model_source"])
    model_source = model_source.expanduser().resolve()
    source_path = (args.loader or workspace / "runtime/tinyvla_loader.py").expanduser().resolve()
    source = source_path.read_text(encoding="utf-8")
    metadata = {
        "version": 1, "dataset": fingerprint(manifest, episodes),
        "loader_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "trainer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_source": str(model_source), "model_files": model_fingerprint(model_source),
        "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
        "steps": steps, "updates_per_task": args.updates_per_task,
    }
    if resumed and resumed["metadata"] != metadata:
        raise ValueError("Resume requires the same dataset, model snapshot, trainer, and training configuration.")
    scope = {"__name__": "tinyvla_training_loader", "__file__": str(source_path)}
    exec(compile(loader_program(source), str(source_path), "exec"), scope)
    scope["arguments"] = lambda: SimpleNamespace(
        workspace=workspace, repo=workspace / "sources/TinyVLA",
        model=model_source,
        output=output / "loader.json", seed=args.seed,
    )
    ctx = scope["main"]()
    torch.set_default_dtype(torch.float32)
    model, cfg, tokenizer = ctx["model"], ctx["cfg"], ctx["tokenizer"]
    if model.noise_scheduler.alphas_cumprod.dtype != torch.float32:
        raise RuntimeError("Expected float32 diffusion coefficients.")
    model.requires_grad_(False)
    model.eval()
    for name in ("embed_out", "proj_to_action"):
        module = getattr(model, name)
        module.float().requires_grad_(True)
        module.train()
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named or any(not name.startswith(("embed_out.", "proj_to_action.")) for name, _ in named):
        raise RuntimeError("Only action-head parameters may be trained.")
    if checkpoint_source or resumed:
        saved_head = resumed["head"] if resumed else load_file(str(checkpoint_source / "action_head.safetensors"))
        if set(saved_head) != {name for name, _ in named}:
            raise ValueError("The saved action head does not match the trainable parameters.")
        with torch.no_grad():
            for name, parameter in named:
                tensor = saved_head[name]
                if tensor.shape != parameter.shape or not torch.isfinite(tensor).all():
                    raise ValueError(f"Invalid saved action-head tensor: {name}")
                parameter.copy_(tensor)
        if not resumed:
            print(f"Continuing from {checkpoint_source} with a new optimizer.", flush=True)
    initial = resumed["initial"] if resumed else {
        name: p.detach().flatten()[:32].float().cpu().clone() for name, p in named
    }
    params = [p for _, p in named]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    marker = ctx["DEFAULT_IMAGE_TOKEN"]
    if getattr(cfg, "mm_use_im_start_end", False):
        marker = ctx["DEFAULT_IM_START_TOKEN"] + marker + ctx["DEFAULT_IM_END_TOKEN"]
    task_ids, offsets = {}, {}
    for instruction, indices in by_task.items():
        conversation = ctx["conv_templates"]["pythia"].copy()
        conversation.append_message(conversation.roles[0], marker + "\n" + instruction)
        conversation.append_message(conversation.roles[1], None)
        task_ids[instruction] = ctx["tokenizer_image_token"](
            conversation.get_prompt() + " <|endoftext|>", tokenizer,
            ctx["IMAGE_TOKEN_INDEX"], return_tensors="pt",
        ).unsqueeze(0).repeat(args.batch_size, 1).to("cuda")
        offsets[instruction] = np.concatenate(([0], np.cumsum([episodes[i]["frames"] for i in indices])))
        if offsets[instruction][-1] < args.batch_size:
            raise ValueError(f"Not enough frames for {instruction}")
    rng = np.random.default_rng(args.seed)
    task_steps = dict.fromkeys(tasks, 0)
    losses, used = [], set()
    start = 0
    if resumed:
        start = resumed["step"]
        tasks, task_steps = resumed["tasks"], resumed["task_steps"]
        losses, used = resumed["losses"], set(resumed["used"])
        optimizer.load_state_dict(resumed["optimizer"])
        rng.bit_generator.state = resumed["sampler_rng"]
        random.setstate(resumed["python_rng"])
        torch.set_rng_state(resumed["torch_rng"])
        torch.cuda.set_rng_state_all(resumed["cuda_rng"])
        if not 1 <= start <= steps or len(losses) != start or sum(task_steps.values()) != start:
            raise ValueError("Invalid training position in resume state.")
        del saved_head, resumed

    state_file = output / "checkpoints" / "training_state.pt"
    state_file.parent.mkdir()

    def save_state(step):
        temporary = state_file.with_suffix(".tmp")
        torch.save({
            "metadata": metadata, "step": step,
            "head": {name: param.detach().cpu() for name, param in named},
            "optimizer": optimizer.state_dict(), "initial": initial,
            "tasks": tasks, "task_steps": task_steps, "losses": losses, "used": sorted(used),
            "sampler_rng": rng.bit_generator.state, "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        }, temporary)
        temporary.replace(state_file)
        if args.keep_intermediate_checkpoints and step < steps:
            snapshot = state_file.with_name(f"training_state_{step:06d}.pt")
            os.link(state_file, snapshot)

    torch.cuda.reset_peak_memory_stats()
    print("train_episodes:", len(episodes), "frames:", total, flush=True)
    print("trainable_parameters:", sum(p.numel() for p in params), flush=True)
    print("batch_size:", args.batch_size, "steps:", steps, "tasks:", len(tasks),
          "updates_per_task:", args.updates_per_task, flush=True)
    print(f"step:{start}/{steps} training_start", flush=True)
    if start == steps:
        save_state(start)
    for step in range(start + 1, steps + 1):
        if (step - 1) % len(tasks) == 0:
            rng.shuffle(tasks)
        instruction = tasks[(step - 1) % len(tasks)]
        task_steps[instruction] += 1
        ids = task_ids[instruction]
        boundaries = offsets[instruction]
        indices = rng.choice(boundaries[-1], size=args.batch_size, replace=False)
        samples = []
        for flat in indices:
            slot = int(np.searchsorted(boundaries[1:], flat, side="right"))
            frame = int(flat - boundaries[slot])
            episode_index = by_task[instruction][slot]
            samples.append((episode_index, frame))
        used.update(samples)
        external, wrist, states, actions, padding = zip(*read_batch(samples))
        external = [Image.fromarray(image) for image in external]
        wrist = [Image.fromarray(image) for image in wrist]
        images = ctx["process_images"](external + wrist, ctx["processor"], cfg)
        if not isinstance(images, torch.Tensor) or images.shape[0] != 2 * args.batch_size:
            raise RuntimeError("Camera batch size mismatch.")
        images = images.to(device="cuda", dtype=torch.bfloat16)
        state = torch.from_numpy(np.stack(states)).to(device="cuda", dtype=torch.float32)
        target = torch.from_numpy(np.stack(actions)).to(device="cuda", dtype=torch.float32)
        is_pad = torch.from_numpy(np.stack(padding)).to("cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = model(
                input_ids=ids, attention_mask=ids.ne(tokenizer.pad_token_id),
                images=images[:args.batch_size], images_r=images[args.batch_size:],
                states=state, actions=target, is_pad=is_pad, eval=False, return_dict=True,
            )
            loss = result.loss
            if isinstance(loss, dict):
                loss = loss["loss"]
        if loss.numel() != 1 or not bool(torch.isfinite(loss).all()):
            raise RuntimeError("Nonfinite training loss.")
        loss.backward()
        grads = [p.grad for p in params if p.grad is not None]
        if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
            raise RuntimeError("Missing or nonfinite gradients.")
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise RuntimeError("Frozen parameters received gradients.")
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError("Nonfinite gradient norm.")
        optimizer.step()
        value = float(loss.detach().float().cpu())
        losses.append(value)
        if step == 1 or step % 20 == 0 or step == steps:
            print(f"step:{step}/{steps} loss:{value:.6f} grad_norm:{float(norm):.4f} samples:{step*args.batch_size} peak_GiB:{torch.cuda.max_memory_allocated()/1024**3:.2f}", flush=True)
        if step % args.save_freq == 0 or step == steps:
            save_state(step)
    if args.updates_per_task is not None and set(task_steps.values()) != {args.updates_per_task}:
        raise RuntimeError("Unexpected number of completed updates per task.")
    changed = [name for name, p in named if not torch.equal(initial[name], p.detach().flatten()[:32].float().cpu())]
    if not changed or not all(bool(torch.isfinite(p).all()) for p in params):
        raise RuntimeError("Expected finite and changed head weights before saving.")
    final_checkpoint = output / "pretrained_model"
    checkpoint = output / ".pretrained_model.tmp"
    checkpoint.mkdir()
    # Keep a small recoverable head checkpoint even if full-model serialization fails.
    save_file({name: p.detach().cpu().contiguous() for name, p in named}, str(checkpoint / "action_head.safetensors"))
    model.save_pretrained(checkpoint, safe_serialization=True, max_shard_size="10GB")
    tokenizer.save_pretrained(checkpoint)
    ctx["processor"].save_pretrained(checkpoint)
    with safe_open(str(checkpoint / "model.safetensors"), framework="pt", device="cpu") as saved:
        for name, param in named:
            if not torch.equal(saved.get_tensor(name), param.detach().cpu()):
                raise RuntimeError(f"Saved action-head tensor differs: {name}")
    checkpoint.rename(final_checkpoint)
    repo = workspace / "sources/TinyVLA"
    report = {
        "status": "TINYVLA_HEAD_TRAINING_SAVED", "steps": steps,
        "batch_size": args.batch_size, "processed_samples": steps * args.batch_size,
        "unique_sample_starts": len(used), "learning_rate": args.lr, "seed": args.seed,
        "training_steps_by_instruction": task_steps,
        "trainable_parameters": sum(p.numel() for p in params), "changed_parameter_samples": changed,
        "dataset_manifest": manifest, "losses": losses, "optimizer_state_saved": True,
        "resume_state": str(state_file), "resumed_from_step": start,
        "initialized_from_checkpoint": str(checkpoint_source) if checkpoint_source else None,
        "checkpoint": str(final_checkpoint),
        "source_commit": scope["revision"](repo), "loader_sha256": metadata["loader_sha256"],
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    (output / "training_result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("TINYVLA_HEAD_TRAINING_SAVED", flush=True)
    print("checkpoint:", final_checkpoint, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--loader", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-python", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--libero-task", nargs=2, metavar=("SUITE", "TASK_ID"))
    parser.add_argument("--output", type=Path, required=True)
    origin = parser.add_mutually_exclusive_group()
    origin.add_argument("--init-checkpoint", type=Path)
    origin.add_argument("--resume-state", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--batch_size", type=int, default=16)
    updates = parser.add_mutually_exclusive_group()
    updates.add_argument("--steps", type=int, default=200)
    updates.add_argument("--updates-per-task", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-freq", type=int, default=100)
    parser.add_argument("--keep-intermediate-checkpoints", action="store_true",
                        help="Keep a step-numbered copy of each saved state before the final step.")
    parser.add_argument("--seed", type=int, default=0)
    run(parser.parse_args())
