"""Launch local Conda commands and record their measured progress."""

import ast
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import os
from pathlib import Path
from pprint import pformat
import re
import runpy
import shutil
import signal
import sys
import subprocess
import time

from adapters import Stage, read_json, require


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NATIVE_STEP = re.compile(r"\bstep:\s*(\d+(?:\.\d+)?)([KMB]?)\b(?![/\d.])", re.I)
TINY_STEP = re.compile(r"\bstep:(\d+)/(\d+)\b")
CHECKPOINT_STEP = re.compile(r"\bCheckpoint policy after step (\d+)\b")
PARALLEL_EPISODES = re.compile(r"^\[Eval progress\] Episodes completed: (\d+)$")
EVAL_BATCH = re.compile(r"Stepping through eval batches[^|]*\|[^|]*\|\s*(\d+)\s*/\s*(\d+)")


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def conda_executable(explicit):
    path = explicit or os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not path:
        location = Path("Scripts/conda.exe") if os.name == "nt" else Path("bin/conda")
        candidates = [Path.home() / name / location for name in ("miniconda3", "anaconda3", "miniforge3")]
        path = next((str(candidate) for candidate in candidates if candidate.is_file()), None)
        if not path:
            raise FileNotFoundError("Use a terminal with Conda on PATH or specify its executable with --conda.")
    return path


def conda_command(conda, name, arguments):
    return [conda, "run", "--no-capture-output", "-n", name, *arguments]


def available_environments(conda, root):
    if not conda:
        return set()
    try:
        result = subprocess.run([conda, "env", "list", "--json"], cwd=root, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=15, check=True)
        return {Path(path).name for path in json.loads(result.stdout)["envs"]}
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return set()


def native_policies(conda, name, root):
    if not conda:
        return []
    probe = """
import json
import lerobot.policies
from lerobot.configs import PreTrainedConfig

policies = PreTrainedConfig.get_known_choices()
print("MANAGER_POLICIES " + json.dumps(sorted(
    name for name, config in policies.items()
    if config.__module__.startswith("lerobot.policies.")
)))
"""
    try:
        result = subprocess.run(conda_command(conda, name, ["python", "-B", "-c", probe]), cwd=root,
                                text=True, encoding="utf-8", errors="replace", capture_output=True,
                                timeout=20, check=True)
        for line in reversed(result.stdout.splitlines()):
            if line.startswith("MANAGER_POLICIES "):
                return json.loads(line.removeprefix("MANAGER_POLICIES "))
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    return []


def resolve_environment(conda, name, root):
    probe = Path(__file__).with_name("provenance.py")
    result = subprocess.run(conda_command(conda, name, ["python", "-B", str(probe)]), cwd=root,
                            text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=90)
    for line in reversed(result.stdout.splitlines()):
        if result.returncode == 0 and line.startswith("MANAGER_ENV "):
            return json.loads(line.removeprefix("MANAGER_ENV "))
    raise RuntimeError(f"Could not inspect Conda environment {name}.\n{result.stderr.strip() or result.stdout.strip()}")


def resolve_arguments(arguments, environments):
    resolved = []
    for argument in arguments:
        key, tag, name = argument.partition("@python:")
        if tag and key in ("", "--policy.inference_python="):
            argument = key + environments[name]["python"]
        resolved.append(argument)
    return resolved


def styled(text, style):
    if sys.stdout.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb":
        return f"\033[{style}m{text}\033[0m"
    return text


def seconds(value):
    if value is None:
        return "N/A"
    mins, secs = divmod(max(0, round(value)), 60)
    hours, mins = divmod(mins, 60)
    days, hours = divmod(hours, 24)
    clock = f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{days}d {clock}" if days else clock


def status_line(percent, count, total, stage, eta, average=None):
    current = count if count is not None else "N/A"
    target = total if total is not None else "N/A"
    parts = [f"[Status] Completion: {percent:.3f} %"]
    if stage == "eval":
        parts.append(f"{current}/{target} Episodes")
        speed = f"{average:.2f} s" if average is not None else "N/A"
        parts.append(f"Seconds Per Episode: {speed}")
    else:
        parts.append(f"{current}/{target} Steps")
    return " | ".join([*parts, f"Estimated Time of Completion: {seconds(eta)}"])


class Progress:
    def __init__(self, stage, settings, started_at=None):
        self.stage = stage
        self.count = stage.start
        self.total = stage.units
        self.anchor = None
        self.baseline = None
        self.started_at = started_at
        self.rate = None
        self.group = 0
        self.last_batch = None
        self.episodes = settings.get("eval", {}).get("episodes", 0)
        self.batch = settings.get("eval", {}).get("batch", 1)
        self.parallel = settings.get("eval", {}).get("parallel_tasks", 1) > 1

    def parse(self, line):
        if self.stage.name == "eval":
            if self.parallel:
                match = PARALLEL_EPISODES.fullmatch(line)
                return int(match[1]) if match else None
            match = EVAL_BATCH.search(line)
            if not match:
                return None
            count, total = map(int, match.groups())
            if total != self.stage.eval_batches or not 0 <= count <= total:
                return None
            if count == 0 and self.last_batch == total:
                self.group += 1
            if count < (self.last_batch or 0) and count != 0:
                return None
            self.last_batch = count
            return self.group * self.episodes + min(self.episodes, count * self.batch)
        if m := TINY_STEP.search(line):
            current, total = map(int, m.groups())
            return current if total == self.total else None
        if m := CHECKPOINT_STEP.search(line):
            return int(m[1])
        if m := NATIVE_STEP.search(line):
            # Abbreviated native counts are estimates and may repeat at high step counts.
            scale = {"": 1, "K": 1000, "M": 1000000, "B": 1000000000}[m[2].upper()]
            return int(float(m[1]) * scale)
        return None

    def update(self, line, when):
        count = self.parse(line)
        if count is None or (self.total is not None and count > self.total):
            return False
        if self.anchor is None:
            if self.count is not None and count < self.count:
                return False
            if self.started_at is not None and self.count is not None and count > self.count and when > self.started_at:
                self.baseline = (self.count, self.started_at)
                self.rate = (when - self.started_at) / (count - self.count)
            else:
                self.baseline = (count, when)
            self.count = count
            self.anchor = (count, when)
            return True
        if count <= self.count:
            return False
        self.count = count
        previous, start = self.anchor
        if when > start:
            speed = (when - start) / (count - previous)
            self.rate = speed if self.rate is None else 0.65 * self.rate + 0.35 * speed
        self.anchor = (count, when)
        return True

    def average(self):
        if self.baseline is None:
            return None
        start_count, started = self.baseline
        count, finished = self.anchor
        return (finished - started) / (count - start_count) if count > start_count else None

    def fraction(self):
        if self.total is None or self.count is None:
            return 0
        return min(0.99, self.count / self.total)

    def eta(self):
        if self.rate is None or self.total is None or self.count >= self.total:
            return None
        return (self.total - self.count) * self.rate


def stage_signature(settings, stage, runtime):
    data = [settings["adapter"], settings["model"], settings[stage.name], stage.environment,
            stage.units, runtime]
    if stage.name == "eval" and "train" in settings:
        data.append(settings["train"])
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def evaluation_timing_key(settings, stage, runtime):
    if settings["adapter"] != "tinyvla" or stage.name != "eval":
        return None
    # TinyVLA uses the same model architecture for base and trained weights.
    # Match the effective evaluation workload and runtime without checkpoint or output paths.
    flags = sorted(arg for arg in stage.arguments if arg.startswith(
        ("--env.", "--eval.", "--policy.device=", "--policy.n_action_steps=")))
    profile = [settings["model"], stage.environment, stage.units, flags, runtime]
    return hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()


def run_log_directories(root, legacy_log_root):
    directories = {path.parent.name: path.parent for path in legacy_log_root.glob("*/run.py")}
    for bucket in ("evaluation", "training"):
        for script in (root / "outputs" / bucket).glob("*/executor/run.py"):
            directories[script.parents[1].name] = script.parent
    return [directories[name] for name in sorted(directories, reverse=True)]


def history(outputs, root=None):
    rates = {}
    if root is None:
        statuses = sorted(outputs.glob("*/status.json"), reverse=True)
    else:
        statuses = [directory / "status.json" for directory in run_log_directories(root, root / "executor/log")]
    for path in statuses[:60]:
        try:
            completed = [(index, item) for index, item in enumerate(read_json(path)["stages"])
                         if item["status"] == "completed" and item.get("seconds", 0) > 0]
            for _, item in completed:
                rates.setdefault(item["signature"], item["seconds"])
                if item.get("timing_key"):
                    rates.setdefault(item["timing_key"], item["seconds"])
            old_evals = [(index, item) for index, item in completed
                         if item["name"] == "eval" and "timing_key" not in item]
            if old_evals:
                nodes = ast.parse(path.with_name("run.py").read_text(encoding="utf-8")).body
                payload = next(node.value for node in nodes if isinstance(node, ast.Assign)
                               and any(isinstance(target, ast.Name) and target.id == "JOB" for target in node.targets))
                job = ast.literal_eval(payload)
                for index, item in old_evals:
                    key = evaluation_timing_key(job["settings"], Stage(**job["stages"][index]), job["runtime"])
                    if key:
                        rates.setdefault(key, item["seconds"])
        except (OSError, ValueError, KeyError, TypeError, SyntaxError, StopIteration, IndexError):
            continue
    return rates


def stop_process(process):
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            return
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


@contextmanager
def progress_display():
    if sys.stdout.isatty() and os.environ.get("TERM") != "dumb":
        try:
            from rich.console import Console
            from rich.live import Live
            from rich.text import Text
        except ImportError:
            pass
        else:
            console = Console(file=sys.stdout, markup=False, highlight=False, emoji=False)
            with Live(console=console, auto_refresh=False,
                      redirect_stdout=False, redirect_stderr=False) as live:
                def refresh(message):
                    live.update(Text(message), refresh=True)

                yield refresh, console.print
            return

    plain = partial(print, flush=True)
    yield plain, plain


def run_stage(command, stage, root, logfile, update, write_message=None):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", **stage.env}
    with logfile.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                   start_new_session=os.name == "posix")
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                clean = ANSI.sub("", line).strip()
                if write_message and clean:
                    write_message(clean)
                update(clean)
            code = process.wait()
        except BaseException:
            stop_process(process)
            raise
        finally:
            process.stdout.close()
    if code != 0:
        raise RuntimeError(f"{stage.name} exited with code {code}. See {logfile}.")
    for item in stage.artifacts:
        require(item)
    read_json(stage.artifacts[0])


def load_settings(path):
    path = Path(path).expanduser()
    if path.suffix != ".py":
        settings = read_json(path)
    else:
        names = {"SETTINGS", "settings", "QUEUE", "queue", "JOBS", "jobs", "JOB"}
        bindings = []
        for node in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body:
            if isinstance(node, ast.Assign):
                bindings.extend((target.id, node.value) for target in node.targets
                                if isinstance(target, ast.Name) and target.id in names)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in names:
                bindings.append((node.target.id, node.value))
        if len(bindings) != 1:
            raise ValueError(f"Define one SETTINGS dictionary or JOBS list in {path}.")
        name, expression = bindings[0]
        value = ast.literal_eval(expression)
        settings = value["settings"] if name == "JOB" else value
    if isinstance(settings, list):
        settings = {"schema": 1, "jobs": settings}
    if not isinstance(settings, dict):
        raise ValueError(f"Settings must be a dictionary or a list of job dictionaries ({path}).")
    # Keep imported values identical to the JSON-compatible settings collected by the prompts.
    return json.loads(json.dumps(settings, ensure_ascii=False, allow_nan=False))


def execute(stages, settings, run_dir, root, conda, verbose):
    names = {stage.environment for stage in stages}
    for stage in stages:
        for argument in stage.arguments:
            if argument.startswith(("@python:", "--policy.inference_python=@python:")):
                names.add(argument.split("@python:", 1)[1])
    runtime = {name: resolve_environment(conda, name, root) for name in sorted(names)}
    entries = []
    for stage in stages:
        entry = stage.arguments[0]
        available = runtime[stage.environment]
        if entry in {"lerobot-train", "lerobot-eval"} and not available["entrypoints"].get(entry, {}).get("launcher"):
            raise FileNotFoundError(f"Command {entry} was not found in Conda environment {stage.environment}.")
        if stage.name == "eval" and settings["adapter"] == "tinyvla" and not available.get("tinyvla_bridge"):
            raise FileNotFoundError(f"Install the TinyVLA evaluation adapter in Conda environment {stage.environment}.")
        item = asdict(stage)
        item.update(arguments=conda_command(conda, stage.environment, resolve_arguments(stage.arguments, runtime)),
                    artifacts=[str(p) for p in stage.artifacts], output=str(stage.output) if stage.output else None)
        entries.append(item)

    job = {"root": str(root), "settings": settings, "runtime": runtime, "stages": entries, "verbose": verbose}
    if stages[0].name == "eval" and stages[0].output and run_dir == stages[0].output / "executor":
        stages[0].output.parent.mkdir(parents=True, exist_ok=True)
        stages[0].output.mkdir()
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "script.py").open("x", encoding="utf-8", newline="\n") as file:
        file.write('"""Reusable settings for this job. Import with --queue or from the project scripts/ directory."""\n\n')
        file.write("JOBS = " + pformat([settings], sort_dicts=False, width=110) + "\n")
    script = run_dir / "run.py"
    source = (
        "#!/usr/bin/env python3\n"
        '"""Execute this LeRobot job with the adjacent executor installation."""\n\n'
        "from pathlib import Path\nimport sys\n\n"
        "JOB = " + pformat(job, sort_dicts=False, width=110) + "\n\n"
        'if __name__ == "__main__":\n'
        "    script = Path(__file__).resolve()\n"
        "    sys.path.insert(0, str(Path(JOB['root']) / 'executor'))\n"
        "    from runner import run_job\n"
        "    run_job(JOB, script.parent)\n"
    )
    with script.open("x", encoding="utf-8", newline="\n") as file:
        file.write(source)
    print(f"Run script: {script}", flush=True)
    runpy.run_path(str(script), run_name="__main__")


def run_job(job, run_dir):
    settings, runtime = job["settings"], job["runtime"]
    root = Path(job["root"])
    stages = []
    for item in job["stages"]:
        stage = Stage(**item)
        stage.artifacts = tuple(map(Path, stage.artifacts))
        stage.output = Path(stage.output) if stage.output else None
        stages.append(stage)
    status_path = run_dir / "status.json"
    log_directories = [run_dir]

    def save_status():
        for directory in log_directories:
            write_json(directory / "status.json", status)

    job_start = time.monotonic()
    status = {"status": "preparing", "started_at": datetime.now(timezone.utc).isoformat(), "stages": []}
    with status_path.open("x", encoding="utf-8") as file:
        json.dump(status, file)
    try:
        signatures = [stage_signature(settings, stage, runtime) for stage in stages]
        timing_keys = [evaluation_timing_key(settings, stage, runtime) for stage in stages]
        previous = history(run_dir.parent, root)
        status["status"] = "running"
        status["stages"] = [{"name": s.name, "status": "waiting", "signature": signature,
                             "timing_key": key, "output": str(s.output) if s.output else None}
                            for s, signature, key in zip(stages, signatures, timing_keys)]
        save_status()
        eta_scope = None
        for index, stage in enumerate(stages):
            saved = status["stages"][index]
            begin = time.monotonic()
            progress = Progress(stage, settings, begin)
            known_future = [previous.get(signature, previous.get(key))
                            for signature, key in zip(signatures[index + 1:], timing_keys[index + 1:])]
            saved["status"] = "running"
            saved["started_at"] = datetime.now(timezone.utc).isoformat()
            save_status()
            label = {"train": "Training", "eval": "Evaluation"}[stage.name]
            future_eta = sum(known_future) if None not in known_future else None
            historical_time = previous.get(signatures[index], previous.get(timing_keys[index]))
            last_print = begin - 5

            def report(now, completed=False):
                nonlocal eta_scope
                eta = progress.eta()
                if completed:
                    eta = 0
                elif eta is None and historical_time is not None and historical_time > now - begin:
                    eta = historical_time - (now - begin)
                overall = eta + future_eta if eta is not None and future_eta is not None else None
                display_eta = overall if overall is not None else (None if completed else eta)
                if display_eta is not None:
                    scope = "whole job" if overall is not None else label.lower()
                    if scope != eta_scope and (scope != "whole job" or eta_scope is not None):
                        write_message(f"Time estimate scope · {scope}")
                    eta_scope = scope
                fraction = 1 if completed else progress.fraction()
                count = stage.units if completed and stage.units is not None else progress.count
                show_status(status_line(100 * (index + fraction) / len(stages), count, stage.units,
                                        stage.name, display_eta, progress.average()))
                saved.update(units_done=count, units_total=stage.units, eta_seconds=eta,
                             overall_eta_seconds=overall)
                if stage.name == "eval":
                    saved["average_seconds_per_episode"] = progress.average()

            def update(line):
                nonlocal last_print
                now = time.monotonic()
                previous_count = progress.count
                if not progress.update(line, now) or progress.count == previous_count:
                    return
                if now - last_print < 2 and progress.count != stage.units:
                    if progress.rate is None or eta_scope is not None:
                        return
                last_print = now
                report(now)
                save_status()

            print(f"\n{label} · Environment {stage.environment}", flush=True)
            stage_log = run_dir
            if stage.output:
                if run_dir.name == "executor" and stage.name == "eval":
                    stage_log = stage.output / "executor"
                    if stage_log == run_dir:
                        if stage.output.is_symlink() or set(stage.output.iterdir()) != {run_dir}:
                            raise FileExistsError(f"Output path already contains results ({stage.output}).")
                    else:
                        stage.output.mkdir(parents=True)
                        stage_log.mkdir()
                        for filename in ("run.py", "script.py"):
                            shutil.copyfile(run_dir / filename, stage_log / filename)
                        log_directories.append(stage_log)
                else:
                    if stage.output.exists() or stage.output.is_symlink():
                        raise FileExistsError(f"Output path already exists ({stage.output}).")
                    stage.output.parent.mkdir(parents=True, exist_ok=True)
                print(f"Output directory: {stage.output}", flush=True)
            with progress_display() as (show_status, write_message):
                report(begin)
                save_status()
                try:
                    run_stage(stage.arguments, stage, root, stage_log / f"{stage.name}.log",
                              update, write_message if job["verbose"] else None)
                finally:
                    saved.update(seconds=time.monotonic()-begin, ended_at=datetime.now(timezone.utc).isoformat())
                saved["status"] = "completed"
                report(time.monotonic(), completed=True)
                save_status()
        status["status"] = "completed"
    except BaseException as error:
        status["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        status["error"] = str(error)
        for stage_status in status["stages"]:
            if stage_status["status"] == "running":
                stage_status["status"] = status["status"]
                stage_status["ended_at"] = datetime.now(timezone.utc).isoformat()
        raise
    finally:
        status.update(ended_at=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-job_start)
        save_status()
