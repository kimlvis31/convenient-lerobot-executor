"""Launch local Conda commands and record their measured progress."""

import ast
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from pprint import pformat
import re
import runpy
import shutil
import signal
import subprocess
import time

from adapters import Stage, read_json, require


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NATIVE_STEP = re.compile(r"\bstep:\s*(\d+(?:\.\d+)?)([KMB]?)\b(?![/\d.])", re.I)
TINY_STEP = re.compile(r"\bstep:(\d+)/(\d+)\b")
CHECKPOINT_STEP = re.compile(r"\bCheckpoint policy after step (\d+)\b")
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


def seconds(value):
    if value is None:
        return "unavailable"
    mins, secs = divmod(max(0, round(value)), 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:d}:{mins:02d}:{secs:02d}"


class Progress:
    def __init__(self, stage, settings):
        self.stage = stage
        self.count = stage.start
        self.total = stage.units
        self.anchor = None
        self.rate = None
        self.group = 0
        self.last_batch = None
        self.episodes = settings.get("eval", {}).get("episodes", 0)
        self.batch = settings.get("eval", {}).get("batch", 1)

    def parse(self, line):
        if self.stage.name == "eval":
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


def history(outputs):
    rates = {}
    for path in sorted(outputs.glob("*/status.json"), reverse=True)[:60]:
        try:
            status = read_json(path)
            for item in status["stages"]:
                if item["status"] == "completed" and item.get("seconds", 0) > 0:
                    rates.setdefault(item["signature"], item["seconds"])
        except (OSError, ValueError, KeyError, TypeError):
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


def run_stage(command, stage, root, logfile, update, verbose):
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
                if verbose and clean:
                    print(clean, flush=True)
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
        "    sys.path.insert(0, str(script.parents[2]))\n"
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
    job_start = time.monotonic()
    status = {"status": "preparing", "started_at": datetime.now(timezone.utc).isoformat(), "stages": []}
    with status_path.open("x", encoding="utf-8") as file:
        json.dump(status, file)
    try:
        signatures = [stage_signature(settings, stage, runtime) for stage in stages]
        previous = history(run_dir.parent)
        status["status"] = "running"
        status["stages"] = [{"name": s.name, "status": "waiting", "signature": signature,
                             "output": str(s.output) if s.output else None}
                            for s, signature in zip(stages, signatures)]
        write_json(status_path, status)
        for index, stage in enumerate(stages):
            saved = status["stages"][index]
            begin = time.monotonic()
            progress = Progress(stage, settings)
            known_future = [previous.get(key) for key in signatures[index + 1:]]
            saved["status"] = "running"
            saved["started_at"] = datetime.now(timezone.utc).isoformat()
            write_json(status_path, status)
            label = {"train": "Training", "eval": "Evaluation"}[stage.name]
            print(f"\nOverall {100*index/len(stages):.0f}% · Starting {label.lower()} · Environment {stage.environment}", flush=True)
            last_print = begin - 5

            def update(line):
                nonlocal last_print
                now = time.monotonic()
                if not progress.update(line, now):
                    return
                if now - last_print < 2 and progress.count != stage.units:
                    return
                last_print = now
                eta = progress.eta()
                overall = eta + sum(known_future) if eta is not None and None not in known_future else None
                fraction = progress.fraction()
                unit = "episodes" if stage.name == "eval" else "steps"
                work = f"{progress.count}/{stage.units}" if stage.units is not None else str(progress.count)
                percent = f" ({100*fraction:.0f}%)" if stage.units is not None else ""
                finalizing = " · Saving results" if progress.count == stage.units else ""
                print(f"Overall {int(100*(index+fraction)/len(stages))}% · {label} {work} {unit}{percent}"
                      f" · Stage time remaining {seconds(eta)} · Total time remaining {seconds(overall)}{finalizing}", flush=True)
                saved.update(units_done=progress.count, units_total=stage.units, eta_seconds=eta,
                             overall_eta_seconds=overall)
                write_json(status_path, status)

            if stage.output:
                if stage.output.exists() or stage.output.is_symlink():
                    raise FileExistsError(f"Output path already exists ({stage.output}).")
                stage.output.parent.mkdir(parents=True, exist_ok=True)
                print(f"Output directory: {stage.output}", flush=True)
            try:
                run_stage(stage.arguments, stage, root, run_dir / f"{stage.name}.log", update, job["verbose"])
            finally:
                saved.update(seconds=time.monotonic()-begin, ended_at=datetime.now(timezone.utc).isoformat())
            saved.update(status="completed", seconds=time.monotonic()-begin, units_done=stage.units,
                         units_total=stage.units, eta_seconds=0, overall_eta_seconds=None,
                         ended_at=datetime.now(timezone.utc).isoformat())
            write_json(status_path, status)
            print(f"Overall {int(100*(index+1)/len(stages))}% · {label} complete", flush=True)
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
        write_json(status_path, status)
