"""Launch local Conda commands and record their measured progress."""

import ast
from contextlib import contextmanager
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
import sys
import subprocess
from tempfile import TemporaryDirectory
import time

from adapters import Stage, read_json, require
from checkpoint_compat import RESULT_PREFIX, checkpoint_candidate, validate_processor_states
from pinned_policy import REVISION_FLAG, pinned_command, requested_revision


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NATIVE_STEP = re.compile(r"\bstep:\s*(\d+(?:\.\d+)?)([KMB]?)\b(?![/\d.])", re.I)
TINY_STEP = re.compile(r"\bstep:(\d+)/(\d+)\b")
CHECKPOINT_STEP = re.compile(r"\bCheckpoint policy after step (\d+)\b")
PARALLEL_EPISODES = re.compile(r"^\[Eval progress\] Episodes completed: (\d+)$")
ROLLOUT_ENDED = re.compile(r"\[Eval progress\] Rollout ended: (\S+)$")
EVAL_BATCH = re.compile(r"Stepping through eval batches[^|]*\|[^|]*\|\s*(\d+)\s*/\s*(\d+)")

ROLLOUT_STEPS = re.compile(
    r"Running rollout(?: \[([^\]]+)\])? with at most \d+ steps:\s*\d+%\|[^|]*\|\s*(\d+)\s*/\s*(\d+)"
)

STAGE_LABELS = {"train": "Training", "eval": "Evaluation"}
TOTAL_REFRESH_SECONDS = 0.5
EPISODE_REFRESH_SECONDS = 0.1
STATUS_SAVE_SECONDS = 2.0


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
        result = subprocess.run([conda, "info", "--json"], cwd=root, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=15, check=True)
        info = json.loads(result.stdout)
        base = Path(info["root_prefix"]) if info.get("root_prefix") else None
        directories = info.get("envs_dirs")
        if directories is None and base is not None:
            directories = [base / "envs"]
        directories = {Path(path) for path in directories or ()}
        return {"base" if path == base else path.name for path in map(Path, info["envs"])
                if path == base or path.parent in directories}
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return set()


def environment_catalog(conda, name, root):
    if not conda:
        return {}
    probe = Path(__file__).with_name("benchmarks.py")
    try:
        result = subprocess.run(conda_command(conda, name, ["python", "-B", str(probe), "--catalog"]),
                                cwd=root, text=True, encoding="utf-8", errors="replace",
                                capture_output=True, timeout=45, check=True)
        for line in reversed(result.stdout.splitlines()):
            if line.startswith("MANAGER_CATALOG "):
                return json.loads(line.removeprefix("MANAGER_CATALOG "))
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return {}


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


def stage_title(stage, model):
    return f"{STAGE_LABELS[stage.name]} · Model {model} · Environment {stage.environment}"


def status_line(percent, count, total, stage, eta, average=None):
    current = count if count is not None else "N/A"
    target = total if total is not None else "N/A"
    parts = [f"[Total] Completion: {percent:.3f} %"]
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


def print_status_detail(message):
    print(f"    • {message}", flush=True)


@contextmanager
def progress_display(evaluation):
    if sys.stdout.isatty() and os.environ.get("TERM") != "dumb":
        try:
            from rich.console import Console
            from rich.live import Live
            from rich.padding import Padding
            from rich.table import Table
            from rich.text import Text
        except ImportError:
            pass
        else:
            console = Console(file=sys.stdout, markup=False, highlight=False, emoji=False)
            with Live(console=console, auto_refresh=False, transient=True,
                      redirect_stdout=False, redirect_stderr=False) as live:
                def gauge(width, completed, total, color):
                    filled = max(0, min(width, int(width * completed / total)))
                    return Text.assemble(("▄" * filled, color), ("▂" * (width - filled), "bar.back"))

                def refresh(message, completion, episode):
                    width = max(8, min(22, console.width // 6))
                    gauges = Table.grid(padding=(0, 1))
                    gauges.add_column(width=1)
                    gauges.add_column(width=width)
                    gauges.add_column()
                    gauges.add_row(
                        Text("•"),
                        gauge(width, completion, 100, "green"),
                        Text(message),
                    )
                    if evaluation:
                        if episode is None:
                            text = "[Episode] No active batch"
                            current, limit = 0, 1
                        else:
                            task, current, limit = episode
                            text = f"[Episode] Completion: {100 * current / limit:.3f} % | {current}/{limit} Steps"
                            if task:
                                text += f" | Task: {task}"
                        gauges.add_row(
                            Text("•"),
                            gauge(width, current, limit, "bright_blue"),
                            Text(text),
                        )
                    live.update(Padding(gauges, (0, 0, 0, 4), expand=False), refresh=True)

                def note(message):
                    console.print(f"    • {message}")

                yield refresh, note, evaluation
            return

        current = None
        erase = "\r\x1b[2K"

        def clear_status():
            nonlocal current
            if current is not None:
                print(erase, end="", flush=True)
                current = None

        def refresh_text(message, _completion, _episode):
            nonlocal current
            width = max(1, shutil.get_terminal_size().columns - 1)
            line = f"    • {message}"
            if len(line) > width:
                line = line[:width - 1] + "…"
            prefix = erase if current is not None else ""
            current = message
            print(prefix + line, end="", flush=True)

        def note_text(message):
            previous = current
            clear_status()
            print_status_detail(message)
            if previous is not None:
                refresh_text(previous, None, None)

        try:
            yield refresh_text, note_text, False
        finally:
            clear_status()
        return

    def status_text(message, _completion, _episode):
        print_status_detail(message)

    yield status_text, print_status_detail, False


class StageFailure(RuntimeError):
    def __init__(self, stage, code, logfile, *, artifact_error=None):
        self.stage = stage
        self.code = code
        self.logfile = logfile
        if artifact_error is None:
            self.description = f"{stage} exited with code {code}"
            self.reason = stage_failure_reason(logfile)
        else:
            self.description = f"{stage} did not produce valid results"
            self.reason = str(artifact_error)
        message = f"{self.description}."
        if self.reason:
            message += f" Cause: {self.reason}"
        super().__init__(f"{message} See {logfile}.")


def stage_failure_reason(logfile):
    try:
        with logfile.open("rb") as log:
            size = log.seek(0, os.SEEK_END)
            start = max(0, size - 64 * 1024)
            log.seek(start)
            data = log.read(64 * 1024)
    except OSError:
        return None
    if start:
        data = data.partition(b"\n")[2]
    lines = [
        re.sub(r"^\[rank\d+\]:\s*", "", line.strip(" │"))
        for line in ANSI.sub("", data.decode("utf-8", errors="replace")).splitlines()
    ]

    def compact(value, limit):
        clean = "".join(char for char in value if char.isprintable() or char.isspace())
        clean = " ".join(clean.split())
        clean = re.sub(r"(?i)(https?://)[^/\s'\"<>]+@", r"\1", clean)
        clean = re.sub(r"(?i)(https?://[^\s?'\"<>#]+)[?#][^\s'\"<>]*", r"\1", clean)
        clean = re.sub(r"(?i)\bBearer\s+[^\s'\",;)\]]+", "Bearer [redacted]", clean)
        clean = re.sub(r"\bhf_[A-Za-z0-9]{8,}\b", "[redacted]", clean)
        return clean if len(clean) <= limit else clean[:limit - 1] + "…"

    exception_line = re.compile(
        r"((?:[A-Za-z_]\w*\.)*(?:[A-Z]\w*(?:Error|Exception)|Error|Exception|OfflineModeIsEnabled)):\s*(.*)"
    )
    ignored_errors = set()
    for index, line in enumerate(lines[:-1]):
        if not line.startswith("Exception ignored in:"):
            continue
        start = index + 1
        if lines[start].startswith("Traceback (most recent call last)"):
            start += 1
        for position in range(start, len(lines)):
            if exception_line.fullmatch(lines[position]):
                ignored_errors.add(position)
                break
            if not lines[position] or lines[position].startswith(("Traceback ", "Exception ignored in:")):
                break
    chain_markers = {
        "The above exception was the direct cause of the following exception:",
        "During handling of the above exception, another exception occurred:",
    }

    def chained_cause(index):
        linked = False
        for position in range(index - 1, -1, -1):
            previous = lines[position]
            if previous in chain_markers:
                linked = True
            elif exception_line.fullmatch(previous):
                return position if linked else None
        return None

    def hub_page(context):
        # Keep the repository only. Resolve URLs can carry signed query strings.
        slug = r"[A-Za-z0-9_-][A-Za-z0-9_.-]*"
        for path in re.findall(r"https?://(?:huggingface\.co|hf\.co)/([^\s?'\"<>#]+)", context, re.IGNORECASE):
            path = path.rstrip(".,;:)/]")
            path = re.split(r"/(?:resolve|raw|blob|tree|commit|commits|revision|discussions|auth-check)(?:/|$)", path, 1)[0]
            parts = path.split("/")
            if parts[0] == "api":
                if len(parts) < 3 or parts[1] not in {"datasets", "spaces", "models"}:
                    continue
                parts = parts[1:]
            prefix = ""
            if parts[0] in {"datasets", "spaces", "models"}:
                kind, *parts = parts
                prefix = "" if kind == "models" else f"{kind}/"
            if (1 <= len(parts) <= 2 and parts[0] not in {"api", "docs", "settings", "login", "join"}
                    and all(re.fullmatch(slug, part) for part in parts)):
                return "https://huggingface.co/" + prefix + "/".join(parts)
        named = re.search(rf"\baccess to (model|dataset|space) ['\"]?({slug}/{slug})['\"]? is restricted", context, re.I)
        if named:
            kind, name = named.groups()
            prefix = "" if kind.lower() == "model" else f"{kind.lower()}s/"
            return f"https://huggingface.co/{prefix}{name}"
        return None

    def hub_access_reason(index, error_type, detail):
        if error_type not in {"GatedRepoError", "HfHubHTTPError", "RepositoryNotFoundError",
                              "HTTPError", "HTTPStatusError", "OSError"}:
            return None
        context = [detail]
        for following in lines[index + 1:index + 8]:
            if following in chain_markers or following.startswith("Traceback ") or exception_line.fullmatch(following):
                break
            context.append(following)
        context = " ".join(context)
        page = hub_page(context)
        gated = (error_type == "GatedRepoError" or "you are trying to access a gated repo" in detail.lower()
                 or re.search(r"\bcannot access (?:a |the )?gated repo", context, re.I)
                 or re.search(r"\baccess to (?:model|dataset|space) \S+ is restricted", context, re.I))
        status = re.search(r"\b(401|403)\s+(?:Client Error|Unauthorized|Forbidden)\b", context, re.I)
        if gated:
            if not page and (previous := chained_cause(index)) is not None:
                page = hub_page(" ".join(lines[previous:min(previous + 8, index)]))
            where = page or "its Hugging Face repository page"
            return compact(f"Gated repository. Open {where} with the Hugging Face account used on this server. "
                           "Review the terms and request access. If already approved, check the token's read permissions.", 320)
        if status and (page or error_type in {"HfHubHTTPError", "RepositoryNotFoundError"}):
            meaning = "could not authenticate the request" if status[1] == "401" else "denied the request"
            where = f" Open {page} to verify the repository." if page else " Verify the repository ID."
            return compact(f"Hugging Face {meaning} ({status[1]}).{where} "
                           "Check the signed-in server account and the token's read permissions.", 320)
        return None

    diagnostic = None
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index]
        match = exception_line.fullmatch(line)
        if match:
            error, detail = match.groups()
            error_type = error.rsplit(".", 1)[-1]
            if index in ignored_errors:
                if diagnostic is None:
                    diagnostic = compact(line, 320)
                continue
            if error_type in {"CalledProcessError", "ChildFailedError"}:
                continue
            if error_type in {"BrokenPipeError", "ConnectionResetError"}:
                diagnostic = compact(line, 320)
                continue
            loading_wrapper = error_type in {"RuntimeError", "OSError"} and re.match(
                r"(?:Failed|Unable) to (?:load|initialize) (?:the |required )?"
                r"(?:tokenizers?|processors?|models?|policy|checkpoints?)\b", detail, re.IGNORECASE,
            )
            hub_wrapper = (error_type == "OSError" and detail.startswith("We couldn't connect to ")
                           and "cached files" in detail)
            config_wrapper = (error in {"draccus.utils.ParsingError", "ParsingError"}
                              and detail.startswith("Couldn't instantiate class "))
            if (loading_wrapper or hub_wrapper or config_wrapper) and chained_cause(index) is not None:
                diagnostic = compact(line, 320)
                continue
            if error_type == "LocalEntryNotFoundError" and "outgoing traffic has been disabled" in detail:
                return "Required files are missing from the Hugging Face cache and network access is disabled."
            if access := hub_access_reason(index, error_type, detail):
                return access
            if "CUDA out of memory" in detail:
                requested = re.search(r"Tried to allocate (\d+(?:\.\d+)? [KMG]iB)", detail)
                free = re.search(r"of which (\d+(?:\.\d+)? [KMG]iB) is free", detail)
                if requested and free:
                    return f"CUDA out of memory. Requested {requested[1]} with {free[1]} free."
                return "CUDA out of memory."
            related = []
            for followup in lines[index + 1:index + 7]:
                feature = re.match(r"(?:[-*•]\s*)?((?:missing|extra|unexpected)(?: [^:]+)?:\s*.+)",
                                   followup.strip(), re.IGNORECASE)
                if feature:
                    related.append(compact(feature[1], 135))
            if detail or related:
                if related:
                    return " | ".join([compact(line, 145), *related[:2]])
                return compact(line, 320)
        elif diagnostic is None and re.match(r"^(?:ERROR:|Fatal:|CUDA error:|\S+: error:)\s*\S",
                                             line, re.IGNORECASE):
            diagnostic = compact(line, 320)
    return diagnostic


@contextmanager
def prepare_stage_checkpoint(command, stage, root, env, logfile, log, native):
    reference = next((item.partition("=")[2] for item in command if item.startswith("--policy.path=")), None)
    if not native or not reference:
        yield command
        return
    conda, model = native
    try:
        revision = requested_revision(command)
        original = pinned_command(command, stage.name, reference, revision, root)
        if revision is not None and original is command:
            log.write(f"Using the selected local checkpoint {reference}. Hub revision {revision} applies only to a repository ID.\n")
        validate_processor_states(reference)
    except (OSError, ValueError) as error:
        log.write(f"CheckpointError: {error}\n")
        log.flush()
        raise StageFailure(stage.name, 1, logfile) from error
    if not checkpoint_candidate(reference, model, stage.name):
        yield original
        return

    with TemporaryDirectory(prefix="convlexecutor-checkpoint-") as directory:
        path = Path(directory) / "pretrained_model"
        helper = Path(__file__).with_name("checkpoint_compat.py").resolve()
        options = ["python", "-B", str(helper), "--policy", reference, "--model", model, "--output", str(path)]
        if model in {"molmoact2", "wall_x"}:
            options.extend(("--stage", stage.name))
        if revision:
            options.extend(("--revision", revision))
        log.write(f"Checking checkpoint processor format: {reference}\n")
        log.flush()
        process = subprocess.Popen(conda_command(conda, stage.environment, options), cwd=root, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", start_new_session=os.name == "posix")
        try:
            output, _ = process.communicate()
        except BaseException:
            stop_process(process)
            raise
        report = None
        for line in output.splitlines():
            if line.startswith(RESULT_PREFIX):
                report = line[len(RESULT_PREFIX):]
            else:
                log.write(line + "\n")
        if process.returncode:
            log.flush()
            raise StageFailure(stage.name, process.returncode, logfile)
        if report == json.dumps(str(path)) and path.is_dir():
            log.write("Using a temporary conversion of the legacy checkpoint.\n")
            log.flush()
            command = [f"--policy.path={path}" if item.startswith("--policy.path=") else item
                       for item in command if not item.startswith(REVISION_FLAG)]
        elif report != "null":
            log.write("CheckpointError: No valid result was returned by checkpoint preparation.\n")
            log.flush()
            raise StageFailure(stage.name, 1, logfile)
        else:
            command = original
        yield command


def run_stage(command, stage, root, logfile, update, write_message=None, *, native=None):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", **stage.env}
    with logfile.open("x", encoding="utf-8") as log, \
         prepare_stage_checkpoint(command, stage, root, env, logfile, log, native) as command:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                   start_new_session=os.name == "posix")
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                clean = ANSI.sub("", line).strip()
                if write_message and clean and not (
                    ROLLOUT_STEPS.search(clean) or ROLLOUT_ENDED.search(clean)
                    or EVAL_BATCH.search(clean) or PARALLEL_EPISODES.fullmatch(clean)
                ):
                    write_message(clean)
                update(clean)
            code = process.wait()
        except BaseException:
            stop_process(process)
            raise
        finally:
            process.stdout.close()
    if code != 0:
        raise StageFailure(stage.name, code, logfile)
    try:
        for item in stage.artifacts:
            require(item)
        if stage.artifacts:
            result = read_json(stage.artifacts[0])
            if not isinstance(result, dict):
                raise ValueError(f"Expected a result object ({stage.artifacts[0]}).")
            return result
    except (OSError, ValueError) as error:
        raise StageFailure(stage.name, code, logfile, artifact_error=error) from error


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

    job = {"root": str(root), "settings": settings, "runtime": runtime, "stages": entries,
           "verbose": verbose, "conda": str(conda)}
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
            label = STAGE_LABELS[stage.name]
            future_eta = sum(known_future) if None not in known_future else None
            historical_time = previous.get(signatures[index], previous.get(timing_keys[index]))
            last_print = begin - TOTAL_REFRESH_SECONDS
            last_episode_print = begin - EPISODE_REFRESH_SECONDS
            last_progress_save = begin - STATUS_SAVE_SECONDS
            active_episodes = {}

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
                count = progress.total if completed and progress.total is not None else progress.count
                completion = 100 * (index + fraction) / len(stages)
                line = status_line(completion, count, progress.total, stage.name, display_eta, progress.average())
                episode = None
                if active_episodes and not completed:
                    task, (current, limit) = min(
                        active_episodes.items(), key=lambda item: item[1][0] / item[1][1]
                    )
                    episode = (task, current, limit)
                show_status(line, completion, episode)
                saved.update(units_done=count, units_total=progress.total, eta_seconds=eta,
                             overall_eta_seconds=overall)
                if stage.name == "eval":
                    saved["average_seconds_per_episode"] = progress.average()

            def update(line):
                nonlocal last_print, last_episode_print, last_progress_save
                now = time.monotonic()
                if show_episode and (rollout := ROLLOUT_STEPS.search(line)):
                    task, current, limit = rollout.groups()
                    current, limit = int(current), int(limit)
                    sample = (current, limit)
                    if limit > 0 and current <= limit and active_episodes.get(task) != sample:
                        active_episodes[task] = sample
                        if now - last_episode_print >= EPISODE_REFRESH_SECONDS or current == limit:
                            last_episode_print = now
                            report(now)
                if show_episode and (ended := ROLLOUT_ENDED.search(line)):
                    if active_episodes.pop(ended[1], None) is not None:
                        report(now)
                previous_count = progress.count
                if not progress.update(line, now) or progress.count == previous_count:
                    return
                active_episodes.pop(None, None)
                if now - last_print < TOTAL_REFRESH_SECONDS and progress.count != stage.units:
                    if progress.rate is None or eta_scope is not None:
                        return
                last_print = now
                report(now)
                if now - last_progress_save >= STATUS_SAVE_SECONDS or progress.count == stage.units:
                    save_status()
                    last_progress_save = now

            if index:
                print()
            print_status_detail(stage_title(stage, settings["model"]))
            if worker := stage.env.get("CONVLEX_ROBOMME_WORKER"):
                print_status_detail(f"RoboMME simulator Python: {json.loads(worker)['python']}")
            print_status_detail(f"Run script: {run_dir / 'run.py'}")
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
                print_status_detail(f"Output directory: {stage.output}")
            with progress_display(stage.name == "eval") as (show_status, write_message, show_episode):
                report(begin)
                save_status()
                try:
                    native = (job["conda"], settings["model"]) if job.get("conda") and settings["adapter"] == "native" else None
                    result = run_stage(stage.arguments, stage, root, stage_log / f"{stage.name}.log",
                                       update, write_message if job["verbose"] else None, native=native)
                finally:
                    saved.update(seconds=time.monotonic()-begin, ended_at=datetime.now(timezone.utc).isoformat())
                saved["status"] = "completed"
                if stage.name == "eval" and result:
                    overall = result.get("overall", result.get("aggregated"))
                    episodes = overall.get("n_episodes") if isinstance(overall, dict) else None
                    if type(episodes) is int and episodes >= 0:
                        progress.total = episodes
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
