"""AI-written ownership and bounded interruption of a stage's process tree.

Linux stages run under a dedicated subreaper, so double-forked and setsid
workers cannot outlive a launcher that exits early. Windows stages enter a
kill-on-close Job Object before starting the command. Neither mechanism
searches for process names or terminates another job's workers.
"""

from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import current_thread, main_thread
import time
import uuid


GRACE_SECONDS = 8.0
TERM_SECONDS = 2.0
KILL_SECONDS = 3.0
POLL_SECONDS = 0.05


@contextmanager
def deferred_interrupts():
    """Keep repeated interrupts from abandoning process creation or cleanup."""
    pending, previous = [], {}
    if current_thread() is main_thread():
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            if (number := getattr(signal, name, None)) is not None:
                previous[number] = signal.getsignal(number)
                signal.signal(number, lambda number, _frame: pending.append(number))
    try:
        yield pending
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


class WindowsJob:
    """A private job; only the manager keeps a persistent handle to it."""

    def __init__(self, name=None):
        import ctypes
        from ctypes import wintypes as w

        self.ctypes, self.w = ctypes, w
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        for function, result, arguments in (
            ("CreateJobObjectW", w.HANDLE, [w.LPVOID, w.LPCWSTR]),
            ("OpenJobObjectW", w.HANDLE, [w.DWORD, w.BOOL, w.LPCWSTR]),
            ("SetInformationJobObject", w.BOOL, [w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD]),
            ("QueryInformationJobObject", w.BOOL, [w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD, w.LPVOID]),
            ("AssignProcessToJobObject", w.BOOL, [w.HANDLE, w.HANDLE]),
            ("TerminateJobObject", w.BOOL, [w.HANDLE, w.UINT]),
            ("GetCurrentProcess", w.HANDLE, []),
            ("CloseHandle", w.BOOL, [w.HANDLE]),
        ):
            method = getattr(self.api, function)
            method.restype, method.argtypes = result, arguments

        class BasicLimits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                        ("flags", w.DWORD), ("minimum", ctypes.c_size_t), ("maximum", ctypes.c_size_t),
                        ("active", w.DWORD), ("affinity", ctypes.c_size_t),
                        ("priority", w.DWORD), ("scheduling", w.DWORD)]

        class Counters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in
                        ("reads", "writes", "others", "read_bytes", "write_bytes", "other_bytes")]

        class Limits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io", Counters),
                        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                        ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

        class Accounting(ctypes.Structure):
            _fields_ = [(name, ctypes.c_longlong) for name in
                        ("user_time", "kernel_time", "period_user", "period_kernel")]
            _fields_ += [(name, w.DWORD) for name in ("faults", "total", "active", "terminated")]

        self.accounting = Accounting
        self.name = name or "Local\\convlex-stage-" + uuid.uuid4().hex
        self.handle = (self.api.OpenJobObjectW(0x000D, False, name) if name else
                       self.api.CreateJobObjectW(None, self.name))
        self.check(self.handle)
        if name is None:
            limits = Limits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; no breakaway.
            try:
                self.check(self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits),
                                                           ctypes.sizeof(limits)))
            except BaseException:
                self.close()
                raise

    def check(self, result):
        if not result:
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def join(self):
        self.check(self.api.AssignProcessToJobObject(self.handle, self.api.GetCurrentProcess()))

    def terminate(self, code=130):
        self.check(self.api.TerminateJobObject(self.handle, code))

    def wait_empty(self):
        deadline = time.monotonic() + KILL_SECONDS
        while True:
            info = self.accounting()
            self.check(self.api.QueryInformationJobObject(self.handle, 1, self.ctypes.byref(info),
                                                         self.ctypes.sizeof(info), None))
            if not info.active:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("Windows stage workers did not finish terminating.")
            time.sleep(POLL_SECONDS)

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


@contextmanager
def owned_process(command, **options):
    """Launch a command whose descendants are cleaned on every exit path."""
    process, job = None, None
    try:
        with deferred_interrupts() as pending:
            wrapper = [sys.executable, "-B", str(Path(__file__).resolve()), str(os.getpid())]
            if os.name == "nt":
                job = WindowsJob()
                wrapper.append(job.name)
                options["creationflags"] = options.get("creationflags", 0) | subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                options["start_new_session"] = True
            process = subprocess.Popen([*wrapper, "--", *command], **options)
            process._convlex_job = job
        if pending:
            raise KeyboardInterrupt("Interrupted while launching the stage.")
        yield process
    finally:
        with deferred_interrupts():
            try:
                if process is not None:
                    if process.poll() is None:
                        stop_process(process)
                    if job is not None:
                        job.terminate()
                        job.wait_empty()
            finally:
                if job is not None:
                    job.close()
                if process is not None:
                    for stream in (process.stdin, process.stdout, process.stderr):
                        if stream is not None and not stream.closed:
                            stream.close()


def stop_process(process):
    """Do not equate a launcher's exit with completion of its process tree."""
    with deferred_interrupts():
        job = getattr(process, "_convlex_job", None)
        if process.poll() is None:
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
            except (ProcessLookupError, OSError):
                if job is not None:
                    job.terminate()
            try:
                process.wait(timeout=GRACE_SECONDS + TERM_SECONDS + KILL_SECONDS + 2)
            except subprocess.TimeoutExpired:
                if job is not None:
                    job.terminate()
                else:
                    # The supervisor normally reaps all descendants itself.
                    # This is a last resort for an unresponsive supervisor.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait(timeout=KILL_SECONDS)
        if job is not None:
            job.terminate()
            job.wait_empty()


def linux_record(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return int(fields[1]), int(fields[19]), fields[0]  # parent, start ticks, state
    except (OSError, ValueError, IndexError):
        return None


def linux_descendants():
    records = {int(path.name): record for path in Path("/proc").iterdir()
               if path.name.isdigit() and (record := linux_record(path.name)) is not None}
    owned = {os.getpid()}
    while new := {pid for pid, (parent, _, _) in records.items() if parent in owned} - owned:
        owned.update(new)
    return {pid: records[pid] for pid in owned - {os.getpid()}}


def signal_linux_worker(pid, record, number):
    """Pin the process identity before signalling, rather than reusing a PID."""
    descriptor = None
    try:
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            descriptor = os.pidfd_open(pid)
        current = linux_record(pid)
        if current is None or current[1] != record[1] or current[2] == "Z":
            return
        if descriptor is not None:
            signal.pidfd_send_signal(descriptor, number)
        else:
            os.kill(pid, number)
    except ProcessLookupError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def supervise_linux(command, owner):
    import ctypes

    stopping = []
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, lambda number, _frame: stopping.append(number) if not stopping else None)
    libc = ctypes.CDLL(None, use_errno=True)
    # Adoption is local to this supervisor, never the long-lived manager.
    # https://man7.org/linux/man-pages/man2/PR_SET_CHILD_SUBREAPER.2const.html
    for option, value in ((36, 1), (1, signal.SIGTERM)):  # CHILD_SUBREAPER, PDEATHSIG
        if libc.prctl(option, value, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Cannot establish stage process ownership.")
    if os.getppid() != owner or stopping:
        return 130
    child = subprocess.Popen(command)
    result, started, sent = None, None, set()
    try:
        while True:
            # Reap the command and adopted descendants, including zombies.
            while True:
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if not pid:
                    break
                if pid == child.pid:
                    result = child.returncode = os.waitstatus_to_exitcode(status)
            if stopping or result is not None:
                if started is None:
                    started = time.monotonic()
                workers = linux_descendants()
                if not workers:
                    return 128 + stopping[0] if stopping else result
                elapsed = time.monotonic() - started
                number = (signal.SIGKILL if elapsed >= GRACE_SECONDS + TERM_SECONDS else
                          signal.SIGTERM if elapsed >= GRACE_SECONDS else
                          stopping[0] if stopping else signal.SIGTERM)
                for pid, record in workers.items():
                    key = pid, record[1], number
                    if key not in sent:
                        signal_linux_worker(pid, record, number)
                        sent.add(key)
                if elapsed >= GRACE_SECONDS + TERM_SECONDS + KILL_SECONDS:
                    raise RuntimeError(f"Stage workers did not finish terminating: {sorted(workers)}")
            time.sleep(POLL_SECONDS)
    finally:
        # Unexpected Python errors must not skip cleanup either.
        deadline = time.monotonic() + KILL_SECONDS
        while workers := linux_descendants():
            for pid, record in workers.items():
                signal_linux_worker(pid, record, signal.SIGKILL)
            while True:
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if not pid:
                    break
                if pid == child.pid:
                    child.returncode = os.waitstatus_to_exitcode(status)
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Stage workers did not finish terminating: {sorted(workers)}")
            time.sleep(POLL_SECONDS)


def supervise_windows(command, name):
    job = WindowsJob(name)
    try:
        job.join()
    finally:
        # Only the manager retains a handle: its death must close the job.
        job.close()
    signal.signal(signal.SIGBREAK, signal.default_int_handler)
    result = 1
    try:
        result = subprocess.call(command)
    except KeyboardInterrupt:
        result = 130
    finally:
        # Also closes inherited output pipes if a launcher exited early.
        job = WindowsJob(name)
        job.terminate(result & 0xFFFFFFFF)
        job.close()
    return result


def main():
    boundary = sys.argv.index("--")
    owner, command = int(sys.argv[1]), sys.argv[boundary + 1:]
    if sys.platform == "linux":
        return supervise_linux(command, owner)
    if os.name == "nt":
        return supervise_windows(command, sys.argv[2])
    # Other POSIX platforms retain process-group interruption.
    return subprocess.call(command)


if __name__ == "__main__":
    try:
        code = main()
    except Exception as error:
        print(f"Stage process cleanup failed: {error}", file=sys.stderr, flush=True)
        code = 1
    raise SystemExit(128 - code if code < 0 else code)
