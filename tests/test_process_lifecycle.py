"""AI-written process-tree regressions using real owned subprocesses."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from process_lifecycle import owned_process, stop_process
from tests import EXECUTOR


LEAF = r'''
import json,os,signal,sys,time
from pathlib import Path
signal.signal(signal.SIGINT, signal.SIG_IGN)
if sys.argv[2] == "stubborn":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(json.dumps({"leaf":os.getpid(), "parent":os.getppid()}))
time.sleep(60)
'''

TREE = r'''
import json,os,subprocess,sys,time
from pathlib import Path
marker,mode,leaf_code = sys.argv[1:]
options = ({"start_new_session":True} if os.name == "posix" else
           {"creationflags":subprocess.CREATE_NEW_PROCESS_GROUP})
child = subprocess.Popen([sys.executable,"-B","-c",leaf_code,marker,
                         "stubborn" if mode == "stubborn" else "normal"],**options)
while not Path(marker).exists():
    time.sleep(.01)
print("READY "+json.dumps({"launcher":os.getpid(), "leaf":child.pid}),flush=True)
if mode in ("success", "failure"):
    raise SystemExit(0 if mode == "success" else 7)
time.sleep(60)
'''


def alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes as w
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.OpenProcess.restype, api.OpenProcess.argtypes = w.HANDLE, [w.DWORD, w.BOOL, w.DWORD]
        api.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
        api.CloseHandle.argtypes = [w.HANDLE]
        handle = api.OpenProcess(0x100000, False, pid)
        if not handle:
            return False
        try:
            return api.WaitForSingleObject(handle, 0) == 258
        finally:
            api.CloseHandle(handle)
    return Path(f"/proc/{pid}").exists()


def wait_for(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for the subprocess condition")
        time.sleep(.02)


@unittest.skipUnless(sys.platform == "linux" or os.name == "nt", "Owned-tree platform required")
class ProcessTreeTests(unittest.TestCase):
    def command(self, root, mode="wait"):
        return [sys.executable, "-B", "-c", TREE, str(root / "leaf.json"), mode, LEAF]

    def ready(self, process):
        line = process.stdout.readline()
        self.assertTrue(line.startswith("READY "), line)
        return json.loads(line.removeprefix("READY "))

    def test_exited_launcher_does_not_leave_detached_child_or_hold_output_pipe(self):
        for mode, expected in (("success", 0), ("failure", 7)):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                with owned_process(self.command(Path(directory), mode), stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True) as process:
                    pids = self.ready(process)
                    output, errors = process.communicate(timeout=15)
                    self.assertEqual(process.returncode, expected, output + errors)
                self.assertTrue(all(not alive(pid) for pid in pids.values()), pids)

    def test_interrupt_reaps_detached_child_after_launcher_exits(self):
        with TemporaryDirectory() as directory:
            with owned_process(self.command(Path(directory)), stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, text=True) as process:
                pids = self.ready(process)
                stop_process(process)
                self.assertEqual(process.returncode, 130)
            self.assertTrue(all(not alive(pid) for pid in pids.values()), pids)

    def test_kill_escalation_reaps_workers_ignoring_interrupt_and_termination(self):
        with TemporaryDirectory() as directory:
            with owned_process(self.command(Path(directory), "stubborn"), stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, text=True) as process:
                pids = self.ready(process)
                stop_process(process)
            self.assertTrue(all(not alive(pid) for pid in pids.values()), pids)

    def test_cleanup_never_stops_an_unrelated_process(self):
        unrelated = subprocess.Popen([sys.executable, "-B", "-c", "import time;time.sleep(60)"])
        try:
            with owned_process([sys.executable, "-B", "-c", "import time;time.sleep(60)"]):
                pass  # Leaving a running context must terminate its own command.
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)

    def test_interrupt_during_process_creation_still_cleans_launched_supervisor(self):
        if os.name == "nt":
            self.skipTest("Uses a real POSIX signal to the creating thread")
        original, launched = subprocess.Popen, []

        def interrupted(*args, **kwargs):
            process = original(*args, **kwargs)
            launched.append(process)
            os.kill(os.getpid(), signal.SIGINT)
            return process

        with patch("process_lifecycle.subprocess.Popen", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                with owned_process([sys.executable, "-c", "import time;time.sleep(60)"]):
                    self.fail("An interrupted launch must not start the stage body")
        self.assertEqual(len(launched), 1)
        self.assertIsNotNone(launched[0].poll())

    def test_stage_exception_cleans_workers_and_closes_pipes(self):
        from runner import Stage, run_stage
        with TemporaryDirectory() as directory:
            root = Path(directory)
            command = self.command(root)
            pids = {}

            def update(line):
                if line.startswith("READY "):
                    pids.update(json.loads(line.removeprefix("READY ")))
                    raise KeyboardInterrupt()

            with self.assertRaises(KeyboardInterrupt):
                run_stage(command, Stage("eval", "test", command, (), 1), root, root / "stage.log", update)
            self.assertTrue(pids)
            self.assertTrue(all(not alive(pid) for pid in pids.values()), pids)


@unittest.skipUnless(sys.platform == "linux" or os.name == "nt", "Owned-tree platform required")
class ManagerLifetimeTests(unittest.TestCase):
    def launch_manager(self, root):
        source = r'''
from pathlib import Path
import json,os,signal,sys
sys.path.insert(0,sys.argv[1])
from runner import Stage,interrupt_on_termination,run_stage
root=Path(sys.argv[2])
command=json.loads(sys.argv[3])
try:
    with interrupt_on_termination():
        def update(line):
            if line.startswith("READY "):
                (root/"ready.json").write_text(line.removeprefix("READY "))
        run_stage(command,Stage("eval","test",command,(),1),root,root/"stage.log",update)
except KeyboardInterrupt:
    (root/"interrupted").touch()
    raise SystemExit(130)
'''
        command = [sys.executable, "-B", "-c", TREE, str(root / "leaf.json"), "stubborn", LEAF]
        process = subprocess.Popen([sys.executable, "-B", "-c", source, str(EXECUTOR), str(root),
                                    json.dumps(command)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(self.finish_manager, process)
        wait_for(lambda: (root / "ready.json").exists() or process.poll() is not None)
        self.assertTrue((root / "ready.json").exists(), process.communicate() if process.poll() is not None else "")
        return process, json.loads((root / "ready.json").read_text())

    @staticmethod
    def finish_manager(process):
        if process.poll() is None:
            process.kill()  # The supervisor must handle owner death too.
        process.communicate(timeout=20)

    @unittest.skipUnless(sys.platform == "linux", "Uses real POSIX signals to the manager")
    def test_repeated_ctrl_c_does_not_interrupt_cleanup(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manager, pids = self.launch_manager(root)
            manager.send_signal(signal.SIGINT)
            for _ in range(4):
                time.sleep(.1)
                manager.send_signal(signal.SIGINT)
            output, errors = manager.communicate(timeout=18)
            self.assertEqual(manager.returncode, 130, output + errors)
            self.assertTrue((root / "interrupted").exists())
            self.assertTrue(all(not alive(pid) for pid in pids.values()), pids)

    def test_abrupt_manager_death_still_cleans_its_workers(self):
        with TemporaryDirectory() as directory:
            manager, pids = self.launch_manager(Path(directory))
            manager.kill()
            manager.wait(timeout=5)
            wait_for(lambda: all(not alive(pid) for pid in pids.values()), seconds=18)
