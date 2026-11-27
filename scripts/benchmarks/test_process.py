"""Owned benchmark processes on this operating system: tuning before the first
instruction, captured output, and no descendant outliving its tree."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import _process
from _process import PLATFORM, Scheduling, Tree, parse_cpus

# A Windows venv's python.exe is a launcher that runs the interpreter as a second
# process; the tests start the interpreter itself, the process whose tuning they read.
PYTHON = sys._base_executable


def alive(pid):
    if PLATFORM == "windows":
        import ctypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        # PROCESS_QUERY_LIMITED_INFORMATION; STILL_ACTIVE is 259.
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
            return code.value == 259
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    if PLATFORM == "linux":
        try:
            return Path(f"/proc/{pid}/stat").read_text().split(") ")[1][0] != "Z"
        except OSError:
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def gone(pid, timeout=10.0):
    deadline = time.monotonic() + timeout
    while alive(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def first_cpu():
    if PLATFORM == "linux":
        return min(os.sched_getaffinity(0))
    return 0


class ProcessorListTests(unittest.TestCase):
    def test_lists_expand_ranges_and_sort_unique_processors(self):
        self.assertEqual(parse_cpus("4-6,0,5"), (0, 4, 5, 6))
        self.assertEqual(parse_cpus("7"), (7,))
        for text in ("", "1-", "-1", "3-1", "a", "1,,2", "١"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_cpus(text)

    def test_scheduling_rejects_what_this_system_cannot_honor(self):
        with self.assertRaises(ValueError):
            Scheduling(priority="realtime")
        for cpus in ((), (2, 1), (1, 1), (-1,)):
            with self.subTest(cpus=cpus), self.assertRaises(ValueError):
                Scheduling(cpus=cpus)
        if PLATFORM == "macos":
            with self.assertRaisesRegex(ValueError, "macOS"):
                Scheduling(cpus=(0,))
        if PLATFORM == "windows":
            with self.assertRaisesRegex(ValueError, "0-63"):
                Scheduling(cpus=(64,))


class ReadbackTests(unittest.TestCase):
    """Readbacks as each platform reports them, judged on any platform."""

    WINDOWS = {"platform": "windows", "priority_class": 0x80, "affinity_mask": 0b1010,
               "execution_speed_control_mask": 1, "execution_speed_state_mask": 0}
    LINUX = {"platform": "linux", "cpus": [1, 3], "nice": _process.HIGH_NICE}
    MACOS = {"platform": "macos", "nice": _process.HIGH_NICE}

    def test_processors_and_priority_are_read_as_the_named_platform_reports_them(self):
        for state in (self.WINDOWS, self.LINUX):
            with self.subTest(platform=state["platform"]):
                self.assertTrue(_process.fulfills(state, (1, 3), "high"))
                self.assertTrue(_process.fulfills(state, None, "inherit"))
                self.assertFalse(_process.fulfills(state, (1, 2), "inherit"))
        self.assertTrue(_process.fulfills(self.MACOS, None, "high"))
        for state in ({**self.WINDOWS, "priority_class": 0x20}, {**self.LINUX, "nice": 0}, {**self.MACOS, "nice": 0}):
            with self.subTest(platform=state["platform"]):
                self.assertFalse(_process.fulfills(state, None, "high"))

    def test_windows_requires_the_power_throttling_opt_out(self):
        for change in ({"execution_speed_control_mask": 0}, {"execution_speed_state_mask": 1}):
            with self.subTest(change=change):
                self.assertFalse(_process.fulfills({**self.WINDOWS, **change}, None, "inherit"))

    def test_only_a_readback_from_this_platform_satisfies_its_scheduling(self):
        native = {"windows": self.WINDOWS, "linux": self.LINUX, "macos": self.MACOS}[PLATFORM]
        foreign = self.LINUX if PLATFORM == "macos" else self.MACOS
        self.assertTrue(_process.satisfies(native, Scheduling(priority="high")))
        self.assertTrue(_process.fulfills(foreign, None, "high"))
        self.assertFalse(_process.satisfies(foreign, Scheduling(priority="high")))
        self.assertFalse(_process.fulfills({**self.LINUX, "platform": "other"}, None, "inherit"))


class TreeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def start(self, code, **options):
        return Tree(PYTHON, ["-c", code], directory=self.root, **options)

    def test_output_exit_status_and_logs_are_kept(self):
        tree = self.start("import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)",
                          log_dir=self.root, log_prefix="leg")
        result = tree.finish()
        self.assertEqual((result.exit_code, result.stdout.strip(), result.stderr.strip()), (3, "out", "err"))
        self.assertEqual((self.root / "leg.stdout.log").read_bytes().strip(), b"out")
        self.assertEqual((self.root / "leg.stderr.log").read_bytes().strip(), b"err")

    def test_output_beyond_a_pipe_buffer_does_not_stall_the_program(self):
        tree = self.start("import sys; sys.stdout.write('x' * 4_000_000)")
        self.assertTrue(tree.wait(30))
        self.assertEqual(len(tree.finish().stdout), 4_000_000)

    def test_stopping_kills_a_running_tree(self):
        tree = self.start("import time; time.sleep(600)")
        self.assertTrue(tree.running())
        started = time.monotonic()
        tree.finish(stop=True)
        self.assertLess(time.monotonic() - started, 9)
        self.assertTrue(gone(tree.pid))

    def test_stopping_a_program_that_has_exited_returns_its_result(self):
        tree = self.start("import sys; print('err', file=sys.stderr); sys.exit(3)")
        # The end of stdout comes with the exit, and nothing has waited for the program yet.
        self.assertFalse(tree.wait_for_stdout(30))
        time.sleep(0.2)
        result = tree.finish(stop=True)
        self.assertEqual((result.exit_code, result.stderr.strip()), (3, "err"))

    def assert_descendant_found_and_killed(self, options):
        """Runs a program that starts a sleeping child with `options` and exits."""
        code = ("import subprocess, sys; child = subprocess.Popen([sys.executable, '-c', "
                f"'import time; time.sleep(600)']{options}); print(child.pid, flush=True)")
        tree = self.start(code, log_dir=self.root, log_prefix="leg")
        self.assertTrue(tree.wait(30))
        with self.assertRaisesRegex(RuntimeError, "live descendants"):
            tree.finish()
        grandchild = int((self.root / "leg.stdout.log").read_text())
        self.assertTrue(gone(grandchild))

    def test_a_descendant_left_alive_fails_the_tree_and_is_killed(self):
        self.assert_descendant_found_and_killed("")

    @unittest.skipUnless(PLATFORM in ("windows", "linux"), "macOS cannot follow a child out of its group")
    def test_a_descendant_that_detaches_is_still_found(self):
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, or a new session.
        self.assert_descendant_found_and_killed(
            ", creationflags=0x208" if PLATFORM == "windows" else ", start_new_session=True")

    def test_a_missing_program_fails_to_start(self):
        with self.assertRaises((OSError, RuntimeError)):
            Tree(self.root / "missing-program", [], directory=self.root)

    @unittest.skipIf(PLATFORM == "macos", "macOS has no processor affinity")
    def test_the_program_runs_on_its_chosen_processors(self):
        cpu = first_cpu()
        code = ("import json, _process; print(json.dumps(_process.current_scheduling()))")
        tree = Tree(PYTHON, ["-c", code], directory=Path(_process.__file__).parent,
                    scheduling=Scheduling(cpus=(cpu,)))
        result = tree.finish()
        self.assertEqual(result.exit_code, 0, result.stderr)
        seen = json.loads(result.stdout)
        if PLATFORM == "windows":
            self.assertEqual(seen["affinity_mask"], 1 << cpu)
            self.assertEqual((seen["execution_speed_control_mask"], seen["execution_speed_state_mask"]), (1, 0))
            self.assertEqual(tree.scheduling["affinity_mask"], 1 << cpu)
        else:
            self.assertEqual(seen["cpus"], [cpu])
            self.assertEqual(tree.scheduling["cpus"], [cpu])

    def test_high_priority_is_applied_or_refused_never_ignored(self):
        code = "import json, _process; print(json.dumps(_process.current_scheduling()))"
        directory = Path(_process.__file__).parent
        if PLATFORM == "windows":
            result = Tree(PYTHON, ["-c", code], directory=directory,
                          scheduling=Scheduling(priority="high")).finish()
            self.assertEqual(json.loads(result.stdout)["priority_class"], 0x80)
            return
        try:
            tree = Tree(PYTHON, ["-c", code], directory=directory, scheduling=Scheduling(priority="high"))
        except RuntimeError as error:
            self.assertNotEqual(os.geteuid(), 0, error)
            return
        self.assertEqual(json.loads(tree.finish().stdout)["nice"], _process.HIGH_NICE)


@unittest.skipUnless(PLATFORM in ("windows", "linux"), "macOS cannot tie a child's life to the runner")
class RunnerDeathTests(unittest.TestCase):
    def test_the_program_dies_with_its_runner(self):
        helper = ("import sys, time; sys.path.insert(0, sys.argv[1]); import _process; "
                  "tree = _process.Tree(sys.executable, ['-c', 'import time; time.sleep(600)'], directory='.'); "
                  "print(tree.pid, flush=True); time.sleep(600)")
        runner = subprocess.Popen([PYTHON, "-c", helper, str(Path(_process.__file__).parent)],
                                  stdout=subprocess.PIPE)
        program = int(runner.stdout.readline())
        self.assertTrue(alive(program))
        if PLATFORM == "windows":
            runner.kill()
        else:
            os.kill(runner.pid, signal.SIGKILL)
        runner.wait()
        runner.stdout.close()
        self.assertTrue(gone(program))


if __name__ == "__main__":
    unittest.main(verbosity=2)
