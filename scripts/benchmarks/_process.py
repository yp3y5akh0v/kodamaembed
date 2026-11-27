"""Benchmark processes tuned before their first instruction and owned with every descendant
until accounted for: in a kill-on-close job object on Windows, a new session elsewhere."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

if sys.platform == "win32":
    PLATFORM = "windows"
elif sys.platform == "darwin":
    PLATFORM = "macos"
elif sys.platform.startswith("linux"):
    PLATFORM = "linux"
else:
    PLATFORM = sys.platform

PRIORITIES = ("high", "inherit")
# The nice value "high" selects on Linux and macOS; it needs CAP_SYS_NICE or root.
HIGH_NICE = -10
# HIGH_PRIORITY_CLASS and PROCESS_POWER_THROTTLING_EXECUTION_SPEED, as Windows reports them.
WINDOWS_HIGH_PRIORITY_CLASS, WINDOWS_EXECUTION_SPEED = 0x80, 0x1
# Waits, in seconds, for an exit, for descendants to finish and for output to drain.
EXIT_WAIT = 10.0
DESCENDANT_GRACE = 2.0


@dataclass(frozen=True)
class Scheduling:
    """Logical processors and priority for a benchmark; None and "inherit" keep the runner's."""

    cpus: tuple[int, ...] | None = None
    priority: str = "inherit"

    def __post_init__(self):
        if self.priority not in PRIORITIES:
            raise ValueError(f"priority must be one of {PRIORITIES}")
        if self.cpus is not None:
            if not self.cpus or list(self.cpus) != sorted(set(self.cpus)) or self.cpus[0] < 0:
                raise ValueError("processors must be a sorted, unique, nonempty list")
            if PLATFORM == "macos":
                raise ValueError("macOS cannot restrict a process to chosen processors")
            if PLATFORM == "windows" and self.cpus[-1] >= 64:
                raise ValueError("Windows affinity masks select logical processors 0-63")


def parse_cpus(text):
    """Logical processors from a list such as 0-7,16."""
    cpus = set()
    for part in text.split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part.strip(), re.ASCII)
        if match is None:
            raise ValueError(f"invalid processor list: {text!r}")
        first, last = int(match[1]), int(match[2] or match[1])
        if last < first:
            raise ValueError(f"descending processor range: {part!r}")
        cpus.update(range(first, last + 1))
    return tuple(sorted(cpus))


@dataclass(frozen=True)
class Finished:
    exit_code: int
    stdout: str
    stderr: str


class _Output:
    """Drains one pipe on its own thread, so a full pipe never blocks the program."""

    def __init__(self, stream):
        self.chunks = []
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._drain, args=(stream,), daemon=True)
        self.thread.start()

    def _drain(self, stream):
        try:
            with stream:
                for chunk in iter(partial(stream.read1, 65536), b""):
                    self.chunks.append(chunk)
                    if len(self.chunks) == 1:
                        self.ready.set()
        finally:
            self.ready.set()

    def collect(self, timeout):
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise RuntimeError("owned output did not drain")
        return b"".join(self.chunks)


if PLATFORM == "windows":
    import ctypes
    from ctypes import wintypes

    _kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    class _Limits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _Limits), ("IoInfo", ctypes.c_uint64 * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class _Accounting(ctypes.Structure):
        _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
                    ("ThisPeriodTotalUserTime", ctypes.c_int64), ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                    ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
                    ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD)]

    class _ThreadEntry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ThreadID", wintypes.DWORD), ("th32OwnerProcessID", wintypes.DWORD),
                    ("tpBasePri", wintypes.LONG), ("tpDeltaPri", wintypes.LONG), ("dwFlags", wintypes.DWORD)]

    class _PowerThrottling(ctypes.Structure):
        _fields_ = [("Version", wintypes.DWORD), ("ControlMask", wintypes.DWORD), ("StateMask", wintypes.DWORD)]

    for _name, _arguments, _result in (
        ("CreateJobObjectW", [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
        ("SetInformationJobObject", [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
        ("QueryInformationJobObject",
         [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL),
        ("AssignProcessToJobObject", [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        ("TerminateJobObject", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
        ("OpenProcess", [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
        ("OpenThread", [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
        ("ResumeThread", [wintypes.HANDLE], wintypes.DWORD),
        ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
        ("CreateToolhelp32Snapshot", [wintypes.DWORD, wintypes.DWORD], wintypes.HANDLE),
        ("Thread32First", [wintypes.HANDLE, ctypes.c_void_p], wintypes.BOOL),
        ("Thread32Next", [wintypes.HANDLE, ctypes.c_void_p], wintypes.BOOL),
        ("GetCurrentProcess", [], wintypes.HANDLE),
        ("GetPriorityClass", [wintypes.HANDLE], wintypes.DWORD),
        ("SetPriorityClass", [wintypes.HANDLE, wintypes.DWORD], wintypes.BOOL),
        ("GetProcessAffinityMask",
         [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)], wintypes.BOOL),
        ("SetProcessAffinityMask", [wintypes.HANDLE, ctypes.c_size_t], wintypes.BOOL),
        ("GetProcessInformation", [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
        ("SetProcessInformation", [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
    ):
        _function = getattr(_kernel, _name)
        _function.argtypes, _function.restype = _arguments, _result

    _ACCOUNTING, _EXTENDED_LIMITS = 1, 9
    _KILL_ON_JOB_CLOSE = 0x2000
    _CREATE_SUSPENDED, _CREATE_NO_WINDOW = 0x4, 0x08000000
    # PROCESS_TERMINATE | PROCESS_SET_QUOTA | PROCESS_SET_INFORMATION | PROCESS_QUERY_INFORMATION.
    _PROCESS_ACCESS = 0x1 | 0x100 | 0x200 | 0x400
    _THREAD_SUSPEND_RESUME = 0x2
    _SNAPSHOT_THREADS = 0x4
    _POWER_THROTTLING = 4
    # ERROR_TIMEOUT, as the exit status of terminated processes.
    _TERMINATED = 1460

    def _checked(value):
        if not value:
            raise ctypes.WinError(ctypes.get_last_error())
        return value

    def _windows_scheduling(handle):
        current, system = ctypes.c_size_t(), ctypes.c_size_t()
        _checked(_kernel.GetProcessAffinityMask(handle, ctypes.byref(current), ctypes.byref(system)))
        power = _PowerThrottling(1, 0, 0)
        _checked(_kernel.GetProcessInformation(handle, _POWER_THROTTLING, ctypes.byref(power), ctypes.sizeof(power)))
        return {"platform": "windows", "priority_class": int(_checked(_kernel.GetPriorityClass(handle))),
                "affinity_mask": current.value,
                "execution_speed_control_mask": power.ControlMask,
                "execution_speed_state_mask": power.StateMask}

    def _tune_windows(handle, scheduling):
        if scheduling.cpus is not None:
            _checked(_kernel.SetProcessAffinityMask(handle, sum(1 << cpu for cpu in scheduling.cpus)))
        if scheduling.priority == "high":
            _checked(_kernel.SetPriorityClass(handle, WINDOWS_HIGH_PRIORITY_CLASS))
        power = _PowerThrottling(1, WINDOWS_EXECUTION_SPEED, 0)
        _checked(_kernel.SetProcessInformation(handle, _POWER_THROTTLING, ctypes.byref(power), ctypes.sizeof(power)))

    def _resume_threads(pid):
        snapshot = _kernel.CreateToolhelp32Snapshot(_SNAPSHOT_THREADS, 0)
        if snapshot in (None, wintypes.HANDLE(-1).value):
            raise ctypes.WinError(ctypes.get_last_error())
        resumed = 0
        try:
            entry = _ThreadEntry(dwSize=ctypes.sizeof(_ThreadEntry))
            more = _kernel.Thread32First(snapshot, ctypes.byref(entry))
            while more:
                if entry.th32OwnerProcessID == pid:
                    thread = _checked(_kernel.OpenThread(_THREAD_SUSPEND_RESUME, False, entry.th32ThreadID))
                    try:
                        if _kernel.ResumeThread(thread) == 0xFFFFFFFF:
                            raise ctypes.WinError(ctypes.get_last_error())
                    finally:
                        _kernel.CloseHandle(thread)
                    resumed += 1
                more = _kernel.Thread32Next(snapshot, ctypes.byref(entry))
        finally:
            _kernel.CloseHandle(snapshot)
        if not resumed:
            raise RuntimeError("the suspended process has no thread to resume")


def current_scheduling():
    """This process's scheduling, as an owned benchmark reports it."""
    if PLATFORM == "windows":
        state = _windows_scheduling(_kernel.GetCurrentProcess())
    elif PLATFORM == "linux":
        state = {"platform": "linux", "cpus": sorted(os.sched_getaffinity(0)),
                 "nice": os.getpriority(os.PRIO_PROCESS, 0)}
    else:
        state = {"platform": PLATFORM, "nice": os.getpriority(os.PRIO_PROCESS, 0)}
    return {**state, "process_id": os.getpid()}


def configure_current(scheduling):
    """Applies `scheduling` to this process and returns the readback; a Windows
    venv launcher does not pass priority or power policy to its interpreter."""
    if PLATFORM == "windows":
        _tune_windows(_kernel.GetCurrentProcess(), scheduling)
    else:
        if scheduling.cpus is not None:
            os.sched_setaffinity(0, scheduling.cpus)
        if scheduling.priority == "high":
            os.setpriority(os.PRIO_PROCESS, 0, HIGH_NICE)
    state = current_scheduling()
    if not satisfies(state, scheduling):
        raise RuntimeError(f"process scheduling readback differs: {state}")
    return state


def fulfills(state, cpus, priority):
    """Whether a readback, read as the platform it names, carries out logical
    processors `cpus` (None for any) at `priority`."""
    if state["platform"] == "windows":
        return ((cpus is None or state["affinity_mask"] == sum(1 << cpu for cpu in cpus))
                and (priority != "high" or state["priority_class"] == WINDOWS_HIGH_PRIORITY_CLASS)
                and state["execution_speed_control_mask"] == WINDOWS_EXECUTION_SPEED
                and state["execution_speed_state_mask"] == 0)
    if state["platform"] in ("linux", "macos"):
        return ((cpus is None or state.get("cpus") == list(cpus))
                and (priority != "high" or state["nice"] == HIGH_NICE))
    return False


def satisfies(state, scheduling):
    """Whether a readback from this platform carries out `scheduling`."""
    return state["platform"] == PLATFORM and fulfills(state, scheduling.cpus, scheduling.priority)


# Applies and reports the scheduling, then execs the benchmark; the report pipe closes
# on a successful exec, so a second report line can only be an exec error.
_TRAMPOLINE = """\
import json, os, sys
settings = json.loads(sys.argv[1])
report = settings["report"]
if settings["cpus"] is not None:
    os.sched_setaffinity(0, settings["cpus"])
if settings["nice"] is not None:
    os.setpriority(os.PRIO_PROCESS, 0, settings["nice"])
if settings["parent"] is not None:
    import ctypes
    if ctypes.CDLL(None, use_errno=True).prctl(1, 9) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG)")
    if os.getppid() != settings["parent"]:
        os._exit(1)
state = {"nice": os.getpriority(os.PRIO_PROCESS, 0)}
if hasattr(os, "sched_getaffinity"):
    state["cpus"] = sorted(os.sched_getaffinity(0))
os.write(report, json.dumps(state).encode() + b"\\n")
os.set_inheritable(report, False)
try:
    os.execv(sys.argv[2], sys.argv[2:])
except OSError as error:
    os.write(report, json.dumps([error.errno, error.strerror]).encode())
    os._exit(127)
"""

_subreaper = False
_owned_leaders = set()


def _become_subreaper():
    global _subreaper
    if not _subreaper:
        import ctypes

        # PR_SET_CHILD_SUBREAPER.
        if ctypes.CDLL(None, use_errno=True).prctl(36, 1) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_CHILD_SUBREAPER)")
        _subreaper = True


def _orphans():
    """Live processes reparented to this runner that no owned tree leads; exited
    ones are reaped, since an unreaped zombie still looks alive."""
    live = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in _owned_leaders:
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        # The command name in parentheses may hold spaces; the parent follows the state.
        if int(stat[stat.rindex(")") + 2:].split()[1]) != os.getpid():
            continue
        try:
            reaped, _ = os.waitpid(int(entry.name), os.WNOHANG)
        except ChildProcessError:
            continue
        if reaped == 0:
            live.append(int(entry.name))
    return live


def _reap(pid):
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass


def _signal_group(leader, number):
    """Whether the group took the signal. For a group whose members have all exited,
    macOS answers that the signal is not permitted where Linux still delivers it."""
    try:
        os.killpg(leader, number)
    except ProcessLookupError:
        return False
    except PermissionError:
        if PLATFORM != "macos":
            raise
        return False
    return True


class Tree:
    """One started program and its descendants, owned until `finish`."""

    def __init__(self, executable, arguments, *, directory, scheduling=None, log_dir=None, log_prefix=""):
        if PLATFORM not in ("windows", "linux", "macos"):
            raise RuntimeError(f"unsupported platform: {PLATFORM}")
        self.log_dir, self.log_prefix = log_dir, log_prefix
        self.finished = False
        self.scheduling = None
        command = [str(executable), *map(str, arguments)]
        if PLATFORM == "windows":
            self._start_windows(command, directory, scheduling)
        else:
            self._start_posix(command, directory, scheduling)
        self.pid = self._process.pid
        self._stdout = _Output(self._process.stdout)
        self._stderr = _Output(self._process.stderr)

    def _start_windows(self, command, directory, scheduling):
        self._job = _checked(_kernel.CreateJobObjectW(None, None))
        handle = None
        try:
            limits = _ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
            _checked(_kernel.SetInformationJobObject(self._job, _EXTENDED_LIMITS, ctypes.byref(limits),
                                                     ctypes.sizeof(limits)))
            # Suspension keeps descendants from starting before the job owns the process.
            self._process = subprocess.Popen(command, cwd=directory, stdin=subprocess.DEVNULL,
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                             creationflags=_CREATE_SUSPENDED | _CREATE_NO_WINDOW)
            try:
                handle = _checked(_kernel.OpenProcess(_PROCESS_ACCESS, False, self._process.pid))
                _checked(_kernel.AssignProcessToJobObject(self._job, handle))
                if scheduling is not None:
                    _tune_windows(handle, scheduling)
                    self.scheduling = _windows_scheduling(handle)
                    if not satisfies(self.scheduling, scheduling):
                        raise RuntimeError(f"process scheduling readback differs: {self.scheduling}")
                _resume_threads(self._process.pid)
            except BaseException:
                self._process.kill()
                self._process.wait(EXIT_WAIT)
                for stream in (self._process.stdout, self._process.stderr):
                    stream.close()
                raise
        except BaseException:
            _kernel.CloseHandle(self._job)
            raise
        finally:
            if handle is not None:
                _kernel.CloseHandle(handle)

    def _start_posix(self, command, directory, scheduling):
        requested = scheduling is not None
        scheduling = scheduling or Scheduling()
        if PLATFORM == "linux":
            _become_subreaper()
        report_read, report_write = os.pipe()
        settings = {"cpus": list(scheduling.cpus) if scheduling.cpus is not None else None,
                    "nice": HIGH_NICE if scheduling.priority == "high" else None,
                    "parent": os.getpid() if PLATFORM == "linux" else None, "report": report_write}
        trampoline = [sys.executable, "-I", "-S", "-c", _TRAMPOLINE, json.dumps(settings)]
        try:
            self._process = subprocess.Popen(trampoline + command, cwd=directory, stdin=subprocess.DEVNULL,
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                             start_new_session=True, pass_fds=(report_write,))
        finally:
            os.close(report_write)
        _owned_leaders.add(self._process.pid)
        with os.fdopen(report_read, "rb") as report:
            lines = report.read().splitlines()
        state = {"platform": PLATFORM, **json.loads(lines[0])} if lines else None
        if len(lines) != 1 or not satisfies(state, scheduling):
            self.pid = self._process.pid
            self._kill_posix()
            self._process.wait(EXIT_WAIT)
            stderr = self._process.stderr.read().decode("utf-8", "replace")
            for stream in (self._process.stdout, self._process.stderr):
                stream.close()
            _owned_leaders.discard(self._process.pid)
            if len(lines) == 2:
                number, message = json.loads(lines[1])
                raise OSError(number, message, command[0])
            if state is None:
                raise RuntimeError(f"the benchmark could not be started with its scheduling: {stderr}")
            raise RuntimeError(f"process scheduling readback differs: {state}")
        self.scheduling = state if requested else None

    def running(self):
        return self._process.poll() is None

    def wait_for_stdout(self, timeout):
        """Whether the program wrote to stdout before EOF or the deadline."""
        return self._stdout.ready.wait(timeout) and bool(self._stdout.chunks)

    def wait(self, timeout):
        """Whether the program exited within `timeout` seconds."""
        try:
            self._process.wait(timeout)
        except subprocess.TimeoutExpired:
            return False
        return True

    def _active_windows(self):
        accounting = _Accounting()
        _checked(_kernel.QueryInformationJobObject(self._job, _ACCOUNTING, ctypes.byref(accounting),
                                                   ctypes.sizeof(accounting), None))
        return accounting.ActiveProcesses

    def _descendants_posix(self):
        """Whether any live process of the tree remains; exited members are reaped
        first, since zombies still count as group members."""
        self._process.poll()
        orphans = _orphans() if PLATFORM == "linux" else []
        return _signal_group(self.pid, 0) or bool(orphans)

    def _kill_posix(self):
        _signal_group(self.pid, signal.SIGKILL)
        if PLATFORM == "linux":
            for orphan in _orphans():
                os.kill(orphan, signal.SIGKILL)
                _reap(orphan)

    def _descendants(self):
        return self._active_windows() > 0 if PLATFORM == "windows" else self._descendants_posix()

    def _kill(self):
        if PLATFORM == "windows":
            if self._active_windows() > 0:
                _checked(_kernel.TerminateJobObject(self._job, _TERMINATED))
        else:
            self._kill_posix()

    def _wait_empty(self, timeout):
        deadline = time.monotonic() + timeout
        while self._descendants():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True

    def finish(self, stop=False):
        """Ends ownership: with `stop`, the tree is killed; otherwise a program that
        left live descendants fails. Output is drained and logged either way."""
        if self.finished:
            raise RuntimeError("owned process already finished")
        self.finished = True
        failure = None
        try:
            if stop:
                self._kill()
            if not self.wait(EXIT_WAIT):
                raise RuntimeError("owned process did not exit within ten seconds")
            if not stop and not self._wait_empty(DESCENDANT_GRACE):
                raise RuntimeError("owned process left live descendants")
        except Exception as error:
            failure = error
        try:
            # A process group is signaled only while a member keeps its number from reuse.
            if self._descendants():
                self._kill()
                if not self._wait_empty(EXIT_WAIT):
                    raise RuntimeError("owned descendants survived bounded termination")
            if not self.wait(EXIT_WAIT):
                raise RuntimeError("owned leader survived bounded termination")
        except Exception as error:
            failure = failure or error
        stdout = stderr = b""
        try:
            stdout = self._stdout.collect(EXIT_WAIT)
            stderr = self._stderr.collect(EXIT_WAIT)
        except Exception as error:
            failure = failure or error
        finally:
            if self.log_dir is not None and self.log_prefix:
                (Path(self.log_dir) / f"{self.log_prefix}.stdout.log").write_bytes(stdout)
                (Path(self.log_dir) / f"{self.log_prefix}.stderr.log").write_bytes(stderr)
            if PLATFORM == "windows":
                _kernel.CloseHandle(self._job)
            else:
                _owned_leaders.discard(self.pid)
        if failure is not None:
            raise failure
        return Finished(self._process.returncode, stdout.decode("utf-8", "replace"),
                        stderr.decode("utf-8", "replace"))
