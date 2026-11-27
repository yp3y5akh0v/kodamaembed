"""NVIDIA GPU readings through nvidia-smi: starting gates and the monitor that
must cover every timed window."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import time

from _process import Tree

FIELDS = ("timestamp", "enforced.power.limit", "clocks.sm", "clocks.mem", "temperature.gpu",
          "utilization.gpu", "clocks_event_reasons.sw_power_cap", "clocks_event_reasons.sw_thermal_slowdown")
QUERY = "--query-gpu=" + ",".join(FIELDS)
FORMAT = "--format=csv,noheader,nounits"
NUMBERS = ("enforced_power_w", "sm_mhz", "memory_mhz", "temperature_c", "utilization_percent")
MONITOR_INTERVAL_MS = 100
TIMING_MARGIN_MS = 200
MAXIMUM_GAP_MS = 350
GATE_SAMPLES, GATE_INTERVAL_SECONDS = 3, 5.0
# How long the monitor runs before a benchmark starts and after it ends.
MONITOR_LEAD_SECONDS = 0.25
# A context the operator names as idle may use the GPU only this much at a gate.
IDLE_CONTEXT_UTILIZATION = 5
POWER_CAP_POLICY = ("Normal board-cap activity is recorded; only low enforced limits and thermal slowdown "
                    "reject timing.")


@dataclass(frozen=True)
class Limits:
    """What every gate reading and timed reading must satisfy."""

    minimum_power_w: float
    maximum_temperature_c: float
    idle_contexts: bool


def _unix_ms(text):
    for pattern in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            # nvidia-smi stamps readings in local time.
            return round(datetime.strptime(text.strip(), pattern).astimezone().timestamp() * 1000)
        except ValueError:
            continue
    raise ValueError(f"malformed GPU timestamp: {text!r}")


def _number(text):
    value = float(text.strip())
    if not math.isfinite(value):
        raise ValueError("GPU telemetry contains a non-finite value")
    return value


def _reason(text):
    value = text.strip()
    if value not in ("Active", "Not Active"):
        raise ValueError("GPU throttle reason is unavailable or malformed")
    return value == "Active"


def parse(text, stopped_monitor=False):
    """Readings and ignored shutdown fragments from nvidia-smi CSV output. Only a
    stopped monitor's final unterminated row may be partial."""
    lines = text.splitlines()
    if not lines:
        raise ValueError("no GPU telemetry samples were returned")
    readings, fragments = [], []
    for number, line in enumerate(lines, 1):
        fields = line.split(",")
        if not line.strip():
            raise ValueError("empty GPU telemetry row")
        if len(fields) != len(FIELDS):
            if (not stopped_monitor or number != len(lines) or text.endswith(("\n", "\r"))
                    or len(fields) > len(FIELDS)):
                raise ValueError("GPU telemetry row has an incomplete or unexpected field count")
            stamp = _unix_ms(fields[0])
            for field in fields[1:min(6, len(fields))]:
                _number(field)
            if len(fields) == 7:
                _reason(fields[6])
            fragments.append({"line_number": number, "raw": line, "fields_present": len(fields),
                              "unix_ms": stamp,
                              "reason": "Final unterminated partial row after the owned telemetry monitor was stopped."})
            continue
        readings.append({"unix_ms": _unix_ms(fields[0]),
                         **{name: _number(value) for name, value in zip(NUMBERS, fields[1:6])},
                         "software_power_cap": _reason(fields[6]),
                         "software_thermal_slowdown": _reason(fields[7])})
    return readings, fragments


def gate_passes(reading, limits):
    return (reading["enforced_power_w"] >= limits.minimum_power_w
            and not reading["software_power_cap"] and not reading["software_thermal_slowdown"]
            and reading["temperature_c"] <= limits.maximum_temperature_c
            and (not limits.idle_contexts or reading["utilization_percent"] <= IDLE_CONTEXT_UTILIZATION))


def check_timing(text, start_ms, end_ms, limits):
    """The stopped monitor's evidence for one timed window."""
    readings, fragments = parse(text, stopped_monitor=True)
    return judge_timing(readings, fragments, start_ms, end_ms, limits)


def judge_timing(readings, fragments, start_ms, end_ms, limits):
    """One timed window's evidence: `valid` when the readings bracket it with no gap, low
    enforced power, thermal slowdown or overlapping partial row."""
    window = [reading for reading in readings
              if start_ms - TIMING_MARGIN_MS <= reading["unix_ms"] <= end_ms + TIMING_MARGIN_MS]
    reasons = []
    if any(fragment["unix_ms"] <= end_ms for fragment in fragments):
        reasons.append("Incomplete shutdown telemetry overlaps the timed interval.")
    if len(window) < 2 or window[0]["unix_ms"] > start_ms or window[-1]["unix_ms"] < end_ms:
        reasons.append("Telemetry does not bracket the timed interval within the 200 ms margins.")
    if any(not 0 < right["unix_ms"] - left["unix_ms"] <= MAXIMUM_GAP_MS for left, right in zip(window, window[1:])):
        reasons.append("Telemetry sample coverage is incomplete.")
    if any(reading["enforced_power_w"] < limits.minimum_power_w for reading in window):
        reasons.append("Enforced GPU power limit fell below the required minimum during timing.")
    if any(reading["software_thermal_slowdown"] for reading in window):
        reasons.append("GPU software thermal slowdown was active during timing.")
    return {"valid": not reasons, "reasons": reasons, "samples": window,
            "total_monitor_samples": len(readings), "ignored_shutdown_fragments": fragments,
            "software_power_cap_samples": sum(reading["software_power_cap"] for reading in window),
            "power_cap_policy": POWER_CAP_POLICY}


class Smi:
    """nvidia-smi as a command (its program, then any leading arguments); every call
    is an owned process with a deadline."""

    def __init__(self, command, directory):
        self.command, self.directory = list(command), directory

    def query(self, *arguments):
        tree = Tree(self.command[0], [*self.command[1:], *arguments], directory=self.directory)
        try:
            if not tree.wait(10):
                raise RuntimeError("nvidia-smi timed out")
            result = tree.finish()
        finally:
            if not tree.finished:
                tree.finish(stop=True)
        if result.exit_code != 0:
            raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
        return result.stdout

    def hardware(self):
        """The one GPU's name, driver and default power limit, None where the board reports none."""
        rows = [line.split(",") for line in self.query(
            "--query-gpu=index,name,driver_version,power.default_limit", FORMAT).splitlines() if line.strip()]
        if len(rows) != 1 or len(rows[0]) != 4 or rows[0][0].strip() != "0":
            raise RuntimeError("exactly one NVIDIA GPU is required")
        _, name, driver, default_limit = (value.strip() for value in rows[0])
        return {"name": name, "driver_version": driver,
                "default_power_limit_w": None if default_limit.startswith("[") else _number(default_limit)}

    def reading(self):
        readings, _ = parse(self.query("-i", "0", QUERY, FORMAT))
        if len(readings) != 1:
            raise RuntimeError("expected telemetry for exactly one GPU")
        return readings[0]

    def compute_processes(self):
        """(process ID, executable name) of every process with a GPU context."""
        text = self.query("-i", "0", "--query-compute-apps=pid,process_name", "--format=csv,noheader").strip()
        if not text or text.startswith("No running processes found"):
            return []
        processes = []
        for line in text.splitlines():
            pid, _, name = line.partition(",")
            processes.append((int(pid.strip()), name.strip().replace("\\", "/").rsplit("/", 1)[-1]))
        return processes

    def monitor(self, log_dir, log_prefix):
        return Tree(self.command[0], [*self.command[1:], "-i", "0", QUERY, FORMAT, "-lms", str(MONITOR_INTERVAL_MS)],
                    directory=self.directory, log_dir=log_dir, log_prefix=log_prefix)


class GpuWatch:
    """One GPU's evidence around each benchmark: no compute process outside the
    allowlist, a gate of consecutive passing readings, and a monitor over the timing."""

    def __init__(self, smi, limits, allowed_pids, max_wait_seconds, interval_seconds=GATE_INTERVAL_SECONDS):
        self.smi, self.limits = smi, limits
        self.allowed_pids = frozenset(allowed_pids)
        self.max_wait_seconds, self.interval_seconds = max_wait_seconds, interval_seconds
        self.observations = []

    def assert_idle(self, stage):
        """Records every compute process; raises unless each is an allowed idle context."""
        processes = [{"process_id": pid, "name": name, "explicitly_allowed": pid in self.allowed_pids}
                     for pid, name in self.smi.compute_processes()]
        self.observations.append({"checked_utc": datetime.now(timezone.utc).isoformat(), "stage": stage,
                                  "processes": processes})
        if not all(process["explicitly_allowed"] for process in processes):
            raise RuntimeError("A GPU compute process is outside the explicit idle-context allowlist; "
                               "no benchmark was started.")

    def gate(self, stage):
        """Yields readings until GATE_SAMPLES consecutive ones pass, with compute
        processes checked before and after; raises when the wait runs out."""
        self.assert_idle(stage)
        started = time.monotonic()
        consecutive = 0
        while consecutive < GATE_SAMPLES:
            reading = self.smi.reading()
            yield reading
            consecutive = consecutive + 1 if gate_passes(reading, self.limits) else 0
            if consecutive < GATE_SAMPLES:
                if time.monotonic() - started >= self.max_wait_seconds:
                    raise RuntimeError("GPU gate timed out.")
                time.sleep(self.interval_seconds)
        self.assert_idle(stage)

    def start_monitor(self, log_dir, log_prefix):
        """The running monitor; raises if it exits before the benchmark can start."""
        monitor = self.smi.monitor(log_dir, log_prefix)
        try:
            if not monitor.wait_for_stdout(self.max_wait_seconds) or monitor.wait(MONITOR_LEAD_SECONDS):
                raise RuntimeError("GPU telemetry monitor exited before the benchmark or produced no data in time.")
        except BaseException:
            monitor.finish(stop=True)
            raise
        return monitor

    @staticmethod
    def stop_monitor(monitor):
        """The monitor's output once its readings extend past the timing."""
        time.sleep(MONITOR_LEAD_SECONDS)
        return monitor.finish(stop=True).stdout
