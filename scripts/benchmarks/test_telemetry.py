"""GPU gate and telemetry rules without a GPU."""

from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

from _telemetry import GpuWatch, Limits, Smi, check_timing, gate_passes, parse

LIMITS = Limits(minimum_power_w=100, maximum_temperature_c=82, idle_contexts=False)
SCRIPTED_SMI = Path(__file__).with_name("scripted_nvidia_smi.py")
IDLE = {"enforced_power_w": 156.85, "temperature_c": 45, "utilization_percent": 3,
        "software_power_cap": False, "software_thermal_slowdown": False}
HOT = {**IDLE, "temperature_c": 95}


def stamp(text):
    return round(datetime.strptime(text, "%Y/%m/%d %H:%M:%S.%f").astimezone().timestamp() * 1000)


def row(time, power=156.85, temperature=64, utilization=67, cap="Not Active", thermal="Not Active"):
    return f"2026/11/24 00:19:{time}, {power}, 1912, 8001, {temperature}, {utilization}, {cap}, {thermal}"


ROWS = [row("40.000"), row("40.100"), row("40.200")]
COMPLETE = "\r\n".join(ROWS) + "\r\n"
START, END = stamp("2026/11/24 00:19:40.050"), stamp("2026/11/24 00:19:40.150")


class ParseTests(unittest.TestCase):
    def test_rows_become_readings_in_local_time(self):
        readings, fragments = parse(COMPLETE)
        self.assertEqual(fragments, [])
        self.assertEqual(readings[0], {
            "unix_ms": stamp("2026/11/24 00:19:40.000"), "enforced_power_w": 156.85, "sm_mhz": 1912.0,
            "memory_mhz": 8001.0, "temperature_c": 64.0, "utilization_percent": 67.0,
            "software_power_cap": False, "software_thermal_slowdown": False})

    def test_malformed_rows_are_rejected(self):
        for text in ("", row("40.000").replace("64", "nan"), row("40.000").replace("Not Active", "[N/A]", 1),
                     row("40.000") + ", extra", "\r\n".join([ROWS[0], " ", ROWS[1]])):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse(text)


class TimingTests(unittest.TestCase):
    def test_a_stopped_monitors_final_partial_row_after_timing_is_recorded(self):
        telemetry = check_timing(COMPLETE + "2026/11/24 00:19:40.300", START, END, LIMITS)
        self.assertTrue(telemetry["valid"], telemetry["reasons"])
        self.assertEqual((len(telemetry["samples"]), len(telemetry["ignored_shutdown_fragments"])), (3, 1))

    def test_a_partial_row_inside_timing_rejects_the_timing(self):
        telemetry = check_timing(COMPLETE + "2026/11/24 00:19:40.120", START, END, LIMITS)
        self.assertFalse(telemetry["valid"])

    def test_other_partial_rows_are_errors(self):
        for name, text, stopped in (
            ("from a running monitor", COMPLETE + "2026/11/24 00:19:40.300", False),
            ("terminated", COMPLETE + "2026/11/24 00:19:40.300\r\n", True),
            ("before the last", ROWS[0] + "\r\n2026/11/24 00:19:40.100, 156.85\r\n" + ROWS[2], True),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                parse(text, stopped_monitor=stopped)

    def test_each_rejection_reason(self):
        cases = {
            "bracket": ("\r\n".join([row("40.100"), row("40.200")]) + "\r\n", "bracket"),
            "gap": ("\r\n".join([row("39.900"), row("40.300")]) + "\r\n", "coverage"),
            "repeat": ("\r\n".join([row("40.000"), row("40.000"), row("40.200")]) + "\r\n", "coverage"),
            "power": ("\r\n".join([ROWS[0], row("40.100", power=15), ROWS[2]]) + "\r\n", "power limit"),
            "thermal": ("\r\n".join([ROWS[0], row("40.100", thermal="Active"), ROWS[2]]) + "\r\n", "thermal"),
        }
        for name, (text, reason) in cases.items():
            with self.subTest(name=name):
                telemetry = check_timing(text, START, END, LIMITS)
                self.assertFalse(telemetry["valid"])
                self.assertTrue(any(reason in item for item in telemetry["reasons"]), telemetry["reasons"])

    def test_a_board_power_cap_is_recorded_without_rejecting(self):
        telemetry = check_timing("\r\n".join([ROWS[0], row("40.100", cap="Active"), ROWS[2]]) + "\r\n",
                                 START, END, LIMITS)
        self.assertTrue(telemetry["valid"])
        self.assertEqual(telemetry["software_power_cap_samples"], 1)


class GateTests(unittest.TestCase):
    def test_every_gate_condition(self):
        reading, _ = parse(ROWS[0])
        good = reading[0]
        self.assertTrue(gate_passes(good, LIMITS))
        for change in ({"enforced_power_w": 99.9}, {"temperature_c": 82.5}, {"software_power_cap": True},
                       {"software_thermal_slowdown": True}):
            with self.subTest(change=change):
                self.assertFalse(gate_passes({**good, **change}, LIMITS))
        idle = Limits(minimum_power_w=100, maximum_temperature_c=82, idle_contexts=True)
        self.assertFalse(gate_passes(good, idle))
        self.assertTrue(gate_passes({**good, "utilization_percent": 5}, idle))


class GpuWatchTests(unittest.TestCase):
    """The gate and monitor around one benchmark, with scripted_nvidia_smi.py
    answering nvidia-smi's queries."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def watch(self, scenario, limits=LIMITS, allowed=(), max_wait_seconds=10.0):
        path = self.root / "scenario.json"
        path.write_text(json.dumps(scenario), encoding="utf-8")
        smi = Smi([sys.executable, str(SCRIPTED_SMI), str(path)], self.root)
        return GpuWatch(smi, limits, allowed, max_wait_seconds, interval_seconds=0.01)

    def test_hardware_reports_the_default_power_limit_or_none(self):
        self.assertEqual(self.watch({"readings": [IDLE]}).smi.hardware(),
                         {"name": "Scripted GPU", "driver_version": "595.00", "default_power_limit_w": 70.0})
        unreported = self.watch({"readings": [IDLE], "default_power_limit": "[N/A]"})
        self.assertIsNone(unreported.smi.hardware()["default_power_limit_w"])

    def test_the_gate_opens_after_three_consecutive_passing_readings(self):
        watch = self.watch({"readings": [IDLE, HOT, IDLE, IDLE, IDLE]})
        self.assertEqual([reading["temperature_c"] for reading in watch.gate("gate")], [45, 95, 45, 45, 45])
        self.assertEqual([observation["stage"] for observation in watch.observations], ["gate", "gate"])

    def test_the_gate_times_out_while_readings_fail(self):
        watch = self.watch({"readings": [HOT]}, max_wait_seconds=0.2)
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            for _ in watch.gate("gate"):
                pass

    def test_a_compute_process_outside_the_allowlist_stops_the_leg_before_any_reading(self):
        watch = self.watch({"readings": [IDLE], "compute_apps": [[999, "C:\\Tools\\trainer.exe"]]})
        readings = []
        with self.assertRaisesRegex(RuntimeError, "allowlist"):
            readings.extend(watch.gate("gate"))
        self.assertEqual(readings, [])
        self.assertEqual(watch.observations[0]["processes"],
                         [{"process_id": 999, "name": "trainer.exe", "explicitly_allowed": False}])

    def test_an_allowed_idle_context_passes_only_while_the_gpu_is_idle(self):
        busy = {**IDLE, "utilization_percent": 40}
        watch = self.watch({"readings": [busy, IDLE, IDLE, IDLE], "compute_apps": [[999, "/usr/bin/idle"]]},
                           limits=Limits(minimum_power_w=100, maximum_temperature_c=82, idle_contexts=True),
                           allowed=(999,))
        self.assertEqual([reading["utilization_percent"] for reading in watch.gate("gate")], [40, 3, 3, 3])

    def test_the_monitor_covers_a_timed_window(self):
        watch = self.watch({"readings": [IDLE]})
        monitor = watch.start_monitor(self.root, "leg-telemetry")
        # A benchmark's timing starts after its own start-up, never with the monitor.
        time.sleep(0.3)
        start = time.time() * 1000
        time.sleep(0.5)
        end = time.time() * 1000
        telemetry = check_timing(watch.stop_monitor(monitor), start, end, LIMITS)
        self.assertTrue(telemetry["valid"], telemetry["reasons"])
        self.assertTrue((self.root / "leg-telemetry.stdout.log").is_file())

    def test_a_monitor_that_ends_at_once_stops_the_leg(self):
        watch = self.watch({"readings": [IDLE], "monitor": "exit", "monitor_delay": 0.4})
        with self.assertRaisesRegex(RuntimeError, "exited before the benchmark"):
            watch.start_monitor(self.root, "leg-telemetry")


if __name__ == "__main__":
    unittest.main(verbosity=2)
