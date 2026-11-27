"""nvidia-smi's CSV answers from a JSON scenario, for GPU-supervision tests without a GPU:
"readings" in order, the last repeated; optional "compute_apps", "default_power_limit", "monitor"."""

from datetime import datetime
import json
from pathlib import Path
import sys
import time

FIXED = {"index": "0", "name": "Scripted GPU", "driver_version": "595.00", "clocks.sm": "2100",
         "clocks.mem": "9000"}
READING_FIELDS = {"enforced.power.limit": "enforced_power_w", "temperature.gpu": "temperature_c",
                  "utilization.gpu": "utilization_percent",
                  "clocks_event_reasons.sw_power_cap": "software_power_cap",
                  "clocks_event_reasons.sw_thermal_slowdown": "software_thermal_slowdown"}


def value(field, reading, scenario):
    if field == "timestamp":
        return datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
    if field == "power.default_limit":
        return scenario.get("default_power_limit", "70.00")
    if field in FIXED:
        return FIXED[field]
    item = reading[READING_FIELDS[field]]
    if isinstance(item, bool):
        return "Active" if item else "Not Active"
    return f"{item:.2f}" if isinstance(item, float) else str(item)


def next_reading(scenario_path, readings):
    """The reading for this query; a counter beside the scenario keeps the order across processes."""
    counter = scenario_path.with_name(scenario_path.name + ".count")
    index = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(index + 1))
    return readings[min(index, len(readings) - 1)]


def main():
    scenario_path = Path(sys.argv[1])
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    arguments = sys.argv[2:]
    query = next(item for item in arguments if item.startswith("--query-"))
    if query.startswith("--query-compute-apps="):
        for pid, executable in scenario.get("compute_apps", []):
            print(f"{pid}, {executable}")
        return
    fields = query.removeprefix("--query-gpu=").split(",")
    if "-lms" in arguments:
        time.sleep(scenario.get("monitor_delay", 0))
        if scenario.get("monitor") == "exit":
            return
        interval = int(arguments[arguments.index("-lms") + 1]) / 1000
        while True:
            print(", ".join(value(field, scenario["readings"][-1], scenario) for field in fields), flush=True)
            time.sleep(interval)
    reading = next_reading(scenario_path, scenario["readings"]) if "timestamp" in fields else None
    print(", ".join(value(field, reading, scenario) for field in fields))


if __name__ == "__main__":
    main()
