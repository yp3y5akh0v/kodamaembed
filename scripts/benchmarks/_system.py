"""Facts about this machine for benchmark evidence: processors, operating system
and the settings that change processor speed, recorded and never changed."""

from __future__ import annotations

import os
from pathlib import Path
import platform
import subprocess

from _process import PLATFORM


def _run(*command):
    return subprocess.run(command, capture_output=True, text=True, timeout=10, check=True).stdout.strip()


def _unique(values):
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def processor_names():
    """Each distinct processor model name."""
    if PLATFORM == "windows":
        import winreg

        names = []
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor") as root:
            for index in range(winreg.QueryInfoKey(root)[0]):
                with winreg.OpenKey(root, winreg.EnumKey(root, index)) as key:
                    names.append(winreg.QueryValueEx(key, "ProcessorNameString")[0])
        return _unique(names)
    if PLATFORM == "macos":
        return _unique([_run("sysctl", "-n", "machdep.cpu.brand_string")])
    lines = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines()
    for field in ("model name", "Model", "Hardware"):
        names = [line.partition(":")[2] for line in lines if line.partition(":")[0].strip() == field]
        if names:
            return _unique(names)
    return [platform.processor() or platform.machine()]


def operating_system():
    return platform.platform()


def _sysfs(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def speed_settings():
    """What can make processor speed vary between runs; None where this system has no such setting."""
    if PLATFORM == "linux":
        governors = _unique(_sysfs(path) or "" for path in
                            Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor"))
        no_turbo = _sysfs("/sys/devices/system/cpu/intel_pstate/no_turbo")
        boost = _sysfs("/sys/devices/system/cpu/cpufreq/boost")
        return {"frequency_governors": governors or None,
                "turbo": None if no_turbo is None and boost is None else (no_turbo == "0" if no_turbo is not None
                                                                          else boost == "1"),
                "isolated_cpus": _sysfs("/sys/devices/system/cpu/isolated"),
                "nohz_full_cpus": _sysfs("/sys/devices/system/cpu/nohz_full")}
    if PLATFORM == "windows":
        return {"power_scheme": _run("powercfg", "/getactivescheme").partition(":")[2].strip()}
    settings = _run("pmset", "-g").splitlines()
    low_power = [line.split()[-1] for line in settings if line.split()[:1] == ["lowpowermode"]]
    return {"low_power_mode": low_power[0] == "1" if low_power else None}


def load_average():
    """The one-minute load average, None where the system reports none."""
    return os.getloadavg()[0] if hasattr(os, "getloadavg") else None
