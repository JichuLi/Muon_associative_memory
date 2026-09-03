"""Read-only laptop monitoring and cooperative throttling; no clock/power changes."""
from __future__ import annotations
import ctypes
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended", ctypes.c_ulonglong)]


class PowerStatus(ctypes.Structure):
    _fields_ = [("ac", ctypes.c_ubyte), ("flags", ctypes.c_ubyte),
                ("percent", ctypes.c_ubyte), ("reserved", ctypes.c_ubyte),
                ("life", ctypes.c_ulong), ("full_life", ctypes.c_ulong)]


def machine_status(path: Path, gpu=True):
    m = MemoryStatus()
    m.length = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    power = PowerStatus()
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power))
    r = dict(time=time.time(), pid=os.getpid(),
             ram_total_gb=m.total_phys / 2**30, ram_free_gb=m.avail_phys / 2**30,
             commit_available_gb=m.avail_page / 2**30,
             disk_free_gb=shutil.disk_usage(path).free / 2**30,
             ac_connected=power.ac == 1, battery_percent=int(power.percent))
    if gpu:
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                timeout=8, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            vals = proc.stdout.strip().splitlines()[0].split(",")
            r.update(dict(zip(["gpu_temp_c", "gpu_util_percent", "gpu_memory_mb", "gpu_power_w"],
                              [float(v.strip()) for v in vals])))
        except Exception as exc:
            r["gpu_query_error"] = str(exc)[:160]
    return r


def process_alive(pid):
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    code = ctypes.c_ulong()
    ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
    kernel.CloseHandle(handle)
    return bool(ok and code.value == 259)


class BudgetExhausted(RuntimeError):
    pass


def atomic_json(path, obj, best_effort=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(temp, path)
            return True
        except PermissionError:
            if attempt==19:
                if best_effort:
                    try:
                        temp.unlink(missing_ok=True)
                    except PermissionError:
                        pass
                    return False
                raise
            time.sleep(min(.05*(attempt+1),.3))


class LaptopGuard:
    def __init__(self, root, config, enabled=True):
        self.root, self.config, self.enabled = Path(root), config, enabled
        self.last_work = time.monotonic()
        self.last_query = 0.0
        self.status = {}
        self.sleep_seconds = 0.0
        self.peak_temp = 0.0
        self.minimum_ram_gb = float("inf")

    def wait(self, seconds):
        # Small sleep chunks keep control files and status observable.
        until = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < until:
            if time.time()>=self.config.get("deadline_unix",float("inf")):
                raise BudgetExhausted("The user-specified experiment time budget is exhausted")
            duration = min(5.0, until - time.monotonic())
            time.sleep(duration)
            self.sleep_seconds += duration

    def checkpoint(self, force=False):
        if time.time()>=self.config.get("deadline_unix",float("inf")):
            raise BudgetExhausted("The user-specified experiment time budget is exhausted")
        if not self.enabled:
            self.last_work = time.monotonic()
            return
        if (self.root / "STOP_AFTER_RUN").exists():
            # The runner handles this after saving the current candidate.
            pass
        now = time.monotonic()
        worked = now - self.last_work
        if force or now - self.last_query >= self.config["monitor_interval_seconds"]:
            self.status = machine_status(self.root)
            self.last_query = time.monotonic()
            self.peak_temp = max(self.peak_temp, self.status.get("gpu_temp_c", 0))
            self.minimum_ram_gb = min(self.minimum_ram_gb, self.status["ram_free_gb"])
            path = self.root / "results" / "telemetry.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(self.status) + "\n")
        duty = self.config["gpu_work_duty"]
        if not self.status.get("ac_connected", True):
            duty = min(duty, self.config["battery_work_duty"])
        self.wait(worked * (1 / duty - 1))
        temperature = self.status.get("gpu_temp_c", 0)
        if temperature >= self.config["soft_temperature_c"]:
            self.wait(min(20, 2 * (temperature - self.config["soft_temperature_c"] + 1)))
            # A cooling pause invalidates the old hot reading; do not repeatedly
            # throttle against a stale temperature while the machine is already cool.
            self.status = machine_status(self.root)
            self.last_query = time.monotonic()
        while True:
            low_battery = (not self.status.get("ac_connected", True)
                           and self.status.get("battery_percent", 255) <= self.config["battery_pause_percent"])
            danger = (self.status.get("gpu_temp_c", 0) >= self.config["hard_temperature_c"]
                      or self.status.get("commit_available_gb", 100) < self.config["min_free_commit_gb"]
                      or self.status.get("disk_free_gb", 100) < self.config["min_free_disk_gb"]
                      or low_battery or (self.root / "PAUSE").exists())
            if not danger:
                break
            atomic_json(self.root / "results" / "resource_pause.json", self.status)
            self.wait(10)
            self.status = machine_status(self.root)
            self.last_query = time.monotonic()
        self.last_work = time.monotonic()


if __name__ == "__main__":
    print(json.dumps(machine_status(Path(__file__).resolve().parent), indent=2))
