#!/usr/bin/env python3
"""Warm live benchmark — 10k × 10k, usearch backend, 50k events at c=10.

Preserves the cache. Clears WAL and crossmap for a clean injection.
On first run the cache is empty so the server builds the index (slow).
On subsequent runs the server loads from cache and starts in ~1s.
Run from the project root:
    python3 benchmarks/live/10kx10k_inject50k_usearch/warm/run_test.py
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import threading
import time

import psutil

TEST_DIR = "benchmarks/live/10kx10k_inject50k_usearch/warm"
BINARY_DEFAULT = "./target/release/meld"
PORT = 8090
ITERATIONS = 50000
CONCURRENCY = 10
A_DATA = "benchmarks/data/dataset_a_10k.csv"
B_DATA = "benchmarks/data/dataset_b_10k.csv"
SERVER_READY_TIMEOUT = 120

# Resource monitoring settings
SAMPLE_INTERVAL_S = 1.0


# ---------------------------------------------------------------------------
# Resource monitor
# ---------------------------------------------------------------------------


def read_gpu_utilisation() -> dict[str, int] | None:
    """Read GPU utilisation percentages from IOAccelerator via ioreg (no sudo)."""
    try:
        out = subprocess.check_output(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
            text=True,
            timeout=5,
        )
        result = {}
        for key in (
            "Device Utilization %",
            "Renderer Utilization %",
            "Tiler Utilization %",
        ):
            m = re.search(rf'"{key}"=(\d+)', out)
            if m:
                result[key] = int(m.group(1))
        return result if result else None
    except Exception:
        return None


class ResourceMonitor:
    """Samples CPU and GPU utilisation in a background thread."""

    def __init__(self, pid: int, interval: float = SAMPLE_INTERVAL_S):
        self.interval = interval
        self.pid = pid
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # CPU samples (per-process % of total system CPU)
        self.cpu_samples: list[float] = []
        # System-wide per-core CPU samples
        self.core_samples: list[list[float]] = []
        # GPU samples
        self.gpu_device_samples: list[int] = []
        self.gpu_renderer_samples: list[int] = []
        self.gpu_tiler_samples: list[int] = []
        # Memory samples (RSS in MB)
        self.mem_samples: list[float] = []

    def start(self):
        try:
            self._proc = psutil.Process(self.pid)
            # Prime the cpu_percent call (first call always returns 0)
            self._proc.cpu_percent()
        except psutil.NoSuchProcess:
            print(f"Warning: PID {self.pid} not found, resource monitoring disabled.")
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        while not self._stop.is_set():
            self._stop.wait(self.interval)
            if self._stop.is_set():
                break
            try:
                # Per-process CPU (can exceed 100% on multi-core)
                cpu = self._proc.cpu_percent()
                self.cpu_samples.append(cpu)

                # System-wide per-core CPU
                cores = psutil.cpu_percent(percpu=True)
                self.core_samples.append(cores)

                # Process RSS
                mem_info = self._proc.memory_info()
                self.mem_samples.append(mem_info.rss / (1024 * 1024))

                # GPU
                gpu = read_gpu_utilisation()
                if gpu:
                    self.gpu_device_samples.append(gpu.get("Device Utilization %", 0))
                    self.gpu_renderer_samples.append(
                        gpu.get("Renderer Utilization %", 0)
                    )
                    self.gpu_tiler_samples.append(gpu.get("Tiler Utilization %", 0))
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                break

    def summary(self) -> str:
        lines = []
        W = 80
        lines.append("")
        lines.append("=" * W)
        lines.append("Resource Utilisation Summary (meld server process)")
        lines.append("=" * W)
        lines.append(
            f"  Sample interval: {self.interval}s  |  Samples collected: {len(self.cpu_samples)}"
        )
        lines.append("")

        num_cores = psutil.cpu_count() or 1

        # CPU — per-process
        if self.cpu_samples:
            avg = sum(self.cpu_samples) / len(self.cpu_samples)
            peak = max(self.cpu_samples)
            mn = min(self.cpu_samples)
            lines.append(f"  CPU (meld process, {num_cores} cores available):")
            lines.append(f"    avg: {avg:6.1f}%   min: {mn:6.1f}%   peak: {peak:6.1f}%")
            lines.append(
                f"    (values > 100% means multiple cores in use; "
                f"max possible = {num_cores * 100}%)"
            )

        # CPU — system-wide per-core
        if self.core_samples:
            n_cores = len(self.core_samples[0])
            core_avgs = []
            for c in range(n_cores):
                vals = [s[c] for s in self.core_samples if c < len(s)]
                core_avgs.append(sum(vals) / len(vals) if vals else 0)
            system_avg = sum(core_avgs) / len(core_avgs)
            active_cores = sum(1 for a in core_avgs if a > 10.0)
            busiest = max(core_avgs)
            lines.append(f"\n  CPU (system-wide, {n_cores} cores):")
            lines.append(
                f"    avg per core: {system_avg:5.1f}%   "
                f"busiest core: {busiest:5.1f}%   "
                f"cores > 10%: {active_cores}/{n_cores}"
            )

        # Memory
        if self.mem_samples:
            avg_mem = sum(self.mem_samples) / len(self.mem_samples)
            peak_mem = max(self.mem_samples)
            lines.append(f"\n  Memory (meld RSS):")
            lines.append(f"    avg: {avg_mem:,.0f} MB   peak: {peak_mem:,.0f} MB")

        # GPU
        if self.gpu_device_samples:
            avg_dev = sum(self.gpu_device_samples) / len(self.gpu_device_samples)
            peak_dev = max(self.gpu_device_samples)
            avg_ren = sum(self.gpu_renderer_samples) / len(self.gpu_renderer_samples)
            peak_ren = max(self.gpu_renderer_samples)
            avg_til = sum(self.gpu_tiler_samples) / len(self.gpu_tiler_samples)
            peak_til = max(self.gpu_tiler_samples)
            lines.append(f"\n  GPU (system-wide via IOAccelerator):")
            lines.append(f"    Device:    avg: {avg_dev:5.1f}%   peak: {peak_dev}%")
            lines.append(f"    Renderer:  avg: {avg_ren:5.1f}%   peak: {peak_ren}%")
            lines.append(f"    Tiler:     avg: {avg_til:5.1f}%   peak: {peak_til}%")
        else:
            lines.append("\n  GPU: no samples collected (ioreg unavailable?)")

        lines.append("=" * W)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Server readiness
# ---------------------------------------------------------------------------


def wait_for_server(log_path, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(log_path):
            with open(log_path) as f:
                if "server listening" in f.read():
                    return True
        time.sleep(2)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=BINARY_DEFAULT)
    args = parser.parse_args()

    if not os.path.exists(args.binary):
        print(f"Binary not found: {args.binary}")
        print("Build with: cargo build --release --features usearch")
        sys.exit(1)

    cache_dir = f"{TEST_DIR}/cache"
    if not (os.path.exists(cache_dir) and os.listdir(cache_dir)):
        print("Note: cache is empty — this run will build the index (slow).")
        print("Run again afterwards for a true warm measurement.\n")

    shutil.rmtree(f"{TEST_DIR}/output", ignore_errors=True)
    os.makedirs(f"{TEST_DIR}/output", exist_ok=True)
    for f in glob.glob(f"{TEST_DIR}/wal/*.ndjson"):
        os.remove(f)
    os.makedirs(f"{TEST_DIR}/wal", exist_ok=True)
    crossmap = f"{TEST_DIR}/crossmap.csv"
    if os.path.exists(crossmap):
        os.remove(crossmap)

    log_path = f"/tmp/meld_bench_{os.getpid()}.log"
    print(f"=== Warm live run: {TEST_DIR} ===")
    print("Starting server...")

    server = subprocess.Popen(
        [
            args.binary,
            "serve",
            "--config",
            f"{TEST_DIR}/config.yaml",
            "--port",
            str(PORT),
        ],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
    )

    if not wait_for_server(log_path, SERVER_READY_TIMEOUT):
        print(f"ERROR: Server did not start within {SERVER_READY_TIMEOUT}s.")
        with open(log_path) as f:
            print(f.read())
        server.terminate()
        sys.exit(1)

    print("Server ready. Startup log:")
    with open(log_path) as f:
        for line in f:
            print(f"  {line}", end="")

    # Start resource monitoring on the meld server process
    monitor = ResourceMonitor(server.pid)
    monitor.start()
    print(
        f"\nResource monitoring started (PID {server.pid}, sampling every {SAMPLE_INTERVAL_S}s)"
    )

    print(f"\nInjecting {ITERATIONS} events at concurrency {CONCURRENCY}...\n")
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/scripts/live_concurrent_test.py",
            "--no-serve",
            "--port",
            str(PORT),
            "--iterations",
            str(ITERATIONS),
            "--concurrency",
            str(CONCURRENCY),
            "--a-path",
            A_DATA,
            "--b-path",
            B_DATA,
        ]
    )

    # Stop monitoring and print summary
    monitor.stop()
    print(monitor.summary())

    server.terminate()
    server.wait()
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
