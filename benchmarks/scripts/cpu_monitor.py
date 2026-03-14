#!/usr/bin/env python3
"""
Monitor CPU utilisation of the meld process during a live benchmark.

Usage:
    python3 bench/cpu_monitor.py --pid <PID>          # monitor a running process
    python3 bench/cpu_monitor.py --port 8090           # find meld by listen port
    python3 bench/cpu_monitor.py --name meld           # find meld by process name

Samples CPU% every 0.5s via `ps`, prints a live ticker, and at the end
prints a summary with min/avg/max/p50/p95 plus an ASCII sparkline chart.

CPU% is reported as a fraction of total machine capacity (800% = 8 cores
fully saturated on an 8-core machine).
"""

import argparse
import subprocess
import sys
import signal
import time


def find_pid_by_port(port: int) -> int | None:
    """Find a PID listening on the given TCP port (macOS lsof)."""
    try:
        out = subprocess.check_output(
            ["lsof", "-iTCP:%d" % port, "-sTCP:LISTEN", "-t"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            return int(out.splitlines()[0])
    except (subprocess.CalledProcessError, ValueError):
        pass
    return None


def find_pid_by_name(name: str) -> int | None:
    """Find a PID by process name (first match via pgrep)."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-x", name],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            return int(out.splitlines()[0])
    except (subprocess.CalledProcessError, ValueError):
        pass
    return None


def sample_cpu(pid: int) -> float | None:
    """Return the %CPU for a given PID, or None if the process is gone."""
    try:
        out = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "%cpu="],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            return float(out)
    except (subprocess.CalledProcessError, ValueError):
        pass
    return None


def sparkline(
    values: list[float], width: int = 60, max_val: float | None = None
) -> str:
    """Render a list of floats as a single-line ASCII sparkline."""
    if not values:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    if max_val is None:
        max_val = max(values) if max(values) > 0 else 1.0
    # Bucket values into `width` bins by averaging
    n = len(values)
    bin_size = max(1, n // width)
    buckets = []
    for i in range(0, n, bin_size):
        chunk = values[i : i + bin_size]
        buckets.append(sum(chunk) / len(chunk))
        if len(buckets) >= width:
            break
    chars = []
    for v in buckets:
        idx = int(v / max_val * (len(blocks) - 1))
        idx = min(idx, len(blocks) - 1)
        chars.append(blocks[idx])
    return "".join(chars)


def main():
    parser = argparse.ArgumentParser(description="Monitor meld CPU utilisation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pid", type=int, help="PID to monitor")
    group.add_argument("--port", type=int, help="Find meld by TCP listen port")
    group.add_argument("--name", type=str, help="Find meld by process name")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Sample interval in seconds (default: 0.5)",
    )
    args = parser.parse_args()

    # Resolve PID
    pid = args.pid
    if args.port:
        pid = find_pid_by_port(args.port)
        if pid is None:
            print(f"No process found listening on port {args.port}", file=sys.stderr)
            sys.exit(1)
        print(f"Found meld PID {pid} on port {args.port}")
    elif args.name:
        pid = find_pid_by_name(args.name)
        if pid is None:
            print(f"No process found matching name '{args.name}'", file=sys.stderr)
            sys.exit(1)
        print(f"Found PID {pid} for '{args.name}'")

    ncpu = 1
    try:
        ncpu = int(
            subprocess.check_output(["sysctl", "-n", "hw.ncpu"], text=True).strip()
        )
    except Exception:
        pass
    max_pct = ncpu * 100.0

    print(f"Monitoring PID {pid} ({ncpu} cores, {max_pct:.0f}% max)")
    print(f"Sampling every {args.interval}s — press Ctrl-C to stop\n")

    samples: list[float] = []
    timestamps: list[float] = []
    start = time.monotonic()

    # Handle Ctrl-C gracefully
    stop = False

    def on_signal(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_signal)

    try:
        while not stop:
            cpu = sample_cpu(pid)
            if cpu is None:
                print("\nProcess exited.")
                break
            samples.append(cpu)
            timestamps.append(time.monotonic() - start)
            elapsed = timestamps[-1]
            bar_len = int(cpu / max_pct * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            print(f"\r  {elapsed:6.1f}s  {cpu:6.1f}%  [{bar}]  ", end="", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass

    print("\n")

    if not samples:
        print("No samples collected.")
        return

    # Summary statistics
    samples_sorted = sorted(samples)
    n = len(samples_sorted)
    avg = sum(samples) / n
    p50 = samples_sorted[int(n * 0.50)]
    p95 = samples_sorted[int(min(n * 0.95, n - 1))]
    lo = samples_sorted[0]
    hi = samples_sorted[-1]
    duration = timestamps[-1]

    print(f"Duration:  {duration:.1f}s  ({n} samples @ {args.interval}s)")
    print(f"Cores:     {ncpu}  ({max_pct:.0f}% = fully saturated)")
    print()
    print(f"  min:     {lo:6.1f}%")
    print(f"  avg:     {avg:6.1f}%")
    print(f"  p50:     {p50:6.1f}%")
    print(f"  p95:     {p95:6.1f}%")
    print(f"  max:     {hi:6.1f}%")
    print()
    print(f"  avg cores used:  {avg / 100:.1f} / {ncpu}")
    print()
    print(f"  Timeline (0% .. {hi:.0f}%):")
    print(f"  {sparkline(samples, width=70, max_val=max_pct)}")


if __name__ == "__main__":
    main()
