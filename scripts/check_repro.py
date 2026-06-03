#!/usr/bin/env python3
"""Quick repro status snapshot."""
import subprocess, time
from pathlib import Path

LOGS = Path("logs")

# Count today's completed evaluate stages (PIT histogram = end of eval)
completed = 0
active = []
for f in sorted(LOGS.glob("gpu*_evaluate@*.log")):
    age = time.time() - f.stat().st_mtime
    text = f.read_text() if f.stat().st_size < 500000 else ""
    if "Generating PIT histogram" in text:
        if age < 3600:
            completed += 1
    elif age < 120:
        active.append(f.name)

# Check running DVC processes
result = subprocess.run(
    "ps aux | grep 'dvc repro' | grep -v grep",
    shell=True, capture_output=True, text=True, timeout=10
)
n_dvc = len([l for l in result.stdout.strip().split("\n") if l])

print(f"Completed today: {completed}/48")
print(f"Active DVC: {n_dvc}")
if active:
    for a in active[:4]:
        print(f"  Running: {a}")
print(f"Parent PID: {Path('/tmp/dvc_repro_pid').read_text().strip() if Path('/tmp/dvc_repro_pid').exists() else 'N/A'}")
