#!/usr/bin/env python3
"""Launch dvc repro across both GPUs in background, survive shell exit."""
import subprocess
import sys
import os

script = os.path.join(os.path.dirname(__file__), "dvc_repro_parallel.sh")
log = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "repro_full.log")
os.makedirs(os.path.dirname(log), exist_ok=True)

with open(log, "a") as f:
    p = subprocess.Popen(
        ["bash", script],
        stdout=f, stderr=subprocess.STDOUT,
        preexec_fn=lambda: os.setsid(),  # new process group, survive parent death
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"},
    )

with open("/tmp/dvc_repro_pid", "w") as f:
    f.write(str(p.pid))

print(f"Launched PID {p.pid} (log: {log})")
