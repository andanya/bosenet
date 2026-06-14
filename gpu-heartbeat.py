"""Periodic small GPU op to keep nvidia-smi-reported utilization > 0%.

YCRC's GPU watchdog auto-cancels jobs that show 0% GPU utilization for one
hour. Long XLA gemm-fusion autotune phases (>1h on busy nodes) trip this
even though the job is doing useful CPU-side compile work — see jobs
10301620 / 10301846 (Apr 30 – May 1 2026, 63/65 inference tasks killed).

For training jobs we want to keep autotune ON (better steady-state kernels),
so we run this heartbeat as a sibling process during the autotune window.
Coexists with the main JAX process by disabling preallocation and capping
its own memory fraction.

Usage from a SLURM script (after conda activate / module load):
    python /home/da753/bosenet/gpu-heartbeat.py 2>/dev/null &
    HEARTBEAT_PID=$!
    ...
    kill $HEARTBEAT_PID 2>/dev/null
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.02"

import time
import jax
import jax.numpy as jnp

x = jax.device_put(jnp.ones((1024, 1024), dtype=jnp.float32))
matmul = jax.jit(lambda a: a @ a)
matmul(x).block_until_ready()

while True:
    # Burn ~2 s of GPU time so any sub-minute monitoring sample sees nonzero
    # util; then idle 60 s. Average util ~3%, negligible relative to the
    # main job but always positive.
    t0 = time.time()
    while time.time() - t0 < 2.0:
        matmul(x).block_until_ready()
    time.sleep(60)
