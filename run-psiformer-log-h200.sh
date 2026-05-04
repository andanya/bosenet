#!/bin/bash
#SBATCH --job-name=andanya-psiformer
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --time=24:00:00
#SBATCH --mem=140G
#SBATCH --output=./andanya-psiformer-jobs/andanya-psiformer-job_%j.out
#SBATCH --error=./andanya-psiformer-jobs/andanya-psiformer-job_%j.err
#SBATCH --no-requeue

cd /home/da753/bosenet

source ~/.bashrc
conda activate ferminet-jax

while true; do
  echo "=== GPU MEM $(date '+%H:%M:%S') ==="
  nvidia-smi --query-gpu=memory.total,memory.used,memory.free \
             --format=csv,noheader,nounits | \
    awk -F', ' '{printf "Total: %s MiB | Used: %s MiB | Free: %s MiB\n", $1, $2, $3}'
  sleep 300
done &
MONITOR_PID=$!

# Background GPU heartbeat — keeps nvidia-smi util > 0% during XLA's slow
# autotune phase so the YCRC watchdog doesn't kill us mid-autotune
# (cf. jobs 10301620/10301846, Apr 30 – May 1 2026).
python /home/da753/bosenet/gpu-heartbeat.py 2>/dev/null &
HEARTBEAT_PID=$!

nvidia-smi | head -3
echo "GPU type: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"

"$@"
EXIT_CODE=$?

kill $MONITOR_PID 2>/dev/null
kill $HEARTBEAT_PID 2>/dev/null
exit $EXIT_CODE