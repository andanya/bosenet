#!/bin/bash
#SBATCH --job-name=andanya-psiformer
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --output=./andanya-psiformer-jobs/andanya-psiformer-job_%j.out
#SBATCH --error=./andanya-psiformer-jobs/andanya-psiformer-job_%j.err

module load CUDA
module load cuDNN

cd /home/da753/bosenet

while true; do
  echo "=== GPU MEM $(date '+%H:%M:%S') ==="
  nvidia-smi --query-gpu=memory.total,memory.used,memory.free \
             --format=csv,noheader,nounits | \
    awk -F', ' '{printf "Total: %s MiB | Used: %s MiB | Free: %s MiB\n", $1, $2, $3}'
  sleep 300
done &
MONITOR_PID=$!

nvidia-smi | head -3
echo "GPU type: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"

"$@"
EXIT_CODE=$?

kill $MONITOR_PID 2>/dev/null
exit $EXIT_CODE