#!/bin/bash
#SBATCH --job-name=andanya-psiformer
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --output=./andanya-psiformer-jobs/andanya-psiformer-job_%j.out
#SBATCH --error=./andanya-psiformer-jobs/andanya-psiformer-job_%j.err
#SBATCH --exclude=r818u03n09
#SBATCH --no-requeue

module load CUDA
module load cuDNN

cd /home/da753/bosenet

# Skip XLA gemm-fusion autotune; the single-threaded compile takes >1h and
# trips the YCRC GPU watchdog (jobs 10301620/10301846, Apr 30 – May 1 2026).
export XLA_FLAGS="--xla_gpu_autotune_level=0"

# Run the command passed as argument
$@
EXIT=$?

exit $EXIT
