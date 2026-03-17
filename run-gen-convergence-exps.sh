#!/bin/bash
# Launch all three convergence experiments as separate SLURM jobs.
#
# Exp 1: Log-scale lambda input
# Exp 2: Conservative optimizer (lower LR, higher damping)
# Exp 3: FiLM conditioning + log-scale
#
# Usage: bash run-gen-convergence-exps.sh

set -e

mkdir -p gen-convergence-jobs

for exp in 1 2 3; do
  case $exp in
    1) SCRIPT="dipole-10e-pbc-gen-exp1-logscale.py"
       NAME="gen-exp1-logscale" ;;
    2) SCRIPT="dipole-10e-pbc-gen-exp2-conservative-lr.py"
       NAME="gen-exp2-conservative-lr" ;;
    3) SCRIPT="dipole-10e-pbc-gen-exp3-film.py"
       NAME="gen-exp3-film" ;;
  esac

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${NAME}
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --output=./gen-convergence-jobs/${NAME}_%j.out
#SBATCH --error=./gen-convergence-jobs/${NAME}_%j.err
#SBATCH --exclude=r818u03n09,r818u09n09

module load CUDA
module load cuDNN

cd /home/da753/bosenet

while true; do
  echo "=== GPU MEM \$(date '+%H:%M:%S') ==="
  nvidia-smi --query-gpu=memory.total,memory.used,memory.free \
             --format=csv,noheader,nounits | \
    awk -F', ' '{printf "Total: %s MiB | Used: %s MiB | Free: %s MiB\n", \$1, \$2, \$3}'
  sleep 300
done &
MONITOR_PID=\$!

nvidia-smi | head -3
echo "GPU type: \$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"

python ${SCRIPT}
EXIT_CODE=\$?

kill \$MONITOR_PID 2>/dev/null
exit \$EXIT_CODE
EOF

  echo "Submitted experiment ${exp}: ${NAME}"
done
