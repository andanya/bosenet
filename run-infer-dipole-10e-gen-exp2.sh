#!/bin/bash
# Inference for dipole-10e-pbc-gen-exp2-conservative-lr-wrapping model.
#
# Before submitting, run:
#   rm -rf ./infer-dipole-10e-pbc-gen-exp2
#   mkdir -p ./infer-dipole-10e-pbc-gen-exp2
#   sbatch run-infer-dipole-10e-gen-exp2.sh

#SBATCH --job-name=infer-gen-exp2
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=2:00:00
#SBATCH --mem=40G
#SBATCH --array=0-5
#SBATCH --output=./infer-dipole-10e-pbc-gen-exp2/slurm_%A_%a.out
#SBATCH --error=./infer-dipole-10e-pbc-gen-exp2/slurm_%A_%a.err
#SBATCH --exclude=r818u03n09,r818u09n09
#SBATCH --no-requeue

module load CUDA
module load cuDNN

cd /home/da753/bosenet

LAMBDAS=(0.1 0.5 1.0 2.5 5.0 10.0)
LAM=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: lambda = $LAM"

python infer-dipole-10e-pbc-gen-exp2.py $LAM
