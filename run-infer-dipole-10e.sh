#!/bin/bash
#SBATCH --job-name=infer-dipole-10e
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=2:00:00
#SBATCH --mem=40G
#SBATCH --array=0-7
#SBATCH --output=./infer-dipole-10e-rep40-pbc/slurm_%A_%a.out
#SBATCH --error=./infer-dipole-10e-rep40-pbc/slurm_%A_%a.err
#SBATCH --exclude=r818u03n09,r818u09n09

module load CUDA
module load cuDNN

cd /home/da753/bosenet

LAMBDAS=(0.1 0.25 0.5 1.0 2.5 5.0 7.5 10.0)
LAM=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: lambda = $LAM"

python infer-dipole-10e-pbc.py $LAM
