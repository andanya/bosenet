#!/bin/bash
#SBATCH --job-name=andanya-psiformer
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --output=./andanya-psiformer-jobs/andanya-psiformer-job_%j.out
#SBATCH --error=./andanya-psiformer-jobs/andanya-psiformer-job_%j.err
#SBATCH --exclude=r818u03n09

module load CUDA
module load cuDNN

cd /home/da753/bosenet

# Run the command passed as argument
$@
