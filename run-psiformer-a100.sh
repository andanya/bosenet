#!/bin/bash
#SBATCH --job-name=andanya-psiformer
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --output=./andanya-psiformer-jobs/andanya-psiformer-job_%j.out
#SBATCH --error=./andanya-psiformer-jobs/andanya-psiformer-job_%j.err


cd /home/da753/bosenet

# Run the command passed as argument
$@
