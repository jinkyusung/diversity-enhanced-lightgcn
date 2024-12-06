#!/bin/bash

#SBATCH --job-name=jinkyu
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --output=./slurm/%x.%j.out

ml purge
ml load cuda/12.1
eval "$(conda shell.bash hook)"
conda activate cf-frame

srun python main.py