#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1

#SBATCH --output=outputs/R-%x.%j.out
#SBATCH --error=outputs/R-%x.%j.err

module load CUDA

./bin/main

