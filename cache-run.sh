#!/bin/bash
#SBATCH --partition=edu-medium
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1

#SBATCH --output=outputs/R-%x.%j.out
#SBATCH --error=outputs/R-%x.%j.err

module load CUDA

ncu --metrics l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct ./bin/main
