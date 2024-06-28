#!/bin/bash
#SBATCH --job-name=Novae_benchmark
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --mail-user=hakim.benkirane@centralesupelec.fr
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --mem=256000
#SBATCH --exclude=ruche-gpu01

# Modules load
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/12.2.1/gcc-11.2.0 

source activate novae_benchmark

python -u main.py -c config/benchmark_union_ruche.yml





