#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/gpfs/workdir/blampeyq/.jobs_outputs/%j
#SBATCH --mem=512G
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=fusion_shm_big

module purge
module load anaconda3/2022.10/gcc-11.2.0
source activate novae

cd /gpfs/workdir/blampeyq/novae_benchmark

python -u novae_benchmark/pan_tissue_umap.py
