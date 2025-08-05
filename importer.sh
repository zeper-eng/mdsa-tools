#!/bin/bash
#SBATCH --job-name="importcpptraj"
#SBATCH --output=batch_outfiles/out_import
#SBATCH --error=batch_outfiles/err_import
#SBATCH -N 1
#SBATCH --partition=mwgpu
#SBATCH --mem=80G

source /zfshomes/lperez/BIN/miniconda3/etc/profile.d/conda.sh
conda activate mdproj 

python /zfshomes/lperez/summer2025/workspace/importer.py
