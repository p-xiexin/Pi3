#!/bin/bash
#SBATCH --job-name=abot_recon_s1
#SBATCH --nodes=1
#SBATCH --exclude=node104,node112,node110
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/%j_abot_recon_stage1_8gpu.out

set -e

source /starmap/nas/miniconda3/etc/profile.d/conda.sh
conda activate pi3Train_dhyao

cd /starmap/nas/workspace/pxx/Pi3

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --module pi3.models.abot_recon.train
