#!/bin/bash
#SBATCH --job-name=abot_recon_s1
#SBATCH --nodes=1
#SBATCH --exclude=node104,node112,node110,node116
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/abot_recon_%j.out

set -e

source /starmap/nas/miniconda3/etc/profile.d/conda.sh
conda activate pi3Train_dhyao

cd /starmap/nas/workspace/pxx/Pi3

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

exp_name="abot_recon2"

# accelerate launch \
#     --multi_gpu \
#     --num_processes 8 \
#     --module pi3.models.abot_recon.train \
#     --config-name stage1 \
#     "name=${exp_name}_stage1" \
#     "model.load_pi3=ckpts/Pi3/model.safetensors"
    
accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --module pi3.models.abot_recon.train \
    --config-name stage2 \
    "name=${exp_name}_stage2" \
    "model.ckpt=outputs/${exp_name}_stage1/ckpts/best_model/pytorch_model.bin"