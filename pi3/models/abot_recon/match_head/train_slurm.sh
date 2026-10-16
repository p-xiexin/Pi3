#!/bin/bash
#SBATCH --job-name=abot_match_coarse
#SBATCH --nodes=1
#SBATCH --exclude=node104,node112,node110,node116
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/abot_match_%j.out

set -e

checkpoint="${1:?ABotRecon checkpoint is required}"

source /starmap/nas/miniconda3/etc/profile.d/conda.sh
conda activate pi3Train_dhyao

cd /starmap/nas/workspace/pxx/Pi3

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

exp_name="abot_match_coarse"
run_id="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"

accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --module pi3.models.abot_recon.match_head.train \
    --config-name coarse \
    "hydra.run.dir=outputs/${exp_name}/${run_id}" \
    "glob3r.backbone_checkpoint=${checkpoint}"
