#!/bin/bash
#SBATCH --job-name=glob3r_train
#SBATCH --nodes=1
#SBATCH --exclude=node104,node112,node110
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j_8gpu.out

# Usage:
#   New two-stage training:
#     sbatch train.sh
#
#   Resume an interrupted coarse stage:
#     sbatch --export=ALL,RUN_DIR=/path/to/existing/run train.sh
#
#   Resume an interrupted refinement stage:
#     sbatch --export=ALL,STAGE=refinement,RUN_DIR=/path/to/existing/run train.sh
#
# Before resubmitting coarse, cancel any old refinement job whose dependency
# can no longer be satisfied.

set -e

source /starmap/nas/miniconda3/etc/profile.d/conda.sh
conda activate pi3Train_dhyao

cd /starmap/nas/workspace/pxx/Pi3

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

STAGE=${STAGE:-coarse}
RUN_DIR=${RUN_DIR:-/starmap/nas/workspace/pxx/Pi3/outputs/glob3r_${SLURM_JOB_ID}}
EXTRA_ARGS=()

if [ "${STAGE}" = "coarse" ]; then
    sbatch \
        --dependency=afterok:${SLURM_JOB_ID} \
        --export=ALL,STAGE=refinement,RUN_DIR=${RUN_DIR} \
        train.sh
else
    EXTRA_ARGS+=(
        glob3r.matching_checkpoint=${RUN_DIR}/coarse/ckpts/best_model/pytorch_model.bin
    )
fi

accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    scripts/train_glob3r.py \
    --config-name glob3r_${STAGE} \
    hydra.run.dir=${RUN_DIR}/${STAGE} \
    hydra.job_logging.handlers.file.filename=${RUN_DIR}/${STAGE}/log.log \
    log.output_dir=${RUN_DIR}/${STAGE}/ \
    log.ckpt_dir=${RUN_DIR}/${STAGE}/ckpts \
    "${EXTRA_ARGS[@]}"
