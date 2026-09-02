#!/bin/bash
#SBATCH --job-name=coast_smoke
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=smoke_baseline_%j.log
# Fast end-to-end smoke: one baseline (st-net), POOLED regime, 2 epochs, all folds.
# Validates the MorphoST.evaluation shim + data + UNI features on real multi-slide
# test folds before the full sweep. Writes to results_smoke/ (throwaway).

set -eu
cd /work/mech-ai-scratch/tirtho/COAST
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONPATH="/work/mech-ai-scratch/tirtho/COAST/STFlow:/work/mech-ai-scratch/tirtho/COAST:${PYTHONPATH:-}"

echo "[smoke] node=$(hostname) job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print('[smoke] cuda:', torch.cuda.is_available())"

python baseline_spatial.py \
  --model stnet --regime POOLED --seed 1 \
  --feature_encoder uni_v1_official \
  --splits_root cross_organ_splits8 \
  --source_dataroot dataset \
  --embed_dataroot embed_dataroot \
  --save_root results_smoke \
  --val_fraction 0.15 \
  --epochs 2 \
  --lr 1e-4 --dim 512 --heads 8 --depth 4 --k 8 \
  --device 0
echo "[smoke] DONE job=$SLURM_JOB_ID"
