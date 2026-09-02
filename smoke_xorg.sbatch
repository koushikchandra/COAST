#!/bin/bash
#SBATCH --job-name=coast_xsmoke
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=smoke_xorg_%j.log
# Re-smoke cross-organ after eigh (magma + CPU-fallback) fix.

set -u
cd /work/mech-ai-scratch/tirtho/COAST
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONPATH="/work/mech-ai-scratch/tirtho/COAST/STFlow:/work/mech-ai-scratch/tirtho/COAST:${PYTHONPATH:-}"

python train_cross_organ.py \
  --film none --regime POOLED --seed 1 \
  --feature_encoder uni_v1_official \
  --splits_root cross_organ_splits8 --source_dataroot dataset --embed_dataroot embed_dataroot \
  --save_root results_smoke --epochs 2 --device 0
echo "===== cross-organ re-smoke rc=$? ====="
