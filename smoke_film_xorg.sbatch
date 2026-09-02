#!/bin/bash
#SBATCH --job-name=coast_smoke2
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=smoke_film_xorg_%j.log
# 2-epoch POOLED smoke of BOTH remaining entry points (FiLM + cross-organ),
# to validate them before the full arrays. Throwaway -> results_smoke/.

set -u
cd /work/mech-ai-scratch/tirtho/COAST
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONPATH="/work/mech-ai-scratch/tirtho/COAST/STFlow:/work/mech-ai-scratch/tirtho/COAST:${PYTHONPATH:-}"

echo "===== FiLM smoke (stnet, desc, POOLED, 2ep) ====="
python FiLM_baselines/film_baselines.py \
  --model stnet --film desc --regime POOLED --seed 1 \
  --feature_encoder uni_v1_official \
  --splits_root cross_organ_splits8 --source_dataroot dataset --embed_dataroot embed_dataroot \
  --save_root results_smoke --val_fraction 0.15 --epochs 2 --device 0
echo "===== FiLM smoke rc=$? ====="

echo "===== cross-organ smoke (none, POOLED, 2ep) ====="
python train_cross_organ.py \
  --film none --regime POOLED --seed 1 \
  --feature_encoder uni_v1_official \
  --splits_root cross_organ_splits8 --source_dataroot dataset --embed_dataroot embed_dataroot \
  --save_root results_smoke --epochs 2 --device 0
echo "===== cross-organ smoke rc=$? ====="
echo "[smoke2] DONE job=$SLURM_JOB_ID"
