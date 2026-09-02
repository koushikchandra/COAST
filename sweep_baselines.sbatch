#!/bin/bash
#SBATCH --job-name=coast_base
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/base_%A_%a.log
#SBATCH --array=0-35%6
# COAST Table 1/2 baselines: 6 backbones x {LOOO,POOLED} x seeds{1,2,3} = 36 tasks.
# Each task runs ALL folds for its (model,regime); fold-level idempotent (skips
# existing fold_*_results.json), so the array is safe to requeue.

set -u
cd /work/mech-ai-scratch/tirtho/COAST
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONPATH="/work/mech-ai-scratch/tirtho/COAST/STFlow:/work/mech-ai-scratch/tirtho/COAST:${PYTHONPATH:-}"
mkdir -p logs

MODELS=(stnet deepspace mlpprobe histogene hist2st genedml)
REGIMES=(LOOO POOLED)
SEEDS=(1 2 3)
i=$SLURM_ARRAY_TASK_ID
s=${SEEDS[$(( i % 3 ))]}
r=${REGIMES[$(( (i / 3) % 2 ))]}
m=${MODELS[$(( i / 6 ))]}

echo "[base] task=$i model=$m regime=$r seed=$s node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
python baseline_spatial.py \
  --model "$m" --regime "$r" --seed "$s" \
  --feature_encoder uni_v1_official \
  --splits_root cross_organ_splits8 \
  --source_dataroot dataset \
  --embed_dataroot embed_dataroot \
  --save_root results_baselines \
  --val_fraction 0.15 \
  --epochs 100 \
  --lr 1e-4 --dim 512 --heads 8 --depth 4 --k 8 \
  --device 0
echo "[base] DONE task=$i"
