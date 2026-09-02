#!/bin/bash
#SBATCH --job-name=coast_film
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/film_%A_%a.log
#SBATCH --array=0-89%6
# COAST Table 3 (FiLM conditioning study): 5 backbones x {none,desc,local}
# x {LOOO,POOLED} x seeds{1,2,3} = 90 tasks. film_baselines.py is fold-idempotent.

set -u
cd /work/mech-ai-scratch/tirtho/COAST
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONPATH="/work/mech-ai-scratch/tirtho/COAST/STFlow:/work/mech-ai-scratch/tirtho/COAST:${PYTHONPATH:-}"
mkdir -p logs

MODELS=(stnet deepspace mlpprobe histogene hist2st)
FILMS=(none desc local)
REGIMES=(LOOO POOLED)
SEEDS=(1 2 3)
i=$SLURM_ARRAY_TASK_ID
s=${SEEDS[$(( i % 3 ))]}
r=${REGIMES[$(( (i / 3) % 2 ))]}
f=${FILMS[$(( (i / 6) % 3 ))]}
m=${MODELS[$(( i / 18 ))]}

echo "[film] task=$i model=$m film=$f regime=$r seed=$s node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
python FiLM_baselines/film_baselines.py \
  --model "$m" --film "$f" --regime "$r" --seed "$s" \
  --feature_encoder uni_v1_official \
  --splits_root cross_organ_splits8 \
  --source_dataroot dataset \
  --embed_dataroot embed_dataroot \
  --save_root results_film \
  --val_fraction 0.15 \
  --epochs 100 \
  --device 0
echo "[film] DONE task=$i"
