#!/bin/bash
#SBATCH --job-name=mmb_ablate
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/mmb_%A_%a.log
#SBATCH --array=0-41%6
# MMBiomorNet ablation: 7 variants x {LOOO,POOLED} x seeds{1,2,3} = 42 tasks.
# Writes into results_baselines/ so it is directly comparable (same pooled metric,
# same folds) to every codebase baseline. Fold-idempotent (skips existing json).
set -u
cd /work/mech-ai-scratch/tirtho/COAST
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONPATH="/work/mech-ai-scratch/tirtho/COAST/STFlow:/work/mech-ai-scratch/tirtho/COAST:${PYTHONPATH:-}"
mkdir -p logs
VARIANTS=(mmbiomor_full mmbiomor_nomod mmbiomor_nograph mmbiomor_noprior mmbiomor_nobio mmbiomor_fixed mmbiomor_norec)
REGIMES=(LOOO POOLED)
SEEDS=(1 2 3)
i=$SLURM_ARRAY_TASK_ID
s=${SEEDS[$(( i % 3 ))]}
r=${REGIMES[$(( (i / 3) % 2 ))]}
m=${VARIANTS[$(( i / 6 ))]}
echo "[mmb] task=$i model=$m regime=$r seed=$s node=$(hostname)"
python baseline_spatial.py --model "$m" --regime "$r" --seed "$s" \
  --feature_encoder uni_v1_official --splits_root cross_organ_splits8 \
  --source_dataroot dataset --embed_dataroot embed_dataroot \
  --save_root results_baselines --val_fraction 0.15 --epochs 100 \
  --lr 1e-4 --dim 512 --heads 8 --depth 4 --k 8 --device 0
echo "[mmb] DONE task=$i"
