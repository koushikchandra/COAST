#!/bin/bash
#SBATCH --job-name=mmb_smoke
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=smoke_mmbiomor_%j.log
# Real-data GPU smoke of MMBiomorNet (full variant), POOLED, 2 epochs. Validates
# the train_fold bio-graph install hook + data path before the ablation sweep.
set -u
cd /work/mech-ai-scratch/tirtho/COAST
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONPATH="/work/mech-ai-scratch/tirtho/COAST/STFlow:/work/mech-ai-scratch/tirtho/COAST:${PYTHONPATH:-}"
python baseline_spatial.py --model mmbiomor_full --regime POOLED --seed 1 \
  --feature_encoder uni_v1_official --splits_root cross_organ_splits8 \
  --source_dataroot dataset --embed_dataroot embed_dataroot \
  --save_root results_smoke --val_fraction 0.15 --epochs 2 \
  --dim 512 --heads 8 --depth 4 --k 8 --device 0
echo "===== mmbiomor smoke rc=$? ====="
