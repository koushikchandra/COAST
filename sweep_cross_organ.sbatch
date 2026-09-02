#!/bin/bash
#SBATCH --job-name=coast_xorg
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/xorg_%A_%a.log
#SBATCH --array=0-17%4
# COAST Table 1/4 STFlow reference: film {none,desc,local} x {LOOO,POOLED}
# x seeds{1,2,3} = 18 tasks.  none=STFlow, desc=+global morphology, local=+local.
# NOTE: uses train_cross_organ.py's ACTUAL args (feature_encoder + splits_root
# overridden from their resnet50/cross_organ_splits defaults to UNI/…splits8).

set -u
cd /work/mech-ai-scratch/tirtho/COAST
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONPATH="/work/mech-ai-scratch/tirtho/COAST/STFlow:/work/mech-ai-scratch/tirtho/COAST:${PYTHONPATH:-}"
mkdir -p logs

FILMS=(none desc local)
REGIMES=(LOOO POOLED)
SEEDS=(1 2 3)
i=$SLURM_ARRAY_TASK_ID
s=${SEEDS[$(( i % 3 ))]}
r=${REGIMES[$(( (i / 3) % 2 ))]}
f=${FILMS[$(( i / 6 ))]}

echo "[xorg] task=$i film=$f regime=$r seed=$s node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
python train_cross_organ.py \
  --film "$f" --regime "$r" --seed "$s" \
  --feature_encoder uni_v1_official \
  --splits_root cross_organ_splits8 \
  --source_dataroot dataset \
  --embed_dataroot embed_dataroot \
  --save_root results_cross_organ \
  --epochs 100 \
  --device 0
echo "[xorg] DONE task=$i"
