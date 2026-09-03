#!/bin/bash
#SBATCH --job-name=mmb_cap
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cap_%A_%a.log
#SBATCH --array=0-29%6
# Capacity-funnel tuning on mmbiomor_full: 5 schedules x {LOOO,POOLED} x seed{1,2,3}.
# Each schedule -> its own results dir (results_cap_<tag>) so no save_dir collision.
set -u
cd /work/mech-ai-scratch/tirtho/COAST
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONPATH="/work/mech-ai-scratch/tirtho/COAST/STFlow:/work/mech-ai-scratch/tirtho/COAST:${PYTHONPATH:-}"
mkdir -p logs
TAGS=(cap050 cap075 cap090 cap100 cap_agg)
CAPS=("1,0.75,0.5,0.5" "1,0.75,0.75,0.75" "1,0.9,0.9,0.9" "1,1,1,1" "1,0.66,0.33,0.33")
REGIMES=(LOOO POOLED); SEEDS=(1 2 3)
i=$SLURM_ARRAY_TASK_ID
s=${SEEDS[$(( i % 3 ))]}; r=${REGIMES[$(( (i/3) % 2 ))]}; j=$(( i / 6 ))
export MMB_CAPACITY="${CAPS[$j]}"
echo "[cap] task=$i sched=${TAGS[$j]} cap=$MMB_CAPACITY regime=$r seed=$s"
python baseline_spatial.py --model mmbiomor_full --regime "$r" --seed "$s" \
  --feature_encoder uni_v1_official --splits_root cross_organ_splits8 \
  --source_dataroot dataset --embed_dataroot embed_dataroot \
  --save_root "results_cap_${TAGS[$j]}" --val_fraction 0.15 --epochs 100 \
  --lr 1e-4 --dim 512 --heads 8 --depth 4 --k 8 --device 0
echo "[cap] DONE task=$i"
