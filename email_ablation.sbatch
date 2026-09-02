#!/bin/bash
#SBATCH --job-name=mmb_email
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=mmb_email_%j.log
# Runs after the ablation array (afterany dep); aggregates + emails the table.
set -u
cd /work/mech-ai-scratch/tirtho/COAST
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONPATH="/work/mech-ai-scratch/tirtho/COAST/STFlow:/work/mech-ai-scratch/tirtho/COAST:${PYTHONPATH:-}"
MAILTO="tirtho@iastate.edu"
TABLE=$(python aggregate_mmbiomor_ablation.py 2>&1)
NDONE=$(find results_baselines -path '*mmbiomor*' -name results_kfold.json 2>/dev/null | wc -l)
BODY="MMbioMOR ablation vs all COAST baselines (pooled mean Pearson, LOOO + POOLED).
mmbiomor variant runs complete: ${NDONE}/42

${TABLE}

-- auto-sent from nova job ${SLURM_JOB_ID}"
if command -v sendmail >/dev/null 2>&1; then
  { echo "Subject: [COAST MMbioMOR] ablation table"; echo "To: $MAILTO";
    echo "Content-Type: text/plain"; echo; echo "$BODY"; } | sendmail -t \
    && echo "[email] sent to $MAILTO" || echo "[email] sendmail failed"
else
  echo "[email] no sendmail"; echo "$BODY"
fi
