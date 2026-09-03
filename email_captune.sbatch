#!/bin/bash
#SBATCH --job-name=mmb_capmail
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=mmb_capmail_%j.log
set -u
cd /work/mech-ai-scratch/tirtho/COAST
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PYTHONPATH="/work/mech-ai-scratch/tirtho/COAST/STFlow:/work/mech-ai-scratch/tirtho/COAST:${PYTHONPATH:-}"
MAILTO="tirtho@iastate.edu"
REPORT=$(python .cap_report.py 2>&1)
# best schedule line (top POOLED entry)
BEST=$(python - <<'PY'
import json,glob,os,numpy as np
rows={}
for d in glob.glob("results_cap_*"):
    tag=d.replace("results_cap_","")
    for kf in glob.glob(os.path.join(d,"*","results_kfold.json")):
        reg=os.path.basename(os.path.dirname(kf)).split("_")[0]
        rows.setdefault((tag,reg),[]).append(json.load(open(kf))["pearson_mean"])
for reg in ["POOLED","LOOO"]:
    xs=sorted(((np.mean(v),t) for (t,r),v in rows.items() if r==reg),reverse=True)
    if xs: print(f"{reg}: BEST = {xs[0][1]} ({xs[0][0]:.4f})")
PY
)
BODY="MMbioMOR capacity-funnel tuning (mmbiomor_full, 5 schedules).

${BEST}

Full ranking (Δ vs best baseline):
${REPORT}

-- nova job ${SLURM_JOB_ID}"
{ echo "Subject: [MMbioMOR] best capacity schedule"; echo "To: $MAILTO";
  echo "Content-Type: text/plain"; echo; echo "$BODY"; } | sendmail -t \
  && echo "[capmail] sent to $MAILTO" || echo "[capmail] failed"
