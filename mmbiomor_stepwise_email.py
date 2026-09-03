#!/usr/bin/env python
"""Email mmbiomor ablation results per SETUP (variant) as each completes.

A variant is "done" when it has both regimes x 3 seeds = 6 results_kfold.json.
Emails that variant's numbers (+ deltas vs baselines) the moment it completes,
then a final summary once the whole array is finished. Polls every 2 min.
"""
import json, glob, os, subprocess, time
import numpy as np
from collections import defaultdict

ROOT = "results_baselines"
MAILTO = "tirtho@iastate.edu"
ARRAY = os.environ.get("ARRAY_ID", "")
VARIANTS = ["mmbiomor_full", "mmbiomor_nomod", "mmbiomor_nograph", "mmbiomor_noprior",
            "mmbiomor_nobio", "mmbiomor_fixed", "mmbiomor_norec"]
BASE = ["stnet", "deepspace", "mlpprobe", "histogene", "hist2st", "genedml", "triplex", "bleep"]


def collect():
    agg = defaultdict(list)
    for kf in glob.glob(os.path.join(ROOT, "*", "results_kfold.json")):
        n = os.path.basename(os.path.dirname(kf))
        try:
            head, _ = n.split("_seed"); reg, mod = head.split("_", 1)
            agg[(mod, reg)].append(json.load(open(kf))["pearson_mean"])
        except Exception:
            pass
    return agg


def best_base(agg, reg):
    xs = [(np.mean(agg[(b, reg)]), b) for b in BASE if (b, reg) in agg]
    return max(xs) if xs else (float("nan"), "-")


def send(subject, body):
    msg = f"Subject: {subject}\nTo: {MAILTO}\nContent-Type: text/plain\n\n{body}\n"
    try:
        subprocess.run(["sendmail", "-t"], input=msg.encode(), check=True)
        print(f"[email] sent: {subject}")
    except Exception as e:
        print(f"[email] FAILED ({e}): {subject}\n{body}")


def variant_block(agg, v):
    lines = []
    for reg in ["POOLED", "LOOO"]:
        f = agg.get((v, reg))
        if not f:
            lines.append(f"  {reg:7} (pending)"); continue
        bm, bn = best_base(agg, reg)
        extra = f"  Δ vs best baseline ({bn} {bm:.4f}) = {np.mean(f)-bm:+.4f}"
        full = agg.get(("mmbiomor_full", reg))
        if v != "mmbiomor_full" and full:
            extra += f"   Δ vs full = {np.mean(f)-np.mean(full):+.4f}"
        lines.append(f"  {reg:7} {np.mean(f):.4f} +/- {np.std(f):.4f} (n={len(f)}){extra}")
    return "\n".join(lines)


def array_active():
    if not ARRAY:
        return False
    try:
        out = subprocess.run(["squeue", "-h", "-j", ARRAY, "-o", "%T"],
                             capture_output=True, text=True).stdout.strip()
        return bool(out)
    except Exception:
        return False


def main():
    emailed = set()
    for _ in range(600):  # ~20 h safety cap
        agg = collect()
        for v in VARIANTS:
            if v in emailed:
                continue
            done = len(agg.get((v, "POOLED"), [])) >= 3 and len(agg.get((v, "LOOO"), [])) >= 3
            if done:
                send(f"[MMbioMOR] setup done: {v}",
                     f"Ablation setup '{v}' complete (LOOO+POOLED x 3 seeds).\n\n"
                     f"{variant_block(agg, v)}\n")
                emailed.add(v)
        if not array_active() and all(v in emailed for v in VARIANTS if agg.get((v, "POOLED")) or agg.get((v, "LOOO"))):
            # final summary over everything present
            body = ["MMbioMOR ablation — FINAL summary (pooled mean Pearson):\n"]
            for reg in ["POOLED", "LOOO"]:
                body.append(f"[{reg}]")
                for m in BASE + VARIANTS:
                    x = agg.get((m, reg))
                    if x:
                        body.append(f"  {m:20} {np.mean(x):.4f} +/- {np.std(x):.4f} (n={len(x)})")
                body.append("")
            send("[MMbioMOR] ablation FINAL summary", "\n".join(body))
            break
        time.sleep(120)


if __name__ == "__main__":
    main()
