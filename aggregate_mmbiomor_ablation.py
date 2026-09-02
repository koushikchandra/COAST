#!/usr/bin/env python
"""Unified comparison: every codebase baseline vs MMBiomorNet ablations.

Reads results_baselines/<REGIME>_<model>_seed<S>/results_kfold.json (pooled
metric, same folds for all models) and prints, per regime, a table of
mean +/- sd over seeds, plus the key ablation deltas:

  * modulation:  full - nomod   (modulated vs unmodulated biological injection)
  * biology:     full - nobio   (multimodal bio vs unimodal histology)
  * bio-graph:   full - nograph
  * prior:       full - noprior
  * recursion:   full - fixed / full - norec
"""
import json, glob, os
import numpy as np
from collections import defaultdict

ROOT = "results_baselines"
BASELINES = ["stnet", "deepspace", "mlpprobe", "histogene", "hist2st", "genedml", "triplex", "bleep"]
MMB = ["mmbiomor_nobio", "mmbiomor_norec", "mmbiomor_fixed", "mmbiomor_nograph",
       "mmbiomor_noprior", "mmbiomor_nomod", "mmbiomor_full"]

def collect():
    agg = defaultdict(list)  # (model, regime) -> [seed pearson_mean, ...]
    for kf in glob.glob(os.path.join(ROOT, "*", "results_kfold.json")):
        name = os.path.basename(os.path.dirname(kf))
        parts = name.split("_seed")
        seed = parts[-1]
        head = parts[0]                       # REGIME_model  (model may contain '_')
        regime, model = head.split("_", 1)
        try:
            agg[(model, regime)].append(json.load(open(kf))["pearson_mean"])
        except Exception:
            pass
    return agg

def main():
    agg = collect()
    for regime in ["LOOO", "POOLED"]:
        print(f"\n===== {regime} — pooled mean Pearson (mean +/- sd over seeds) =====")
        print(f"{'model':22} {'pearson':>16} {'n':>3}")
        for m in BASELINES + MMB:
            v = agg.get((m, regime))
            if not v:
                continue
            tag = "  <- proposed" if m == "mmbiomor_full" else ""
            print(f"{m:22} {np.mean(v):.4f} +/- {np.std(v):.4f} {len(v):>3}{tag}")
        f = agg.get(("mmbiomor_full", regime))
        if f:
            fm = np.mean(f)
            def delta(other):
                o = agg.get((other, regime))
                return f"{fm-np.mean(o):+.4f}" if o else "   n/a"
            print(f"  Δ modulation (full - nomod):   {delta('mmbiomor_nomod')}")
            print(f"  Δ biology    (full - nobio):   {delta('mmbiomor_nobio')}")
            print(f"  Δ bio-graph  (full - nograph): {delta('mmbiomor_nograph')}")
            print(f"  Δ prior      (full - noprior): {delta('mmbiomor_noprior')}")
            print(f"  Δ recursion  (full - fixed):   {delta('mmbiomor_fixed')}")
            print(f"  Δ recursion  (full - norec):   {delta('mmbiomor_norec')}")
            best_base = max((np.mean(agg[(b, regime)]) for b in BASELINES if (b, regime) in agg),
                            default=float("nan"))
            print(f"  Δ vs best baseline:            {fm-best_base:+.4f}")

if __name__ == "__main__":
    main()
