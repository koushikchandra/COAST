import json,glob,os,numpy as np
best_base={"POOLED":0.7637,"LOOO":0.4358}  # deepspace / genedml
rows={}
for d in glob.glob("results_cap_*"):
    tag=d.replace("results_cap_","")
    for kf in glob.glob(os.path.join(d,"*","results_kfold.json")):
        reg=os.path.basename(os.path.dirname(kf)).split("_")[0]
        rows.setdefault((tag,reg),[]).append(json.load(open(kf))["pearson_mean"])
for reg in ["POOLED","LOOO"]:
    print(f"[{reg}] best_baseline={best_base[reg]:.4f}")
    xs=sorted(((np.mean(v),t,len(v)) for (t,r),v in rows.items() if r==reg),reverse=True)
    for m,t,n in xs: print(f"   {t:8} {m:.4f} (n={n})  Δbase={m-best_base[reg]:+.4f}")
