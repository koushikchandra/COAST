# MMbioMOR runbook (branch `MMbioMOR`)

Multimodal bioMoR for histology → spatial-transcriptomics on the COAST cross-organ
benchmark, plus a faithful reproduction of the paper's baselines/STFlow tables.

## What's new on this branch
- `mmbiomor/` — standalone multimodal bioMoR (`MMBiomorNet`): per-spot UNI features →
  gene-identity tokens → Mixture-of-Recursions (keep-priority router) → per-gene head.
  Biology injected via **zero-init FiLM**, a **bio-graph router residual**, and a
  **centrality prior**. 7 ablation variants (`VARIANTS` in `mmbiomor/model.py`).
- `mmbiomor/bio_graph.py` — leakage-safe co-expression gene-graph from training spots.
- `MorphoST/evaluation.py` — the eval utilities the COAST scripts import
  (`expression_metrics` = **pooled** per-gene Pearson, `train_val_split`,
  `save_predictions`, `spot_role_index`). This module was missing from the public repo.
- `baseline_spatial.py` — adds `mmbiomor*` models (dispatch + per-fold graph install).
- Sweeps/aggregators: `sweep_baselines/film/cross_organ/mmbiomor/captune.sbatch`,
  `aggregate_mmbiomor_ablation.py`, `.cap_report.py`.
- `patch_stflow_eigh.py` — fixes STFlow's `eigh` crash on torch 2.10.

## 1. Environment
```bash
pip install torch torch_geometric timm einops scanpy pandas numpy scipy \
            huggingface_hub h5py
git clone <STFlow repo> STFlow && pip install -e STFlow/    # STFlow backbone
python patch_stflow_eigh.py STFlow                          # torch-2.10 eigh fix
huggingface-cli login                                       # UNI + HEST are gated
export PYTHONPATH="$PWD/STFlow:$PWD"
```

## 2. Data (same as COAST README §Data pipeline)
Produce these three, then everything runs:
```
dataset/<COHORT>/adata/*.h5ad                     # 10 HEST-bench cohorts
embed_dataroot/<COHORT>/uni_v1_official/fp32/*.h5 # UNI patch features
cross_organ_splits8/{LOOO,POOLED}/...             # splits + 50-gene panels
```
(Use `data/download_cohorts.py`, `data/extract_features_uni.py`,
`data/make_cross_organ_splits.py`.) On our cluster these are symlinked from a shared
copy; on yours point the `--source_dataroot/--embed_dataroot/--splits_root` flags at them.

## 3. EDIT before running (the sbatch are cluster-specific)
In every `*.sbatch`: the `cd /work/mech-ai-scratch/tirtho/COAST` path, `#SBATCH
--account=`, the `.venv` activation path, and the `MAILTO` in the email scripts.

## 4. Run
```bash
# Reproduction (paper tables) — pooled Pearson, LOOO + POOLED, 3 seeds
sbatch sweep_baselines.sbatch        # 6 backbones           -> results_baselines/
sbatch sweep_film.sbatch             # FiLM study            -> results_film/
sbatch sweep_cross_organ.sbatch      # STFlow reference      -> results_cross_organ/

# MMbioMOR ablation (7 variants) — writes into results_baselines/ for a unified table
sbatch sweep_mmbiomor.sbatch
python aggregate_mmbiomor_ablation.py   # baselines vs variants + Δmod/Δbio/Δbaseline

# Capacity-funnel tuning (optional): 5 schedules on mmbiomor_full
sbatch sweep_captune.sbatch
python .cap_report.py                    # ranked schedules vs best baseline
```
A single run: `python baseline_spatial.py --model mmbiomor_full --regime POOLED
--seed 1 --feature_encoder uni_v1_official --splits_root cross_organ_splits8
--source_dataroot dataset --embed_dataroot embed_dataroot --save_root results_baselines
--epochs 100 --dim 512 --heads 8 --depth 4 --k 8 --device 0`.
Tune the funnel with `MMB_CAPACITY="1,0.9,0.9,0.9"` (default `1,0.75,0.75,0.75`).

## 5. Expected numbers (our runs)
- Reproduction is faithful: STFlow POOLED ≈ 0.784 (paper 0.785); baselines within
  ~0.02–0.04; conditioning-helps result reproduces.
- `mmbiomor_full` ≈ **0.745 POOLED / 0.41 LOOO** — competitive at ~2–3× fewer params
  than the transformer baselines (5.3M/4.2M vs 9–12M). Headline ablation:
  **Δ modulation (full − nomod) ≈ +0.019** (modulated FiLM beats additive).
- Capacity: bioMoR's 0.75 floor is the default; gentler funnels are worth checking
  (their 0.5 floor over-prunes pooling models).

## Metric note
`expression_metrics` is **pooled** per-gene Pearson over all eval spots (matches how the
paper's numbers were produced), not per-slide averaging.
