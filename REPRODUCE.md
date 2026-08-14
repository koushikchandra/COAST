# Reproducing COAST Results

This file gives the exact commands, hyperparameters, and seeds needed to reproduce
every number in the paper. Run steps in order; each step is idempotent (safe to rerun).

---

## Environment

```bash
pip install torch torch_geometric timm einops scanpy pandas numpy scipy \
            huggingface_hub wandb h5py
pip install -e STFlow/          # STFlow must be on PYTHONPATH for all training commands
export PYTHONPATH=STFlow         # add to every python call below, or set once here
huggingface-cli login            # required for UNI encoder and HEST dataset access
```

---

## Step 1 — Data

Follow `data/README.md` (or root `README.md` §Data pipeline) in full.
Key outputs needed before training:

| Path | Contents |
|------|----------|
| `dataset/<COHORT>/adata/*.h5ad` | Expression (10 HEST cohorts) |
| `embed_dataroot/<COHORT>/uni_v1_official/fp32/*.h5` | UNI patch features |
| `cross_organ_splits8/` | LOOO + POOLED splits + gene panels |

---

## Step 2 — Baseline experiments (Table 1 / Table 2)

Run all 5 baselines × 2 regimes × 3 seeds. Each task is independent and idempotent.

```bash
for MODEL in stnet histogene hist2st triplex bleep; do
  for REGIME in LOOO POOLED; do
    for SEED in 1 2 3; do
      python baseline_spatial.py \
        --model $MODEL --regime $REGIME --seed $SEED \
        --feature_encoder uni_v1_official \
        --splits_root cross_organ_splits8 \
        --source_dataroot dataset \
        --embed_dataroot embed_dataroot \
        --save_root results_baselines \
        --val_fraction 0.15 \
        --epochs 100 \
        --lr 1e-4 \
        --dim 512 \
        --heads 8 \
        --depth 4 \
        --k 8 \
        --device 0
    done
  done
done
```

**Key hyperparameters (shared across all models):**

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam, lr=1e-4 |
| Epochs | 100 (early stop patience=20) |
| Val fraction | 0.15 (from training set) |
| Gene panel | 50 HVGs per fold (from `genes_<fold>.json`) |
| Features | UNI v1 (1024-dim) |
| Normalization | log1p |
| kNN neighbors | k=8 |
| Seeds | 1, 2, 3 (results averaged) |

**Model-specific dims:**

| Model | `--dim` | `--heads` | `--depth` |
|-------|---------|-----------|-----------|
| HisToGene | 512 | 8 | 4 |
| Hist2ST | 512 | 8 | 4 (transformer) + 4 (GCN) |
| ST-Net | 512 | — | — |
| TRIPLEX | 512 | 8 | 4 |
| BLEEP | 512 | — | — |

Results written to: `results_baselines/<REGIME>_<MODEL>_seed<S>/`

---

## Step 3 — FiLM conditioning study (Table 3)

Tests whether adding organ descriptor conditioning to each baseline improves cross-organ performance.

```bash
for MODEL in stnet histogene hist2st triplex bleep; do
  for FILM in none desc local; do
    for REGIME in LOOO POOLED; do
      for SEED in 1 2 3; do
        python FiLM_baselines/film_baselines.py \
          --model $MODEL --film_type $FILM \
          --regime $REGIME --seed $SEED \
          --splits_root cross_organ_splits8 \
          --source_dataroot dataset \
          --embed_dataroot embed_dataroot \
          --save_root results_film \
          --val_fraction 0.15 \
          --epochs 100 \
          --device 0
      done
    done
  done
done
```

**FiLM types:**

| `--film_type` | Description |
|---------------|-------------|
| `none` | No conditioning (baseline) |
| `desc` | Organ name as learned embedding (descriptor) |
| `local` | Local image statistics as conditioning signal |

---

## Step 4 — Cross-organ training (Table 4)

```bash
for REGIME in LOOO POOLED; do
  for SEED in 1 2 3; do
    python train_cross_organ.py \
      --regime $REGIME --seed $SEED \
      --splits_root cross_organ_splits8 \
      --source_dataroot dataset \
      --embed_dataroot embed_dataroot \
      --save_root results_cross_organ \
      --val_fraction 0.15 \
      --epochs 150 \
      --lr 1e-4 \
      --dim 256 \
      --n_layers 4 \
      --n_heads 4 \
      --k 8 \
      --device 0
  done
done
```

---

## Step 5 — Aggregate results

```bash
python analysis/aggregate_comparison.py \
    --baselines_root results_baselines \
    --cross_organ_root results_cross_organ \
    --out results_summary.csv

python analysis/final_significance.py \
    --summary results_summary.csv

python analysis/make_figure.py \
    --summary results_summary.csv \
    --out paper/figures/figure2.pdf
```

---

## Evaluation metric

All results use **mean Pearson correlation** across the 50 gene panel, averaged over spots within a slide, then averaged over test slides. Computed by `analysis/aggregate_comparison.py` using the per-fold `fold_*_results.json` files.

For 3-seed runs: report mean ± std across seeds.

---

## SLURM (HPC)

For cluster runs, use the sbatch scripts in the repo root. Each script is a job array:

```bash
sbatch mouse_baselines.sbatch        # 30-task array: 5 models × 2 regimes × 3 seeds
sbatch mouse_baselines_nova.sbatch   # same, nova partition (idle GPU nodes)
```

All scripts are **fold-level idempotent**: if `fold_<X>_results.json` exists it is skipped,
so jobs can be requeued safely after preemption.

---

## Expected runtimes (single A100 GPU)

| Task | Time per seed |
|------|--------------|
| Baseline LOOO (7 folds) | ~2–4 h |
| Baseline POOLED (5 folds) | ~1–3 h |
| Cross-organ training LOOO | ~4–6 h |

---

## Checklist

- [ ] HEST-1k downloaded (`dataset/`)
- [ ] UNI features extracted (`embed_dataroot/`)
- [ ] Cross-organ splits built (`cross_organ_splits8/`)
- [ ] Baselines run (3 seeds each, LOOO + POOLED)
- [ ] FiLM study run (3 film types × 3 seeds)
- [ ] Results aggregated (`results_summary.csv`)
- [ ] Figures generated (`paper/figures/`)
