# COAST: Cross-Organ Benchmark and Conditioning Study for Histology-to-Transcriptomics Prediction

COAST is a controlled evaluation framework for predicting spatial gene expression from H&E histology.
It introduces two complementary cross-organ protocols on top of the standard within-cohort benchmark:

- **LOOO** (leave-one-organ-out): train on all organs except one, test on the held-out organ — measures unseen-organ transfer
- **POOLED**: train on all organs together, test on held-out patients — measures multi-organ deployment without organ shift

The same conditioning intervention (FiLM-style descriptor injection) is tested against five backbone architectures.
See `paper/` for the full write-up.

---

## Repository layout

```
COAST/
  data/                   # Data acquisition pipeline (see below)
  analysis/               # Aggregation, table / figure generation, significance tests
  FiLM_baselines/         # Conditioning-is-general study (FiLM on top of each backbone)
  MorphoST/               # MorphoST reference model (model code, training, evaluation, plots)
  stimage/                # STImage-1K4M dataset aggregation utilities
  utils/                  # Standalone helpers (gpu check, param count, equivariance test)
  experimental/           # Prototype models and scratchpads
  paper/                  # WACV 2027 paper draft (self-contained LaTeX, compiles with tectonic)

  baseline_spatial.py     # Feature-matched spatial baselines (ST-Net, HisToGene, Hist2ST, Gene-DML, HyperST)
  hest_baselines.py       # Baselines under STFlow's per-cohort HEST protocol
  train_cross_organ.py    # FiLM-conditioned cross-organ training (LOOO / POOLED)
```

---

## Dependencies

```bash
pip install torch torch_geometric timm einops scanpy pandas numpy scipy \
            huggingface_hub wandb openreview
```

The `STFlow` package (the backbone we build on) must be installed separately:

```bash
git clone https://github.com/bowang-lab/scGPT  # or the correct STFlow repo
pip install -e STFlow/
```

UNI and the full HEST dataset are **gated** on HuggingFace.  
Accept the licenses and authenticate before any download step:

```bash
huggingface-cli login          # paste your HF token when prompted
# OR: export HF_TOKEN=hf_...
```

---

## Data pipeline

All scripts below live in `data/`. Run them in order.

### Step 1 — Download HEST-1k benchmark cohorts (human)

Downloads the 10 standard HEST-bench cohorts into `dataset/`.
Each cohort directory contains `adata/`, `patches/`, `splits/`, and `var_50genes.json`.

```bash
# All 10 cohorts (CCRCC, COAD, HCC, IDC, LUNG, LYMPH_IDC, PAAD, PRAD, READ, SKCM)
python STFlow/stflow/app/hest/benchmark.py \
    --datasets all \
    --source_dataroot dataset \
    --embed_dataroot embed_dataroot \
    --weights_root weights_root \
    --encoders uni_v1_official \
    --batch_size 128

# Or download just the remaining cohorts (IDC, LYMPH_IDC, COAD, READ) if the others exist:
python data/download_cohorts.py
```

### Step 2 — Extract UNI patch features (human HEST-1k)

Reads `dataset/<COHORT>/patches/<ID>.h5` and writes
`embed_dataroot/<COHORT>/uni_v1_official/fp32/<ID>.h5` (shape: N × 1024).

```bash
python data/extract_features_uni.py \
    --dataset_root dataset \
    --embed_dataroot embed_dataroot \
    --weights_root weights_root \
    --device 0
```

### Step 3 — Build cross-organ splits (human HEST-1k)

Groups the 10 cohorts into 8 organ groups and produces leakage-safe LOOO + POOLED splits
with organ-specific 50-gene panels (gene selection uses only training spots).

```bash
python data/make_cross_organ_splits.py \
    --hest_root dataset \
    --out_root cross_organ_splits8
```

Output layout:
```
cross_organ_splits8/
  LOOO/splits/train_<organ>.csv  test_<organ>.csv
  LOOO/genes_<organ>.json
  POOLED/splits/train_<fold>.csv  test_<fold>.csv
  POOLED/genes_<fold>.json
  manifest.json
```

---

### Mouse data (Mus musculus extension)

#### Step 4 — Download mouse HEST samples

Downloads `st/<ID>.h5ad` and `patches/<ID>.h5` for 59 selected Visium slides
across 7 mouse organs (brain, liver, skin, heart, bowel, kidney, muscle).

Requires access to the gated `MahmoodLab/hest` dataset
(request at https://huggingface.co/datasets/MahmoodLab/hest).

```bash
python data/download_mouse_hest.py \
    --dataset_root dataset
```

Downloads into `dataset/MOUSE_<ORGAN>/adata/` and `dataset/MOUSE_<ORGAN>/patches/`.

#### Step 5 — Extract UNI features (mouse)

```bash
python data/extract_mouse_features.py \
    --dataset_root dataset \
    --embed_dataroot embed_dataroot \
    --weights_root weights_root \
    --device 0
```

#### Step 6 — Build mouse cross-organ splits

```bash
python data/make_mouse_splits.py \
    --dataset_root dataset \
    --out_root mouse_cross_organ_splits
```

---

### STImage-1K4M (optional second benchmark)

```bash
# 1. Select slides (leakage-safe organ-balanced subset)
python data/select_stimage_slides.py

# 2. Download raw coords / counts / images
python data/download_stimage_slides.py --manifest stimage/manifest.csv

# 3. Convert to HEST-style layout
python data/build_stimage.py

# 4. Build cross-organ splits
python data/make_stimage_splits.py --out_root stimage_splits
```

---

## Running evaluations

### Feature-matched baselines (LOOO / POOLED)

```bash
python baseline_spatial.py \
    --model histogene \          # st-net | histogene | hist2st | genedml | hyperst
    --regime LOOO \              # LOOO | POOLED | INTRA
    --splits_root cross_organ_splits8 \
    --source_dataroot dataset \
    --embed_dataroot embed_dataroot \
    --save_root results_baselines \
    --seed 1
```

### FiLM conditioning study

```bash
python FiLM_baselines/film_baselines.py \
    --model histogene \
    --film_type desc \           # none | desc | local
    --splits_root cross_organ_splits8 \
    --seed 1
```

### Cross-organ training (LOOO / POOLED)

```bash
python train_cross_organ.py \
    --regime LOOO \
    --film_type context \        # none | desc | local | context | meta
    --splits_root cross_organ_splits8 \
    --source_dataroot dataset \
    --embed_dataroot embed_dataroot \
    --seed 1
```

### MorphoST reference model

```bash
python MorphoST/train.py \
    --regime LOOO \
    --splits_root cross_organ_splits8 \
    --source_dataroot dataset \
    --embed_dataroot embed_dataroot \
    --n_layers 4 --n_sample_steps 5 \
    --seed 1
```

---

## Aggregating results

```bash
python analysis/aggregate_comparison.py    # main cross-organ comparison table
python analysis/aggregate_ablation.py      # FiLM ablation (none vs desc vs local vs context)
python analysis/final_significance.py      # paired significance tests
python analysis/make_figure.py             # per-organ LOOO bar chart
```

---

## Building the paper

```bash
cd paper/
tectonic main.tex      # produces main.pdf
tectonic appendix.tex  # produces appendix.pdf
```

Requires [tectonic](https://tectonic-typesetting.github.io/) (self-contained, downloads TeX packages automatically).
All style files (`wacv.sty`, `ieeenat_fullname.bst`, `references.bib`) are bundled in `paper/`.
