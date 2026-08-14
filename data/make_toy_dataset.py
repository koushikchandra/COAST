"""Generate a tiny synthetic COAST-format dataset for testing without HuggingFace access.

Creates sample_data/ with 2 toy slides (TOY01 train, TOY02 test) mimicking the exact
file layout and array shapes expected by baseline_spatial.py and train_cross_organ.py.

Usage:
    python data/make_toy_dataset.py [--out_root sample_data] [--n_spots 300] [--n_genes 50]
"""
import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd
import scanpy as sc

COHORT = "TOY_ORGAN"
SAMPLES = ["TOY01", "TOY02"]
ENCODER = "uni_v1_official"
SEED = 0


def make_barcodes(n):
    return np.array([f"ACGT{i:04d}-1".encode() for i in range(n)], dtype="S18").reshape(-1, 1)


def make_coords(n, grid_size=20):
    rng = np.random.default_rng(SEED)
    rows = rng.integers(0, grid_size, size=n)
    cols = rng.integers(0, grid_size, size=n)
    return np.stack([rows * 100, cols * 100], axis=1).astype(np.int64)


def make_expression(n_spots, gene_names, seed):
    rng = np.random.default_rng(seed)
    counts = rng.negative_binomial(2, 0.3, size=(n_spots, len(gene_names))).astype(np.float32)
    return counts


def write_embed_h5(path, n_spots, dim=1024, seed=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        f.create_dataset("barcodes",   data=make_barcodes(n_spots))
        f.create_dataset("coords",     data=make_coords(n_spots))
        f.create_dataset("embeddings", data=rng.standard_normal((n_spots, dim)).astype(np.float32))


def write_adata_h5ad(path, n_spots, gene_names, seed):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    X = make_expression(n_spots, gene_names, seed)
    barcodes = [f"ACGT{i:04d}-1" for i in range(n_spots)]
    coords = make_coords(n_spots)
    adata = sc.AnnData(X=X, obs=pd.DataFrame(index=barcodes),
                       var=pd.DataFrame(index=gene_names))
    adata.obsm["spatial"] = coords.astype(np.float64)
    adata.write_h5ad(path)


def write_splits(splits_dir, gene_dir, gene_names):
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(gene_dir, exist_ok=True)

    def row(sid):
        return {"sample_id": sid,
                "patches_path": f"{COHORT}/patches/{sid}.h5",
                "expr_path":    f"{COHORT}/adata/{sid}.h5ad"}

    # single fold: TOY01 train, TOY02 test
    pd.DataFrame([row("TOY01")]).to_csv(os.path.join(splits_dir, "train_0.csv"), index=False)
    pd.DataFrame([row("TOY02")]).to_csv(os.path.join(splits_dir, "test_0.csv"),  index=False)

    with open(os.path.join(gene_dir, "genes_0.json"), "w") as f:
        json.dump({"genes": list(gene_names), "fold": 0}, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", default="sample_data")
    p.add_argument("--n_spots",  type=int, default=300)
    p.add_argument("--n_genes",  type=int, default=50)
    args = p.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out  = os.path.join(root, args.out_root)

    rng = np.random.default_rng(SEED)
    gene_names = [f"Gene{i:03d}" for i in range(args.n_genes)]

    print(f"Writing toy dataset to {out}/")
    for i, sid in enumerate(SAMPLES):
        # UNI embeddings
        embed_path = os.path.join(out, "embed_dataroot", COHORT, ENCODER, "fp32", f"{sid}.h5")
        write_embed_h5(embed_path, args.n_spots, dim=1024, seed=i)
        print(f"  wrote {embed_path}")

        # Expression h5ad
        adata_path = os.path.join(out, "dataset", COHORT, "adata", f"{sid}.h5ad")
        write_adata_h5ad(adata_path, args.n_spots, gene_names, seed=i + 10)
        print(f"  wrote {adata_path}")

    # Splits (INTRA single fold)
    splits_dir = os.path.join(out, "toy_splits", "INTRA", "splits")
    gene_dir   = os.path.join(out, "toy_splits", "INTRA")
    write_splits(splits_dir, gene_dir, gene_names)
    print(f"  wrote splits → {splits_dir}")
    print(f"  wrote gene panel → {gene_dir}/genes_0.json")

    print("\nDone. Test with:")
    print(f"  PYTHONPATH=STFlow python baseline_spatial.py \\")
    print(f"    --model stnet --regime INTRA --seed 1 \\")
    print(f"    --splits_root {os.path.join(args.out_root, 'toy_splits')} \\")
    print(f"    --source_dataroot {os.path.join(args.out_root, 'dataset')} \\")
    print(f"    --embed_dataroot {os.path.join(args.out_root, 'embed_dataroot')} \\")
    print(f"    --save_root {os.path.join(args.out_root, 'results')} --device cpu")


if __name__ == "__main__":
    main()
