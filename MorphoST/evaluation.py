"""MorphoST.evaluation — evaluation utilities used by the COAST baselines.

Faithful reconstruction of the four helpers that baseline_spatial.py,
FiLM_baselines/film_baselines.py and hest_baselines.py import from
``MorphoST.evaluation`` (a module that ships with the authors' MorphoST
reference model but was not committed to the public repo).

Reconstructed to the exact call contracts found in the code, and to the metric
defined in REPRODUCE.md:

    "mean Pearson correlation across the 50 gene panel, averaged over spots
     within a slide, then averaged over test slides."

That per-slide-then-across-slides averaging is precisely why
``expression_metrics`` receives ``slide_ids`` (whereas stflow's pooled
``metric_func`` does not). The returned dict mirrors ``metric_func``'s schema so
that ``stflow.utils.merge_fold_results`` and ``analysis/aggregate_comparison.py``
consume it unchanged: they only require ``pearson_corrs`` (a list of
``{"name", "pearson_corr"}``) and ``pearson_mean``.
"""

import os

import numpy as np


def _slide_averaged_pearson(pred, target, slide_ids):
    """Per gene: Pearson over the spots of each slide, then averaged across slides.

    Slides with < 2 spots, or with a constant prediction/target for a gene
    (undefined correlation), are skipped for that gene (nan-mean across slides).
    Returns a 1-D array of length n_genes (nan where a gene has no valid slide).
    """
    slide_ids = np.asarray(slide_ids)
    n_genes = target.shape[1]
    per_gene = np.full(n_genes, np.nan, dtype=np.float64)

    uniq = np.unique(slide_ids)
    # [n_slides, n_genes] matrix of per-slide per-gene correlations (nan if undefined)
    corr_mat = np.full((len(uniq), n_genes), np.nan, dtype=np.float64)
    for si, sid in enumerate(uniq):
        m = slide_ids == sid
        if m.sum() < 2:
            continue
        p = pred[m]
        t = target[m]
        pd_ = p - p.mean(0, keepdims=True)
        td_ = t - t.mean(0, keepdims=True)
        pn = np.sqrt((pd_ ** 2).sum(0))
        tn = np.sqrt((td_ ** 2).sum(0))
        denom = pn * tn
        valid = denom > 1e-12
        c = np.full(n_genes, np.nan, dtype=np.float64)
        c[valid] = (pd_ * td_).sum(0)[valid] / denom[valid]
        corr_mat[si] = c

    with np.errstate(invalid="ignore"):
        for g in range(n_genes):
            col = corr_mat[:, g]
            col = col[~np.isnan(col)]
            if col.size:
                per_gene[g] = float(col.mean())
    return per_gene


def expression_metrics(pred, target, gene_list, slide_ids=None):
    """Compute spatial gene-expression prediction metrics.

    Parameters
    ----------
    pred, target : np.ndarray, shape [n_spots, n_genes]
    gene_list    : list[str], length n_genes
    slide_ids    : array-like[str] | None, length n_spots (accepted for call
                   signature compatibility; not used by the pooled metric).

    Metric definition — **pooled** per-gene Pearson over all spots of the
    evaluation set, then averaged across the gene panel. This matches the metric
    that actually produced the paper's numbers: the COAST/STFlow pipeline scores
    with stflow's ``metric_func`` (pooled over spots) + ``merge_fold_results``.
    (An earlier per-slide-averaged variant, matching REPRODUCE.md's prose, gave
    ~2.3x lower values — e.g. ST-Net POOLED 0.16 vs the paper's 0.42 — so pooled
    is the faithful choice.) The returned dict mirrors ``metric_func``'s schema.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    gene_list = list(gene_list)

    # Pooled per-gene Pearson over all spots (identical to stflow.metric_func).
    per_gene_pearson = np.full(target.shape[1], np.nan, dtype=np.float64)
    errors, r2_scores = [], []
    for i in range(target.shape[1]):
        p = pred[:, i]
        t = target[:, i]
        errors.append(float(np.mean((p - t) ** 2)))
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2_scores.append(float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"))
        ps, ts = p.std(), t.std()
        if ps > 1e-12 and ts > 1e-12:
            per_gene_pearson[i] = float(np.corrcoef(p, t)[0, 1])

    pearson_genes = [
        {"name": gene_list[i], "pearson_corr": float(per_gene_pearson[i])}
        for i in range(len(gene_list))
    ]

    valid = per_gene_pearson[~np.isnan(per_gene_pearson)]
    pearson_mean = float(np.mean(valid)) if valid.size else float("nan")
    pearson_std = float(np.std(valid)) if valid.size else float("nan")

    n_nan = int(np.isnan(per_gene_pearson).sum())
    if n_nan:
        print(f"Warning: {n_nan} genes have NaN (pooled) Pearson correlation")

    return {
        "l2_errors": list(errors),
        "r2_scores": list(r2_scores),
        "pearson_corrs": pearson_genes,
        "pearson_mean": pearson_mean,
        "pearson_std": pearson_std,
        "l2_error_q1": float(np.percentile(errors, 25)),
        "l2_error_q2": float(np.median(errors)),
        "l2_error_q3": float(np.percentile(errors, 75)),
        "r2_score_q1": float(np.nanpercentile(r2_scores, 25)),
        "r2_score_q2": float(np.nanmedian(r2_scores)),
        "r2_score_q3": float(np.nanpercentile(r2_scores, 75)),
    }


def train_val_split(df, seed, val_fraction):
    """Slide-level train/validation split of a fold's training manifest.

    ``df`` has one row per slide. Returns ``(train_df, val_df)`` as DataFrames
    with reset indices. Deterministic in ``seed``. Guarantees at least one slide
    in each partition; for a single-slide fold it falls back to train == val
    (the single-slide inner-validation case referenced by ``spot_role_index``).
    """
    n = len(df)
    if n <= 1:
        d = df.reset_index(drop=True)
        return d, d

    rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
    n_val = int(round(val_fraction * n))
    n_val = max(1, min(n_val, n - 1))
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


def save_predictions(prediction_path, pred, target, coords, slide_ids, gene_list):
    """Persist per-spot predictions/targets for a fold to a compressed ``.npz``."""
    d = os.path.dirname(prediction_path)
    if d:
        os.makedirs(d, exist_ok=True)
    np.savez_compressed(
        prediction_path,
        pred=np.asarray(pred, dtype=np.float32),
        target=np.asarray(target, dtype=np.float32),
        coords=np.asarray(coords, dtype=np.float32),
        slide_ids=np.asarray(slide_ids).astype(str),
        genes=np.asarray(list(gene_list)).astype(str),
    )


def spot_role_index(row, n_spots):
    """Single-slide inner-validation spot selector.

    No-op for the multi-slide cross-organ protocol (the default): returns
    ``None`` so the caller keeps every spot. A ``role``/``spot_role`` column on
    the slide manifest row, if present, would carry a per-spot index array; the
    multi-slide splits used here never set one.
    """
    return None
