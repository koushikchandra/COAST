"""Standalone biological gene-gene graph over a fold's gene panel.

Same concept as bioMoR's gene-gene network (STRING/KEGG/Reactome union) but
self-contained: a **co-expression** graph estimated from the fold's *training*
spots -- no external database download. Genes that co-vary across training spots
are linked; the row-stochastic operator drives the MoR bio-router message
passing and the FiLM conditioner, and the degree gives the router centrality
prior. Estimated on train only (leakage-safe).
"""

import numpy as np
import torch


def build_coexpr_operator(label_arrays, n_genes, topk=8):
    """Build (operator[G,G] row-stochastic, prior[G]) from training expression.

    Parameters
    ----------
    label_arrays : list of np.ndarray, each [n_spots_i, n_genes] (log1p expr)
    n_genes      : int, G (panel size)
    topk         : keep the strongest ``topk`` |corr| neighbours per gene
    """
    X = np.concatenate([np.asarray(a, dtype=np.float64) for a in label_arrays], axis=0)  # [Ntot, G]
    if X.shape[0] < 3:
        # too few spots to estimate correlation -> no graph
        return torch.zeros(n_genes, n_genes), torch.zeros(n_genes)

    C = np.corrcoef(X.T)                          # [G, G]
    C = np.nan_to_num(np.abs(C), nan=0.0)
    np.fill_diagonal(C, 0.0)

    if topk and topk < n_genes - 1:               # sparsify: keep top-k per row
        keep = np.zeros_like(C)
        order = np.argsort(-C, axis=1)[:, :topk]
        rows = np.arange(n_genes)[:, None]
        keep[rows, order] = C[rows, order]
        C = keep
    C = np.maximum(C, C.T)                         # symmetric union

    prior = C.sum(axis=1)                          # degree centrality
    prior = (prior - prior.mean()) / (prior.std() + 1e-6)

    row = C.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    operator = C / row                             # row-stochastic

    return (torch.tensor(operator, dtype=torch.float32),
            torch.tensor(prior, dtype=torch.float32))
