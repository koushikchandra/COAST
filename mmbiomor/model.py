"""Multimodal bioMoR for histology -> spatial-transcriptomics (COAST).

STANDALONE reimplementation of bioMoR's concepts (no dependency on the original
recursive_marker_transformer package), adapted from cell-type *classification*
(gene-expression in) to *spatial expression regression* (histology in):

  * Mixture-of-Recursions (MoR): one shared transformer block applied K times,
    with a keep-priority (expert-choice) router that gives each token an
    adaptive depth -- ``RecursiveStack`` + ``ExpertChoiceRouter``.
  * Biological gene-gene graph injected in a *modulated* way:
      - as a zero-init message-passing residual on the router logits
        (``token_graph``), so depth respects the biological neighbourhood;
      - as a router ``prior`` (gene centrality);
      - and as a zero-init **FiLM** modulation of the gene tokens (``BioFiLM``),
        matching COAST's "conditioning-is-general" thesis: biology starts as an
        identity map and the model *learns* how much to inject.

Multimodality: modality 1 = histology (UNI patch features, per spot); modality 2
= biology (learnable gene-identity tokens + the gene-gene graph over the panel).
The 50 panel genes are the token set (M = n_genes); each spot is a sample (B).

Forward contract matches COAST's ``baseline_spatial.py`` harness:
    forward(feat[N,feature_dim], gxy[N,2], adj[N,N], labels=None) -> [N, n_genes]
``needs_labels = True`` only so the harness adds ``self.aux_loss`` (router
regularizers); labels themselves are unused by the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedTransformerBlock(nn.Module):
    """Pre-norm transformer block (MHSA + FFN); applied K times by the stack."""

    def __init__(self, dim, heads, ff_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ff_mult * dim, dim),
        )

    def forward(self, x, attn_bias=None):
        # x: [B, M, d]; attn_bias: [M, M] additive attention mask or None
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_bias, need_weights=False)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x


class ExpertChoiceRouter(nn.Module):
    """Keep-priority MoR router (expert-choice).

    At each of K steps: score the currently-active tokens, keep the top
    ``capacity`` fraction, and apply the shared block to survivors. The number of
    steps a token survives is its adaptive recursion depth (= importance). The
    biological ``token_graph`` enters as a *zero-init* graph-conv residual on the
    routing logits, so it can only learn to help.
    """

    def __init__(self, dim, depth, capacity=(1.0, 0.75, 0.75, 0.75)):
        super().__init__()
        self.depth = depth
        cap = list(capacity)
        if len(cap) < depth:
            cap = cap + [cap[-1]] * (depth - len(cap))
        self.capacity = cap[:depth]
        self.routers = nn.ModuleList([nn.Linear(dim, 1) for _ in range(depth)])
        # zero-init biological message-passing residual (concept: bio-router)
        self.graph_gate = nn.Parameter(torch.zeros(depth))
        self.graph_proj = nn.ModuleList([nn.Linear(2 * dim, 1) for _ in range(depth)])
        for gp in self.graph_proj:
            nn.init.zeros_(gp.weight)
            nn.init.zeros_(gp.bias)

    def forward(self, tokens, block, prior=None, prior_weight=0.0, token_graph=None):
        B, M, d = tokens.shape
        cand = torch.ones(B, M, device=tokens.device, dtype=torch.bool)
        depth_count = torch.zeros(B, M, device=tokens.device)
        z_loss = tokens.new_zeros(())
        for t in range(self.depth):
            logits = self.routers[t](tokens).squeeze(-1)               # [B, M]
            if token_graph is not None:
                agg = token_graph @ tokens                             # [B, M, d]
                gres = self.graph_proj[t](
                    torch.cat([agg, tokens - agg], dim=-1)).squeeze(-1)
                logits = logits + torch.sigmoid(self.graph_gate[t]) * gres
            if prior is not None:
                logits = logits + prior_weight * prior.unsqueeze(0)    # [B, M]
            z_loss = z_loss + (logits ** 2).mean()
            logits = logits.masked_fill(~cand, float("-inf"))
            k = min(M, max(1, int(round(self.capacity[t] * M))))
            topk = logits.topk(k, dim=1).indices                       # [B, k]
            keep = torch.zeros_like(cand)
            keep.scatter_(1, topk, True)
            keep = keep & cand
            # Mask-based keep-priority: run the block over the FULL token set (every
            # token keeps full attention context) but only *update* the survivors.
            # At M=50 tokens the FLOP saving from gathering survivors is marginal and
            # empirically hurt accuracy (deeper steps lost context), so we keep full
            # context here -- parameter-efficient (one shared block) without the loss.
            updated = block(t, tokens)                                 # [B, M, d]
            gate = keep.unsqueeze(-1).float()
            tokens = tokens + gate * (updated - tokens)                # update survivors only
            depth_count = depth_count + keep.float()
            cand = keep
        info = {"depth_per_token": depth_count, "z_loss": z_loss / max(1, self.depth)}
        return tokens, info


class RecursiveStack(nn.Module):
    """Weight-shared transformer with configurable recursion (ablation axis).

    recursion:
      * "mor"   -- keep-priority expert-choice router (adaptive depth) [proposed]
      * "fixed" -- apply the shared block ``depth`` times to ALL tokens (no router)
      * "none"  -- a single block application (recursion off)
    """

    def __init__(self, dim, heads, depth, dropout=0.1, share_weights=True,
                 capacity=(1.0, 0.75, 0.75, 0.75), recursion="mor"):
        super().__init__()
        self.depth = depth
        self.recursion = recursion
        eff_depth = 1 if recursion == "none" else depth
        n_blocks = 1 if share_weights else eff_depth
        self.blocks = nn.ModuleList([
            SharedTransformerBlock(dim, heads, dropout=dropout) for _ in range(n_blocks)
        ])
        self.router = ExpertChoiceRouter(dim, depth, capacity) if recursion == "mor" else None

    def _block(self, t):
        return self.blocks[0] if len(self.blocks) == 1 else self.blocks[t]

    def forward(self, tokens, prior=None, prior_weight=0.0, token_graph=None, attn_bias=None):
        if self.recursion == "mor":
            block = lambda t, x: self._block(t)(x, attn_bias)
            return self.router(tokens, block, prior=prior,
                               prior_weight=prior_weight, token_graph=token_graph)
        steps = 1 if self.recursion == "none" else self.depth
        for t in range(steps):
            tokens = self._block(t)(tokens, attn_bias)
        info = {"depth_per_token": tokens.new_full(tokens.shape[:2], float(steps)),
                "z_loss": tokens.new_zeros(())}
        return tokens, info


class BioInject(nn.Module):
    """Biological conditioner injection with a switchable mode (ablation axis).

      * "film" -- zero-init FiLM: x -> (1 + gamma(c)) * x + beta(c)   [MODULATED]
      * "add"  -- unmodulated:    x -> x + proj(c)                    [same biology,
                  injected additively, i.e. WITHOUT modulation -- the key contrast]
      * "none" -- no biological conditioner injection at all

    For "film"/"add" the projection is zero-initialized so each variant starts at
    identity and only *learns* to use biology (fair, no init advantage).
    """

    def __init__(self, dim, mode="film"):
        super().__init__()
        self.mode = mode
        if mode == "film":
            self.to_gamma = nn.Linear(dim, dim)
            self.to_beta = nn.Linear(dim, dim)
            for lin in (self.to_gamma, self.to_beta):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
        elif mode == "add":
            self.proj = nn.Linear(dim, dim)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        # x: [B, M, d]; cond: [M, d]
        if self.mode == "film":
            gamma = self.to_gamma(cond).unsqueeze(0)                   # [1, M, d]
            beta = self.to_beta(cond).unsqueeze(0)
            return x * (1.0 + gamma) + beta
        if self.mode == "add":
            return x + self.proj(cond).unsqueeze(0)
        return x


class MMBiomorNet(nn.Module):
    """Multimodal bioMoR: histology (UNI) -> spatial gene expression, with the
    biological gene-graph injected via a zero-init FiLM + MoR bio-router."""

    needs_labels = True   # only to route router regularizers through self.aux_loss

    def __init__(self, feature_dim, dim, depth, heads, n_genes, dropout,
                 capacity=(1.0, 0.75, 0.75, 0.75), prior_weight=0.1,
                 aux_weight=1e-3, share_weights=True,
                 bio_mode="film", use_graph=True, use_prior=True, recursion="mor"):
        super().__init__()
        self.n_genes = n_genes
        self.use_graph = use_graph
        self.use_prior = use_prior
        # per-spot histology encoder: input projection + a residual MLP block
        # (the single-linear version underfit the image->expression map).
        self.feat_in = nn.Linear(feature_dim, dim)
        self.feat_mlp = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * dim, dim),
        )
        self.feat_norm = nn.LayerNorm(dim)
        # biological gene-identity tokens (modality 2)
        self.gene_ident = nn.Parameter(torch.randn(n_genes, dim) * 0.02)
        self.inject = BioInject(dim, mode=bio_mode)
        self.stack = RecursiveStack(dim, heads, depth, dropout, share_weights,
                                    capacity, recursion=recursion)
        self.head = nn.Linear(dim, 1)
        self.prior_weight = prior_weight
        self.aux_weight = aux_weight
        # gene-gene graph over the panel (installed per fold), row-stochastic
        self.register_buffer("token_graph", torch.zeros(n_genes, n_genes))
        self.register_buffer("gene_prior", torch.zeros(n_genes))
        self._have_graph = False
        self.aux_loss = torch.zeros(())

    @torch.no_grad()
    def set_bio_graph(self, operator, prior=None):
        """Install the [G,G] row-stochastic biological operator + [G] prior."""
        self.token_graph.copy_(operator.to(self.token_graph))
        if prior is not None:
            self.gene_prior.copy_(prior.to(self.gene_prior))
        self._have_graph = True

    def forward(self, feat, gxy, adj, labels=None):
        # feat: [N, feature_dim] -> predict [N, n_genes]
        h = self.feat_in(feat)                                         # [N, d] per-spot histology
        h = self.feat_norm(h + self.feat_mlp(h))                       # residual MLP block
        tok = self.gene_ident.unsqueeze(0).expand(feat.shape[0], -1, -1)   # [N, G, d] biology
        tok = tok + h.unsqueeze(1)                                     # multimodal fusion
        tok = self.inject(tok, self.gene_ident)                       # bio injection (film / add / none)
        tg = self.token_graph if (self._have_graph and self.use_graph) else None
        pr = self.gene_prior if (self._have_graph and self.use_prior) else None
        out, info = self.stack(tok, prior=pr, prior_weight=self.prior_weight, token_graph=tg)
        pred = self.head(out).squeeze(-1)                              # [N, G]
        self.aux_loss = self.aux_weight * info["z_loss"]
        return pred


# ---------------------------------------------------------------------------
# Ablation variants. Each maps a --model name to an MMBiomorNet config.
# Headline contrast: modulated (full, FiLM) vs unmodulated (nomod, additive)
# biological injection. Plus component knockouts and a recursion ablation.
# ---------------------------------------------------------------------------
VARIANTS = {
    # proposed: modulated bio-injection (FiLM) + bio-graph router + prior + MoR
    "mmbiomor_full":    dict(bio_mode="film", use_graph=True,  use_prior=True,  recursion="mor"),
    # WITHOUT modulation: same biology, injected additively (FiLM -> add)
    "mmbiomor_nomod":   dict(bio_mode="add",  use_graph=True,  use_prior=True,  recursion="mor"),
    # component knockouts
    "mmbiomor_nograph": dict(bio_mode="film", use_graph=False, use_prior=True,  recursion="mor"),
    "mmbiomor_noprior": dict(bio_mode="film", use_graph=True,  use_prior=False, recursion="mor"),
    # no biology at all (unimodal histology + MoR)
    "mmbiomor_nobio":   dict(bio_mode="none", use_graph=False, use_prior=False, recursion="mor"),
    # recursion ablation: full biology but fixed-depth (no keep-priority router)
    "mmbiomor_fixed":   dict(bio_mode="film", use_graph=True,  use_prior=True,  recursion="fixed"),
    # recursion off entirely (single block) with full biology
    "mmbiomor_norec":   dict(bio_mode="film", use_graph=True,  use_prior=True,  recursion="none"),
}
VARIANTS["mmbiomor"] = VARIANTS["mmbiomor_full"]   # alias


def build_variant(name, feature_dim, dim, depth, heads, n_genes, dropout):
    cfg = VARIANTS.get(name, VARIANTS["mmbiomor_full"])
    return MMBiomorNet(feature_dim, dim, depth, heads, n_genes, dropout, **cfg)
