"""
Multi-relation dynamic graph encoder
=====================================================================
Three parts make up the encoder:

A  Multi-relation dynamic graph  (HGT, WWW'20 / HAN, WWW'19 / R-GCN, ESWC'18)
   R relation-specific low-rank bilinear score functions, each producing a
   row-stochastic edge distribution, combined per node by a node-level semantic
   attention weight beta_i^(r) that expresses the preference of node i over the
   relation types.

B  Layer-wise independent refinement  (SuperGlue, CVPR'20 / GATv2, ICLR'22)
   With num_layers=2 every layer and every modality owns its own adjacency head
   and re-estimates the graph from H^(l-1), so the layers no longer share a
   single graph and multi-hop reasoning is genuine.

C  Soft prior bias
   A = softmax( log(A_raw+eps) + lambda * log(C_norm+eps) )
   lambda=0 reduces the model to a purely dynamic graph and lambda -> inf to a
   purely prior-driven one, giving a single continuous and differentiable knob.
   A large lambda early in training lets the knowledge prior guide the graph and
   the learned value lets the data take over later.

A continuous softmax replaces hard Gumbel sampling with a straight-through
estimator (whose high variance made the sharpness of P oscillate), which trains
more stably, while sparsity is still enforced by the top-k mask.
=====================================================================
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from MGCN.lib.util_C_GCN import *


def l2norm(X, dim=-1, eps=1e-12):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def gen_adj_sym(A):
    """Symmetric normalisation, supporting both (N,N) and (B,N,N) shapes."""
    deg = A.sum(dim=-1) + 1e-8
    d = torch.pow(deg, -0.5)
    if A.dim() == 2:
        D = torch.diag(d)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
    else:
        D = torch.diag_embed(d)
        adj = torch.matmul(torch.matmul(A, D).transpose(-1, -2), D)
    return torch.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)


class DynamicAdjHead(nn.Module):
    """
    Multi-relation dynamic adjacency head
    ------------------------------------------------------------------
    scores:  for every relation r, S_r = (H U_r)(H V_r)^T, symmetrised and
             divided by a learnable temperature
    masking: threshold, a per-row floor, and an optional top-k (sparsity)
    relation probabilities: p_r = softmax(masked S_r), row-stochastic per relation
    semantic attention, node level (HGT/HAN):
        z_r = p_r @ H ; score_i^r = <h_i, z_ir>/sqrt(D)
        beta_i = softmax_r score_i        preference of node i over relation types
        A_raw = sum_r beta_i^r p_r        row-stochastic
    soft prior bias (C):
        A = softmax( log(A_raw+eps) + lambda log(C_norm+eps) )
    output:  association() returns the row-stochastic A, forward() returns the
             propagation matrix with self loops and symmetric normalisation
    """

    def __init__(self, embed_size, sparse_threshold=0.0, topk=None,
                 alpha_init=-2.0, prior_init=None, rank=None, num_heads=4,
                 edge_sampler='softmax', gumbel_tau=1.0, gumbel_hard=False,
                 edge_norm='row', sinkhorn_iters=10, use_batch_ctx=False,
                 topk_sample=5, tau_init=None, num_relations=None,
                 prior_lambda_init=2.0, **kwargs):
        super(DynamicAdjHead, self).__init__()
        self.embed_size = embed_size
        self.sparse_threshold = float(sparse_threshold)
        self.topk = topk if (topk is not None and int(topk) > 0) else None

        self.num_relations = int(num_relations) if (num_relations is not None and int(num_relations) > 0) \
            else max(1, int(num_heads))
        self.num_heads = self.num_relations          # keeps the old attribute name

        self.rank = int(rank) if (rank is not None and int(rank) > 0) else max(embed_size // 8, 16)
        self.head_dim = max(self.rank // self.num_relations, 4)
        self.effective_rank = self.head_dim * self.num_relations

        self.use_batch_ctx = bool(use_batch_ctx)

        # Relation-specific low-rank bilinear projections, S_r = (H U_r)(H V_r)^T
        self.W_a_u = Parameter(torch.Tensor(self.num_relations, embed_size, self.head_dim))
        self.W_a_v = Parameter(torch.Tensor(self.num_relations, embed_size, self.head_dim))

        # Batch-level (per-sample) context modulation, used by forward_batch only
        self.W_ctx_u = Parameter(torch.Tensor(embed_size, self.rank))
        self.W_ctx_v = Parameter(torch.Tensor(embed_size, self.rank))

        # Learnable temperature
        if tau_init is not None and float(tau_init) > 0:
            _t0 = max(float(tau_init), 1.0 + 1e-3)
            self._tau_init_used = _t0
            self.log_tau = nn.Parameter(torch.tensor(math.log(math.expm1(_t0 - 1.0))))
        else:
            self._tau_init_used = math.log(math.sqrt(embed_size) + 1.0) + 1.0
            self.log_tau = nn.Parameter(torch.tensor(math.log(math.sqrt(embed_size))))

        # Knowledge prior C
        self.C = None
        if prior_init is not None:
            init = prior_init.clone().float() if isinstance(prior_init, torch.Tensor) \
                else torch.Tensor(prior_init).float()
            self.C = Parameter(init)

        # Prior bias strength lambda = softplus(prior_lambda_raw)
        self.prior_lambda_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(max(float(prior_lambda_init), 1e-3)))))

        # Kept for the old reporting interface; this design uses no convex combination
        self.alpha = Parameter(torch.tensor(float(alpha_init)))

        self.reset_parameters()

    def reset_parameters(self):
        for r in range(self.num_relations):
            nn.init.xavier_uniform_(self.W_a_u[r])
            nn.init.xavier_uniform_(self.W_a_v[r])
        stdv = 1. / math.sqrt(self.embed_size)
        for w in (self.W_ctx_u, self.W_ctx_v):
            w.data.uniform_(-stdv, stdv)
        if self.C is not None:
            with torch.no_grad():
                self.C.data = self.C.data / (self.C.data.sum(dim=1, keepdim=True) + 1e-8)

    # ---------------- temperature ----------------
    def _tau(self):
        return F.softplus(self.log_tau) + 1.0

    @torch.no_grad()
    def set_temperature(self, tau):
        t = max(float(tau), 1.0 + 1e-3)
        self.log_tau.requires_grad_(False)
        self.log_tau.copy_(torch.tensor(math.log(math.expm1(t - 1.0)),
                                        device=self.log_tau.device, dtype=self.log_tau.dtype))

    @torch.no_grad()
    def tau_value(self):
        return float(self._tau().detach())

    # ---------------- prior ----------------
    def _get_prior(self, N, device, dtype):
        if self.C is None or self.C.size(0) != N:
            self.C = Parameter(torch.eye(N, device=device, dtype=dtype))
        return self.C

    def prior_lambda(self):
        return F.softplus(self.prior_lambda_raw)

    # ---------------- scoring ----------------
    def _relation_logits(self, H):
        """(R,N,N) symmetric low-rank bilinear logits per relation, already divided by the temperature."""
        logits = []
        for r in range(self.num_relations):
            pu = torch.matmul(H, self.W_a_u[r])
            pv = torch.matmul(H, self.W_a_v[r])
            s = torch.matmul(pu, pv.t())
            s = 0.5 * (s + s.t())
            logits.append(s)
        x = torch.stack(logits, dim=0) / self._tau()
        x = x.clamp(-10.0, 10.0)
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _ctx_bias(self, H, ctx_vec):
        ctx_vec = F.normalize(ctx_vec, p=2, dim=-1)
        ctx_proj = torch.matmul(self.W_ctx_v.t(), ctx_vec)
        return torch.matmul(torch.matmul(H, self.W_ctx_u), ctx_proj)

    def _apply_mask(self, logits, N):
        """The last two dimensions of logits are (N,N); applies the threshold, the per-row floor and the optional top-k."""
        row_max = logits.max(dim=-1, keepdim=True)[0]
        mask = (logits > self.sparse_threshold) | (logits == row_max)
        if self.topk is not None:
            k = min(self.topk, N)
            topk_vals = logits.topk(k, dim=-1)[0]
            th = topk_vals[..., -1].unsqueeze(-1)
            mask = mask & (logits >= th)
        return logits.masked_fill(~mask, float('-inf'))

    def _semantic_weights(self, H, p):
        """Node-level relation-type attention beta, returns (N,R).
        z_r = p_r @ H is the neighbourhood representation under relation r and
        score = <h,z>/sqrt(D).
        """
        z = torch.matmul(p, H.unsqueeze(0))                    # (R,N,D)
        scale = math.sqrt(max(H.size(-1), 1))
        score = (z * H.unsqueeze(0)).sum(dim=-1) / scale               # (R,N)
        return torch.softmax(score.permute(1, 0), dim=-1)       # (N,R)

    def _apply_prior(self, A_raw, N, device, dtype):
        """Log-linear soft prior bias, A = softmax(log A_raw + lambda log C)."""
        C = self._get_prior(N, device, dtype)
        C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        C_norm = C / (C.sum(dim=-1, keepdim=True) + 1e-8)
        Clog = torch.log(C_norm.clamp_min(1e-8))
        logit = torch.log(A_raw.clamp_min(1e-8)) + self.prior_lambda() * Clog
        A = torch.softmax(logit, dim=-1)
        return torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    def kl_to_prior(self, A):
        """Auxiliary regulariser KL(A || C_norm), the row-wise KL between the dynamic edge distribution and the knowledge prior."""
        N = A.size(0)
        C = self._get_prior(N, A.device, A.dtype)
        C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        C_norm = C / (C.sum(dim=-1, keepdim=True) + 1e-8)
        kl = (A * (torch.log(A.clamp_min(1e-8))
                   - torch.log(C_norm.clamp_min(1e-8)))).sum(dim=-1)
        return torch.nan_to_num(kl.mean(), nan=0.0, posinf=0.0, neginf=0.0)

    # ---------------- main path ----------------
    def _raw_association(self, H, ctx=None):
        """Learned association before the knowledge prior is applied."""
        N = H.size(0)
        logits = self._relation_logits(H)
        if self.use_batch_ctx and ctx is not None:
            bias = self._ctx_bias(H, ctx)
            logits = logits + bias.view(1, N, 1) + bias.view(1, 1, N)
        logits_m = self._apply_mask(logits, N)
        p = torch.softmax(logits_m, dim=-1)
        p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        beta = self._semantic_weights(H, p)
        A_raw = torch.einsum('nr,rnj->nj', beta, p)
        return torch.nan_to_num(A_raw, nan=0.0, posinf=0.0, neginf=0.0)

    def association(self, H, ctx=None):
        return self._apply_prior(self._raw_association(H, ctx), H.size(0), H.device, H.dtype)

    def _finalize(self, A, N, device, dtype):
        eye = torch.eye(N, device=device, dtype=dtype)
        return gen_adj_sym(A + eye)

    def forward(self, H, ctx=None, return_assoc=False):
        N = H.size(0)
        A = self.association(H, ctx)
        adj = self._finalize(A, N, H.device, H.dtype)
        if return_assoc:
            return adj, A
        return adj

    def forward_batch(self, H, ctx_batch):
        """Per sample, (B,N,N): base relation logits plus a per-sample context bias."""
        N = H.size(0)
        B = ctx_batch.size(0)
        base = self._relation_logits(H)                        # (R,N,N)
        ctx_vec = F.normalize(ctx_batch, p=2, dim=-1)
        ctx_proj = torch.matmul(ctx_vec, self.W_ctx_v)         # (B,rank)
        bias = torch.matmul(torch.matmul(H, self.W_ctx_u), ctx_proj.t()).t()  # (B,N)
        logits = base.unsqueeze(0) + bias.view(B, 1, N, 1) + bias.view(B, 1, 1, N)
        logits = logits.clamp(-10.0, 10.0)
        logits_m = self._apply_mask(logits, N)                 # (B,R,N,N)
        p = torch.softmax(logits_m, dim=-1)
        p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        z = torch.matmul(p, H.view(1, 1, N, -1))              # (B,R,N,D)
        scale = math.sqrt(max(H.size(-1), 1))
        score = (z * H.view(1, 1, N, -1)).sum(dim=-1) / scale           # (B,R,N)
        beta = torch.softmax(score.permute(0, 2, 1), dim=-1)   # (B,N,R)
        A_raw = torch.einsum('bnr,brnj->bnj', beta, p)        # (B,N,N)
        A_raw = torch.nan_to_num(A_raw, nan=0.0, posinf=0.0, neginf=0.0)
        C = self._get_prior(N, H.device, H.dtype)
        C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        C_norm = C / (C.sum(-1, keepdim=True) + 1e-8)
        Clog = torch.log(C_norm.clamp_min(1e-8))
        logit = torch.log(A_raw.clamp_min(1e-8)) + self.prior_lambda() * Clog.unsqueeze(0)
        A = torch.softmax(logit, dim=-1)
        A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        eye = torch.eye(N, device=H.device, dtype=H.dtype)
        return gen_adj_sym(A + eye.unsqueeze(0))

    # ---------------- KL and export interfaces ----------------
    def prior_kl(self, H):
        # The prior enters as a soft bias, so KL is no longer needed; the interface is kept for return_kl callers.
        return torch.zeros((), device=H.device, dtype=H.dtype)

    @torch.no_grad()
    def export_adjacency(self, H, ctx=None):
        was = self.training
        self.eval()
        try:
            N = H.size(0)
            A_raw = self._raw_association(H, ctx)
            A = self._apply_prior(A_raw, N, H.device, H.dtype)
            C = self._get_prior(N, H.device, H.dtype)
            C_norm = C / (C.sum(-1, keepdim=True) + 1e-8)
            out = {'prior': torch.nan_to_num(C_norm).float().cpu(),
                   'dynamic': A_raw.float().cpu(),
                   'fused': A.float().cpu(),
                   'lambda': float(self.prior_lambda().item()),
                   'alpha': float(torch.sigmoid(self.alpha).item())}
        finally:
            self.train(was)
        return out


class ImplicitInteractionLayer(nn.Module):
    """
    Implicit knowledge interaction unit
    ------------------------------------------------------------------
        M = A H W                              neighbourhood message
        G = sigmoid(W_g [H; M])                gate, per-dimension propagation weight
        H' = LeakyReLU( G*M + (1-G)*H + b )    gated propagation with a residual term
    """

    def __init__(self, embed_size, bias=False, dropout=0.2):
        super(ImplicitInteractionLayer, self).__init__()
        self.embed_size = embed_size
        self.proj_w = Parameter(torch.Tensor(embed_size, embed_size))
        self.gate = nn.Linear(embed_size * 2, embed_size, bias=True)   # gate network
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=dropout)
        if bias:
            self.bias = Parameter(torch.Tensor(embed_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj_w)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.embed_size)
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, H, A):
        # L2-normalise the input so that feature scale drift cannot overflow the aggregation
        H_norm = l2norm(H, dim=-1)
        M = torch.matmul(A, torch.matmul(H_norm, self.proj_w))   # aggregated message
        M = M.clamp(-50.0, 50.0)                                  # clamp to stay finite
        G = torch.sigmoid(self.gate(torch.cat([H_norm, M], dim=-1)))  # gate
        out = G * M + (1.0 - G) * H_norm                          # gated propagation
        if self.bias is not None:
            out = out + self.bias
        out = self.dropout(self.relu(out))
        return out


class MMGCN_Enc(nn.Module):
    """
    Multi-relation dynamic graph encoder
    ------------------------------------------------------------------
    Layer-wise refinement with num_layers=2: every layer and every modality owns
    its own DynamicAdjHead and re-estimates the graph from H^(l-1) (SuperGlue, CVPR'20).
    Each layer
      applies one head and one gated propagation inside text and inside visual;
      concatenates both into the full graph, applies the cross-modal head and gated
      propagation, then splits the result back for the next layer.
    The output is residual plus an L2 normalisation.
    """

    def __init__(self, graph_embed_size, bias=False, dropout=0.2, num_layers=2,
                 sparse_threshold=0.0, topk=None, alpha_init=-2.0, adj_prior=None,
                 edge_sampler='softmax', gumbel_tau=1.0, gumbel_hard=False,
                 edge_norm='row', sinkhorn_iters=10, use_batch_ctx=False,
                 topk_sample=5, tau_init=None, rank=None, num_heads=4,
                 num_relations=None, prior_lambda_init=2.0):
        super(MMGCN_Enc, self).__init__()
        self.graph_embed_size = graph_embed_size
        self.num_layers = max(1, int(num_layers))
        self.dropout = dropout
        self.use_batch_ctx = bool(use_batch_ctx)

        prior_text = adj_prior.get('adj_text') if adj_prior is not None else None
        prior_visual = adj_prior.get('adj_visual') if adj_prior is not None else None
        prior_all = adj_prior.get('adj_all') if adj_prior is not None else None

        def _head(prior):
            return DynamicAdjHead(
                graph_embed_size, sparse_threshold=sparse_threshold, topk=topk,
                alpha_init=alpha_init, prior_init=prior, edge_sampler=edge_sampler,
                gumbel_tau=gumbel_tau, gumbel_hard=gumbel_hard, edge_norm=edge_norm,
                sinkhorn_iters=sinkhorn_iters, use_batch_ctx=use_batch_ctx,
                topk_sample=topk_sample, tau_init=tau_init, rank=rank,
                num_heads=num_heads, num_relations=num_relations,
                prior_lambda_init=prior_lambda_init)

        # Three adjacency heads per layer
        self.heads = nn.ModuleList()
        for _ in range(self.num_layers):
            self.heads.append(nn.ModuleDict({
                'text': _head(prior_text),
                'visual': _head(prior_visual),
                'all': _head(prior_all),
            }))

        # Three gated interaction units per layer
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(nn.ModuleDict({
                'text': ImplicitInteractionLayer(graph_embed_size, bias=bias, dropout=dropout),
                'visual': ImplicitInteractionLayer(graph_embed_size, bias=bias, dropout=dropout),
                'all': ImplicitInteractionLayer(graph_embed_size, bias=bias, dropout=dropout),
            }))

    # ===================== compatibility interfaces =====================
    def adj_heads(self):
        """Return the (text, visual, all) adjacency heads of the first layer."""
        return (self.heads[0]['text'], self.heads[0]['visual'], self.heads[0]['all'])

    @property
    def adj_head_text(self):
        return self.heads[0]['text']

    @property
    def adj_head_visual(self):
        return self.heads[0]['visual']

    @property
    def adj_head_all(self):
        return self.heads[0]['all']

    @torch.no_grad()
    def set_temperature(self, tau):
        for layer in self.heads:
            for h in layer.values():
                h.set_temperature(tau)

    @torch.no_grad()
    def tau_report(self):
        # Average over the two layers
        names = ('text', 'visual', 'all')
        out = {n: 0.0 for n in names}
        for layer in self.heads:
            for n in names:
                out[n] += layer[n].tau_value()
        return {n: out[n] / self.num_layers for n in names}

    @torch.no_grad()
    def capacity_report(self):
        h = self.heads[0]['text']
        return {'requested_rank': h.rank, 'effective_rank': h.effective_rank,
                'num_heads': h.num_heads, 'head_dim': h.head_dim}

    # ===================== forward =====================
    def forward(self, text_embeddings, visual_embeddings, adj_text=None, adj_visual=None,
                adj_all=None, ctx_text=None, ctx_visual=None, ctx_all=None,
                return_adj=False, return_kl=False):
        Nt = text_embeddings.size(0)
        _x = torch.cat([text_embeddings, visual_embeddings], dim=0)

        text_nodes, visual_nodes = text_embeddings, visual_embeddings
        assoc_all = None
        for l in range(self.num_layers):
            h, layer = self.heads[l], self.layers[l]

            adj_t = h['text'](text_nodes, ctx=ctx_text)
            adj_v = h['visual'](visual_nodes, ctx=ctx_visual)
            text_nodes = l2norm(layer['text'](text_nodes, adj_t), dim=-1)
            visual_nodes = l2norm(layer['visual'](visual_nodes, adj_v), dim=-1)

            all_pre = torch.cat([text_nodes, visual_nodes], dim=0)
            adj_a, assoc = h['all'](all_pre, ctx=ctx_all, return_assoc=True)
            all_nodes = l2norm(layer['all'](all_pre, adj_a), dim=-1)

            text_nodes = all_nodes[:Nt]
            visual_nodes = all_nodes[Nt:]
            assoc_all = assoc

        concept_nodes = l2norm(_x + torch.cat([text_nodes, visual_nodes], dim=0), dim=-1)
        concept_nodes = torch.nan_to_num(concept_nodes, nan=0.0, posinf=0.0, neginf=0.0)

        kl = self.heads[-1]['all'].kl_to_prior(assoc_all) if return_kl else None
        if return_adj:
            if return_kl:
                return concept_nodes, assoc_all, kl
            return concept_nodes, assoc_all
        return concept_nodes

    @torch.no_grad()
    def export_adjacency(self, text_embeddings, visual_embeddings,
                         ctx_text=None, ctx_visual=None, ctx_all=None):
        was = self.training
        self.eval()
        try:
            out = {
                'text': self.heads[0]['text'].export_adjacency(text_embeddings, ctx=ctx_text),
                'visual': self.heads[0]['visual'].export_adjacency(visual_embeddings, ctx=ctx_visual),
            }
            Nt = text_embeddings.size(0)
            text_nodes, visual_nodes = text_embeddings, visual_embeddings
            for l in range(self.num_layers):
                h = self.heads[l]
                adj_t = h['text'].association(text_nodes, ctx=ctx_text)
                adj_v = h['visual'].association(visual_nodes, ctx=ctx_visual)
                text_nodes = l2norm(self.layers[l]['text'](text_nodes, adj_t), -1)
                visual_nodes = l2norm(self.layers[l]['visual'](visual_nodes, adj_v), -1)
                all_pre = torch.cat([text_nodes, visual_nodes], 0)
                if l == self.num_layers - 1:
                    out['all'] = h['all'].export_adjacency(all_pre, ctx=ctx_all)
        finally:
            self.train(was)
        return out


class GraphConvolution(nn.Module):
    """
    Standard GCN layer, kept for backward compatibility with older callers.
    The training path now uses DynamicAdjHead plus ImplicitInteractionLayer.
    """

    def __init__(self, in_features, out_features, bias=False, dropout=0.2):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return self.dropout(self.relu(output + self.bias))
        else:
            return self.dropout(self.relu(output))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
