import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math

from lib.loss import loss_select
from lib.mca import AdjacencyModel
from MGCN.lib.MMGCN import MMGCN_Enc

from MGCN.models.model.decoder import ConceptDecoder
from MGCN.models.layers.multi_head_attention import MyMultiHeadAttention
import logging

from collections import OrderedDict
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.init
import torchtext
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm

from MGCN.lib.utils import *
from MGCN.lib.C_GCN import C_GCN
import pickle
from scipy import linalg
import os
from torch.nn import Parameter

def extra_parameters(parser):

    # loss function
    parser.add_argument('--base_loss', default='trip', type=str, help='the loss function for the initial embeddings.')
    parser.add_argument('--gnn_loss', default='trip', type=str, help='the loss function for the enhanced embeddings.')

    # ---- Random seed for reproducibility; run with different seeds to report mean and std ----
    parser.add_argument('--seed', default=1, type=int,
                        help='random seed for reproducibility; run with different seeds to report mean±std in the main table.')

    # warmup for training
    # A long warmup delays the interaction blocks and their cold start then produces a
    # large loss spike, so the default is kept short and paired with a linear learning
    # rate warmup (--lr_warmup_iters), which lets the blocks take part from the start.
    parser.add_argument('--warmup', default=5000, type=int, help='warmup iteration for instance-level interaction network')

    # Graph modelling on fragment-level 
    parser.add_argument('--residual_weight', default=0.8, type=float, help='the weight of residual operation for pooling')

    # Graph modelling on instance-level 
    parser.add_argument('--num_layers_enc', default=1, type=int, help='the num_layers of Transformer encoder')   
    parser.add_argument('--nhead', default=16, type=int, help='the num_head for Transformer encoder')
    parser.add_argument('--dropout', default=0.1, type=float, help='the dropout rate for Transformer encoder')
    parser.add_argument('--graph_lr_factor', default=1., type=float, help='the learning rate factor for the interaction model')

    # connection and relevance relation 
    parser.add_argument('--mask_weight', default=1, type=float, help='use extra weight for the attention matrix')
    parser.add_argument('--threshold', default=0.5, type=float, help='give a threshold for the mask proportion, 0-1')
    parser.add_argument('--topk', default=10, type=int, help='the topk for the region-word pair selection')
    parser.add_argument('--reg_loss_weight', default=0.1, type=float, help='the values for the regularization loss')
    parser.add_argument('--norm_input', default=1, type=int, help='if use L2-norm embeddings as input')

    # ============ Multi-relation graph and concept-guided sampling ============
    # Multi-relation types
    parser.add_argument('--num_relations', default=4, type=int,
                        help='number of relation types (0 falls back to a single relation); each relation has its own low-rank bilinear head')
    # Initial strength of the soft prior bias
    parser.add_argument('--prior_lambda_init', default=2.0, type=float,
                        help='initial strength lambda of the log-linear prior bias (0 is purely dynamic, larger favours the prior)')
    # Concept-guided hard negative sampling (CG-HNM)
    parser.add_argument('--use_cg_hnm', default=1, type=int,
                        help='1 enables concept-guided hard negative sampling (CG-HNM)')
    parser.add_argument('--cg_hnm_mix', default=0.5, type=float,
                        help='mixing ratio between the concept-guided and the distance-based weights')
    parser.add_argument('--cg_hnm_temp', default=1.0, type=float,
                        help='softmax temperature of the hard negative guidance')
    parser.add_argument('--cgig_soft_gamma', default=0.1, type=float,
                        help='fraction gamma of the concept-graph gradient passed back (0 detaches fully, 1 passes it all)')
    # Multi-granularity alignment and temperature annealing
    parser.add_argument('--cmga_num_local', default=4, type=int,
                        help='number of local topology tokens (0 aligns the global view only)')
    parser.add_argument('--cmga_temp_start', default=0.5, type=float,
                        help='starting temperature of the CMGA annealing schedule')
    parser.add_argument('--cmga_temp_warmup', default=5000, type=int,
                        help='number of iterations over which the CMGA temperature anneals to its target')
    
    # loss function
    parser.add_argument('--cross_loss', default=1, type=int, help='if compute the loss for cross embeddings')
    # Weight of emb_loss, the matching loss on the concept-enhanced embeddings emb_v/emb_t,
    # exposed so that its sensitivity can be measured. The default is 0.4. When the concept
    # branch is disabled (use_concept_graph=0) the term is set to 0 and this weight has no effect.
    parser.add_argument('--emb_loss_weight', default=0.4, type=float, help='weight of emb_loss (concept-enhanced embedding matching loss)')

    # Hyper-parameters of the dynamic GCN adjacency head are defined in arguments.py

    return parser
class Fusion(nn.Module):
    def __init__(self, opt):
        super(Fusion, self).__init__()
        self.f_size = opt.embed_size
        # self.gate0 = nn.Linear(self.f_size, self.f_size)
        # self.gate1 = nn.Linear(self.f_size, self.f_size)

        self.fusion1 = nn.Linear(self.f_size * 2, self.f_size)
        self.relu = nn.ReLU()
        self.fusion2 = nn.Linear(self.f_size, self.f_size)
        self.bn = nn.BatchNorm1d(self.f_size)
        # self.fusion3 = nn.Linear(self.f_size * 2, self.f_size)
        # self.sigmoid = torch.sigmoid()

    def forward(self, vec1, vec2):
        vec_cat = torch.cat([vec1, vec2], dim=-1)
        map1_out = self.fusion1(vec_cat)
        map1_out = self.relu(map1_out)
        map2_out = self.fusion2(map1_out)
        gate = torch.sigmoid(torch.mul(map2_out, vec2))
        gate_result = gate * vec2
        # concat_result = torch.cat([vec1, gate_result], dim=-1)
        # concat_result = self.relu(self.fusion3(concat_result))
        gate_result = self.bn(gate_result.permute(0, 2, 1)).permute(0, 2, 1)
        f = gate_result + vec1
        return f
logger = logging.getLogger(__name__)


# =====================================================================
# Prior-quality control (--prior_mode)
#   The control matrix must differ from the real prior in semantics only, not in
#   scale, sparsity or degree distribution, otherwise a drop in score could come
#   from the graph becoming sparser rather than from the knowledge being destroyed.
# =====================================================================
def make_prior_variants(adj_prior, mode='real', seed=20260920):
    """Build the control matrices selected by --prior_mode.

    A single fixed random permutation perm is applied to the concept indices so that
    M'[i, j] = M[perm[i], perm[j]]. This is a graph isomorphism: the degree of every
    node, the edge count and the multiset of weights are unchanged, while which two
    concepts are connected is scrambled. Any performance gap between M' and M can
    therefore only come from semantics.

    :param adj_prior: dict with the three (N,N) tensors 'adj_text', 'adj_visual', 'adj_all'
    :param mode: 'real'     returns the input unchanged
                 'shuffle'  isomorphic permutation, same degree distribution, scrambled semantics
                 'identity' identity matrix, no prior knowledge beyond self loops. This is not
                            an isomorphic control because the scale changes too; it answers
                            whether the prior can be dropped altogether
    :return: dict with the three matrices
    """
    mode = str(mode)
    if mode == 'real':
        return adj_prior

    out = {k: v.clone() for k, v in adj_prior.items()}
    n_all = out['adj_all'].size(0)
    n_t = out['adj_text'].size(0)
    n_v = out['adj_visual'].size(0)
    g = torch.Generator().manual_seed(int(seed))

    if mode == 'identity':
        for k, v in out.items():
            out[k] = torch.eye(v.size(0), dtype=v.dtype)
        return out

    if mode != 'shuffle':
        raise ValueError("prior_mode supports real, shuffle and identity, got: {}".format(mode))

    # The three prior matrices share one set of concept indices, text concept i and visual
    # concept i, so the same permutation must be applied to all of them. Otherwise the
    # cross-modal correspondence breaks and the control destroys the alignment instead of
    # the semantics, which makes the result uninterpretable.
    if n_t != n_v:
        raise ValueError(
            'shuffle needs num_text_nodes == num_visual_nodes, got {} vs {}'.format(n_t, n_v))

    perm = torch.randperm(n_t, generator=g)          # one permutation shared by all three graphs

    out['adj_text'] = out['adj_text'][perm][:, perm]
    out['adj_visual'] = out['adj_visual'][perm][:, perm]
    if n_all == n_t:
        # adj_all has the same size as the blocks, so apply the same permutation directly
        out['adj_all'] = out['adj_all'][perm][:, perm]
    elif n_all == n_t + n_v:
        # adj_all is the 2N x 2N concatenation with the text block first and the visual block
        # second, so concat(perm, perm + N) keeps both the within-block scramble and the
        # cross-modal correspondence.
        perm_all = torch.cat([perm, perm + n_t])
        out['adj_all'] = out['adj_all'][perm_all][:, perm_all]
    else:
        raise ValueError(
            'shuffle cannot handle adj_all of size {} (text {}, visual {})'.format(n_all, n_t, n_v))
    return out


# =====================================================================
# Memory control: per-sample chunked computation in CMGA
# =====================================================================
class _PriorAdjSource(object):
    """Present the knowledge prior C as an adjacency head so that chunked_struct_readout
    can be reused.

    Used by --cmga_graph_source prior. By default the topology alignment uses the
    dynamically generated A_t/A_v; this shim supplies the control that aligns on the
    static knowledge prior C instead, which shows whether the gain comes from the dynamic
    topology or from any topology at all.
    It only has to implement forward_batch(H, ctx_batch) -> (b, N, N), the same interface
    as DynamicAdjHead. The prior does not depend on the sample, so every sample receives
    the same C_norm replicated over the batch.
    """

    def __init__(self, head):
        self.head = head

    def forward_batch(self, H, ctx_batch):
        N = H.size(0)
        C = self.head._get_prior(N, H.device, H.dtype)
        C_norm = C / (C.sum(dim=1, keepdim=True) + 1e-8)
        C_norm = torch.nan_to_num(C_norm, nan=0.0, posinf=0.0, neginf=0.0)
        b = ctx_batch.size(0) if ctx_batch is not None else 1
        return C_norm.unsqueeze(0).expand(b, -1, -1)


def chunked_struct_readout(adj_head, struct_encoder, basis, ctx, chunk=0, return_local=False):
    """Produce the per-sample structural descriptor, optionally in chunks of samples.

    Why chunking is valid
    ---------------------
    Both steps inside CMGA are independent across samples.
      * In the adjacency generation adj_head.forward_batch(H, ctx) the node features H are
        shared parameters and only the conditioning vector ctx[i] varies per sample;
        with use_batch_ctx=False (the default) the matrix A^i of sample i depends on
        ctx[i] alone and not on the rest of the batch.
      * The readout struct_encoder(A^i, basis) is a purely per-sample operation.
    Computing the whole batch at once is therefore mathematically identical to computing
    chunks and concatenating the results.

    Why chunking is needed
    ----------------------
    The two largest intermediate tensors of the forward pass live here
        A (B,N,N) with N=num_attribute (300 by default) and B=128  -> about 46 MB each
        H (B,N,D) with D=embed_size (1024)                        -> about 157 MB each
    struct_encoder materialises three to four (B,N,D) tensors at once, about 0.6 GB for
    one modality and 1.2 GB for text and visual together, and the Gumbel-Top-K sampling
    adds another four to six (B,N,N) tensors, which puts the CMGA peak near 2 GB.
    Chunking lowers the peak in proportion to chunk/B.

    :param adj_head:      a DynamicAdjHead, or any object exposing forward_batch(H, ctx)
    :param struct_encoder: GraphStructureEncoder
    :param basis:         (N, D) concept basis of the modality
    :param ctx:           (B, D) conditioning vector of each sample (cap_emb or img_emb)
    :param chunk:         chunk size when >0 and <B; 0, None or >=B disables chunking
    :return:              (B, D) structural descriptor, already L2 normalised

    Note: in training mode Gumbel-Top-K samples fresh noise for every sub-batch, so
    chunked and unchunked runs consume the random stream in a different order and are not
    bit-identical, although they are statistically equivalent. To reproduce a curve, keep
    the value of --cmga_chunk fixed.
    """
    B = ctx.size(0)

    def _run(c):
        A = adj_head.forward_batch(basis, c)
        return struct_encoder(A, basis, return_local=return_local)

    if not chunk or chunk <= 0 or chunk >= B:
        return _run(ctx)

    gs, Ls = [], []
    for s in range(0, B, chunk):
        out = _run(ctx[s:min(s + chunk, B)])
        if return_local:
            gs.append(out[0]); Ls.append(out[1])
        else:
            gs.append(out)
    if return_local:
        return torch.cat(gs, dim=0), torch.cat(Ls, dim=0)
    return torch.cat(gs, dim=0)


# =====================================================================
# Structural readout used by the cross-modal topology alignment (shared weights)
# =====================================================================
class GraphStructureEncoder(nn.Module):
    """
    Graph structure encoder
    ------------------------------------------------------------------
    Maps a concept graph plus the concept basis to
      global g : mean_n( MLP(A X) ), a permutation-invariant graph-level summary
      local  L : K learnable anchors attend over the structure-aware nodes and yield K
                 local topology tokens, each covering a different relational substructure,
                 which gives a multi-granularity description of the topology.
    Across modalities the global vectors are aligned with each other and the local tokens
    anchor by anchor.
    """

    def __init__(self, dim, dropout=0.1, num_local=4):
        super(GraphStructureEncoder, self).__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.num_local = int(num_local)
        if self.num_local > 0:
            self.local_anchors = Parameter(torch.zeros(self.num_local, dim))
            nn.init.normal_(self.local_anchors, std=1.0 / math.sqrt(dim))
            self.local_out = nn.Linear(dim, dim)

    def _nodes(self, A, X):
        H = torch.matmul(A, X.unsqueeze(0))          # (B,N,D)
        H = self.node_mlp(self.proj(H))
        return self.dropout(H)

    def forward(self, A, X, return_local=False):
        """
        :param A: (B,N,N) per-sample propagation matrix of the concept graph, symmetrically normalised
        :param X: (N,D)   concept basis of the modality, shared across samples
        :return: g of shape (B,D), or (g, L) with L of shape (B,K,D) when return_local is set
        """
        H = self._nodes(A, X)
        g = l2norm(H.mean(dim=1), dim=-1)            # (B,D)
        if not return_local or self.num_local <= 0:
            return g
        scale = math.sqrt(max(self.dim, 1))
        attn = torch.softmax(
            torch.matmul(self.local_anchors, H.transpose(1, 2)) / scale, dim=-1)  # (B,K,N)
        L = torch.matmul(attn, H)                    # (B,K,D)
        L = l2norm(self.local_out(L), dim=-1)
        return g, L


# instance-level relation modeling
class GraphLoss(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        self.iter_count = 0
        self.embed_size = opt.embed_size
        self.GT_label_ratio = opt.Concept_label_ratio
        self.fuse_weight = opt.fuse_weight
        self.score_weight = opt.score_weight
        self.cat_weight = opt.cat_weight

        # concept embeddings
        text_concept_nodes = np.load(os.path.join(opt.data_path, 'Concept_annotations',
                                                  opt.data_name + '_concepts_word2vec_{}.pkl'.format(
                                                      str(opt.num_attribute))), allow_pickle=True)
        visual_concept_nodes = np.load(os.path.join(opt.data_path, 'Concept_annotations',
                                                    opt.data_name + '_concepts_visual2vec_{}.pkl'.format(
                                                        str(opt.num_attribute))), allow_pickle=True)
        self.text_concept_nodes = Parameter(torch.Tensor(text_concept_nodes), requires_grad=False)
        self.visual_concept_nodes = Parameter(torch.Tensor(visual_concept_nodes), requires_grad=False)
        self.num_text_nodes = self.text_concept_nodes.size(0)
        self.num_visual_nodes = self.visual_concept_nodes.size(0)

        # concept adj matrix
        with open(os.path.join(opt.data_path, 'Concept_annotations',
                               'adj_matrix_{}.pkl'.format(str(opt.num_attribute))), 'rb') as f:
            adj_matrix = torch.Tensor(pickle.load(f))

        diag = torch.diag_embed(torch.diag(adj_matrix))
        adj_matrix = adj_matrix - diag
        self.adj_all = Parameter(adj_matrix / adj_matrix.sum(dim=1, keepdim=True), requires_grad=False)

        # Concept-level similarity matrix, used for the within-modality priors
        with open(os.path.join(opt.data_path, 'Concept_annotations',
                               'similarity_matrix_{}.pkl'.format(str(opt.num_attribute))), 'rb') as f:
            similarity_matrix = torch.Tensor(pickle.load(f))
            mask = torch.eye(similarity_matrix.size(0), dtype=int)
            similarity_matrix = similarity_matrix.masked_fill(mask == 1, 1.0)

        self.adj_text = Parameter(similarity_matrix[:self.num_text_nodes, :self.num_text_nodes], requires_grad=False)
        self.adj_visual = Parameter(similarity_matrix[self.num_text_nodes:, self.num_text_nodes:], requires_grad=False)

        # Dynamic adjacency head plus gated implicit knowledge interaction.
        # The external co-occurrence and similarity matrices only serve as a weak initial
        # value for the learnable prior C and take no part in inference.
        # edge_sampler='gumbel_topk' keeps k edges per row: the sampling is discrete in the
        #   structure, through a differentiable top-k decision, and continuous in the edge
        #   weights, through a softmax over the candidates.
        # use_batch_ctx is off by default so that the dynamic graph does not depend on the
        #   composition of the batch.
        # Prior strength: adj_alpha defaults to -2.0, sigmoid(-2) is about 0.12, so the
        #   dynamic part carries roughly 88 per cent of the weight.
        adj_prior = {
            'adj_text': self.adj_text.data.clone(),
            'adj_visual': self.adj_visual.data.clone(),
            'adj_all': self.adj_all.data.clone(),
        }

        # =================================================================
        # Master switch of the concept-level branch, plus the prior-quality control
        # =================================================================
        # use_concept_graph=0 removes the whole concept-level branch. This is a master
        #   switch: without A_c the concept-graph interaction and the multi-granularity
        #   alignment are disabled as well, so the variant corresponds to "without the
        #   concept-level branch" rather than "without the dynamic graph". To isolate the
        #   dynamic part, freeze the graph on the static prior instead
        #   (--fix_adj_alpha 1 --adj_alpha 12 keeps A_c but pins it to the prior).
        self.use_concept_graph = bool(int(getattr(opt, 'use_concept_graph', 1)))

        # prior_mode: 'real' (default), 'shuffle' (isomorphic permutation) or 'identity'
        self.prior_mode = str(getattr(opt, 'prior_mode', 'real'))
        adj_prior = make_prior_variants(adj_prior, mode=self.prior_mode)
        if self.prior_mode != 'real':
            print('[prior_mode={}] the prior has been permuted or replaced by the identity '
                  '(same degree distribution, semantics destroyed), this is a control run'.format(self.prior_mode))
            if not self.use_concept_graph:
                print('[warning] prior_mode has no effect with use_concept_graph=0, the concept branch is off')

        self.concept_enc = None
        if self.use_concept_graph:
            self.concept_enc = MMGCN_Enc(opt.embed_size, bias=True,
                                     num_layers=getattr(opt, 'num_gcn_layers', 2),
                                     sparse_threshold=getattr(opt, 'sparse_threshold', 0.0),
                                     topk=getattr(opt, 'adj_topk', 0) or None,
                                     alpha_init=getattr(opt, 'adj_alpha', -2.0),
                                     adj_prior=adj_prior,
                                     edge_sampler=getattr(opt, 'edge_sampler', 'softmax'),
                                     gumbel_tau=getattr(opt, 'gumbel_tau', 1.0),
                                     gumbel_hard=bool(getattr(opt, 'gumbel_hard', 0)),
                                     edge_norm=getattr(opt, 'edge_norm', 'row'),
                                     sinkhorn_iters=getattr(opt, 'sinkhorn_iters', 10),
                                     use_batch_ctx=bool(getattr(opt, 'use_batch_ctx', 0)),
                                     topk_sample=getattr(opt, 'edge_topk_sample', 5),
                                     tau_init=getattr(opt, 'tau_init', 0.0) or None,
                                     rank=(getattr(opt, 'adj_rank', 0) or None),
                                     num_heads=getattr(opt, 'adj_num_heads', 4),
                                     num_relations=(getattr(opt, 'num_relations', 0) or None),
                                     prior_lambda_init=getattr(opt, 'prior_lambda_init', 2.0))
            # ---- adj_alpha fixed or learnable ----
            # --fix_adj_alpha 1 freezes alpha on the three adjacency heads and treats
            # --adj_alpha as the fixed prior weight w=sigmoid(adj_alpha). This is how the
            # sensitivity curve of alpha and the two end points (w close to 0 for a purely
            # dynamic graph, w close to 1 for a purely static prior) are produced.
            if int(getattr(opt, 'fix_adj_alpha', 0)):
                _a0 = float(getattr(opt, 'adj_alpha', -2.0))
                with torch.no_grad():
                    for _h in self.concept_enc.adj_heads():
                        _h.alpha.requires_grad_(False)
                        _h.alpha.copy_(torch.tensor(_a0, dtype=_h.alpha.dtype,
                                                    device=_h.alpha.device))
                print('[fix_adj_alpha=1] alpha of the three adjacency heads frozen at {:.3f} '
                      '(prior weight w=sigmoid({:.3f})={:.4f})'.format(
                          _a0, _a0, 1.0 / (1.0 + math.exp(-_a0))))
        self.text_dec = ConceptDecoder(d_output=opt.embed_size,
                                        d_model=opt.embed_size,
                                        ffn_hidden=2 * opt.embed_size,
                                        n_head=opt.n_head_concept_encoder,
                                        n_layers=opt.n_layers_concept_decoder,
                                        drop_prob=opt.dropout_rate_concept_encoder)
        self.visual_dec = ConceptDecoder(d_output=opt.embed_size,
                                          d_model=opt.embed_size,
                                          ffn_hidden=2 * opt.embed_size,
                                          n_head=opt.n_head_concept_encoder,
                                          n_layers=opt.n_layers_concept_decoder,
                                          drop_prob=opt.dropout_rate_concept_encoder)

        # Initialize the dml objective function for embeddings learning.
        self.base_loss = loss_select(opt, loss_type=opt.base_loss)
        self.gnn_loss = loss_select(opt, loss_type=opt.gnn_loss)

        # the fusion interaction mechanism (intra-inter four-block sample-level matrix module)
        encoder_layer = nn.TransformerEncoderLayer(d_model=opt.embed_size, nhead=opt.nhead, 
                                                    dim_feedforward=opt.embed_size, dropout=opt.dropout)          
        self.gnn = nn.TransformerEncoder(encoder_layer, num_layers=opt.num_layers_enc)

        # construct the cross-embedding graph (AdjacencyModel)
        self.adj_model = AdjacencyModel(hidden_size=opt.embed_size, threshold=opt.threshold, topk=opt.topk, detach=True)

        self.criterion_KL_softmax = KL_loss_softmax()
        self.text_linear = nn.Linear(self.text_concept_nodes.size(-1), opt.embed_size)
        self.visual_linear = nn.Linear(self.visual_concept_nodes.size(-1), opt.embed_size)
        self.fusion1 = Fusion(opt)
        self.gated_fusion = GatedFusion(opt.embed_size)

        # =================================================================
        # Cross-modal graph topology alignment (CMGA)
        #   Replaces a single-modality self-comparison (text graph against text graph)
        #   with an alignment between the visual concept graph and the text concept graph.
        # =================================================================
        self.use_cmga = bool(getattr(opt, 'use_cmga', 1))
        self.cmga_weight = float(getattr(opt, 'cmga_weight', 0.05))
        # The reg_loss weight is a real CLI parameter rather than a hard-coded factor,
        # so the value used by the total loss and the value reported by the argument agree.
        # The default is 0.1 and --reg_loss_weight overrides it.
        self.reg_loss_weight = float(getattr(opt, 'reg_loss_weight', 0.1))
        # emb_loss weight, exposed as a CLI parameter. The default is 0.4 and
        # --emb_loss_weight overrides it when running a sensitivity sweep.
        self.emb_loss_weight = float(getattr(opt, 'emb_loss_weight', 0.4))
        self.cmga_temperature = float(getattr(opt, 'cmga_temperature', 0.1))
        self.topo_mode = str(getattr(opt, 'topo_mode', 'cross'))   # 'cross' | 'intra'
        self.topo_intra_weight = float(getattr(opt, 'topo_intra_weight', 0.0))
        # Chunk size of the CMGA computation (0 disables chunking)
        self.cmga_chunk = int(getattr(opt, 'cmga_chunk', 0))
        # Memory diagnostics: print the allocated memory of each stage for the first mem_debug iterations
        self.mem_debug = int(getattr(opt, 'mem_debug', 0))
        # ---- Controls for the CMGA ablation ----
        # cmga_detach_graph detaches the input of the structural readout.
        #   CMGA calls chunked_struct_readout(adj_head_*, ...), so gradients do flow back
        #   into the dynamic adjacency heads and the gap between --use_cmga 0 and 1 mixes
        #   two effects: the InfoNCE term itself and the extra training signal it gives the
        #   dynamic heads. Setting this to 1 removes the second effect and isolates the loss
        #   term contribution. The CGIG branch already detaches the association matrix.
        self.cmga_detach_graph = bool(int(getattr(opt, 'cmga_detach_graph', 0)))
        # cmga_graph_source: 'dyn' (generated by the dynamic heads, default) or 'prior'
        #   (uses the knowledge prior C instead), which shows whether the gain comes from
        #   the dynamic topology or from any topology at all.
        self.cmga_graph_source = str(getattr(opt, 'cmga_graph_source', 'dyn'))
        # cmga_share_encoder: 1 (default) shares one structural readout across modalities,
        #   so the local anchors correspond and the outputs can be aligned anchor by anchor;
        #   0 gives each modality its own readout plus a projection head, as a control.
        self.cmga_share_encoder = bool(int(getattr(opt, 'cmga_share_encoder', 1)))
        # Number of local topology tokens (0 aligns the global vector only)
        self.cmga_num_local = int(getattr(opt, 'cmga_num_local', 4))
        # Temperature annealing from cmga_temp_start towards cmga_temperature over the first
        # cmga_temp_warmup iterations
        self.cmga_temp_start = float(getattr(opt, 'cmga_temp_start', 0.5))
        self.cmga_temp_warmup = int(getattr(opt, 'cmga_temp_warmup', 5000))
        if self.use_cmga:
            self.struct_encoder = GraphStructureEncoder(
                opt.embed_size, dropout=getattr(opt, 'dropout', 0.1),
                num_local=self.cmga_num_local)
            if not self.cmga_share_encoder:
                self.struct_encoder_visual = GraphStructureEncoder(
                    opt.embed_size, dropout=getattr(opt, 'dropout', 0.1),
                    num_local=self.cmga_num_local)
                # Separate projection heads map both descriptors into a shared alignment space
                self.cmga_proj_text = nn.Sequential(
                    nn.Linear(opt.embed_size, opt.embed_size), nn.ReLU(inplace=True),
                    nn.Linear(opt.embed_size, opt.embed_size))
                self.cmga_proj_visual = nn.Sequential(
                    nn.Linear(opt.embed_size, opt.embed_size), nn.ReLU(inplace=True),
                    nn.Linear(opt.embed_size, opt.embed_size))
                print('[cmga] separate structural readouts with two projection heads, multi-granularity K=%d'
                      % self.cmga_num_local)

        # =================================================================
        # Structure injection from the concept level to the sample level (CGIG)
        #   The concept-level dynamic graph A_c and the concept activation distribution P of
        #   each sample give the sample-to-sample relation in concept-association space
        #       S = rownorm( P  A_c  P^T ),
        #   which is then fused with the instance-level relation matrix. This is what couples
        #   the two levels instead of merely stacking them: the concept graph is not just a
        #   feature bank queried by the decoder, it decides which samples interact.
        # =================================================================
        self.use_concept_guided_graph = bool(getattr(opt, 'use_concept_guided_graph', 1))
        self.concept_guide_weight = float(getattr(opt, 'concept_guide_weight', 0.3))
        self.concept_guide_tau = float(getattr(opt, 'concept_guide_tau', 10.0))
        # concept_guide_source decides where A_c comes from in CGIG.
        #   'dyn'      A_c generated by the dynamic heads (default)
        #   'prior'    the knowledge prior C instead. If this degrades to the level of
        #              --use_concept_guided_graph 0, the benefit depends on the dynamic
        #              graph; if it matches the default, any structured matrix would do and
        #              the dynamic part is redundant.
        #   'identity' A_c = I, so S = rownorm(P P^T) and the quadratic form carries no graph
        #              structure, which tests whether that intermediate level is needed.
        self.concept_guide_source = str(getattr(opt, 'concept_guide_source', 'dyn'))

        # ---- CGIG injection: two switches that keep S signed and avoid over-smoothing ----
        # A row-normalised S added to attn_mask would be cancelled by the row invariance of
        # softmax, so the injection would carry no information at all.
        self.cgig_demean = bool(int(getattr(opt, 'cgig_demean', 1)))
        self.cgig_unit_rms = bool(int(getattr(opt, 'cgig_unit_rms', 1)))
        # Gradient pass-through strength gamma (0 blocks it entirely, 1 passes all of it; default 0.1)
        self.cgig_soft_gamma = float(getattr(opt, 'cgig_soft_gamma', 0.1))
        # Master switch for concept-guided hard negative sampling
        self.use_cg_hnm = bool(int(getattr(opt, 'use_cg_hnm', 1)))
        # ---- Source of the concept response P: cos (default), zscore or attn ----
        self.concept_response_source = str(getattr(opt, 'concept_response_source', 'cos')).lower()
        if self.cgig_demean or self.cgig_unit_rms:
            logger.info('[CGIG] cgig_demean=%d cgig_unit_rms=%d -- the row constant cancelled by '
                        'softmax is removed; this changes the numbers.',
                        int(self.cgig_demean), int(self.cgig_unit_rms))
        if self.concept_response_source != 'cos':
            logger.info('[concept response] source=%s (tau=%.4g)',
                        self.concept_response_source, self.concept_guide_tau)

        # =================================================================
        # CLIP patch branch (disabled by default)
        #   The visual side otherwise has only the Faster-RCNN regions (36x2048) and the
        #   CLIP global embedding (512). CLIP ViT-B/32 also produces 49 patch tokens of
        #   width 512, which carry the spatial information of where a concept sits, and
        #   they need no additional model or weights.
        #   The patches are pooled into a D-dimensional vector and fused by a gate into the
        #   concept-enhanced visual feature. With use_patch_fusion=0 no parameter is created
        #   and nothing enters the forward pass.
        # =================================================================
        self.use_patch_fusion = bool(int(getattr(opt, 'use_patch_fusion', 0)))
        self.patch_pool = str(getattr(opt, 'patch_pool', 'concept')).lower()
        self.patch_beta = float(getattr(opt, 'patch_beta', 10.0))
        self.patch_dim = int(getattr(opt, 'patch_dim', 512))
        self.patch_num = int(getattr(opt, 'patch_num', 49))
        if self.use_patch_fusion:
            self.patch_proj = nn.Linear(self.patch_dim, opt.embed_size)
            # Note the argument name: the third parameter of graph_lib.GatedFusion is
            # init_gate_bias, while lib.encoders.GatedFusion names it gate_bias. They are
            # different classes.
            self.patch_gate = GatedFusion(
                opt.embed_size,
                init_gate_bias=float(getattr(opt, 'patch_gate_bias', -1.0)))
            if self.patch_pool == 'attn':
                # Learnable query, independent of the sample so that inference stays per-sample
                self.patch_query = Parameter(torch.zeros(opt.embed_size))
                nn.init.normal_(self.patch_query, std=0.02)
            logger.info('[patch branch] use_patch_fusion=1 pool=%s beta=%.4g '
                        '(patch_dim=%d, num=%d)', self.patch_pool, self.patch_beta,
                        self.patch_dim, self.patch_num)

        # =================================================================
        # KL regulariser on the prior
        #   A convex combination A = w*C_norm + (1-w)*A_dyn mixes two distributions with a
        #   single scalar and, when the dynamic graph is sparse, the dense prior dominates.
        #   The alternative adds L_kl = mean_i KL( pi_i || p_i ) with pi = softmax(S_masked)
        #   and p = rownorm(C), which has a clear probabilistic reading: the learned edge
        #   distribution should not drift far from the knowledge prior.
        #   The default 0.0 disables it; a positive value runs it alongside the convex
        #   combination and controls the prior strength.
        # =================================================================
        self.kl_prior_weight = float(getattr(opt, 'kl_prior_weight', 0.0))
        self.track_kl = self.kl_prior_weight > 0.0

    # ==================================================================
    # Reporting interfaces for the ablation and sensitivity studies
    # (all under @torch.no_grad, no effect on the forward pass)
    # ==================================================================
    @torch.no_grad()
    def alpha_report(self):
        """Return the current prior weight w = sigmoid(alpha) of the three adjacency heads.

        alpha is the mixing coefficient of the convex combination between the weak prior C
        and the dynamic graph A_dyn
            A = w * C_norm + (1 - w) * A_dyn,   w = sigmoid(alpha)
        so w is the quantity to report. alpha itself lives in logit space, where a distance
        from -2 to -1 is not comparable to one from 1 to 2, and plotting it directly would
        distort the curve.
        :return: {'text':.., 'visual':.., 'all':.., 'mean':..}, all nan when the concept
                 branch is disabled
        """
        if not self.use_concept_graph:
            return {'text': float('nan'), 'visual': float('nan'),
                    'all': float('nan'), 'mean': float('nan')}
        heads = (('text', self.concept_enc.adj_head_text),
                 ('visual', self.concept_enc.adj_head_visual),
                 ('all', self.concept_enc.adj_head_all))
        out = {}
        for name, h in heads:
            out[name] = float(torch.sigmoid(h.alpha.detach()).item())
        out['mean'] = (out['text'] + out['visual'] + out['all']) / 3.0
        return out

    @torch.no_grad()
    def tau_report(self):
        """Effective temperature of the three adjacency heads, as {name: float}.

        The CLI value --tau_init is only a request; annealing (--tau_anneal_epochs>0)
        drives it towards --tau_min. The horizontal axis of a temperature sensitivity
        curve must use these actual values rather than the CLI values.
        """
        if not self.use_concept_graph:
            return {'text': float('nan'), 'visual': float('nan'), 'all': float('nan')}
        return self.concept_enc.tau_report()

    @torch.no_grad()
    def capacity_report(self):
        """Capacity of the scoring function, the requested rank against the effective rank.

        effective_rank = head_dim * num_heads with head_dim = max(rank//num_heads, 4), so
        floor division and the minimum of 4 can make the effective rank smaller than the
        request (rank=100 with 8 heads gives head_dim=12 and an effective rank of 96).
        A capacity sensitivity study must therefore report effective_rank.
        """
        if not self.use_concept_graph:
            return {}
        return self.concept_enc.capacity_report()

    @torch.no_grad()
    def set_temperature(self, tau):
        """Set the effective temperature of the three heads and freeze log_tau."""
        if self.use_concept_graph:
            self.concept_enc.set_temperature(tau)

    # ==================================================================
    # Concept-level enhancement, shared by training and inference
    # ==================================================================
    # ==================================================================
    # CLIP patch branch
    # ==================================================================
    def _patch_focus(self, patch, concept_basis=None):
        """(B, P, patch_dim) -> (B, D), pooling the P patches of each image into one vector.

        Four pooling modes (--patch_pool)
          'mean'    plain average, parameter free
          'max'     per-dimension maximum, parameter free
          'attn'    attention pooling with a learnable query
          'concept' concept guided: the weight of a patch is set by how strongly it responds
                    to the concept nodes. A patch with a strong response carries spatial
                    evidence for that concept and should be kept, which amounts to
                    replacing word-level patch selection with concept-response weighting,
                    using soft weights rather than a hard selection.
        """
        p = l2norm(self.patch_proj(patch), dim=-1)          # (B, P, D)
        mode = self.patch_pool

        if mode == 'mean':
            v = p.mean(dim=1)
        elif mode == 'max':
            v = p.max(dim=1).values
        elif mode == 'attn':
            score = torch.matmul(p, self.patch_query)       # (B, P)
            w = torch.softmax(score * self.patch_beta / math.sqrt(max(p.size(-1), 1)), dim=-1)
            v = (w.unsqueeze(-1) * p).sum(dim=1)
        else:                                               # 'concept'
            if concept_basis is None:
                v = p.mean(dim=1)
            else:
                Xc = l2norm(concept_basis, dim=-1)          # (Nc, D)
                scale = self.patch_beta / math.sqrt(max(p.size(-1), 1))
                logits = torch.matmul(p, Xc.t()) * scale    # (B, P, Nc)
                # The strongest concept response of a patch measures its importance; normalise over patches
                imp = logits.max(dim=-1).values             # (B, P)
                w = torch.softmax(imp, dim=-1)              # (B, P)
                v = (w.unsqueeze(-1) * p).sum(dim=1)
        return l2norm(v, dim=-1)

    def _patch_fuse(self, emb_v, emb_t, patch, concept_basis):
        """Gated fusion of the patch branch. Returns the inputs unchanged when the branch is
        off or no patches are given.

        Only the visual side is affected, since patches come from the image.
        """
        if not self.use_patch_fusion or patch is None:
            return emb_v, emb_t
        v_focus = self._patch_focus(patch, concept_basis)
        return l2norm(self.patch_gate(emb_v, v_focus), dim=-1), emb_t

    def concept_enhance(self, img_emb, cap_emb, need_kl=False, patch=None):
        """
        Concept-level graph enhancement: concept nodes -> decoder -> gated fusion
        ------------------------------------------------------------------
        This is a separate method rather than inline code in forward because the inference
        path VSEModel.forward_emb(graph=True) reuses it, which is what lets the dynamic
        association enter the retrieval path. Training and inference then run the same
        code, so they cannot drift apart.

        Batch invariance: the concept nodes (num_attribute, D) are fixed parameters and do
        not depend on the composition of the batch. Only the decoder and the gated fusion
        see the sample, and both act per sample, so this path is safe at inference time.

        :param img_emb:  (B, D) image features, already L2 normalised
        :param cap_emb:  (B, D) text features, already L2 normalised
        :param need_kl:  also compute the KL prior regulariser, controlled by kl_prior_weight
        :return: dict with
                 emb_v / emb_t         (B, D) concept-enhanced retrieval features
                 concept_nodes         (Nc, D) concept node features
                 concept_assoc         (Nc, Nc) concept-level association matrix A_c
                 text_basis/visual_basis (Nc, D) concept basis used by CMGA
                 kl                    a scalar, or None
        """
        # With --use_concept_graph 0 the concept branch is removed and this degrades to an
        # identity mapping. concept_nodes and concept_assoc are None, so the downstream CGIG
        # skips itself, and the dict keeps the same keys so callers need no branch.
        if not self.use_concept_graph:
            # The patch branch still works when the concept branch is off, which gives the
            # "patches without a concept graph" variant. With use_patch_fusion=0 the fusion
            # returns its inputs unchanged.
            _v, _t = self._patch_fuse(img_emb, cap_emb, patch, None)
            return {'emb_v': _v, 'emb_t': _t,
                    'concept_nodes': None, 'concept_assoc': None,
                    'text_basis': None, 'visual_basis': None, 'kl': None,
                    'resp': None}

        text_basis = self.text_linear(self.text_concept_nodes)
        visual_basis = self.visual_linear(self.visual_concept_nodes)

        # By default the context is not built from a batch mean, which keeps batch invariance
        if self.concept_enc.use_batch_ctx:
            ctx_text = cap_emb.mean(dim=0)
            ctx_visual = img_emb.mean(dim=0)
            ctx_all = torch.cat([img_emb, cap_emb], dim=0).mean(dim=0)
        else:
            ctx_text = ctx_visual = ctx_all = None

        out = self.concept_enc(text_basis, visual_basis,
                               ctx_text=ctx_text, ctx_visual=ctx_visual, ctx_all=ctx_all,
                               return_adj=True, return_kl=need_kl)
        if need_kl:
            concept_nodes, concept_assoc, kl = out
        else:
            concept_nodes, concept_assoc = out
            kl = None

        cb = concept_nodes.unsqueeze(dim=0)
        # The cross-attention of the decoder is kept rather than discarded. Its shape is
        # (B, n_head, 1, Nc), which is exactly the softmaxed response distribution of a
        # sample over the concepts, the P used by --concept_response_source attn. It is
        # returned in the resp field for _concept_guided_relation.
        concept_v, attn_v = self.visual_dec(img_emb.unsqueeze(dim=1),
                                            cb.expand(img_emb.size(0), cb.size(1), cb.size(2)),
                                            trg_mask=None, src_mask=None)
        concept_t, attn_t = self.text_dec(cap_emb.unsqueeze(dim=1),
                                          cb.expand(cap_emb.size(0), cb.size(1), cb.size(2)),
                                          trg_mask=None, src_mask=None)
        concept_v = concept_v.squeeze(dim=1)
        concept_t = concept_t.squeeze(dim=1)

        # Per-dimension adaptive gated fusion, the dimension is unchanged
        emb_v = l2norm(self.gated_fusion(img_emb, concept_v), dim=-1)
        emb_t = l2norm(self.gated_fusion(cap_emb, concept_t), dim=-1)

        # CLIP patch branch, off by default, so emb_v is unchanged unless enabled
        emb_v, emb_t = self._patch_fuse(emb_v, emb_t, patch, visual_basis)

        return {'emb_v': emb_v, 'emb_t': emb_t,
                'concept_nodes': concept_nodes, 'concept_assoc': concept_assoc,
                'text_basis': text_basis, 'visual_basis': visual_basis,
                'kl': kl, 'resp': (attn_v, attn_t)}

    # ==================================================================
    # Sample-level four-block intra-inter matrix module, shared by training and inference
    # ==================================================================
    def _four_block(self, img_emb, cap_emb, img_emb_notnorm, cap_emb_notnorm,
                    img_emb_pre_pool, cap_emb_pre_pool, img_len, cap_len,
                    concept_nodes=None, concept_assoc=None, concept_resp=None):
        """
        AdjacencyModel (connection and relevance blocks) -> concept-guided fusion (CGIG)
        -> gnn relation modelling.

        :return: (img_emb_gnn, cap_emb_gnn, reg_loss), the first two being the L2 normalised
                 sample-level enhanced features, the same tensors gnn_loss uses in training.

        This was factored out of forward() so that the inference path
        (VSEModel.forward_emb) reuses exactly the same code, which rules out the
        distribution shift that two separate implementations would introduce.
        """
        mask_weight = self.opt.mask_weight
        batch_c, batch_r, reg_loss = self.adj_model(img_emb, cap_emb,
                                                    img_regions=img_emb_pre_pool,
                                                    cap_words=cap_emb_pre_pool,
                                                    img_len=img_len,
                                                    cap_len=cap_len,)

        # === Structure injection from the concept level (CGIG) ===
        # The concept-level association matrix A_c and the concept activation distribution P
        # of each sample give the sample-to-sample structure in concept-association space
        # through a quadratic form, fused with the instance-level relevance:
        #      S = rownorm( P A_c P^T )
        #      R' = (1-w) R + w * scale(R) * S
        # Two samples whose activated concepts are strongly connected in A_c should also
        # interact more strongly at the instance level, which couples the two levels
        # instead of merely stacking the two modules.
        S_vt_guide = None
        if self.use_concept_guided_graph and concept_nodes is not None:
            # concept_resp is the cross-attention of the concept decoder, of shape
            # (B,n_head,1,Nc), used only by --concept_response_source attn.
            S_vv, S_tt, S_vt = self._concept_guided_relation(
                concept_nodes, concept_assoc, img_emb, cap_emb, resp=concept_resp)
            batch_r['i2i'] = self._fuse_relation(batch_r['i2i'], S_vv)
            batch_r['t2t'] = self._fuse_relation(batch_r['t2t'], S_tt)
            batch_r['i2t'] = self._fuse_relation(batch_r['i2t'], S_vt)
            batch_r['t2i'] = self._fuse_relation(batch_r['t2i'], S_vt.t())
            # For CG-HNM: concept-guided hard negative sampling, detached and used as sampling weights only
            S_vt_guide = S_vt.detach()

        # connection relation
        connect_mask = torch.cat((torch.cat((batch_c['i2i'], batch_c['i2t']), dim=1),
                                torch.cat((batch_c['t2i'], batch_c['t2t']), dim=1)), dim=0)

        # relevance relation
        relation_mask = torch.cat((torch.cat((batch_r['i2i'], batch_r['i2t']), dim=1),
                                torch.cat((batch_r['t2i'], batch_r['t2t']), dim=1)), dim=0)

        mask = mask_weight * relation_mask.masked_fill_(~connect_mask, float('-inf'))

        # Safety: a fully masked row makes softmax return NaN and corrupts the whole batch.
        # This does not occur in training, where a sample is always connected to itself, but
        # it can occur at inference when the batch composition differs, for example 100
        # images against 500 captions during evaluation, so it is handled explicitly.
        row_ok = torch.isfinite(mask).any(dim=-1)
        if not bool(row_ok.all()):
            idx = torch.arange(mask.size(0), device=mask.device)
            bad = idx[~row_ok]
            mask[bad, bad] = 0.0

        # concat embeddings, batch as the dim=1
        if self.opt.norm_input:
            all_embs = torch.cat((img_emb, cap_emb), dim=0)
        else:
            all_embs = torch.cat((img_emb_notnorm, cap_emb_notnorm), dim=0)

        # get the instance-level relation modeling
        all_embs_gnn = self.gnn(all_embs.unsqueeze(1), mask).squeeze(1)

        # get the enhanced embeddings
        img_emb_gnn, cap_emb_gnn = torch.split(all_embs_gnn, img_emb.size(0), dim=0)

        # L2 normalization for the relation-enhanced embeddings
        img_emb_gnn = F.normalize(img_emb_gnn)
        cap_emb_gnn = F.normalize(cap_emb_gnn)

        return img_emb_gnn, cap_emb_gnn, reg_loss, S_vt_guide

    def instance_enhance(self, img_out, cap_out, patch=None):
        """
        Entry point for the reusable sample-level four-block enhancement at inference.

        :param img_out: the 5-tuple returned by the image encoder with graph=True,
                        (images, image_lengths, img_emb, img_emb_notnorm, img_emb_pre_pool)
        :param cap_out: the 5-tuple returned by the text encoder with graph=True,
                        (bert_emb, cap_len, cap_emb, cap_emb_notnorm, cap_emb_pre_pool)
        :return: dict with emb_v / emb_t, the sample-level enhanced retrieval features, and reg_loss

        Implementation notes
          - The four-block inputs match training exactly, that is the raw encoder outputs
            img_emb and cap_emb, so the trained AdjacencyModel and gnn never see an unseen
            feature distribution.
          - The concept graph only serves as a structural prior inside CGIG, so
            concept_enhance is called once here to obtain concept_nodes and concept_assoc.
            The embeddings it returns are not used on the inference path.
          - The output depends on the rest of the batch, so this is not a per-sample encoder.
        """
        img_feat, img_len, img_emb, img_emb_notnorm, img_emb_pre_pool = img_out
        cap_feat, cap_len, cap_emb, cap_emb_notnorm, cap_emb_pre_pool = cap_out

        # Same order as training: the patches go into concept_enhance, which only affects
        # the output of ce and leaves the raw encoder features fed to _four_block intact.
        ce = self.concept_enhance(img_emb, cap_emb, need_kl=False, patch=patch)

        img_emb_gnn, cap_emb_gnn, reg_loss, _ = self._four_block(
            img_emb, cap_emb, img_emb_notnorm, cap_emb_notnorm,
            img_emb_pre_pool, cap_emb_pre_pool, img_len, cap_len,
            ce['concept_nodes'], ce['concept_assoc'], ce['resp'])

        return {'emb_v': img_emb_gnn, 'emb_t': cap_emb_gnn, 'reg_loss': reg_loss}

    # ------------------------------------------------------------------
    # Memory diagnostics
    #   With --mem_debug N the first N iterations print the allocated memory, the peak and
    #   the memory reserved from the driver for each stage, which shows where an out of
    #   memory error comes from. Together with the reset_peak_memory_stats call at the start
    #   of each iteration in train.py, the peak covers the current stage.
    # ------------------------------------------------------------------
    def _mem(self, tag):
        if self.mem_debug <= 0 or self.iter_count > self.mem_debug:
            return
        if not torch.cuda.is_available():
            return
        logger.info('[mem][iter %d] %-18s cur=%6.2fGB  peak=%6.2fGB  reserved=%6.2fGB',
                    self.iter_count, tag,
                    torch.cuda.memory_allocated() / 2 ** 30,
                    torch.cuda.max_memory_allocated() / 2 ** 30,
                    torch.cuda.memory_reserved() / 2 ** 30)

    def forward(self, img_emb, cap_emb, concept_labels, img_ids, patch=None):
        # patch holds (B, P, patch_dim) CLIP patch features, or None. With None, or with
        # use_patch_fusion=0, the function behaves as if the branch did not exist.

        # get latent features and embeddings
        # include the pre-pooling and after-pooling features
        # The encoders return 5-tuples: (images, lengths, emb, emb_notnorm, emb_pre_pool)
        img_feat, img_len, img_emb, img_emb_notnorm, img_emb_pre_pool = img_emb
        cap_feat, cap_len, cap_emb, cap_emb_notnorm, cap_emb_pre_pool = cap_emb

        bs = img_emb_notnorm.shape[0]
        assert img_emb_notnorm.shape[0] == cap_emb_notnorm.shape[0]
        self._mem('after_encoders')

        # ---- Concept-level enhancement, shared with the inference path ----
        ce = self.concept_enhance(img_emb, cap_emb, need_kl=self.track_kl, patch=patch)
        emb_v, emb_t = ce['emb_v'], ce['emb_t']
        concept_nodes, concept_assoc = ce['concept_nodes'], ce['concept_assoc']
        text_concept_basis, visual_concept_basis = ce['text_basis'], ce['visual_basis']
        kl_prior = ce['kl']
        self._mem('+concept_enhance')

        # Concept sharpness diagnostics, one line every --concept_sharp_log_every iterations
        # (0 disables it). It shows within a single epoch whether the branch is active.
        _every = int(getattr(self.opt, 'concept_sharp_log_every', 100))
        if (_every > 0 and self.use_concept_guided_graph and concept_nodes is not None
                and self.iter_count % _every == 0):
            self._log_concept_sharpness(img_emb, cap_emb, concept_nodes,
                                        concept_assoc, ce['resp'])

        # basic matching loss
        base_loss = self.base_loss(img_emb, cap_emb, img_ids)
        # With the concept branch off emb_v equals img_emb, so computing emb_loss would count
        # base_loss twice. The ablation has to drop the term, otherwise it compares a doubled
        # base_loss weight rather than the presence of the concept graph.
        if self.use_concept_graph:
            emb_loss = self.base_loss(emb_v, emb_t, img_ids)
        else:
            emb_loss = torch.tensor(0.0, device=img_emb.device)
        self._mem('+base_emb_loss')

        # === Intra-inter four-block sample-level matrix module (AdjacencyModel + gnn + gnn_loss/reg_loss) ===
        if self.iter_count >= self.opt.warmup:
            img_emb_gnn, cap_emb_gnn, reg_loss, S_vt_guide = self._four_block(
                img_emb, cap_emb, img_emb_notnorm, cap_emb_notnorm,
                img_emb_pre_pool, cap_emb_pre_pool, img_len, cap_len,
                concept_nodes, concept_assoc, ce['resp'])

            # compute loss
            if self.opt.cross_loss:
                gnn_loss1 = self.gnn_loss(img_emb, cap_emb_gnn, img_ids)
                gnn_loss2 = self.gnn_loss(img_emb_gnn, cap_emb, img_ids)
            else:
                gnn_loss1 = 0.
                gnn_loss2 = 0.

            # gnn_loss3 uses the concept structure S_vt to guide hard negative sampling (triplet loss only)
            if (self.use_cg_hnm and S_vt_guide is not None
                    and hasattr(self.gnn_loss, 'loss_forward')):
                gnn_loss3 = self.gnn_loss(img_emb_gnn, cap_emb_gnn, img_ids,
                                          guide=S_vt_guide)
            else:
                gnn_loss3 = self.gnn_loss(img_emb_gnn, cap_emb_gnn, img_ids)
            gnn_loss = (gnn_loss1 + gnn_loss2 + gnn_loss3 )
                
        else:
            gnn_loss = 0.
            reg_loss = 0.
        self._mem('+four_block')

        # === Cross-modal graph topology alignment loss L_cmga ===
        # Without the concept branch there is no concept basis and CMGA cannot run, since it
        # depends on A_c, so it is removed together with --use_concept_graph 0.
        if self.use_cmga and self.use_concept_graph:
            cmga_loss = self._compute_cmga_loss(text_concept_basis, visual_concept_basis,
                                                cap_emb, img_emb, img_ids)
        else:
            cmga_loss = torch.tensor(0.0, device=img_emb.device)
        self._mem('+cmga')

        # === KL prior regulariser, off by default, active when kl_prior_weight > 0 ===
        if self.track_kl and kl_prior is not None and self.use_concept_graph:
            kl_loss = kl_prior
        else:
            kl_loss = torch.tensor(0.0, device=img_emb.device)

        # Total loss: base_loss + gnn_loss + emb_loss_weight * emb_loss (0.4 by default)
        #             + reg_loss_weight * reg_loss (0.1 by default)
        #             + cmga_weight * L_cmga     cross-modal graph topology alignment
        #             + kl_prior_weight * L_kl   KL between the edge distribution and the prior
        loss = (base_loss + gnn_loss + self.emb_loss_weight * emb_loss
                + self.reg_loss_weight * reg_loss
                + self.cmga_weight * cmga_loss
                + self.kl_prior_weight * kl_loss)

        # Diagnostics run only when the total loss is not finite. Any single non-finite term
        # makes the total non-finite, and the plain message from vse.train_emb does not say
        # which term broke. Here the state of every term is reported once, the non-finite ones
        # are cleaned in place so that the iteration can still backpropagate, and the check is
        # kept inside the if to avoid six GPU synchronisations per step.
        if not torch.isfinite(loss):
            items = [('base_loss', base_loss), ('gnn_loss', gnn_loss),
                     ('emb_loss', emb_loss), ('reg_loss', reg_loss),
                     ('cmga_loss', cmga_loss), ('kl_loss', kl_loss)]
            status, cleaned = [], []
            for name, t in items:
                v = t if torch.is_tensor(t) else torch.tensor(float(t), device=img_emb.device)
                ok = bool(torch.isfinite(v).all().item())
                status.append('%s=%s' % (name, 'OK' if ok else 'NaN/Inf'))
                if not ok:
                    v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                cleaned.append(v)
            logger.error('[NaN][iter %d] the total loss is not finite | term status: %s',
                         self.iter_count, ', '.join(status))
            logger.error('[NaN][iter %d] cleaned values: base=%.4f gnn=%.4f emb=%.4f reg=%.4f cmga=%.4f kl=%.4f',
                         self.iter_count, float(cleaned[0]), float(cleaned[1]),
                         float(cleaned[2]), float(cleaned[3]),
                         float(cleaned[4]), float(cleaned[5]))
            self._nan_terms = getattr(self, '_nan_terms', 0) + 1
            loss = (cleaned[0] + cleaned[1] + 0.4 * cleaned[2]
                    + self.reg_loss_weight * cleaned[3]
                    + self.cmga_weight * cleaned[4]
                    + self.kl_prior_weight * cleaned[5])

        self.iter_count += 1

        return loss

    # ==================================================================
    # Cross-modal graph topology alignment
    # ==================================================================
    @staticmethod
    def _infonce(A_emb, B_emb, pos_mask, temperature, neg_mask=None):
        """Standard InfoNCE with A_emb (B,D) as anchors and B_emb (B,D) as the candidate pool.
        The denominator excludes the columns marked by neg_mask, or only the diagonal when
        neg_mask is None.
        COCO has several captions per image, so the other captions with the same img_id are
        positive samples as well. Excluding the diagonal alone would put positives in the
        denominator, which makes the numerator and the denominator inconsistent.
        """
        B = A_emb.size(0)
        sim = torch.matmul(A_emb, B_emb.t()) / temperature          # (B,B)
        sim = sim - sim.max(dim=-1, keepdim=True)[0].detach()        # numerical stability
        exp_sim = torch.exp(sim)
        if neg_mask is None:
            denom_mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
        else:
            denom_mask = neg_mask.bool()
        denom = (exp_sim * denom_mask.float()).sum(dim=-1, keepdim=True)
        # When a row excludes every candidate, fall back to everything but itself to avoid log(0)
        empty = (denom <= 1e-8)
        if bool(empty.any()):
            eye = ~torch.eye(B, dtype=torch.bool, device=sim.device)
            denom_fb = (exp_sim * eye.float()).sum(-1, keepdim=True)
            denom = torch.where(empty, denom_fb.clamp_min(1e-8), denom)
        log_prob = sim - torch.log(denom + 1e-8)
        loss = -(log_prob * pos_mask.float()).sum() / (pos_mask.float().sum() + 1e-8)
        return loss

    def _project_struct(self, which, g, L):
        """With separate readouts each projection head maps into the shared space; with a shared readout the inputs pass through."""
        if self.cmga_share_encoder:
            return g, L
        proj = self.cmga_proj_text if which == 'text' else self.cmga_proj_visual
        g = l2norm(proj(g), dim=-1)
        if L is not None:
            L = l2norm(proj(L), dim=-1)
        return g, L

    def _cmga_readout(self, which, basis, ctx):
        """Per-sample structural descriptor, returning (g, L) where g is the (B,D) global
        topology description and L the (B,K,D) local topology tokens, or None.
        The controls are cmga_graph_source (dyn or prior) and cmga_detach_graph.
        """
        assert which in ('text', 'visual'), which
        head = (self.concept_enc.adj_head_text if which == 'text'
                else self.concept_enc.adj_head_visual)
        encoder = self.struct_encoder
        if (not self.cmga_share_encoder) and which == 'visual':
            encoder = self.struct_encoder_visual      # separate readout
        if self.cmga_graph_source == 'prior':
            head = _PriorAdjSource(head)              # align on the static prior instead

        want_local = self.cmga_num_local > 0

        if not self.cmga_detach_graph:
            # Chunked, so the peak drops from B to chunk; returns (g,L) when return_local is set
            out = chunked_struct_readout(head, encoder, basis, ctx,
                                        self.cmga_chunk, return_local=want_local)
            if want_local:
                g, L = out
            else:
                g, L = out, None
            return self._project_struct(which, g, L)

        # Detached path: no_grad on the adjacency side, gradients only through basis and projections
        B = ctx.size(0)
        gs, Ls = [], []
        step = self.cmga_chunk if (self.cmga_chunk and 0 < self.cmga_chunk < B) else B
        for s in range(0, B, step):
            e = min(s + step, B)
            with torch.no_grad():
                A = head.forward_batch(basis, ctx[s:e])       # (b,N,N)
            if want_local:
                gi, Li = encoder(A.detach(), basis, return_local=True)
                Ls.append(Li)
            else:
                gi = encoder(A.detach(), basis)
            gs.append(gi)
        g = torch.cat(gs, dim=0)
        L = torch.cat(Ls, dim=0) if want_local else None
        return self._project_struct(which, g, L)


    def _current_cmga_temperature(self):
        """Temperature annealing from cmga_temp_start to cmga_temperature over the first cmga_temp_warmup iterations."""
        p = min(self.iter_count / max(self.cmga_temp_warmup, 1), 1.0)
        return self.cmga_temp_start + (self.cmga_temperature - self.cmga_temp_start) * p

    def _compute_cmga_loss(self, text_basis, visual_basis, cap_emb, img_emb, img_ids):
        """
        Multi-granularity cross-modal graph topology alignment
        ------------------------------------------------------------------
        Per sample, build the text and the visual graph and read them out as a global vector
        g and K local topology tokens L, then align with InfoNCE globally and anchor by
        anchor. A positive pair shares the same img_id and the denominator excludes every
        candidate with that id. The temperature anneals from 0.5 to 0.1, staying smooth while
        the structure is learned and becoming sharper later.
        """
        device = cap_emb.device
        img_ids_1d = img_ids.to(device).view(-1)
        B = cap_emb.size(0)

        same = (img_ids_1d.unsqueeze(1) == img_ids_1d.unsqueeze(0))   # (B,B)
        neg_mask = ~same                                              # denominator excludes every same id

        g_t, L_t = self._cmga_readout('text', text_basis, cap_emb)

        if self.topo_mode == 'intra':
            # Control: compare the text graph with itself, positives excluding the diagonal
            pos_mask = same.clone()
            pos_mask.fill_diagonal_(False)
            ga, gb, La, Lb = g_t, g_t, L_t, L_t
        else:
            pos_mask = same.clone()
            g_v, L_v = self._cmga_readout('visual', visual_basis, img_emb)
            ga, gb, La, Lb = g_t, g_v, L_t, L_v

        if not pos_mask.any():
            return torch.tensor(0.0, device=device)

        temp = self._current_cmga_temperature()

        def _bidir(a, b):
            return 0.5 * (self._infonce(a, b, pos_mask, temp, neg_mask=neg_mask)
                          + self._infonce(b, a, pos_mask, temp, neg_mask=neg_mask))

        loss = _bidir(ga, gb)                              # global topology alignment
        if La is not None and Lb is not None and La.size(1) > 0:
            K = La.size(1)
            loc = 0.0
            for k in range(K):
                loc = loc + _bidir(La[:, k], Lb[:, k])     # local topology alignment, anchor by anchor
            loss = loss + loc / float(K)
        return loss

    # ==================================================================
    # Structure injection from the concept level (CGIG)
    # ==================================================================
    def _fuse_relation(self, R, S):
        """
        Relation matrix fusion, R' = (1-w) R + w * ref(R) * S.
        ref(R) is the maximum row magnitude of R; it maps S, a row-normalised probability
        distribution, onto the scale of R so that the weight w keeps a clear meaning.
        """
        w = self.concept_guide_weight
        if w <= 0:
            return R
        ref = R.abs().max(dim=-1, keepdim=True)[0]
        return (1.0 - w) * R + w * (S * ref)

    # ------------------------------------------------------------------
    # Three ways of reading the concept response P
    #   The plain form is P = softmax( cos(e_i, X_c) * tau / sqrt(D) ) with D = embed_size
    #   equal to 1024. Since std(cos) scales as 1/sqrt(D), std(logit) scales as tau/D, which
    #   is far smaller than the naive reading of tau/sqrt(D) suggests: with tau=10 the logit
    #   spread is about 0.01 over 300 concepts, so P is almost flat. Reaching a logit spread
    #   of order 1.5 would need tau in the thousands, so a sensitivity sweep over small tau
    #   values stays on the plateau and necessarily produces a flat curve.
    # ------------------------------------------------------------------
    @staticmethod
    def _row_zscore(X, eps=1e-6):
        """Standardise each row: (x - mean) / (std + eps).

        After the z-score the row standard deviation is 1, so multiplying by tau gives a
        logit spread of tau independent of D.
        """
        mu = X.mean(dim=-1, keepdim=True)
        sd = X.std(dim=-1, keepdim=True, unbiased=False)
        return (X - mu) / (sd + eps)

    @staticmethod
    def _attn_to_response(attn):
        """Reduce the cross-attention of the concept decoder, (B, n_head, 1, Nc), to the
        concept response P of shape (B, Nc).

        The head dimension is averaged, so the result does not depend on the value of
        n_head_concept_encoder.
        """
        a = attn
        if a is None:
            return None
        if a.dim() == 4:
            a = a.mean(dim=1)                       # (B, 1, Nc)
        if a.dim() == 3:
            a = a[:, 0, :] if a.size(1) == 1 else a.mean(dim=1)
        if a.dim() != 2:
            return None
        a = a.clamp_min(0.0)                        # the attention comes from a softmax, so it is non-negative
        row = a.sum(dim=-1, keepdim=True)
        if not bool(torch.isfinite(row).all()) or bool((row <= 0).any()):
            return None
        return a / row

    def _concept_response(self, Xc, img_emb, cap_emb, resp=None):
        """Return (P_v, P_t) of shape (B, Nc); Xc is the L2 normalised concept basis (Nc, D).

        'cos'    the plain cosine form, the default
        'zscore' softmax( rowzscore(cos) * tau )
        'attn'   the cross-attention of the concept decoder
        Any failure, such as a missing resp or a mismatched batch or Nc, falls back to
        'cos' without raising.
        """
        src = str(getattr(self, 'concept_response_source', 'cos')).lower()
        scale = self.concept_guide_tau / math.sqrt(max(Xc.size(-1), 1))
        cos_v = torch.matmul(l2norm(img_emb, dim=-1), Xc.t())
        cos_t = torch.matmul(l2norm(cap_emb, dim=-1), Xc.t())

        if src == 'zscore':
            tau = self.concept_guide_tau
            return (torch.softmax(self._row_zscore(cos_v) * tau, dim=-1),
                    torch.softmax(self._row_zscore(cos_t) * tau, dim=-1))

        if src == 'attn':
            if isinstance(resp, (tuple, list)) and len(resp) == 2:
                av, at = resp
            else:
                av = at = resp
            pv = self._attn_to_response(av)
            pt = self._attn_to_response(at)
            ok = (pv is not None and pt is not None
                  and pv.size(0) == cos_v.size(0) and pt.size(0) == cos_t.size(0)
                  and pv.size(-1) == cos_v.size(-1) and pt.size(-1) == cos_t.size(-1))
            if ok:
                return pv.to(cos_v.dtype), pt.to(cos_t.dtype)
            if not getattr(self, '_attn_fallback_warned', False):
                self._attn_fallback_warned = True
                logger.warning('[concept_response_source=attn] the attention is unavailable or has the wrong shape '
                               '(P_v=%s vs cos_v=%s), falling back to cos.',
                               None if pv is None else tuple(pv.shape), tuple(cos_v.shape))
            # fall through -> cos

        return (torch.softmax(cos_v * scale, dim=-1),
                torch.softmax(cos_t * scale, dim=-1))

    def _concept_guided_relation(self, concept_basis, concept_assoc, img_emb, cap_emb,
                                 resp=None):
        """
        Concept-guided instance graph
        ------------------------------------------------------------------
        1) The concept association matrix A_c comes from the concept-level dynamic head as a
           row-stochastic measure of the strength between concepts. Self loops are removed,
           since they carry no information between concepts, then the matrix is symmetrised
           and row-normalised.
        2) The concept activation distribution P gives the normalised response of sample i to
           concept node c
                P^i_c = softmax( tau * < normalize(e_i), normalize(X_c) > )
           --concept_response_source can replace this with a z-score or the decoder attention.
        3) The sample-to-sample structure in concept-association space follows from the
           quadratic form, which amounts to diffusion along the concept graph
                S_ij = P^i  A_c  (P^j)^T
        Returns (S_vv, S_tt, S_vt) for the i2i, t2t and i2t blocks.
        """
        Xc = l2norm(concept_basis, dim=-1)                            # (Nc, D)
        P_v, P_t = self._concept_response(Xc, img_emb, cap_emb, resp)

        # ---- Where A_c comes from (--concept_guide_source) ----
        #   'dyn'      A_c generated by the dynamic heads (default)
        #   'prior'    the knowledge prior C instead. If this degrades to the level of
        #              --use_concept_guided_graph 0, the benefit of CGIG depends on the
        #              dynamic graph; if it matches the default, any structured matrix would do.
        #   'identity' A_c = I, so S = rownorm(P P^T) and the quadratic form carries no graph
        #              structure, which tests whether that intermediate level is needed.
        if self.concept_guide_source == 'identity':
            Nc0 = concept_basis.size(0)
            A_c = torch.eye(Nc0, device=concept_basis.device, dtype=concept_basis.dtype)
        elif self.concept_guide_source == 'prior':
            Nc0 = concept_basis.size(0)
            A_c = self.concept_enc.adj_head_all._get_prior(
                Nc0, concept_basis.device, concept_basis.dtype).detach()
        else:
            # Partial gradient pass-through: the gamma share lets the sample-level loss nudge
            # the concept graph while the remaining (1-gamma) is detached, which is stabler
            # than a full detach.
            _gamma = float(getattr(self, 'cgig_soft_gamma', 0.1))
            A_c = _gamma * concept_assoc + (1.0 - _gamma) * concept_assoc.detach()
        A_c = 0.5 * (A_c + A_c.t())                                   # symmetrise, the graph is undirected
        A_c = A_c - torch.diag(torch.diag(A_c))                       # drop self loops, keep only concept-to-concept mass
        A_c = A_c / (A_c.sum(dim=-1, keepdim=True) + 1e-8)            # row normalisation

        def quadratic(P_a, P_b):
            S = torch.matmul(torch.matmul(P_a, A_c), P_b.t())         # (B, B)
            S = S / (S.sum(dim=-1, keepdim=True) + 1e-8)
            # ---- Injection amount lost to the row invariance of softmax ----
            # _four_block passes R' as attn_mask to nn.TransformerEncoder, and softmax is
            # invariant to adding the same constant across a row. A row-normalised S is
            # therefore almost constant per row and the injection carries almost no
            # information.
            #   cgig_demean   subtracts the row mean, so that the within-row differences are
            #                 the only thing injected
            #   cgig_unit_rms rescales to unit RMS; without it the varying part is of order
            #                 CV/B and stays negligible after multiplication by ref(R)
            # Both switches default to 0, in which case nothing is applied.
            if self.cgig_demean:
                S = S - S.mean(dim=-1, keepdim=True)
            if self.cgig_unit_rms:
                S = S / (S.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-12)
            return S

        return quadratic(P_v, P_v), quadratic(P_t, P_t), quadratic(P_v, P_t)

    # ==================================================================
    # Concept response and structure sharpness diagnostics, one log line every N iterations
    #   A single epoch is enough to tell whether CGIG is doing anything:
    #     max/mean of P for text and visual, near 1 means P is flat
    #     row CV of S_vt, very small means S is essentially a constant matrix
    #     visible share = CV^2/(1+CV^2), how much of the injection softmax can see
    #   With both switches off the CV is around 1e-5; with --cgig_demean 1 and
    #   --cgig_unit_rms 1 it should rise to around 1e-1, which shows the path is active.
    # ==================================================================
    @torch.no_grad()
    def _log_concept_sharpness(self, img_emb, cap_emb, concept_nodes, concept_assoc,
                               resp=None):
        def _p_stat(P):
            n = max(P.size(-1), 1)
            mx = P.max(dim=-1).values.mean().item() * n
            rs = P.std(dim=-1, unbiased=False).mean().item()
            return 'max/mean=%.3f rowstd=%.2e' % (mx, rs)

        Xc = l2norm(concept_nodes, dim=-1)
        P_v, P_t = self._concept_response(Xc, img_emb, cap_emb, resp)
        S = self._concept_guided_relation(concept_nodes, concept_assoc,
                                          img_emb, cap_emb, resp)[2]
        cv = (S.std(dim=-1, unbiased=False) / (S.mean(dim=-1).abs() + 1e-12)).mean().item()
        logger.info('[concept sharpness][iter %d] source=%s tau=%.4g demean=%d unit_rms=%d | '
                    'text P %s | visual P %s | row CV of S_vt=%.3e (visible share %.2e)',
                    self.iter_count, self.concept_response_source,
                    self.concept_guide_tau, int(self.cgig_demean),
                    int(self.cgig_unit_rms), _p_stat(P_t), _p_stat(P_v),
                    cv, cv * cv / (1.0 + cv * cv))


if __name__ == '__main__':

    pass


''' Visual self-attention module '''
class V_single_modal_atten(nn.Module):
    """
    Single Visual Modal Attention Network.
    """

    def __init__(self, image_dim, embed_dim, activation_type, dropout_rate):
        """
        param image_dim: dim of visual feature
        param embed_dim: dim of embedding space
        """
        super(V_single_modal_atten, self).__init__()

        self.fc1 = nn.Linear(image_dim, embed_dim)  # embed visual feature to common space

        self.fc2 = nn.Linear(image_dim, embed_dim)  # embed memory to common space
        self.fc2_2 = nn.Linear(embed_dim, embed_dim)

        self.fc3 = nn.Linear(embed_dim, 1)  # turn fusion_info to attention weights
        self.fc4 = nn.Linear(image_dim, embed_dim)  # embed attentive feature to common space


        if activation_type == 'tanh':
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2_2 = nn.Sequential(self.fc2_2,
                                               nn.Tanh(),
                                               nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3,
                                               nn.Dropout(dropout_rate))
        else:
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2_2 = nn.Sequential(self.fc2_2,
                                               nn.BatchNorm1d(embed_dim),
                                               nn.Sigmoid(),
                                               nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3)

        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, v_t, m_v):
        """
        Forward propagation.
        :param v_t: encoded images, shape: (batch_size, num_regions, image_dim)
        :param m_v: previous visual memory, shape: (batch_size, image_dim)
        :return: attention weighted encoding, weights
        """
        W_v = self.embedding_1(v_t)

        if m_v.size()[-1] == v_t.size()[-1]:
            W_v_m = self.embedding_2(m_v)
        else:
            W_v_m = self.embedding_2_2(m_v)

        W_v_m = W_v_m.unsqueeze(1).repeat(1, W_v.size()[1], 1)

        h_v = W_v.mul(W_v_m)

        a_v = self.embedding_3(h_v)
        a_v = a_v.squeeze(2)
        weights = self.softmax(a_v)

        v_att = ((weights.unsqueeze(2) * v_t)).sum(dim=1)

        # l2 norm
        v_att = l2norm((v_att))

        return v_att, weights


''' Textual self-attention module '''
class T_single_modal_atten(nn.Module):
    """
    Single Textual Modal Attention Network.
    """

    def __init__(self, embed_dim, activation_type, dropout_rate):
        """
        param embed_dim: dim of embedding space/ dim of input text features
        """
        super(T_single_modal_atten, self).__init__()

        self.fc1 = nn.Linear(embed_dim, embed_dim)  # embed visual feature to common space
        self.fc2 = nn.Linear(embed_dim, embed_dim)  # embed memory to common space
        self.fc3 = nn.Linear(embed_dim, 1)  # turn fusion_info to attention weights

        if activation_type == 'tanh':
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.Tanh(),
                                             nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3)
        elif activation_type == 'sigmoid':
            self.embedding_1 = nn.Sequential(self.fc1,
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_2 = nn.Sequential(self.fc2,
                                             nn.Sigmoid(),
                                             nn.Dropout(dropout_rate))
            self.embedding_3 = nn.Sequential(self.fc3)

        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, u_t, m_u):
        """
        Forward propagation.
        :param v_t: encoded images, shape: (batch_size, num_regions, image_dim)
        :param m_v: previous visual memory, shape: (batch_size, image_dim)
        :return: attention weighted encoding, weights
        """

        W_u = self.embedding_1(u_t)

        W_u_m = self.embedding_2(m_u)
        W_u_m = W_u_m.unsqueeze(1).repeat(1, W_u.size()[1], 1)

        h_u = W_u.mul(W_u_m)

        a_u = self.embedding_3(h_u)
        a_u = a_u.squeeze(2)
        weights = self.softmax(a_u)

        u_att = ((weights.unsqueeze(2) * u_t)).sum(dim=1)

        # l2 norm
        u_att = l2norm(u_att)

        return u_att, weights




'''KL regularizer for softmax prob distribution'''
class KL_loss_softmax(nn.Module):
    """
    Compute KL_divergence between all prediction score (already sum=1, omit softmax function)
    """
    def __init__(self):
        super(KL_loss_softmax, self).__init__()

        self.KL_loss = nn.KLDivLoss(reduce=False)

    def forward(self, im, s):
        # Clamp to keep log(0) from giving -inf and log of a negative value from giving nan
        im_safe = im.clamp(min=1e-8)
        s_safe = s.clamp(min=1e-8)
        img_prob = torch.log(im_safe)
        s_prob = s_safe
        KL_loss = self.KL_loss(img_prob, s_prob)
        loss = KL_loss.sum()
        # Final guard: replace any remaining nan or inf with 0 so the total loss stays finite
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss

class Consensus_level_feature_learning(nn.Module):
    """
    Consensus-level feature learning module .
    """
    def __init__(self, image_dim, embed_dim, use_bn, activation_type, dropout_rate, attribute_num,
                 no_imgnorm=False, ):
        """
        param image_dim: dim of visual feature
        param embed_dim: dim of embedding space
        """
        super(Consensus_level_feature_learning, self).__init__()

        self.no_imgnorm = no_imgnorm
        self.fc1 = nn.Linear(image_dim, embed_dim)  # embed visual feature to common space
        self.fc2 = nn.Linear(embed_dim, embed_dim)  # embed attribute to common space
        self.fc3 = nn.Linear(embed_dim, 1)  # turn fusion_info to attention weights
        # The ground-truth label has attribute_num dimensions while there are 2*attribute_num
        # concept nodes. Creating the projection inside forward on every call would build a
        # new layer each step with untrained parameters outside the optimiser, and those
        # parameters would never reach state_dict. It is therefore registered once here.
        self.GT_label_proj = nn.Linear(attribute_num, 2 * attribute_num)

        if use_bn == True and activation_type == 'tanh':
            self.embedding_1 = nn.Sequential(
                                             self.fc1,
                                             nn.BatchNorm1d(embed_dim),
                                             nn.Tanh()
                                             )

            self.embedding_2 = nn.Sequential(
                                             self.fc2,
                                             nn.BatchNorm1d(embed_dim),
                                             nn.Tanh()
                                             )
            self.embedding_3 = nn.Sequential(self.fc3)
        elif use_bn == False and activation_type == 'tanh':
            self.embedding_1 = nn.Sequential(
                                             self.fc1,
                                             nn.Tanh()
                                             )
            self.embedding_2 = nn.Sequential(
                                             self.fc2,
                                             nn.Tanh()
                                             )
            self.embedding_3 = nn.Sequential(self.fc3)
        elif use_bn == True and activation_type == 'sigmoid':
            self.embedding_1 = nn.Sequential(
                                             self.fc1,
                                             nn.BatchNorm1d(embed_dim),
                                             nn.Sigmoid()
                                             )

            self.embedding_2 = nn.Sequential(
                                             self.fc2,
                                             nn.BatchNorm1d(embed_dim),
                                             nn.Sigmoid()
                                             )
            self.embedding_3 = nn.Sequential(self.fc3)
        else:
            self.embedding_1 = nn.Sequential(
                                             self.fc1,
                                             nn.Dropout(dropout_rate)
                                             )
            self.embedding_2 = nn.Sequential(
                                             self.fc2,
                                             nn.Dropout(dropout_rate)
                                             )
            self.embedding_3 = nn.Sequential(self.fc3)

        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
        self.smooth_coef = 10


    def forward(self, emb_instance, concept_feature, input_modal, GT_label, GT_label_ratio):
        """
        Forward propagation.
        :param emb_instance: encoded images or text, shape: (batch_size, emb_dim)
        :param concept_feature: concept feature, shape: (att_num, emb_dim)
        :return: emb_concept: consensus-level feature
                 weights_u, weights_v: predicted concept score
        """
        W_s = self.embedding_1(concept_feature) #（600,1014） # (concept_num, emb_dim)

        W_v_m = self.embedding_2(emb_instance) #（2,1024）  # (bs, emb_dim)
        W_v_m = W_v_m.unsqueeze(1).repeat(1, W_s.size()[0], 1) #（2,600,1024）  # (bs, att_num, emb_dim)

        h_s = W_s.mul(W_v_m) #（2,600,1024）   # (bs, concept_num, emb_dim)

        a_s = self.embedding_3(h_s) # (bs, concept_num, 1)
        a_s = a_s.squeeze(2)  #（2,600,1）      # (bs, concept_num)

        weights = self.softmax(a_s * self.smooth_coef)#（2,600）

        if input_modal == 'textual':
            linear_layer = self.GT_label_proj

            GT_label=GT_label.cuda()
            GT_label=linear_layer(GT_label)
            GT_label_scale = self.softmax(GT_label* self.smooth_coef).cuda()
            weights_u = GT_label_ratio * GT_label_scale + (1 - GT_label_ratio) * weights
            concept_feature = l2norm(concept_feature)
            #
            emb_concept = (weights_u.unsqueeze(2) * concept_feature).sum(dim=1)
            #
            if not self.no_imgnorm:
              emb_concept = l2norm(emb_concept)
            return emb_concept, weights_u
        elif input_modal == 'visual':

            weights_v = weights#（2,600）
            concept_feature = l2norm(concept_feature)#（600,1024）

            emb_concept = (weights_v.unsqueeze(2) * concept_feature).sum(dim=1)#（2,1024）
            if not self.no_imgnorm:
                emb_concept = l2norm(emb_concept)
            return emb_concept, weights_v

class Multi_feature_fusing(nn.Module):
    """
    Emb the features from both modalities to the joint attribute label space.
    """

    def __init__(self, embed_dim, fuse_type='weight_sum'):
        """
        param image_dim: dim of visual feature
        param embed_dim: dim of embedding space
        """
        super(Multi_feature_fusing, self).__init__()

        self.fuse_type = fuse_type
        self.embed_dim = embed_dim
        if fuse_type == 'concat':
            input_dim = int(2*embed_dim)
            self.joint_emb_v = nn.Linear(input_dim, embed_dim)
            self.joint_emb_t = nn.Linear(input_dim, embed_dim)
            self.init_weights_concat()
        if fuse_type == 'adap_sum':
            self.joint_emb_v = nn.Linear(embed_dim, 1)
            self.joint_emb_t = nn.Linear(embed_dim, 1)
            self.init_weights_adap_sum()

    def init_weights_concat(self):
        """Xavier initialization"""
        r = np.sqrt(6.) / np.sqrt(self.embed_dim + 2*self.embed_dim)
        self.joint_emb_v.weight.data.uniform_(-r, r)
        self.joint_emb_v.bias.data.fill_(0)
        self.joint_emb_t.weight.data.uniform_(-r, r)
        self.joint_emb_t.bias.data.fill_(0)

    def init_weights_adap_sum(self):
        """Xavier initialization"""
        r = np.sqrt(6.) / np.sqrt(self.embed_dim + 1)
        self.joint_emb_v.weight.data.uniform_(-r, r)
        self.joint_emb_v.bias.data.fill_(0)
        self.joint_emb_t.weight.data.uniform_(-r, r)
        self.joint_emb_t.bias.data.fill_(0)

    def forward(self, v_emb_instance, t_emb_instance, v_emb_concept, t_emb_concept, alpha=0.75):
        """
        Forward propagation.
        :param v_emb_instance, t_emb_instance: instance-level visual or textual features, shape: (batch_size, emb_dim)
        :param v_emb_concept, t_emb_concept: consensus-level concept features, shape: (batch_size, emb_dim)
        :return: joint embbeding features for both modalities
        """
        if self.fuse_type == 'multiple':
            v_fused_emb = v_emb_instance.mul(v_emb_concept);
            v_fused_emb = l2norm(v_fused_emb)
            t_fused_emb = t_emb_instance.mul(t_emb_concept);
            t_fused_emb = l2norm(t_fused_emb)

        elif self.fuse_type == 'concat':
            v_fused_emb = torch.cat([v_emb_instance, v_emb_concept], dim=1)
            v_fused_emb = self.joint_emb_instance_v(v_fused_emb)
            v_fused_emb = l2norm(v_fused_emb)

            t_fused_emb = torch.cat([t_emb_instance, t_emb_concept], dim=1)
            t_fused_emb = self.joint_emb_instance_v(t_fused_emb)
            t_fused_emb = l2norm(t_fused_emb)

        elif self.fuse_type == 'adap_sum':
            v_mean = (v_emb_instance + v_emb_concept) / 2
            v_emb_instance_mat = self.joint_emb_instance_v(v_mean)
            alpha_v = F.sigmoid(v_emb_instance_mat)
            v_fused_emb = alpha_v * v_emb_instance + (1 - alpha_v) * v_emb_concept
            v_fused_emb = l2norm(v_fused_emb)

            t_mean = (t_emb_instance + t_emb_concept) / 2
            t_emb_instance_mat = self.joint_emb_instance_t(t_mean)
            alpha_t = F.sigmoid(t_emb_instance_mat)
            t_fused_emb = alpha_t * t_emb_instance + (1 - alpha_t) * t_emb_concept
            t_fused_emb = l2norm(t_fused_emb)

        elif self.fuse_type == 'weight_sum':
            # alpha = 0.75

            v_fused_emb = alpha * v_emb_instance + (1 - alpha) * v_emb_concept
            v_fused_emb = l2norm(v_fused_emb)
            t_fused_emb = alpha * t_emb_instance + (1 - alpha) * t_emb_concept
            t_fused_emb = l2norm(t_fused_emb)

        return v_fused_emb, t_fused_emb


def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

class GatedFusion(nn.Module):
    """
    Gated adaptive fusion, controlling the mixing ratio between the input feature and the
    enhanced feature per dimension rather than with a single global weight.
        g = sigmoid( Linear( [x; y] ) )
        out = g * y + (1 - g) * x
    x is the base feature, y the enhanced one and g the per-dimension gate in [0,1].
    """
    def __init__(self, embed_size, bias=True, init_gate_bias=-1.0):
        super(GatedFusion, self).__init__()
        self.embed_size = embed_size
        # Gate layer: the two features are concatenated and mapped to a per-dimension gate
        self.gate_linear = nn.Linear(embed_size * 2, embed_size, bias=bias)
        self.sigmoid = nn.Sigmoid()
        # Initialised towards the base feature so that training starts out stable
        if bias:
            nn.init.constant_(self.gate_linear.bias, init_gate_bias)
        nn.init.xavier_uniform_(self.gate_linear.weight)

    def forward(self, x_raw, y_enhance):
        """
        :param x_raw:     (..., D) base feature (img_emb or cap_emb)
        :param y_enhance: (..., D) graph-enhanced feature (concept_v or concept_t)
        :return:          (..., D) fused feature
        """
        # Concatenate both features and compute the per-dimension gate
        cat_feat = torch.cat([x_raw, y_enhance], dim=-1)
        gate = self.sigmoid(self.gate_linear(cat_feat))
        # Residual gated fusion: the gate goes to the enhanced branch, the rest to the base
        out = gate * y_enhance + (1.0 - gate) * x_raw
        return out
