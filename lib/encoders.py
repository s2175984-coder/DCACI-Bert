"""
Image and text encoders
=====================================================================

Local-feature path
    image: Faster-RCNN region features (2048) -> Dropout -> FC_MLP -> MaxK pooling (res)
           -> TransformerEncoder (aggr, with padding mask) -> mean pooling
           -> img = beta*res + (1-beta)*pooled
    text:  BERT (768) -> Dropout -> Linear -> MaxK pooling (res)
           -> TransformerEncoder (aggr, with padding mask) -> mean pooling
           -> cap = beta*res + (1-beta)*pooled

CLIP path, enabled by --use_clip_fusion 1
    clip   = fc_clip(CLIP 512 -> 1024)              # CLIP is frozen, only the projection trains
    pooled = GGLA(clip, local sequence)             # the CLIP global feature is the query
    f      = beta*res + (1-beta)*pooled             # residual fusion
    ------- f up to here is written to *_emb_notnorm -------
    out    = GatedFusion(l2norm(f), l2norm(clip))   # gated fusion, returned as the retrieval feature

Three details of the attention block
    1) MultiHeadAttention normalises both Q and K with F.normalize before the dot
       product, so the scores are cosine similarities, and no padding mask is
       applied. Padded positions therefore still contribute to the scores, which
       is the intended behaviour of this aggregation.
    2) Eight heads. The head count scales memory linearly and can be lowered with
       --clip_mha_heads, but the default stays at 8.
    3) *_emb_notnorm holds the local fused vector f from before the CLIP fusion.

With --use_clip_fusion 0 the else branch runs and the computation graph reduces to
the local-feature path alone.
=====================================================================
"""
import os
import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel
import logging
from lib.mlp import FC_MLP
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# 'True' represents to be masked （Do not participate in the calculation of attention）
# 'False' represents not to be masked
def padding_mask(embs, lengths):

    mask = torch.ones(len(lengths), embs.shape[1], device=lengths.device)
    for i in range(mask.shape[0]):
        end = int(lengths[i])
        mask[i, :end] = 0.

    return mask.bool()


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)


def l2norm(X, dim, eps=1e-8):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk(x, dim, k):
    _x, index = x.topk(k, dim=dim)
    return _x


# uncertain length
def maxk_pool1d_var(x, dim, k, lengths):
    # k >= 1
    results = []
    # assert len(lengths) == x.size(0)

    for idx in range(x.size(0)):
        # keep use all number of features
        k = min(k, int(lengths[idx].item()))

        tmp = torch.split(x[idx], split_size_or_sections=lengths[idx], dim=dim-1)[0]

        max_k_i = maxk_pool1d(tmp, dim-1, k)
        results.append(max_k_i)

    # construct with the batch
    results = torch.stack(results, dim=0)

    return results


def avg_pool1d_var(x, dim, lengths):

    results = []
    # assert len(lengths) == x.size(0)

    for idx in range(x.size(0)):

        # keep use all number of features
        tmp = torch.split(x[idx], split_size_or_sections=lengths[idx], dim=dim-1)[0]
        avg_i = tmp.mean(dim-1)

        results.append(avg_i)

    # construct with the batch
    results = torch.stack(results, dim=0)

    return results


class Maxk_Pooling_Variable(nn.Module):
    def __init__(self, dim=1, k=2):
        super(Maxk_Pooling_Variable, self).__init__()

        self.dim = dim
        self.k = k

    def forward(self, features, lengths):

        pool_weights = None
        pooled_features = maxk_pool1d_var(features, dim=self.dim, k=self.k, lengths=lengths)

        return pooled_features, pool_weights


class Avg_Pooling_Variable(nn.Module):
    def __init__(self, dim=1):
        super(Avg_Pooling_Variable, self).__init__()

        self.dim = dim

    def forward(self, features, lengths):

        pool_weights = None
        pooled_features = avg_pool1d_var(features, dim=self.dim, lengths=lengths)

        return pooled_features, pool_weights


# ======================================================================
# CLIP wrapper used for on-the-fly encoding.
#   The weight path is configurable through --clip_model_path.
#   CLIP runs entirely under requires_grad=False and torch.no_grad(), so any gain
#   cannot be attributed to fine-tuning a strong backbone, which keeps the
#   comparison with local-feature-only methods fair.
# ======================================================================
class ClipEncoder(object):
    """Load open_clip ViT-B/32 on demand, shared by the DataLoader (tokenizer and
    preprocess) and the encoder (encode_image/encode_text). Built once per process."""

    DEFAULT_MODEL_PATH = '/root/autodl-tmp/data/CLIP/open_clip_pytorch_model.bin'

    def __init__(self, model_path=None, device=None, verbose=True, encode_chunk=0):
        super(ClipEncoder, self).__init__()
        import open_clip

        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.arch = 'ViT-B-32'
        # Chunk size for the per-sample forward pass; 0 or negative disables chunking
        self.encode_chunk = int(encode_chunk or 0)

        # open_clip builds differ slightly in their kwargs, so fall back one by one
        last_err = None
        for kwargs in ({'pretrained': self.model_path, 'load_weights_only': False},
                       {'pretrained': self.model_path},
                       {'pretrained': self.model_path, 'load_weights_only': True}):
            try:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    self.arch, **kwargs)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
        else:
            raise RuntimeError('cannot load CLIP weights {}: {}'.format(self.model_path, last_err))

        self.clip_tokenizer = open_clip.get_tokenizer(self.arch)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        if verbose:
            logger.info('CLIP loaded: %s (arch=%s, device=%s, frozen)',
                        self.model_path, self.arch, self.device)

    def get_clip_image_encoder(self):
        return self.model.encode_image

    def get_clip_text_encoder(self):
        return self.model.encode_text

    # ---- Per-sample chunked forward pass (mathematically equivalent, lower peak memory) ----
    # Every ViT operator (LayerNorm, attention, MLP) acts within a single sample and
    # samples never interact, and there is no cross-sample statistic such as BatchNorm, so
    #           f(x[0:B])  ==  cat(f(x[0:c]), f(x[c:2c]), ...)
    # holds exactly. The only difference is that cuBLAS may pick a different kernel for a
    # different batch shape (an effect of order 1e-6 in the last digits), while the peak
    # drops from O(B) to O(chunk).
    # Chunking exists only to avoid running out of memory at batch_size=128 and does not
    # change the feature of any sample. --clip_encode_chunk 0 restores the full-batch pass.
    def _encode(self, fn, x, chunk=None):
        n = int(x.shape[0])
        c = self.encode_chunk if chunk is None else int(chunk or 0)
        if not c or c <= 0 or n <= c:
            return fn(x)
        return torch.cat([fn(x[i:i + c]) for i in range(0, n, c)], dim=0)

    def encode_image(self, images, chunk=None):
        return self._encode(self.model.encode_image, images, chunk)

    def encode_text(self, texts, chunk=None):
        return self._encode(self.model.encode_text, texts, chunk)


def build_clip_encoder(opt, verbose=True):
    """Build a ClipEncoder from opt, or return None when CLIP fusion is disabled."""
    if not bool(getattr(opt, 'use_clip_fusion', 0)):
        return None
    src = str(getattr(opt, 'clip_source', 'online')).lower()
    if src != 'online':
        return None
    return ClipEncoder(model_path=getattr(opt, 'clip_model_path', None) or None,
                       verbose=verbose,
                       encode_chunk=int(getattr(opt, 'clip_encode_chunk', 0) or 0))


def get_text_encoder(opt, embed_size, no_txtnorm=False, clip_encoder=None):

    text_encoder = EncoderText_BERT(opt, embed_size, no_txtnorm=no_txtnorm,
                                    clip_encoder=clip_encoder)

    return text_encoder


def get_image_encoder(opt, img_dim, embed_size, no_imgnorm=False, clip_encoder=None):

    img_enc = EncoderImageAggr(opt, img_dim, embed_size, no_imgnorm,
                               clip_encoder=clip_encoder)

    return img_enc


# ======================================================================
# Components of the CLIP global-semantics branch
#   GGLA = Global semantics-Guided adaptive Local feature Aggregation
# ======================================================================
class MultiHeadAttention(nn.Module):
    """Globally guided local aggregation (GGLA).

    The CLIP global feature acts as the query and the local features (regions or
    words) as key and value, so the pooling weights are predicted from the global
    semantics instead of being fixed as in mean or MaxK pooling.

    Both Q and K are normalised with F.normalize before the dot product and no mask
    is applied by default. region_mask stays optional and is only passed when
    --clip_use_mask 1, which allows the masked variant to be compared.
    """

    def __init__(self, feature_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = max(1, int(num_heads))
        self.feature_dim = feature_dim

        self.query = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(self.num_heads)])
        self.key = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(self.num_heads)])
        self.value = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(self.num_heads)])

    def forward(self, clip_emb, region_emb, region_mask=None):
        """
        :param clip_emb:   (B, D)    global feature after the CLIP projection
        :param region_emb: (B, N, D) local features (regions or words)
        :param region_mask:(B, N) bool, True marks padding; None means no mask is applied
        :return:           (B, D) attended local feature
        """
        heads = []
        for i in range(self.num_heads):
            Q = self.query[i](clip_emb)
            K = self.key[i](region_emb)
            V = self.value[i](region_emb)

            Q = F.normalize(Q, p=2, dim=-1)
            K = F.normalize(K, p=2, dim=-1)

            attention_scores = torch.bmm(K, Q.unsqueeze(-1)).squeeze(-1)
            if region_mask is not None:
                attention_scores = attention_scores.masked_fill(region_mask, float('-inf'))
            attention_weights = F.softmax(attention_scores, dim=-1)
            # An all-padding row can only occur with a mask and yields NaN, so zero it out
            attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

            weighted_regions = torch.bmm(attention_weights.unsqueeze(-1).transpose(2, 1), V).squeeze(1)

            heads.append(weighted_regions)

        return torch.mean(torch.stack(heads), dim=0)


class GatedFusion(nn.Module):
    """Gated fusion of two feature streams, g * feat1 + (1 - g) * feat2.

    fc_gate keeps the default PyTorch initialisation, with no hand-set bias.
    gate_bias only takes effect when it is explicitly non-zero, which allows the
    effect of biasing the gate towards either stream to be measured.
    """

    def __init__(self, embed_dim, gate_bias=0.0):
        super(GatedFusion, self).__init__()
        self.fc_gate = nn.Linear(embed_dim * 2, embed_dim)
        if gate_bias != 0.0:
            nn.init.constant_(self.fc_gate.bias, float(gate_bias))

    def forward(self, feat1, feat2):
        # Concatenate the two feature vectors
        concat_feat = torch.cat([feat1, feat2], dim=-1)  # (B, 2*embed_dim)

        # Predict the gate value with a feed-forward network
        gate_values = torch.sigmoid(self.fc_gate(concat_feat))  # (B, embed_dim)

        # Weighted sum of the two feature vectors
        fused_feat = gate_values * feat1 + (1 - gate_values) * feat2  # (B, embed_dim)

        return fused_feat


class ClipGlobalBranch(nn.Module):
    """CLIP global branch, in the order projection -> GGLA -> residual -> gated fusion.

    Public contract:
        out = branch(local_res, local_seq, clip_raw, residual_weight)
    forward_with_notnorm(...) additionally returns the local vector from before the
    CLIP fusion, which the encoder stores in *_emb_notnorm.
    """

    def __init__(self, opt, embed_size):
        super(ClipGlobalBranch, self).__init__()
        self.opt = opt
        self.clip_dim = int(getattr(opt, 'clip_dim', 512))
        self.fc_clip = FC_MLP(self.clip_dim, embed_size, embed_size, 1, bn=False)
        self.fc_clip.apply(init_weights)
        # 8 heads by default; the head count scales memory linearly and --clip_mha_heads can lower it
        self.mha_pool = MultiHeadAttention(
            embed_size, num_heads=int(getattr(opt, 'clip_mha_heads', 8)))
        self.gated_fusion = GatedFusion(embed_size,
                                        gate_bias=float(getattr(opt, 'clip_gate_bias', 0.0)))
        # No mask by default
        self.use_mask = bool(getattr(opt, 'clip_use_mask', 0))

    def forward_with_notnorm(self, local_res, local_seq, clip_raw, residual_weight,
                             local_mask=None, normalize=True):
        """Run projection -> GGLA -> residual fusion -> optional L2 -> gated fusion.

        :param normalize: whether to L2 normalise both inputs of the gated fusion,
                          equivalent to `if not self.no_imgnorm`. Default True.
        :return: (out, local_notnorm)
                 out           = retrieval feature after the gated fusion
                 local_notnorm = local vector from before the CLIP fusion and before
                                 L2 normalisation, matching the *_emb_notnorm convention
        """
        clip_emb = self.fc_clip(clip_raw)                                  # (B, D)
        mask = local_mask if self.use_mask else None
        pooled = self.mha_pool(clip_emb, local_seq, region_mask=mask)      # GGLA
        local_fusion = residual_weight * local_res + (1.0 - residual_weight) * pooled

        # notnorm keeps the local vector from before the normalisation
        local_notnorm = local_fusion
        if normalize:
            local_fusion = l2norm(local_fusion, dim=-1)
            clip_emb = l2norm(clip_emb, dim=-1)

        out = self.gated_fusion(local_fusion, clip_emb)
        return out, local_notnorm

    def forward(self, local_res, local_seq, clip_raw, residual_weight, local_mask=None,
                normalize=True):
        return self.forward_with_notnorm(local_res, local_seq, clip_raw, residual_weight,
                                         local_mask, normalize)[0]


class EncoderImageAggr(nn.Module):
    """Image encoder, region features -> FC -> Transformer aggregation -> residual fusion.
    The optional CLIP branch adds globally guided local aggregation (GGLA) and gated fusion.
    """

    def __init__(self, opt, img_dim=2048, embed_size=1024, no_imgnorm=False,
                 clip_encoder=None):
        super(EncoderImageAggr, self).__init__()

        self.opt = opt

        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm

        # B * N * 2048 -> B * N * 1024
        # N = 36 for region features
        self.fc = FC_MLP(img_dim, embed_size // 2, embed_size, 2, bn=True)
        self.fc.apply(init_weights)

        # fragment-level relation modeling (for local features)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=opt.nhead,
                                                   dim_feedforward=embed_size, dropout=opt.dropout)
        self.aggr = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=None)

        # pooling function
        self.graph_pool = Avg_Pooling_Variable()
        self.gpool = Maxk_Pooling_Variable()
        self.dropout = nn.Dropout(0.1)

        # ===== CLIP global branch, disabled by default =====
        self.use_clip_fusion = bool(getattr(opt, 'use_clip_fusion', 0))
        self.clip_source = str(getattr(opt, 'clip_source', 'online')).lower()
        if self.use_clip_fusion:
            self.clip_branch = ClipGlobalBranch(opt, embed_size)
            # Encoding raw images needs the CLIP model itself, passed in as a shared instance in online mode
            if self.clip_source == 'online':
                self.clip_encoder = clip_encoder if clip_encoder is not None \
                    else ClipEncoder(model_path=getattr(opt, 'clip_model_path', None) or None,
                                     verbose=False)

    # ---- CLIP global feature: encoded from raw images in online mode, loaded otherwise ----
    def _clip_global_from_image(self, clip_emb, full_image):
        if clip_emb is not None:
            return clip_emb
        if full_image is None:
            return None
        with torch.no_grad():
            # ClipEncoder.encode_image chunks samples according to --clip_encode_chunk, which
            # is mathematically equivalent to a full-batch pass but lowers the peak from O(B) to O(chunk).
            if hasattr(self.clip_encoder, 'encode_image'):
                return self.clip_encoder.encode_image(full_image)
            return self.clip_encoder.get_clip_image_encoder()(full_image)

    def forward(self, images, image_lengths, graph=False, clip_emb=None, full_image=None):
        images = self.dropout(images)
        img_emb = self.fc(images)

        # initial visual embedding (MaxK over valid regions)
        img_emb_res, _ = self.gpool(img_emb, image_lengths)

        img_emb_pre_pool = img_emb

        # get padding mask
        src_key_padding_mask = padding_mask(img_emb, image_lengths)

        # switch the dim
        img_emb = img_emb.transpose(1, 0)
        img_emb = self.aggr(img_emb, src_key_padding_mask=src_key_padding_mask)
        img_emb = img_emb.transpose(1, 0)

        clip_raw = self._clip_global_from_image(clip_emb, full_image) \
            if self.use_clip_fusion else None

        if clip_raw is not None:
            # ---- CLIP path ----
            # *_notnorm = local vector from before the CLIP fusion and before L2 normalisation
            img_emb, img_emb_notnorm = self.clip_branch.forward_with_notnorm(
                img_emb_res, img_emb, clip_raw, self.opt.residual_weight,
                local_mask=src_key_padding_mask,
                normalize=not self.no_imgnorm)
        else:
            # ---- Local-feature path without CLIP ----
            img_emb, _ = self.graph_pool(img_emb, image_lengths)
            # the final global embedding
            img_emb = self.opt.residual_weight * img_emb_res + (1 - self.opt.residual_weight) * img_emb

            img_emb_notnorm = img_emb
            if not self.no_imgnorm:
                img_emb = l2norm(img_emb, dim=-1)

        if graph:
            return images, image_lengths, img_emb, img_emb_notnorm, img_emb_pre_pool
        else:
            return img_emb


# Language Model with BERT backbone
class EncoderText_BERT(nn.Module):
    """Text encoder, BERT -> Linear -> Transformer aggregation -> residual fusion.
    The optional CLIP branch adds globally guided local aggregation (GGLA) and gated fusion.
    """

    def __init__(self, opt, embed_size=1024, no_txtnorm=False, clip_encoder=None):
        super(EncoderText_BERT, self).__init__()

        self.opt = opt

        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        bert_path = getattr(opt, 'bert_path', '/root/autodl-tmp/data/bert-base-uncased')
        self.bert = BertModel.from_pretrained(bert_path)

        # ===== BERT activation checkpointing =====
        # Why: the activation memory of BERT grows with batch_size times the longest
        # sequence in the batch, and caption lengths in f30k have a heavy tail, so a
        # single long batch at batch_size=128 can need about 10 GB for BERT alone.
        # How: each layer keeps only its input tensor and recomputes its forward pass
        # during the backward pass, which reduces the activation memory from O(depth)
        # to O(1) at the cost of one extra encoder forward pass (about 20-30% slower).
        # Numerical effect: torch.utils.checkpoint restores the RNG state before
        # recomputation by default (preserve_rng_state=True), so the same dropout masks
        # are drawn again and the gradients are identical to the non-checkpointed run.
        # It saves memory without changing the optimisation semantics.
        self.grad_ckpt = bool(int(getattr(opt, 'grad_ckpt', 0)))
        if self.grad_ckpt:
            try:
                # Some transformers versions warn or refuse with use_cache=True, so disable it
                self.bert.config.use_cache = False
            except Exception:
                pass
            try:
                self.bert.gradient_checkpointing_enable()
            except TypeError:
                self.bert.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={'preserve_rng_state': True})

        # backbone features -> embbedings
        self.linear = nn.Linear(768, embed_size)

        # relation modeling for local feature
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=opt.nhead,
                                                   dim_feedforward=embed_size, dropout=opt.dropout)
        self.aggr = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=None)

        # pooling function
        self.graph_pool = Avg_Pooling_Variable()
        self.gpool = Maxk_Pooling_Variable()
        self.dropout = nn.Dropout(0.1)

        # ===== CLIP global branch, disabled by default =====
        self.use_clip_fusion = bool(getattr(opt, 'use_clip_fusion', 0))
        self.clip_source = str(getattr(opt, 'clip_source', 'online')).lower()
        if self.use_clip_fusion:
            self.clip_branch = ClipGlobalBranch(opt, embed_size)
            # Built lazily as on the image side, see _ensure_clip_encoder
            self.clip_encoder = clip_encoder

    def _ensure_clip_encoder(self):
        if getattr(self, 'clip_encoder', None) is None:
            self.clip_encoder = ClipEncoder(
                model_path=getattr(self.opt, 'clip_model_path', None) or None, verbose=False)
        return self.clip_encoder

    def _clip_global_from_text(self, clip_emb, clip_tokens):
        if clip_emb is not None:
            return clip_emb
        if clip_tokens is None:
            return None
        with torch.no_grad():
            # Chunked per-sample forward pass as on the image side; padding tokens do not
            # take part in the pooling, so chunking does not change any sentence feature
            ce = self._ensure_clip_encoder()
            if hasattr(ce, 'encode_text'):
                return ce.encode_text(clip_tokens)
            return ce.get_clip_text_encoder()(clip_tokens)

    def forward(self, x, lengths, graph=False, clip_emb=None, clip_tokens=None):

        # Embed word ids to vectors
        # pad 0 for redundant tokens in previous process
        bert_attention_mask = (x != 0).float()

        # all hidden features, D=768 in bert-base model
        bert_emb = self.bert(input_ids=x, attention_mask=bert_attention_mask)[0]  # B x N x D
        bert_emb = self.dropout(bert_emb)
        cap_len = lengths

        # B x N x embed_size
        cap_emb = self.linear(bert_emb)

        # initial textual embedding
        cap_emb_res, _ = self.gpool(cap_emb, cap_len)

        cap_emb_pre_pool = cap_emb

        # fragment-level relation modeling for word features
        # get padding mask
        src_key_padding_mask = padding_mask(cap_emb, cap_len)

        # switch the dim
        cap_emb = cap_emb.transpose(1, 0)
        cap_emb = self.aggr(cap_emb, src_key_padding_mask=src_key_padding_mask)
        cap_emb = cap_emb.transpose(1, 0)

        clip_raw = self._clip_global_from_text(clip_emb, clip_tokens) \
            if self.use_clip_fusion else None

        if clip_raw is not None:
            # ---- CLIP path ----
            # As on the image side, *_notnorm is the local vector from before the CLIP fusion and L2
            cap_emb, cap_emb_notnorm = self.clip_branch.forward_with_notnorm(
                cap_emb_res, cap_emb, clip_raw, self.opt.residual_weight,
                local_mask=src_key_padding_mask,
                normalize=not self.no_txtnorm)
        else:
            # ---- Local-feature path without CLIP ----
            cap_emb, _ = self.graph_pool(cap_emb, cap_len)
            cap_emb = self.opt.residual_weight * cap_emb_res + (1 - self.opt.residual_weight) * cap_emb

            # the final global embedding
            cap_emb_notnorm = cap_emb
            if not self.no_txtnorm:
                cap_emb = l2norm(cap_emb, dim=-1)

        if graph:
            return bert_emb, cap_len, cap_emb, cap_emb_notnorm, cap_emb_pre_pool
        else:
            return cap_emb


if __name__ == '__main__':

    pass
