import arguments
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init
from torch.nn.utils import clip_grad_norm_

from lib.encoders import get_image_encoder, get_text_encoder
from lib.loss import *
from graph_lib import *


logger = logging.getLogger(__name__)


def compute_contrastive_loss(view1, view2, temperature=0.1):
    sims = view1.mm(view2.t())
    mask = torch.eye(sims.size(0)).bool().cuda()

    l_pos = sims[mask].unsqueeze(-1)
    l_neg = sims[~mask].view(sims.shape[0], -1)

    logits = torch.cat([l_pos, l_neg], dim=1)

    logits /= temperature

    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

    loss = torch.nn.functional.cross_entropy(logits, labels)

    return loss


class VSEModel(object):
    def __init__(self, opt, eval=False, clip_encoder=None):

        self.opt = opt
        self.grad_clip = opt.grad_clip

        # CLIP branch settings, initialised before the encoders are built.
        # use_clip_fusion : 1 enables the global CLIP branch; 0 disables the whole
        #                   branch, so the graph is identical to the local-only model.
        # clip_source     : 'online' encodes raw images / tokens inside the encoder,
        #                   'npy' consumes pre-computed 512-d vectors.
        # clip_encoder    : shared ClipEncoder built by train.py. It is None when
        #                   clip_source='npy' or use_clip_fusion=0, in which case the
        #                   encoders run the local path only and need no CLIP weights.
        self.use_clip_fusion = bool(getattr(opt, 'use_clip_fusion', 0))
        self.clip_source = getattr(opt, 'clip_source', 'online')
        self.clip_encoder = clip_encoder

        # Route the dynamic concept graph through the inference path (off by default).
        # The checkpoint must contain the concept-graph weights (see state_dict),
        # otherwise the graph stays randomly initialised and retrieval collapses.
        self.use_graph_inference = bool(getattr(opt, 'use_graph_inference', 0))

        # Route the sample-level four-block module through the inference path
        # (off by default). This branch builds sample-to-sample relations inside a
        # batch, so its output depends on batch composition and departs from the
        # per-sample evaluation protocol. Intended for exploratory analysis only.
        self.use_instance_inference = bool(getattr(opt, 'use_instance_inference', 0))

        # Image / text encoders. With clip_source='online' the shared ClipEncoder is
        # passed in and the encoders call encode_image / encode_text under no_grad,
        # keeping CLIP frozen.
        self.img_enc = get_image_encoder(opt, opt.img_dim, opt.embed_size,
                                         no_imgnorm=opt.no_imgnorm,
                                         clip_encoder=self.clip_encoder)
        self.txt_enc = get_text_encoder(opt, opt.embed_size,
                                        no_txtnorm=opt.no_txtnorm,
                                        clip_encoder=self.clip_encoder)

        if not eval:
            self.criterion = GraphLoss(opt)
            print(self.criterion)

            # Set up the lr for different parts of the VSE model
            decay_factor = 1e-4
            criterion_lr_factor = opt.graph_lr_factor

            self.params = list(self.txt_enc.parameters()) + list(self.img_enc.parameters()) + list(self.criterion.parameters())

            all_text_params = list(self.txt_enc.parameters())
            bert_params = list(self.txt_enc.bert.parameters())

            # Tensor.data_ptr() → int, Returns the address of the first element
            bert_params_ptr = [p.data_ptr() for p in bert_params]
            text_params_no_bert = list()

            # select other parameters except BERT
            for p in all_text_params:
                if p.data_ptr() not in bert_params_ptr:
                    text_params_no_bert.append(p)

            self.optimizer = torch.optim.AdamW([
                {'params': text_params_no_bert, 'lr': opt.learning_rate},
                {'params': bert_params, 'lr': opt.learning_rate * 0.1},
                {'params': self.img_enc.parameters(), 'lr': opt.learning_rate},
                {'params': self.criterion.parameters(), 'lr': opt.learning_rate * criterion_lr_factor},
            ],
                lr=opt.learning_rate, weight_decay=decay_factor)
    
        else:
            # Inference mode only needs ContrastiveLoss by default. GraphLoss must be
            # constructed when use_graph_inference or use_instance_inference is set,
            # otherwise the graph modules have no weights to load.
            if self.use_graph_inference or self.use_instance_inference:
                self.criterion = GraphLoss(opt)
            else:
                self.criterion = ContrastiveLoss(opt)
        
        # iteration
        self.Eiters = 0
        self.data_parallel = False

        # use the gpu
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.criterion.cuda()
            torch.backends.cudnn.benchmark = True

    def set_max_violation(self, max_violation=True):
        """Enable or disable hard negative mining.

        The switch is forwarded to every loss term that implements
        max_violation_on / max_violation_off. A loss term that does not implement
        them logs a single warning instead of silently ignoring the request.
        """
        targets = []
        for name in ('base_loss', 'gnn_loss'):
            crit = getattr(self.criterion, name, None)
            if crit is not None:
                targets.append((name, crit))
        if not targets:
            targets = [('criterion', self.criterion)]

        for name, crit in targets:
            fn = getattr(crit, 'max_violation_on' if max_violation else 'max_violation_off', None)
            if fn is None:
                if not hasattr(self, '_mv_warned'):
                    self._mv_warned = set()
                if name not in self._mv_warned:
                    self._mv_warned.add(name)
                    logger.warning(
                        "set_max_violation: %s(%s) does not implement "
                        "max_violation_on/off, so hard negatives stay disabled "
                        "for this loss term.",
                        name, type(crit).__name__)
                continue
            fn()

    def state_dict(self):
        """Return the three checkpoints of the model.

          [0] img_enc   [1] txt_enc   [2] criterion

        The third entry holds the GraphLoss weights (concept graph, sample-level
        blocks and structure reader), which are required for resuming training and
        for explaining or visualising the learned graph.
        """
        state_dict = [
            self.img_enc.state_dict(),
            self.txt_enc.state_dict(),
            ]
        # Append the third entry only when the criterion holds concept-graph modules.
        if hasattr(self.criterion, 'concept_enc'):
            state_dict.append(self.criterion.state_dict())
        return state_dict

    def load_state_dict(self, state_dict, ):
        # strict=True, ensure keys match
        self.img_enc.load_state_dict(state_dict[0], strict=True)

        # Unexpected key(s) in state_dict: "bert.embeddings.position_ids".
        # incompatible problem of transformers package version
        self.txt_enc.load_state_dict(state_dict[1], strict=False)

        # (3) Concept-graph weights; checkpoints without them are still accepted.
        need_graph = hasattr(self.criterion, 'concept_enc')
        has_graph_ckpt = len(state_dict) > 2
        if need_graph and has_graph_ckpt:
            self.criterion.load_state_dict(state_dict[2], strict=False)
        elif need_graph and not has_graph_ckpt:
            logger.warning(
                'checkpoint has no concept-graph weights, so concept_enc and '
                'AdjacencyModel stay randomly initialised and results obtained with '
                'use_graph_inference enabled are not meaningful.')

    def train_start(self):
        self.img_enc.train()
        self.txt_enc.train()
        self.criterion.train()

    def val_start(self):
        self.img_enc.eval()
        self.txt_enc.eval()
        self.criterion.eval()

    def make_data_parallel(self):
        self.img_enc = nn.DataParallel(self.img_enc)
        self.txt_enc = nn.DataParallel(self.txt_enc)
        self.data_parallel = True
        logger.info('Image/Text encoder is data paralleled (use multi GPUs).')

    @property
    def is_data_parallel(self):
        return self.data_parallel

    # Map the CLIP payload produced by the DataLoader to encoder keyword arguments.
    def _clip_kwargs(self, clip_imgs, clip_caps):
        """Entries 9 and 10 of a batch carry the CLIP payload, whose meaning is
        determined by --clip_source:

        online : payload = raw images (B,3,224,224) / CLIP tokens (B,77),
                 forwarded as full_image / clip_tokens and encoded inside the encoder.
        npy    : payload = pre-computed vectors (B,512), forwarded as clip_emb.

        Returns (img_kwargs, txt_kwargs).
        """
        if not self.use_clip_fusion or (clip_imgs is None and clip_caps is None):
            return {}, {}
        if self.clip_source == 'online':
            return {'full_image': clip_imgs}, {'clip_tokens': clip_caps}
        return {'clip_emb': clip_imgs}, {'clip_emb': clip_caps}

    # Compute the image and caption embeddings
    def forward_emb(self, images, captions, img_ids, lengths, image_lengths=None, graph=None,
                    clip_imgs=None, clip_caps=None, patch_feats=None):
        """
        :param graph: whether the dynamic concept graph takes part in this forward
                      pass; None falls back to self.use_graph_inference. When enabled
                      the returned features are concept-enhanced and live in the same
                      space as emb_loss during training, so inference matches training.
                      The concept node set is fixed, hence this path is independent of
                      batch composition.

        When self.use_instance_inference is set, the features enhanced by the
        sample-level four-block module (four AdjacencyModel matrices, concept gating
        and the GNN) are used for retrieval. The pipeline is
        encoder -> four-block module -> concept gating -> retrieval feature. This path
        depends on batch composition and is therefore not per-sample independent.

        The meaning of clip_imgs / clip_caps is determined by --clip_source (raw
        images and tokens, or pre-computed vectors). Both cases are per-sample
        independent and do not affect the evaluation protocol.
        """
        use_graph = self.use_graph_inference if graph is None else bool(graph)
        use_instance = self.use_instance_inference and hasattr(self.criterion, 'instance_enhance')

        images = images.cuda()
        image_lengths = image_lengths.cuda()
        captions = captions.cuda()
        lengths = lengths.cuda()
        if clip_imgs is not None:
            clip_imgs = clip_imgs.cuda()
        if clip_caps is not None:
            clip_caps = clip_caps.cuda()
        # Patch features (B, P, 512); None when the patch branch is disabled.
        if patch_feats is not None:
            patch_feats = patch_feats.cuda()

        ikw, tkw = self._clip_kwargs(clip_imgs, clip_caps)

        if use_instance:
            # The four-block relations need pre-pooled local features (regions and
            # words), so the five-element output of graph=True is requested.
            img_out = self.img_enc(images, image_lengths, graph=True, **ikw)
            # The raw image (B,3,224,224) is consumed once by the image encoder.
            # Dropping the reference releases roughly 77 MB at B=128, which lowers
            # the peak memory.
            ikw.pop('full_image', None)
            clip_imgs = None
            cap_out = self.txt_enc(captions, lengths, graph=True, **tkw)
            ie = self.criterion.instance_enhance(img_out, cap_out, patch=patch_feats)
            img_emb, cap_emb = ie['emb_v'], ie['emb_t']
        else:
            img_emb = self.img_enc(images, image_lengths, **ikw)
            ikw.pop('full_image', None)      # release the raw image as above
            clip_imgs = None
            cap_emb = self.txt_enc(captions, lengths, **tkw)

        # Route the dynamic concept graph into the inference path.
        if use_graph and hasattr(self.criterion, 'concept_enhance'):
            ce = self.criterion.concept_enhance(img_emb, cap_emb, need_kl=False,
                                                patch=patch_feats)
            return ce['emb_v'], ce['emb_t']

        return img_emb, cap_emb

    # One training step given images and captions

    def train_emb(self, images, captions, lengths, concept_labels, image_lengths=None, img_ids=None,
                  clip_imgs=None, clip_caps=None, patch_feats=None):

        self.Eiters += 1
        self.logger.update('Iter', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # With --mem_debug N enabled the peak statistics are reset at the start of
        # each iteration, so the peak printed by GraphLoss._mem refers to the
        # current forward pass rather than the historical maximum.
        if int(getattr(self.opt, 'mem_debug', 0)) > 0 \
                and self.Eiters <= int(self.opt.mem_debug) \
                and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # compute images embs
        images = images.cuda()
        image_lengths = image_lengths.cuda()
        if clip_imgs is not None:
            clip_imgs = clip_imgs.cuda()
        # compute caption embs
        captions = captions.cuda()
        lengths = lengths.cuda()
        if clip_caps is not None:
            clip_caps = clip_caps.cuda()
        # Patch features (B, P, 512); None when the patch branch is disabled.
        if patch_feats is not None:
            patch_feats = patch_feats.cuda()

        ikw, tkw = self._clip_kwargs(clip_imgs, clip_caps)

        img_emb = self.img_enc(images, image_lengths, graph=True, **ikw)
        # As in forward_emb, release the raw image as soon as it is consumed.
        ikw.pop('full_image', None)
        clip_imgs = None
        cap_emb = self.txt_enc(captions, lengths, graph=True, **tkw)

        img_feat, img_len, img_view1, img_emb_notnorm, img_emb_pre_pool = img_emb
        cap_feat, cap_len, cap_view1, cap_emb_notnorm, cap_emb_pre_pool = cap_emb
        #img_feat, img_len, img_view2, img_emb_notnorm, img_emb_pre_pool = img_emb1
        #cap_feat, cap_len, cap_view2, cap_emb_notnorm, cap_emb_pre_pool = cap_emb1
        self.optimizer.zero_grad()

        #loss1 = compute_contrastive_loss(img_view1, img_view2) + compute_contrastive_loss(cap_view1, cap_view2)
        # compute loss
        loss = self.criterion(img_emb, cap_emb, concept_labels, img_ids=img_ids,
                              patch=patch_feats)

        self.logger.update('Loss', loss.item(), self.opt.batch_size)

        # compute gradient and update
        # Guard against non-finite losses. torch.isfinite covers both nan and inf,
        # and the iteration index is logged so the event can be traced.
        if not torch.isfinite(loss):
            logger.error("[NaN] iteration %d: loss is not finite, batch skipped",
                         self.Eiters)
            self.optimizer.zero_grad(set_to_none=True)
            return 0.

        loss.backward()

        # Guard against non-finite gradients. clip_grad_norm_ cannot handle nan or
        # inf: the norm it returns is itself nan and is then multiplied back into
        # every parameter, so a single anomalous batch corrupts all weights and the
        # parameters can never recover. The returned norm is therefore checked for
        # finiteness and, if invalid, the gradients are simply dropped so that the
        # parameters are untouched and training continues.
        if self.grad_clip > 0:
            total_norm = clip_grad_norm_(self.params, self.grad_clip)
        else:
            # Compute the global norm for the guard; max_norm=inf rescales nothing.
            total_norm = clip_grad_norm_(self.params, float('inf'))

        if not torch.isfinite(total_norm):
            self._nan_skips = getattr(self, '_nan_skips', 0) + 1
            logger.error("[NaN] iteration %d: gradient norm is nan/inf, update "
                         "dropped (%d dropped so far)", self.Eiters, self._nan_skips)
            self.optimizer.zero_grad(set_to_none=True)
            return 0.

        self.optimizer.step()








if __name__ == '__main__':

    pass
    



