import torch
import torch.nn as nn
import torch.nn.functional as F


def pos_neg_mask(labels):

    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) 
    neg_mask = labels.unsqueeze(0) != labels.unsqueeze(1)

    return pos_mask, neg_mask


def pos_neg_mask_xy(labels_col, labels_row):

    pos_mask = (labels_row.unsqueeze(0) == labels_col.unsqueeze(1)) 
    neg_mask = (labels_row.unsqueeze(0) != labels_col.unsqueeze(1))

    return pos_mask, neg_mask


def loss_select(opt, loss_type='vse'):

    if loss_type == 'vse':
        # the default loss
        criterion = ContrastiveLoss(opt=opt, margin=opt.margin, max_violation=opt.max_violation)
    elif loss_type == 'trip':
        # Triplet loss with the distance-weight sampling
        criterion = TripletLoss(opt=opt)
    else:
        raise ValueError('Invalid loss {}'.format(loss_type))
    
    return criterion


class ContrastiveLoss(nn.Module):

    def __init__(self, opt, margin=0.2, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation
        self.mask_repeat = opt.mask_repeat

        self.false_hard = []

    def max_violation_on(self):
        self.max_violation = True
        # print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        # print('Use VSE0 objective.')

    def forward(self, im, s, img_ids=None):

        # compute image-sentence score matrix
        scores = get_sim(im, s)
        
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval, i->t
        cost_s = (self.margin + scores - d1).clamp(min=0)

        # compare every diagonal score to scores in its row
        # image retrieval t->i
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        if not self.mask_repeat:
            mask = torch.eye(scores.size(0), dtype=torch.bool, device=scores.device)
        else:
            img_ids = img_ids.cuda()
            mask = (img_ids.unsqueeze(1) == img_ids.unsqueeze(0))

        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s, idx_s = cost_s.max(1)
            cost_im, idx_im = cost_im.max(0)

        loss = cost_s.sum() + cost_im.sum()

        return loss


def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities


# Triplet loss + DistanceWeight Miner
# Sampling Matters in Deep Embedding Learning, ICCV, 2017
# more information refer to https://kevinmusgrave.github.io/pytorch-metric-learning/miners/#distanceweightedminer
class TripletLoss(nn.Module):

    def __init__(self, opt=None, margin=0.2, ):
        super().__init__()

        self.opt = opt
        self.margin = margin
        
        self.cut_off = 0.5
        self.d = 512

        if opt.dataset == 'coco':
            self.nonzero_loss_cutoff = 1.9         
        else:
            self.nonzero_loss_cutoff = 1.7

        # The max_violation switch is required by vse.set_max_violation; without it
        # the "enable hard negatives after epoch 1" schedule in train.py has no
        # effect when base_loss='trip'.
        # With hard_negative=0 the switch is a no-op and training is unchanged;
        # with hard_negative=1 it replaces distance-weighted random sampling with
        # the least similar valid negative, i.e. a true hard negative.
        self.hard_negative = bool(getattr(opt, 'trip_hard_negative', 0)) if opt is not None else False
        self.max_violation = False

        # CG-HNM : use the concept-structure relation S to guide negative sampling.
        # guide_mix  : blending ratio between concept guidance and distance weighting
        #              (1 = concept guidance only).
        # guide_temp : softmax temperature; S is RMS-normalised, so 1 is sharp enough.
        self.guide_mix = float(getattr(opt, 'cg_hnm_mix', 0.5)) if opt is not None else 0.5
        self.guide_temp = float(getattr(opt, 'cg_hnm_temp', 1.0)) if opt is not None else 1.0

    def max_violation_on(self):
        self.max_violation = True

    def max_violation_off(self):
        self.max_violation = False

    def forward(self, im, s, img_ids, guide=None):

        sim_mat = get_sim(im, s)
        img_ids = img_ids.cuda()

        if im.size(0) == s.size(0):
            pos_mask, neg_mask = pos_neg_mask(img_ids)
        else:
            pos_mask, neg_mask = pos_neg_mask_xy(torch.unique(img_ids), img_ids)

        loss_im = self.loss_forward(sim_mat, pos_mask, neg_mask, guide=guide)
        loss_s = self.loss_forward(sim_mat.t(), pos_mask.t(), neg_mask.t(),
                                   guide=(guide.t() if guide is not None else None))

        loss = loss_im + loss_s

        return loss        

    def loss_forward(self, sim_mat, pos_mask, neg_mask, guide=None): 

        pos_pair_idx = pos_mask.nonzero(as_tuple=False)
        anchor_idx = pos_pair_idx[:, 0]
        pos_idx = pos_pair_idx[:, 1]

        dist = (2 - 2 * sim_mat).sqrt()
        dist = dist.clamp(min=self.cut_off)

        log_weight = (2.0 - self.d) * dist.log() - ((self.d - 3.0) / 2.0) * (1.0 - 0.25 * (dist * dist)).log()
        inf_or_nan = torch.isinf(log_weight) | torch.isnan(log_weight)

        log_weight = log_weight * neg_mask  
        log_weight[inf_or_nan] = 0.      

        weight = (log_weight - log_weight.max(dim=1, keepdim=True)[0]).exp()
        weight = weight * (neg_mask * (dist < self.nonzero_loss_cutoff)).float() 
     
        weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-20)

        # CG-HNM : concept-structure guided negative sampling.
        # A candidate that is close in concept space (large guide value) and is not
        # a positive pair receives a higher sampling weight, which makes it a hard
        # negative that is also conceptually confusable.
        if guide is not None and guide.shape == sim_mat.shape:
            g = guide.detach()
            g = g - g.mean(dim=1, keepdim=True)              # remove row-wise bias
            gw = torch.softmax(g / max(self.guide_temp, 1e-6), dim=1)
            gw = gw * neg_mask.float()
            gw = gw / (gw.sum(dim=1, keepdim=True) + 1e-20)
            weight = self.guide_mix * gw + (1.0 - self.guide_mix) * weight
            weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-20)

        weight = weight[anchor_idx]

        if self.max_violation and self.hard_negative:
            # Compute the triplet on the hardest negative, i.e. the valid negative
            # with the lowest similarity (largest distance).
            valid_rows = neg_mask.bool()[anchor_idx] & (dist[anchor_idx] < self.nonzero_loss_cutoff)
            sim_rows = sim_mat[anchor_idx].masked_fill(~valid_rows, float('inf'))
            neg_idx = sim_rows.argmin(dim=1)
            no_valid = ~valid_rows.any(dim=1)
            if no_valid.any():
                # An anchor without any valid negative falls back to itself, which
                # gives relu(margin + s_ap - s_ap) = margin and no wrong gradient.
                neg_idx = torch.where(no_valid, anchor_idx, neg_idx)
        else:
            # maybe not exist
            try:
                neg_idx = torch.multinomial(weight, 1).squeeze(1)   
            except Exception:
                return torch.zeros([], requires_grad=True, device=sim_mat.device) 


        s_ap = sim_mat[anchor_idx, pos_idx]
        s_an = sim_mat[anchor_idx, neg_idx]  

        loss = F.relu(self.margin + s_an - s_ap) 
        loss = loss.sum() 

        return loss


if __name__ == '__main__':

    pass
    