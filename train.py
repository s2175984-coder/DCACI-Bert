import os
import sys

# =====================================================================
# CUDA allocator configuration. This does not change any computation but it
# must be applied before torch is imported and before the first CUDA call.
#
# Large-batch training can fail with CUDA out of memory even though the total
# amount of free memory is sufficient, because the allocator cannot find one
# contiguous block. Limiting the maximum split size keeps large blocks reusable
# and reduces fragmentation.
#
# setdefault is used so that a value exported in the shell takes precedence.
# =====================================================================
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')

import time

import numpy as np
import torch
import logging
from transformers import BertTokenizer
import tensorboard_logger as tb_logger
import arguments

from lib import evaluation
from lib import image_caption
from lib.vse import VSEModel
from lib.encoders import build_clip_encoder
from lib.evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, compute_sim
from graph_lib import *

def main():

    # Hyper Parameters
    parser = arguments.get_argument_parser()
    parser = extra_parameters(parser)
    opt = parser.parse_args()

    # Fix the random seeds so that a run is reproducible. Repeating the command
    # with a different --seed gives the mean and standard deviation across runs.
    import random
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    opt.text_channel = 768  # dimension of initial text concept embedding
    opt.visual_channel = 2048  # dimension of initial visual concept embedding

    opt.model_name = opt.logger_name

    # ---------------------------------------------------------------
    # Single-instance lock and GPU headroom check. It must run before
    # logging.basicConfig(filemode='w'), otherwise a second process would truncate
    # the log of the first one before failing. No numerical computation is involved.
    # See lib/run_guard.py for details.
    # ---------------------------------------------------------------
    from lib.run_guard import acquire_or_exit
    acquire_or_exit(opt.model_name,
                    enabled=getattr(opt, 'single_instance', 1),
                    min_free_mb=getattr(opt, 'min_free_gpu_mb', 0),
                    gpu_id=getattr(opt, 'gpu_id', 0))

    # set the gpu-id for training
    if not opt.multi_gpu:
        torch.cuda.set_device(opt.gpu_id)

    # create the folder for logger and checkpoint
    if not os.path.exists(opt.model_name):
        os.makedirs(opt.model_name)

    # initialize logger
    logging.basicConfig(filename=os.path.join(opt.logger_name, 'train.txt'), filemode='w', 
                        format='%(asctime)s %(message)s', level=logging.INFO)
    # Log to stdout as well, so that progress is visible when the run is started
    # with nohup and the output is piped to a file.
    _console = logging.StreamHandler(stream=sys.stdout)
    _console.setLevel(logging.INFO)
    _console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logging.getLogger().addHandler(_console)
    logger = logging.getLogger(__name__)
    logger.info(opt)  

    tb_logger.configure(opt.logger_name, flush_secs=5)

    # record parameters
    arguments.save_parameters(opt, opt.logger_name)

    # load tokenizer for TextEncoder
    # The BERT location is taken from opt.bert_path (see arguments.py) so that no
    # code change is needed when the data root differs between machines.
    tokenizer = BertTokenizer.from_pretrained(opt.bert_path)
    # ======================================================================
    # Build the CLIP branch. A single ClipEncoder is created per process and
    # shared by the DataLoader (tokenizer and preprocess, needed in online mode)
    # and by VSEModel, whose encoders call encode_image / encode_text. When
    # --use_clip_fusion is 0 or --clip_source is npy, build_clip_encoder returns
    # None and no CLIP weights are loaded.
    # ======================================================================
    clip_encoder = build_clip_encoder(opt)
    clip_tokenizer = clip_encoder.clip_tokenizer if clip_encoder is not None else None
    preprocess = clip_encoder.preprocess if clip_encoder is not None else None

    # load the multi-modal model
    model = VSEModel(opt, clip_encoder=clip_encoder)

    # get the train-set 
    train_loader = image_caption.get_train_loader(opt.data_path, tokenizer, opt.batch_size,
                                                  opt.workers, opt,
                                                  clip_tokenizer=clip_tokenizer,
                                                  preprocess=preprocess)

    # get the test-set
    split = 'testall' if opt.dataset == 'coco' else 'test'
    test_loader = image_caption.get_test_loader(opt.data_path, split, tokenizer, opt.batch_size,
                                                opt.workers, opt,
                                                clip_tokenizer=clip_tokenizer,
                                                preprocess=preprocess)

    # Model selection follows the evaluation protocol used for the reported
    # results: MS-COCO uses testall (5000 images) and Flickr30K uses test
    # (1000 images). The rSum of every epoch is measured on the test split and
    # the best checkpoint is kept as model_best.pth.
    # Note: dataset.num_images is the number of rows in the npy file
    # (images x captions), not the number of distinct images.

    # Release the test-split arrays (about 1.37 GB) and load them on demand during
    # validation, so that training does not exhaust the physical memory.
    test_loader.dataset.release_images()

    logger.info('Number of images for train-set: {}'.format(train_loader.dataset.num_images))

    start_epoch = 0

    # use the multi gpu
    if (not model.is_data_parallel) and opt.multi_gpu:
        model.make_data_parallel()

    best_rsum = 0

    # start the training process
    for epoch in range(start_epoch, opt.num_epochs):

        if epoch == 0:
            logger.info('Log saving path: ' + opt.logger_name)
            logger.info('Models saving path: ' + opt.model_name)

        adjust_learning_rate(opt, model.optimizer, epoch)

        # =================================================================
        # Read-only bookkeeping of the values that actually take effect, printed at
        # the start of every epoch. The requested command-line value can differ from
        # the effective one because the temperature is annealed and the rank is
        # truncated to an integer number of heads.
        # =================================================================
        _core = getattr(model, 'module', model)
        _crit = getattr(_core, 'criterion', None)
        if _crit is not None and getattr(_crit, 'use_concept_graph', False):
            # Temperature annealing is active only when --tau_anneal_epochs > 0.
            _n_anneal = int(getattr(opt, 'tau_anneal_epochs', 0) or 0)
            if _n_anneal > 0:
                _t0 = float(getattr(opt, 'tau_init', 0.0) or 0.0) or None
                _tmin = float(getattr(opt, 'tau_min', 1.1))
                if _t0 is None:
                    import math as _math
                    _t0 = _math.log(opt.embed_size) / 2.0 + 1.0     # softplus(log sqrt D)+1
                _frac = min(float(epoch) / float(_n_anneal), 1.0)
                _tau = _t0 + (_tmin - _t0) * _frac
                _crit.set_temperature(_tau)
                logger.info('epoch {}: tau anneal -> {:.4f} (from {:.4f} to {:.4f} over {} ep)'.format(
                    epoch, _tau, _t0, _tmin, _n_anneal))

            if epoch == 0:
                _cap = _crit.capacity_report()
                if _cap:
                    logger.info('idea1 capacity: requested_rank={requested_rank} '
                                'effective_rank={effective_rank} num_heads={num_heads} '
                                'head_dim={head_dim}'.format(**_cap))
            _tau_d = _crit.tau_report()
            logger.info('epoch {}: effective tau = text {:.4f} | visual {:.4f} | all {:.4f}'.format(
                epoch, _tau_d['text'], _tau_d['visual'], _tau_d['all']))

        # set hard negative for vse loss
        if (epoch >= opt.vse_mean_warmup_epochs):
            opt.max_violation = True
            model.set_max_violation(opt.max_violation)

        # Warm-up for graph-enhanced inference. Early in training the concept graph
        # is not yet reliable, so validating with it would understate the accuracy and
        # distort model selection. The first graph_inference_warmup epochs therefore
        # validate with encoder features only.
        if getattr(opt, 'use_graph_inference', 0):
            model.use_graph_inference = bool(epoch >= getattr(opt, 'graph_inference_warmup', 0))
            logger.info('epoch {}: use_graph_inference = {}'.format(epoch, model.use_graph_inference))

        # Warm-up for the sample-level four-block inference path.
        if getattr(opt, 'use_instance_inference', 0):
            model.use_instance_inference = bool(epoch >= getattr(opt, 'instance_inference_warmup', 0))
            logger.info('epoch {}: use_instance_inference = {}'.format(epoch, model.use_instance_inference))

        # train for one epoch
        train(opt, train_loader, model, epoch)

        # Log the prior weight learned by each of the three relation heads,
        # w = sigmoid(alpha). Starting from different initial values, a consistent
        # range of w indicates that the prior strength is driven by the data rather
        # than hand-tuned. w is reported instead of alpha because alpha lives in logit
        # space, where distances are not comparable.
        if int(getattr(opt, 'alpha_log', 1)):
            _core2 = getattr(model, 'module', model)
            _crit2 = getattr(_core2, 'criterion', None)
            if _crit2 is not None and hasattr(_crit2, 'alpha_report'):
                _a = _crit2.alpha_report()
                logger.info('epoch {}: prior weight w=sigmoid(alpha) = '
                            'text {:.4f} | visual {:.4f} | all {:.4f} | mean {:.4f}'.format(
                                epoch, _a['text'], _a['visual'], _a['all'], _a['mean']))

        # evaluate on test set for every epoch
        rsum = validate(opt, test_loader, model)

        # remember best rsum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)

        logger.info("Epoch: [{}], Best rsum: {:.1f}".format(epoch, best_rsum))

        # save the checkpoint
        state = {'model': model.state_dict(), 'opt': opt, 'epoch': epoch + 1, 'best_rsum': best_rsum, 'Eiters': model.Eiters}
        save_checkpoint(state, is_best, prefix=opt.model_name)

    logger.info('Train finish.')    

    # evaluation after training process
    logger.info('Evaluate the model')
    
    base = opt.logger_name
    logging.basicConfig(filename=os.path.join(base, 'eval.txt'), filemode='w', 
                        format='%(asctime)s %(message)s', level=logging.INFO, force=True)
    logger = logging.getLogger()

    logger.info('Evaluating {}'.format(base))
    model_path = os.path.join(base, 'model_best.pth')

    # Save the final results for computing ensemble results
    save_path = os.path.join(base, 'results_{}.npy'.format(opt.dataset)) if opt.save_results else None

    if opt.dataset == 'coco':
        # Evaluate COCO 5-fold 1K
        # Evaluate COCO 5K
        evaluation.evalrank(model_path, opt=opt, tokenizer=tokenizer, model=model, split='testall', fold5=True,
                            save_path=save_path, clip_encoder=clip_encoder)
    else:
        # Evaluate Flickr30K
        evaluation.evalrank(model_path, opt=opt, tokenizer=tokenizer, model=model, split='test', fold5=False,
                            save_path=save_path, clip_encoder=clip_encoder)

    logger.info('Evaluation finish')    


def train(opt, train_loader, model, epoch):
   
    logger = logging.getLogger(__name__)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    if epoch == 0:
        logger.info('image encoder trainable parameters: {}M'.format(count_params(model.img_enc)))
        logger.info('txt encoder trainable parameters: {}M'.format(count_params(model.txt_enc)))
        logger.info('criterion trainable parameters: {}M'.format(count_params(model.criterion)))

    end = time.time()
    repeat_list = []
    n_batch = len(train_loader.dataset) // opt.batch_size 

    model.train_start()    

    for i, train_data in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        (images, img_lengths, captions, lengths, ids, img_ids, repeat, attribute_labels,
         clip_imgs, clip_caps, patch_feats) = train_data

        # Iteration-level linear learning-rate warm-up over lr_warmup_iters steps.
        set_warmup_lr(opt, model.optimizer, model.Eiters)

        model.train_emb(images, captions, lengths, attribute_labels, image_lengths=img_lengths,
                        img_ids=img_ids, clip_imgs=clip_imgs, clip_caps=clip_caps,
                        patch_feats=patch_feats)
  
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if model.Eiters % opt.log_step == 0:               
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Batch-Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                    .format(
                    epoch, i+1, n_batch, batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

    return repeat_list


def validate(opt, val_loader, model):

    logger = logging.getLogger(__name__)
    model.val_start()

    # Load the test-split arrays for validation and release them immediately
    # afterwards (f30k->test_ims.npy, coco->testall_ims.npy).
    val_loader.dataset.ensure_images()

    with torch.no_grad():

        # compute the encoding for all the validation images and captions
        img_embs, cap_embs = encode_data(model, val_loader, opt.log_step, logging.info)

    # Release the test-split arrays once validation is done.
    val_loader.dataset.release_images()

    # have repetitive image features
    img_embs = img_embs[::5]
    npts = img_embs.shape[0]

    sims = compute_sim(img_embs, cap_embs)

    (r1, r5, r10, medr, meanr) = i2t(npts, sims)
    logging.info("Image to text (R@1, R@5, R@10): %.1f, %.1f, %.1f" % (r1, r5, r10))

    (r1i, r5i, r10i, medri, meanr) = t2i(npts, sims)
    logging.info("Text to image (R@1, R@5, R@10): %.1f, %.1f, %.1f" % (r1i, r5i, r10i))

    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i
    logger.info('Current rsum is {}'.format(round(currscore, 1)))

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)

    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    
    tb_logger.log_value('rsum', currscore, step=model.Eiters)    

    return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth', prefix=''):
    logger = logging.getLogger(__name__)
    tries = 2

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            # don't save checkpoint
            # torch.save(state, prefix + filename)
            if is_best:
                torch.save(state, os.path.join(prefix, 'model_best.pth'))
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        logger.info('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def set_warmup_lr(opt, optimizer, iters):
    """Iteration-level linear learning-rate warm-up.

    The randomly initialised graph modules (AdjacencyModel, the dynamic GCN and the
    concept decoder) share the learning rate of the pretrained encoders. Starting
    them directly at the full rate produces a large loss oscillation, so the rate is
    ramped linearly from zero over the first lr_warmup_iters iterations.

    The target rate is the base learning rate maintained by the step decay in
    adjust_learning_rate and stored in optimizer._base_lrs.
    """
    warmup = int(getattr(opt, 'lr_warmup_iters', 0) or 0)
    if warmup <= 0:
        return
    if not hasattr(optimizer, '_base_lrs'):
        optimizer._base_lrs = [g['lr'] for g in optimizer.param_groups]
    if len(optimizer._base_lrs) != len(optimizer.param_groups):
        optimizer._base_lrs = [g['lr'] for g in optimizer.param_groups]
    scale = min(1.0, float(iters + 1) / float(warmup))
    for g, base in zip(optimizer.param_groups, optimizer._base_lrs):
        g['lr'] = base * scale


def adjust_learning_rate(opt, optimizer, epoch):
    logger = logging.getLogger(__name__)

    decay_rate = opt.decay_rate
    lr_schedules = opt.lr_schedules

    if epoch in lr_schedules:
        logger.info('Current epoch num is {}, decrease all lr by 10'.format(epoch, ))
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * decay_rate
            param_group['lr'] = new_lr
            logger.info('new lr: {}'.format(new_lr))
        # Refresh the warm-up reference, otherwise the ramp would overwrite the
        # decay schedule once it ends.
        optimizer._base_lrs = [g['lr'] for g in optimizer.param_groups]


def count_params(model):

    # The unit is M (million)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    params = round(params/(1024**2), 2)

    return params


if __name__ == '__main__':
    
    main()
