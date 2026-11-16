import os

# CUDA allocator setting, kept consistent with train.py.
# It must be applied before torch is imported.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:256')

import torch
import argparse
import logging

from lib import evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/root/autodl-tmp/data/f30k/', type=str,
                        help='path to datasets')
    parser.add_argument('--dataset', default='f30k', type=str,
                        help='the dataset choice, coco or f30k')
    parser.add_argument('--save_results', type=int, default=0,
                        help='if save the similarity matrix for ensemble')
    parser.add_argument('--model_path', default='runs_test/model_best.pth', type=str,
                        help='path to the trained checkpoint (model_best.pth)')
    parser.add_argument('--save_embs_dir', default='', type=str,
                        help='export dir of img/cap embeddings for visualization; '
                             'empty -> same dir as the checkpoint')
    parser.add_argument('--gpu-id', type=int, default=0, help='the gpu-id for evaluation')
    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu_id)

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # The checkpoint is taken from --model_path only, and a missing file is
    # reported immediately instead of silently evaluating the wrong model.
    model_path = opt.model_path
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            'checkpoint not found: {0} (pass model_best.pth via --model_path)'.format(model_path))
    logger.info('Evaluating {0}...'.format(model_path))

    if opt.save_results:
        save_dir = os.path.dirname(model_path) or 'runs_test'
        save_path = os.path.join(save_dir, 'results_{0}.npy'.format(opt.dataset))
    else:
        save_path = None

    if opt.dataset == 'coco':
        # COCO 5-fold 1K and 5K
        evaluation.evalrank(model_path, split='testall', fold5=True,
                            save_path=save_path, data_path=opt.data_path,
                            save_embs_dir=opt.save_embs_dir)
    else:
        # Flickr30K, single test split
        evaluation.evalrank(model_path, split='test', fold5=False,
                            save_path=save_path, data_path=opt.data_path,
                            save_embs_dir=opt.save_embs_dir)


if __name__ == '__main__':
    main()
