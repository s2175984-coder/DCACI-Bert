"""Generate the text and visual concept vectors used by the concept graph.

The concept graph is built on two sets of node embeddings that are derived from
the training split and cached next to the concept annotations,

    {data_path}/Concept_annotations/{data_name}_concepts_word2vec_{n}.pkl
    {data_path}/Concept_annotations/{data_name}_concepts_visual2vec_{n}.pkl

where n is --num_attribute (300 by default). The text side averages the GloVe
vectors of the concept words, the visual side averages the pooled region
features of every image annotated with the concept. Both tables are computed
once and then frozen.

The concept annotations themselves (the per-image concept labels, the
co-occurrence matrix and the concept similarity matrix) must already be in
{data_path}/Concept_annotations; this script only derives the two embedding
tables from them.

The visual side needs a GPU because the region features are moved to the
device. Run the script once per dataset before the first training run, an
existing file is left untouched unless --overwrite is given.

Usage
-----
    python gen_concept_features.py --data_path /path/to/data/f30k \
        --data_name f30k_precomp --bert_path /path/to/bert-base-uncased

    # regenerate both tables even if the files already exist
    python gen_concept_features.py --data_path /path/to/data/f30k \
        --data_name f30k_precomp --bert_path /path/to/bert-base-uncased --overwrite

    # only the text side, which needs no GPU
    python gen_concept_features.py --data_path /path/to/data/f30k \
        --data_name f30k_precomp --bert_path /path/to/bert-base-uncased --skip_visual
"""
import os

import torch
from transformers import BertTokenizer

import arguments
from MGCN.lib.visual2vec import visual2vec
from MGCN.lib.word2vec import word2vec


def concept_path(opt, kind):
    return os.path.join(opt.data_path, 'Concept_annotations',
                        '{0}_concepts_{1}_{2}.pkl'.format(opt.data_name, kind, opt.num_attribute))


def main():
    parser = arguments.get_argument_parser()
    parser.add_argument('--overwrite', action='store_true',
                        help='regenerate the files even if they already exist')
    parser.add_argument('--skip_visual', action='store_true',
                        help='only generate the text side, no GPU needed')
    parser.add_argument('--skip_text', action='store_true',
                        help='only generate the visual side')
    opt = parser.parse_args()

    annotation_dir = os.path.join(opt.data_path, 'Concept_annotations')
    if not os.path.isdir(annotation_dir):
        raise FileNotFoundError('{0} not found'.format(annotation_dir))

    text_out = concept_path(opt, 'word2vec')
    visual_out = concept_path(opt, 'visual2vec')

    if not opt.skip_text:
        if os.path.exists(text_out) and not opt.overwrite:
            print('[skip] {0} already exists'.format(text_out))
        else:
            word2vec(opt)
            print('[done] {0}'.format(text_out))

    if not opt.skip_visual:
        if os.path.exists(visual_out) and not opt.overwrite:
            print('[skip] {0} already exists'.format(visual_out))
        else:
            torch.cuda.set_device(opt.gpu_id)
            tokenizer = BertTokenizer.from_pretrained(opt.bert_path)
            visual2vec(opt, tokenizer)
            print('[done] {0}'.format(visual_out))


if __name__ == '__main__':
    main()
