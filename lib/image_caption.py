"""Data pipeline for Flickr30K and MS-COCO.

--clip_source selects between two mathematically equivalent sources of the CLIP
global features:

  online (default)
      __getitem__ returns the raw image -> CLIP preprocess -> (3,224,224),
      and the caption -> CLIP tokenizer -> (77,).
      After collate: full_images (B,3,224,224) and clip_captions (B,77).
      The encoder runs CLIP under torch.no_grad, so CLIP stays frozen.

  npy (optional, faster)
      Reads pre-computed features {split}_clip_imgs.npy / {split}_clip_caps.npy
      into (B,512). The values match the online path because both use the same
      frozen weights and the same preprocess, but no image is decoded.

In both modes collate_fn returns a 10-tuple whose last two entries are the CLIP
payload; VSEModel interprets them as (raw images, tokens) or as pre-computed
vectors according to opt.clip_source.

Region features, concept labels, size augmentation and im_div follow the standard
VSE setting.
"""
import torch
import torch.utils.data as data
import os
import numpy as np
import random
import logging
import pickle
from imageio import imread
import random
import json
import cv2
cv2.setNumThreads(0)
import logging
import json
import pickle
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import nltk
from PIL import Image
logger = logging.getLogger(__name__)


class PrecompRegionDataset(data.Dataset):
    def __init__(self, data_path, data_split, tokenizer, opt, train,
                 clip_tokenizer=None, preprocess=None):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.opt = opt
        self.train = train
        self.clip_tokenizer = clip_tokenizer
        self.preprocess = preprocess

        # The precomputed directory is selected from opt.dataset so that switching
        # datasets never reads the files of the other one:
        #     coco -> {data_path}/coco_precomp  + {data_path}/images
        #     f30k -> {data_path}/f30k_precomp  + {data_path}/flickr30k-images
        data_base = os.path.join(data_path)
        is_coco = 'coco' in opt.dataset
        loc = os.path.join(data_base, 'coco_precomp' if is_coco else 'f30k_precomp')
        loc_mapping = os.path.join(data_base, 'id_mapping.json')
        self.image_base = os.path.join(
            data_base, 'images' if is_coco else 'flickr30k-images')

        with open(loc_mapping, 'r') as f_mapping:
            self.id_to_path = json.load(f_mapping)
        # Raw captions
        self.captions = []
        with open(os.path.join(loc, '%s_caps.txt' % data_split), 'r',encoding='utf-8') as f:
            for line in f:
                self.captions.append(line.strip())

        # Region features. The path of the current split is kept so that the array
        # can be loaded and released on demand (released during training, loaded
        # temporarily for validation).
        self.images_path = os.path.join(loc, '%s_ims.npy' % data_split)
        # Optional memory mapping (--region_mmap 1).
        # MS-COCO train_ims.npy is 33.4 GB (31.1 GiB); loading it fully into RAM
        # together with the CLIP features keeps about 32.5 GiB resident and a
        # container can be killed by the cgroup OOM killer. On Linux, mmap_mode='r'
        # turns this into demand paging and cuts the resident set substantially, at
        # the cost of random reads from disk. Large-file mmap has triggered
        # 0xC0000005 on Windows, so it is disabled by default.
        self.region_mmap = bool(getattr(opt, 'region_mmap', 0))
        self.images = np.load(self.images_path,
                              mmap_mode='r' if self.region_mmap else None)

        # ==================================================================
        # CLIP global payload:
        #   clip_source == 'online' : raw images + CLIP tokenizer
        #   clip_source == 'npy'    : pre-computed {split}_clip_imgs.npy / _caps.npy
        #   use_clip_fusion == 0    : zeros of the same shape, nothing is read from disk
        # ==================================================================
        self.use_clip_fusion = bool(getattr(opt, 'use_clip_fusion', 0))
        self.clip_source = str(getattr(opt, 'clip_source', 'online')).lower()
        if self.clip_source not in ('online', 'npy'):
            raise ValueError("--clip_source must be 'online' or 'npy', got: %r" % self.clip_source)
        self.clip_dim = int(getattr(opt, 'clip_dim', 512))

        self.clip_imgs = None
        self.clip_caps = None
        self.clip_imgs_path = None
        self.clip_caps_path = None

        if self.use_clip_fusion and self.clip_source == 'online':
            if self.clip_tokenizer is None or self.preprocess is None:
                raise ValueError(
                    "clip_source='online' requires the CLIP tokenizer and preprocess.\n"
                    "  train.py / eval.py pass them from the shared ClipEncoder; when "
                    "building a DataLoader manually, pass clip_tokenizer=..., "
                    "preprocess=..., or use --clip_source npy.")
        elif self.use_clip_fusion:
            clip_dir = getattr(opt, 'clip_feat_dir', '') or loc
            self.clip_imgs_path = os.path.join(clip_dir, '%s_clip_imgs.npy' % data_split)
            self.clip_caps_path = os.path.join(clip_dir, '%s_clip_caps.npy' % data_split)
            if not (os.path.exists(self.clip_imgs_path) and os.path.exists(self.clip_caps_path)):
                raise FileNotFoundError(
                    'clip_source=npy but the pre-computed CLIP features are missing:\n'
                    '  {}\n  {}\n'
                    'Run gen_clip_features.py for the missing split, or point '
                    '--clip_feat_dir at the feature directory'.format(
                        self.clip_imgs_path, self.clip_caps_path))
            self.clip_imgs = np.load(self.clip_imgs_path)
            self.clip_caps = np.load(self.clip_caps_path)
            logger.info('CLIP features [%s]: imgs=%s caps=%s'
                        % (data_split, self.clip_imgs.shape, self.clip_caps.shape))

        # ==================================================================
        # CLIP patch features, read from disk only when --use_patch_fusion 1.
        # The files are produced by gen_patch_features.py and may use either layout:
        #     dedup: {split}_clip_patches.npy (N_uniq, P, patch_dim)
        #            + {split}_clip_patches_index.npy (N_row,) int32
        #     row  : {split}_clip_patches.npy (N_row, P, patch_dim), no index file
        # Both behave identically in __getitem__: img_index selects the patches of
        # that image. When the branch is off, zeros of the same shape are returned
        # so that collate and the downstream code need no special case.
        # ==================================================================
        self.use_patch_fusion = bool(getattr(opt, 'use_patch_fusion', 0))
        self.patch_dim = int(getattr(opt, 'patch_dim', 512))
        self.patch_num = int(getattr(opt, 'patch_num', 49))
        self.patch_feats = None
        self.patch_index = None
        self.patch_feats_path = None
        if self.use_patch_fusion:
            patch_dir = getattr(opt, 'patch_feat_dir', '') or loc
            self.patch_feats_path = os.path.join(
                patch_dir, '%s_clip_patches.npy' % data_split)
            idx_path = os.path.join(patch_dir, '%s_clip_patches_index.npy' % data_split)
            if not os.path.exists(self.patch_feats_path):
                raise FileNotFoundError(
                    'use_patch_fusion=1 but the patch features are missing:\n  %s\n'
                    'Run: python gen_patch_features.py --data_path <data_path> '
                    '--dataset %s --splits %s' % (self.patch_feats_path,
                                                  opt.dataset, data_split))
            self.patch_feats = np.load(
                self.patch_feats_path, mmap_mode='r' if self.region_mmap else None)
            if os.path.exists(idx_path):
                self.patch_index = np.load(idx_path)
            logger.info('[patch] %s: feats=%s index=%s',
                        data_split, self.patch_feats.shape,
                        None if self.patch_index is None else self.patch_index.shape)

        # Global image ids
        with open(os.path.join(loc, '{}_ids.txt'.format(data_split)), 'r') as f:
            image_ids = f.readlines()
            self.full_images = [int(x.strip()) for x in image_ids]

        # num_captions
        self.length = len(self.captions)
        self.num_images = len(self.images)

        if self.num_images != self.length:
            # one images to five captions (train set)
            self.im_div = 5
        else:
            # one images to one captions (test set)
            self.im_div = 1

        if data_split == 'dev':
            self.length = 5000

        '''load the concrete words of concepts'''
        categroy_concepts_dir = os.path.join(opt.data_path,
                                            'Concept_annotations', 'concepts_{}.pkl'.format(str(opt.num_attribute)))
        with open(categroy_concepts_dir, "rb") as names_concepts:
            concepts = pickle.load(names_concepts)
            self.text_concepts = concepts['text_concepts']
            self.visual_objects = concepts['visual_objects']

        self.text_classes = len(self.text_concepts)
        self.visual_classes = len(self.visual_objects)

    def _patch_row(self, img_index):
        """Return the patch-feature row that belongs to this image.

        dedup layout: {split}_clip_patches.npy stores unique images only and the
                      mapping is given by _index.npy.
        row layout  : there is no index file, so the row equals img_index.
        Keeping this in one method makes the mapping directly testable, because a
        wrong row raises no error and would silently pair a caption with the
        patches of another image.
        """
        if self.patch_index is None:
            return int(img_index)
        return int(self.patch_index[img_index])

    def _load_full_image(self, img_index):
        """Decode the raw image via id_mapping and apply CLIP preprocess.

        Returns a (1,3,H,W) tensor.
        """
        image_id = self.full_images[img_index]
        image_path = os.path.join(self.image_base, self.id_to_path[str(image_id)])
        with Image.open(image_path) as im:
            rgb = im.convert('RGB')
        return self.preprocess(rgb).unsqueeze(0)

    def __getitem__(self, index):

        # handle the image redundancy
        # index for captions, img_index for images
        img_index = index // self.im_div
        caption = self.captions[index]

        caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption)

        # Convert caption (string) to word ids (with Size Augmentation at training time).
        target = process_caption_bert(self.tokenizer, caption_tokens, self.train)

        image = self.images[img_index]

        # Size augmentation: randomly drop region features to make the model more
        # robust to the number of detected regions.
        if self.train and self.opt.size_augment:
            num_features = image.shape[0]
            rand_list = np.random.rand(num_features)
            image = image[np.where(rand_list > 0.20)]

        image = torch.Tensor(image)
        ''' generate the concept labels for each sentence'''
        concept_labels = np.zeros(self.text_classes + self.visual_classes, np.float32)
        for (i, word) in enumerate(caption_tokens):
            try:
                word, tag = nltk.pos_tag([word])[0]
                if tag.startswith('NN'):
                    word_lemma = lemmatizer.lemmatize(word, pos='n')
                elif tag.startswith('VB'):
                    word_lemma = lemmatizer.lemmatize(word, pos='v')
                elif tag.startswith('JJ'):
                    word_lemma = lemmatizer.lemmatize(word, pos='a')
                elif tag.startswith('R'):
                    word_lemma = lemmatizer.lemmatize(word, pos='r')
                else:
                    word_lemma = word
            except:
                continue
            if word_lemma in self.text_concepts:
                inx_concept = self.text_concepts.index(word_lemma)
                concept_labels[inx_concept] = 1

        for i, visual_object in enumerate(self.visual_objects):
            if visual_object in caption:
                concept_labels[i + self.text_classes] = 1

        concept_labels = torch.Tensor(concept_labels)

        # CLIP payload, stored in the last two entries of the returned tuple.
        if self.use_clip_fusion and self.clip_source == 'online':
            # Raw image tensor together with the CLIP tokens.
            clip_img = self._load_full_image(img_index)
            clip_cap = self.clip_tokenizer(caption)
        elif self.use_clip_fusion:
            # Pre-computed features, row-aligned with the region features:
            # img_index selects the image and index selects the caption.
            if self.clip_imgs is None or self.clip_caps is None:
                self.ensure_images()
            clip_img = torch.from_numpy(
                np.ascontiguousarray(self.clip_imgs[img_index], dtype=np.float32))
            clip_cap = torch.from_numpy(
                np.ascontiguousarray(self.clip_caps[index], dtype=np.float32))
        else:
            clip_img = torch.zeros(self.clip_dim, dtype=torch.float32)
            clip_cap = torch.zeros(self.clip_dim, dtype=torch.float32)

        # Patch payload; zeros of the same shape when the branch is off.
        if self.use_patch_fusion:
            if self.patch_feats is None:
                self.ensure_images()
            patch_feat = torch.from_numpy(
                np.ascontiguousarray(self.patch_feats[self._patch_row(img_index)],
                                     dtype=np.float32))
        else:
            patch_feat = torch.zeros(self.patch_num, self.patch_dim, dtype=torch.float32)

        return (image, target, index, img_index, concept_labels, clip_img, clip_cap,
                patch_feat)

    def __len__(self):
        return self.length

    # ------------------------------------------------------------------
    # On-demand loading and release of the region-feature arrays.
    # Keeping the whole test ims.npy (about 1.37 GB) resident is wasteful during
    # training, so callers use ensure_images() before validation and
    # release_images() afterwards. With --clip_source npy the pre-computed CLIP
    # features are loaded and released the same way.
    # ------------------------------------------------------------------
    def ensure_images(self):
        if self.images is None:
            self.images = np.load(self.images_path,
                                  mmap_mode='r' if self.region_mmap else None)
        if self.use_clip_fusion and self.clip_source == 'npy':
            if self.clip_imgs is None:
                self.clip_imgs = np.load(self.clip_imgs_path)
            if self.clip_caps is None:
                self.clip_caps = np.load(self.clip_caps_path)
        # Patch features support the same on-demand loading and release; MS-COCO
        # train is about 5.7 GB and should not stay resident.
        if getattr(self, 'use_patch_fusion', False) and self.patch_feats is None:
            self.patch_feats = np.load(
                self.patch_feats_path, mmap_mode='r' if self.region_mmap else None)
        return self.images

    def release_images(self):
        self.images = None
        if self.use_clip_fusion and self.clip_source == 'npy':
            self.clip_imgs = None
            self.clip_caps = None
        if getattr(self, 'use_patch_fusion', False):
            self.patch_feats = None


def process_caption_bert(tokenizer, tokens, train=True):

    output_tokens = []
    deleted_idx = []

    for i, token in enumerate(tokens):
        # text -> token (basic_tokenizer.tokenize) -> sub_token (wordpiece_tokenizer.tokenize)
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)

        prob = random.random()

        # first, 20% probability use the augmenation operations
        if prob < 0.20 and train:  # mask/remove the tokens only during training
            prob /= 0.20

            # 50% change token to mask token
            if prob < 0.5:
                for sub_token in sub_tokens:
                    output_tokens.append("[MASK]")
            # 10% randomly change token to random token from the BERT-vocab
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
            # -> 40% delete the token
            else:
                for sub_token in sub_tokens:
                    output_tokens.append(sub_token)
                    # record the index of sub_token
                    deleted_idx.append(len(output_tokens) - 1)
        # 80% probability keep the token
        else:
            for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                output_tokens.append(sub_token)

    if len(deleted_idx) != 0:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

    # and first and last notations for BERT model
    output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']

    # Convert token to vocabulary indices, torch.float32
    target = tokenizer.convert_tokens_to_ids(output_tokens)

    # convert to the torch-tenfor
    target = torch.Tensor(target)
    return target


def make_collate_fn(online_clip):
    """online_clip=True treats the payload as raw images (1,3,224,224) and
    tokens (77,), combined with cat; otherwise it is a pre-computed vector (512,)
    combined with stack."""

    def collate_fn(data):

        # Sort a data list by caption length
        data.sort(key=lambda x: len(x[1]), reverse=True)

        (images, captions, ids, img_ids, attribute_labels, clip_imgs, clip_caps,
         patch_feats) = zip(*data)

        img_ids = torch.tensor(img_ids)
        ids = torch.tensor(ids)

        # print(img_ids)
        repeat = len(img_ids) - len(torch.unique(img_ids))

        # Merge images (convert tuple of 3D tensor to 4D tensor)
        # images = torch.stack(images, 0)

        img_lengths = [len(image) for image in images]

        # dataset_size * max_lengths (maybe 36) * 2048
        all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
        for i, image in enumerate(images):
            end = img_lengths[i]
            all_images[i, :end] = image[:end]

        img_lengths = torch.tensor(img_lengths)

        # Merget captions (convert tuple of 1D tensor to 2D tensor)
        # count the length of each captions
        lengths = [len(cap) for cap in captions]

        # pad the redundancy with zero, in order to input BERT model as a batch
        targets = torch.zeros(len(captions), max(lengths)).long()

        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        lengths = torch.tensor(lengths)
        attribute_labels = torch.stack(attribute_labels, 0)

        # CLIP payload
        if online_clip:
            clip_imgs = torch.cat(clip_imgs, 0)     # (B,3,224,224)
            clip_caps = torch.cat(clip_caps, 0)     # (B,77)
        else:
            clip_imgs = torch.stack(clip_imgs, 0)   # (B,512)
            clip_caps = torch.stack(clip_caps, 0)   # (B,512)

        # Patch payload. When the branch is off every item is a zero tensor of
        # shape (P, patch_dim), so stacking keeps the shape valid; graph_lib only
        # consumes it when use_patch_fusion=1.
        patch_feats = torch.stack(patch_feats, 0)   # (B, P, patch_dim)

        # all_images: Batch_size * max_img_lengths * 2048 (the dimension of region-features)
        # targets:  Batch_size * max_cap_lengths
        return (all_images, img_lengths, targets, lengths, ids, img_ids, repeat, attribute_labels,
                clip_imgs, clip_caps, patch_feats)

    return collate_fn


def get_loader(data_path, data_split, tokenizer, opt, batch_size=100,
               shuffle=True, num_workers=1, train=True,
               clip_tokenizer=None, preprocess=None):

    drop_last = True if train else False

    dset = PrecompRegionDataset(data_path, data_split, tokenizer, opt, train,
                                clip_tokenizer=clip_tokenizer, preprocess=preprocess)

    online_clip = bool(getattr(opt, 'use_clip_fusion', 0)) and \
        str(getattr(opt, 'clip_source', 'online')).lower() == 'online'

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                pin_memory=True,
                                                collate_fn=make_collate_fn(online_clip),
                                                num_workers=num_workers,
                                                drop_last=drop_last)

    return data_loader


def get_train_loader(data_path, tokenizer, batch_size, workers, opt,
                     clip_tokenizer=None, preprocess=None):

    train_loader = get_loader(data_path, 'train', tokenizer, opt, batch_size, True, workers,
                              train=True, clip_tokenizer=clip_tokenizer, preprocess=preprocess)

    return train_loader


def get_test_loader(data_path, split_name, tokenizer, batch_size, workers, opt,
                    clip_tokenizer=None, preprocess=None):

    test_loader = get_loader(data_path, split_name, tokenizer, opt, batch_size, False, workers,
                             train=False, clip_tokenizer=clip_tokenizer, preprocess=preprocess)

    return test_loader


if __name__ == '__main__':

    pass
