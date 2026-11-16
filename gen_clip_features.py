# -*- coding: utf-8 -*-
"""
gen_clip_features.py -- generate or complete the precomputed CLIP global features
used when --use_clip_fusion 1.

Overview
--------
A dataset directory usually ships CLIP features for the train and test splits,
    train_clip_imgs.npy  (29000, 512)    train_clip_caps.npy  (145000, 512)
    test_clip_imgs.npy   (5000,  512)    test_clip_caps.npy   (5000,  512)
while other splits may be missing. This script fills in the missing splits using
the same row-index convention, and can also regenerate any split from scratch.

Row-index convention (must match lib/image_caption.py::__getitem__)
------------------------------------------------------------------
    image row i  <->  line i of {split}_ids.txt   -> id_mapping.json -> file name
    caption row j <-> line j of {split}_caps.txt  (raw caption text)

Images are therefore stored once per caption, with 5 captions per image:
    train : 29000 image rows (29000 unique) / 145000 caption rows
    dev   : 5070  image rows (1014  unique) / 5070  caption rows
    test  : 5000  image rows (1000  unique) / 5000  caption rows

Verify before generating
------------------------
The built-in --verify mode re-encodes the first N rows of a split that already
has features on disk (test by default) and compares the cosine similarity with
the stored *.npy files.
    cos close to 1.0 -> preprocessing, model and normalisation all match, so a
                        newly generated split can be trusted;
    cos clearly below 1 -> the model name or the preprocessing is wrong; do not
                        use the generated output in that case.

Usage
-----
    # generate the dev split (CPU is enough, about 1-3 minutes)
    python gen_clip_features.py --data_path /path/to/data/f30k \
        --clip_model_path /path/to/CLIP/open_clip_pytorch_model.bin \
        --verify test --splits dev

    # verification only, no generation
    python gen_clip_features.py --data_path /path/to/data/f30k --verify test --verify_only

    # force regeneration
    python gen_clip_features.py --data_path /path/to/data/f30k --splits dev --overwrite

    # MS-COCO, all splits (113k unique images and 566k captions, ~10-15 min on GPU)
    #   the coco layout uses {data_path}/coco_precomp and {data_path}/images, detected automatically
    python gen_clip_features.py --data_path /path/to/data/coco --dataset coco \
        --clip_model_path /path/to/CLIP/open_clip_pytorch_model.bin \
        --splits train dev test --num_workers 8 --batch_size 128

    # resume an interrupted run (image features only, a partial file every 10 minutes)
    python gen_clip_features.py --data_path /path/to/data/coco --dataset coco --splits train --resume

Dependencies: torch, torchvision, open-clip-torch, numpy, pillow
    pip install open-clip-torch   (installs timm/ftfy/regex automatically)
"""
import argparse
import json
import os
import sys


# --------------------------------------------------------------------------
# Basic utilities
# --------------------------------------------------------------------------
def load_id_mapping(precomp_dir):
    """Load id_mapping.json.

    The file does not live in the same place for every dataset:
      * f30k: both {data_path}/f30k_precomp/id_mapping.json and {data_path}/id_mapping.json exist;
      * coco: only {data_path}/id_mapping.json exists (coco_precomp has none).
    This matches loc_mapping = {data_base}/id_mapping.json in lib/image_caption.py,
    so the precomputed directory is searched first and its parent afterwards.
    """
    cands = [os.path.join(precomp_dir, 'id_mapping.json'),
             os.path.join(os.path.dirname(os.path.abspath(precomp_dir)),
                          'id_mapping.json')]
    for path in cands:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    raise FileNotFoundError('id_mapping.json not found, tried:\n  ' + '\n  '.join(cands))


def read_lines(path, strip_newline_only=False):
    with open(path, 'r', encoding='utf-8') as f:
        if strip_newline_only:
            return [l.rstrip('\n') for l in f if l.strip() != '']
        return [l.strip() for l in f if l.strip() != '']


def build_file_lists(precomp_dir, split, id_to_path):
    """Return (image_names, captions) in the exact row order of {split}_ids.txt and {split}_caps.txt."""
    ids_path = os.path.join(precomp_dir, '{}_ids.txt'.format(split))
    caps_path = os.path.join(precomp_dir, '{}_caps.txt'.format(split))
    if not os.path.exists(ids_path):
        raise FileNotFoundError('missing {}'.format(ids_path))
    if not os.path.exists(caps_path):
        raise FileNotFoundError('missing {}'.format(caps_path))

    image_ids = read_lines(ids_path)
    captions = read_lines(caps_path, strip_newline_only=True)

    image_names = []
    for i, image_id in enumerate(image_ids):
        key = str(image_id)
        if key not in id_to_path:
            raise KeyError('id={} not found in id_mapping.json (line {})'.format(image_id, i))
        image_names.append(id_to_path[key])
    return image_names, captions


def resolve_weights(opt, data_path):
    candidates = [opt.clip_model_path] if opt.clip_model_path else []
    if not opt.clip_model_path:
        candidates += [
            os.path.join(data_path, 'CLIP', 'open_clip_pytorch_model.bin'),
            os.path.join(os.path.dirname(os.path.abspath(data_path)), 'CLIP',
                         'open_clip_pytorch_model.bin'),
            os.path.join('CLIP', 'open_clip_pytorch_model.bin'),
        ]
    weights = next((p for p in candidates if p and os.path.exists(p)), None)
    return weights, candidates


def _save_partial(partial_path, arr, done, np_dtype):
    """Atomically write the unique-image features and the completed row count for resuming."""
    import numpy as np
    try:
        tmp = partial_path + '.tmp.npy'
        np.save(tmp, arr[:done].astype(np_dtype, copy=False))
        os.replace(tmp, partial_path)
        with open(partial_path + '.done', 'w') as f:
            f.write(str(int(done)))
    except Exception as e:
        print('[WARN] failed to write the partial result (continuing anyway): {}'.format(e))


class ImageListDataset(object):
    """Read images one by one from a file-name list and apply the CLIP preprocessing.

    This must stay a module-level class. On Windows, DataLoader(num_workers>0)
    starts workers with spawn and pickles the dataset object, so a class defined
    inside a function raises `AttributeError: Can't get local object`. Only Linux
    with fork is immune to that.
    """

    def __init__(self, names, image_base, preprocess):
        self.names = names
        self.image_base = image_base
        self.preprocess = preprocess

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        from PIL import Image
        p = os.path.join(self.image_base, self.names[i])
        with Image.open(p) as im:
            img = im.convert('RGB')
        return self.preprocess(img), i


def encode_images(model, preprocess, names, image_base, device, batch_size, num_workers,
                  log_prefix='    img', dtype='float32', amp=True, partial_path=None,
                  save_every_s=600, resume=False):
    """Encode images in the order of names, deduplicating by file name.

      * The scatter-back step indexes the unique list directly, which keeps the
        whole routine O(N) even for 113k unique images.
      * --amp runs half precision on cuda (about 2x faster) and --dtype float16
        halves the size on disk.
      * Resuming is supported: the unique-image features are written every
        save_every_s seconds, and --resume continues from the completed row.
    """
    import time
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    np_dtype = np.float16 if str(dtype).lower().startswith('float16') else np.float32

    # Deduplicate: unique file names and their positions in the original list
    uniq = list(dict.fromkeys(names))
    name_to_pos = {n: [] for n in uniq}
    for i, n in enumerate(names):
        name_to_pos[n].append(i)
    n_uniq = len(uniq)

    uniq_feats = np.zeros((n_uniq, 512), dtype=np_dtype)
    start = 0
    if resume and partial_path and os.path.exists(partial_path) and \
            os.path.exists(partial_path + '.done'):
        try:
            prev = np.load(partial_path)
            done_prev = int(open(partial_path + '.done').read().strip())
            if prev.ndim == 2 and prev.shape[1] == 512 and 0 < done_prev <= n_uniq:
                uniq_feats[:done_prev] = prev[:done_prev].astype(np_dtype, copy=False)
                start = done_prev
                print('{}: --resume continuing from row {} / {}'.format(log_prefix, start, n_uniq))
        except Exception as e:
            print('{}: --resume failed ({}), restarting from scratch'.format(log_prefix, e))
            start = 0
    tail = uniq[start:]
    loader = DataLoader(ImageListDataset(tail, image_base, preprocess),
                        batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(str(device).startswith('cuda')),
                        persistent_workers=(num_workers > 0))
    if n_uniq < len(names):
        print('{}: deduplicated {} -> {} unique images'.format(log_prefix, len(names), n_uniq))

    use_amp = bool(amp) and str(device).startswith('cuda')
    t0 = time.time()
    t_save = t0
    done = start
    with torch.no_grad():
        for bi, (batch, idx) in enumerate(loader):
            batch = batch.to(device, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    emb = model.encode_image(batch)
            else:
                emb = model.encode_image(batch)
            emb = F.normalize(emb.float(), dim=-1)   # matches the existing *_clip_imgs.npy
            uniq_feats[idx.numpy() + start] = emb.cpu().numpy().astype(np_dtype, copy=False)
            done = min(start + (bi + 1) * batch_size, n_uniq)
            if bi % 20 == 0 or done >= n_uniq:
                el = time.time() - t0
                eta = el / max(done - start, 1) * (n_uniq - done)
                print('{}: {}/{} ({:.1f}%)  elapsed {:.1f}min  eta {:.1f}min'.format(
                    log_prefix, done, n_uniq, 100.0 * done / n_uniq, el / 60, eta / 60),
                    flush=True)
            if partial_path and (time.time() - t_save > save_every_s):
                _save_partial(partial_path, uniq_feats, done, np_dtype)
                t_save = time.time()
    if partial_path:
        _save_partial(partial_path, uniq_feats, done, np_dtype)

    # Scatter back: unique images -> one row per entry of names (O(N))
    feats = np.zeros((len(names), 512), dtype=np_dtype)
    for k, n in enumerate(uniq):
        v = uniq_feats[k]
        for pos in name_to_pos[n]:
            feats[pos] = v
    return feats


def encode_texts(model, tokenizer, captions, device, batch_size, log_prefix='    cap',
                 dtype='float32', amp=True):
    """Encode captions in order, one by one and without deduplication."""
    import time
    import numpy as np
    import torch
    import torch.nn.functional as F

    np_dtype = np.float16 if str(dtype).lower().startswith('float16') else np.float32
    feats = np.zeros((len(captions), 512), dtype=np_dtype)
    bs = batch_size * 4
    use_amp = bool(amp) and str(device).startswith('cuda')
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(captions), bs):
            chunk = captions[start:start + bs]
            tokens = tokenizer(chunk).to(device)
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    emb = model.encode_text(tokens)
            else:
                emb = model.encode_text(tokens)
            emb = F.normalize(emb.float(), dim=-1)
            feats[start:start + len(chunk)] = emb.cpu().numpy().astype(np_dtype, copy=False)
            if (start // bs) % 20 == 0 or start + bs >= len(captions):
                done = min(start + bs, len(captions))
                el = time.time() - t0
                eta = el / max(done, 1) * (len(captions) - done)
                print('{}: {}/{} ({:.1f}%)  elapsed {:.1f}min  eta {:.1f}min'.format(
                    log_prefix, done, len(captions), 100.0 * done / len(captions),
                    el / 60, eta / 60), flush=True)
    return feats


def verify(model, preprocess, tokenizer, precomp_dir, image_base, split, n, device,
           batch_size, num_workers):
    """Re-encode the first n rows of a split and compare cosines with the stored features."""
    import numpy as np

    img_path = os.path.join(precomp_dir, '{}_clip_imgs.npy'.format(split))
    cap_path = os.path.join(precomp_dir, '{}_clip_caps.npy'.format(split))
    if not (os.path.exists(img_path) and os.path.exists(cap_path)):
        print('[VERIFY] skipped: no existing features for {}'.format(split))
        return None

    id_to_path = load_id_mapping(precomp_dir)
    names, captions = build_file_lists(precomp_dir, split, id_to_path)
    n = min(n, len(names), len(captions))

    print('\n[VERIFY] re-encoding the first {} rows of {} and comparing with the stored npy ...'.format(n, split))
    my_img = encode_images(model, preprocess, names[:n], image_base, device,
                           batch_size, num_workers, log_prefix='    [v] img')
    my_cap = encode_texts(model, tokenizer, captions[:n], device, batch_size,
                          log_prefix='    [v] cap')
    ref_img = np.load(img_path, mmap_mode='r')[:n]
    ref_cap = np.load(cap_path, mmap_mode='r')[:n]

    def cosstats(a, b):
        a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
        b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
        c = (a * b).sum(axis=1)
        return c

    ci = cosstats(my_img, ref_img)
    cc = cosstats(my_cap, ref_cap)
    print('[VERIFY] image cos: mean={:.6f}  min={:.6f}'.format(ci.mean(), ci.min()))
    print('[VERIFY] text  cos: mean={:.6f}  min={:.6f}'.format(cc.mean(), cc.min()))

    ok = (ci.min() > 0.99) and (cc.min() > 0.99)
    if ok:
        print('[VERIFY] >>> PASS: the pipeline matches the stored features, the new split can be used.')
    else:
        print('[VERIFY] >>> FAIL: the model name or preprocessing does not match the stored features!')
        print('         Check --clip_model_name (ViT-B-32 vs ViT-B-32-quickgelu) and the image preprocessing.')
        if ci.min() <= 0.99 < cc.min():
            print('         Images differ while texts match -> look at the image preprocessing or the visual branch.')
        if cc.min() <= 0.99 < ci.min():
            print('         Texts differ while images match -> the activation (quick_gelu) is probably wrong.')
    return ok


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', required=True,
                    help='root directory of the dataset')
    ap.add_argument('--dataset', default='f30k', help='f30k or coco')
    ap.add_argument('--splits', nargs='+', default=['dev'],
                    help='splits to generate or complete (dev by default)')
    ap.add_argument('--precomp_dir', default='',
                    help='region-feature directory; empty means {data_path}/f30k_precomp')
    ap.add_argument('--image_base', default='',
                    help='raw image directory; empty means {data_path}/flickr30k-images')
    ap.add_argument('--clip_model_path', default='',
                    help='open_clip ViT-B-32 weights; empty means look under data_path/CLIP')
    ap.add_argument('--clip_model_name', default='ViT-B-32',
                    help='open_clip model name; LAION-2B weights need ViT-B-32, OpenAI weights need ViT-B-32-quickgelu')
    ap.add_argument('--batch_size', type=int, default=64,
                    help='image batch size; 128 is fine on GPU, 32 is recommended on CPU')
    ap.add_argument('--num_workers', type=int, default=0,
                    help='4 on Windows (the script must be importable), 0 if PIL/spawn errors appear, 8 on Linux')
    ap.add_argument('--overwrite', action='store_true', help='regenerate even if the files exist')
    ap.add_argument('--amp', type=int, default=1,
                    help='1 enables half precision on cuda (about 2x faster), 0 disables it (closest to fp32)')
    ap.add_argument('--dtype', default='float32', choices=['float32', 'float16'],
                    help='on-disk precision; float16 halves the size and is cast back to float32 on load')
    ap.add_argument('--resume', action='store_true',
                    help='resume from {split}_clip_imgs.partial.npy[.done]')
    ap.add_argument('--save_every_s', type=int, default=600,
                    help='interval in seconds between partial writes of the image features')
    ap.add_argument('--verify', default='',
                    help='split used for self-verification (e.g. test); re-encodes --verify_n rows and compares')
    ap.add_argument('--verify_n', type=int, default=32, help='number of rows used for self-verification')
    ap.add_argument('--verify_only', action='store_true', help='run the self-verification only')
    opt = ap.parse_args()

    try:
        import numpy as np
        import torch
        import open_clip
    except ImportError as e:
        print('[ERROR] missing dependency: {}\nrun pip install open-clip-torch torch torchvision numpy pillow'
              .format(e))
        sys.exit(1)

    data_path = opt.data_path
    # The coco precomputed directory is {data_path}/coco_precomp.
    precomp_dir = opt.precomp_dir or os.path.join(
        data_path, 'f30k_precomp' if opt.dataset == 'f30k' else 'coco_precomp')
    image_base = opt.image_base or os.path.join(
        data_path, 'flickr30k-images' if opt.dataset == 'f30k' else 'images')

    weights, candidates = resolve_weights(opt, data_path)
    if weights is None:
        print('[ERROR] CLIP weights not found, tried:\n  ' + '\n  '.join(c for c in candidates if c))
        print('Pass the path explicitly with --clip_model_path.')
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[INFO] data_path   = {}'.format(data_path))
    print('[INFO] precomp_dir = {}'.format(precomp_dir))
    print('[INFO] image_base  = {}'.format(image_base))
    print('[INFO] device      = {}'.format(device))
    print('[INFO] loading CLIP {} <- {}'.format(opt.clip_model_name, weights))
    model, _, preprocess = open_clip.create_model_and_transforms(
        opt.clip_model_name, pretrained=weights)
    tokenizer = open_clip.get_tokenizer(opt.clip_model_name)
    model = model.to(device).eval()

    # ---------- 1) self-verification ----------
    if opt.verify:
        verify(model, preprocess, tokenizer, precomp_dir, image_base,
               opt.verify, opt.verify_n, device, opt.batch_size, opt.num_workers)
        if opt.verify_only:
            return

    # ---------- 2) generation ----------
    id_to_path = load_id_mapping(precomp_dir)
    print('[INFO] splits      = {}'.format(opt.splits))
    for split in opt.splits:
        out_img = os.path.join(precomp_dir, '{}_clip_imgs.npy'.format(split))
        out_cap = os.path.join(precomp_dir, '{}_clip_caps.npy'.format(split))
        need_img = opt.overwrite or not os.path.exists(out_img)
        need_cap = opt.overwrite or not os.path.exists(out_cap)
        if not (need_img or need_cap):
            print('[SKIP] {} already exists (use --overwrite to regenerate)'.format(split))
            continue

        names, captions = build_file_lists(precomp_dir, split, id_to_path)
        print('\n[INFO] {}: {} image rows / {} captions'.format(split, len(names), len(captions)))

        # Row-count check: clip_imgs must have as many rows as ims.npy, and clip_caps as many
        # as caps.txt, because image_caption.py indexes both directly by row.
        ims_path = os.path.join(precomp_dir, '{}_ims.npy'.format(split))
        if os.path.exists(ims_path):
            n_ims = int(np.load(ims_path, mmap_mode='r').shape[0])
            if len(names) != n_ims:
                print('[FATAL] row count mismatch: {}_ids.txt={} but {}_ims.npy={}\n'
                      '        the generated clip_imgs would be misaligned with the region features, aborting.'
                      .format(split, len(names), split, n_ims))
                sys.exit(2)
            print('[CHECK] row counts consistent: ids={} == ims={} | caps={}'.format(
                len(names), n_ims, len(captions)))

        partial = os.path.join(precomp_dir, '{}_clip_imgs.partial.npy'.format(split))
        if need_img:
            feats = encode_images(model, preprocess, names, image_base, device,
                                  opt.batch_size, opt.num_workers,
                                  dtype=opt.dtype, amp=bool(opt.amp),
                                  partial_path=partial, resume=opt.resume,
                                  save_every_s=opt.save_every_s)
            tmp = out_img + '.tmp.npy'
            np.save(tmp, feats)
            os.replace(tmp, out_img)          # atomic write, so a crash never leaves a half-written file
            print('[OK] {} shape={} dtype={}'.format(out_img, feats.shape, feats.dtype))
            for suf in ('', '.done'):
                if os.path.exists(partial + suf):
                    os.remove(partial + suf)

        if need_cap:
            feats = encode_texts(model, tokenizer, captions, device, opt.batch_size,
                                 dtype=opt.dtype, amp=bool(opt.amp))
            tmp = out_cap + '.tmp.npy'
            np.save(tmp, feats)
            os.replace(tmp, out_cap)
            print('[OK] {} shape={} dtype={}'.format(out_cap, feats.shape, feats.dtype))

    # ---------- 3) output manifest with md5, useful to verify an upload ----------
    import hashlib
    print('\n[CHECKSUM] md5 of the generated files:')
    for split in opt.splits:
        for kind in ('clip_imgs', 'clip_caps'):
            p = os.path.join(precomp_dir, '{}_{}.npy'.format(split, kind))
            if os.path.exists(p):
                h = hashlib.md5()
                with open(p, 'rb') as f:
                    for chunk in iter(lambda: f.read(1 << 20), b''):
                        h.update(chunk)
                sz = os.path.getsize(p) / 1024.0 / 1024.0
                print('  {:<26s} {:>9.1f} MB  md5={}'.format(
                    os.path.basename(p), sz, h.hexdigest()))

    print('\n[DONE] finished. If the files were generated locally, upload the npy files to the same directory on the server.')


if __name__ == '__main__':
    main()
