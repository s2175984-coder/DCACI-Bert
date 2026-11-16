# -*- coding: utf-8 -*-
"""
gen_patch_features.py -- generate CLIP patch-token features for --use_patch_fusion 1.

Why this exists
---------------
The visual side normally has two streams:
    (a) Faster-RCNN region features   36 x 2048   ({split}_ims.npy)
    (b) CLIP global embedding         512         ({split}_clip_imgs.npy)
CLIP ViT-B/32 already splits the image into 7x7 = 49 patches and encodes them,
producing a 49 x 768 patch-token sequence after ln_post. Applying the same
proj (768 -> 512) maps those tokens into the space of the global embedding, so
no extra model and no extra download are needed: the weights are the
open_clip_pytorch_model.bin already used by the global branch.
These features carry the spatial information of where a concept sits in the image.

What the script does
--------------------
For every unique image of every split it extracts (49, 512) patch features and
writes them as

    dedup (default)  {split}_clip_patches.npy        (N_uniq, 49, 512)
                     {split}_clip_patches_index.npy  (N_row,)  int32   row -> unique image
    row              {split}_clip_patches.npy        (N_row, 49, 512)   no index

Both layouts are equivalent for the dataloader, which always looks the patches up
by img_index. The dedup layout only saves disk space: f30k train has 29000 unique
images against 145000 rows, a factor of five.

Why float16 and dedup by default
--------------------------------
    f30k train : 29000 x 49 x 512 x 2B =  1.4 GB   (7.3 GB in the row layout)
    coco train : 113k   x 49 x 512 x 2B =  5.7 GB   (28  GB in the row layout)
The loader casts back with .astype(float32), which is numerically equivalent
because CLIP outputs are of order 1 and fp16 precision is sufficient.

Row-index convention, identical to gen_clip_features.py and lib/image_caption.py
--------------------------------------------------------------------------------
    image row i -> line i of {split}_ids.txt -> id_mapping.json -> file name
Note that f30k train stores five captions per image, so clip_imgs also repeats
the rows 145000 times, while the patch features are stored once per unique image
and _index.npy restores the correspondence.

Verify before generating
------------------------
The script first runs the token-extraction pipeline on --verify images and
compares l2norm(proj(CLS token)) with the official model.encode_image(img).
    cos > 0.9999 -> the token sequence, projection and spatial layout all match;
    otherwise    -> the open_clip version, preprocessing or projection layer is
                    wrong, and the generated output must not be used.

Usage
-----
    # CPU run over the three f30k splits (about 10-20 minutes)
    python gen_patch_features.py --data_path /path/to/data/f30k \
        --clip_model_path /path/to/CLIP/open_clip_pytorch_model.bin \
        --splits train dev test

    # verification only
    python gen_patch_features.py --data_path /path/to/data/f30k --verify_only

    # GPU run, same command plus a larger batch
    python gen_patch_features.py --data_path /path/to/data/f30k \
        --clip_model_path /path/to/CLIP/open_clip_pytorch_model.bin \
        --splits train dev test --num_workers 8 --batch_size 128

    # MS-COCO (needs about 6 GB of disk, keep the default dedup layout)
    python gen_patch_features.py --data_path /path/to/data/coco --dataset coco \
        --clip_model_path /path/to/CLIP/open_clip_pytorch_model.bin \
        --splits train dev test --num_workers 8 --batch_size 128

    # resume an interrupted run (dedup layout only)
    python gen_patch_features.py --data_path /path/to/data/f30k --splits train --resume

Dependencies: torch, torchvision, open-clip-torch, numpy, pillow
"""
import argparse
import hashlib
import json
import os
import sys


# --------------------------------------------------------------------------
# Basic utilities, shared with gen_clip_features.py so that the row-index convention matches
# --------------------------------------------------------------------------
def load_id_mapping(precomp_dir):
    """Load id_mapping.json (f30k has one inside and one outside precomp, coco only outside)."""
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
    """Return (image_names, captions) in the exact row order of {split}_ids.txt."""
    ids_path = os.path.join(precomp_dir, '{}_ids.txt'.format(split))
    caps_path = os.path.join(precomp_dir, '{}_caps.txt'.format(split))
    if not os.path.exists(ids_path):
        raise FileNotFoundError('missing {}'.format(ids_path))
    image_ids = read_lines(ids_path)
    captions = read_lines(caps_path, strip_newline_only=True) if os.path.exists(caps_path) else []
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


def md5_of(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Patch-token extraction
# --------------------------------------------------------------------------
def expected_num_patches(v):
    """Infer the number of patches from positional_embedding (total tokens minus the CLS token).

    ViT-B/32 at 224x224 gives pos_embed (50, 768), hence 49 patches (7x7).
    The value is reported by the verification step and cross-checked there.
    """
    pe = getattr(v, 'positional_embedding', None)
    if pe is None:
        return None
    if pe.dim() == 3:                 # some versions store (1, N, D)
        return int(pe.shape[1]) - 1
    return int(pe.shape[0]) - 1


def make_patch_extractor(model):
    """Return fn(batch) -> (pooled_raw, patches_raw), both in the 512-d projected space.

    The implementation mirrors the three inner steps of VisionTransformer.forward in
    open_clip 3.x
        x = visual._embeds(img)             # conv patch embed + CLS + pos + ln_pre
        x = visual.transformer(x)           # blocks
        pooled, tokens = visual._pool(x)    # ln_post + CLS pooling
    Important detail: the tokens returned by _pool already exclude the CLS token,
    because open_clip's _global_pool applies x[:, 1:] internally. Slicing them again
    silently drops one patch (49 -> 48) and the CLS self-check cannot detect it, so
    the patch count derived from positional_embedding is asserted here as a guard.

    If a build has no _embeds/_pool, the code falls back to a forward hook on the
    transformer. That path returns the full sequence including CLS, so the first
    token has to be dropped explicitly.
    """
    v = model.visual
    proj = getattr(v, 'proj', None)
    if proj is None:
        raise RuntimeError('the visual tower of this open_clip build has no proj parameter, so patch '
                           'tokens cannot be mapped to the global feature space; check --clip_model_name.')
    expect = expected_num_patches(v)

    use_private = hasattr(v, '_embeds') and hasattr(v, '_pool')

    if use_private:
        def _extract(batch):
            x = v._embeds(batch)
            x = v.transformer(x)
            pooled, tokens = v._pool(x)          # tokens already exclude CLS
            if expect is not None and tokens.size(1) != expect:
                # Some builds return the CLS token from _pool; decide by length instead of assuming
                if tokens.size(1) == expect + 1:
                    tokens = tokens[:, 1:]
                else:
                    raise RuntimeError(
                        'patch count disagrees with positional_embedding: tokens=%d, expected %d '
                        '(open_clip version %s)' % (tokens.size(1), expect,
                                                 getattr(__import__('open_clip'),
                                                         '__version__', '?')))
            return pooled @ proj, tokens @ proj   # (B,512) / (B,P,512)
        return _extract

    # ---- fallback: hook the transformer output (works without private methods) ----
    holder = {}

    def _hook(module, inp, out):
        holder['x'] = out

    def _extract(batch):
        h = v.transformer.register_forward_hook(_hook)
        try:
            pooled = model.encode_image(batch)
        finally:
            h.remove()
        x = holder.pop('x')                       # (B, 1+P, width), includes CLS
        if hasattr(v, 'ln_post'):
            x = v.ln_post(x)
        return pooled, x[:, 1:] @ proj            # drop CLS explicitly

    return _extract


def embed_and_patches(extract_fn, model, preprocess, names, image_base, device,
                      batch_size, num_workers, log_prefix='    patch', amp=True,
                      max_images=0, partial_path=None, save_every_s=600, resume=False):
    """Extract patches in the order of names, computing each unique image only once.

    :return: (uniq_patches (N_uniq, P, 512) float32, uniq_names, pos_list)
    """
    import time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    fn = extract_fn if extract_fn is not None else make_patch_extractor(model)

    uniq = list(dict.fromkeys(names))
    if max_images and max_images > 0:
        uniq = uniq[:max_images]
    name_to_pos = {n: [] for n in uniq}
    uniq_set = set(uniq)
    for i, n in enumerate(names):
        if n in uniq_set:
            name_to_pos[n].append(i)
    n_uniq = len(uniq)

    patches = None
    start = 0
    if resume and partial_path and os.path.exists(partial_path) and \
            os.path.exists(partial_path + '.done'):
        try:
            prev = np.load(partial_path)
            done_prev = int(open(partial_path + '.done').read().strip())
            if prev.ndim == 3 and 0 < done_prev <= n_uniq:
                patches = np.zeros((n_uniq,) + prev.shape[1:], dtype=np.float32)
                patches[:done_prev] = prev[:done_prev]
                start = done_prev
                print('{}: --resume continuing from unique image {} / {}'.format(log_prefix, start, n_uniq))
        except Exception as e:
            print('{}: --resume failed ({}), restarting from scratch'.format(log_prefix, e))
            start = 0

    # The first forward pass determines the patch count, the array is allocated afterwards
    _first = None
    if patches is None:
        # Probe the patch count on the first image
        tail0 = uniq[start:start + 1]
        if tail0:
            probe_loader = DataLoader(ImageListDataset(tail0, image_base, preprocess),
                                      batch_size=1, shuffle=False, num_workers=0)
            with torch.no_grad():
                b0, _ = next(iter(probe_loader))
                _, p0 = fn(b0.to(device))
            _first = p0.shape
            print('{}: detected patch shape {}'.format(log_prefix, tuple(p0.shape)))
            patches = np.zeros((n_uniq,) + tuple(p0.shape[1:]), dtype=np.float32)
            patches[0] = p0[0].float().cpu().numpy()

    tail = uniq[start:]
    loader = DataLoader(ImageListDataset(tail, image_base, preprocess),
                        batch_size=batch_size, shuffle=False,
                        num_workers=num_workers,
                        pin_memory=str(device).startswith('cuda'),
                        persistent_workers=(num_workers > 0))
    if n_uniq < len(names) and max_images == 0:
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
                    _, pat = fn(batch)
            else:
                _, pat = fn(batch)
            if isinstance(pat, tuple):
                pat = pat[0]
            assign = idx.numpy() + start
            if _first is not None and bi == 0:
                # Image 0 was already written by the probe, skip it to avoid rewriting the same values
                keep = assign != 0
                assign = assign[keep]
                pat = pat[keep]
            patches[assign] = pat.float().cpu().numpy()
            done = int(assign.max()) + 1 if len(assign) else done
            if bi % 20 == 0 or done >= n_uniq:
                el = time.time() - t0
                eta = el / max(done - start - 1, 1) * (n_uniq - done)
                print('{}: {}/{} ({:.1f}%)  elapsed {:.1f}min  eta {:.1f}min'.format(
                    log_prefix, done, n_uniq, 100.0 * done / max(n_uniq, 1),
                    el / 60, eta / 60), flush=True)
            if partial_path and (time.time() - t_save > save_every_s):
                _save_partial(partial_path, patches, done)
                t_save = time.time()
    if partial_path:
        _save_partial(partial_path, patches, done)

    return patches, uniq, name_to_pos


def _save_partial(partial_path, arr, done):
    import numpy as np
    try:
        tmp = partial_path + '.tmp.npy'
        np.save(tmp, arr[:done].astype(np.float16, copy=False))
        os.replace(tmp, partial_path)
        with open(partial_path + '.done', 'w') as f:
            f.write(str(int(done)))
    except Exception as e:
        print('[WARN] failed to write the partial result (continuing anyway): {}'.format(e))


class ImageListDataset(object):
    """Read images one by one from a file-name list and apply the CLIP preprocessing.

    This must stay a module-level class. On Windows, DataLoader(num_workers>0) starts
    workers with spawn and pickles the dataset, so a locally defined class fails with
    AttributeError: Can't get local object.
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


# --------------------------------------------------------------------------
def verify_pipeline(extract_fn, model, preprocess, id_to_path, precomp_dir,
                    image_base, split, n, device):
    """Self-check: the CLS token captured by the same hook must match encode_image after projection."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    names, _ = build_file_lists(precomp_dir, split, id_to_path)
    n = min(n, len(names))
    if n <= 0:
        print('[VERIFY] skipped: no usable images for {}'.format(split))
        return True

    from torch.utils.data import DataLoader
    loader = DataLoader(ImageListDataset(names[:n], image_base, preprocess),
                        batch_size=n, shuffle=False, num_workers=0)
    with torch.no_grad():
        batch, _ = next(iter(loader))
        batch = batch.to(device)
        pooled_from_tokens, patches = extract_fn(batch)
        official = model.encode_image(batch)

    a = F.normalize(pooled_from_tokens.float(), dim=-1)
    b = F.normalize(official.float(), dim=-1)
    cos = (a * b).sum(dim=-1)
    print('\n[VERIFY] cosine between the captured CLS (after proj) and the official encode_image:')
    print('[VERIFY]   mean={:.8f}  min={:.8f}  (n={})'.format(
        cos.mean().item(), cos.min().item(), n))
    expect = expected_num_patches(model.visual)
    print('[VERIFY] patch shape = {}  (expected (B, {}, 512))'.format(
        tuple(patches.shape), expect))

    norm = patches.float().norm(dim=-1)
    print('[VERIFY] raw patch norm: mean={:.4f} min={:.4f} max={:.4f}'.format(
        norm.mean().item(), norm.min().item(), norm.max().item()))

    n_ok = (expect is None) or (patches.size(1) == expect)
    if not n_ok:
        print('[VERIFY] !! wrong patch count: got {}, positional_embedding implies {}.'
              .format(patches.size(1), expect))
        print('         Most likely CLS was sliced once too many or too few; _pool tokens already exclude it.')

    ok = cos.min().item() > 0.9999 and patches.dim() == 3 and n_ok
    if ok:
        print('[VERIFY] >>> PASS: tokens, projection and spatial layout all match, the output can be used.')
    else:
        print('[VERIFY] >>> FAIL: the captured tokens disagree with the official forward pass, do not use the output.')
        print('        Check the open_clip version ({} here), the visual type {} and whether proj exists.'.format(
            getattr(__import__('open_clip'), '__version__', '?'), type(model.visual).__name__))
    return ok


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', required=True,
                    help='root directory of the dataset')
    ap.add_argument('--dataset', default='f30k', help='f30k or coco')
    ap.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'],
                    help='splits to generate (all three by default)')
    ap.add_argument('--precomp_dir', default='',
                    help='region-feature directory; empty means {data_path}/f30k_precomp or coco_precomp')
    ap.add_argument('--image_base', default='',
                    help='raw image directory; empty means {data_path}/flickr30k-images or images')
    ap.add_argument('--clip_model_path', default='',
                    help='open_clip ViT-B-32 weights; empty means look under data_path/CLIP')
    ap.add_argument('--clip_model_name', default='ViT-B-32',
                    help='open_clip model name; LAION-2B weights need ViT-B-32, OpenAI weights need ViT-B-32-quickgelu')
    ap.add_argument('--batch_size', type=int, default=64, help='image batch size')
    ap.add_argument('--num_workers', type=int, default=0,
                    help='4 on Windows (the script must be importable), 0 if spawn/PIL errors appear, 8 on Linux')
    ap.add_argument('--layout', default='dedup', choices=['dedup', 'row'],
                    help='dedup stores unique images plus an index (default, 5x smaller); row expands per row like clip_imgs')
    ap.add_argument('--dtype', default='float16', choices=['float16', 'float32'],
                    help='on-disk precision; the loader casts back to float32, which is numerically equivalent')
    ap.add_argument('--amp', type=int, default=1, help='1 runs fp16 inference on cuda (about 2x faster)')
    ap.add_argument('--overwrite', action='store_true', help='regenerate even if the files exist')
    ap.add_argument('--resume', action='store_true', help='resume from .partial.npy[.done]')
    ap.add_argument('--save_every_s', type=int, default=600, help='interval in seconds between partial writes')
    ap.add_argument('--verify', type=int, default=8, help='number of images used for self-verification, 0 skips it')
    ap.add_argument('--verify_only', action='store_true', help='run the self-verification only')
    ap.add_argument('--max_images', type=int, default=0,
                    help='process only the first N unique images (smoke test); 0 means all')
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
    precomp_dir = opt.precomp_dir or os.path.join(
        data_path, 'f30k_precomp' if opt.dataset == 'f30k' else 'coco_precomp')
    image_base = opt.image_base or os.path.join(
        data_path, 'flickr30k-images' if opt.dataset == 'f30k' else 'images')

    weights, candidates = resolve_weights(opt, data_path)
    if weights is None:
        print('[ERROR] CLIP weights not found, tried:\n  ' + '\n  '.join(c for c in candidates if c))
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[INFO] data_path   = {}'.format(data_path))
    print('[INFO] precomp_dir = {}'.format(precomp_dir))
    print('[INFO] image_base  = {}'.format(image_base))
    print('[INFO] device      = {}'.format(device))
    print('[INFO] layout      = {}   dtype={}'.format(opt.layout, opt.dtype))
    print('[INFO] loading CLIP {} <- {}'.format(opt.clip_model_name, weights))
    model, _, preprocess = open_clip.create_model_and_transforms(
        opt.clip_model_name, pretrained=weights)
    model = model.to(device).eval()
    extract_fn = make_patch_extractor(model)

    id_to_path = load_id_mapping(precomp_dir)

    # ---------- 1) self-verification ----------
    if opt.verify and opt.verify > 0:
        vsplit = 'test' if os.path.exists(
            os.path.join(precomp_dir, 'test_ids.txt')) else opt.splits[0]
        ok = verify_pipeline(extract_fn, model, preprocess, id_to_path, precomp_dir,
                             image_base, vsplit, int(opt.verify), device)
        if opt.verify_only:
            sys.exit(0 if ok else 3)
        if not ok:
            print('\n[FATAL] self-verification failed, generation aborted (pass --verify 0 to skip it).')
            sys.exit(3)

    # ---------- 2) generation ----------
    np_dtype = np.float16 if opt.dtype == 'float16' else np.float32
    # Smoke mode (--max_images N) computes features for the first N unique images only, so
    # the remaining rows would have no value. Keeping the regular file names would make the
    # generated index point every uncomputed row at image 0, a silent misalignment, so the
    # output is prefixed with smoke_ and no index is written.
    smoke = bool(opt.max_images and opt.max_images > 0)
    if smoke:
        print('[WARN] --max_images {} enables smoke mode: outputs get a smoke_ prefix, '
              'no index is written and the regular files are left untouched.'.format(opt.max_images))
    print('[INFO] splits      = {}'.format(opt.splits))
    for split in opt.splits:
        stem = ('smoke_' if smoke else '') + '{}_clip_patches'.format(split)
        out_p = os.path.join(precomp_dir, '{}.npy'.format(stem))
        out_i = os.path.join(precomp_dir, '{}_index.npy'.format(stem))
        if os.path.exists(out_p) and not opt.overwrite:
            print('[SKIP] {} already exists (use --overwrite to regenerate)'.format(out_p))
            continue

        names, captions = build_file_lists(precomp_dir, split, id_to_path)
        print('\n[INFO] {}: {} image rows / {} captions'.format(split, len(names), len(captions)))

        # Row-count check: the row count must equal the row count of ims.npy, otherwise row lookup misaligns silently
        ims_path = os.path.join(precomp_dir, '{}_ims.npy'.format(split))
        if os.path.exists(ims_path):
            n_ims = int(np.load(ims_path, mmap_mode='r').shape[0])
            if len(names) != n_ims:
                print('[FATAL] row count mismatch: {}_ids.txt={} but {}_ims.npy={}\n'
                      '        the generated patches would be misaligned with the region features, aborting.'
                      .format(split, len(names), split, n_ims))
                sys.exit(2)
            print('[CHECK] row counts consistent: ids={} == ims={}'.format(len(names), n_ims))

        partial = os.path.join(precomp_dir, '{}.partial.npy'.format(stem))
        uniq_p, uniq_names, name_to_pos = embed_and_patches(
            extract_fn, model, preprocess, names, image_base, device,
            opt.batch_size, opt.num_workers, log_prefix='    [{}]'.format(split),
            amp=bool(opt.amp), max_images=opt.max_images,
            partial_path=partial, save_every_s=opt.save_every_s, resume=opt.resume)

        # L2 normalisation, same convention as {split}_clip_imgs.npy so cosines can be used downstream
        norm = np.linalg.norm(uniq_p, axis=-1, keepdims=True)
        uniq_p = uniq_p / np.clip(norm, 1e-12, None)

        if opt.layout == 'dedup':
            save_p = uniq_p.astype(np_dtype, copy=False)
            # row -> unique-image index
            index = np.zeros(len(names), dtype=np.int32)
            pos_of = {n: k for k, n in enumerate(uniq_names)}
            for i, n in enumerate(names):
                if n in pos_of:
                    index[i] = pos_of[n]
            tmp = out_p + '.tmp.npy'
            np.save(tmp, save_p)
            os.replace(tmp, out_p)
            print('[OK] {} shape={} dtype={}'.format(out_p, save_p.shape, save_p.dtype))
            if smoke:
                print('[WARN] smoke mode writes no index, otherwise uncomputed rows would point at image 0')
            else:
                tmp = out_i + '.tmp.npy'
                np.save(tmp, index)
                os.replace(tmp, out_i)
                print('[OK] {} shape={} dtype={}'.format(out_i, index.shape, index.dtype))
        else:   # row layout, expanded per row
            n_uniq, P, D = uniq_p.shape
            flat = np.zeros((len(names), P, D), dtype=np_dtype)
            for k, n in enumerate(uniq_names):
                for pos in name_to_pos[n]:
                    flat[pos] = uniq_p[k].astype(np_dtype, copy=False)
            tmp = out_p + '.tmp.npy'
            np.save(tmp, flat)
            os.replace(tmp, out_p)
            print('[OK] {} shape={} dtype={}'.format(out_p, flat.shape, flat.dtype))
            if os.path.exists(out_i):
                os.remove(out_i)      # the row layout needs no index

        for suf in ('', '.done'):
            if os.path.exists(partial + suf):
                os.remove(partial + suf)

    # ---------- 3) output manifest with md5 ----------
    print('\n[CHECKSUM] md5 of the generated files:')
    for split in opt.splits:
        for suf in ('clip_patches', 'clip_patches_index'):
            p = os.path.join(precomp_dir, '{}_{}.npy'.format(split, suf))
            if os.path.exists(p):
                sz = os.path.getsize(p) / 1024.0 / 1024.0
                print('  {:<30s} {:>9.1f} MB  md5={}'.format(
                    os.path.basename(p), sz, md5_of(p)))

    print('\n[DONE] finished. Upload the npy files to the same directory on the server, then enable')
    print('       --use_patch_fusion 1 --patch_pool concept when training.')


if __name__ == '__main__':
    main()
