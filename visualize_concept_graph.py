# -*- coding: utf-8 -*-
"""
Concept graph interpretability visualisation
================================================================================
Shows what the dynamic association graph has actually learned.

Requirement: the checkpoint must store the criterion as its third entry, which
`VSEModel.state_dict()` does because GraphLoss holds concept_enc. A checkpoint without
that entry has no concept-graph weights, and the script stops with an explicit message
instead of drawing a random graph.

Output written to --out_dir
  01_prior_vs_dynamic.png
      Rows are the knowledge prior C, the dynamic association A_dyn and the fused result
      A_c; columns are the text, visual and all heads. This shows that the combination of
      the two sources genuinely restructures the graph rather than only reweighting it.
  02_topk_network.png
      The learned Top-K concept association network, with text concepts on the left half
      circle and visual concepts on the right, and the edge width giving the weight.
  03_alpha_and_sparsity.png
      The learned sigmoid(alpha) of each head together with the out-degree distribution of
      the nodes, which documents the prior weight and the sparsity of the graph.
  04_case_study.png
      Bar chart of the Top-N neighbours of the selected concepts, or of the highest-degree
      concepts when none are given.
  top_edges.csv
      All top edges with source, target, weight and whether the edge is cross-modal.
  adjacency_<head>_<kind>.npy
      The raw matrices, for reproduction or further plotting.

Usage
  # a real checkpoint
  python visualize_concept_graph.py --model_path runs_test/model_best.pth --out_dir viz_concept

  # self test without a checkpoint, random weights, validates the plotting pipeline
  python visualize_concept_graph.py --self_test --out_dir viz_concept_selftest

  # choose the concepts to inspect (needs concept names, see --concept_names)
  python visualize_concept_graph.py --model_path ... --concepts dog,car,run

Dependencies: torch, numpy, matplotlib, networkx
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

# Headless servers need the Agg backend set before pyplot is imported
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Render the minus sign with the default font instead of the unicode glyph,
# which is missing on some headless installations.
plt.rcParams["axes.unicode_minus"] = False

# ==================================================================
# Data loading
# ==================================================================
def load_checkpoint(path):
    import torch
    if not os.path.isfile(path):
        raise SystemExit("[error] checkpoint does not exist: {}".format(path))
    ckpt = torch.load(path, map_location="cpu")
    if "model" not in ckpt or "opt" not in ckpt:
        raise SystemExit("[error] the checkpoint has no 'model'/'opt' entries, check that train.py produced it")
    opt = ckpt["opt"]
    state = ckpt["model"]
    if not isinstance(state, (list, tuple)) or len(state) < 3:
        raise SystemExit(
            "[error] this checkpoint holds no concept-graph weights, its state_dict has {} entries.\n"
            "        state_dict() stores the criterion (GraphLoss) as the third entry.\n"
            "        Retrain with this code, or run --self_test to validate the script alone.".format(
                len(state) if isinstance(state, (list, tuple)) else "?"))
    return opt, state


def build_graphloss(opt, criterion_state):
    """Build GraphLoss on CPU in eval mode and load the concept-graph weights."""
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from graph_lib import GraphLoss          # graph_lib depends on torchtext

    gl = GraphLoss(opt)
    missing, unexpected = gl.load_state_dict(criterion_state, strict=False)
    # Every concept-graph parameter has to be matched, otherwise the plot is meaningless
    crit = [k for k in missing if k.startswith(("concept_enc", "text_linear", "visual_linear"))]
    if crit:
        raise SystemExit(
            "[error] concept-graph parameters missing from the checkpoint: {}\n"
            "        Stopping, the plot would otherwise show a random graph.".format(crit[:8]))
    gl.eval()
    for p in gl.parameters():
        p.requires_grad_(False)
    return gl


def export_detail(gl):
    """Read the concept basis out of GraphLoss and export the matrices of the three heads."""
    import torch
    with torch.no_grad():
        text_basis = gl.text_linear(gl.text_concept_nodes)
        visual_basis = gl.visual_linear(gl.visual_concept_nodes)
        detail = gl.concept_enc.export_adjacency(text_basis, visual_basis)
    return detail, text_basis.shape[0], visual_basis.shape[0]


def self_test_detail(args):
    """
    Self test with random weights: builds an MMGCN_Enc directly, without the dataset
    pickles or a checkpoint, to validate the script, its dependencies and the plotting
    pipeline before training.
    """
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from MGCN.lib.MMGCN import MMGCN_Enc

    nt, nv, d = args.self_test_text, args.self_test_visual, 64
    torch.manual_seed(args.seed)
    prior = torch.rand(nt + nv, nt + nv)
    enc = MMGCN_Enc(d, num_layers=1, sparse_threshold=0.05, topk=min(10, nt + nv - 1),
                    alpha_init=-2.0,
                    adj_prior={"adj_text": prior[:nt, :nt].clone(),
                               "adj_visual": prior[nt:, nt:].clone(),
                               "adj_all": prior.clone()},
                    edge_sampler="softmax")   # deterministic softmax, easier to reproduce
    enc.eval()
    with torch.no_grad():
        tb = torch.nn.functional.normalize(torch.randn(nt, d), dim=-1)
        vb = torch.nn.functional.normalize(torch.randn(nv, d), dim=-1)
        detail = enc.export_adjacency(tb, vb)
    return detail, nt, nv


# ==================================================================
# Concept names
# ==================================================================
def load_concept_names(opt, n_text, n_visual, explicit_path):
    """
    Concept names are not always shipped as a separate list, so the lookup order is
      1) --concept_names, one name per line, the first n_text for text concepts and the rest
         for visual concepts
      2) otherwise a few common file names are probed under Concept_annotations/
      3) failing that, TextConcept_i and VisualConcept_j placeholders are used
    :return: (names, source_description)
    """
    names = None
    src = ""
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise SystemExit("[error] the file given to --concept_names does not exist: {}".format(explicit_path))
        with open(explicit_path, "r", encoding="utf-8") as f:
            names = [l.strip() for l in f if l.strip()]
        src = explicit_path
    else:
        data_path = getattr(opt, "data_path", "")
        data_name = getattr(opt, "data_name", "f30k")
        num_attr = getattr(opt, "num_attribute", 300)
        cand_dirs = [os.path.join(data_path, "Concept_annotations"), data_path]
        cand_files = [
            "{}_concept_names_{}.txt".format(data_name, num_attr),
            "concept_names_{}.txt".format(num_attr),
            "{}_concepts_{}.txt".format(data_name, num_attr),
            "attribute_vocabulary_{}.txt".format(num_attr),
            "attribute_vocabulary.txt",
            "concepts.txt",
        ]
        for d in cand_dirs:
            for fn in cand_files:
                p = os.path.join(d, fn)
                if os.path.isfile(p):
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        names = [l.strip() for l in f if l.strip()]
                    src = p
                    break
            if names:
                break

    if not names:
        names = ["TextConcept_{}".format(i) for i in range(n_text)] + \
                ["VisualConcept_{}".format(j) for j in range(n_visual)]
        src = "(no concept-name list found, index placeholders used; pass --concept_names to fix)"
    elif len(names) < n_text + n_visual:
        # Only half as many names as nodes, which happens when text and visual share one set
        if len(names) >= n_text:
            names = names[:n_text] + names[:n_visual]
            src += " (only {} names, the visual side reuses them)".format(len(names[:n_text]))
        else:
            pad = ["Concept_{}".format(i) for i in range(len(names), n_text + n_visual)]
            names = names + pad
            src += " (fewer than {} names, padded with placeholders)".format(n_text + n_visual)
    return names[:n_text + n_visual], src


# ==================================================================
# Top edges
# ==================================================================
def top_edges(A, k, exclude_self=True):
    """Take the top-k edges of each row and return [(src, dst, w), ...] in descending weight order."""
    A = np.asarray(A, dtype=np.float64).copy()
    if exclude_self:
        np.fill_diagonal(A, 0.0)
    n = A.shape[0]
    k = max(1, min(int(k), n - 1))
    src, dst = [], []
    for i in range(n):
        idx = np.argpartition(-A[i], k - 1)[:k] if n > k else np.arange(n)
        for j in idx:
            if A[i, j] > 0:
                src.append(i)
                dst.append(int(j))
    if not src:
        return []
    w = np.array([A[s, d] for s, d in zip(src, dst)])
    order = np.argsort(-w)
    return [(int(src[o]), int(dst[o]), float(w[o])) for o in order]


# ==================================================================
# Figure 1: prior, dynamic and fused graphs side by side
# ==================================================================
def fig_prior_vs_dynamic(detail, heads, out_path):
    fig, axes = plt.subplots(3, len(heads), figsize=(5.2 * len(heads), 13.5))
    if len(heads) == 1:
        axes = axes.reshape(3, 1)
    rows = [("prior", "knowledge prior C (static)"),
            ("dynamic", "dynamic association A_dyn (learned)"),
            ("fused", "fused result A_c")]
    for r, (key, row_title) in enumerate(rows):
        for c, head in enumerate(heads):
            A = np.asarray(detail[head][key], dtype=np.float64)
            ax = axes[r, c]
            pos = A[A > 0]
            if pos.size == 0:
                ax.set_axis_off()
                ax.set_title("{} / {} (all zero)".format(head, key), fontsize=10)
                continue
            vmin = max(np.percentile(pos, 1), 1e-9)
            im = ax.imshow(A, cmap="magma", norm=LogNorm(vmin=vmin, vmax=pos.max()),
                           interpolation="nearest", aspect="auto")
            ax.set_title("{} head / {}\nshape={}  alpha={:.3f}".format(
                head, row_title, A.shape, detail[head]["alpha"]), fontsize=10)
            ax.set_xlabel("target concept")
            ax.set_ylabel("source concept")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Concept-level association: knowledge prior vs learned dynamic graph\n"
                 "(log scale; every row is row-normalised, so each row sums to one)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


# ==================================================================
# Figure 2: Top-K association network
# ==================================================================
def fig_topk_network(A, names, n_text, edges, out_path, max_edges=80,
                     top_concepts=80, label_topn=25):
    """
    With the real settings there are about 600 concept nodes (f30k with num_attribute=300),
    and drawing all of them produces an unreadable blob. The subgraph therefore keeps the
    top_concepts highest-degree nodes, and only the label_topn highest-degree among them
    are labelled.
    """
    try:
        import networkx as nx
    except ImportError:
        print("[skip] networkx is not installed, the network figure needs pip install networkx")
        return None

    n = A.shape[0]
    Ad = np.asarray(A, dtype=np.float64).copy()
    np.fill_diagonal(Ad, 0.0)
    rel = max(Ad.max() * 1e-3, 1e-12)
    deg_all = (Ad > rel).sum(axis=1)

    if top_concepts and n > top_concepts:
        keep = sorted(np.argsort(-deg_all)[:top_concepts].tolist())
        note = " (the {} highest-degree of {} concepts)".format(len(keep), n)
    else:
        keep = list(range(n))
        note = ""
    keep_set = set(keep)

    sub_edges = [e for e in edges if e[0] in keep_set and e[1] in keep_set][:max_edges]

    G = nx.DiGraph()
    for i in keep:
        G.add_node(i, name=names[i], mod="text" if i < n_text else "visual")
    for s, d, w in sub_edges:
        G.add_edge(s, d, weight=w)
    if G.number_of_edges() == 0:
        return None

    # Two half circles, text concepts on the left and visual concepts on the right, ordered by index
    kept_text = [i for i in keep if i < n_text]
    kept_vis = [i for i in keep if i >= n_text]
    pos = {}
    for a, i in enumerate(kept_text):
        ang = math.pi / 2 + math.pi * (a + 0.5) / max(len(kept_text), 1)
        pos[i] = (math.cos(ang), math.sin(ang))
    for b, i in enumerate(kept_vis):
        ang = -math.pi / 2 + math.pi * (b + 0.5) / max(len(kept_vis), 1)
        pos[i] = (math.cos(ang), math.sin(ang))

    ws = np.array([G[u][v]["weight"] for u, v in G.edges()], dtype=np.float64)
    wmax = ws.max() if ws.size else 1.0
    colors = ["#d62728" if G.nodes[i]["mod"] == "text" else "#1f77b4" for i in G.nodes()]
    deg_sub = dict(G.degree())

    fig, ax = plt.subplots(figsize=(14.5, 11))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#7f7f7f",
                           width=[0.4 + 3.2 * (w / wmax) for w in ws],
                           alpha=0.45, arrows=True, arrowsize=7,
                           connectionstyle="arc3,rad=0.12")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                           node_size=110 if len(keep) <= 120 else 55, alpha=0.9)
    # Label only the highest-degree nodes, otherwise the labels overlap
    lab_nodes = sorted(G.nodes(), key=lambda i: -deg_sub.get(i, 0))[:max(1, label_topn)]
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={i: names[i] for i in lab_nodes}, font_size=7)
    ax.set_title("Learned Top-{} concept associations ({} edges shown){}"
                 "\nred = text concept, blue = visual concept, edge width = association weight"
                 .format(len(sub_edges), G.number_of_edges(), note), fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


# ==================================================================
# Figure 3: alpha and sparsity
# ==================================================================
def fig_alpha_sparsity(detail, heads, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

    alphas = [detail[h]["alpha"] for h in heads]
    axes[0].bar(heads, alphas, color="#2ca02c", alpha=0.85)
    for i, v in enumerate(alphas):
        axes[0].text(i, v, "{:.3f}".format(v), ha="center", va="bottom", fontsize=11)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("sigmoid(alpha)  (higher means more weight on the prior)")
    axes[0].set_title("learned prior weight alpha\n0 = purely dynamic, 1 = purely static prior")
    axes[0].grid(axis="y", alpha=0.3)

    # The fused matrix adds the dense prior C, so almost every entry is positive while most
    # are tiny, and counting non-zeros would wrongly show a fully connected graph. Hence
    #   the dynamic graph counts entries above 0, which is genuinely sparse, since the number
    #     of edges per row is fixed by the top-k mask
    #   the fused graph counts entries above the relative threshold max*1e-3
    dyn, fus, labels = [], [], []
    for h in heads:
        Dm = np.asarray(detail[h]["dynamic"], dtype=np.float64).copy()
        Fs = np.asarray(detail[h]["fused"], dtype=np.float64).copy()
        np.fill_diagonal(Dm, 0.0)
        np.fill_diagonal(Fs, 0.0)
        rel = max(Fs.max() * 1e-3, 1e-12)
        dyn.append((Dm > 0).sum(axis=1))
        fus.append((Fs > rel).sum(axis=1))
        labels.append(h)

    positions = np.arange(len(heads))
    bp1 = axes[1].boxplot(dyn, positions=positions - 0.18, widths=0.32,
                          patch_artist=True, showfliers=False)
    bp2 = axes[1].boxplot(fus, positions=positions + 0.18, widths=0.32,
                          patch_artist=True, showfliers=False)
    for b in bp1["boxes"]:
        b.set_facecolor("#1f77b4"); b.set_alpha(0.75)
    for b in bp2["boxes"]:
        b.set_facecolor("#ff7f0e"); b.set_alpha(0.75)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("out-degree of each concept node")
    axes[1].set_title("out-degree distribution\nblue: dynamic A_dyn (>0);  orange: fused A_c (>max*1e-3)\n"
                      "the fused graph looks denser because of the dense static prior")
    axes[1].legend([bp1["boxes"][0], bp2["boxes"][0]], ["dynamic A_dyn", "fused A_c"], fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


# ==================================================================
# Figure 4: case study
# ==================================================================
def fig_case_study(A, names, targets, out_path, topn=10):
    valid = [t for t in targets if 0 <= t < A.shape[0]]
    if not valid:
        return None
    fig, axes = plt.subplots(1, len(valid), figsize=(4.6 * len(valid), 5.4), squeeze=False)
    for c, i in enumerate(valid):
        row = np.asarray(A[i], dtype=np.float64).copy()
        row[i] = -1.0
        idx = np.argsort(-row)[:topn]
        w = row[idx]
        lbl = [names[j] for j in idx]
        ax = axes[0, c]
        y = np.arange(len(idx))[::-1]
        ax.barh(y, w, color="#ff7f0e", alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(lbl, fontsize=8)
        ax.set_title("concept: {}\n(id={})".format(names[i], i), fontsize=10)
        ax.set_xlabel("association weight")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Case study: Top-{} learned neighbours of selected concepts".format(topn),
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


# ==================================================================
# Main entry
# ==================================================================
def main():
    ap = argparse.ArgumentParser(description="concept graph interpretability visualisation")
    ap.add_argument("--model_path", default="runs_test/model_best.pth", type=str,
                    help="checkpoint produced by training, including the concept-graph weights")
    ap.add_argument("--out_dir", default="viz_concept", type=str, help="output directory")
    ap.add_argument("--topk", default=8, type=int, help="Top-K neighbours kept per concept")
    ap.add_argument("--max_edges", default=80, type=int, help="maximum number of edges in the network figure")
    ap.add_argument("--network_top_concepts", default=80, type=int,
                    help="draw only this many highest-degree concepts, since all 600 are unreadable; 0 means all")
    ap.add_argument("--label_topn", default=25, type=int,
                    help="label only this many highest-degree nodes, to avoid overlapping labels")
    ap.add_argument("--case_topn", default=10, type=int, help="number of neighbours shown in the case study")
    ap.add_argument("--concepts", default="", type=str,
                    help="comma separated concept names for the case study, e.g. dog,car; empty takes the highest-degree ones")
    ap.add_argument("--concept_names", default="", type=str,
                    help="file with one concept name per line; probed automatically when unset")
    ap.add_argument("--heads", default="text,visual,all", type=str,
                    help="association heads to visualise")
    ap.add_argument("--seed", default=0, type=int)
    # ---- self test, no dataset or checkpoint needed ----
    ap.add_argument("--self_test", action="store_true",
                    help="run the whole plotting pipeline with random weights to validate the setup")
    ap.add_argument("--self_test_text", default=40, type=int, help="self test: number of text concepts")
    ap.add_argument("--self_test_visual", default=30, type=int, help="self test: number of visual concepts")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    made = []

    # ---------- load the matrices ----------
    if args.self_test:
        print("[self test] random weights, no dataset and no checkpoint are read")
        detail, n_text, n_visual = self_test_detail(args)
        names = ["TextConcept_{}".format(i) for i in range(n_text)] + \
                ["VisualConcept_{}".format(j) for j in range(n_visual)]
        name_src = "(self test: placeholder names)"
    else:
        opt, state = load_checkpoint(args.model_path)
        print("[info] checkpoint loaded: {}".format(args.model_path))
        gl = build_graphloss(opt, state[2])
        detail, n_text, n_visual = export_detail(gl)
        names, name_src = load_concept_names(opt, n_text, n_visual, args.concept_names)

    heads = [h for h in heads if h in detail]
    if not heads:
        raise SystemExit("[error] no usable association head, --heads must be a subset of: {}".format(list(detail)))

    print("[info] concept nodes: text {} / visual {} (total {})".format(n_text, n_visual, n_text + n_visual))
    print("[info] concept names from: {}".format(name_src))
    summary_lines = []

    def _emit(s):
        print(s)
        summary_lines.append(s)

    _emit("concept nodes: text {} / visual {} (total {})".format(n_text, n_visual, n_text + n_visual))
    _emit("concept names from: {}".format(name_src))

    # Expected edges per row, the smaller of the sample count and the mask top-k, which shows
    # which mechanism actually limits the sparsity
    if args.self_test:
        intended_k = 5
        thr_desc = "0.05 (fixed in the self test)"
    else:
        intended_k = max(1, min(int(getattr(opt, "edge_topk_sample", 5) or 5),
                                int(getattr(opt, "adj_topk", 10) or 10)))
        thr_desc = str(getattr(opt, "sparse_threshold", "n/a"))

    for h in heads:
        Dm = np.asarray(detail[h]["dynamic"], dtype=np.float64)
        Fs = np.asarray(detail[h]["fused"], dtype=np.float64)
        dyn_deg = (Dm > 0).sum() / Dm.shape[0]
        rel = max(Fs.max() * 1e-3, 1e-12)
        fus_deg = (Fs > rel).sum() / Fs.shape[0]
        _emit("  - {} head: shape={}  sigmoid(alpha)={:.4f} | mean out-degree of the dynamic graph={:.2f} "
              "(genuinely sparse, set by sparse_threshold and the top-k mask) | fused graph={:.2f} (threshold max*1e-3)".format(
                  h, Fs.shape, detail[h]["alpha"], dyn_deg, fus_deg))
        if dyn_deg < 0.5 * intended_k:
            _emit("    [note] the mean out-degree of this head is {:.2f}, far below the intended {}. "
                  "The sparsity is therefore limited by sparse_threshold({}), which removes most edges "
                  "before the top-k mask is applied and makes the top-k sampling ineffective. "
                  "Setting --sparse_threshold to 0.0 or a negative value lets top-k decide the graph."
                  .format(dyn_deg, intended_k, thr_desc))

    # ---------- save the raw matrices ----------
    for h in heads:
        for kind in ("prior", "dynamic", "fused"):
            p = os.path.join(args.out_dir, "adjacency_{}_{}.npy".format(h, kind))
            np.save(p, np.asarray(detail[h][kind], dtype=np.float32))
    print("[info] raw matrices written to adjacency_<head>_<kind>.npy")

    # ---------- figure 1 ----------
    p = os.path.join(args.out_dir, "01_prior_vs_dynamic.png")
    made.append(fig_prior_vs_dynamic(detail, heads, p))
    print("[figure] " + p)

    # ---------- top edges and the network figure ----------
    main_head = "all" if "all" in heads else heads[0]
    A_main = np.asarray(detail[main_head]["fused"], dtype=np.float64)
    edges = top_edges(A_main, args.topk)

    csv_path = os.path.join(args.out_dir, "top_edges.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "source_name", "target_id", "target_name",
                    "weight", "type"])
        for s, d, wt in edges:
            typ = "cross_modal" if (s < n_text) != (d < n_text) else "intra_modal"
            w.writerow([s, names[s], d, names[d], "{:.8f}".format(wt), typ])
    print("[info] " + csv_path + "  ({} edges)".format(len(edges)))

    p = os.path.join(args.out_dir, "02_topk_network.png")
    r = fig_topk_network(A_main, names, n_text, edges, p, max_edges=args.max_edges,
                         top_concepts=args.network_top_concepts,
                         label_topn=args.label_topn)
    if r:
        made.append(r)
        print("[figure] " + p)

    # ---------- figure 3 ----------
    p = os.path.join(args.out_dir, "03_alpha_and_sparsity.png")
    made.append(fig_alpha_sparsity(detail, heads, p))
    print("[figure] " + p)

    # ---------- figure 4: case study ----------
    targets = []
    if args.concepts.strip():
        want = [c.strip() for c in args.concepts.split(",") if c.strip()]
        name2idx = {}
        for i, nm in enumerate(names):
            name2idx.setdefault(nm.lower(), i)
        for c in want:
            if c.lower() in name2idx:
                targets.append(name2idx[c.lower()])
            else:
                print("[warning] concept name not found: {} (skipped)".format(c))
    if not targets:
        A_d = A_main.copy()
        np.fill_diagonal(A_d, 0.0)
        targets = list(np.argsort(-(A_d > 1e-8).sum(axis=1))[:4])
        print("[info] --concepts not given, using the four highest-degree concepts")
    p = os.path.join(args.out_dir, "04_case_study.png")
    r = fig_case_study(A_main, names, targets, p, topn=args.case_topn)
    if r:
        made.append(r)
        print("[figure] " + p)

    # Share of cross-modal edges among the top edges, that is how much of the learned
    # association is text concept to visual concept
    if edges:
        cross = sum(1 for s, d, _ in edges if (s < n_text) != (d < n_text))
        _emit("top-{} edges: {} in total, {} cross-modal ({:.1f}%) and {} intra-modal".format(
            args.topk, len(edges), cross, 100.0 * cross / len(edges), len(edges) - cross))

    with open(os.path.join(args.out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("concept graph interpretability statistics (produced by visualize_concept_graph.py)\n")
        f.write("=" * 60 + "\n")
        if not args.self_test:
            f.write("checkpoint: {}\n".format(args.model_path))
        else:
            f.write("mode: --self_test (random weights, script validation only)\n")
        for line in summary_lines:
            f.write(line + "\n")

    print("\ndone, {} figures written to {}".format(len(made), os.path.abspath(args.out_dir)))


if __name__ == "__main__":
    main()
