import argparse
import os
DATASET_NAME='f30k' # {coco,f30k}_precomp
BACKBONE_NAME='_butd_region_bert'
PARAMS_NAME='_debug'
DATA_PATH='/root/autodl-tmp/data/f30k/'

def get_argument_parser():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='/root/autodl-tmp/data/f30k/', type=str,
                        help='path to datasets')
    parser.add_argument('--dataset', default='f30k', type=str,
                        help='dataset coco or f30k')
    parser.add_argument('--data_name', default=DATASET_NAME,
                        help='{coco,f30k}_precomp')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=5e-4, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=200, type=int,
                        help='Number of steps to logger.info and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs_test/',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='runs_test/',
                        help='Path to save the model.')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    
    parser.add_argument('--vse_mean_warmup_epochs', type=int, default=1,
                        help='The number of warmup epochs using mean vse loss') 
    
    parser.add_argument('--multi_gpu', type=int, default=0, 
                        help='whether use the multi gpus for training')
    parser.add_argument('--size_augment', type=int, default=1, 
                        help='whether use the size augmention for training')
    
    parser.add_argument('--mask_repeat', type=int, default=1, 
                        help='whether mask the repeat images in the batch for vse loss')  
    parser.add_argument('--save_results', type=int, default=1,
                        help='whether save the similarity matrix for the evaluation')
    parser.add_argument('--gpu-id', type=int, default=0, 
                        help='the gpu-id for training')
    parser.add_argument('--bert_path', type=str, default='/root/autodl-tmp/data/bert-base-uncased',
                        help='the path of pretrained checkpoint')      
    parser.add_argument("--lr_schedules", default=[15, 25], type=int, nargs="+",
                        help='Epochs at which the learning rate is decayed.')
    parser.add_argument("--decay_rate", default=0.1, type=float, 
                        help='lr decay_rate for optimizer')

    parser.add_argument('--reset_start_epoch', action='store_true',
                        help='Whether restart the start epoch when load weights')
    parser.add_argument('--backbone_warmup_epochs', type=int, default=5,
                        help='The number of epochs for warmup')
    parser.add_argument('--embedding_warmup_epochs', type=int, default=2,
                        help='The number of epochs for warming up the embedding layers')
    parser.add_argument('--input_scale_factor', type=float, default=1.0,
                        help='The factor for scaling the input image')

    parser.add_argument('--activation_type', default='tanh',
                        help='choose type of activation functions.')
    parser.add_argument('--use_BatchNorm', action='store_false',
                        help='Whether to use BN in Consensus_level_feature_learning.')
    parser.add_argument('--dropout_rate', default=0.4, type=float,
                        help='dropout rate.')
    parser.add_argument('--feature_fuse_type', default='weight_sum',
                        help='choose the fusing type for raw feature and attribute feature '
                             '(multiple|concat|adap_sum|weight_sum))')
    parser.add_argument('--fuse_weight', default=0.9, type=float,
                        help='fusing weight for raw feature and attribute feature')
    parser.add_argument('--cat_weight', default=0.9, type=float,
                        help='emb_final = emb_origin * cat_weight + '
                             'emb_fuse * (1-cat_weight) + score_concept * score_weight')
    parser.add_argument('--score_weight', default=0.2, type=float,
                        help='emb_final = emb_origin * cat_weight + '
                             'emb_fuse * (1-cat_weight) + score_concept * score_weight')
    #parser.add_argument('--lr_MLGCN', default=.0002, type=float,
                        #help='learning rate of module of MLGCN.')
    parser.add_argument('--Concept_label_ratio', default=0.4, type=float,
                        help='The ratio of concept label.')

    parser.add_argument('--n_head_concept_encoder', type=int, default=1,
                        help='The number of heads for concept encoder layers')
    parser.add_argument('--n_layers_concept_decoder', type=int, default=1,
                        help='The number of layers for concept encoder')
    parser.add_argument('--dropout_rate_concept_encoder', type=float, default=0.1,
                        help='The dropout rate of concept encoder layers')

    parser.add_argument('--num_attribute', type=int, default=300,
                        help='The number of visual concepts')#300

    # =====================================================================
    # Master switch of the concept-level branch.
    #   0 disables the whole branch: no concept encoder, no concept decoding and no
    #     gated fusion, so emb_v / emb_t fall back to the encoder outputs and
    #     emb_loss is dropped as well. CGIG and CMGA also become inactive because
    #     both depend on the concept association matrix A_c. This is the baseline
    #     reported as "w/o concept-level branch".
    #   1 enables the branch (default).
    # To measure the effect of the dynamic association alone, keep the branch on and
    # freeze the prior coefficient (--fix_adj_alpha 1) so that A_c becomes static.
    # =====================================================================
    parser.add_argument('--use_concept_graph', type=int, default=1,
                        help='1=enable the concept-level association branch (default); '
                             '0=disable it and use the encoder features as baseline. '
                             'Disabling it also deactivates CGIG, CMGA and the KL prior '
                             'regulariser, because all three depend on the concept graph.')

    # Dynamic graph head hyper-parameters.
    parser.add_argument('--num_gcn_layers', type=int, default=2,
                        help='Number of layers in the dynamic association head.')
    parser.add_argument('--sparse_threshold', type=float, default=0,#0.05
                        help='Threshold of the sparse mask that filters noisy edges in the '
                             'dynamic association head.')
    parser.add_argument('--adj_topk', type=int, default=10,
                        help='Top-k edges per node for sparse adjacency, 0 means disable')
    # =====================================================================
    # Capacity of the association scorer, a multi-head low-rank bilinear form
    # S_k = (H U_k)(H V_k)^T.
    #   rank: rank of the low-rank decomposition. 0 reproduces the default
    #         max(embed_size // 8, 16), which is 128 at embed_size=1024.
    #   Note that rank and num_heads are coupled through
    #   head_dim = max(rank // num_heads, 4), so they are not independent knobs.
    #   When plotting a capacity sweep, use the effective_rank printed in the log
    #   rather than the value requested on the command line.
    # =====================================================================
    parser.add_argument('--adj_rank', type=int, default=0,
                        help='Rank of the low-rank association scorer. 0 uses '
                             'max(embed_size//8, 16); a positive value sets it explicitly.')
    parser.add_argument('--adj_num_heads', type=int, default=4,
                        help='Number of heads in the association scorer (default 4). '
                             'head_dim = max(rank//num_heads, 4).')
    parser.add_argument('--adj_alpha', type=float, default=-2,
                        help='Initial value of the learnable coefficient alpha of the weak '
                             'prior C. The induced prior weight is w = sigmoid(alpha).')#-2
    parser.add_argument('--fix_adj_alpha', type=int, default=0,
                        help='1=freeze the prior coefficient alpha (requires_grad=False) and '
                             'treat --adj_alpha as a fixed prior weight w=sigmoid(adj_alpha); '
                             '0=learn it (default).')
    parser.add_argument('--alpha_log', type=int, default=1,
                        help='1=log sigmoid(alpha) of the three relation heads every epoch; '
                             '0=off.')

    # ===== Edge sampling. Continuous softmax is the default; Gumbel is optional.
    parser.add_argument('--edge_sampler', type=str, default='soft',
                        choices=['gumbel_topk', 'gumbel', 'soft', 'softmax'],
                        help="'soft/softmax': continuous softmax graph (default); "
                             "'gumbel': Gumbel-Softmax straight-through discrete sampling.")
    parser.add_argument('--gumbel_tau', type=float, default=1.0,
                        help='Gumbel-Softmax temperature; smaller values approach one-hot.')
    parser.add_argument('--gumbel_hard', type=int, default=0,
                        help='1=straight-through estimator producing 0/1 edges; 0=soft relaxation.')

    # =====================================================================
    # Temperature: learnable by default, with an optional initial value and an
    # optional linear annealing schedule.
    # ---------------------------------------------------------------------
    # Each relation head holds a learnable log_tau with
    # _tau() = softplus(log_tau) + 1 >= 1, initialised to sqrt(embed_size), which
    # is 32 at embed_size=1024.
    # Annealing is disabled by default; when enabled it overrides the learnable
    # temperature and drives tau linearly towards tau_min.
    # For a temperature sweep, read the effective value from the log rather than
    # the command-line value, because annealing moves tau away from tau_init.
    # =====================================================================
    parser.add_argument('--tau_init', type=float, default=0.0,
                        help='Initial temperature of the relation head. 0 keeps '
                             'sqrt(embed_size), which is 32.0 at embed_size=1024.')
    parser.add_argument('--tau_anneal_epochs', type=int, default=0,
                        help='>0 anneals tau linearly from tau_init to tau_min over the first '
                             'N epochs and freezes log_tau; 0 disables annealing (default).')
    parser.add_argument('--tau_min', type=float, default=1.1,
                        help='Final value of the annealed temperature. It must exceed 1 '
                             'because _tau() = softplus(log_tau) + 1; smaller values behave '
                             'closer to hard top-k.')

    # =====================================================================
    # Controls for the content of the knowledge prior.
    #   'real'     : concept co-occurrence / similarity prior (default).
    #   'shuffle'  : the same fixed permutation is applied to the rows and to the
    #                columns of the prior. Every concept keeps its degree, but the
    #                pairing between concepts is destroyed, so any difference from
    #                'real' can only be attributed to the semantic content of the
    #                prior rather than to scale or sparsity.
    #   'identity' : identity matrix, no prior knowledge at all apart from self loops.
    # =====================================================================
    parser.add_argument('--prior_mode', type=str, default='real',
                        choices=['real', 'shuffle', 'identity'],
                        help="'real'=concept co-occurrence prior (default); "
                             "'shuffle'=fixed random permutation of the concept indices, "
                             "which preserves the degree distribution; "
                             "'identity'=identity matrix, self loops only.")
    parser.add_argument('--edge_norm', type=str, default='row',
                        choices=['row', 'sinkhorn'],
                        help="'row'=row normalisation (random walk); 'sinkhorn'=doubly "
                             "stochastic normalisation.")
    parser.add_argument('--sinkhorn_iters', type=int, default=10,
                        help='Number of Sinkhorn-Knopp iterations.')
    # ===== Gumbel-Top-K edge sampling together with the KL prior regulariser. =====
    parser.add_argument('--edge_topk_sample', type=int, default=5,
                        help='Number of edges sampled per row by Gumbel-Top-K; it should be '
                             'smaller than adj_topk to have any effect. 0 disables sampling.')
    parser.add_argument('--kl_prior_weight', type=float, default=0.2,
                        help='Weight of the KL divergence between the edge distribution '
                             'pi=softmax(S) and the normalised knowledge prior '
                             'p=rownorm(C). 0 disables it; a positive value applies it on '
                             'top of the convex combination.')#0.2

    # ===== Batch-invariance control. =====
    parser.add_argument('--use_batch_ctx', type=int, default=0,
                        help='1=enables batch-mean context modulation, which breaks batch '
                             'invariance; intended for ablation only.')

    # ===== Making the dynamic graph take part in the retrieval path. =====
    parser.add_argument('--use_graph_inference', type=int, default=0,
                        help='1=retrieve with the concept-enhanced features, which requires '
                             'a checkpoint containing the graph weights; 0=the graph acts as '
                             'a training-time structural regulariser only.')
    parser.add_argument('--graph_inference_warmup', type=int, default=5,
                        help='Number of initial epochs during which validation does not use '
                             'the graph, because the concept graph is not yet trained. This '
                             'keeps model selection consistent with the final metric.')

    # ===== Whether the sample-level four-block module also enters the retrieval path.
    # The four-block module (AdjacencyModel together with the GNN) builds a
    # sample-to-sample graph inside a batch, so its output depends on batch
    # composition. Enabling it means a single image or sentence is no longer encoded
    # independently, which departs from the per-sample evaluation protocol of
    # image-text retrieval. It is therefore disabled by default and should be stated
    # explicitly when reporting results.
    parser.add_argument('--use_instance_inference', type=int, default=0,
                        help='1=also feed the sample-level four-block features into the '
                             'retrieval path; 0=the four-block module contributes to the '
                             'training loss only (default).')
    parser.add_argument('--instance_inference_warmup', type=int, default=5,
                        help='Number of initial epochs whose validation does not use the '
                             'sample-level four-block enhancement.')

    # ===== Training stability and negative mining. =====
    parser.add_argument('--lr_warmup_iters', type=int, default=1000,
                        help='Number of iterations over which the learning rate is ramped '
                             'linearly from zero to the base rate; 0 disables the ramp. '
                             'Without it the randomly initialised graph modules start at '
                             'the full rate and the loss oscillates strongly.')
    parser.add_argument('--trip_hard_negative', type=int, default=0,
                        help='1=mine the hardest negative in TripletLoss; 0=keep the '
                             'distance-weighted random sampling. Only used when '
                             'base_loss or gnn_loss is set to trip.')

    # ===== CLIP global branch. =====
    # Design notes:
    #   1) CLIP is frozen throughout (requires_grad=False and no_grad); only the
    #      512->1024 projection and the gate are trained, so any gain cannot be
    #      attributed to fine-tuning a stronger backbone.
    #   2) The local feature path is kept unchanged (region and word features ->
    #      Transformer aggregation -> residual). CLIP provides a global semantic
    #      query that guides local aggregation (GGLA) before gated fusion.
    #   3) The order of operations is:
    #        clip   = fc_clip(CLIP_512 -> 1024)
    #        pooled = GGLA(clip, local sequence)   # Q/K L2-normalised, optional mask
    #        f      = beta*res + (1-beta)*pooled
    #        *_emb_notnorm = f                     # before CLIP fusion, unnormalised
    #        out    = GatedFusion(l2norm(f), l2norm(clip))
    parser.add_argument('--use_clip_fusion', type=int, default=1,
                        help='1=enable the CLIP global branch; 0=baseline without CLIP.')
    parser.add_argument('--clip_source', type=str, default='online',
                        choices=['online', 'npy'],
                        help="Source of the CLIP features.\n"
                             "  'online' reads the raw image and the CLIP tokenizer and "
                             "encodes them inside the encoder under no_grad, keeping CLIP "
                             "frozen. This is the default.\n"
                             "  'npy' reads the pre-computed {split}_clip_imgs.npy and "
                             "_caps.npy files. The values are equivalent because the same "
                             "weights and the same preprocess are used, but roughly 1.5 GB "
                             "of GPU memory is saved. It is a memory-saving fallback rather "
                             "than the setting used for the reported results.")
    parser.add_argument('--clip_model_path', type=str,
                        default='/root/autodl-tmp/data/CLIP/open_clip_pytorch_model.bin',
                        help='Path to the open_clip ViT-B/32 weights.')
    parser.add_argument('--clip_dim', type=int, default=512,
                        help='Dimension of the CLIP features (512 for ViT-B/32).')
    parser.add_argument('--clip_feat_dir', type=str, default='',
                        help='Directory of the pre-computed CLIP features when '
                             'clip_source=npy. Empty means the same directory as the '
                             'region features ({data_path}/f30k_precomp or coco_precomp).')
    parser.add_argument('--clip_mha_heads', type=int, default=8,
                        help='Number of attention heads in GGLA. Memory grows linearly '
                             'with this value, so it can be lowered when memory is tight.')
    parser.add_argument('--clip_use_mask', type=int, default=1,
                        help='0=apply no padding mask; 1=mask the padding positions, which '
                             'is mathematically cleaner. Default 1.')#0
    parser.add_argument('--clip_gate_bias', type=float, default=0,
                        help='Initial bias of the gated fusion. 0 uses the default '
                             'initialisation; a positive value favours the local features '
                             'and a negative value favours the CLIP features.')#0.0
    parser.add_argument('--clip_encode_chunk', type=int, default=32,
                        help="Sample-chunk size of the CLIP forward pass in online mode. "
                             "This is a memory control only.\n"
                             "Every operator of the ViT acts within a single sample, and "
                             "there is no BatchNorm, so\n"
                             "  f(x[0:B]) == cat(f(x[0:c]), f(x[c:2c]), ...)\n"
                             "The result is numerically identical; only the transient peak "
                             "of the CLIP forward pass drops from O(B) to O(chunk).\n"
                             "0 runs the whole batch in one pass. The default of 32 keeps "
                             "batch_size=128 within a 24 GB card.")

    # ===== Activation recomputation for the BERT layers. =====
    parser.add_argument('--grad_ckpt', type=int, default=0,
                        help="Gradient checkpointing of the BERT layers.\n"
                             "1 stores only the layer inputs and recomputes the forward "
                             "pass during backpropagation, reducing the BERT activation "
                             "memory from O(layers) to O(1). The gradients are identical "
                             "because preserve_rng_state=True restores the dropout masks "
                             "exactly; the text encoder becomes about 20-30%% slower.\n"
                             "0 disables it and trades memory for speed.\n"
                             "Caption lengths have a long tail, and BERT activations grow "
                             "with batch_size times the longest sequence, so a long-caption "
                             "batch at batch_size=128 can reach about 10 GB on its own.")

    # ===== Cross-modal graph topology alignment (CMGA). =====
    parser.add_argument('--use_cmga', type=int, default=1,
                        help='1=enable the cross-modal graph topology alignment loss L_cmga.')
    parser.add_argument('--cmga_weight', type=float, default=0.05,
                        help='Weight of L_cmga.')
    parser.add_argument('--cmga_temperature', type=float, default=0.1,
                        help='InfoNCE temperature used in L_cmga.')
    parser.add_argument('--topo_mode', type=str, default='cross',
                        choices=['cross', 'intra'],
                        help="'cross'=align the visual graph with the textual graph; "
                             "'intra'=contrast the textual graph with itself.")
    # The alignment loss reads the structure through the dynamic head, so the
    # gradient of L_cmga also reaches that head. The difference between
    # --use_cmga 0 and 1 therefore mixes two effects: the InfoNCE term itself and
    # the extra training signal it provides to the dynamic head. Setting
    # --cmga_detach_graph 1 removes the second one so that each can be measured
    # separately.
    parser.add_argument('--cmga_detach_graph', type=int, default=0,
                        help='1=detach the input of the structure reader, so that the '
                             'alignment loss is still computed but no gradient reaches the '
                             'dynamic head; 0=propagate the gradient (default).')
    parser.add_argument('--cmga_graph_source', type=str, default='dyn',
                        choices=['dyn', 'prior'],
                        help="Graph used for the alignment. 'dyn'=produced by the dynamic "
                             "head (default); 'prior'=the knowledge prior C, which shows "
                             "whether the gain requires the dynamic topology.")
    parser.add_argument('--cmga_share_encoder', type=int, default=1,
                        help='1=both modalities share one structure reader, which makes the '
                             'structure descriptions comparable; 0=one reader per modality.')

    # ===== Concept-to-sample structural injection (CGIG). =====
    parser.add_argument('--use_concept_guided_graph', type=int, default=1,
                        help='1=enable concept-guided instance graph fusion (P A_c P^T).')
    parser.add_argument('--concept_guide_weight', type=float, default=0.3,
                        help='Weight w of the concept-guided relation against the '
                             'instance-level relation.')
    parser.add_argument('--concept_guide_tau', type=float, default=10.0,
                        help='Sharpness of the concept activation distribution.')
    # ---- Source of A_c used by CGIG. ----
    # CGIG computes S = rownorm(P A_c P^T) with A_c the dynamic concept graph.
    #   'dyn'      : A_c produced by the dynamic head (default).
    #   'prior'    : A_c replaced by the knowledge prior C. If this setting falls back
    #                to the level of --use_concept_guided_graph 0, the benefit of the
    #                injection depends on the dynamic graph; if it stays close to the
    #                default, any structured matrix would suffice.
    #   'identity' : A_c = I, so S = rownorm(P P^T) and no graph structure remains.
    #                This isolates whether the intermediate graph is needed at all.
    parser.add_argument('--concept_guide_source', type=str, default='dyn',
                        choices=['dyn', 'prior', 'identity'],
                        help="Source of A_c in CGIG: 'dyn'=dynamic concept graph "
                             "(default); 'prior'=knowledge prior C; 'identity'=identity "
                             "matrix.")#dyn

    # =====================================================================
    # Normalisation of the injected relation matrix.
    # ---------------------------------------------------------------------
    # The four-block module passes S as an additive bias to the transformer
    # encoder, and softmax is invariant to a constant added to an entire row.
    # Because S = rownorm(P A_c P^T) is row-normalised, every row is nearly
    # constant and the injected information is therefore close to zero.
    # Two operations, both required, restore a meaningful contribution:
    #       S <- S - row_mean(S)   # remove the row constant that softmax cancels
    #       S <- S / rms(S)        # normalise to unit RMS so that w is a true ratio
    # After row normalisation the varying part has an RMS of only about CV, so
    # scaling it directly would add almost nothing; the unit-RMS step restores the
    # ratio semantics of the fusion weight w.
    # =====================================================================
    parser.add_argument('--cgig_demean', type=int, default=1,
                        help='1=subtract the row mean of S so that the injected signal is '
                             'visible to the softmax; 0=keep S unchanged.')
    parser.add_argument('--cgig_unit_rms', type=int, default=1,
                        help='1=normalise S to unit RMS so that the fusion weight w keeps '
                             'its ratio semantics; 0=keep S unchanged.')
    # ---- Computation of the concept response P. ----
    # The plain form is P = softmax( cos(e_i, X_c) * tau / sqrt(D) ). Since
    # D = 1024 and std(cos) is about 1/sqrt(D), the standard deviation of the logits
    # is about tau/D rather than tau/sqrt(D), which is far too flat to rank concepts.
    #   'cos'    : the plain form above.
    #   'zscore' : standardise each cos row with (x-mu)/sigma before scaling by tau,
    #              which decouples the scale from D and gives a usable range of tau.
    #   'attn'   : use the cross-attention of the concept decoder as P, which is
    #              naturally peaked and needs no temperature tuning.
    parser.add_argument('--concept_response_source', type=str, default='attn',
                        choices=['cos', 'zscore', 'attn'],
                        help="Source of the concept response P: 'cos'=plain cosine form; "
                             "'zscore'=row standardisation followed by scaling by tau; "
                             "'attn'=cross-attention of the concept decoder.")#cos
    parser.add_argument('--concept_sharp_log_every', type=int, default=100,
                        help='Print a concept-sharpness diagnostic every N iterations, '
                             'covering max/mean of P, the within-row CV of S and the '
                             'visible fraction; 0 disables it.')

    # =====================================================================
    # CLIP patch branch.
    # The visual side otherwise uses (a) Faster-RCNN region features of size
    # 36x2048 and (b) a 512-d CLIP global vector. The ViT-B/32 of CLIP already
    # produces 49 patch tokens of 512 dimensions from the same weights, so this
    # branch needs no additional model or download. It supplies the spatial
    # evidence of where on the image a concept appears.
    # gen_patch_features.py pre-computes {split}_clip_patches.npy; training and
    # inference then read (B, 49, 512) by img_index, pool the patches and fuse
    # them into the visual retrieval feature.
    # =====================================================================
    parser.add_argument('--use_patch_fusion', type=int, default=0,
                        help='1=enable the CLIP patch branch, which requires '
                             '{split}_clip_patches.npy from gen_patch_features.py.')
    parser.add_argument('--patch_feat_dir', type=str, default='',
                        help='Directory of the patch features. Empty means the same '
                             'directory as the region features.')
    parser.add_argument('--patch_dim', type=int, default=512,
                        help='Dimension of a patch token (512 for ViT-B/32).')
    parser.add_argument('--patch_num', type=int, default=49,
                        help='Number of patches per image (49 for ViT-B/32 at 224x224).')
    parser.add_argument('--patch_pool', type=str, default='concept',
                        choices=['mean', 'max', 'attn', 'concept'],
                        help="Patch pooling: 'mean'/'max' are parameter-free baselines, "
                             "'attn' uses a learnable query, and 'concept' weights the "
                             "patches by the concept response.")
    parser.add_argument('--patch_gate_bias', type=float, default=-1.0,
                        help='Initial bias of the patch gate. A value of -1 starts from '
                             'the original feature and lets the branch grow in.')
    parser.add_argument('--patch_beta', type=float, default=10.0,
                        help="Sharpness of the patch weights in the 'concept' pooling "
                             "mode, i.e. the inverse softmax temperature.")

    # =====================================================================
    # Memory controls.
    # The CLIP branch adds the (K,V) activations of GGLA to the forward pass, and
    # CMGA is already the most memory-hungry part because it materialises a
    # (B, num_attribute, num_attribute) concept graph together with a
    # (B, num_attribute, D) structural representation. The options below reduce
    # the peak without changing any numerical result; only the order in which
    # random numbers are consumed can differ.
    # =====================================================================
    parser.add_argument('--cmga_chunk', type=int, default=32,
                        help='Chunk size of the CMGA loss over samples. The loss builds a '
                             'concept graph and reads a structure vector per sample, so '
                             'processing the batch in chunks is exactly equivalent while '
                             'the peak of (B,N,N) and (B,N,D) drops by chunk/B. '
                             '0 disables chunking and uses the most memory.')
    parser.add_argument('--mem_debug', type=int, default=0,
                        help='>0 prints the current, peak and reserved memory of every '
                             'stage for the first N iterations and resets the peak '
                             'statistics at the start of each one, which locates the '
                             'stage where memory runs out.')
    parser.add_argument('--region_mmap', type=int, default=0,
                        help='1=read {split}_ims.npy with mmap_mode="r" so that pages are '
                             'loaded on demand; 0=load the whole array into memory. '
                             'MS-COCO train_ims.npy is 33.4 GB, which keeps about 32.5 GiB '
                             'resident and can be killed by the cgroup OOM killer. '
                             'Large-file mmap has triggered 0xC0000005 on Windows, so the '
                             'default is 0.')

    parser.add_argument('--debug', action='store_true',
                        help='Debug mode will train only one batch size.')
    parser.add_argument('--precomp_enc_type', default='basic',
                        help='basic|backbone')


    # ---- Single-instance guard, see lib/run_guard.py. No numerical effect. ----
    parser.add_argument('--single_instance', type=int, default=1,
                        help='1=allow only one training process per logger_name using a file '
                             'lock, so that a second launch cannot compete for the GPU or '
                             'truncate the log of the first one; 0=disable the lock.')
    parser.add_argument('--min_free_gpu_mb', type=int, default=0,
                        help='Minimum free GPU memory required before training starts, in '
                             'MiB. 0 derives the threshold from the device size (60%%); a '
                             'positive value sets it explicitly and -1 only reports it.')

    return parser


def save_parameters(opt, save_path):

    varx = vars(opt)
    base_str = ''
    for key in varx:
        base_str += str(key)
        if isinstance(varx[key], dict):
            for sub_key, sub_item in varx[key].items():
                base_str += '\n\t'+str(sub_key)+': '+str(sub_item)
        else:
            base_str += '\n\t'+str(varx[key])
        base_str+='\n'
    
    with open(os.path.join(save_path, 'Parameters.txt'), 'w') as f:
        f.write(base_str)


if __name__ == '__main__':

    pass
    