# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MaskDINO https://github.com/IDEA-Research/MaskDINO by Tan-Cong Nguyen
# ------------------------------------------------------------------------

from detectron2.config import CfgNode as CN


def add_simaskformer_config(cfg):
    """
    Add config for SiMaskFormer.
    """
    # NOTE: configs from original mask2former
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "SiMaskFormer_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    cfg.SOLVER.T_MAX = 4333
    cfg.SOLVER.ETA_MIN = 1e-6
    cfg.SOLVER.LAST_EPOCH = -1
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # SiMaskFormer model config
    cfg.MODEL.SiMaskFormer = CN()
    cfg.MODEL.SiMaskFormer.LEARN_TGT = False

    # loss
    cfg.MODEL.SiMaskFormer.PANO_BOX_LOSS = False
    cfg.MODEL.SiMaskFormer.SEMANTIC_CE_LOSS = False
    cfg.MODEL.SiMaskFormer.SEMANTIC_CE_LOSS_KEEP_IGNORE = False
    cfg.MODEL.SiMaskFormer.DEEP_SUPERVISION = True
    cfg.MODEL.SiMaskFormer.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.SiMaskFormer.CLASS_WEIGHT = 4.0
    cfg.MODEL.SiMaskFormer.DICE_WEIGHT = 5.0
    cfg.MODEL.SiMaskFormer.MASK_WEIGHT = 5.0
    cfg.MODEL.SiMaskFormer.BOX_WEIGHT = 5.
    cfg.MODEL.SiMaskFormer.GIOU_WEIGHT = 2.
    cfg.MODEL.SiMaskFormer.PROPOSAL_WEIGHT = 5.0


    # cost weight
    cfg.MODEL.SiMaskFormer.COST_CLASS_WEIGHT = 4.0
    cfg.MODEL.SiMaskFormer.COST_DICE_WEIGHT = 5.0
    cfg.MODEL.SiMaskFormer.COST_MASK_WEIGHT = 5.0
    cfg.MODEL.SiMaskFormer.COST_BOX_WEIGHT = 5.
    cfg.MODEL.SiMaskFormer.COST_GIOU_WEIGHT = 2.
    
    # transformer config
    cfg.MODEL.SiMaskFormer.NHEADS = 8
    cfg.MODEL.SiMaskFormer.DROPOUT = 0.1
    cfg.MODEL.SiMaskFormer.DIM_FEEDFORWARD = 2048
    cfg.MODEL.SiMaskFormer.ENC_LAYERS = 0
    cfg.MODEL.SiMaskFormer.DEC_LAYERS = 6
    cfg.MODEL.SiMaskFormer.DEC_N_POINTS = 8
    cfg.MODEL.SiMaskFormer.TYPE_SAMPLING_LOCATIONS = 'mask'        #['both','mask','bbox']
    cfg.MODEL.SiMaskFormer.TYPE_MASK_EMBED = 'MaskSimpleCNN' #[MaskSimpleCNN]
    cfg.MODEL.SiMaskFormer.MASK_EMBED_SPATINAL_SHAPE_LEVEL= 0 #[None,0,1,2]    None is full-size
    
    cfg.MODEL.SiMaskFormer.INITIAL_PRED = True
    cfg.MODEL.SiMaskFormer.PRE_NORM = False
    cfg.MODEL.SiMaskFormer.BOX_LOSS = True
    cfg.MODEL.SiMaskFormer.HIDDEN_DIM = 256
    cfg.MODEL.SiMaskFormer.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.SiMaskFormer.ENFORCE_INPUT_PROJ = False
    cfg.MODEL.SiMaskFormer.TWO_STAGE = True
    cfg.MODEL.SiMaskFormer.INITIALIZE_BOX_TYPE = 'no'  # ['no', 'bitmask', 'mask2box']
    cfg.MODEL.SiMaskFormer.DN="seg"
    cfg.MODEL.SiMaskFormer.DN_NOISE_SCALE=0.4
    cfg.MODEL.SiMaskFormer.DN_NUM=100
    cfg.MODEL.SiMaskFormer.PRED_CONV=False

    cfg.MODEL.SiMaskFormer.EVAL_FLAG = 1

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8
    cfg.MODEL.SEM_SEG_HEAD.DIM_FEEDFORWARD = 1024
    cfg.MODEL.SEM_SEG_HEAD.NUM_FEATURE_LEVELS = 3
    cfg.MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS = 4
    cfg.MODEL.SEM_SEG_HEAD.FEATURE_ORDER = 'high2low'  # ['low2high', 'high2low'] high2low: from high level to low level

    # SiMaskFormer inference config
    cfg.MODEL.SiMaskFormer.TEST = CN()
    cfg.MODEL.SiMaskFormer.TEST.TEST_FOUCUS_ON_BOX = False
    cfg.MODEL.SiMaskFormer.TEST.SEMANTIC_ON = True
    cfg.MODEL.SiMaskFormer.TEST.INSTANCE_ON = False
    cfg.MODEL.SiMaskFormer.TEST.PANOPTIC_ON = False
    cfg.MODEL.SiMaskFormer.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.SiMaskFormer.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.SiMaskFormer.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False
    cfg.MODEL.SiMaskFormer.TEST.PANO_TRANSFORM_EVAL = True
    cfg.MODEL.SiMaskFormer.TEST.PANO_TEMPERATURE = 0.06
    cfg.MODEL.SiMaskFormer.TEST.VISUALIZE = False
    # cfg.MODEL.SiMaskFormer.TEST.EVAL_FLAG = 1

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.SiMaskFormer.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "SiMaskFormerEncoder"

    # transformer module
    cfg.MODEL.SiMaskFormer.TRANSFORMER_DECODER_NAME = "SiMaskFormerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.SiMaskFormer.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.SiMaskFormer.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.SiMaskFormer.IMPORTANCE_SAMPLE_RATIO = 0.75

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    cfg.Default_loading=True  # a bug in my d2. resume use this; if first time ResNet load, set it false

    #Extra config
    cfg.TEST.EVAL_START_ITER=0
