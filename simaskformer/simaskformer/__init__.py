# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MaskDINO https://github.com/IDEA-Research/MaskDINO by Tan-Cong Nguyen
# ------------------------------------------------------------------------
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_simaskformer_config

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.detr_dataset_mapper import DetrDatasetMapper

from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

from .data.dataset_mappers.polyp_ins_semantic_dataset_mapper import (
    PolypInsSemanticDatasetMapper,
)
from .data.dataset_mappers.vofo_semantic_dataset_mapper import (
    VoFoSemanticDatasetMapper,
)
from .data.dataset_mappers.isic_semantic_dataset_mapper import (
    ISICSemanticDatasetMapper,
)

# models
from .simaskformer import SiMaskFormer
# from .data.datasets_detr import coco
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
from .evaluation.semantic_evaluation import PolypDBSemSegEvaluator
from .evaluation.isic_semantic_evaluation import ISICSemSegEvaluator
from .evaluation.semantic_vofo_evaluation import VoFoSemSegEvaluator
# util
from .utils import box_ops, misc, utils
