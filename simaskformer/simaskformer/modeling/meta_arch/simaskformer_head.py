# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MaskDINO https://github.com/IDEA-Research/MaskDINO by Tan-Cong Nguyen
# ------------------------------------------------------------------------
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

from torch import nn

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.simaskformer_decoder import build_transformer_decoder
from ..pixel_decoder.simaskformer_encoder import build_pixel_decoder


@SEM_SEG_HEADS_REGISTRY.register()
class SiMaskFormerHead(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        transformer_predictor: nn.Module,
    ):
        """
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight
        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM                                                              #256
        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES                                           #["res2", "res3", "res4", "res5"]
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,                                                                          #256
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,                                                                            #2
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,                                                                            #1
            "transformer_predictor": build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            ),
        }

    def forward(self, features, mask=None,targets=None):
        return self.layers(features, mask,targets=targets)

    def layers(self, features, mask=None,targets=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features, mask)

        predictions = self.predictor(multi_scale_features, mask_features, mask, targets=targets)

        return predictions
