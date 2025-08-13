# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
from typing import Optional, Union
import pycocotools.mask as mask_util
import torch
from PIL import Image
import copy
import time
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from detectron2.evaluation import DatasetEvaluator

_CV2_IMPORTED = True
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    _CV2_IMPORTED = False


def load_image_into_numpy_array(
    filename: str,
    copy: bool = False,
    dtype: Optional[Union[np.dtype, str]] = None,
) -> np.ndarray:
    with PathManager.open(filename, "rb") as f:
        #array = np.array(Image.open(f), copy=copy, dtype=dtype)
        array = np.asarray(Image.open(f), dtype=dtype)
    return array

def evaluate_single(pred, gt):

    pred_binary = (pred >= 0.5).float()
    pred_binary_inverse = (pred_binary == 0).float()

    gt_binary = (gt >= 0.5).float()
    gt_binary_inverse = (gt_binary == 0).float()

    TP = pred_binary.mul(gt_binary).sum()
    FP = pred_binary.mul(gt_binary_inverse).sum()
    TN = pred_binary_inverse.mul(gt_binary_inverse).sum()
    FN = pred_binary_inverse.mul(gt_binary).sum()

    if TP.item() == 0:
        # print('TP=0 now!')
        # print('Epoch: {}'.format(epoch))
        # print('i_batch: {}'.format(i_batch))
        TP = torch.Tensor([1]).cuda()

    # recall
    Recall = TP / (TP + FN)

    # Specificity or true negative rate
    if (TN + FP).item() == 0:
        Specificity = torch.tensor(0.0, device=pred.device)
    else:
        Specificity = TN / (TN + FP)


    # Precision or positive predictive value
    if (TP + FP).item() == 0:
        Precision = torch.tensor(0.0, device=pred.device)
    else:
        Precision = TP / (TP + FP)

    # F1 score = Dice
    F1 = 2 * Precision * Recall / (Precision + Recall)

    # F2 score
    F2 = 5 * Precision * Recall / (4 * Precision + Recall)

    # Overall accuracy
    ACC_overall = (TP + TN) / (TP + FP + FN + TN)

    # IoU for poly
    IoU_poly = TP / (TP + FP + FN)

    # IoU for background
    IoU_bg = TN / (TN + FP + FN)

    # mean IoU
    IoU_mean = (IoU_poly + IoU_bg) / 2.0

    return Recall, Specificity, Precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean

def Average(lst):
    return sum(lst) / len(lst)



class ISICSemSegEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        visualize=False,
        visualize_path="./visualize",
        *,
        sem_seg_loading_fn=load_image_into_numpy_array,
        num_classes=None,
        ignore_label=None,

    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            sem_seg_loading_fn: function to read sem seg file and load into numpy array.
                Default provided, but projects can customize.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)
        self.visualize=visualize
        self.visualize_path=visualize_path
        if num_classes is not None:
            self._logger.warn(
                "SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata."
            )
        if ignore_label is not None:
            self._logger.warn(
                "SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self._class_names = meta.stuff_classes
        self.sem_seg_loading_fn = sem_seg_loading_fn
        print(meta.stuff_classes)
        self._num_classes = len(meta.stuff_classes)
        if num_classes is not None:
            assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

        # This is because cv2.erode did not work for int datatype. Only works for uint8.
        self._compute_boundary_iou = True
        if not _CV2_IMPORTED:
            self._compute_boundary_iou = False
            self._logger.warn(
                """Boundary IoU calculation requires OpenCV. B-IoU metrics are
                not going to be computed because OpenCV is not available to import."""
            )
        if self._num_classes >= np.iinfo(np.uint8).max:
            self._compute_boundary_iou = False
            self._logger.warn(
                f"""SemSegEvaluator(num_classes) is more than supported value for Boundary IoU calculation!
                B-IoU metrics are not going to be computed. Max allowed value (exclusive)
                for num_classes for calculating Boundary IoU is {np.iinfo(np.uint8).max}.
                The number of classes of dataset {self._dataset_name} is {self._num_classes}"""
            )

    def reset(self):
        self._bin_metrics = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """

        if self.visualize is True and not os.path.exists(self.visualize_path): 
          os.makedirs(self.visualize_path)

        for input, output in zip(inputs, outputs):
                        
            #output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            output = output["sem_seg"].to(self._cpu_device)

            output = (output - output.min()) / (output.max() - output.min() + 1e-8)

            if self.visualize==True:
              mask_image1 = (copy.deepcopy(output).squeeze().numpy() * 255).astype(np.uint8)
              mask_image1 = Image.fromarray(mask_image1)
              mask_image1.save(self.visualize_path+'/'+os.path.basename(input["file_name"]))


            pred_bin=output.numpy().astype(float)
            
            
            gt_filename = self.input_file_to_gt_file[input["file_name"]]
            gt = self.sem_seg_loading_fn(gt_filename, dtype=int)
            #gt = gt.mean(axis=2)
            gt[gt<128]=0
            gt[gt>=128]=1
            gt_bin=gt.astype(float)

            Recall, Specificity, Precision, F1, F2, ACC_overall, IoU_lesion, IoU_bg, IoU_mean=evaluate_single(torch.from_numpy(pred_bin),torch.from_numpy(gt_bin))


            self._bin_metrics.append({"Recall":Recall,
                                          "Specificity":Specificity,
                                          "Precision":Precision,
                                          "F1":F1,
                                          "F2":F2,
                                          "ACC_overall":ACC_overall,
                                          "IoU_lesion":IoU_lesion,
                                          "IoU_bg":IoU_bg,
                                          "IoU_mean":IoU_mean
                                          })

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()

            self._bin_metrics = all_gather(self._bin_metrics)
            self._bin_metrics = list(itertools.chain(*self._bin_metrics))

            if not is_main_process():
                return


        ####
        Recall_list=[]
        Specificity_list=[]
        Precision_list=[]
        F1_list=[]
        F2_list=[]
        ACC_overall_list=[]
        IoU_lesion_list=[]
        IoU_bg_list=[]
        IoU_mean_list=[]
        print(f"Binary segmentation:{len(self._bin_metrics)} images")
        self._logger.info(f"Binary segmentation:{len(self._bin_metrics)} images")
        for metrics in self._bin_metrics:
          Recall=metrics["Recall"]
          Specificity=metrics["Specificity"]
          Precision=metrics["Precision"]
          F1=metrics["F1"]
          F2=metrics["F2"]
          ACC_overall=metrics["ACC_overall"]
          IoU_lesion=metrics["IoU_lesion"]
          IoU_bg=metrics["IoU_bg"]
          IoU_mean=metrics["IoU_mean"]
          Recall_list.append(Recall)
          Specificity_list.append(Specificity)
          Precision_list.append(Precision)
          F1_list.append(F1)
          F2_list.append(F2)
          ACC_overall_list.append(ACC_overall)
          IoU_lesion_list.append(IoU_lesion)
          IoU_bg_list.append(IoU_bg)
          IoU_mean_list.append(IoU_mean)
        print("Recall:",Average(Recall_list))
        print("Specificity:",Average(Specificity_list))
        print("Precision:",Average(Precision_list))
        print("F1:",Average(F1_list))
        print("F2:",Average(F2_list))
        print("ACC_overall:",Average(ACC_overall_list))
        print("IoU_lesion:",Average(IoU_lesion_list))
        print("IoU_bg:",Average(IoU_bg_list))
        print("IoU_mean:",Average(IoU_mean_list))
        self._logger.info(f"Recall:{Average(Recall_list)}")
        self._logger.info(f"Specificity:{Average(Specificity_list)}")
        self._logger.info(f"Precision:{Average(Precision_list)}")
        self._logger.info(f"F1:{Average(F1_list)}")
        self._logger.info(f"F2:{Average(F2_list)}")
        self._logger.info(f"ACC_overall:{Average(ACC_overall_list)}")
        self._logger.info(f"IoU_lesion:{Average(IoU_lesion_list)}")
        self._logger.info(f"IoU_bg:{Average(IoU_bg_list)}")
        self._logger.info(f"IoU_mean:{Average(IoU_mean_list)}")
        res = {}
        res["Recall"] = Average(Recall_list).item()
        res["Specificity"] = Average(Specificity_list).item()
        res["Precision"] = Average(Precision_list).item()
        res["F1"] = Average(F1_list).item()
        res["F2"] = Average(F2_list).item()
        res["ACC_overall"] = Average(ACC_overall_list).item()
        res["IoU_poly"] = Average(IoU_lesion_list).item()
        res["IoU_bg"] = Average(IoU_bg_list).item()
        res["IoU_mean"] = Average(IoU_mean_list).item()

        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results

    
