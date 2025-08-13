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
from detectron2.data import detection_utils as utils
from detectron2.evaluation import DatasetEvaluator

from scipy.spatial.distance import cdist
from medpy import metric
from sklearn.metrics import fbeta_score,accuracy_score
_CV2_IMPORTED = True
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    _CV2_IMPORTED = False

convert_L_to_Class={
            0:0,                #background: 0->0
            147:1,              #R_Vofo: 147 ->1
            111:2,              #R_Arytenoid: 111 ->2
            230:3,              #Benign_Les: 230 ->3
            104:4,              #Malignant_les: 104 ->4
            124:5,              #L_Vofo: 124 ->5
            127:6               #L_Arytenoid: 127 ->6
        }

#Visualize
color_palette={
    0:[0, 0, 0],                    #background    
    1:[255, 0, 0],                  #R_Vofo
    2:[255, 165, 0],                #R_Arytenoid
    3:[255, 255, 0],                #Benign_Les
    4:[0, 255, 0],                  #Malignant_les
    5:[0, 0, 255],                  #L_Vofo
    6:[148, 0, 211]                 #L_Arytenoid
}

def visualize_mask(mask_output, save_path, color_palette, image_path, opacity=0.5): 
    """
    Overlay the mask on the original image, but only change the pixels corresponding to class_id > 0 to preserve the quality of the original image.

    Args:
        mask_output (numpy.ndarray): Mask with shape (H, W), containing class ids (0 to 6).
        save_path (str): Path to save the resulting image.
        color_palette (dict): A dictionary mapping class ids (0-6) to RGB color values.
        image_path (str): Path to the original image.
        opacity (float): Opacity of the mask overlay (0.0 - 1.0), default is 0.5.

    Returns:
        None: The overlay image will be saved at `save_path`.
    """
    # Read the original image
    img = np.array(Image.open(image_path).convert('RGB'), dtype=np.uint8)
    img_h, img_w = img.shape[:2]

    # Create a color mask based on the color_palette
    mask_h, mask_w = mask_output.shape
    color_mask = np.zeros((mask_h, mask_w, 3), dtype=np.uint8)
    for class_id, color in color_palette.items():
        if class_id != 0:  # Skip background (class_id 0)
            color_mask[mask_output == class_id] = color  

    # Resize the color mask to match the dimensions of the original image
    color_mask = np.array(Image.fromarray(color_mask).resize((img_w, img_h), Image.NEAREST))

    # Create a binary mask to identify the areas that need overlay
    binary_mask = np.any(color_mask > 0, axis=-1)  # True for pixels that have color

    # Create the overlay image by blending only the masked pixels
    overlay = img.copy()
    overlay[binary_mask] = (opacity * color_mask[binary_mask] + (1 - opacity) * img[binary_mask]).astype(np.uint8)

    # Convert the result to a PIL image and save
    Image.fromarray(overlay).save(save_path)

def load_groundtruth_vofo(path):
    sem_seg_gt = utils.read_image(path)
    sem_seg_gt = Image.fromarray(sem_seg_gt)
    sem_seg_gt=np.expand_dims(np.array(sem_seg_gt.convert('L')), axis=2) 
    mapping_array = np.vectorize(convert_L_to_Class.get)
    sem_seg_gt=mapping_array(sem_seg_gt).squeeze()
    return sem_seg_gt

def load_image_into_numpy_array(
    filename: str,
    copy: bool = False,
    dtype: Optional[Union[np.dtype, str]] = None,
) -> np.ndarray:
    with PathManager.open(filename, "rb") as f:
        #array = np.array(Image.open(f), copy=copy, dtype=dtype)
        array = np.asarray(Image.open(f), dtype=dtype)
    return array

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    iou = metric.binary.jc(pred, gt)
    
    specificity=metric.binary.specificity(pred, gt)
    
    if np.sum(gt) == 0 and np.sum(pred) == 0:
        f2 = 1.0  
        recall = 1.0
        precision= 1.0
    else:
        recall = metric.binary.recall(pred, gt)
        f2 = fbeta_score(gt.flatten(), pred.flatten(), beta=2, zero_division=0)
        precision=metric.binary.precision(pred, gt)
    accuracy = accuracy_score(gt.flatten(), pred.flatten())
    return dice,  iou, recall, specificity, precision, f2, accuracy
    
    
def Average(lst):
    return sum(lst) / len(lst)



class VoFoSemSegEvaluator(DatasetEvaluator):
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
        sem_seg_loading_fn=load_groundtruth_vofo,
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
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._b_conf_matrix = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.int64
        )
        self._predictions = []
        self._raw_predictions = []

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
             
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred_raw=copy.deepcopy(output.numpy())
            #pred_raw=np.where(pred_raw == self._num_classes,0, pred_raw + 1).astype(int)


            if self.visualize==True:
                mask_image = (copy.deepcopy(pred_raw).squeeze()).astype(np.uint8)
                visualize_mask(mask_image, self.visualize_path+'/'+input["file_name"][31:-4].replace("/","_")+".png", color_palette, input["file_name"], opacity=0.7)
                print("Visualize:",self.visualize_path+'/'+input["file_name"][31:-4].replace("/","_")+".png")

              
            #output=torch.where(output == 0, 255, output - 1)
            #output[output==-1]=255
            pred = np.array(output, dtype=int)        # 255: _ignore_label
            
            
            gt_filename = self.input_file_to_gt_file[input["file_name"]]
            gt = self.sem_seg_loading_fn(gt_filename)
            gt_raw=copy.deepcopy(gt).astype(int)
            
            
            #gt=np.where(gt == 0, self._num_classes, gt - 1)
            gt = np.array(gt, dtype=int)        

            #print(np.unique(pred))
            #print(np.unique(gt))
            
            #pred[pred == self._ignore_label] = self._num_classes
            #gt[gt == self._ignore_label] = self._num_classes

            
            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            #print(self._conf_matrix)
            if self._compute_boundary_iou:
                b_gt = self._mask_to_boundary(gt.astype(np.uint8))
                b_pred = self._mask_to_boundary(pred.astype(np.uint8))

                self._b_conf_matrix += np.bincount(
                    (self._num_classes + 1) * b_pred.reshape(-1) + b_gt.reshape(-1),
                    minlength=self._conf_matrix.size,
                ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred_raw, input["file_name"]))
            self._raw_predictions.append({"pred":pred_raw,"gt":gt_raw})

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
            conf_matrix_list = all_gather(self._conf_matrix)
            b_conf_matrix_list = all_gather(self._b_conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))

            self._raw_predictions = all_gather(self._raw_predictions)
            self._raw_predictions = list(itertools.chain(*self._raw_predictions))

            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

            self._b_conf_matrix = np.zeros_like(self._b_conf_matrix)
            for b_conf_matrix in b_conf_matrix_list:
                self._b_conf_matrix += b_conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        """
        acc = np.full(self._num_classes+1, np.nan, dtype=float)
        iou = np.full(self._num_classes+1, np.nan, dtype=float)
        tp = self._conf_matrix.diagonal().astype(float)
        pos_gt = np.sum(self._conf_matrix, axis=0).astype(float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix, axis=1).astype(float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        if self._compute_boundary_iou:
            b_iou = np.full(self._num_classes+1, np.nan, dtype=float)
            b_tp = self._b_conf_matrix.diagonal().astype(float)
            b_pos_gt = np.sum(self._b_conf_matrix, axis=0).astype(float)
            b_pos_pred = np.sum(self._b_conf_matrix, axis=1).astype(float)
            b_union = b_pos_gt + b_pos_pred - b_tp
            b_iou_valid = b_union > 0
            b_iou[b_iou_valid] = b_tp[b_iou_valid] / b_union[b_iou_valid]

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res[f"IoU-{name}"] = 100 * iou[i]
            if self._compute_boundary_iou:
                res[f"BoundaryIoU-{name}"] = 100 * b_iou[i]
                res[f"min(IoU, B-Iou)-{name}"] = 100 * min(iou[i], b_iou[i])
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res[f"ACC-{name}"] = 100 * acc[i]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        """
        ###

        metric_list_per_class = {}
        
        for img in self._raw_predictions:
            pred = img["pred"]
            msk = img["gt"]
            for i in range(1, self._num_classes):
                gt_class = (msk == i).astype(int)
                pred_class = (pred == i).astype(int)
                dice,iou,recall, specificity, precision, f2, accuracy = calculate_metric_percase(pred_class, gt_class)
                if i not in metric_list_per_class:
                    metric_list_per_class[i] = []
                metric_list_per_class[i].append([dice, iou,recall, specificity, precision, f2, accuracy])

        metric_list = []
        for i in range(1, self._num_classes):
            metric_list_per_class[i] = np.array(metric_list_per_class[i])     
            metric_list_per_class[i] = np.nanmean(metric_list_per_class[i], axis=0)
            metric_list.append([metric_list_per_class[i][0], 
                                metric_list_per_class[i][1],
                                metric_list_per_class[i][2],
                                metric_list_per_class[i][3],
                                metric_list_per_class[i][4],
                                metric_list_per_class[i][5],
                                metric_list_per_class[i][6]])

        mean_dice = [metric_list[i][0] for i in range(len(metric_list))]  # Mean dice
        mean_iou = [metric_list[i][1] for i in range(len(metric_list))]  # Mean iou
        mean_recall = [metric_list[i][2] for i in range(len(metric_list))]
        mean_specificity = [metric_list[i][3] for i in range(len(metric_list))]
        mean_precision = [metric_list[i][4] for i in range(len(metric_list))]
        mean_f2 = [metric_list[i][5] for i in range(len(metric_list))]
        mean_accuracy = [metric_list[i][6] for i in range(len(metric_list))]

        for i in range(1, self._num_classes): 
            print(f"Test epoch; Class {i} mean_dice: {mean_dice[i-1]:.5f}, mean_iou: {mean_iou[i-1]:.5f}, mean_recall: {mean_recall[i-1]:.5f}, mean_specificity: {mean_specificity[i-1]:.5f}, mean_precision: {mean_precision[i-1]:.5f}, mean_f2: {mean_f2[i-1]:.5f}, mean_accuracy: {mean_accuracy[i-1]:.5f}")
            self._logger.info(f"Test epoch; Class {i} mean_dice: {mean_dice[i-1]:.5f}, mean_iou: {mean_iou[i-1]:.5f}, mean_recall: {mean_recall[i-1]:.5f}, mean_specificity: {mean_specificity[i-1]:.5f}, mean_precision: {mean_precision[i-1]:.5f}, mean_f2: {mean_f2[i-1]:.5f}, mean_accuracy: {mean_accuracy[i-1]:.5f}")

        all_class_mean_dice=np.mean(mean_dice)
        all_class_mean_iou=np.mean(mean_iou)
        all_class_mean_recall=np.mean(mean_recall)
        all_class_mean_specificity=np.mean(mean_specificity)
        all_class_mean_precision=np.mean(mean_precision)
        all_class_mean_f2=np.mean(mean_f2)
        all_class_mean_accuracy=np.mean(mean_accuracy)
        print("ALL class: mean_dice: %.5f, mean_iou: %.5f, mean_recall: %.5f, mean_specificity: %.5f, mean_precision: %.5f, mean_f2: %.5f, mean_accuracy: %.5f"% (all_class_mean_dice,all_class_mean_iou,all_class_mean_recall,all_class_mean_specificity,all_class_mean_precision,all_class_mean_f2,all_class_mean_accuracy))
        self._logger.info("ALL class: mean_dice: %.5f, mean_iou: %.5f, mean_recall: %.5f, mean_specificity: %.5f, mean_precision: %.5f, mean_f2: %.5f, mean_accuracy: %.5f"% (all_class_mean_dice,all_class_mean_iou,all_class_mean_recall,all_class_mean_specificity,all_class_mean_precision,all_class_mean_f2,all_class_mean_accuracy))
        
        res = {}
        res["all_class_mean_dice"] = all_class_mean_dice
        res["all_class_mean_iou"] = all_class_mean_iou
        res["all_class_mean_recall"] = all_class_mean_recall
        res["all_class_mean_specificity"] = all_class_mean_specificity
        res["all_class_mean_precision"] = all_class_mean_precision
        res["all_class_mean_f2"] = all_class_mean_f2
        res["all_class_mean_accuracy"] = all_class_mean_accuracy

        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results

    def encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert (
                    label in self._contiguous_id_to_dataset_id
                ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append(
                {"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle}
            )
        return json_list

    def _mask_to_boundary(self, mask: np.ndarray, dilation_ratio=0.02):
        assert mask.ndim == 2, "mask_to_boundary expects a 2-dimensional image"
        h, w = mask.shape
        diag_len = np.sqrt(h**2 + w**2)
        dilation = max(1, int(round(dilation_ratio * diag_len)))
        kernel = np.ones((3, 3), dtype=np.uint8)

        padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        eroded_mask_with_padding = cv2.erode(padded_mask, kernel, iterations=dilation)
        eroded_mask = eroded_mask_with_padding[1:-1, 1:-1]
        boundary = mask - eroded_mask
        return boundary
