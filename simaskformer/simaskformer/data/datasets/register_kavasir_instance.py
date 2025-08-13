# Copyright (c) Selab-HCMUS and its affiliates.

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import register_coco_instances

#META

#All polyp categories, together with their nice-looking visualization colors 
KVASIR_INS_CATEGORIES = [
    {"color": [0, 255, 0], "id": 1, "name": "pedunculated"},
    {"color": [255, 0, 0], "id": 2, "name": "sessile"},
]


def _get_kvasir_instances_meta():
    ids = [k["id"] for k in KVASIR_INS_CATEGORIES]
    colors = [k["color"] for k in KVASIR_INS_CATEGORIES]
    classes = [k["name"] for k in KVASIR_INS_CATEGORIES]
    dataset_id_to_contiguous_id = {k: i for i, k in enumerate(ids)}
    ret = {
        "thing_dataset_id_to_contiguous_id": dataset_id_to_contiguous_id,
        "thing_classes": classes,
        "thing_colors": colors,
    }
    return ret

# Polyp instance dataset
# ==== Predefined datasets and splits for PolypDB_INS ==========

_PREDEFINED_SPLITS_KVASIR_INS={
    "kvasir_ins_train": (".","kvasir_instance_train.json"),
    "kvasir_ins_val": (".","kvasir_instance_val.json"),
    "kvasir_ins_test": (".","kvasir_instance_test.json"),
}


def register_all_kvasir_instance(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_KVASIR_INS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_kvasir_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
      
_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "../datasets/Kvasir_INS"))
register_all_kvasir_instance(_root)

