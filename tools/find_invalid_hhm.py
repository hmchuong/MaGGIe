import glob
import torch, detectron2

import argparse
# import some common libraries
import numpy as np
import os, json, cv2, random
from PIL import Image
import tqdm
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split", type=int, default=0)
    parser.add_argument("--num_splits", type=int, default=8)

    args = parser.parse_args()

    cuda_id = args.split % 4
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)


    root_dir = "/mnt/localssd/HHM/train"
    invalid_name = []
    all_images = os.listdir(root_dir + "/images")
    all_images.sort()
    all_images = all_images[args.split::args.num_splits]

    for image_name in tqdm.tqdm(all_images):
        image_path = f"{root_dir}/images/{image_name}"
        # Predict mask
        img = cv2.imread(image_path)
        outputs = predictor(img)
        
        no_people = (outputs['instances'].pred_classes == 0).sum().item()
        if no_people != 1:
            invalid_name.append(image_name)
    
    with open(f"invalid_{args.split}.txt", "w") as f:
        f.write("\n".join(invalid_name))