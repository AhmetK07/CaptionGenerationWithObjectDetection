import os
import random
import json
import cv2
import numpy as np
import torch
import time
import shutil
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetMapper
import datetime
from tqdm import tqdm

# Define function to prepare dataset
def get_coco_dicts(img_dir, annot_file):
    with open(annot_file, 'r') as f:
        annots = json.load(f)
    dataset_dicts = []
    for i, annot in enumerate(annots['annotations']):
        record = {}
        image_id = annot['image_id']
        if image_id not in annots['images']:
            continue
        filename = os.path.join(img_dir, annots['images'][image_id]['file_name'])
        record['file_name'] = filename
        record['image_id'] = i
        record['height'] = annots['images'][image_id]['height']
        record['width'] = annots['images'][image_id]['width']
        objs = []
        for bbox, cat_id in zip(annot['bbox'], annot['category_id']):
            obj = {
                'bbox': bbox,
                'bbox_mode': BoxMode.XYWH_ABS,
                'category_id': cat_id
            }
            objs.append(obj)
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts

# Register dataset with DatasetCatalog
img_dir = "/home/kaan/DL/detectron2/train2017"
annot_file = "/home/kaan/DL/detectron2/annotations/instances_train2017.json"
DatasetCatalog.register("my_dataset", lambda: get_coco_dicts(img_dir, annot_file))
MetadataCatalog.get("my_dataset").set(thing_classes=["person"])

# Load COCO metadata
coco_metadata = MetadataCatalog.get("my_dataset")

# Check if the thing classes are correct
assert coco_metadata.thing_classes[0] == "person", "Thing classes are not correct!"

# Define trainer
class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))

# Define config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 3000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

# Define trainer and train
trainer = CocoTrainer(cfg)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
start_training_time = time.time()
trainer.resume_or_load(resume=False)
total_steps = len(trainer.data_loader.dataset) // cfg.SOLVER.IMS_PER_BATCH * cfg.SOLVER.MAX_ITER

with open(os.path.join(cfg.OUTPUT_DIR, "log.txt"), "a") as log_file:
    for iteration, d in enumerate(trainer.data_loader, start=trainer.iter):
        iteration_time = time.time()
        trainer.run_step


checkpoint_period = 500
next_checkpoint = checkpoint_period

print("PROGRESS BAR DENEME")
for iteration, d in tqdm(enumerate(trainer.data_loader, start=trainer.iter), total=total_steps):
    iteration_time = time.time()
    trainer.run_step()
    if iteration == next_checkpoint:
        
        eta_seconds = ((time.time() - start_training_time) / iteration) * (total_steps - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        print(f"Iteration {iteration}/{total_steps}. ETA: {eta_string}")

        checkpoint_path = os.path.join(cfg.OUTPUT_DIR, f"model_{iteration:07}.pth")
        trainer.checkpointer.save(checkpoint_path)

        next_checkpoint += checkpoint_period

    if iteration >= total_steps:
        break
