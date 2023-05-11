import torch
import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


# Set up Detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

# Set up Hugging Face Transformers
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Load image
image_path = "/home/kaan/DL/detectron2/dog-puppy-on-garden-royalty-free-image-1586966191.jpg"
image = cv2.imread(image_path)
print(image.shape)
cv2.imshow("Image",image)


# Generate caption using Hugging Face Transformers
inputs = []
outputs = predictor(image)
for i, box in enumerate(outputs['instances'].pred_boxes):
    x1, y1, x2, y2 = box
    crop = image[int(y1):int(y2), int(x1):int(x2), :]
    caption_input = f"{metadata.thing_classes[outputs['instances'].pred_classes[i]]} {tokenizer.sep_token} {tokenizer.cls_token} {tokenizer.eos_token}"
    inputs.append((crop, caption_input))

captions = []
for crop, caption_input in inputs:
    input_ids = tokenizer.encode(caption_input, return_tensors="pt")
    features = model(input_ids=input_ids)
    generated_caption = tokenizer.decode(features["logits"].squeeze().argmax(dim=-1).tolist())
    captions.append(generated_caption.strip())


print(captions)
# Print captions
for i, caption in enumerate(captions):
    print(f"{metadata.thing_classes[outputs['instances'].pred_classes[i]]}: {caption}")

