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
tokenizer_config = {"padding_side": "left"}
tokenizer = AutoTokenizer.from_pretrained("gpt2", **tokenizer_config)
model = AutoModelForCausalLM.from_pretrained("gpt2", num_beams=5)

# Load image
image_path = "/home/kaan/DL/detectron2/Taiwan-Blue-Magpie-web-824x549.jpg"
image = cv2.imread(image_path)
print("Image shape:", image.shape)
cv2.imshow("Image",image)
cv2.waitKey(0)

# Perform object detection
outputs = predictor(image)
print("Object detection outputs:", outputs)

# Generate captions
inputs = []
for i, box in enumerate(outputs['instances'].pred_boxes):
    x1, y1, x2, y2 = box
    crop = image[int(y1):int(y2), int(x1):int(x2), :]
    cv2.imshow("Object " + str(i), crop)
    cv2.waitKey(0)
    caption_input = f"A cute {metadata.thing_classes[outputs['instances'].pred_classes[i]]} is sitting in the photo"
    inputs.append((crop, caption_input))

captions = []
for crop, caption_input in inputs:
    input_ids = tokenizer.encode(caption_input, return_tensors="pt")
    features = model.generate(input_ids=input_ids, num_beams=10, max_length=30, early_stopping=True)
    generated_caption = tokenizer.decode(features.squeeze().tolist())
    captions.append(generated_caption.strip())

# Sort the objects by confidence score
sorted_objects = sorted(enumerate(outputs["instances"].scores.tolist()), key=lambda x: x[1], reverse=True)

# Print captions for the top 3 accurate objects
for i in range(min(len(sorted_objects), 3)):
    obj_idx = sorted_objects[i][0]
    print(f"{metadata.thing_classes[outputs['instances'].pred_classes[obj_idx]]}: {captions[obj_idx]}")

cv2.destroyAllWindows()
