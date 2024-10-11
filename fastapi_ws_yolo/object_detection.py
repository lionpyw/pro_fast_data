from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import YolosForObjectDetection, YolosImageProcessor

BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "etc"
PIC_PATH = ASSETS_DIR / "shop.jpg"

image = Image.open(PIC_PATH)


image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")


inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)

target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(
    outputs, threshold=0.7,  target_sizes=target_sizes
)[0]

draw = ImageDraw.Draw(image)
font_path = ASSETS_DIR / "OpenSans-ExtraBold.ttf"
font = ImageFont.truetype(str(font_path), 24)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box_values = box.tolist()
    label = model.config.id2label[label.item()]
    draw.rectangle(box_values, outline="red", width=5)
    draw.text(box_values[0:2], label, fill="red", font=font)

image.save(ASSETS_DIR/"obj_detect.jpg")

