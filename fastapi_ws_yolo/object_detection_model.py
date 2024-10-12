import asyncio
import io
from fastapi import WebSocket
from pydantic import BaseModel
from PIL import Image
import torch
from transformers import YolosForObjectDetection, YolosImageProcessor



class Object(BaseModel):
    box: tuple[float, float, float, float]
    label: str


class Objects(BaseModel):
    objects: list[Object]


class ObjectDetection:
    image_processor: YolosImageProcessor | None = None
    model: YolosForObjectDetection | None = None

    def load_model(self) -> None:
        """Loads the model"""
        self.image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
        self.model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")

    def predict(self, image: Image.Image) -> Object:
        
        if not self.image_processor or not self.model:
            raise RuntimeError("Model is not loaded")
        
        inputs = self.image_processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.image_processor.post_process_object_detection(
            outputs, threshold=0.7, target_sizes=target_sizes
        )[0]

        objects: list[Object] = []
        
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            box_values = box.tolist()
            label = self.model.config.id2label[label.item()]
            objects.append(Object(box=box_values, label=label))
        
        return Objects(objects=objects)
    

async def receive(websocket: WebSocket, queue: asyncio.Queue):
    while True:
        bytes = await websocket.receive_bytes()
        try:
            queue.put_nowait(bytes)
        except asyncio.QueueFull:
            pass


async def detect(websocket: WebSocket, queue: asyncio.Queue, object_detection: ObjectDetection):
    while True:
        bytes = await queue.get()
        image = Image.open(io.BytesIO(bytes))
        objects = object_detection.predict(image)
        await websocket.send_json(objects.model_dump())
