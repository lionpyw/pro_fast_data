## Fastapi Image Object Detection

In this simple app, the browser will send into the WebSocket a stream of images from the webcam, and our application will run an object detection algorithm and send back the coordinates and label of each detected object in the image. For this task, we’ll rely on HuggingFace pretrained AI model, YOLOS.