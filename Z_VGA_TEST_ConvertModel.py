from ultralytics import YOLO
model = YOLO("models/yolo/best.pt")
# สั่ง export เป็น engine (TensorRT)
model.export(format="engine", device=0, half=True)


