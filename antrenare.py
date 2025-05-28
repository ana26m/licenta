from ultralytics import YOLO

model = YOLO("best.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    device='cuda'  
)