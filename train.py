from ultralytics import YOLO

model = YOLO(r'/Users/john.jr/Documents/TCS/git repo1/runs/detect/train/weights/best.pt')

model.train(
    data=r"/Users/john.jr/Documents/TCS/git repo1/dataset/dataset.yaml",
    epochs=75,
    imgsz=640,
    lr0=0.001,
    save_period=5,
    device="mps",
    workers=0,
    batch=4,
)
