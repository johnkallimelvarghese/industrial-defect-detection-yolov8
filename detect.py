from ultralytics import YOLO

# Load your newly trained model
model = YOLO(r"runs/detect/train3/weights/best.pt")

# Run inference
results = model.predict(
    source=r"/Users/john.jr/Documents/TCS/git repo1/dataset/test/patches_241.jpg",
    imgsz=640,          # match your training image size for best accuracy
    conf=0.5,
    save=True,          # saves output images
    save_txt=True,      # saves YOLO txt output
    name="prediction",
    device="mps",
     show_labels=True,
    show_conf=True
)
