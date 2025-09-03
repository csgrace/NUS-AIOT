from ultralytics import YOLO

# Load a model
model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training)

# Train the model with MPS and your own dataset
data_yaml = "uecfood256.yaml"
results = model.train(data=data_yaml, epochs=100, imgsz=640, device="mps")