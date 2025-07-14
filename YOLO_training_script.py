# Import necessary packages
from ultralytics import YOLO
import torch
# Model training
def main():
    # Using GPU instead of CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    # Trains a model from scratch
    model = YOLO("yolov8n.yaml")
    # Configurations
    results = model.train(
        data="configs.yaml",
        epochs=100,
        batch=4  
    )
# Only runs directly
if __name__ == "__main__":
    main()
