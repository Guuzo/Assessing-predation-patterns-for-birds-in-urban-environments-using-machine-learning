
import cv2  
import os 
import torch  
from collections import defaultdict  
from ultralytics import YOLO  

# Paths
model_path = "Your path to the model"
input_folder = " Your path to input folder"
output_folder = " Your path to output folder"

# Use GPU instead of CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("CUDA available:", torch.cuda.is_available())


# Load YOLO model 
model = YOLO(model_path)
model.to(device)

# Supported video files
video_extensions = ('.mp4')


# Label mapping following categories
label_mapping = {
    0: "Green",
    1: "Other",
    2: "Non caterpillar"
}

# Create the output folder 
os.makedirs(output_folder, exist_ok=True)

# Gathering videos in a list
video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(video_extensions)]
total_videos = len(video_files)

# Number of predicted videos per category
category_counts = defaultdict(int)

# Video Processing 

# Process each video one by one
for idx, filename in enumerate(video_files, start=1):
    input_path = os.path.join(input_folder, filename)
    print(f"[{idx}/{total_videos}] Processing: {filename}")
# Open the video file
    cap = cv2.VideoCapture(input_path)  
    if not cap.isOpened():
        print(f"Failed to open {input_path}")
        continue

# Retrieve video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Default to 30 FPS if unknown

# temporary output path and video writer
    temp_output_path = os.path.join(output_folder, f"temp_{filename}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

#  highest confidence and corresponding label
    highest_conf = 0
    top_label = "Background"

# Frame processing
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break  
# Run YOLO model and obtain bounding boxes
        results = model(frame)  
        boxes = results[0].boxes  
# detections with a confidence < 0.5 are skipped
        if boxes is not None and boxes.conf is not None:
            for i in range(len(boxes.conf)):
                conf = boxes.conf[i].item()  
                if conf < 0.5:
                    continue  
# assign correct label following the label map
                cls_id = int(boxes.cls[i].item())
                label = label_mapping.get(cls_id) 

# Track the highest-confidence label for this video
                if label and conf > highest_conf:
                    highest_conf = conf
                    top_label = label

# Draw boundingbox on video frames
                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist()) 
                label_text = f"{label} {conf:.2f}" if label else f"Class {cls_id} {conf:.2f}"
                color = (0, 255, 0)  
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)  

# Obtain resources from temporary files
    cap.release()
    out.release()

# Sorting predictions per category

# Count how many videos belong to each label
    category_counts[top_label] += 1

# Create a folder for the predicted category
    label_folder = os.path.join(output_folder, top_label)
    os.makedirs(label_folder, exist_ok=True)

# Move the temporary annotated video into the correct category folder
    final_output_path = os.path.join(label_folder, f"annotated_{filename}")
    os.replace(temp_output_path, final_output_path)

    print(f"Sorted as '{top_label}' → {final_output_path}")

# summary of processed videos
print(f"Total videos processed: {total_videos}")
for category in ["Green", "Other", "Non caterpillar", "Background"]:
    print(f"{category}: {category_counts[category]} video(s)")

