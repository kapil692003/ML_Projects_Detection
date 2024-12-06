import random
import os
import cv2
import numpy as np
from ultralytics import YOLO

# Function to create a resizable window
def create_resizable_window(window_name, width, height):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

# Specify the path to the utils directory
utils_path = "C:\\Users\\KKI43\\Desktop\\Dassault Projects\\Object Detection\\utils"
file_name = "Kapil.txt"  # Replace with your actual file name
file_path = os.path.join(utils_path, file_name)

# Initialize class_list
class_list = []

# Check if the file exists and read class names
if os.path.isfile(file_path):
    with open(file_path, "r") as my_file:
        class_list = [line.strip() for line in my_file.readlines()]
    print("Class list loaded:", class_list)
else:
    print(f"File '{file_name}' does not exist in the directory '{utils_path}'.")

# Generate random colors for class list
detection_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in class_list]

# Load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8")

# Open video capture
cap = cv2.VideoCapture("inference/videos/videoplayback.mp4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Define desired window size
output_window_size = (800, 600)  # Width, Height
create_resizable_window("Object Detection", *output_window_size)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Resize frame for faster processing (optional)
    frame_resized = cv2.resize(frame, (640, 640))

    # Predict on the image
    detect_params = model.predict(source=[frame_resized], conf=0.45, save=False)

    # Convert tensor array to numpy
    DP = detect_params[0].numpy()
    print(DP)

    if len(DP) > 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]  # returns one box
            clsID = int(box.cls.numpy()[0])
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            # Draw bounding box
            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[clsID],
                3,
            )

            # Display class name and confidence
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                f"{class_list[clsID]} {conf:.1%}",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )
    else:
        print("No objects detected.")

    # Resize the frame for display
    frame_display = cv2.resize(frame, output_window_size)

    # Display the resulting frame
    cv2.imshow("Object Detection", frame_display)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
