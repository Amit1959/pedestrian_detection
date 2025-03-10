import cv2
import torch
import platform
from ultralytics import YOLO

# 🚀 Check if running on Jetson Nano
RUNNING_ON_JETSON = "aarch64" in platform.machine()

# 🚀 Use GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 🚀 Load YOLOv8 Model
model = YOLO("yolov8n.pt")  # Ensure the file exists in your directory

# 🚀 Use Camera on Jetson, Video File on Other Systems
video_source = 0 if RUNNING_ON_JETSON else "input_video.mp4"
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("❌ Error: Could not open camera or video file.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 🚀 Run pedestrian detection
    results = model(frame, device=device)

    # 🚀 Draw bounding boxes
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:  # COCO class 0 = "person"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()

                # Draw bounding box & label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Person {confidence:.2f}', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 🚀 Show output
    cv2.imshow("YOLOv8 Pedestrian Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
