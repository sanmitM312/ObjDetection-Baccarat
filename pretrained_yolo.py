from ultralytics import YOLO
import cv2

# Load YOLOv8 model (pre-trained)
model = YOLO("yolov8n.pt")  # 'n' = nano model (small & fast)

# Load haystack image (the big image where you want to find objects)
image = cv2.imread("casino2.jpg")

# Perform detection
results = model(image)

# Draw bounding boxes on the image
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        conf = box.conf[0].item()  # Confidence score
        label = r.names[int(box.cls[0])]  # Object label

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Output the image
cv2.imwrite("output.jpg", image)
# Show result
# cv2.imshow("YOLO Detection", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
