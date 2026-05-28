import cv2
from ultralytics import YOLO

# Load pre-trained YOLO model (detects 'person' out of the box)
model = YOLO("yolov8n.pt") 

# Read image in grayscale
gray_img = cv2.imread(r"_2d_confidence/2026_05_27_10_09_20.png", cv2.IMREAD_GRAYSCALE)

# Replicate the single channel into 3 identical channels
rgb_like_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

# Run inference
results = model(rgb_like_img)

result = results[0]

# 4. Extract bounding box coordinates and confidence metrics
boxes = result.boxes.xyxy    # Bounding box coordinates in [xmin, ymin, xmax, ymax] format
scores = result.boxes.conf   # Confidence scores ranging from 0.0 to 1.0
classes = result.boxes.cls   # Class IDs corresponding to each detection

for box, score, cls in zip(boxes, scores, classes):
    name = result.names[int(cls)]
    print(f"Object: {name} | Bounding Box: {box.tolist()} | Confidence Score: {score:.2f}")
    
    cv2.rectangle(rgb_like_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)
    
    cv2.imshow("image with bounding box", rgb_like_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    