# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
import cv2

# download model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# load model
model = YOLO(model_path)

# inference
image_path = r"_2d_intensity/2026_05_27_11_01_49.png"

grayImage = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

rgb_like_img = cv2.cvtColor(grayImage, cv2.COLOR_GRAY2RGB)
print("rgb intensity image: %s" %rgb_like_img.dtype)
output = model(rgb_like_img)
results = Detections.from_ultralytics(output[0])

result = results[0]

boxes = result.xyxy
scores = result.confidence
classes = result.class_id

for box, score, cls in zip(boxes, scores, classes):

    print(f"Object: {cls} | Bounding Box: {box.tolist()} | Confidence Score: {score:.2f}")
    
    cv2.rectangle(rgb_like_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)
    
    #Text
    text = f"{score:.2f}"
    coordinates = (int(box[0]), int(box[1])-20)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255 , 0, 0) #Red
    thickness = 2
    
    cv2.putText(rgb_like_img,  text, coordinates, font, font_scale, color, thickness, cv2.LINE_AA)
    
    cv2.imshow("image with bounding box", rgb_like_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
