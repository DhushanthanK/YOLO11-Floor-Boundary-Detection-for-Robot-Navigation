from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np

CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="AuX83b3TUkz1802QnozK"
)

# Perform inference
result = CLIENT.infer('/Users/dhushanthankumararatnam/Documents/DMIE_RE/DMIE_AMR/floor detection/Floor_Detection_Yolo11_Final/image.webp', model_id="room-lyc7i/2")

# Load the image
image = cv2.imread('/Users/dhushanthankumararatnam/Documents/DMIE_RE/DMIE_AMR/Floor_Detection_Yolo11_Final/image.webp')

# Loop through predictions and draw bounding boxes and points
for prediction in result['predictions']:

    # Draw the points
    for point in prediction['points']:
        px, py = int(point['x']), int(point['y'])
        cv2.circle(image, (px, py), 3, (255, 0, 0), -1)

# Save or display the resulting image
cv2.imwrite('output.jpg', image)
cv2.imshow('Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()