import cv2
from ultralytics import YOLO

# Load a YOLO model
model = YOLO("best.pt")

# Capture the video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if there are no frames left

    # Perform object detection on the current frame
    results = model(frame)

    # Draw detections on the frame
    for result in results:
        # Iterate through detected objects
        for box in result.boxes:
            # Get the label and confidence
            label = box.cls  # Class label
            
            # Get the confidence value and convert to float
            confidence = float(box.conf.item())

            # Check if the detected class is the floor (replace `floor_label` with the actual index of the floor class)
            floor_label = 0  # Update with the correct index for 'floor' from your class labels
            if label == floor_label:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                # Draw a rectangle around the detected floor
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
                cv2.putText(frame, f'Floor {confidence:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow("Floor Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()