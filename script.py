from ultralytics import YOLO

import cv2
import math

# Define the source: 0 for webcam or the path to your video file
VIDEO_SOURCE = 0  # Use "samples/v1.mp4" for a video file
VIDEO_FILE = "test_videos/delhi.mp4"

# Initialize video capture
cap = cv2.VideoCapture(VIDEO_FILE)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
resize_width = 640  # Adjust based on your needs
resize_height = 480  # Adjust based on your needs
if frame_width > 0:
    resize_height = int((resize_width / frame_width) * frame_height)

skip_frames = 2  # Process every 3rd frame
frame_count = 0

# Load the YOLO model
chosen_model = YOLO("yolov8n.pt")  # Adjust model version as needed

def predict(chosen_model, img, classes=[], conf=0.2):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.2):
    results = predict(chosen_model, img, classes, conf)

    for result in results:
        for box in result.boxes:
            # Calculate the center of the bounding box
            x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
            y_center = (box.xyxy[0][1] + box.xyxy[0][3]) / 2

            # Calculate the distance from the center of the image (vehicle reference point)
            img_center_x = img.shape[1] / 2
            img_center_y = img.shape[0] / 2
            distance = math.sqrt((x_center - img_center_x) ** 2 + (y_center - img_center_y) ** 2)
            
            
            # Define a threshold distance for proximity alert
            threshold_distance = 130  # Adjust as needed

            # Check if the detected object is a person
            if result.names[int(box.cls[0])] == "person":
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 255, 0), 2)
                cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                # Check proximity
                if distance < threshold_distance:
                    cv2.putText(img, "ALERT: Person too close!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 0, 255), 2)
                cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
    return img, results

while True:
    success, img = cap.read()

    if not success:
        break
    # Skip frames to speed up processing
    frame_count += 1
    if frame_count % skip_frames != 0:
        continue
    img = cv2.resize(img, (resize_width, resize_height))
    result_img, _ = predict_and_detect(chosen_model, img, classes=[], conf=0.2)

    cv2.imshow("Image", result_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
