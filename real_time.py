from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8m.pt")
video_path = 0
cap = cv2.VideoCapture(video_path)
prev_inference_time = 0

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    current_time = time.time()

    if current_time - prev_inference_time >= 0.5:
        results = model(frame)

        for result in results:
            annotated_frame = result.plot()
            cv2.imshow("YOLOv8", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
