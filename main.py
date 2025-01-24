﻿import time
import cv2
import requests
from ultralytics import YOLO
from paddleocr import PaddleOCR

API_URL = "http://127.0.0.1:5000/license_plate_from_entrance"


def main():
    ip_camera_url = "http://192.168.0.83:8080/video"
    cap = cv2.VideoCapture(ip_camera_url)

    if not cap.isOpened():
        print("Error: Unable to connect to the IP camera.")
        return

    # Process every n-th frame
    frame_skip_interval = 15
    frame_count = 0
    last_detection = None  # Track the last detected license plate
    last_sent_time = time.time()  # Track the time of the last API call

    # Load YOLO model
    model = YOLO("best_plate_detector_model.pt")

    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en')

    while True:
        ret, frame = cap.read()  # Capture a frame from the video stream
        if not ret:
            print("Failed to grab frame from IP camera.")
            break

        frame_count += 1
        if frame_count % frame_skip_interval != 0:
            continue

        # Perform inference on the current frame
        results = model(frame)

        # Extract bounding box information
        if results and results[0].boxes:  # Check if any detections exist
            for box in results[0].boxes:  # Iterate over detected boxes
                xyxy = box.xyxy.cpu().numpy()[0]  # Extract coordinates
                x1, y1, x2, y2 = map(int, xyxy)

                # Crop the image using the coordinates
                cropped_img = frame[y1:y2, x1:x2]  # Crop the license plate region

                # Perform OCR
                result = ocr.ocr(cropped_img, cls=True)

                if result and result[0]:
                    detected_text = result[0][0][1][0]

                    if detected_text and (detected_text != last_detection or time.time() - last_sent_time >= 4) and len(
                            detected_text) >= 7 and detected_text[0].isalpha():
                        try:
                            detected_text = detected_text.replace(" ", "").replace("-", "").upper()
                            response = requests.post(API_URL, json={'license_plate': detected_text})
                            if response.status_code == 201:
                                print(f"License plate '{detected_text}' sent successfully!")
                                last_detection = detected_text  # Update the last detected plate
                                last_sent_time = time.time()  # Update the last sent time
                            else:
                                print(f"Failed to send license plate '{detected_text}': {response.text}")
                        except Exception as e:
                            print(f"Error during API call: {e}")

        # Break the loop when the user presses 'Esc'
        if cv2.waitKey(1) & 0xFF == 27:
            print("Execution terminated by user.")
            break


if __name__ == "__main__":
    main()
