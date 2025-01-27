import time
import cv2
import requests
from ultralytics import YOLO
from paddleocr import PaddleOCR

import logging
logging.getLogger("ppocr").setLevel(logging.WARNING)

API_URL = "http://127.0.0.1:5000/license_plate_from_entrance"


def send_license_plate(license_plate):
    try:
        last_detection = license_plate.replace(" ", "").replace("-", "").upper()
        response = requests.post(API_URL, json={'license_plate': last_detection})
        if response.status_code == 201:
            print(f"License plate '{last_detection}' sent successfully!")

        else:
            print(f"Failed to send license plate '{last_detection}': {response.text}")
    except Exception as e:
        print(f"Error during API call: {e}")


def main():
    # ip_camera_url = "http://192.168.0.83:8080/video"
    ip_camera_url = "../data/rejestracje.mp4"
    cap = cv2.VideoCapture(ip_camera_url)

    if not cap.isOpened():
        print("Error: Unable to connect to the IP camera.")
        return

    # Process every n-th frame
    frame_skip_interval = 15
    frame_count = 0
    last_time = time.time()

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

                # Show the binary image
                # cv2.imshow("Binary Image", binary)

                # Perform OCR
                result = ocr.ocr(cropped_img, cls=True)

                if result and result[0]:
                    detected_text = result[0][0][1][0]

                    # Ensure the license plate is valid and not a duplicate
                    if detected_text  and len(detected_text) >= 7 and detected_text[0].isalpha():
                        # Check if enough time has passed to send license plate (5 seconds)
                        if time.time() - last_time >= 3:
                            last_time = time.time()
                            send_license_plate(detected_text)

        # Display the frame
        cv2.imshow("IP Camera", frame)

        # Break the loop when the user presses 'Esc'
        if cv2.waitKey(1) & 0xFF == 27:
            print("Execution terminated by user.")
            break


if __name__ == "__main__":
    main()
