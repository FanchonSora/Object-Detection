from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import random

class ObjectRecognitionGame:
    def __init__(self, model_path, target_objects):
        self.model = YOLO(model_path)
        self.target_objects = target_objects
        self.classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                           "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                           "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                           "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                           "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                           "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                           "teddy bear", "hair drier", "toothbrush"]
        self.current_target = self.get_random_target()

    def get_random_target(self):
        return random.choice(self.target_objects)

    def run_game(self):
        cap = cv2.VideoCapture(0)  # For Webcam
        cap.set(3, 1280)  # Set width
        cap.set(4, 720)  # Set height

        prev_frame_time = 0
        new_frame_time = 0

        while True:
            new_frame_time = time.time()
            success, img = cap.read()

            if not success:
                print("Failed to grab frame")
                break

            # Perform YOLO detection
            results = self.model(img, stream=True)
            detected_objects = set()

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(img, (x1, y1, w, h))

                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    class_name = self.classNames[cls]
                    detected_objects.add(class_name)

                    text = f'{class_name} {conf}'
                    cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Check if the current target object is detected
            if self.current_target in detected_objects:
                cv2.putText(img, f"Target Found {self.current_target}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Choose a new target after finding the current one
                self.current_target = self.get_random_target()
            else:
                cv2.putText(img, f"Find the target {self.current_target}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Calculate FPS
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.putText(img, f"FPS: {fps:.2f}", (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Show the image
            cv2.imshow("Object Recognition Game", img)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "path_to_yolo_weights/yolov8n.pt"
    target_objects = ["cup", "book"]

    game = ObjectRecognitionGame(model_path, target_objects)
    game.run_game()
