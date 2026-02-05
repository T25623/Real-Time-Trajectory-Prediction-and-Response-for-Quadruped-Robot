from picamera2 import Picamera2
import time
import cv2
from ultralytics import YOLO

picam2 = Picamera2()
import numpy as np
config = picam2.create_video_configuration(
    main={"size": (640, 640), "format": "RGB888"},
    controls={"FrameRate": 120}
)
picam2.configure(config)
picam2.start()

frame_count = 0
start_time = time.time()
fps = 0

model = YOLO("./best.pt")

def detect_balloon(frame):
    results = model(frame)[0]
    center_x, center_y = 0,0
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id != 0:
            continue
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255, 0), 2)
        break
    
    return frame

try:
    while True:
        frame = picam2.capture_array()
        frame_count += 1

        frame = frame.to_ndarray(format="bgr24")
        frame = detect_balloon(frame)

        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()
            print(f"FPS: {fps}")

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
