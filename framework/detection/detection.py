from hailo_apps.python.core.common.toolbox import (
    init_input_source,
    get_labels,
    load_json_file,
    preprocess,
    visualize,
    FrameRateTracker,
)

from hailo_apps.python.standalone_apps.object_detection.object_detection_post_process import (
    extract_detections,
    draw_detections,
    inference_result_handler
)
from hailo_apps.python.standalone_apps.object_detection.object_detection import (
    infer,
)

from hailo_apps.python.core.common.hailo_inference import HailoInfer

import queue
import threading
import cv2
import time
from framework.detection.kalman_filter import KalmanFilter
import numpy as np
from collections import deque

class DetectionPipeline:
    def __init__(self, hef_path, config_path, labels_path, source="rpi", resolution=(1280, 720), framerate=60, colour_format="RGB888", batch_size=1, kalman_filter=KalmanFilter.xyz_predict(), FPS_counter = True, trail_length=30) -> None:
        # Camera config
        self.source = source
        self.resolution = resolution
        self.framerate = framerate
        self.colour_format = colour_format
        
        # Model Config
        self.hef_path = hef_path
        self.config_path = config_path
        self.labels_path = labels_path
        self.batch_size = batch_size
        
        # Detected Global
        self.center_x = 0
        self.center_y = 0
        self.predicted_center_x = 0
        self.predicted_center_y = 0
        self.future_center_x = 0
        self.future_center_y = 0
        self.future_distance = 0
        self.detected = False
        self.min_distance = 0
        self.predicted_distances = deque(maxlen=round(framerate/2))
        self.predicted_distance = 0
        
        # Video Stream Queues
        self.FPS_counter = FPS_counter
        self.input_queue = queue.Queue(maxsize=2)
        self.output_queue = queue.Queue(maxsize=2)
        self.display_queue = queue.Queue(maxsize=1)

        self.trail = deque(maxlen=trail_length)

        self.kalman_filter = kalman_filter

    def setup_camera(self):
        resolution = {"size": self.resolution, "format": self.colour_format}
        framerate = {"FrameRate": self.framerate}
        return init_input_source(self.source, self.batch_size, resolution, framerate)

    def load_model(self):
        self.labels = get_labels(self.labels_path)
        self.config_data = load_json_file(self.config_path)
        self.hailo_inference = HailoInfer(self.hef_path, self.batch_size)
        return self.hailo_inference.get_input_shape()

    def start_threads(self, images, cap, width, height):
        threading.Thread(target=preprocess, daemon=True, args=(images, cap, self.framerate, self.batch_size, self.input_queue, width, height)).start()
        threading.Thread(target=infer, daemon=True, args=(self.hailo_inference, self.input_queue, self.output_queue)).start()
        threading.Thread(target=self._display_loop, daemon=True).start()

    def _display_loop(self):
        cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        while True:
            frame = self.display_queue.get()
            if frame is None:
                break
            cv2.imshow("Output", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    
    def estimate_distance(self, x0, y0, x1, y1, real_height, focal_length, scale_factor):
        # Calculate area of detected object
        box_height = abs(y1 - y0)

        distance = (((real_height * focal_length) / box_height) * 2) * scale_factor
        return distance

    def process_detections(self, detections):

        bbox = detections['detection_boxes']

        if len(bbox) > 0:
            # Extract coordinates of bounding box 
            y0, x0, y1, x1 = bbox[0]

            # Calculate center of bounding box
            self.center_x = ((y0 + y1) / 2) / self.resolution[0]
            self.center_y = ((x0 + x1) / 2) / self.resolution[1]
            px = int(self.center_x * self.resolution[0])
            py = int(self.center_y * self.resolution[1])
            self.trail.append((px, py))
            
            # Calculate distance to detected object
            self.min_distance = self.estimate_distance(x0, y0, x1, y1, 24, 0.275, 15)

            # Create array for kalman filter of center x, center y, and distance 
            z = np.array([
                [self.center_x],
                [self.center_y],
                [self.min_distance]
            ])

            u = np.zeros((2,1))
            # Predict next position of detected object
            self.kalman_filter.predict(u)
            state = self.kalman_filter.update(z)

            self.predicted_center_x = state[0,0]
            self.predicted_center_y = state[1,0]
            self.predicted_distance = state[2,0]
            self.detected = True

        else:
            self.detected = False

    def draw_trail(self, frame, trail_list):
        total = len(trail_list)
        for i in range(1, total):
            alpha = i / total
            color = (
                int(0   * (1 - alpha) + 0   * alpha),  # B
                int(255 * (1 - alpha) + 100 * alpha),  # G
                int(0   * (1 - alpha) + 255 * alpha),  # R
            )
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, trail_list[i - 1], trail_list[i], color, thickness)
            cv2.circle(frame, trail_list[i], max(1, int(4 * alpha)), color, -1)
        return frame

    

    def run(self, predict_steps=20):
        cap, images = self.setup_camera()
        height, width, _ = self.load_model()
        self.start_threads(images, cap, width, height)

        if self.FPS_counter:
            prev_frame_time = time.time()
            fps_time = prev_frame_time 
            fps_total = 0
            frame_count = 0
            fps_text = "FPS: 0"
        try:
            while True:
                result = self.output_queue.get()
                if result is None:
                    break

                if self.FPS_counter:
                    new_frame_time = time.time()
                    fps_total += 1 / (new_frame_time - prev_frame_time + 1e-5)
                    frame_count += 1
                    prev_frame_time = new_frame_time

                    if new_frame_time >= fps_time + 1:
                        fps_text = f"FPS: {fps_total / frame_count:.1f}"
                        fps_total, frame_count, fps_time = 0, 0, new_frame_time

                original_frame, inference_result = result
                detections = extract_detections(original_frame, inference_result, self.config_data)
                self.process_detections(detections)
                frame = draw_detections(detections, original_frame.copy(), self.labels)
                pz = 0

                if self.kalman_filter is not None and self.detected:

                    future_positions = self.kalman_filter.predict_n_steps(predict_steps)
                    self.future_center_x = future_positions[-1][0]
                    self.future_center_y = future_positions[-1][1]
                    
                    self.predicted_distances.append(future_positions[-1][2])
                    self.future_distance = min(self.predicted_distances)
                    for px, py, pz in future_positions:
                        px = int(px * self.resolution[0])
                        py = int(py * self.resolution[1])
                        
                        radius = int(max(1, (pz**-1)*3))
                        cv2.circle(frame, (px, py), radius, (255,0,255), -1)
                
                elif not self.detected:
                    self.future_center_x = 0
                    self.future_center_y = 0
                    self.future_distance = 0

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                frame = self.draw_trail(frame, self.trail)

                if self.FPS_counter:
                    cv2.putText(frame, fps_text, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, f"Distance {round(self.min_distance,2)}", (7, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, f"Predicted Distance {round(self.future_distance,2)}", (7, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

                if not self.display_queue.full():
                    self.display_queue.put(frame)
        
        finally:
            self.display_queue.put(None)
            cap.release()