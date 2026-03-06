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

class VideoInput:
    def __init__(self) -> None:
        self.resolution = (1280, 720)
        self.framerate = 30
        self.colour_format = "RGB888"

    def rpi_camera_setup():
        resolution = {"size": self.resolution, "format": self.colour_format}
        framerate = {"FrameRate": self.framerate}

        return resolution, framerate
    
    def usb_camera_setup():

class ObjectDetection:
    def __init__(self) -> None:
        self.hef_path = None
        self.config_path = None
        self.labels_path = None

        self.batch_size = 1
        self.save_output = False
        self.output_dir = None
        self.output_resolution = None
        
        self.input_queue = queue.Queue(maxsize=2)
        self.output_queue = queue.Queue(maxsize=2)


def input_image():
    global center_x, center_y, detected, min_distance
    resolution = {"size": (1280, 720), "format": "RGB888"}
    batch_size = 1
    frame_rate = 60
    framerate = {"FrameRate": frame_rate}
    save_output=False
    output_dir=None
    output_resolution=None
    
    input_queue = queue.Queue(maxsize=2)
    output_queue = queue.Queue(maxsize=2)
    
    cap, images = init_input_source("rpi", batch_size, resolution, framerate)
    labels = "/home/go2/FYP/Real-Time-Trajectory-Prediction-and-Response-for-Quadruped-Robot/testing/balloon.txt"
    labels = get_labels(labels)
    config_data = load_json_file("/home/go2/FYP/Real-Time-Trajectory-Prediction-and-Response-for-Quadruped-Robot/testing/config.json")
    hef_path = "/home/go2/FYP/Real-Time-Trajectory-Prediction-and-Response-for-Quadruped-Robot/testing/balloonv8s.hef"

    hailo_inference = HailoInfer(hef_path, batch_size)
    height, width, _ = hailo_inference.get_input_shape()

    preprocess_thread = threading.Thread(
        target=preprocess, 
        args=(images, cap, frame_rate, batch_size, input_queue, width, height),
        daemon=True
    )

    infer_thread = threading.Thread(
        target=infer, 
        args=(hailo_inference, input_queue, output_queue),
        daemon=True
    )

    preprocess_thread.start()
    infer_thread.start()

    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

    prev_frame_time = time.time()
    new_frame_time = 0
    fps_time = prev_frame_time
    fps_total = 0
    frame_count = 0
    fps_text = "FPS: 0"
    
    while True:
        result = output_queue.get()
        if result == None:
            break

        new_frame_time = time.time()

        fps = 1/(new_frame_time - prev_frame_time + 1e-5)
        fps_total += fps
        frame_count += 1 
        prev_frame_time = new_frame_time

        if new_frame_time >= fps_time + 1:
            fps_avg = fps_total / frame_count
            fps_text = f"FPS: {fps_avg:.1f}"
            fps_total = 0
            frame_count = 0
            fps_time = new_frame_time
        
        original_frame, inference_result = result
        
        detections = extract_detections(original_frame, inference_result, config_data)
        bbox = detections['detection_boxes']
        if len(bbox) > 0:
            y0 = bbox[0][0]
            x0 = bbox[0][1]
            y1 = bbox[0][2]
            x1 = bbox[0][3]

            min_distance = estimate_distance(x0, y0, x1, y1, 30, 0.275)
            detected = True

            center_x = ((y0 + y1) / 2) / 1280
            center_y = ((x0 + x1) / 2) / 720
        else:
            detected = False

        frame_with_detections = draw_detections(detections, original_frame.copy(), labels)

        frame_with_detections = cv2.cvtColor(frame_with_detections, cv2.COLOR_RGB2BGR)

        cv2.putText(frame_with_detections, fps_text, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow("Output", frame_with_detections)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        