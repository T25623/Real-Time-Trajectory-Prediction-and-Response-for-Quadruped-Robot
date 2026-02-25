import threading
import asyncio
import time
import queue
import numpy as np
from pathlib import Path
import cv2
from functools import partial


# Hailo imports
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

# Go2 imports
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD

# -------------------- GLOBALS --------------------

center_x = 0
center_y = 0
min_distance = 1000
detected = False
y = 0
last_y = 0
perfrom_action = False
cooldown_timer = 0
lidar_queue = queue.Queue(maxsize=5)
conn = None


# -------------------- LIDAR --------------------
def lidar_callback(message):
    if not lidar_queue.full():
        lidar_queue.put(message)

def lidar_distance_loop():
    global lidar_queue, min_distance
    while True:
        message = lidar_queue.get()
        try:
            data = message["data"]
            points = np.array(data["data"]["points"], dtype=float)
            origin = np.array(data["origin"], dtype=float)
            width = np.array(data["width"], dtype=float)
            resolution = np.array(data["resolution"], dtype=float)
            min_distance = 1000
            center = origin + (width / 2.0) * resolution
            center_x_l, center_y_l, center_z_l = center
            mask = np.abs(points[:, 2] - 1) <= 1
            filtered_points = points[mask]
            if len(filtered_points) > 0:
                distances = np.linalg.norm(filtered_points - np.array([center_x_l, center_y_l, center_z_l]), axis=1)
                min_distance = float(np.min(distances))
        except Exception as e:
            print(f"LiDAR error: {e}")

# -------------------- GO2 MOVEMENT --------------------
def compute_z():
    deadzone = 0.01
    offset = center_x - 0.5
    if abs(offset) < deadzone:
        offset = 0.0
    return offset * 2

def compute_y():
    global y
    deadzone = 0.01
    if abs(center_y - 0.5) < deadzone:
        return y
    temp_y = (center_y - 0.5) * 0.02
    if -0.4 <= (y + temp_y) <= 0.4:
        y += temp_y
    return y

def compute_x():
    deadzone = 0.1
    target_distance = 0.7
    x = 0.0
    if min_distance < 2:
        if min_distance < target_distance - deadzone:
            x = -min_distance
        elif min_distance > target_distance + deadzone:
            x = min_distance
    return x / 4

def action_cooldown_check(cooldown_seconds=5):
    global cooldown_timer
    current_time = time.time()
    if current_time >= cooldown_timer + cooldown_seconds:
        cooldown_timer = current_time
        return True
    return False

def go2_interact(action):
    global perfrom_action, conn
    conn.datachannel.pub_sub.publish_request_no_wait(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD[action], "parameter": {"data": True}}
    )
    perfrom_action = False

async def go2_setup():
    global conn
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
    await conn.connect()
    await conn.datachannel.disableTrafficSaving(True)
    print("Go2 setup complete")
    return conn

async def go2_movement_loop(conn):
    while True:
        if detected:
            z = compute_z()
            x_val = compute_x()
        else:
            z = 0.0
            x_val = 0.0

        if detected and abs(min_distance - 0.5) <= 0.1:
            if action_cooldown_check():
                print("Hello")
        else:
            print(f"X val: {x_val}")
            print(f"z: {z}")
            conn.datachannel.pub_sub.publish_request_no_wait(
                RTC_TOPIC["SPORT_MOD"],
                {"api_id": SPORT_CMD["Move"], "parameter": {"x": x_val, "y": 0, "z": z}}
            )
        await asyncio.sleep(0.02)

async def go2_movement_loop2(conn):
    global last_y
    last_y = 0.0
    while True:
        if detected:
            y_val = compute_y()
            last_y = y_val
            print(f"Y val: {y_val}")
            conn.datachannel.pub_sub.publish_request_no_wait(
                RTC_TOPIC["SPORT_MOD"],
                {"api_id": SPORT_CMD["Euler"], "parameter": {"x": 0, "y": y_val, "z": 0}}
            )
        await asyncio.sleep(0.02)

def start_async_loop():
    asyncio.run(main_async())

async def main_async():
    conn = await go2_setup()
    task1 = asyncio.create_task(go2_movement_loop(conn))
    task2 = asyncio.create_task(go2_movement_loop2(conn))
    await asyncio.gather(task1, task2)


# -------------------- Hailo --------------------
def input_image():
    resolution = {"size": (1280, 720), "format": "RGB888"}
    batch_size = 1
    frame_rate = 80
    framerate = {"FrameRate": frame_rate}
    save_output=False
    output_dir=None
    output_resolution=None
    
    input_queue = queue.Queue(maxsize=2)
    output_queue = queue.Queue(maxsize=2)
    
    cap, images = init_input_source("rpi", batch_size, resolution, framerate)
    labels = "balloon.txt"
    labels = get_labels(labels)
    config_data = load_json_file("config.json")
    hef_path = "balloonv8s.hef"

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
        frame_with_detections = draw_detections(detections, original_frame.copy(), labels)

        frame_with_detections = cv2.cvtColor(frame_with_detections, cv2.COLOR_RGB2BGR)

        cv2.putText(frame_with_detections, fps_text, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow("Output", frame_with_detections)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


    


# -------------------- MAIN --------------------
def main():
    # ----- Set hardcoded args -----
    # BASE = Path("/home/go2/FYP/Real-Time-Trajectory-Prediction-and-Response-for-Quadruped-Robot/testing").expanduser()
    # hef_path = str(BASE / "balloonv8s.hef")
    # labels = str(BASE / "balloon.json")
    # input_src = "rpi"       # your camera input
    # batch_size = 1
    # output_dir = str(BASE / "output")
    # save_output = False
    # camera_resolution = "sd"
    # output_resolution = "sd"
    # enable_tracking = False
    # show_fps = True
    # frame_rate = 60
    # draw_trail = False
    
    threading.Thread(target=start_async_loop, daemon=True).start()
    threading.Thread(target=lidar_distance_loop, daemon=True).start()

    input_image()

if __name__ == "__main__":
    main()