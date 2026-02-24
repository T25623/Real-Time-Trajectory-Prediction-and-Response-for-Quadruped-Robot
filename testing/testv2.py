import sys
from pathlib import Path
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
import hailo
import asyncio
import threading
import json
import logging
import numpy as np
import time
from queue import Queue

logging.basicConfig(level=logging.FATAL)
logging.getLogger("go2_webrtc_driver").setLevel(logging.ERROR)
logging.getLogger("aiortc").setLevel(logging.ERROR)
logging.getLogger("aioice").setLevel(logging.ERROR)

from hailo_apps.python.pipeline_apps.detection.detection_pipeline import GStreamerDetectionApp
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD

conn = None

center_x = 0
center_y = 0
detected = False
y = 0
min_distance = 1000
lidar_queue = Queue(maxsize=5)
last_z = 0
last_y = 0
perfrom_action = False
cooldown_timer = 0

BASE = Path("/home/go2/FYP/Real-Time-Trajectory-Prediction-and-Response-for-Quadruped-Robot/testing").expanduser()

sys.argv = [
    sys.argv[0],
    "--hef-path", str(BASE / "balloonv8s.hef"),
    "--labels-json", str(BASE / "balloon.json"),
    "--input", "rpi",
    "--frame-rate", "60",
    "--show-fps",
]

class AppState(app_callback_class):
    pass

def lidar_callback(message):
    if not lidar_queue.full():
        lidar_queue.put(message)

def lidar_distance():
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

            x = points[:, 0]
            y_p = points[:, 1]
            z_p = points[:, 2]
            
            mask = np.abs(z_p-1) <= 1
            filtered_points = points[mask]
            
            if len(filtered_points) > 0:
                distances = np.linalg.norm(filtered_points - np.array([center_x_l, center_y_l, center_z_l]), axis=1)
                min_distance = float(np.min(distances))
        except Exception as e:
            logging.error(f"LiDAR callback error {e}")

def estimate_distance(x0, y0, x1, y1, real_height, focal_length):
    box_height = abs(y1 - y0)

    if box_height == 0:
        return None

    distance = (((real_height * focal_length) / box_height) * 4) / 100
    return distance 

def app_callback(element, buffer, state):
    global center_x, center_y, detected, min_distance
    if buffer is None:
        detected = False
        return

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    if not detections:
        detected = False
        return

    best = max(detections, key=lambda d: d.get_confidence())
    bbox = best.get_bbox()
    center_x = (bbox.xmax() + bbox.xmin()) / 2
    center_y = (bbox.ymax() + bbox.ymin()) / 2

    min_distance = estimate_distance(bbox.xmin(),  bbox.ymin(), bbox.xmax(), bbox.ymax(), 30, 0.275)
    detected = True

def compute_z():
    global center_x
    deadzone = 0.01 

    offset = center_x - 0.5
    if abs(offset) < deadzone:
        offset = 0.0

    return offset * 2  

def compute_y():
    global center_y, y
    deadzone = 0.01
    if abs(center_y - 0.5) < deadzone:
        return y

    temp_y = (center_y - 0.5) * 0.02
    
    
    if -0.4 <= (y + temp_y) <= 0.4:
        y += temp_y
    
    return y

def compute_x():
    global min_distance, perfrom_action
    deadzone = 0.1
    target_distance = 0.7
    x = 0.0
    if min_distance < 2:
        if min_distance < target_distance - deadzone:
            x = -min_distance
        elif min_distance > target_distance + deadzone:
            x = min_distance

    return x / 4

def action_cooldown_check(cooldown_seconds = 5):
    global cooldown_timer
    
    current_time = time.time()

    if current_time >= cooldown_timer+cooldown_seconds:
        print(f"Current: {current_time}")
        print(f"Cooldown: {cooldown_timer}")
        print(f"Cooldown Seconds: {cooldown_seconds}")
        cooldown_timer = current_time
        return True
    else:
        return False


def go2_interact(action):
    global perfrom_action, conn
    
    conn.datachannel.pub_sub.publish_request_no_wait(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD[action],
            "parameter": {"data": True}
        }
    )            
    perfrom_action = False

async def go2_setup():
    global conn
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
    await conn.connect()

    await conn.datachannel.disableTrafficSaving(True)
    #conn.datachannel.set_decoder(decoder_type="native")

    #conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")
    
    #conn.datachannel.pub_sub.subscribe(
    #        "rt/utlidar/voxel_map_compressed", lidar_callback
    #)

    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["MOTION_SWITCHER"], {"api_id": 1001}
    )
    if response["data"]["header"]["status"]["code"] == 0:
        data = json.loads(response["data"]["data"])
        mode = data["name"]
        if mode != "normal":
            await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"], {"api_id": 1002, "parameter": {"name": "normal"}}
            )
    print("Go2 setup complete")
    return conn

async def go2_movement_loop(conn):
    global perfrom_action
    print("Go2 movement loop started")
    while True:
        if detected:
            z = compute_z()      
            x_val = compute_x()   

        else:
            z = 0.0
            x_val = 0.0
        
        if detected:
            print(f"min Distance: {min_distance}")
            if abs(min_distance - 0.5) <= 0.1:
                if action_cooldown_check():
                    # go2_interact("Hello")
                    print("Hello")
            else:
                response = conn.datachannel.pub_sub.publish_request_no_wait(
                    RTC_TOPIC["SPORT_MOD"],
                    {
                        "api_id": SPORT_CMD["Move"],
                        "parameter": {"x": x_val, "y": 0, "z": z},
                        
                    },
                )

        await asyncio.sleep(0.02) 

async def go2_movement_loop2(conn):
    global last_y
    last_y = 0.0
    print("Go2 movement2 loop started")
    while True:
        if detected:
            y_val = compute_y()
            last_y = y_val

            print(f"Y val: {y_val}")

            response = conn.datachannel.pub_sub.publish_request_no_wait(
                RTC_TOPIC["SPORT_MOD"],
                {
                    "api_id": SPORT_CMD["Euler"],
                    "parameter": {"x": 0, "y": y_val, "z": 0},
                },
            )

        await asyncio.sleep(0.02) 

def run_lidar_loop():
    lidar_distance()


async def main_async():

    conn = await go2_setup()

    task1 = asyncio.create_task(go2_movement_loop(conn))
    task2 = asyncio.create_task(go2_movement_loop2(conn))

    await asyncio.gather(task1, task2)

def start_async_loop():
    asyncio.run(main_async())


def main():
    state = AppState()
    app = GStreamerDetectionApp(app_callback, state)

    threading.Thread(
        target=start_async_loop,
        daemon=True
    ).start()

    app.run()

if __name__ == "__main__":
    main()
