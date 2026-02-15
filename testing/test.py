import sys
from pathlib import Path
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
import hailo
import asyncio
import threading
import json
import time
import logging
import numpy as np
from queue import Queue


logging.basicConfig(level=logging.FATAL)
logging.getLogger("go2_webrtc_driver").setLevel(logging.ERROR)
logging.getLogger("aiortc").setLevel(logging.ERROR)
logging.getLogger("aioice").setLevel(logging.ERROR)


from hailo_apps.python.pipeline_apps.detection.detection_pipeline import (
    GStreamerDetectionApp,
)
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class

from go2_webrtc_driver.webrtc_driver import (
    Go2WebRTCConnection,
    WebRTCConnectionMethod,
)
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD

# Y / Z movement
center_x = 0
center_y = 0
detected = False
y = 0

# X movement
min_distance = 1000
lidar_queue = Queue(maxsize=5)


BASE = Path(
    "/home/go2/FYP/Real-Time-Trajectory-Prediction-and-Response-for-Quadruped-Robot/testing"
).expanduser()

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
            center_x, center_y, center_z = center

            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]
            
            mask = np.abs(z-1)<=1
            

            filtered_points = points[mask]
            
            if len(filtered_points) > 0:
                distances = np.linalg.norm(filtered_points - np.array([center_x, center_y, center_z]), axis=1)
                min_distance = float(np.min(distances))
            
            #print(f"min distance: {min_distance}")

        except Exception as e:
            logging.error(f"LiDAR callback error {e}")
        


def app_callback(element, buffer, state):
    global center_x, center_y, detected
    if buffer is None:
        return

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    if not detections:
        detected = False
        return

    best = max(detections, key=lambda d: d.get_confidence())
    bbox = best.get_bbox()

    x0 = bbox.xmin()
    y0 = bbox.ymin()
    x1 = bbox.xmax()
    y1 = bbox.ymax()
    center_x = (x1 + x0)/2
    center_y = (y1 + y0)/2

    detected = True


def compute_z():
    global center_x
    deadzone = 0.1

    if abs(center_x) < deadzone:
        return 0.0

    z = (center_x - 0.5) * 2
    return z

def compute_y():
    global center_y, y
    deadzone = 0.1

    if abs(center_y) < deadzone:
        return 0.0
    temp_y = (center_y - 0.5) * 0.25
    if temp_y > 0.05:
        temp_y = 0.05
    elif temp_y < -0.05:
        temp_y = -0.05
    
    if y <= 0.4 and y >= -0.4:
        y += temp_y
    
    return y

def compute_x():
    global min_distance
    deadzone = 0.1
    target_distance = 0.3
    x = 0
    if min_distance < 10:
        if min_distance < target_distance - deadzone:
            x = (min_distance - target_distance)
        elif min_distance > target_distance + deadzone:
            x = (min_distance - target_distance)
    else:
        x = 0.0
    
    return x



async def go2_connect():
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
    await conn.connect()
    
    await conn.datachannel.disableTrafficSaving(True)
    
    conn.datachannel.set_decoder(decoder_type="native")
    conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")
    conn.datachannel.pub_sub.subscribe(
        "rt/utlidar/voxel_map_compressed", lidar_callback
    )

    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["MOTION_SWITCHER"], {"api_id": 1001}
    )

    if response["data"]["header"]["status"]["code"] == 0:
        data = json.loads(response["data"]["data"])
        mode = data["name"]
        #print(f"Current motion mode: {mode}")

        if mode != "normal":
            await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"],
                {"api_id": 1002, "parameter": {"name": "normal"}},
            )

    
    print("Go2 control loop started")
    while True:
        #print(detected)
        if detected:
            z = compute_z()
            y = compute_y()
            x = compute_x()
            # print(f"min dist: {min_distance}")
            # print(f"Z is: {z}")
            # print(f"Y is: {y}")
            # print(f"X is: {x}")

            await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {
                    "api_id": SPORT_CMD["Move"],
                    "parameter": {
                        "x": 0,
                        "y": 0,
                        "z": z,
                    },
                },
            )
            await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {
                    "api_id": SPORT_CMD["Euler"],
                    "parameter": {"x": 0, "y": y, "z": 0},
                },
            )
        
        await asyncio.sleep(0.1)


        


def go2_thread_fn():
    asyncio.run(go2_connect())


def main():
    t = threading.Thread(target=go2_thread_fn, daemon=True)
    t.start()
    lidar_thread = threading.Thread(target=lidar_distance, daemon=True)
    lidar_thread.start()

    state = AppState()
    app = GStreamerDetectionApp(app_callback, state)
    app.run()

if __name__ == "__main__":
    main()
