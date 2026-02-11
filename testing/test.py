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


logging.basicConfig(level=logging.FATAL)


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

# X movement
latest_points = None
mask_state = 0
latest_center = None
min_distance = 1000

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
    global latest_points, mask_state, latest_center, min_distance
    #print(message)

    try:
        points = np.array(message["data"]["data"]["points"])
        origin = np.array(message["data"]["origin"])
        width = np.array(message["data"]["width"])
        resolution = np.array(message["data"]["resolution"])

        center_x = origin[0] + width[0]/2 * resolution[0]
        center_y = origin[1] + width[1]/2 * resolution[1]
        center_z = origin[2] + width[2]/2 * resolution[2]

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        mask = 0
        mask_state = 6
        #center = points.mean(axis=0)
        origin_x = origin[0] 
        origin_y = origin[1]
        origin_z = origin[2]
        
        max_x = np.max(x)
        max_y = np.max(y)
        max_z = np.max(z)

        # print(f"max x: {max_x}")
        # print(f"max y: {max_y}")
        # print(f"max z: {max_z}")

        min_x = np.min(x)
        min_y = np.min(y)
        min_z = np.min(z)

        # print(f"min x: {min_x}")
        # print(f"min y: {min_y}")
        # print(f"min z: {min_z}")

        # center_x = (max_x - min_x) / 2
        # center_y = (max_y - min_y) / 2
        # center_z = (max_z - min_z) / 2

        # for point in points:
        #     i = (point[0] - origin_x) / 0.05
        #     j = (point[1] - origin_y) / 0.05
        #     k = (point[2] - origin_z) / 0.05

        #     if i == 64 and j == 64:
        #         center_x = point[0]
        #         center_y = point[1]
        #         center_z = 0.2

        print(f"center x: {center_x}")
        print(f"center y: {center_y}")
        print(f"center z: {center_z}")

        if mask_state == 0:
            mask = (np.abs(x+center_x * 0.81) + y+center_y) <= 0
        elif mask_state == 1:
            mask = (np.abs(x+center_x * 0.81) - y+center_y) <= 0
        elif mask_state == 2:
            mask = (np.abs(y+center_y * 0.81) + x+center_x) <= 0
        elif mask_state == 3:
            mask = (np.abs(y+center_y * 0.81) - x+center_x) <= 0
        elif mask_state == 4:
            mask = (x+center_x)**2 + (y+center_y)**2 <= 10
        elif mask_state == 5:
            mask = (x+center_x)**2 + (y+center_y)**2 >= 2
        elif mask_state == 6:
            mask = np.abs(z-1)<=1
        else:
            mask = np.ones(len(points), dtype=bool)

        filtered_points = points[mask]
        
        for point in filtered_points:
            distance = np.linalg.norm(point - np.array([center_x, center_y, center_z]))
            if distance <= min_distance:
                min_distance = distance
        
        print(f"min distance: {min_distance}")
        with points_lock:
            latest_points = filtered_points
            latest_center = np.array([center_x, center_y, center_z])

    except Exception as e:
        logging.error(f"LiDAR callback error: {e}")

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

    #print(f"X0={round(x0, 3)} X1={round(x1, 3)} Y0={round(y0, 3)} Y1={round(y1, 3)}")

def compute_z():
    global center_x
    deadzone = 0.1

    if abs(center_x) < deadzone:
        return 0.0

    z = (center_x - 0.5)
    return z

def compute_y():
    global center_y
    deadzone = 0.1

    if abs(center_y) < deadzone:
        return 0.0
    
    y = (center_y - 0.5)
    return y

def compute_x():
    global min_distance
    deadzone = 0.5
    target_distance = 4
    x = 0

    if min_distance < target_distance - deadzone:
        x = (min_distance - target_distance) / 100
    elif min_distance > target_distance + deadzone:
        x = (min_distance - target_distance) / 100
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
        print(f"Current motion mode: {mode}")

        if mode != "normal":
            await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"],
                {"api_id": 1002, "parameter": {"name": "normal"}},
            )

    await asyncio.sleep(1)
    print("Go2 control loop started")
    while True:
        print(detected)
        if detected:
            z = compute_z()
            y = compute_y()

            print(f"Z is: {z}")
            print(f"Y is: {y}")

            await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {
                    "api_id": SPORT_CMD["Move"],
                    "parameter": {
                        "x": 0.0,
                        "y": y,
                        "z": z,
                    },
                },
            )  

        await asyncio.sleep(0.1)


def go2_thread_fn():
    asyncio.run(go2_connect())


def main():
    t = threading.Thread(target=go2_thread_fn, daemon=True)
    t.start()

    state = AppState()
    app = GStreamerDetectionApp(app_callback, state)
    app.run()

if __name__ == "__main__":
    main()
