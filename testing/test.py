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

center_x = 0
center_y = 0
z = 0
detected = False

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

    z = (center_x - 0.5) * 1.2
    return z


async def go2_connect():
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
    await conn.connect()

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
            print(f"Z is: {z}")
            await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {
                    "api_id": SPORT_CMD["Move"],
                    "parameter": {
                        "x": 0.0,
                        "y": 0.0,
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
