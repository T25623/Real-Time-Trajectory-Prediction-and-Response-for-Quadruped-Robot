import sys
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")

import hailo
from gi.repository import Gst

from hailo_apps.python.pipeline_apps.detection.detection_pipeline import (
    GStreamerDetectionApp,
)
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class


BASE = Path("~/FYP/test").expanduser()

sys.argv = [
    sys.argv[0],
    "--hef-path", str(BASE / "balloonv8s.hef"),
    "--labels-json", str(BASE / "balloon.json"),
    "--input", "rpi",
    "--frame-rate", "70",
    "--show-fps",
]

class AppState(app_callback_class):
    pass

def app_callback(element, buffer, state):
    if buffer is None:
        return

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    if not detections:
        return

    best = max(detections, key=lambda d: d.get_confidence())

    bbox = best.get_bbox()
    x1_norm = bbox.xmin()
    y1_norm = bbox.ymin()
    x2_norm = bbox.xmax()
    y2_norm = bbox.ymax()
    x1 = int(x1_norm * 640)
    y1 = int(y1_norm * 640)
    x2 = int(x2_norm * 640)
    y2 = int(y2_norm * 640)

    print(f"{(x1 - x2)/2} {(y1 - y2)/2}")
def main():
    state = AppState()
    app = GStreamerDetectionApp(app_callback, state)
    app.run()


if __name__ == "__main__":
    main()
