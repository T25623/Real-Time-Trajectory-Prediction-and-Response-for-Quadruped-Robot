from framework.detection.detection import DetectionPipeline

hef_path = "config/models/balloonv8s.hef"
config_path = "config/json/balloon.json"
labels_path = "config/labels/balloon.txt"

detection = DetectionPipeline(hef_path, config_path, labels_path, FPS_counter=True, headless=False)

detection.run(predict_steps=10)