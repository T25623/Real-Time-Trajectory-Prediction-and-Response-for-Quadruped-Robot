from framework.detection.detection import DetectionPipeline

hef_path = "config/balloonv8s.hef"
config_path = "config/config.json"
labels_path = "config/balloon.txt"

detection = DetectionPipeline(hef_path, config_path, labels_path, FPS_counter=True)

detection.run(predict_steps=10)