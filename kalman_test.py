from detection import DetectionPipeline

hef_path = "testing/balloonv8s.hef"
config_path = "testing/config.json"
labels_path = "testing/balloon.txt"

detection = DetectionPipeline(hef_path, config_path, labels_path)

detection.run()