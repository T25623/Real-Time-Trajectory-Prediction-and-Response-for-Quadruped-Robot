from go2_setup import WebRTCConnection
import go2_setup as go2
import display_lidar as dl
from detection import DetectionPipeline
import asyncio
import go2_movement as move
import threading

hef_path = "testing/balloonv8s.hef"
config_path = "testing/config.json"
labels_path = "testing/balloon.txt"

detection = DetectionPipeline(hef_path, config_path, labels_path)


async def go2():
    dog = WebRTCConnection()
    await dog.connection_setup()
    print("connected")
    pitch = 0
    while True:
        if detection.detected:
            if detection.min_distance >= 0.2 and detection.min_distance <= 0.30: 
                await asyncio.sleep(1.2)
                await move.perform_action(dog.conn, "Hello")
            else:
                forward = move.calculate_movement(detection.future_distance, 0.1, 0.2, 0.25)
                rotate = move.calculate_rotation(detection.future_center_x, 0.1, 2)
                
                pitch = move.calculate_pitch(detection.future_center_y, 0.1, 0.25, pitch) 
                await move.move_pitch(dog.conn, forward, 0, rotate, 0, pitch, 0)
        
        await asyncio.sleep(0.1)
       
detection_thread = threading.Thread(target=detection.run, daemon=True)
detection_thread.start()

asyncio.run(go2())