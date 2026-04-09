from framework.go2.go2_setup import WebRTCConnection
import framework.go2.go2_setup as go2
import framework.lidar.display_lidar as dl
from framework.detection.detection import DetectionPipeline
import asyncio
import framework.go2.go2_movement as move
import threading
import numpy as np
import pyvista as pv
from framework.utils.go2_utils import rotate_vector, calculate_facing_correction

hef_path = "config/balloonv8s.hef"
config_path = "config/config.json"
labels_path = "config/balloon.txt"

detection = DetectionPipeline(hef_path, config_path, labels_path)
detection.FPS_counter = True

message = None
dog = None
vector = None 

def lidar_visuals():
    global message, dog, vector, slider_x, slider_y
    plotter = pv.Plotter(off_screen=False)
    plotter.show(interactive_update=True)
    while True:
        if not message == None:
            points = np.array(message["data"]["data"]["points"])
 
            adjusted_origin = (
                dog.lidar_origin[0],
                dog.lidar_origin[1],
                dog.lidar_origin[2],  
            )

            vector = dl.run(plotter, points, adjusted_origin, 0.3, dog.orientation)

 

async def go2():
    global message, dog, vector
    dog = WebRTCConnection()
    await dog.connection_setup()
    dog.lidar_setup(True, 0)

    print("connected")
    pitch = 0
    while True:
        dog.state_call()
        message = dog.lidar_queue
        
        if vector is not None:

            facing_correction = calculate_facing_correction(dog.orientation)
    
            new_vector = rotate_vector(vector, facing_correction)
            await move.go2_movement(dog.conn, (new_vector[1]*0.1), (-new_vector[0]*0.1), 0)
        elif detection.detected:
            if detection.future_distance >= -0.1 and detection.future_distance <= 0.3: 
                await asyncio.sleep(1.5)
                print("now")
                resp = await move.perform_action(dog.conn, "Hello")
                print(resp)
                pitch = 0
            else:
                forward = move.calculate_movement(detection.future_distance, 0.1, 0.2, 0.25)
                rotate = move.calculate_rotation(detection.future_center_x, 0.1, 2)
                
                pitch = move.calculate_pitch(detection.future_center_y, 0.1, 0.25, pitch) 
                await move.move_pitch(dog.conn, forward, 0, rotate, 0, pitch, 0)
        
        await asyncio.sleep(0.05)
       
detection_thread = threading.Thread(target=detection.run, daemon=True)
detection_thread.start()


lidar_thread = threading.Thread(target=lidar_visuals, daemon=True)
lidar_thread.start()

asyncio.run(go2())