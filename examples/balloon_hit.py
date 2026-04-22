from framework.robot.go2.setup import WebRTCConnection
import framework.robot.go2.setup as go2
import framework.robot.go2.lidar as dl
from framework.robot.go2.setup import Objective
from framework.detection.detection import DetectionPipeline
import asyncio
import framework.robot.go2.movement as move
import threading
import numpy as np
import time 

hef_path = "config/models/balloonv8s.hef"
config_path = "config/json/balloon.json"
labels_path = "config/labels/balloon.txt"

detection = None
ms_per_frame = None
robot = None
robot_manual_control = False
lidar_image = None

def start_detection():
    global detection

    detection = DetectionPipeline(hef_path, config_path, labels_path, headless=True, resolution=(1280, 720), framerate=60)

    detection.running = True
    detection_thread = threading.Thread(target=detection.run, args=(robot,), daemon=True)

    detection_thread.start()



# Robot Connection
async def robot_connection_setup():
    global robot
    response_action = "Hello"
    while robot is None:
        robot = WebRTCConnection()
        await robot.connection_setup("LocalAP" )
        await asyncio.sleep(1)
    
    lidar_thread()
    robot.movement_speed = 0.2
    robot.rotate_speed = 2
    robot.pitch_speed = 0.5

    while robot is not None:
        if robot.conn.datachannel.pub_sub.channel.readyState != "open":
            robot = None
            break

        robot.status_check()
        await robot.low_battery_action()

        if detection is not None:
            no_detection_time = detection.no_detection_time
            detection_time = detection.detection_time
            detected = detection.detected
            future_distance = detection.future_distance
            future_center_x = detection.future_center_x
            future_center_y = detection.future_center_y

            if None not in (detection_time, future_distance, future_center_x, future_center_y):
                await move.movement_response(robot, detection_time, no_detection_time, detected, future_distance, future_center_x, future_center_y, Objective.Track_Hit, response_action)
  
        await asyncio.sleep(0.1)

    


def lidar_display():
    global robot

    st = time.time()
    while robot is not None and robot.lidar_data is not None:

        if (time.time() - st >= 0.2):
            points = robot.lidar_data

            facing = np.array(robot.orientation)
            facing = facing / np.linalg.norm(facing) 

            sensor_offset = 0.3
            origin = np.array(robot.lidar_origin) + facing * sensor_offset

            robot.avoid_vector = go2_lidar.run(points, origin, 0.3, robot.orientation)

            if not bool(boundary_var):
                robot.avoid_vector = None

            st = time.time()
    

def lidar_thread():
    if robot is not None:
        robot.lidar_setup(True, 0)

        lidar_thread = threading.Thread(target=lidar_display, daemon=True)
        lidar_thread.start() 


start_detection()
asyncio.run(robot_connection_setup())
