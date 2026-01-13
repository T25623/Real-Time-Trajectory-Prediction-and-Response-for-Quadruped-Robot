import asyncio
import cv2
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from enum import Enum
import time

class CameraSource(Enum):
    LocalCamera = 0
    WebRTC = 1
    ROS = 2

def getCameraOutput(source = 0, webRTC_connection = None, local_camera = 0):
    frame = None
    match source:
        case 0:            
            ret, frame = cam.read() 
        case 1:
            print("WebRTC Camera")
            webRTC_connection.video.switchVideoChannel(True)
            webRTC_connection.datachannel.disableTrafficSaving(True)
            webRTC_connection.video.add_track_callback(frame)
            webRTC_connection.video.set_buffer_size(1)
            webRTC_connection.datachannel.set_decoder(decoder_type="libvoxel")
        case 2:
            print("todo")
    
    return frame

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 140)
prev_frame_time = 0
new_frame_time = 0
one_sec_timer = 0

while True:
    new_frame_time = time.time()
    
    frame = getCameraOutput()
    print(time.time())
    if time.time() > one_sec_timer+1:
        one_sec_timer = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        
        fps = int(fps)
        fps = str(fps)

    prev_frame_time = new_frame_time
    cv2.putText(frame, fps, (3, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord("q"):
        break
    
