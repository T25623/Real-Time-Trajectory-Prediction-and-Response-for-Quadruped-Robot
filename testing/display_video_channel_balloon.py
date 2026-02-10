import math
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np
from ultralytics import YOLO
from collections import deque

height, width = 720, 1280
img = np.zeros((height, width, 3), dtype=np.uint8)
cv2.imshow("Video", img)
cv2.waitKey(1)  

import asyncio
import logging
import threading
import time
from queue import Queue
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
from aiortc import MediaStreamTrack
import json

model = YOLO("/home/igor/Nextcloud/College/Year_4/FYP_Geo2/runs/detect/train4/weights/best.pt")
logging.basicConfig(level=logging.FATAL)

last_euler_time = 0
y = 0
lidar_data = 0
motion_magnitude = 0
dy = 0

fps = 0.0
last_fps_time = time.time()
frame_count = 0

balloon_history = deque(maxlen=3)


def set_lidar_data(message):
    global lidar_data
    lidar_data = message["data"]
    #print(lidar_data)


def main():
    frame_queue = Queue(maxsize=1)
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)

    def locate_green(img):
        global go2_sit_state
        green_found = False
        center_x = 0
        center_y = 0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = (45, 120, 120)
        upper_green = (75, 255, 255)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            if (cv2.contourArea(largest_contour)>200):
                biggest_mask = np.zeros_like(mask) 

                cv2.drawContours(biggest_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

                result = cv2.bitwise_and(img, img, mask=biggest_mask)

                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                center_x = x+w/2
                center_y = y+h/2

                green_found = True
                    
            else:
                green_found = False

        
        return img, green_found, center_x, center_y

    def detect_balloon(frame):
        results = model(frame)[0] 
        balloon_found = False
        center_x, center_y = 0, 0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id != 0: 
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            balloon_found = True
            break  

        return frame, balloon_found, center_x, center_y


    async def maintain_euler():
        global y
        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            {
                "api_id": SPORT_CMD["Euler"],
                "parameter": {"x": 0, "y": y, "z": 0},
            },
        )
        await asyncio.sleep(0.1)
    
    async def handle_green_detection(center_x, center_y):
        global last_euler_time
        global y

        x_tolerance = 50
        y_tolerance = 30
        z = 0
        

        if (center_x > width / 2 + x_tolerance or center_x < width / 2 - x_tolerance):
            x_offset = center_x - width / 2
            x_percentage_offset = x_offset / (width / 2)
            z = -(x_percentage_offset * 60) * 0.01

        if (center_y > height/2 + y_tolerance or center_y < height/2 - y_tolerance):
            if (center_y > height/2 + y_tolerance and y <= 0.4):
                if motion_magnitude > 100:
                    y+=0.03
                else:
                    y += 0.02
            elif (center_y < height/2 - y_tolerance and y >= -0.4):
                if motion_magnitude > 100:
                    y+= -0.03
                else:
                    y += -0.02


        print(f"y={y}, z={z}")

        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            {
                "api_id": SPORT_CMD["Move"],
                "parameter": {"x": 0, "y": 0, "z": z},
            },
        )

    def draw_future_path(img, history, steps=10):
        global motion_magnitude, dy
        if len(history) < 2:
            return img

        (x1, y1), (x2, y2) = history[-2], history[-1]

        dx = (x2 - x1)
        dy = (y2 - y1)

        start_point = (int(x2), int(y2))
        end_point = (
            int(x2 + dx * steps),
            int(y2 + dy * steps),
        )

        motion_magnitude = math.hypot(dx, dy)

        print(f"Object speed: {motion_magnitude}")

        cv2.arrowedLine(
            img,
            start_point,
            end_point,
            (0, 0, 255),
            2,
            tipLength=0.15
        )

        return img


    async def recv_camera_stream(track: MediaStreamTrack):
        global y
        global fps, last_fps_time, frame_count
        no_detection_count = 0


        while True:
            tolerance = 150
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            img, balloon_found, center_x, center_y = detect_balloon(img)

            frame_count += 1
            current_time = time.time()
            elapsed = current_time - last_fps_time

            if elapsed >= 1.0: 
                fps = frame_count / elapsed
                frame_count = 0
                last_fps_time = current_time

            cv2.putText(
                img,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            increment = False
            # if lidar_data:
            #     resolution = lidar_data['resolution']        
            #     origin = np.array(lidar_data['origin'], dtype=np.float32)

            #     positions_data = lidar_data['data']['positions']

            #     positions = np.array(positions_data, dtype=np.float32).reshape(-1, 3)
            #     points_world = positions * resolution + origin

            #     distances = np.linalg.norm(points_world, axis=1)
            #     if len(distances) > 0:
            #         # mask = distances >= 3
            #         # filtered = distances[mask]

            #         nearest_distance = np.min(distances)
            #         print(f"distances {distances.mean()}")
            #         print(f"Nearest obstacle distance (any direction): {nearest_distance:.2f} m")
            #     else:
            #         print("No obstacles detected.")
            
            

            #img_clean = img.copy()
            
            if balloon_found and center_x != 0 and center_y != 0:
                z = 0
                no_detection_count = 0
                asyncio.run_coroutine_threadsafe(maintain_euler(), loop)
                asyncio.create_task(handle_green_detection(center_x, center_y))
                balloon_history.append((center_x, center_y))
                img = draw_future_path(img, balloon_history, steps=8)
            else:
                no_detection_count+=1
            
            print(f"no detection count {no_detection_count}")
            if no_detection_count >= 20 and not balloon_found:
                #img = img_clean.copy()
                print("here")
                if y > 0.015:
                    print("down")
                    y = y - 0.015
                    asyncio.run_coroutine_threadsafe(maintain_euler(), loop)
                elif y < -0.015:
                    print("up")
                    y = y + 0.015
                    asyncio.run_coroutine_threadsafe(maintain_euler(), loop)
                else:
                    print("level")
                    y = 0
                
            
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except:
                    pass
            frame_queue.put_nowait(img)


    def run_asyncio_loop(loop):
        asyncio.set_event_loop(loop)

        async def setup():
            try:
                await conn.connect()
                
                conn.video.switchVideoChannel(True)
                await conn.datachannel.disableTrafficSaving(True)
                conn.video.add_track_callback(recv_camera_stream)
                conn.video.set_buffer_size(1)
                conn.datachannel.set_decoder(decoder_type="libvoxel")

                #conn.datachannel.pub_sub.subscribe("rt/utlidar/voxel_map_compressed", set_lidar_data)

                #conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")


                try:
                    response = await conn.datachannel.pub_sub.publish_request_new(
                        RTC_TOPIC["MOTION_SWITCHER"],
                        {"api_id": 1002, "parameter": {"name": "normal"}},
                    )
                    print("standing up")
                    print(response)

                    if response["data"]["header"]["status"]["code"] == 0:
                        data = json.loads(response["data"]["data"])
                        current_motion_switcher_mode = data["name"]
                        print(f"Current motion mode: {current_motion_switcher_mode}")
                except Exception as e:
                    print(f"Failed to send sit command: {e}")  

            except Exception as e:
                logging.error(f"Error in WebRTC connection: {e}")

        loop.run_until_complete(setup())
        loop.run_forever()

    loop = asyncio.new_event_loop()

    asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,))
    asyncio_thread.start()

    try:
        while True:
            if not frame_queue.empty():
                img = frame_queue.get()
                
                cv2.imshow("Video", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(.1)
    finally:
        cv2.destroyAllWindows()
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join()


if __name__ == "__main__":
    main()



