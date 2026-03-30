import tkinter as tk
from tkinter.ttk import Combobox
import cv2
import threading
import asyncio
import logging
from queue import Queue
from PIL import Image, ImageTk
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
from aiortc import MediaStreamTrack
import json
import numpy as np

hold_euler = False
euler_target = {"x": 0, "y": 0, "z": 0}
lidar_data = 0
write = True


async def go2_camera_stream(track: MediaStreamTrack):
    while True:
        frame = await track.recv()
        img = frame.to_ndarray(format="bgr24")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im_pil)

        videoLabel.after(0, lambda imgtk=imgtk: update_video_label(imgtk))
        await asyncio.sleep(0.01)

def write_lidar_data():
    global write
    if write:
        try:
            with open("lidar_data2.json", "w") as f:
                json.dump({
                    "stamp": lidar_data["stamp"],
                    "frame_id": lidar_data["frame_id"],
                    "resolution": lidar_data["resolution"],
                    "src_size": lidar_data["src_size"],
                    "origin": lidar_data["origin"],            
                    "width": lidar_data["width"],              
                    "point_count": lidar_data["data"]["point_count"],
                    "face_count": lidar_data["data"]["face_count"],
                    "positions": lidar_data["data"]["positions"].tolist(),
                    "uvs": lidar_data["data"]["uvs"].tolist(),
                    "indices": lidar_data["data"]["indices"].tolist()
                }, f)
            write = False
            print("written")
        except Exception as e:
            print("Error writing file:", e)


def update_video_label(imgtk):
    videoLabel.imgtk = imgtk
    videoLabel.configure(image=imgtk)

def set_lidar_data(message):
    global lidar_data
    lidar_data = message["data"]
    #print("💡 LiDAR data updated")

def run_loop(loop):
    asyncio.set_event_loop(loop)

    async def setup():
        try:
            await conn.connect()
            conn.video.switchVideoChannel(True)
            conn.video.add_track_callback(go2_camera_stream)
            await conn.datachannel.disableTrafficSaving(True)
            conn.datachannel.set_decoder(decoder_type="libvoxel")
            #conn.datachannel.pub_sub.subscribe("rt/utlidar/voxel_map_compressed", set_lidar_data)
            #conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")
            print("🔧 Switching to 'normal' motion mode...")
            response = await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"],
                {"api_id": 1002, "parameter": {"name": "sport"}},
            )
            
            print("✅ Motion mode set:", response)
        except Exception as e:
            logging.error(f"Error in WebRTC connection {e}")
        
        asyncio.create_task(maintain_euler())

    async def main():
        asyncio.create_task(setup())
        asyncio.create_task(check_min_distance())
        await asyncio.Event().wait()  
    
    loop.run_until_complete(main())


def capture_frame():
    global lidar_data

    ret, frame = video.read()
    if ret:
        cv2Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2Image)
        imgtk = ImageTk.PhotoImage(image=img)
        videoLabel.imgtk = imgtk
        videoLabel.configure(image=imgtk)
    
    videoLabel.after(10, capture_frame)

async def jump():
    await conn.datachannel.pub_sub.publish_request_new(
    RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD["ContinuousGait"],
        "parameter": {"x": 0.6, "y": 0.0, "z": 0.0}}
    )
    await asyncio.sleep(0.2)
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD["BodyHeight"], "parameter": {"data": +0.10}}
    )



async def perform_action(action):
    try:
        print(f"\n=== ACTION REQUESTED: {action} ===")

        fast_actions = []

        request_payload = 0

        if action in fast_actions:
            print("Setting SpeedLevel to 3 for fast-response action...")
            rsp = await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {"api_id": SPORT_CMD["SpeedLevel"], "parameter": {"data": 3}},
            )
            print(f"SpeedLevel response: {rsp}")

            await asyncio.sleep(0.05) 
        elif action == "BodyHeight":
            request_payload = {
                "api_id": SPORT_CMD[action],
                "parameter": {"data": body_height_var.get()}
            }
        else:
            request_payload = {
                "api_id": SPORT_CMD[action],
                "parameter": {"data": True}
            }



        print(f"➡️ Sending action: {action}")


        response = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            request_payload
        )

        print("RAW RESPONSE FROM ROBOT:")
        print(response)

        success = False
        if response and isinstance(response, dict):
            if "message" in response:
                print(f" Robot says: {response['message']}")
            if "result" in response:
                success = response["result"] == 0
                print(f"Result code: {response['result']}")

        # Final status message
        if success:
            print(f"Action '{action}' executed successfully!\n")
        else:
            print(f" Action '{action}' FAILED!\n")

        return success

    except Exception as e:
        print(f"Exception while sending command {action}: {e}")
        return False


async def maintain_euler():
    while True:
        if hold_euler:
            try:
                await conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {
                        "api_id": SPORT_CMD["Euler"],
                        "parameter": euler_target,
                    },
                )
            except Exception as e:
                print(f" Euler hold failed: {e}")
        await asyncio.sleep(0.05)  # 20 Hz update rate


async def move_action(x=0, y=0, z=0):
    try:
        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            {"api_id": SPORT_CMD["Pose"], "parameter": {"data": False}},
        )
        await asyncio.sleep(0.05)  # 20 Hz update rate


        print(f" Moving: x={x}, y={y}, z={z}")
        response = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            {
                "api_id": SPORT_CMD["Move"],
                "parameter": {"x": x, "y": y, "z": z},  
            },
        )
    
        print(" Move response:", response)
    except Exception as e:
        print(f"❌ Failed to send move command {e}")


def move_action_btn(x=0, y=0, z=0):
    asyncio.run_coroutine_threadsafe(move_action(x, y, z), loop)


def on_close():
    video.release()
    root.destroy()

async def check_min_distance():
    while True:
        if lidar_data:
            try:
                positions = lidar_data["data"]["positions"]
                if positions is None or len(positions) < 3:
                    await asyncio.sleep(0.5)
                    continue

                origin = np.array(lidar_data["origin"])
                resolution = lidar_data["resolution"]
                width = np.array(lidar_data["width"])

                pos_q = positions.reshape(-1, 3).astype(np.float32)
                scale = width * resolution
                pos = origin + (pos_q / 255.0) * scale

                front_pts = pos

                if front_pts.size == 0:
                    await asyncio.sleep(0.5)
                    continue

                dists = np.linalg.norm(front_pts[:, :2], axis=1)
                closest = float(np.min(dists))
                # print("mean:", dists.mean())
                # print("Closest distance:", closest)

            except Exception as e:
                print(f"❌ Failed distance calc: {e}")

        await asyncio.sleep(0.001)




# ---------- SETUP ----------
logging.basicConfig(level=logging.FATAL)
conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
loop = asyncio.new_event_loop()



root = tk.Tk()
root.title("Go2 WebRTC GUI")

videoLabel = tk.Label(root)
videoLabel.grid(column=0, row=0, rowspan=4, padx=10, pady=10)

go2_actions = ("StandDown", "StandUp", "Sit", "RiseSit", "Hello", "Handstand" , "Pose", "FrontJump", "Wallow", "Stretch", "MoonWalk", "Handstand", "FrontPounce", "BodyHeight")
go2_actions_combo = Combobox(root, values=go2_actions, state="readonly")
go2_actions_combo.grid(column=1, row=0, padx=5, pady=5)

action_button = tk.Button(
    root,
    text="Perform Action",
    command=lambda: asyncio.run_coroutine_threadsafe(
        perform_action(go2_actions_combo.get()), loop
    ),
)
action_button.grid(column=1, row=1, padx=5, pady=5)

movement_frame = tk.LabelFrame(root, text="Movement Controls", padx=10, pady=10)
movement_frame.grid(column=1, row=2, padx=10, pady=10)

euler_frame = tk.LabelFrame(root, text="Posture Control", padx=10, pady=10)
euler_frame.grid(column=1, row=3, padx=10, pady=10)

def set_euler(y=0):
    global euler_target, hold_euler
    euler_target = {"x": 0, "y": y, "z": 0}
    hold_euler = True
    print(f" Holding Euler posture y={y}")

def clear_euler():
    global hold_euler
    hold_euler = False
    print(" Stopped Euler hold")


tk.Button(euler_frame, text="Tilt Forward", width=12,
          command=lambda: set_euler(-0.5)).grid(row=0, column=0, padx=5)
tk.Button(euler_frame, text="Level", width=12,
          command=lambda: set_euler(0)).grid(row=0, column=1, padx=5)
tk.Button(euler_frame, text="Tilt Back", width=12,
          command=lambda: set_euler(0.5)).grid(row=0, column=2, padx=5)
tk.Button(euler_frame, text="Stop Hold", width=12,
          command=clear_euler).grid(row=1, column=1, pady=5)

tk.Button(movement_frame, text="Forward", width=10,
          command=lambda: move_action_btn(x=.2)).grid(row=0, column=1, pady=2)
tk.Button(movement_frame, text="Left", width=10,
          command=lambda: move_action_btn(y=.2)).grid(row=1, column=0, padx=2)
tk.Button(movement_frame, text="Stop", width=10,
          command=lambda: move_action_btn(0, 0, 0)).grid(row=1, column=1, pady=2)
tk.Button(movement_frame, text="Right", width=10,
          command=lambda: move_action_btn(y=-.2)).grid(row=1, column=2, padx=2)
tk.Button(movement_frame, text="Backward", width=10,
          command=lambda: move_action_btn(x=-.2)).grid(row=2, column=1, pady=2)

tk.Button(movement_frame, text="Rotate Left", width=12,
          command=lambda: move_action_btn(z=.5)).grid(row=3, column=0, pady=5)
tk.Button(movement_frame, text="Rotate Right", width=12,
          command=lambda: move_action_btn(z=-.5)).grid(row=3, column=2, pady=5)

body_height_var = tk.DoubleVar()  # allows floating-point values
slider = tk.Scale(
    movement_frame,
    from_=-2.0,
    to=2.0,
    resolution=0.01,      # step size
    orient=tk.HORIZONTAL,
    length=150,
    label="Body Height",
    variable=body_height_var
)
slider.grid(row=4, column=0, columnspan=3, pady=5)


asyncio_thread = threading.Thread(target=run_loop, args=(loop,), daemon=True)
asyncio_thread.start()



root.after(100, lambda: root.focus_force())
root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
