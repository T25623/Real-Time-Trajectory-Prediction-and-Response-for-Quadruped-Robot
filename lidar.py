import asyncio
import logging
import sys
import numpy as np
import pandas as pd
import open3d as o3d
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
import threading
import time

write = True

latest_points = None
points_lock = threading.Lock()
mask_state = 0
time_tracker = 0
latest_center = None


def lidar_callback(message):
    global latest_points, mask_state, latest_center
    #print(message)

    try:
        points = np.array(message["data"]["data"]["points"])
        origin = np.array(message["data"]["origin"])
        width = np.array(message["data"]["width"])
        resolution = np.array(message["data"]["resolution"])

        center_x = origin[0] + width[0]/2 * resolution[0]
        center_y = origin[1] + width[1]/2 * resolution[1]
        center_z = origin[2] + width[2]/2 * resolution[2]


        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        mask = 0
        mask_state = 6
        #center = points.mean(axis=0)
        origin_x = origin[0] 
        origin_y = origin[1]
        origin_z = origin[2]

        
        max_x = np.max(x)
        max_y = np.max(y)
        max_z = np.max(z)

        # print(f"max x: {max_x}")
        # print(f"max y: {max_y}")
        # print(f"max z: {max_z}")

        min_x = np.min(x)
        min_y = np.min(y)
        min_z = np.min(z)

        # print(f"min x: {min_x}")
        # print(f"min y: {min_y}")
        # print(f"min z: {min_z}")

        # center_x = (max_x - min_x) / 2
        # center_y = (max_y - min_y) / 2
        # center_z = (max_z - min_z) / 2

        min_distance = 100
        # for point in points:
        #     i = (point[0] - origin_x) / 0.05
        #     j = (point[1] - origin_y) / 0.05
        #     k = (point[2] - origin_z) / 0.05

        #     if i == 64 and j == 64:
        #         center_x = point[0]
        #         center_y = point[1]
        #         center_z = 0.2


        

        print(f"center x: {center_x}")
        print(f"center y: {center_y}")
        print(f"center z: {center_z}")
        


        if mask_state == 0:
            mask = (np.abs(x+center_x * 0.81) + y+center_y) <= 0
        elif mask_state == 1:
            mask = (np.abs(x+center_x * 0.81) - y+center_y) <= 0
        elif mask_state == 2:
            mask = (np.abs(y+center_y * 0.81) + x+center_x) <= 0
        elif mask_state == 3:
            mask = (np.abs(y+center_y * 0.81) - x+center_x) <= 0
        elif mask_state == 4:
            mask = (x+center_x)**2 + (y+center_y)**2 <= 10
        elif mask_state == 5:
            mask = (x+center_x)**2 + (y+center_y)**2 >= 2
        elif mask_state == 6:
            mask = np.abs(z-1)<=1
        else:
            mask = np.ones(len(points), dtype=bool)

        
        

        filtered_points = points[mask]
        
        for point in filtered_points:
            distance = np.linalg.norm(point - np.array([center_x, center_y, center_z]))
            if distance <= min_distance:
                min_distance = distance
        
        print(f"min distance: {min_distance}")

        with points_lock:
            latest_points = filtered_points
            latest_center = np.array([center_x, center_y, center_z])


    except Exception as e:
        logging.error(f"LiDAR callback error: {e}")

def visualizer_loop():
    global latest_points, mask_state, time_tracker, latest_center

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="LiDAR Point Cloud",
        width=1000,
        height=800
    )

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    center_sphere.paint_uniform_color([1, 0, 0])  
    vis.add_geometry(center_sphere)

    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([0, 0, 0])

    first_frame = True

    while True:
        with points_lock:
            if latest_points is not None:
                pcd.points = o3d.utility.Vector3dVector(latest_points)
                vis.update_geometry(pcd)

                if first_frame:
                    vis.reset_view_point(True)
                    first_frame = False

                if latest_center is not None:
                    center_sphere.translate(-center_sphere.get_center(), relative=True)
                    center_sphere.translate(latest_center, relative=True)
                    vis.update_geometry(center_sphere)

                    if first_frame:
                        vis.reset_view_point(True)
                        first_frame = False
        
        if time.time() >= time_tracker+5:
            print(f"State {mask_state}")
            time_tracker = time.time()
            mask_state += 1
            mask_state = mask_state % 4

        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.03)

async def main():
    try:
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
        await conn.connect()

        await conn.datachannel.disableTrafficSaving(True)
        conn.datachannel.set_decoder(decoder_type="native")

        conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")

        conn.datachannel.pub_sub.subscribe(
            "rt/utlidar/voxel_map_compressed", lidar_callback
        )

        while True:
            await asyncio.sleep(1)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    try:
        vis_thread = threading.Thread(target=visualizer_loop, daemon=True)
        vis_thread.start()

        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(0)
