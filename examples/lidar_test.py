# from framework.robot.go2.setup import WebRTCConnection
# import framework.robot.go2.setup as go2
# import framework.robot.go2.lidar as dl
# import asyncio
# import numpy as np
# import pyvista as pv

# async def main():
#     dog = WebRTCConnection()
#     await dog.connection_setup()
#     dog.lidar_setup(True, 0)
#     print("connected")
#     plotter = pv.Plotter(off_screen=False)
#     plotter.show(interactive_update=True)

#     while True:
#         await asyncio.sleep(0.1)
#         message = dog.lidar_queue
#         if not message == None:
#             state = dog.state_call()
#             points = np.array(message["data"]["data"]["points"])
#             dl.run(plotter, points, dog.lidar_origin, 0.5, dog.orientation)

# asyncio.run(main())


import asyncio
import numpy as np
import pyvista as pv

from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod


lidar_points = None


def lidar_callback(message):
    global lidar_points

    try:
        pts = message["data"]["data"]["points"]
        lidar_points = np.array(pts)

    except Exception as e:
        print("lidar decode error:", e)


async def main():

    # connect to robot
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

    print("Connecting...")
    await conn.connect()
    print("Connected")

    await conn.datachannel.disableTrafficSaving(True)

    # enable lidar
    conn.datachannel.pub_sub.publish_without_callback(
        "rt/utlidar/switch", "on"
    )

    # choose decoder
    conn.datachannel.set_decoder(decoder_type="native")

    # subscribe to lidar topic
    conn.datachannel.pub_sub.subscribe(
        "rt/utlidar/voxel_map_compressed",
        lidar_callback
    )

    # setup plotter
    plotter = pv.Plotter()
    plotter.show(interactive_update=True)

    cloud = None

    while True:

        if lidar_points is not None:

            plotter.clear()

            plotter.add_points(
                lidar_points,
                color="black",
                point_size=5,
                render_points_as_spheres=True
            )

            plotter.update()

        await asyncio.sleep(0.1)


asyncio.run(main())