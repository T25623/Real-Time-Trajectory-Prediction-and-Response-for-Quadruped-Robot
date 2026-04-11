from framework.robot.go2.setup import WebRTCConnection
import framework.robot.go2.setupas go2
import framework.robot.go2.lidar as dl
import asyncio
import numpy as np
import pyvista as pv

async def main():
    dog = WebRTCConnection()
    await dog.connection_setup()
    dog.lidar_setup(True, 0)
    print("connected")
    plotter = pv.Plotter(off_screen=False)
    plotter.show(interactive_update=True)

    while True:
        await asyncio.sleep(0.1)
        message = dog.lidar_queue
        if not message == None:
            state = dog.state_call()
            points = np.array(message["data"]["data"]["points"])
            dl.run(plotter, points, dog.lidar_origin, 0.5, dog.orientation)

asyncio.run(main())