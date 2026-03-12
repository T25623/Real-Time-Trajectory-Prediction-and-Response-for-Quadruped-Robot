from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD 
import asyncio
import re
from enum import Enum
import queue
import numpy as np
import go2_utils as utils


class LidarDecoder(Enum):
    Native = 0
    Libvoxel = 1


class WebRTCConnection:
    def __init__(self):
        self.conn = None
        self.lidar_queue = None
        self.state_of_charge = None
        self.orientation = None
        self.motor_temperature_list = None
        self.lidar_origin = None
    
    async def connection_setup(self, connection_method=WebRTCConnectionMethod.LocalAP,  ip=None, serial_number=None, username=None, password=None):
        match connection_method:
            case WebRTCConnectionMethod.LocalAP:
                self.conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)
            case WebRTCConnectionMethod.LocalSTA:
                if utils.valid_serial_number(serial_number):
                    self.conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber=serial_number)
                elif utils.valid_ip(ip):
                    self.conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ip)
            case WebRTCConnectionMethod.Remote:
                self.conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.Remote, serialNumber=serial_number, username=username, password=password)

        await self.conn.connect()
        await self.conn.datachannel.disableTrafficSaving(True)
        # await motion_switcher(self.conn)

    def lidar_sensor_activate(self, lidar_on=True):
        if lidar_on:
            lidar_on = "on"
        else:
            lidar_on = "off"

        self.conn.datachannel.pub_sub.publish_without_callback("rt/utlidar/switch", "on")

    def lidar_data_type(self, decoder_type=0):
        if decoder_type == 0:
            decoder_type = "native"
        else:
            decoder_type = "libvoxel"

        self.conn.datachannel.set_decoder(decoder_type=decoder_type)
    
    def lidar_setup(self, lidar_on, decoder_type):
        self.lidar_data_type(decoder_type)
        self.lidar_sensor_activate(lidar_on)
        
        self.conn.datachannel.pub_sub.subscribe(
           "rt/utlidar/voxel_map_compressed", self.lidar_callback
        )
    
    def state_call(self):
        self.conn.datachannel.pub_sub.subscribe(
           "rt/lf/lowstate", self.state_callback
        )

    def state_callback(self, message):
        self.orientation = utils.orientation_calculation(message)
        self.state_of_charge = utils.battery_state_of_charge_data(message)
        self.motor_temperature_list = utils.motor_temperature_data(message)

    def lidar_callback(self, message):
        self.lidar_queue = message 
        self.lidar_origin = utils.lidar_origin_calculation(message)
        print("got lidar data")


        
        