import numpy as np
from hailo_platform import Device
import subprocess
import os
import psutil

def parse_response(response):
    code = response['data']['header']['status']['code']

    return code

def orientation_calculation(state):
    yaw = state["data"]["imu_state"]["rpy"][2]
    orientation = np.array([np.cos(yaw), np.sin(yaw), 0.0])

    return orientation

def battery_state_of_charge_data(state):
    bms = state["data"]["bms_state"]

    return bms["soc"]

def motor_temperature_data(state):
    motor_state = state["data"]["motor_state"]
    motor_temperature_list = []
    for motor in motor_state:
        motor_temperature_list.append(motor["temperature"])

    return motor_temperature_list

def lidar_origin_calculation(message):
    origin = np.array(message["data"]["origin"], dtype=float)
    width = np.array(message["data"]["width"], dtype=float)
    resolution = np.array(message["data"]["resolution"], dtype=float)
    center = origin + (width * resolution) / 2
    return center

def valid_serial_number(serial_number):
    valid = False
    if "B42D2000" in serial_number:
        print("Valid Serial Number")
        valid = True
    else:
        print("Invalid Serial Number")
    
    return valid

def valid_ip(ip):
    valid = False
    if ip != None:
        valid = True
    else:
        print("Invalid IP")

def calculate_facing_correction(orientation):
    target = np.array([0, 1, 0])
    cross = np.cross(orientation, target)
    dot = np.dot(orientation, target)
    angle = np.arctan2(np.linalg.norm(cross), dot)  
    sign = np.sign(cross[2])
    return sign * angle     

def rotate_vector(orientation, angle):
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    R = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    return R @ orientation

target = Device()

def npu_temp():
    return round(target.control.get_chip_temperature().ts0_temperature, 2)

def npu_load():
    return 0

def pi_battery_soc():
    result = subprocess.run("vcgencmd pmic_read_adc | grep EXT5V_V", shell=True, capture_output=True, text=True)

    voltage = result.stdout.strip().split("=")[-1].replace("V", "")
    voltage = float(voltage)
    return round(voltage, 2)

def cpu_temp():
    result = subprocess.run("vcgencmd measure_temp", shell=True, capture_output=True, text=True)

    temp = result.stdout.strip().split("=")[-1].replace("'C", "")
    temp = float(temp)
    return round(temp, 2)

def cpu_load():
    return psutil.cpu_percent(interval=None)


def robot_velocity(message):
    return message['data']["velocity"]
    
def robot_gyroscope(message):
    return message['data']['imu_state']['gyroscope']

def robot_temperature(message):
    return message['data']['imu_state']['temperature']

def sportmode_state_print(message):
    print(message)
    
    # quaternion = imu_state['quaternion']
    # accelerometer = imu_state['accelerometer']
    # rpy = imu_state['rpy']
    # temperature = imu_state['temperature']

    # mode = message['mode']
    # progress = message['progress']
    # gait_type = message['gait_type']
    # foot_raise_height = message['foot_raise_height']
    # position = message['position']
    # body_height = message['body_height']
    # velocity = message['velocity']