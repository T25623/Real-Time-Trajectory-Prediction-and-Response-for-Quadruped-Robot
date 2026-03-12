import numpy as np

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