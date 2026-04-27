import framework.utils.utils as utils
import asyncio
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
import time
from framework.utils.utils import Objective


async def go2_movement(conn, x, y, z):
    # await enable_movement(conn)
    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        # X = Forwards/Backwards, Y = Left/Right, Z = Rotation
        {"api_id": SPORT_CMD["Move"], "parameter": {"x": x, "y": y, "z": z}}
    )

async def enable_movement(conn):
    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["MOTION_SWITCHER"], 
        {"api_id": 1001}
    )
    print(response)

    if response['data']['header']['status']['code'] == 0:
        data = json.loads(response['data']['data'])
        current_motion_switcher_mode = data['name']
    
    print(f"Current motion mode: {current_motion_switcher_mode}")

    # Switch to "normal" mode if not already
    if current_motion_switcher_mode != "normal":
        print(f"Switching motion mode from {current_motion_switcher_mode} to 'normal'...")
        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"], 
            {
                "api_id": 1002,
                "parameter": {"name": "normal"}
            }
        )
        await asyncio.sleep(2)


async def go2_euler(conn, x, y, z):
    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        # X = Role, Y = Pitch, Z = Yaw
        {"api_id": SPORT_CMD["Euler"], "parameter": {"x": x, "y": y, "z": z}}
    )

async def getState(conn):
    response = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"], 
            {"api_id": 1001}
        )
    print(response)
    

async def perform_action(conn, action):
    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD[action], "parameter": {"data": True}}
    )
    if action == "FrontPounce":
        response = await go2_movement(conn, -0.4, 0, 0)
    
    return response

async def move_pitch(conn, move_x, move_y, move_z, pitch_x, pitch_y, pitch_z):
    move_task = asyncio.create_task(go2_movement(conn, move_x, move_y, move_z))
    pitch_task = asyncio.create_task(go2_euler(conn, pitch_x, pitch_y, pitch_z))

    await asyncio.gather(move_task, pitch_task)

async def system_error_reposnse(conn, status_list):
    bms_data = status_list["data"]["bms"]
    battery_percent = bms_data["SOC"]
    temperature = bms_data["temperature"]

    if battery_percent <= 5:
        # Put robot into a safe position
        while True:
            sit_response = await perform_action(conn, "StandDown")

            if utils.parse_response(sit_response) == 0:
                break

            await asyncio.sleep(5)
    
def calculate_movement(distance_to_object, deadzone, target_distance, scale_factor):
    x = 0.0
    # Prevent extremely fast movement
    if distance_to_object < 3:
        # move backward
        if distance_to_object < target_distance - deadzone:
            x = -distance_to_object
        # move forward
        elif distance_to_object > target_distance + deadzone:
            x = distance_to_object
    # 0.25 seems to work well
    return x * scale_factor


def calculate_rotation(offset_ratio, deadzone, scale_factor, max_step=0.5):
    # Shifting center to 0 from 0.5
    offcenter_ratio = offset_ratio - 0.5

    if abs(offcenter_ratio) <= deadzone:
        offcenter_ratio = 0.0

    # 2 seems to work well
    rotate = -offcenter_ratio * scale_factor  
    if rotate > max_step:
        rotate = max_step
    elif rotate < -max_step:
        rotate = -max_step
    
    return rotate

def calculate_pitch(offset_ratio, deadzone, scale_factor, current_pitch, max_pitch=0.05):
    MAX_ABS_PITCH = 0.4
    # Shifting center to 0 from 0.5
    offcenter_ratio = offset_ratio - 0.5

    if abs(offcenter_ratio) <= deadzone:
        return current_pitch

    # amount to pitch
    temp_pitch_position = offcenter_ratio * scale_factor
    
    if temp_pitch_position > max_pitch:
        temp_pitch_position = max_pitch
    elif temp_pitch_position < -max_pitch:
        temp_pitch_position = -max_pitch

    # future pitch position
    pitch_position = current_pitch + temp_pitch_position
    # Check if pitch is within max pitch limit
    if abs(pitch_position) < MAX_ABS_PITCH:
        pitch_position = pitch_position
    else:
        if pitch_position > 0:
            pitch_position = MAX_ABS_PITCH
        else:
            pitch_position = -MAX_ABS_PITCH
    
    return pitch_position

async def avoid_obstacle(robot):
    facing_correction = utils.angle_between_vectors(robot.orientation)
    vector = utils.rotate_vector(robot.avoid_vector, facing_correction)
            
    await go2_movement(robot.conn, (vector[1]*0.1), (-vector[0]*0.1), 0)

async def avoid_response(robot, detected, future_distance):
    global st
    if robot.avoid_vector is not None:
        await avoid_obstacle(robot)

    elif (time.time() - st) >= 5 and detected:
        if future_distance is not None and future_distance >= -0.1 and future_distance <= 0.3: 
            await move_pitch(robot.conn, -0.5, 0, 0, 0, 0, 0)
            


pitch = 0
st = time.time()
async def movement_response(robot, no_detection_time, detected, future_distance, future_center_x, future_center_y, track, response_action):
    global st, pitch
    if robot.avoid_vector is not None and robot.orientation is not None:
        await avoid_obstacle(robot)
        
    elif (time.time() - no_detection_time) >= 5 and not detected and (time.time() - st) >= 3 and track == Objective.Track_Hit:
        st = time.time()
        await go2_movement(robot.conn, 0, 0, 2)
            
    elif detected:
        no_detection_time = time.time()
        if future_distance is not None and future_distance >= -0.1 and future_distance <= 0.3:
            if track == Objective.Track_Hit:
                await asyncio.sleep(1.5)
            await perform_action(robot.conn, response_action)
            pitch = 0
                
        elif track == Objective.Track_Hit and future_distance is not None and future_center_x is not None and future_center_y is not None:
            forward = calculate_movement(future_distance, 0.1, 0.2, robot.movement_speed)
            rotate = calculate_rotation(future_center_x, 0.1, robot.rotate_speed)
                   
            pitch = calculate_pitch(future_center_y, 0.1, robot.pitch_speed, pitch)
            await move_pitch(robot.conn, forward, 0, rotate, 0, pitch, 0)
    
    

