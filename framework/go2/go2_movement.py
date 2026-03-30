import framework.utils.go2_utils as utils
import asyncio
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD

async def go2_movement(conn, x, y, z):
    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        # X = Forwards/Backwards, Y = Left/Right, Z = Rotation
        {"api_id": SPORT_CMD["Move"], "parameter": {"x": x, "y": y, "z": z}}
    )


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

