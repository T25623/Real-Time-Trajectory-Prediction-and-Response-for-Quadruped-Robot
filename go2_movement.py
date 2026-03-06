
async def go2_movement(conn, x, y, z):
    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD["Move"], "parameter": {"x": x, "y": y, "z": z}}
    )


async def go2_euler(conn, x, y, z):
    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD["Euler"], "parameter": {"x": x, "y": y, "z": z}}
    )
    

async def perform_action(conn, action):
    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {"api_id": SPORT_CMD[action], "parameter": {"data": True}}
    )

    

def calculate_x(distance_to_object, deadzone, target_distance, scale_factor):
    x = 0.0
    if min_distance < 2:
        if min_distance < target_distance - deadzone:
            x = -min_distance
        elif min_distance > target_distance + deadzone:
            x = min_distance

    return x / 4
