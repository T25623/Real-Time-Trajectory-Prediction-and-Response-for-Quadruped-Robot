from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
import asyncio

async def motion_switcher(conn, timeout=10):

    while True:
        response = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"], 
            {"api_id": 1001}
        )

        if response['data']['header']['status']['code'] == 0:
            data = json.loads(response['data']['data'])
            
            if data['name'] == "normal":
                break
            else:
                await conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["MOTION_SWITCHER"], 
                    {
                        "api_id": 1002,
                        "parameter": {"name": "normal"}
                    }
                )
                await asyncio.sleep(5)



