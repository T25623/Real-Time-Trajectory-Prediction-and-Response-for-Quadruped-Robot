from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD 
import asyncio
import re

async def connection_setup(connection_method=WebRTCConnectionMethod.LocalAP, ip=None, serial_number=None, username=None, password=None):
    conn = None

    match connection_method:
        case WebRTCConnectionMethod.LocalAP:
            conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
        case WebRTCConnectionMethod.LocalSTA:
            if valid_serial_number(serial_number):
                conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber=serial_number)
            elif valid_ip(ip):
                conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ip)
        case WebRTCConnectionMethod.Remote:
            conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.Remote, serialNumber=serial_number, username=username, password=password)

    await conn.connect()
    await motion_switcher(conn)


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
    
 

def valid_serial_number(serial_number):
    valid = False
    if "B42D2000" in serial_number:
        print("Valid Serial Number")
        valid = True
    elif serial_number == None:
        continue
    else:
        print("Invalid Serial Number")
    
    return valid

def valid_ip(ip):
    valid = False
    if ip != None:
        valid = True
    else:
        print("Invalid IP")
        