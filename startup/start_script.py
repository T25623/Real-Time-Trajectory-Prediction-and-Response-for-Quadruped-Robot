import RPi.GPIO as GPIO
import subprocess
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setup(16, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(22, GPIO.OUT)

venv_path = "/home/go2/Real-Time-Trajectory-Prediction-and-Response-for-Quadruped-Robot/hailo-apps/venv_hailo_apps/bin/python"
script_path = "/home/go2/Real-Time-Trajectory-Prediction-and-Response-for-Quadruped-Robot/examples/balloon_hit.py"


process = None
try:
    while True:
        print(f"Pin 16 {GPIO.input(16)}")
        print(f"Pin 18 {GPIO.input(18)}")
        print(f"Pin 22 {GPIO.input(22)}")

        if not GPIO.input(16) and process == None:
            process = subprocess.Popen([venv_path, script_path])
            time.sleep(0.5)

        if not GPIO.input(18) and process != None:
            process.kill()
            process = None
            time.sleep(0.5)
        
        if process != None:
            GPIO.output(22, GPIO.HIGH)
        else:
            GPIO.output(22, GPIO.LOW)
        
        time.sleep(0.2)
finally:
    GPIO.cleanup()
    