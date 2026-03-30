import RPi.GPIO as GPIO
import subprocess
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(13, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(15, GPIO.OUT)

venv_path = "/home/go2/FYP/Real-Time-Trajectory-Prediction-and-Response-for-Quadruped-Robot/hailo-apps/venv_hailo_apps/bin/python"
script_path = "/home/go2/FYP/Real-Time-Trajectory-Prediction-and-Response-for-Quadruped-Robot/"


process = None
try:
    while True:
        print(f"Pin 11 {GPIO.input(11)}")
        print(f"Pin 13 {GPIO.input(13)}")
        print(f"Pin 15 {GPIO.input(15)}")

        if not GPIO.input(11) and process == None:
            process = subprocess.Popen([venv_path, script_path])
            time.sleep(0.5)

        if not GPIO.input(13) and process != None:
            process.kill()
            process = None
            time.sleep(0.5)
        
        if process != None:
            GPIO.output(15, GPIO.HIGH)
        else:
            GPIO.output(15, GPIO.LOW)
        
        time.sleep(0.2)
finally:
    GPIO.cleanup()
    