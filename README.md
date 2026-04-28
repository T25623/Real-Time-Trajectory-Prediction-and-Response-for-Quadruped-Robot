# Real-Time Trajectory Prediction and Response for Quadruped Robot

A system that enables the Unitree Go2 quadruped robot to detect moving objects, predict their trajectory, and respond in real time using a Raspberry Pi 5 and Hailo AI Hat. The robot uses a combination of camera-based object detection, lidar obstacle avoidance, and a Kalman filter to track and react to objects such as balloons, or avoid moving hazards.

---

## Setup

Requires a Raspberry Pi 5 with a Hailo AI Hat.

```bash
git clone https://github.com/T25623/Real-Time-Trajectory-Prediction-and-Response-for-Quadruped-Robot.git
cd Real-Time-Trajectory-Prediction-and-Response-for-Quadruped-Robot/hailo-apps/
```

Follow the setup guide in [hailo-apps/README.md](./hailo-apps/README.md).

Download and install the Hailo drivers from the [Hailo Developer Zone](https://hailo.ai/developer-zone/), then run the installer:

```bash
sudo ./install.sh
```

Once complete:

```bash
source venv_hailo_apps/bin/activate
cd ..
pip install -r requirements.txt
```

---

## Running

A connection to the Go2 robot is required.

**GUI app** — monitor and configure the system:
```bash
python -m examples.gui_app
```

**Headless mode** — robot autonomously chases and intercepts a balloon:
```bash
python -m examples.balloon_hit
```

**Button-triggered startup** — wire buttons to the Pi as shown in [wiring.png](./images/wiring.png), then:
```bash
sudo cp startup/gpio_startup.service /etc/systemd/system/
sudo systemctl enable gpio_startup.service
sudo systemctl start gpio_startup.service
```

This runs `balloon_hit` automatically on boot based on button state.

---

## Adding Models

Place files in the following locations and the GUI will pick them up automatically:

- Model: `./config/models/<name>.hef`
- Config: `./config/config/<name>.json`
- Labels: `./config/labels/<name>.txt`

---

## Training Models

The Hailo AI Hat requires models in `.hef` format. Follow [this guide](https://towardsai.net/p/artificial-intelligence/custom-dataset-with-hailo-ai-hat-yolo-raspberry-pi-5-and-docker) to train and convert custom YOLO models.

---

## 3D Printing

Print the following parts at 100% infill to maximise strength. Settings for other parts are not critical.

- [pi5_case.stl](./3d_models/pi5_case.stl)
- [din_stop.stl](./3d_models/din_stop.stl)
- [mounting_rail.stl](./3d_models/mounting_rail.stl)
- [din_clamp.stl](./3d_models/din_clamp.stl)