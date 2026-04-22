from framework.robot.go2.setup import WebRTCConnection
import framework.robot.go2.setup as go2
from framework.utils.utils import Objective
import framework.robot.go2.lidar as go2_lidar
from framework.detection.detection import DetectionPipeline
import asyncio
import framework.robot.go2.movement as move
import threading
import numpy as np
import framework.utils.utils as utils
from tkinter import *
from tkinter.ttk import Combobox, Notebook, Style
from PIL import Image, ImageTk
import time
import cv2
from collections import deque
from pathlib import Path

# Colours & Fonts
BG = "#0f1117"
BG2 = "#181c27"
BG3 = "#1f2535"   
BORDER = "#2a3047"
ACCENT = "#00b4d8"
ACCENT_DIM = "#0077a0"
DANGER = "#e63946"
DANGER_DIM = "#9e1c27"
SUCCESS = "#2dc653"
SUCCESS_DIM = "#1a7a33"
FG = "#d0d6e8"
FG_DIM = "#6b738f"
FG_HEAD = "#ffffff"
MONO = ("Courier", 9)
FONT_HEAD = ("Helvetica", 9, "bold")
FONT_LABEL = ("Helvetica", 9)
FONT_TITLE = ("Helvetica", 13, "bold")

# Root Window 
window = Tk()
window.configure(bg=BG)
screen_width  = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
window.geometry(f"{screen_width}x{screen_height}")
window.title("PARS — Prediction and Response System")

# Tkinter Items Style
style = Style()
style.theme_use("clam")

style.configure("TNotebook",background=BG2, borderwidth=0, tabmargins=[2, 2, 0, 0])

style.configure("TNotebook.Tab",background=BG3, foreground=FG_DIM, font=FONT_HEAD, padding=[10, 4], borderwidth=0)
style.map("TNotebook.Tab",background=[("selected", BG), ("active", BG3)], foreground=[("selected", ACCENT), ("active", FG)])

style.configure("TCombobox", fieldbackground=BG3, background=BG3, foreground=FG, selectbackground=ACCENT, selectforeground=BG, arrowcolor=ACCENT, borderwidth=1, relief="flat")
style.map("TCombobox", fieldbackground=[("readonly", BG3)], foreground=[("readonly", FG)], background=[("readonly", BG3)])

# Tkinter Item Helpers
def panel(parent, **kw):
    f = Frame(parent, bg=BG2, highlightbackground=BORDER, highlightthickness=1, **kw)

    return f

def section_label(parent, text):
    outer = Frame(parent, bg=BG2)

    bar = Frame(outer, bg=ACCENT, width=3)
    bar.pack(side=LEFT, fill=Y, padx=(0, 6))

    lbl = Label(outer, text=text.upper(), font=FONT_HEAD, bg=BG2, fg=ACCENT, anchor=W)
    lbl.pack(side=LEFT)

    return outer

def info_label(parent, text, **kw):
    return Label(parent, text=text, font=MONO, bg=BG2, fg=FG, anchor=W, **kw)

def dim_label(parent, text, **kw):
    return Label(parent, text=text, font=FONT_LABEL, bg=BG2, fg=FG_DIM, anchor=W, **kw)

def make_button(parent, text, color=ACCENT, hover=ACCENT_DIM, fg=BG, width=10, **kw):
    btn = Button(parent, text=text, font=FONT_HEAD, bg=color, fg=fg, activebackground=hover, activeforeground=fg, relief="flat", bd=0, cursor="hand2", padx=8, pady=4, width=width, **kw)
    
    btn.bind("<Enter>", lambda e: btn.config(bg=hover))
    btn.bind("<Leave>", lambda e: btn.config(bg=color))
    
    return btn

def make_scale(parent, **kw):
    return Scale(parent, font=FONT_LABEL, bg=BG2, fg=FG, troughcolor=BG3, activebackground=ACCENT, highlightthickness=0, orient=HORIZONTAL, sliderrelief="flat", bd=0, **kw)

def make_entry(parent, **kw):
    return Entry(parent, font=FONT_LABEL, bg=BG3, fg=FG, insertbackground=ACCENT, relief="flat", bd=4, **kw)

def make_check(parent, text, var):
    return Checkbutton(parent, text=text, variable=var, font=FONT_LABEL, bg=BG2, fg=FG, selectcolor=BG3, activebackground=BG2, activeforeground=FG, highlightthickness=0)

def separator(parent, color=BORDER):
    return Frame(parent, bg=color, height=1)

def file_map(extension, path):
    folder = Path(path)
    mapping = {}

    for file in folder.iterdir():
        if file.suffix == f".{extension}":
            mapping[file.stem] = str(file)

    return mapping

# Title Bar
title_bar = Frame(window, bg=BG3, highlightbackground=BORDER, highlightthickness=1)
title_bar.grid(column=0, row=0, columnspan=3, sticky=EW, padx=0, pady=0)
window.grid_columnconfigure(1, weight=1)

Label(title_bar, text="PARS", font=("Helvetica", 15, "bold"), bg=BG3, fg=ACCENT).pack(side=LEFT, padx=(14, 4), pady=8)
Label(title_bar, text="Prediction and Response System", font=("Helvetica", 10), bg=BG3, fg=FG_DIM).pack(side=LEFT, pady=8)

connection_status_label = Label(title_bar, text="● NO CONNECTION", font=FONT_HEAD, bg=BG3, fg=DANGER)
connection_status_label.pack(side=RIGHT, padx=14, pady=8)

# Column Layout
left_container = Frame(window, bg=BG)
left_container.grid(column=0, row=1, padx=(8,4), pady=8, sticky=N)

center_container = Frame(window, bg=BG)
center_container.grid(column=1, row=1, padx=4, pady=8, sticky=N)

right_container = Frame(window, bg=BG)
right_container.grid(column=2, row=1, padx=(4,8), pady=8, sticky=N)

# LEFT COLUMN

# Robot Status 
robot_status_container = panel(left_container)
robot_status_container.grid(column=0, row=0, padx=0, pady=(0,6), sticky=EW)

section_label(robot_status_container, "Robot Status").pack(fill=X, padx=8, pady=(8,6))
separator(robot_status_container).pack(fill=X, padx=8, pady=(0,6))

robot_battery_label = info_label(robot_status_container, "Robot Battery : N/A")
robot_speed_label = info_label(robot_status_container, "Roll          : N/A")
robot_temp_label = info_label(robot_status_container, "Pitch         : N/A")
robot_motor_temperature_label = info_label(robot_status_container, "Yaw           : N/A")

pi_label = section_label(robot_status_container, "Pi Status")
pi_separator = separator(robot_status_container)

pi_battery_label = info_label(robot_status_container, "Pi Battery    : N/A")
cpu_temp_label = info_label(robot_status_container, "CPU Temp      : N/A")
cpu_load_label = info_label(robot_status_container, "CPU Load      : N/A")
npu_temp_label = info_label(robot_status_container, "NPU Temp      : N/A")
npu_load_label = info_label(robot_status_container, "NPU Load      : N/A")

for lbl in (robot_battery_label, robot_speed_label, robot_temp_label, robot_motor_temperature_label, pi_label, pi_separator, pi_battery_label, cpu_temp_label, cpu_load_label, npu_temp_label, npu_load_label):
    lbl.pack(fill=X, padx=12, pady=1)

Frame(robot_status_container, bg=BG2, height=8).pack()

# Response & Motion 
reponse_and_motion_container = panel(left_container)
reponse_and_motion_container.grid(column=0, row=1, padx=0, pady=(0,6), sticky=EW)

section_label(reponse_and_motion_container, "Response & Motion").pack(fill=X, padx=8, pady=(8,6))
separator(reponse_and_motion_container).pack(fill=X, padx=8, pady=(0,6))

objectives = ("Move & Hit", "Stand & Hit", "Move & Dodge", "Stand & Dodge")
objectives_combobox = Combobox(reponse_and_motion_container, values=objectives, state="readonly", font=FONT_LABEL)
objectives_combobox.set(objectives[0])
objectives_combobox.pack(fill=X, padx=12, pady=(0,8))

actions = ("FrontJump", "FrontPounce", "Hello")
actions_combobox = Combobox(reponse_and_motion_container, values=actions, state="readonly", font=FONT_LABEL)
actions_combobox.set(actions[2])
actions_combobox.pack(fill=X, padx=12, pady=(0,8))


move_speed_row = Frame(reponse_and_motion_container, bg=BG2)
move_speed_row.pack(fill=X, padx=12, pady=2)
dim_label(move_speed_row, "Movement Speed", width=14).pack(side=LEFT, anchor=W)
move_speed_slider = make_scale(move_speed_row, from_=0.0, to=50.0, resolution=0.1)
move_speed_slider.pack(side=LEFT, fill=X, expand=True)
move_speed_slider.set(0.25)

rotate_speed_row = Frame(reponse_and_motion_container, bg=BG2)
rotate_speed_row.pack(fill=X, padx=12, pady=2)
dim_label(rotate_speed_row, "Rotate Speed", width=14).pack(side=LEFT, anchor=W)
rotate_speed_slider = make_scale(rotate_speed_row, from_=0.0, to=50.0, resolution=0.1)
rotate_speed_slider.pack(side=LEFT, fill=X, expand=True)
rotate_speed_slider.set(2)

pitch_speed_row = Frame(reponse_and_motion_container, bg=BG2)
pitch_speed_row.pack(fill=X, padx=12, pady=2)
dim_label(pitch_speed_row, "Pitch Speed", width=14).pack(side=LEFT, anchor=W)
pitch_speed_slider = make_scale(pitch_speed_row, from_=0.0, to=5.0, resolution=0.1)
pitch_speed_slider.pack(side=LEFT, fill=X, expand=True)
pitch_speed_slider.set(0.5)

auto_run_button = make_button(reponse_and_motion_container, "Auto Run", color=SUCCESS, hover=SUCCESS_DIM, fg=BG, width=9)
auto_run_button.pack(side=LEFT, fill=X, expand=True)

auto_run_stop_button = make_button(reponse_and_motion_container, "Stop Auto Run", color=DANGER, hover=DANGER_DIM, fg=BG, width=9)
auto_run_stop_button.pack(side=LEFT, fill=X, expand=True)

Frame(reponse_and_motion_container, bg=BG2, height=8).pack()

# Detection Status 
detection_status_container = panel(left_container)
detection_status_container.grid(column=0, row=2, padx=0, pady=0, sticky=EW)

section_label(detection_status_container, "Detection").pack(fill=X, padx=8, pady=(8,6))
separator(detection_status_container).pack(fill=X, padx=8, pady=(0,6))

detection_status_label = info_label(detection_status_container, "Object Detected : N/A")
detection_confidence_label = info_label(detection_status_container, "Confidence      : N/A")
detection_distance_label = info_label(detection_status_container, "Detected Object Distance : N/A")
detection_predicted_distance_label = info_label(detection_status_container, "Detected Object Predicted Distance      : N/A")
detection_status_label.pack(fill=X, padx=12, pady=1)
detection_confidence_label.pack(fill=X, padx=12, pady=1)
detection_distance_label.pack(fill=X, padx=12, pady=1)
detection_predicted_distance_label.pack(fill=X, padx=12, pady=1)
Frame(detection_status_container, bg=BG2, height=8).pack()

# CENTER COLUMN

# Connection bar 
connection_setup_container = panel(center_container)
connection_setup_container.grid(column=0, row=0, padx=0, pady=(0,6), sticky=EW)

conn_inner = Frame(connection_setup_container, bg=BG2)
conn_inner.pack(fill=X, padx=8, pady=8)

dim_label(conn_inner, "Method").grid(row=0, column=0, padx=(0,4), pady=0, sticky=W)

connection_methods = ("LocalAP", "LocalSTA", "Remote")
connection_methods_combobox = Combobox(conn_inner, values=connection_methods, state="readonly", font=FONT_LABEL, width=10)
connection_methods_combobox.grid(row=0, column=1, padx=(0,8))

# Dynamic fields
ip_label = dim_label(conn_inner, "Robot IP")
ip_input = make_entry(conn_inner, width=14)
serial_number_label = dim_label(conn_inner, "Serial #")
serial_number_input = make_entry(conn_inner, width=14)
username_label = dim_label(conn_inner, "Username")
username_input = make_entry(conn_inner, width=12)
password_label = dim_label(conn_inner, "Password")
password_input = make_entry(conn_inner, width=12, show="*")

connection_button = make_button(conn_inner, "Connect", color=SUCCESS, hover=SUCCESS_DIM, fg=BG, width=9)
connection_button.grid(row=0, column=10, padx=(8,0))

def update_connection_fields(*args):
    for w in (ip_label, ip_input, serial_number_label, serial_number_input, username_label, username_input, password_label, password_input):
        w.grid_remove()

    method = connection_methods_combobox.get()

    if method == "LocalSTA":
        ip_label.grid(row=0, column=2, padx=(0,4))
        ip_input.grid(row=0, column=3, padx=(0,8))
        serial_number_label.grid(row=0, column=4, padx=(0,4))
        serial_number_input.grid(row=0, column=5, padx=(0,8))

    elif method == "Remote":
        serial_number_label.grid(row=0, column=2, padx=(0,4))
        serial_number_input.grid(row=0, column=3, padx=(0,8))
        username_label.grid(row=0, column=4, padx=(0,4))
        username_input.grid(row=0, column=5, padx=(0,8))
        password_label.grid(row=0, column=6, padx=(0,4))
        password_input.grid(row=0, column=7, padx=(0,8))

connection_methods_combobox.bind("<<ComboboxSelected>>", update_connection_fields)

# Video Feed
video_panel = panel(center_container)
video_panel.grid(column=0, row=1, padx=0, pady=(0,6), sticky=EW)

video = Label(video_panel, bg="#000000", text="No Signal", font=("Courier", 12), fg=FG_DIM)
video.pack(padx=2, pady=2)

# Manual Controls
robot_controls_container = panel(center_container)
robot_controls_container.grid(column=0, row=2, padx=0, pady=0, sticky=EW)

section_label(robot_controls_container, "Manual Controls").pack(fill=X, padx=8, pady=(8,6))
separator(robot_controls_container).pack(fill=X, padx=8, pady=(0,8))

ctrl_grid = Frame(robot_controls_container, bg=BG2)
ctrl_grid.pack(padx=12, pady=(0,8))

# D-pad + rotate group
rotate_left_button = make_button(ctrl_grid, "↺ Left", width=9)
forward_button = make_button(ctrl_grid, "▲ Fwd", width=9)
rotate_right_button = make_button(ctrl_grid, "↻ Right", width=9)
left_button = make_button(ctrl_grid, "◀ Left", width=9)
_center_spacer = Frame(ctrl_grid, bg=BG2, width=78, height=32)
right_button = make_button(ctrl_grid, "Right ▶", width=9)
_empty = Frame(ctrl_grid, bg=BG2, width=78, height=32)
backward_button = make_button(ctrl_grid, "▼ Back", width=9)

rotate_left_button.grid( row=0, column=0, padx=3, pady=3)
forward_button.grid(row=0, column=1, padx=3, pady=3)
rotate_right_button.grid(row=0, column=2, padx=3, pady=3)
left_button.grid(row=1, column=0, padx=3, pady=3)
_center_spacer.grid(row=1, column=1, padx=3, pady=3)
right_button.grid(row=1, column=2, padx=3, pady=3)
_empty.grid(row=2, column=0, padx=3, pady=3)
backward_button.grid(row=2, column=1, padx=3, pady=3)

# Separator between d-pad and actions
Frame(ctrl_grid, bg=BORDER, width=1).grid(row=0, column=3, rowspan=3, padx=(10,10), sticky=NS)

# Action buttons
sit_button = make_button(ctrl_grid, "Sit", width=8)
stand_button = make_button(ctrl_grid, "Stand", width=8)
hello_button = make_button(ctrl_grid, "Hello", width=8)
pounce_button = make_button(ctrl_grid, "Pounce", width=8)

sit_button.grid(row=0, column=4, padx=3, pady=3)
stand_button.grid(row=0, column=5, padx=3, pady=3)
hello_button.grid(row=1, column=4, padx=3, pady=3)
pounce_button.grid(row=1, column=5, padx=3, pady=3)

# RIGHT COLUMN

# Mode Override 
override_container = panel(right_container)
override_container.grid(column=0, row=0, padx=0, pady=(0,6), sticky=EW)

section_label(override_container, "Mode Override").pack(fill=X, padx=8, pady=(8,6))
separator(override_container).pack(fill=X, padx=8, pady=(0,8))

mode_row = Frame(override_container, bg=BG2)
mode_row.pack(padx=12, pady=(0,10))

manual_mode_button = make_button(mode_row, "Manual", color=ACCENT, hover=ACCENT_DIM, width=9)
auto_mode_button = make_button(mode_row, "Auto", color=SUCCESS, hover=SUCCESS_DIM, fg=BG, width=9)
power_off_mode_button = make_button(mode_row, "Power Off", color=DANGER, hover=DANGER_DIM, fg=FG_HEAD, width=9)

manual_mode_button.pack(side=LEFT, padx=(0,6))
auto_mode_button.pack(side=LEFT, padx=(0,6))
power_off_mode_button.pack(side=LEFT)

# Tab panel
tab_container = Frame(right_container, bg=BG)
tab_container.grid(column=0, row=1, padx=0, pady=0, sticky=EW)

tabs = Notebook(tab_container)
tabs.pack(fill=BOTH, expand=True)

setup_tab = Frame(tabs, bg=BG2)
tabs.add(setup_tab, text="  Robot Setup  ")

lidar_tab = Frame(tabs, bg=BG2)
tabs.add(lidar_tab, text="  Lidar  ")

log_tab = Frame(tabs, bg=BG2)
tabs.add(log_tab, text="  Error Log  ")

# Setup Tab
def tab_row(parent, row, label_text, widget):
    dim_label(parent, label_text).grid(row=row, column=0, padx=(12,6), pady=3, sticky=W)
    widget.grid(row=row, column=1, padx=(0,12), pady=3, sticky=EW)
    parent.grid_columnconfigure(1, weight=1)

# Camera section
section_label(setup_tab, "Camera").grid(row=0, column=0, columnspan=2, padx=8, pady=(10,4), sticky=W)
separator(setup_tab).grid(row=1, column=0, columnspan=2, padx=8, pady=(0,4), sticky=EW)

camera_source_combobox = Combobox(setup_tab, values=("rpi","usb"), state="readonly", font=FONT_LABEL)
camera_source_combobox.set("rpi")
resolution_combobox = Combobox(setup_tab, values=("(1920x1080)","(1280x720)","(640x360)","(320x180)"), state="readonly", font=FONT_LABEL)
resolution_combobox.set("(640x360)")
frame_rate_input = make_entry(setup_tab, width=10)
frame_rate_input.insert(0, "120")
run_detection_button = make_button(setup_tab, "Run Detection", width=22)

tab_row(setup_tab, 2, "Camera Source", camera_source_combobox)
tab_row(setup_tab, 3, "Resolution", resolution_combobox)
tab_row(setup_tab, 4, "Frame Rate", frame_rate_input)
tab_row(setup_tab, 5, "", run_detection_button)

# Detection model section
section_label(setup_tab, "Detection Model").grid(row=6, column=0, columnspan=2, padx=8, pady=(10,4), sticky=W)
separator(setup_tab).grid(row=7, column=0, columnspan=2, padx=8, pady=(0,4), sticky=EW)

model_paths_map = file_map("hef", "config/models")
config_paths_map = file_map("json", "config/json")
label_paths_map = file_map("txt", "config/labels")

models = tuple(model_paths_map.keys()) + ("None",)
configs = tuple(config_paths_map.keys()) + ("None",)
labels = tuple(label_paths_map.keys()) + ("None",)

model_path_combobox  = Combobox(setup_tab, values=models,  state="readonly", font=FONT_LABEL)
config_path_combobox = Combobox(setup_tab, values=configs, state="readonly", font=FONT_LABEL)
labels_path_combobox = Combobox(setup_tab, values=labels,  state="readonly", font=FONT_LABEL)
model_path_combobox.set(models[0])
config_path_combobox.set(configs[0])
labels_path_combobox.set(labels[0])

tab_row(setup_tab, 8,  "Model",  model_path_combobox)
tab_row(setup_tab, 9,  "Config", config_path_combobox)
tab_row(setup_tab, 10, "Labels", labels_path_combobox)


# Prediction section
section_label(setup_tab, "Prediction").grid(row=11, column=0, columnspan=2, padx=8, pady=(10,4), sticky=W)
separator(setup_tab).grid(row=12, column=0, columnspan=2, padx=8, pady=(0,4), sticky=EW)

prediction_steps_slider = make_scale(setup_tab, from_=1, to=100)
object_height_input = make_entry(setup_tab, width=10)
object_height_input.insert(0, "20")
distance_calibration_slider = make_scale(setup_tab, from_=1, to=100)

prediction_steps_slider.set(15)
distance_calibration_slider.set(15)

tab_row(setup_tab, 13, "Prediction Steps", prediction_steps_slider)
tab_row(setup_tab, 14, "Object Height", object_height_input)
tab_row(setup_tab, 15, "Calibrate Distance", distance_calibration_slider)

# Video section
section_label(setup_tab, "Video").grid(row=16, column=0, columnspan=2, padx=8, pady=(10,4), sticky=W)
separator(setup_tab).grid(row=17, column=0, columnspan=2, padx=8, pady=(0,4), sticky=EW)

trail_length_slider = make_scale(setup_tab, from_=1, to=100)
trail_length_slider.set(15)

show_fps_var = IntVar()
show_fps_counter_switch = make_check(setup_tab, "Show FPS Counter", show_fps_var)
show_fps_counter_switch.grid(row=18, column=0, columnspan=2, padx=12, pady=3, sticky=W)
tab_row(setup_tab, 19, "Trail Length", trail_length_slider)

Frame(setup_tab, bg=BG2, height=10).grid(row=20, column=0)

# Lidar Tab
section_label(lidar_tab, "Lidar").grid(row=0, column=0, columnspan=2, padx=8, pady=(10,4), sticky=W)
separator(lidar_tab).grid(row=1, column=0, columnspan=2, padx=8, pady=(0,4), sticky=EW)

boundary_var = IntVar()
lidar_on_button = make_button(lidar_tab, "Lidar On", color=ACCENT, hover=ACCENT_DIM, width=9)
lidar_off_button = make_button(lidar_tab, "Lidar Off", color=DANGER, hover=DANGER_DIM, width=9)
boundry_detection_switch = make_check(lidar_tab, "Boundary Detection On", boundary_var)
lidar_on_button.grid(row=2, column=0, columnspan=2, padx=12, pady=3, sticky=W)
lidar_off_button.grid(row=3, column=0, columnspan=2, padx=12, pady=3, sticky=W)
boundry_detection_switch.grid(row=4, column=0, columnspan=2, padx=12, pady=3, sticky=W)

refresh_row = Frame(lidar_tab, bg=BG2)
refresh_row.grid(row=5, column=0, columnspan=2, padx=12, pady=2, sticky=EW)
lidar_tab.grid_columnconfigure(0, weight=1)
dim_label(refresh_row, "Refresh Rate", width=14).pack(side=LEFT, anchor=W)
lidar_refresh_rate_slider = make_scale(refresh_row, from_=1, to=10)
lidar_refresh_rate_slider.set(1)
lidar_refresh_rate_slider.pack(side=LEFT, fill=X, expand=True)

lidar_panel = panel(lidar_tab)
lidar_panel.grid(row=6, column=0, columnspan=2, padx=8, pady=(4,6), sticky=EW)
lidar = Label(lidar_panel, bg="#000000", text="No Lidar", font=("Courier", 12), fg=FG_DIM)
lidar.pack(padx=2, pady=2)

Frame(lidar_tab, bg=BG2, height=10).grid(row=5, column=0)

# Log Tab
log_text = Text(log_tab, bg=BG3, fg=FG, font=MONO, relief="flat", bd=0, state="disabled", insertbackground=ACCENT)
log_text.pack(fill=BOTH, expand=True, padx=8, pady=8)

# Getters
def get_connection_method(): 
    return connection_methods_combobox.get()

def get_ip(): 
    return ip_input.get()

def get_serial_number(): 
    return serial_number_input.get()

def get_username(): 
    return username_input.get()

def get_password(): 
    return password_input.get()
    

def get_objective():
    if robot is not None and detection is not None:
        objective = objectives_combobox.get()
        
        track = True if "Move" in objective.split("&")[0] else False
        hit = True if "Hit" in objective.split("&")[1] else False
        
        if hit and track:
            robot.objective = Objective.Track_Hit
        elif hit and not track:
            robot.objective = Objective.Stand_Hit
        elif not hit and track:
            robot.objective = Objective.Move_Dodge
        elif not hit and not track:
            robot.objective = Objective.Stand_Dodge

def get_action():
    if robot is not None and detection is not None:
        return actions_combobox.get()
    else:
        return "Hello"

        
def get_move_speed():
    if robot is not None:
        robot.movement_speed = move_speed_slider.get()

def get_rotate_speed():
    if robot is not None:
        robot.rotate_speed = rotate_speed_slider.get()

def get_pitch_speed():
    if robot is not None:
        robot.pitch_speed = pitch_speed_slider.get()

def get_camera_source():
    source = camera_source_combobox.get()
    if detection is not None and source != "":
        detection.source = source
            

def get_resolution():
    res = resolution_combobox.get()
    if res != "" and detection is not None:
        result = res.replace("(", "").replace(")", "").split("x")
        resolution = (int(result[0]), int(result[1]))
        detection.resolution = resolution

def get_frame_rate():
    frame_rate = frame_rate_input.get()
    if frame_rate != "" and detection is not None:
        frame_rate = int(frame_rate)
        detection.framerate = frame_rate

def get_prediction_steps():
    if detection is not None:
        detection.predict_steps = int(prediction_steps_slider.get())

def get_object_height():
    height = object_height_input.get()
    if detection is not None and height != "":
        detection.real_object_height = float(height)

def get_distance_calibration():
    if detection is not None:
        detection.distance_scale_factor = int(distance_calibration_slider.get())

def get_trail_length():
    if detection is not None:
        detection.trail = deque(maxlen=int(trail_length_slider.get())) 

def get_lidar_refresh_rate():
    return int(lidar_refresh_rate_slider.get())

def get_show_fps():
    if detection is not None:
        detection.FPS_counter = bool(show_fps_var.get())


def get_model():
    model = model_path_combobox.get()
    
    if model != "" and model != "None":
        return model_paths_map[model]
    else:
        return None

def get_config():
    config = config_path_combobox.get()
    
    if config != "" and config != "None":
        return config_paths_map[config]
    else:
        return None

def get_labels():
    labels = labels_path_combobox.get()
    
    if labels != "" and labels != "None":
        return label_paths_map[labels]
    else:
        return None

# Button commands
def on_connect():
    global robot
    if get_connection_method() != "":
        if robot is None:
            robot_connection_setup()
        elif robot is not None:
            robot = None

def on_forward():
    global robot, robot_manual_control
    if robot is not None and robot_manual_control:
        move_speed = 5 * (move_speed_slider.get() / 100)
        run_async(move.go2_movement(robot.conn, move_speed, 0, 0))

def on_backward():
    global robot, robot_manual_control
    if robot is not None and robot_manual_control:
        move_speed = 5 * (move_speed_slider.get() / 100)
        run_async(move.go2_movement(robot.conn, -move_speed, 0, 0))

def on_left():
    global robot, robot_manual_control
    if robot is not None and robot_manual_control:
        move_speed = 5 * (move_speed_slider.get() / 100)
        run_async(move.go2_movement(robot.conn, 0, move_speed, 0))

def on_right():
    global robot, robot_manual_control
    if robot is not None and robot_manual_control:
        move_speed = 5 * (move_speed_slider.get() / 100)
        run_async(move.go2_movement(robot.conn, 0, -move_speed, 0))

def on_rotate_left():
    global robot, robot_manual_control
    if robot is not None and robot_manual_control:
        rotate_speed = 2 * (rotate_speed_slider.get() / 100)
        run_async(move.go2_movement(robot.conn, 0, 0, rotate_speed))

def on_rotate_right():
    global robot, robot_manual_control
    if robot is not None and robot_manual_control:
        rotate_speed = 2 * (rotate_speed_slider.get() / 100)
        run_async(move.go2_movement(robot.conn, 0, 0, -rotate_speed))

def on_sit():
    global robot, robot_manual_control
    if robot is not None and robot_manual_control:
        run_async(move.perform_action(robot.conn, "StandDown"))

def on_stand():
    global robot, robot_manual_control
    if robot is not None and robot_manual_control:
        run_async(move.perform_action(robot.conn, "StandUp"))

def on_hello():
    global robot, robot_manual_control
    if robot is not None and robot_manual_control:
        run_async(move.perform_action(robot.conn, "Hello"))

def on_pounce():
    global robot, robot_manual_control
    if robot is not None and robot_manual_control:
        run_async(move.perform_action(robot.conn, "FrontPounce"))

def on_power_off():
    global robot
    if robot is not None:
        run_async(move.perform_action(robot.conn, "StandDown"))

def on_manual_mode():
    global robot_manual_control
    robot_manual_control = True


def on_auto_mode():
    global robot_manual_control
    robot_manual_control = False


def on_power_off():
    global robot
    if robot is not None:
        run_async(move.perform_action(robot.conn, "StandDown"))

def on_run_detection():
    if detection is not None:
        start_detection()
    

def on_auto_run():
    if robot is not None:
        get_objective()  

def on_auto_run_stop():
    if robot is not None:
        robot.objective = Objective.Stop  

def on_lidar():
    if robot is not None:
        lidar_thread()

def on_lidar_off():
    global lidar_image
    if robot is not None:
        robot.lidar_disable()
        lidar_image = None


connection_button.config(command=on_connect)
forward_button.config(command=on_forward)
backward_button.config(command=on_backward)
left_button.config(command=on_left)
right_button.config(command=on_right)
rotate_left_button.config(command=on_rotate_left)
rotate_right_button.config(command=on_rotate_right)
sit_button.config(command=on_sit)
stand_button.config(command=on_stand)
hello_button.config(command=on_hello)
pounce_button.config(command=on_pounce)
manual_mode_button.config(command=on_manual_mode)
auto_mode_button.config(command=on_auto_mode)
power_off_mode_button.config(command=on_power_off)
run_detection_button.config(command=on_run_detection)
auto_run_button.config(command=on_auto_run)
auto_run_stop_button.config(command=on_auto_run_stop)
lidar_on_button.config(command=on_lidar)
lidar_off_button.config(command=on_lidar_off)


detection = None
ms_per_frame = None
detection_thread = None
robot = None
robot_manual_control = False
lidar_image = None

# Detection Pipeline

def start_detection():
    global detection_thread, detection, ms_per_frame

    hef_path = get_model()
    config_path = get_config()
    labels_path = get_labels()

    if detection_thread is not None and detection_thread.is_alive():
        detection.running = False
        detection.stop()
        detection_thread.join()
        detection_thread = None
        detection = None

    detection = DetectionPipeline(hef_path, config_path, labels_path, headless=True, resolution=(1280, 720), framerate=60)
    ms_per_frame = ms_per_frame = int(1000 / detection.framerate)
    
    get_camera_source()
    get_resolution()
    get_frame_rate()

    if hef_path is None or config_path is None:
        detection.running = True
        detection_thread = threading.Thread(target=detection.camera_output, daemon=True)
    else:
        detection.running = True
        detection_thread = threading.Thread(target=detection.run, args=(robot,), daemon=True)

    detection_thread.start()

start_detection()



# Robot Connection

def run_async(task):
    threading.Thread(target=lambda: asyncio.run(task), daemon=True).start()

def robot_connection_setup():

    async def setup():
        global robot
        response_action = "Hello"
        while robot is None:
            robot = WebRTCConnection()
            await robot.connection_setup(get_connection_method(), get_ip(), get_serial_number(), get_username(), get_password())
            await asyncio.sleep(1)

        while robot is not None:
            if robot.conn.datachannel.pub_sub.channel.readyState != "open":
                robot = None
                break

            robot.status_check()
            await robot.low_battery_action()
                    
            if detection is not None and robot.objective != Objective.Stop:
                    state = detection.snapshot()
 
                    await move.movement_response(robot,state["detection_time"], state["no_detection_time"], state["detected"], state["future_distance"], state["future_center_x"], state["future_center_y"], Objective.Track_Hit, get_action())

            await asyncio.sleep(0.1)

        
    def async_setup():
        asyncio.run(setup())

    robot_thread = threading.Thread(target=async_setup, daemon=True)
    robot_thread.start()

def lidar_display():
    global robot, lidar_image

    st = time.time()
    while robot is not None and robot.lidar_data is not None:

        if (time.time() - st >= 1/get_lidar_refresh_rate()):
            points = robot.lidar_data

            facing = np.array(robot.orientation)
            facing = facing / np.linalg.norm(facing) 

            sensor_offset = 0.3
            origin = np.array(robot.lidar_origin) + facing * sensor_offset

            robot.avoid_vector, lidar_image = go2_lidar.run_with_plot(points, origin, 0.3, robot.orientation)

            if not bool(boundary_var.get()):
                robot.avoid_vector = None

            st = time.time()
    



def lidar_thread():
    robot.lidar_setup(True, 0)

    lidar_thread = threading.Thread(target=lidar_display, daemon=True)
    lidar_thread.start() 

def status_check(value, suffix=""):
    return f"{value}{suffix}" if value is not None else "N/A"

def update_robot_status():
    global robot
    robot_battery = robot.state_of_charge if robot is not None else None
    robot_velocity = robot.velocity if robot is not None else None
    robot_temp = robot.temperature if robot is not None else None
    robot_motor_temperature = robot.motor_temperature if robot is not None else None
    
    pi_battery = utils.pi_battery_soc()
    cpu_temp = utils.cpu_temp()
    cpu_load = utils.cpu_load()
    npu_temp = utils.npu_temp()
    npu_load = utils.npu_load()

    
    
    robot_battery_label.config(text=f"Robot Battery        : {status_check(robot_battery, "%")}")
    robot_speed_label.config(text=f"Robot Speed          : {status_check(robot_velocity, "m/s")}")
    robot_temp_label.config(text=f"Robot Temp           : {status_check(robot_temp, "°C")}")
    robot_motor_temperature_label.config(text=f"Robot Max Motor Temp : {status_check(robot_motor_temperature, "°C")}")
    
    pi_battery_label.config(text=f"Pi Battery           : {status_check(pi_battery)}")
    cpu_temp_label.config(  text=f"CPU Temp             : {status_check(cpu_temp, "°C")}")
    cpu_load_label.config(  text=f"CPU Load             : {status_check(cpu_load, "%")}")
    npu_temp_label.config(  text=f"NPU Temp             : {status_check(npu_temp, "°C")}")
    npu_load_label.config(  text=f"NPU Load             : {status_check(npu_load, "%")}")

    if robot is not None:
        if robot.conn.datachannel.pub_sub.channel.readyState == "open":
            connection_status_label.config(text="● CONNECTED", fg=SUCCESS)
        else:
            connection_status_label.config(text="● NO CONNECTION", fg=DANGER)
    else:
        connection_status_label.config(text="● NO CONNECTION", fg=DANGER)

    window.after(500, update_robot_status)

def update_detection_status():
    if detection is not None:
        detected = detection.detected
        detection_confidence = status_check(detection.confidence_score, "%")
        distance = status_check(detection.min_distance, "m")
        predicted_distance = status_check(detection.future_distance, "m")

    detection_status_label.config(            text=f"Detected                  : {status_check(detected)}")
    detection_confidence_label.config(        text=f"Object Confidence         : {status_check(detection_confidence)}")
    detection_distance_label.config(          text=f"Object Distance           : {status_check(distance)}")
    detection_predicted_distance_label.config(text=f"Object Predicted Distance : {status_check(predicted_distance)}")

    window.after(500, update_detection_status)

def input_field_check():
    get_move_speed()
    get_rotate_speed()
    get_pitch_speed()
    get_prediction_steps()
    get_object_height()
    get_distance_calibration()
    get_trail_length()
    get_lidar_refresh_rate()
    get_show_fps()
    window.after(500, input_field_check)



def update_video():
    global lidar_image

    window_width  = window.winfo_width()
    window_height = window.winfo_height()
    video_size = (int(window_width * 0.5), int(window_height * 0.5))
    lidar_size = (int(window_width * 0.25), int(window_height * 0.25))

    if detection.display_queue is not None:
        frame = detection.display_queue[:, :, ::-1]

        if frame.shape[:2][::-1] != video_size:
            frame = cv2.resize(frame, video_size, interpolation=cv2.INTER_NEAREST)

        image = Image.fromarray(frame)
        tk_image = ImageTk.PhotoImage(image=image)

        video.imgtk = tk_image
        video.configure(image=tk_image)

    if lidar_image is not None:
        if lidar_image.shape[:2][::-1] != lidar_size:
            lidar_image = cv2.resize(lidar_image, lidar_size, interpolation=cv2.INTER_NEAREST)

        image = Image.fromarray(lidar_image)
        tk_image = ImageTk.PhotoImage(image=image)

        lidar.imgtk = tk_image
        lidar.configure(image=tk_image)
    else:
        lidar.configure(image="", text="No Lidar")

    

    window.after(int(1000/30), update_video)



input_field_check()
update_video()
update_robot_status()
update_detection_status()


window.mainloop()