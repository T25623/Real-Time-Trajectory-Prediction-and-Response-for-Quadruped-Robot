import pandas as pd
import pyvista as pv
import numpy as np
import framework.robot.go2.setup as go2
from framework.robot.go2.setup import WebRTCConnection
import asyncio
from framework.detection.detection import DetectionPipeline
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

def voxel_downsample(points, nth_point):
    return points[::nth_point]


def filter_and_nearby(points, center, x_dist=1, y_dist=1, z_dist=0.3, inrange_distance=1.0):
    cx, cy, cz = center
    mask = (
        (points[:, 0] > cx - x_dist) & (points[:, 0] < cx + x_dist) &
        (points[:, 1] > cy - y_dist) & (points[:, 1] < cy + y_dist) &
        (points[:, 2] > cz - z_dist) & (points[:, 2] < cz)
    )
    filtered = points[mask]
    
    dx = filtered[:, 0] - cx
    dy = filtered[:, 1] - cy
    distances = np.hypot(dx, dy)
    nearby = filtered[distances < inrange_distance]
    
    return filtered, nearby


def vector_towards_emptyness(points, center):
    if len(points) == 0:
        return None
    
    dx = points[:, 0] - center[0]
    dy = points[:, 1] - center[1]
    angles = np.arctan2(dy, dx)
    angles.sort()
    
    gaps = np.empty(len(angles))
    gaps[:-1] = angles[1:] - angles[:-1]
    gaps[-1] = (angles[0] + 2 * np.pi) - angles[-1]
    
    idx = np.argmax(gaps)
    best_angle = angles[idx] + gaps[idx] / 2
    
    return np.array([np.cos(best_angle), np.sin(best_angle), 0.0])
    

def plot_lidar(filtered_points, nearby_points, center, vector, facing):
    ax.clear()

    if len(filtered_points) > 0:
        ax.scatter(filtered_points[:, 0], filtered_points[:, 1], c='black', s=10)
    if len(nearby_points) > 0:
        ax.scatter(nearby_points[:, 0], nearby_points[:, 1], c='red', s=25)
        
    ax.scatter(*center[:2], c='orange', s=30, zorder=5)
    
    if vector is not None:
        ax.annotate("", xy=(center[0] + vector[0]*0.5, center[1] + vector[1]*0.5),
                    xytext=center[:2], arrowprops=dict(arrowstyle="->", color='green'))
    if facing is not None:
        ax.annotate("", xy=(center[0] + facing[0]*0.5, center[1] + facing[1]*0.5),
                    xytext=center[:2], arrowprops=dict(arrowstyle="->", color='blue'))
    
    ax.set_xlim(center[0] - 1.5, center[0] + 1.5)
    ax.set_ylim(center[1] - 1.5, center[1] + 1.5)
    
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    frame = frame[:, :, :3]

    return frame

def run(points, center, inrange_distance, facing):
    downsample_points = voxel_downsample(points, 5)
    filtered_points, nearby_points = filter_and_nearby(downsample_points, center, inrange_distance=inrange_distance)
    vector = vector_towards_emptyness(nearby_points, center)

    return vector


def run_with_plot(points, center, inrange_distance, facing):
    downsample_points = voxel_downsample(points, 5)
    filtered_points, nearby_points = filter_and_nearby(downsample_points, center, inrange_distance=inrange_distance)
    vector = vector_towards_emptyness(nearby_points, center)

    frame = plot_lidar(filtered_points, nearby_points, center, vector, facing)
    return vector, frame
    