import pandas as pd
import pyvista as pv
import numpy as np
import framework.go2.go2_setup as go2
from framework.go2.go2_setup import WebRTCConnection
import asyncio
from framework.detection.detection import DetectionPipeline
from framework.detection.kalman_filter import KalmanFilter
import time

def lidar_point_cloud_processing(points, center, x_distance=1, y_distance=1, z_distance=0.3):
    x_min = center[0] - x_distance
    x_max = center[0] + x_distance

    y_min = center[1] - y_distance
    y_max = center[1] + y_distance

    z_min = center[2] - z_distance
    z_max = center[2] 

    mask = (
        (points[:,0] > x_min) & (points[:,0] < x_max) &
        (points[:,1] > y_min) & (points[:,1] < y_max) &
        (points[:,2] > z_min) & (points[:,2] < z_max)
    )

    return points[mask]

def inrange_points(points, center, inrange_distance):
    diff = points - center
    diff[:,2] = 0

    distances = np.linalg.norm(diff, axis=1)

    return points[distances < inrange_distance]


def vector_towards_emptyness(points, center):

    if len(points) == 0:
        return None

    rel = points - center
    rel[:,2] = 0

    angles = np.arctan2(rel[:,1], rel[:,0])

    angles = np.sort(angles)

    angles = np.concatenate([angles, angles[:1] + 2*np.pi])

    gaps = np.diff(angles)

    idx = np.argmax(gaps)

    best_angle = angles[idx] + gaps[idx]/2

    direction = np.array([
        np.cos(best_angle),
        np.sin(best_angle),
        0
    ])
    return direction
    

def update_plotter(plotter, filtered_points, nearby_points, arrow_direction, center, facing=None):
    plotter.clear()
    plotter.add_points(filtered_points, color='black', point_size=2, render_points_as_spheres=True)
    
    if len(nearby_points) > 0:
        plotter.add_points(nearby_points, color='red', point_size=5, render_points_as_spheres=True)
    
    if arrow_direction is not None:
        arrow = pv.Arrow(start=center, direction=arrow_direction, scale=0.5)
        plotter.add_mesh(arrow, color='green')
    
    if facing is not None:
        facing_arrow = pv.Arrow(start=center, direction=facing, scale=0.5)
        plotter.add_mesh(facing_arrow, color='blue')
    
    plotter.add_points(np.array([center]), color='yellow', point_size=15, render_points_as_spheres=True)
    plotter.update()

def run(plotter, points, center, inrange_distance, facing):
    filtered_points = lidar_point_cloud_processing(points, center)
    nearby_points = inrange_points(filtered_points, center, inrange_distance)
    vector = vector_towards_emptyness(nearby_points, center)

    return vector
    
def run_with_plot(plotter, points, center, inrange_distance, facing):
    filtered_points = lidar_point_cloud_processing(points, center)
    nearby_points = inrange_points(filtered_points, center, inrange_distance)
    vector = vector_towards_emptyness(nearby_points, center)

    update_plotter(plotter, filtered_points, nearby_points, vector, center, facing=facing)
    return vector
    