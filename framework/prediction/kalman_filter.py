from filterpy.kalman import KalmanFilter as FilterPyKF
import numpy as np

class KalmanFilter3D:
    def __init__(self, g=0.0, min_observations=6):
        self.g = g
        self.min_observations = min_observations
        self.observation_count = 0
        self.kf = self.build()

    def build(self):
        dt = 1.0
        kf = FilterPyKF(dim_x=6, dim_z=3)
        kf.F = np.array([
            [1, dt, 0,  0, 0,  0],
            [0,  1, 0,  0, 0,  0],
            [0,  0, 1, dt, 0,  0],
            [0,  0, 0,  1, 0,  0],
            [0,  0, 0,  0, 1, dt],
            [0,  0, 0,  0, 0,  1],
        ])
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ])
        kf.P *= 100
        kf.R = np.eye(3) * 0.1
        kf.Q = np.eye(6) * 0.01
        return kf

    def update(self, x, y, z):
        self.kf.predict()
        self.kf.update([x, y, z])
        self.observation_count += 1

    def predict_n_steps(self, n):
        if self.observation_count < self.min_observations:
            return [(0.0, 0.0, 0.0)]
        x_orig = self.kf.x.copy()
        P_orig = self.kf.P.copy()
        predictions = []
        for i in range(n):
            self.kf.predict()
            state = self.kf.x.flatten()
            px, vx, py, vy, pz, vz = state
            dt = i + 1
            py_corrected = py + 0.5 * self.g * dt ** 2
            predictions.append((px, py_corrected, pz))
        self.kf.x = x_orig
        self.kf.P = P_orig
        return predictions