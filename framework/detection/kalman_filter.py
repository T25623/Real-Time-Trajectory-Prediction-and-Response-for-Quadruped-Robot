import numpy as np

class KalmanFilter:
    def __init__(self, F, B, H, Q, R, x0, P0):
        self.F = F
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        self.gravity = np.zeros_like(x0)

    def predict(self, u=None):
        if u is not None:
            self.x = np.dot(self.F, self.x) + np.dot(self.B, u) + self.gravity
        else:
            self.x = np.dot(self.F, self.x) + self.gravity
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.linalg.solve(S.T, np.dot(self.P, self.H.T).T).T  # more stable than inv(S)
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x

    def predict_n_steps(self, n, u=None):
        predictions = []
        x_future = self.x.copy()
        P_future = self.P.copy()
        for _ in range(n):
            if u is not None:
                x_future = np.dot(self.F, x_future) + np.dot(self.B, u) + self.gravity
            else:
                x_future = np.dot(self.F, x_future) + self.gravity
            P_future = np.dot(self.F, np.dot(P_future, self.F.T)) + self.Q
            predictions.append((x_future[0, 0], x_future[1, 0], x_future[2, 0]))
        return predictions

    @staticmethod
    def xyz_predict(g=0.0005):
        dt = 1
        F = np.array([
            [1, 0, 0, dt, 0,  0 ],
            [0, 1, 0, 0,  dt, 0 ],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0 ],
            [0, 0, 0, 0,  1,  0 ],
            [0, 0, 0, 0,  0,  1 ]
        ])
        B = np.zeros((6, 2))
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        Q = np.eye(6) * 0.01
        R = np.eye(3) * 0.1
        x0 = np.zeros((6, 1))
        P0 = np.eye(6) * 100

        kf = KalmanFilter(F, B, H, Q, R, x0, P0)
        kf.gravity = np.array([
            [0],
            [0.5 * g * dt**2],
            [0],
            [0],
            [g * dt],
            [0]
        ])
        return kf