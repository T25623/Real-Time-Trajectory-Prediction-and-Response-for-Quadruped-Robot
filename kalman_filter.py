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

    def predict(self, u):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x
    
    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x
    
    def predict_n_steps(self, n):
        predictions = []
        
        x_future = self.x.copy()
        P_future = self.P.copy()

        for _ in range(n):
            x_future = np.dot(self.F, x_future)
            P_future = np.dot(self.F, np.dot(P_future, self.F.T)) + self.Q

            predictions.append((x_future[0,0], x_future[1,0], x_future[2,0]))

        return predictions

    def xyz_predict():
        dt = 1  

        F = np.array([
            [1,0,0,dt,0,0],
            [0,1,0,0,dt,0],
            [0,0,1,0,0,dt],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]
        ])

        B = np.zeros((6,2))  

        H = np.array([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,1,0,0,0]
        ])

        Q = np.eye(6) * 0.01
        R = np.eye(3) * 0.1

        x0 = np.array([[0],[0],[0],[0],[0],[0]])
        P0 = np.eye(6) * 100

        return KalmanFilter(F, B, H, Q, R, x0, P0)