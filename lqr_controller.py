import numpy as np
from scipy.linalg import solve_continuous_are

class LQRController:
    def __init__(self, Ix=100, Iy=150, Iz=200, torque_max=5.0):
        self.torque_max = torque_max
        # Linearised system: A and B Matrices
        # State: [phi, theta, psi, p, q, r]
        self.A = np.zeros((6, 6))
        self.A[0, 3] = 1.0 #dphi= p
        self.A[1, 4] = 1.0 #dtheta = q
        self.A[2, 5] = 1.0 #dpsi = r

        self.B = np.zeros((6, 3))
        self.B[3, 0] = 1.0 / Ix
        self.B[4, 1] = 1.0 / Iy
        self.B[5, 2] = 1.0 / Iz

        # penalise angles (first 3) more than rates (last 3)
        self.Q = np.diag([50.0, 50.0, 50.0, 5.0, 5.0, 5.0])
        self.R = np.diag([1.0, 1.0, 1.0])

        p = solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R) @ self.B.T @ p
        print(f"LQR k shape: {self.K.shape} - computed Ok" )

    def compute(self, state, target=None):
            if target is None:
                target = np.zeros(6)
            error = target - state
            torque = self.K @ error
            return np.clip(torque, -self.torque_max, self.torque_max)