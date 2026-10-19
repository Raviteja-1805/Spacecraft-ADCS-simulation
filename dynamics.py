import numpy as np

class Spacecraft:
    def __init__(self, dt=0.05):
        self.dt = dt

        #Moments of inertia (kg.m^2)- assymetric spacecraft
        self.Ix = 100.0
        self.Iy = 150.0
        self.Iz = 200.0

    def step(self, state, torque, disturbance=None):
        # state = [phi, theta, psi, p, q, r]
        # torque = [Tx, Ty, Tz] from reaction wheels
        phi, theta, psi, p, q, r = state
        Tx, Ty, Tz = torque
        if disturbance is None:
            disturbance = np.zeros(3)
        Tx += disturbance[0]
        Ty += disturbance[1]
        Tz += disturbance[2]

        # Euler's rotation equations (Newton's law for rotation )
        dp = (Tx - (self.Iz - self.Iy) * q * r) / self.Ix
        dq = (Ty - (self.Ix - self.Iz) * p * r) / self.Iy
        dr = (Tz - (self.Iy - self.Ix) * p * q) / self.Iz

        # Kinematics : angle rates from angular velocities
        dphi = p + (q * np.sin(phi) + r * np.cos(phi)) * np.tan(theta)
        dtheta = q * np.cos(phi) - r * np.sin(phi)
        dpsi = (q * np.sin(phi) + r * np.cos(phi)) / (np.cos(theta) + 1e-6)

        # Euler integration
        new_state = state + self.dt * np.array([dphi, dtheta, dpsi, dp, dq, dr])
        return new_state