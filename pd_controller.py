import numpy as np

class PDController:
    def __init__(self, kp=5.0, kd=20.0, torque_max=5.0):
        self.kp = kp
        self.kd = kd
        self.torque_max = torque_max

    def compute(self, angles, rates, target=None):
        # angles = [phi, theta, psi], rates = [p, q, r]
        # target = desired angles 
        if target is None:
            target = np.zeros(3)
        angle_error = target - angles
        rate_error  = -rates
        torque = self.kp * angle_error + self.kd * rate_error
        # clamp to reaction wheel limits
        return np.clip(torque, -self.torque_max, self.torque_max)