import numpy as np

def gravity_gradient(phi, theta, mu=3.986e14, R=6.8e6, Ix=100, Iy=150, Iz=200):
    # Gravity gradient disturbance torque
    n = np.sqrt(mu / R**3) # Orbital rate
    Tx = 3 * n**2 * (Iz - Iy) * np.sin(phi) * np.cos(phi) * np.cos(theta)**2
    Ty = 3 * n**2 * (Iz - Ix) * np.sin(theta) * np.cos(theta)
    Tz = 0.0
    return np.array([Tx, Ty, Tz]) * 0.001 # scale to realisic magnitude