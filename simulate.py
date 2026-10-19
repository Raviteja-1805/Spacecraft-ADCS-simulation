import numpy as np
import matplotlib.pyplot as plt
from dynamics import Spacecraft
from pd_controller import PDController
from lqr_controller import LQRController
from disturbance import gravity_gradient

def run(controller, use_lqr=False, n=600):
    sc = Spacecraft()
    # start tumbing: 20 deg off on each axis, small initial rates
    state = np.array([np.deg2rad(20), np.deg2rad(15),
                      np.deg2rad(10), 0.05, -0.03, 0.02])
    hist = []
    for _ in range(n):
        phi, theta, psi = state[0], state[1], state[2]
        dist = gravity_gradient(phi, theta)
        if use_lqr:
            torque = controller.compute(state)
        else:
            torque = controller.compute(state[:3], state[3:])
        state = sc.step(state, torque, dist)
        hist.append(state.copy())
    return np.array(hist)

pd = PDController()
lqr= LQRController()
h_pd = run(pd, use_lqr=False)
h_lqr = run(lqr, use_lqr=True)

t = np.arange(len(h_pd)) * 0.05
labels = [ 'Roll ', 'Pitch', 'Yaw' ]
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for i in range(3):
    axes[i].plot(t, np.rad2deg(h_pd[:, i]), 'b-' , label= 'PD' )
    axes[i].plot(t, np.rad2deg(h_pd[:, i]), 'b-' , label= 'LQR' )
    axes[i].axhline(0, color= 'k' , lw=0.5, ls= '--' )
    axes[i].set_ylabel(f'{labels[i]} (deg)' )
    axes[i].legend(); axes[i].grid(True)
axes[2].set_xlabel( 'Time(s)')
axes[0].set_title( 'Spacecraft attitude stabilisation: PD vs LQR with gravity gradient disturbance')
plt.tight_layout()
plt.savefig( 'Spacecraft_results.png', dpi=150)
plt.show()

# print comparision metrics
settle_pd = np.argmax(np.all(np.abs(h_pd[:, :3]) < np.deg2rad(1), axis=1)) * 0.05
settle_lqr = np.argmax(np.all(np.abs(h_lqr[:, :3]) < np.deg2rad(1), axis=1)) * 0.05
print( f"PD settling time: {settle_pd:.1f}s" )
print( f"LQR settling time: {settle_lqr:.1f}s" )