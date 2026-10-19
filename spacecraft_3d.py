

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.linalg import solve_continuous_are


class Spacecraft:
    def __init__(self, dt=0.05):
        self.dt = dt
        self.Ix = 100.0
        self.Iy = 150.0
        self.Iz = 200.0

    def step(self, state, torque):
        phi, theta, psi, p, q, r = state
        Tx, Ty, Tz = torque

        # Gravity gradient disturbance
        n   = 0.00113
        Tx += 3 * n**2 * (self.Iz - self.Iy) * np.sin(phi) * np.cos(phi) * 0.001
        Ty += 3 * n**2 * (self.Iz - self.Ix) * np.sin(theta) * np.cos(theta) * 0.001

        # Euler rotation equations
        dp = (Tx - (self.Iz - self.Iy) * q * r) / self.Ix
        dq = (Ty - (self.Ix - self.Iz) * p * r) / self.Iy
        dr = (Tz - (self.Iy - self.Ix) * p * q) / self.Iz

        # Kinematic equations
        dphi   = p + (q * np.sin(phi) + r * np.cos(phi)) * np.tan(theta)
        dtheta = q * np.cos(phi) - r * np.sin(phi)
        dpsi   = (q * np.sin(phi) + r * np.cos(phi)) / (np.cos(theta) + 1e-6)

        new_state = state + self.dt * np.array([dphi, dtheta, dpsi, dp, dq, dr])
        new_state[:3] = np.clip(new_state[:3], -np.pi, np.pi)
        new_state = np.nan_to_num(new_state, nan=0.0)
        return new_state


class PD:
    def __init__(self, Kp=5.0, Kd=20.0):
        self.Kp = Kp
        self.Kd = Kd

    def compute(self, state):
        torque = -self.Kp * state[:3] - self.Kd * state[3:]
        return np.clip(torque, -5.0, 5.0)


class LQR:
    def __init__(self):
        A = np.zeros((6, 6))
        A[0, 3] = A[1, 4] = A[2, 5] = 1.0
        B = np.zeros((6, 3))
        B[3, 0] = 1.0/100
        B[4, 1] = 1.0/150
        B[5, 2] = 1.0/200
        Q = np.diag([50.0, 50.0, 50.0, 5.0, 5.0, 5.0])
        R = np.diag([1.0, 1.0, 1.0])
        P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.inv(R) @ B.T @ P

    def compute(self, state):
        return np.clip(self.K @ (-state), -5.0, 5.0)


def run(ctrl, use_lqr=False, n=500):
    sc = Spacecraft()
    state = np.array([np.deg2rad(25), np.deg2rad(20),
                      np.deg2rad(15), 0.08, -0.05, 0.04])
    hist = []
    for _ in range(n):
        torque = ctrl.compute(state)
        state  = sc.step(state, torque)
        hist.append(state.copy())
    return np.array(hist)

print("Running PD simulation...")
pd_hist  = run(PD())
print("Running LQR simulation...")
lqr_hist = run(LQR())


def euler_to_R(phi, theta, psi):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi),  np.cos(phi)]])
    Ry = np.array([[ np.cos(theta), 0, np.sin(theta)],
                   [0,              1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi),  np.cos(psi), 0],
                   [0,            0,            1]])
    return Rz @ Ry @ Rx

def make_box_faces(R, size=np.array([2.0, 1.5, 0.8])):
    h = size / 2
    corners_local = np.array([
        [-h[0], -h[1], -h[2]], [ h[0], -h[1], -h[2]],
        [ h[0],  h[1], -h[2]], [-h[0],  h[1], -h[2]],
        [-h[0], -h[1],  h[2]], [ h[0], -h[1],  h[2]],
        [ h[0],  h[1],  h[2]], [-h[0],  h[1],  h[2]]
    ])
    c = np.array([R @ corner for corner in corners_local])
    return [
        [c[0], c[1], c[2], c[3]],
        [c[4], c[5], c[6], c[7]],
        [c[0], c[1], c[5], c[4]],
        [c[2], c[3], c[7], c[6]],
        [c[1], c[2], c[6], c[5]],
        [c[0], c[3], c[7], c[4]],
    ]


fig = plt.figure(figsize=(16, 8))
fig.patch.set_facecolor('black')

ax1 = fig.add_subplot(121, projection='3d')
ax1.set_facecolor('black')

ax2 = fig.add_subplot(122, projection='3d')
ax2.set_facecolor('black')

def style_ax(ax, title):
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_xlabel('X', color='silver', fontsize=9)
    ax.set_ylabel('Y', color='silver', fontsize=9)
    ax.set_zlabel('Z', color='silver', fontsize=9)
    ax.tick_params(colors='dimgrey', labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('darkslategrey')
    ax.yaxis.pane.set_edgecolor('darkslategrey')
    ax.zaxis.pane.set_edgecolor('darkslategrey')
    ax.set_title(title, color='white', fontsize=12, pad=12)

style_ax(ax1, 'PD Controller')
style_ax(ax2, 'LQR Controller')

# Draw inertial axes
for ax in [ax1, ax2]:
    ax.quiver(0, 0, 0, 2.5, 0, 0, color='red',   arrow_length_ratio=0.15, linewidth=1, alpha=0.3)
    ax.quiver(0, 0, 0, 0, 2.5, 0, color='green', arrow_length_ratio=0.15, linewidth=1, alpha=0.3)
    ax.quiver(0, 0, 0, 0, 0, 2.5, color='blue',  arrow_length_ratio=0.15, linewidth=1, alpha=0.3)

pd_box  = Poly3DCollection([], alpha=0.55, facecolor='steelblue',  edgecolor='lightskyblue', linewidth=0.5)
lqr_box = Poly3DCollection([], alpha=0.55, facecolor='darkgreen',  edgecolor='limegreen',    linewidth=0.5)
ax1.add_collection3d(pd_box)
ax2.add_collection3d(lqr_box)

pd_arrows  = [ax1.quiver(0, 0, 0, 1, 0, 0, color=c, linewidth=2, arrow_length_ratio=0.2)
               for c in ['salmon', 'lightgreen', 'cornflowerblue']]
lqr_arrows = [ax2.quiver(0, 0, 0, 1, 0, 0, color=c, linewidth=2, arrow_length_ratio=0.2)
               for c in ['salmon', 'lightgreen', 'cornflowerblue']]

pd_text  = ax1.text2D(0.02, 0.95, '', transform=ax1.transAxes,
                       color='lightskyblue', fontsize=9, va='top', fontfamily='monospace')
lqr_text = ax2.text2D(0.02, 0.95, '', transform=ax2.transAxes,
                       color='limegreen', fontsize=9, va='top', fontfamily='monospace')

fig.text(0.5, 0.01,
         'Body frame arrows (salmon=X, lightgreen=Y, cornflowerblue=Z) rotate with the spacecraft',
         ha='center', color='grey', fontsize=9)


def redraw_arrows(ax, old_arrows, R, scale=1.8):
    directions = [R[:, 0], R[:, 1], R[:, 2]]
    colors     = ['salmon', 'lightgreen', 'cornflowerblue']
    for arrow in old_arrows:
        arrow.remove()
    return [ax.quiver(0, 0, 0,
                       d[0]*scale, d[1]*scale, d[2]*scale,
                       color=c, linewidth=2, arrow_length_ratio=0.2)
            for d, c in zip(directions, colors)]


def update(frame):
    global pd_arrows, lqr_arrows
    i = min(frame * 2, len(pd_hist) - 1)

    phi_pd,  theta_pd,  psi_pd  = pd_hist[i,  :3]
    phi_lqr, theta_lqr, psi_lqr = lqr_hist[i, :3]

    R_pd  = euler_to_R(phi_pd,  theta_pd,  psi_pd)
    R_lqr = euler_to_R(phi_lqr, theta_lqr, psi_lqr)

    pd_box.set_verts( make_box_faces(R_pd))
    lqr_box.set_verts(make_box_faces(R_lqr))

    pd_arrows  = redraw_arrows(ax1, pd_arrows,  R_pd)
    lqr_arrows = redraw_arrows(ax2, lqr_arrows, R_lqr)

    pd_text.set_text(
        f"Roll  : {np.rad2deg(phi_pd):+6.1f} deg\n"
        f"Pitch : {np.rad2deg(theta_pd):+6.1f} deg\n"
        f"Yaw   : {np.rad2deg(psi_pd):+6.1f} deg\n"
        f"t = {i*0.05:.1f} s"
    )
    lqr_text.set_text(
        f"Roll  : {np.rad2deg(phi_lqr):+6.1f} deg\n"
        f"Pitch : {np.rad2deg(theta_lqr):+6.1f} deg\n"
        f"Yaw   : {np.rad2deg(psi_lqr):+6.1f} deg\n"
        f"t = {i*0.05:.1f} s"
    )
    return pd_box, lqr_box

ani = animation.FuncAnimation(fig, update, frames=250, interval=60, blit=False)

plt.suptitle('Spacecraft Attitude Stabilisation — PD vs LQR',
             color='white', fontsize=14, y=1.01)
plt.tight_layout()

print("Saving GIF... please wait 2-3 minutes")
ani.save('spacecraft_3d.gif', writer='pillow', fps=15, dpi=80)
print("Saved: spacecraft_3d.gif")
plt.show()