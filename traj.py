import numpy as np
import scipy as sp
from scipy import integrate

import physconst as c

def traj_dxdt_ballistic(t, x):
    gravity = c.gravity()
    r = x[0:3]
    v = x[3:]

    accel = gravity * np.array([0,0,1])
    dx = np.concatenate((v, accel))

    return dx

def traj_ballistic(dt, tf, x0):
    n = int(tf/dt) + 1
    t = np.linspace(0, tf, n)

    x = np.zeros((6, n))

    x[:, 0] = x0

    for idx in np.arange(1,n+1):
        y0 = x[:, idx-1]

        x[:, idx] = integrate.RK45(traj_dxdt_ballistic, t[idx-1], y0, t[idx])

        if x[3, idx] < 0:
            ntimes = idx-1
            x = x[:, :ntimes]
            t = t[:ntimes]

            return t, x

    return t, x

def traj_ballistic2(dt, tf, x0):
    n = int(tf/dt) + 1
    t = np.linspace(0, tf, n)

    y = integrate.solve_ivp(traj_dxdt_ballistic, (0, tf), x0, events=[at_ground,apex], dense_output=True)

    return t, y

def at_ground(t, y):
    return y[2]

def apex(t, y):
    return y[5]

at_ground.terminal=True

x0 = np.array([0,0,1,1,2,15])
t, y = traj_ballistic2(0.1, 10, x0)
print(y.t)
print(y.y)


dat = y.y
px = dat[0, :]
py = dat[1, :]
pz = dat[2, :]
vx = dat[3, :]
vy = dat[4, :]
vz = dat[5, :]
print(py)
