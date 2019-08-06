import numpy as np
from scipy import integrate

import physconst as c

def at_ground(t, y):
    return y[2]

def apex(t, y):
    return y[5]

def traj_dxdt_ballistic(t, x):
    gravity = c.gravity()
    v = x[3:]

    accel = gravity * np.array([0,0,1])
    dx = np.concatenate((v, accel))

    return dx

def traj_ballistic(tf, x0):
    
    at_ground.terminal=True
    y = integrate.solve_ivp(traj_dxdt_ballistic, (0, tf), x0, events=[at_ground,apex], dense_output=True)

    return y
