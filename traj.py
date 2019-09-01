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

def traj_dxdt_boost(t, x):
    gravity = c.gravity()
    if t < 5:
        thrust = 75
    else:
        thrust = 0
    v = x[3:]
    
    a_grav = gravity* np.array([0,0,1])
    a_thrust = thrust * (v/np.linalg.norm(v))
    
    accel = a_grav + a_thrust
    
    dx = np.concatenate((v, accel))
    
    return dx

def traj_boost(tf, x0):
    
    at_ground.terminal=True
    y = integrate.solve_ivp(traj_dxdt_boost, (0, tf), x0, events=[at_ground,apex], dense_output=True)
    
    return y
    
def formObs_rbe(x, o):
    px = x[0] - o[0]
    py = x[1] - o[1]
    pz = x[2] - o[2]
    r = np.sqrt(px*px + py*py + pz*pz)
    b = np.arctan2(py, px)
    e = np.arctan2(pz, np.sqrt(px*px + py*py))
    return r,b,e

def formObs_rbed(x, o):
    px = x[0] - o[0]
    py = x[1] - o[1]
    pz = x[2] - o[2]
    vx = x[3]
    vy = x[4]
    vz = x[5]
    r = np.sqrt(px*px + py*py + pz*pz)
    b = np.arctan2(py, px)
    e = np.arctan2(pz, np.sqrt(px*px + py*py))
    d = (px*vx + py*vy + pz*vz)/r
    return r,b,e,d