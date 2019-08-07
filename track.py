# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 18:47:28 2019

@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import traj

def formObs_rbe(x, o):
    px = x[0] - o[0]
    py = x[1] - o[1]
    pz = x[2] - o[2]
    r = np.sqrt(px*px + py*py + pz*pz)
    b = np.arctan2(py, px)
    e = np.arctan2(pz, np.sqrt(px*px + py*py))
    return r,b,e
    

x0 = np.array([0,0,1,30, 30, 30])
y = traj.traj_ballistic(10, x0)
o = np.array([100,0,0])
#y = traj.traj_boost(25, x0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

tf = y.t[-1]

t = np.linspace(0, tf)

x = y.sol(t)
px = x[0]
py = x[1]
pz = x[2]

r, b, e = formObs_rbe(x, o)

ax.scatter(px, py, pz, c='r', marker='.')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

#ax.set_xlim([-5, 5])
#ax.set_ylim([-10, 10])
ax.scatter(o[0], o[1], o[2], c='b', marker='*')

plt.show()