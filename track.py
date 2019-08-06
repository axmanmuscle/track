# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 18:47:28 2019

@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import traj

x0 = np.array([0,0,1,1,2,25])
y = traj.traj_ballistic(10, x0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

tf = y.t[-1]

t = np.linspace(0, tf)

x = y.sol(t)
px = x[0]
py = x[1]
pz = x[2]

ax.scatter(px, py, pz, c='r', marker='.')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.set_xlim([-5, 5])
ax.set_ylim([-10, 10])

plt.show()