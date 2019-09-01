# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 16:38:30 2019

@author: Alex

This script was just meant to test that the ballistic propagation contained in 
traj.py is suitable for the motion model time update of the kalman filter

to actually see what's going on here, run track.py first and then run this

will be deprectated in a future version

your mileage may vary

"""

i_ex = 15
t_ex = t[i_ex]

new_x = x[:, i_ex]


y_new = traj.traj_ballistic(tf, new_x)

tf_new = y_new.t[-1]
t_new = np.linspace(t_ex, tf_new)

xx = y_new.sol(t_new)
px2 = xx[0]
py2 = xx[1]
pz2 = xx[2]

ax.scatter(px2, py2, pz2, c='b', marker='.')