# -*- coding: utf-8 -*-
"""
Created on Sun May 31 14:45:46 2020

@author: Alex
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 18:47:28 2019

@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ukf

import traj
    
x0 = np.array([0,0,1,30,30,30])
y = traj.traj_ballistic(10, x0)
#y = traj.traj_boost(25, x0)
o = np.array([300,-200,0])

tf = y.t[-1]

t = np.linspace(0, tf, 100)
dt = t[1] - t[0]

cov_all_3 = np.zeros((3, 3, len(t)))
cov_all_6 = np.zeros((6, 6, len(t)))

x = y.sol(t)
x_filt = np.zeros(x.shape)
px = x[0]
py = x[1]
pz = x[2]

vx = x[3]
vy = x[4]
vz = x[5]

r, b, e, d = traj.formObs_rbed(x, o)

## Add noise to obs

sig_r = 0.0005
sig_b = 0.001
sig_e = 0.004
sig_d = 0.001

#sig_r = 0
#sig_b = 0
#sig_e = 0
#sig_d = 0

r_noise = np.random.normal(0, sig_r, len(r))
b_noise = np.random.normal(0, sig_b, len(b))
e_noise = np.random.normal(0, sig_e, len(e))
d_noise = np.random.normal(0, sig_d, len(d))

obs_r = r + r_noise
obs_b = b + b_noise
obs_e = e + e_noise
obs_d = d + d_noise
obs = np.zeros((4, len(t)))
obs[0, :] = obs_r
obs[1, :] = obs_b
obs[2, :] = obs_e
obs[3, :] = obs_d
 
## give to kalman filter
init_state_cov = np.array([50,50,50,50,50,50])
proc = np.array([50,50,50,50,50,50])

P0 = np.diag(init_state_cov)
Q = np.diag(proc)
R = np.diag([sig_r, sig_b, sig_e, sig_d])

xhat = x0
Phat = P0 
x_filt[:, 0] = x0

measDim = obs.shape[0]

for idx in range(1, len(t)):
    ## Time update
    sigma_points_x = ukf.sigmaPoints(xhat, Phat) # get apriori sigma points
    two_n = sigma_points_x.shape[1]
    xhat_k = ukf.propSigmaPoints(sigma_points_x, dt) # propagate sigma points // time update
    xhat_kminus = np.sum(xhat_k, 1)/two_n # recombine to a priori state estimate
    P_kminus = ukf.aPrioriCov(xhat_kminus, xhat_k, Q) # get a priori covariance estimate
    
    ## Measurement update
    sigma_points_meas = ukf.sigmaPoints(xhat_kminus, P_kminus)
    
    yhat_k, yhat_ki = ukf.computedMeas(sigma_points_meas, measDim, o)
    Py = ukf.measCov(yhat_k, yhat_ki, R)
    Pxy = ukf.crossCov(sigma_points_meas,xhat_kminus,yhat_ki,yhat_k)
    Py_inv = np.linalg.inv(Py)
    
    y = obs[:, idx]
    resid = y - yhat_k
    
    K = np.matmul(Pxy, Py_inv)
    xhat_kplus = xhat_kminus + np.dot(K, resid)
    KP = np.matmul(K, Py)
    KPKt = np.matmul(KP, K.T)
    P_kplus = P_kminus - KPKt
    
    xhat = xhat_kplus
    Phat = P_kplus
    
    cov_all_6[:,:,idx] = Phat
    cov_all_3[:,:,idx] = Phat[0:3, 0:3]
    
    x_filt[:, idx] = xhat
    
filt_obs = np.zeros(obs.shape)    
for idx in range(x_filt.shape[1]):
    state = x_filt[:, idx]
    s_obs = traj.formObs_rbed(state, o)
    filt_obs[:, idx] = s_obs

## covariance ground estimate
ground_hat_3 = np.zeros((2, x_filt.shape[1]))
nn = np.zeros((1, x_filt.shape[1]))
for idx in range(1, x_filt.shape[1]):
    state = x_filt[:, idx]
    cov_3 = cov_all_3[:, :, idx]
    
    v_hat, w_hat = np.linalg.eigh(cov_3)
    vec = w_hat[:, 0]
    
    scaling_factor = state[2]/vec[2]
    scaled_pt = vec * scaling_factor
    
    x_hat_ground = state[0] - scaled_pt[0]
    y_hat_ground = state[1] - scaled_pt[1]
    
    ground_hat_3[0,idx] = x_hat_ground
    ground_hat_3[1,idx] = y_hat_ground
    
    nn[:, idx] = np.linalg.norm(ground_hat_3[:, idx] - o[0:2])
    
    
## plotting

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Ground Loc Estimates, 3D')
ax.scatter(o[0], o[1], c = 'b', marker='*')
ax.scatter(ground_hat_3[0, 0:50],ground_hat_3[1, 0:50], c='r', marker='.')
ax.scatter(ground_hat_3[0, 50:-1],ground_hat_3[1, 50:-1], c='g', marker='.')

#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#ax2.set_title('Hist Norm Difference')
#ax2.hist(nn)
    
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.set_title('Trajectory Plot')
#
#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#
#fig3 = plt.figure()
#ax3 = fig3.add_subplot(111)
#
#fig4 = plt.figure()
#ax4 = fig4.add_subplot(111)
#
#fig5 = plt.figure()
#ax5 = fig5.add_subplot(111)
#
#fig6 = plt.figure()
#pos_x_er = fig6.add_subplot(231)
#pos_y_er = fig6.add_subplot(232)
#pos_z_er = fig6.add_subplot(233)
#vel_x_er = fig6.add_subplot(234)
#vel_y_er = fig6.add_subplot(235)
#vel_z_er = fig6.add_subplot(236)
#
#ax.scatter(px, py, pz, c='r', marker='.')
#ax.scatter(x_filt[0,:],x_filt[1,:],x_filt[2,:],'b*')
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
#
##ax.set_xlim([-5, 5])
##ax.set_ylim([-10, 10])
#ax.scatter(o[0], o[1], o[2], c='b', marker='*')
#
#ax2.scatter(b*180/np.pi, e*180/np.pi, c=r, marker='.')
#ax2.set_title('Truth Obs')
#ax3.scatter(obs_b*180/np.pi, obs_e*180/np.pi, c=obs_r, marker='.')
#ax3.set_title('Noised Obs')
#
#ax4.scatter(obs_b*180/np.pi, obs_e*180/np.pi, c=obs_r, marker='.')
#ax4.scatter(filt_obs[1, :]*180/np.pi, filt_obs[2,:]*180/np.pi, c=filt_obs[0, :], marker='*')
#ax4.set_title('Noised Obs with Filtered Obs')
#
#ax5.scatter(b*180/np.pi, e*180/np.pi, c=r, marker='.')
#ax5.scatter(filt_obs[1, :]*180/np.pi, filt_obs[2,:]*180/np.pi, c=filt_obs[0, :], marker='*')
#ax5.set_title('Truth Obs with Filtered Obs')
#
#pos_x_er.plot(px - x_filt[0,:])
#pos_y_er.plot(py - x_filt[1,:])
#pos_z_er.plot(pz - x_filt[2,:])
#
#pos_x_er.set_title('Position Error - x')
#pos_y_er.set_title('Position Error - y')
#pos_z_er.set_title('Position Error - z')
#
#vel_x_er.plot(vx - x_filt[3,:])
#vel_y_er.plot(vy - x_filt[4,:])
#vel_z_er.plot(vz - x_filt[5,:])
#
#vel_x_er.set_title('Velocity Error - x')
#vel_y_er.set_title('Velocity Error - y')
#vel_z_er.set_title('Velocity Error - z')

plt.show()