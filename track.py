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
    
x0 = np.array([0,0,1,30, 30, 30])
#y = traj.traj_ballistic(10, x0)
y = traj.traj_boost(25, x0)
o = np.array([300,-200,0])

tf = y.t[-1]

t = np.linspace(0, tf, 100)
dt = t[1] - t[0]

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
init_state_cov = 0.1*np.array([50,50,50,50,50,50])
proc = 0.1*np.array([50,50,50,50,50,50])

P0 = np.diag(init_state_cov)
Q = np.diag(proc)
R = np.diag([sig_r, sig_b, sig_e, sig_d])

xhat = x0
Phat = P0 
x_filt[:, 0] = x0

measDim = obs.shape[0]

cov = np.zeros((len(t),6,6))
cov[0, :, :] = P0

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
    
    cov[idx, :, :] = Phat
    x_filt[:, idx] = xhat
    
filt_obs = np.zeros(obs.shape)    
for idx in range(x_filt.shape[1]):
    state = x_filt[:, idx]
    s_obs = traj.formObs_rbed(state, o)
    filt_obs[:, idx] = s_obs
  
## plotting

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Trajectory Plot')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)

fig4 = plt.figure()
ax4 = fig4.add_subplot(111)

fig5 = plt.figure()
ax5 = fig5.add_subplot(111)

fig6 = plt.figure()
pos_x_er = fig6.add_subplot(231)
pos_y_er = fig6.add_subplot(232)
pos_z_er = fig6.add_subplot(233)
vel_x_er = fig6.add_subplot(234)
vel_y_er = fig6.add_subplot(235)
vel_z_er = fig6.add_subplot(236)

fig7 = plt.figure()
axpmd = fig7.add_subplot(221)
axvmd = fig7.add_subplot(222)
axptr = fig7.add_subplot(223)
axvtr = fig7.add_subplot(224)

fig8 = plt.figure()
axmd = fig8.add_subplot(121)
axtr = fig8.add_subplot(122)

ax.scatter(px, py, pz, c='r', marker='.')
ax.scatter(x_filt[0,:],x_filt[1,:],x_filt[2,:],'b*')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

#ax.set_xlim([-5, 5])
#ax.set_ylim([-10, 10])
ax.scatter(o[0], o[1], o[2], c='b', marker='*')

ax2.scatter(b*180/np.pi, e*180/np.pi, c=r, marker='.')
ax2.set_title('Truth Obs')
ax3.scatter(obs_b*180/np.pi, obs_e*180/np.pi, c=obs_r, marker='.')
ax3.set_title('Noised Obs')

ax4.scatter(obs_b*180/np.pi, obs_e*180/np.pi, c=obs_r, marker='.')
ax4.scatter(filt_obs[1, :]*180/np.pi, filt_obs[2,:]*180/np.pi, c=filt_obs[0, :], marker='*')
ax4.set_title('Noised Obs with Filtered Obs')

ax5.scatter(b*180/np.pi, e*180/np.pi, c=r, marker='.')
ax5.scatter(filt_obs[1, :]*180/np.pi, filt_obs[2,:]*180/np.pi, c=filt_obs[0, :], marker='*')
ax5.set_title('Truth Obs with Filtered Obs')

pos_x_er.plot(px - x_filt[0,:])
pos_y_er.plot(py - x_filt[1,:])
pos_z_er.plot(pz - x_filt[2,:])

pos_x_er.set_title('Position Error - x')
pos_y_er.set_title('Position Error - y')
pos_z_er.set_title('Position Error - z')

vel_x_er.plot(vx - x_filt[3,:])
vel_y_er.plot(vy - x_filt[4,:])
vel_z_er.plot(vz - x_filt[5,:])

vel_x_er.set_title('Velocity Error - x')
vel_y_er.set_title('Velocity Error - y')
vel_z_er.set_title('Velocity Error - z')

poserr = x_filt[0:3, :] - x[0:3]
velerr = x_filt[3:, :] - x[3:]
ferr = x_filt - x
pmd = []
vmd = []
ptrace = []
vtrace = []

fullmd = []
fulltr = []

for idx in range(poserr.shape[1]):
    perr1 = poserr[:, idx]
    verr1 = velerr[:, idx]
    ferr1 = ferr[:, idx]
    
    cov_est = cov[idx, :, :]
    pcov = cov_est[0:3, 0:3]
    vcov = cov_est[3:, 3:]
    
    tmp1 = np.matmul(perr1, np.linalg.inv(pcov))
    pmd.append(np.matmul(tmp1,perr1.T))
    
    tmp2 = np.matmul(verr1, np.linalg.inv(vcov))
    vmd.append(np.matmul(tmp2,verr1.T))
    
    ptrace.append(np.sqrt(pcov.trace()))
    vtrace.append(np.sqrt(vcov.trace()))
    
    tmp3 = np.matmul(ferr1, np.linalg.inv(cov_est))
    fullmd.append(np.matmul(tmp3,ferr1.T))
    
    fulltr.append(np.sqrt(cov_est.trace()))

axpmd.plot(pmd)
axpmd.set_title('Pos MD')

axvmd.plot(vmd)
axvmd.set_title('Vel MD')

axptr.plot(ptrace)
axptr.set_title('Pos Trc')

axvtr.plot(vtrace)
axvtr.set_title('Vel Trc')   

axmd.plot(fullmd)
axmd.set_title('Full MD')

axtr.plot(fulltr)
axtr.set_title('Full Tr') 

plt.show()