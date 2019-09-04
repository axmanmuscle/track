# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 18:53:20 2019

@author: Alex
"""
import numpy as np
import traj

def sigmaPoints(x, P):
    n = x.size
    
    sigx = np.zeros((n, 2*n))
    
    P = P/2 + P.T/2
    
    mx = np.linalg.cholesky(n*P)
    
    for idx in range(n):
#        sigx[:, idx] = x + mx[idx, :]
#        sigx[:, idx+n] = x - mx[idx, :]
        sigx[:, idx] = x + mx[:, idx]
        sigx[:, idx+n] = x - mx[:, idx]
        
    return sigx
    
def propSigmaPoints(sigx, dt):
    xhat_k = np.zeros(sigx.shape)
    
    for idx in range(xhat_k.shape[1]):        
        xhat_km1 = sigx[:, idx]
        yhat = traj.traj_ballistic(5*dt, xhat_km1)
        xhat_k[:, idx] = yhat.sol(dt)
    
    return xhat_k

def computedMeas(sigx, ndim, o):
    # ndim is number of measurement space dimensions
    # o is origin location
    two_n = sigx.shape[1]
    yhat_ki = np.zeros((ndim, two_n))
    for idx in range(two_n):
        xki = sigx[:,idx]
        r,b,e,d = traj.formObs_rbed(xki, o)
        if b<0:
            b += 2*np.pi
        yhat_ki[:, idx] = np.array([r,b,e,d])
    
    yhat_k = np.sum(yhat_ki, 1)/two_n
    return yhat_k, yhat_ki
    

def aPrioriCov(xhat_kminus, xhat_k, Q):
    n = xhat_k.shape[0]
    two_n = xhat_k.shape[1]
    
    P_kminus = np.zeros((n,n))
    
    for idx in range(two_n):
        xki = xhat_k[:, idx]
        dif = xki - xhat_kminus
        for idx2 in range(dif.shape[0]):
            if abs(dif[idx2]) < 1e-7:
                dif[idx2] = 0
        out = np.outer(dif, dif)
        P_kminus += out
        
    P_kminus *= (1/two_n)
    P_kminus += Q
    return P_kminus
        
def measCov(yhat_k, yhat_ki, R):
    n = yhat_k.shape[0]
    Py = np.zeros((n,n))
    
    two_n = yhat_ki.shape[1]
    
    for idx in range(two_n):
        yki = yhat_ki[:, idx]
        dif = yki - yhat_k
        for idx2 in range(dif.shape[0]):
            if abs(dif[idx2]) < 1e-7:
                dif[idx2] = 0
        out = np.outer(dif,dif)
        Py += out
        
    Py *= (1/two_n)
    Py += R
    return Py
    
def crossCov(xhat_ki, xhat_k, yhat_ki, yhat_k):
    nx = xhat_ki.shape[0]
    ny = yhat_ki.shape[0]
    two_n = xhat_ki.shape[1]
    
    Pxy = np.zeros((nx,ny))
    
    for idx in range(two_n):
        xki = xhat_ki[:, idx]
        yki = yhat_ki[:, idx]
        
        difx = xki - xhat_k
        dify = yki - yhat_k
        
        out = np.outer(difx, dify)
        
        Pxy += out
    
    Pxy *= (1/two_n)
    return Pxy
        
    
    
    
    
    