import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from scipy.optimize import minimize

TOTAL_TIME = 50

def receiver_position(time):
    start_pos = np.array([-60, -50, 40]).T
    end_pos = np.array([30, 40, 40]).T
    tof = TOTAL_TIME

    # if time > tof:
    #     print('Time needs to be between 0 and {}'.format(tof))
    #     return

    s_t = time / tof # scaled time for parameterization, so time goes 0 -> 1
    pos = s_t * end_pos + (1 - s_t) * start_pos
    return pos


def get_meas(tower, time, c):
    """
    the issue here is that the receiver moves, so we have to iterate to find the actual time of arrival
    INPUTS:
        - tower, 3x1 position of launch
        - time, time of launch (scalar)
        - c, speed of flight of signal (distance per time)
    """

    rng = lambda t: np.linalg.norm(receiver_position(t) - tower)
    f = lambda t: rng(t) / c + time
    tol = 1e-8
    old_t = 0
    maxIter = 20
    for i in range(maxIter):
        new_t = f(old_t)
        if np.abs(new_t - old_t) < tol:
            return new_t
        old_t = new_t
    print('reach maxed iters')
    return new_t

def make_measurements(ist, launch, towers, nmeas):
    """
    make time of arrival measurements given inputs
    INPUTS:
        - ist, inter-signal time (Time between signals)
        - launch, launch time of first signal
        - towers, tower positions (should be a list of 3x1 vectors)
    PARAMETERS:
        - c, flight speed of signal
    """
    c = 50 # arbitrary, may need to change

    #nmeas = int(np.floor((TOTAL_TIME - launch)/ist) + 1)
    launch_times = [launch+(ist*i) for i in range(nmeas)]

    measurements = []
    for tower in towers:
        mgrp = []
        for lt in launch_times:
            meas = get_meas(tower, lt, c)
            mgrp.append(meas)
        measurements.append(mgrp)

    return measurements

def objf_2(x):
    ## split up arguments for optimization
    time = x[-1]
    locs = x[0:15]
    locs2 = [locs[3*i:3*(i+1)] for i in range(5)]
    return locs2, time

def objf_3(x):
    ## split up arguments but enforce 0 altitude
    time = x[-1]
    locs = []
    for i in range(5):
        ll = []
        ll.append(x[2*i])
        ll.append(x[2*i+1])
        ll.append(0)
        locs.append(ll)
    return locs, time

def estimation(measurements, inter_signal_time):
    ntower = len(measurements)
    nmeas = len(measurements[0])

    meas = np.array(measurements)

    meas_flatten = np.array([m for sublist in measurements for m in sublist]).T

    forward_model = lambda est_locs, est_time: make_measurements(inter_signal_time, est_time, est_locs, nmeas)

    x0 = np.zeros((16,))
    objf = lambda x: np.linalg.norm(forward_model(*objf_2(x)) - meas)**2

    res = minimize(objf, x0, method='BFGS')
    print(res.success)
    
def estimation_2d(measurements, inter_signal_time):
    ntower = len(measurements)
    nmeas = len(measurements[0])

    meas = np.array(measurements)


    forward_model = lambda est_locs, est_time: make_measurements(inter_signal_time, est_time, est_locs, nmeas)
    x0 = np.zeros((11,))
    objf = lambda x: np.linalg.norm(forward_model(*objf_3(x)) - meas)**2

    res = minimize(objf, x0, method='BFGS')
    print(res.x)
    
plot = False
tower_1 = np.array([20, -40, 0]).T
tower_2 = np.array([20, 20, 0]).T
tower_3 = np.array([-20, -10, 0]).T
tower_4 = np.array([-8, 25, 0]).T
tower_5 = np.array([8, -5, 0]).T

towers = [tower_1, tower_2, tower_3, tower_4, tower_5]

inter_signal_time = 8 # time between signals, assuming known
launch_time = 14 # launch time of first signal, assuming unknown, we'll estimate this

nmeas = int(np.floor((TOTAL_TIME - launch_time)/inter_signal_time) + 1)
m = make_measurements(inter_signal_time, launch_time, towers, nmeas)

estimation_2d(m, inter_signal_time)

if plot:

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    time = np.linspace(0, 50, 100)
    r_pos = np.zeros((len(time), 3))
    for nt in range(len(time)):
        t = time[nt]
        rp = receiver_position(t)
        r_pos[nt, :] = rp

    ax.scatter(r_pos[:, 0], r_pos[:, 1], r_pos[:, 2], c='r', marker='.')

    ax.scatter(tower_1[0], tower_1[1], tower_1[2], c='b', marker='x')
    ax.scatter(tower_2[0], tower_2[1], tower_2[2], c='b', marker='x')
    ax.scatter(tower_3[0], tower_3[1], tower_3[2], c='b', marker='x')
    ax.scatter(tower_4[0], tower_4[1], tower_4[2], c='b', marker='x')
    ax.scatter(tower_5[0], tower_5[1], tower_5[2], c='b', marker='x')


    plt.show()
