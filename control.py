#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np
import pinocchio as pin
from bezier import Bezier
from config import DT, LEFT_HAND, RIGHT_HAND, LEFT_HOOK, RIGHT_HOOK
from math import comb
import quadprog

# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 300.               # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)   # derivative gain (D of PD)
Ki = 2000.              # integral gain (I of PID): removes the steady offset left by model errors
ie = np.zeros(15)       # running total of the error over time (the e_i of Tutorial 5)
damping = np.zeros(15)  # joint damping that pybullet applies (read from the simulator in __main__)
K = 1000
S = 50

def controllaw(sim, robot, trajs, tcurrent, cube):
    global ie                                 # ie must survive from one call to the next
    q, vq = sim.getpybulletstate()
    q_ref = trajs[0](tcurrent)
    vq_ref = trajs[1](tcurrent)
    vvq_ref = trajs[2](tcurrent)

    pos, quat = sim.p.getBasePositionAndOrientation(sim.cubeId)
    oMcube = pin.XYZQUATToSE3(np.concatenate([pos, quat]))
    hookL = (oMcube * cube.data.oMf[cube.model.getFrameId(LEFT_HOOK)]).translation
    hookR = (oMcube * cube.data.oMf[cube.model.getFrameId(RIGHT_HOOK)]).translation                 

    # where the hands are now
    pin.framesForwardKinematics(robot.model, robot.data, q)
    handL = robot.data.oMf[robot.model.getFrameId(LEFT_HAND)].translation
    handR = robot.data.oMf[robot.model.getFrameId(RIGHT_HAND)].translation                    # YOU

    JL = pin.computeFrameJacobian(robot.model, robot.data, q, robot.model.getFrameId(LEFT_HAND), pin.LOCAL_WORLD_ALIGNED)[:3,:]  # only the linear part of the Jacobian is needed for the hands
    JR = pin.computeFrameJacobian(robot.model, robot.data, q, robot.model.getFrameId(RIGHT_HAND), pin.LOCAL_WORLD_ALIGNED)[:3,:]  # only the linear part of the Jacobian is needed for the hands

    inward = (handR - handL) / np.linalg.norm(handR - handL)   # 1-unit-long arrow from the left hand to the right hand
    FL = K * (hookL - handL) + S * inward     # left hand: spring onto its hook + squeeze towards the right hand
    FR = K * (hookR - handR) - S * inward     # right hand: the same, squeezing the other way
    
    e = q_ref - q
    de = vq_ref - vq
    ie = ie + e * DT                          # add this tick's error to the running total
    a_des = vvq_ref + Kp * e + Kv * de + Ki * ie
    M = pin.crba(robot.model, robot.data, q)
    M = np.triu(M) + np.triu(M, 1).T          # rebuild the full symmetric M (older pinocchio fills half)
    nle = pin.nle(robot.model, robot.data, q, vq)
    torques = M @ a_des + nle + damping * vq + JL.T @ FL + JR.T @ FR # + the joint damping pybullet applies, which pinocchio does not model

    sim.step(torques)


# time parametrisation of a straight line between two configurations (level 0):
# smooth start and stop, zero velocity at both ends
def maketraj(q0, q1, T):
    dq = q1 - q0                  # how far each joint has to travel (15 numbers)

    def q_of_t(t):                # where each joint should be at time t
        u = t / T                 # fraction of the time that has passed, 0 -> 1
        s = 3*u**2 - 2*u**3       # the slider: 0 -> 1, gentle start and stop
        return q0 + s * dq

    def vq_of_t(t):               # how fast each joint should be moving
        u = t / T                       # YOU: slider speed (6u - 6u^2)/T, times dq
        return (6*u - 6*u**2) * dq / T

    def vvq_of_t(t):              # how fast the speed should be changing
        u = t / T                      # YOU: slider acceleration (6 - 12u)/T^2, times dq
        return (6 - 12*u) * dq / T**2

    return q_of_t, vq_of_t, vvq_of_t


# the same idea for a path made of many configurations (level 0 along the planned path):
# the slider decides how far along the route we are, the route is followed piece by piece
def maketrajpath(path,T):

    path = np.array(path)
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)

    cum = np.concatenate(([0], np.cumsum(steps)))
    total = cum[-1]

    def q_of_t(t):
        u = t / T
        s = 3*u**2 - 2*u**3
        d = s * total
        return np.array([np.interp(d, cum, path[:,i]) for i in range(path.shape[1])])

    def direction(t):             # unit vector (15 numbers): which way the joints move in the piece we are in
        u = t / T
        dist = (3*u**2 - 2*u**3) * total                      # distance along the route, as in q_of_t
        k = np.searchsorted(cum, dist, side='right') - 1      # number of the piece we are in
        k = min(max(k, 0), len(steps) - 1)                    # stay between the first and last piece (t = T)
        return (path[k+1] - path[k]) / max(steps[k], 1e-12)   # end minus start of the piece, divided by its length

    def vq_of_t(t):               # how fast each joint should be moving
        u = t / T
        speed = total * (6*u - 6*u**2) / T                    # speed along the route (distance per second)
        return speed * direction(t)

    def vvq_of_t(t):              # how fast that speed should be changing (the slider's acceleration only)
        u = t / T
        acc = total * (6 - 12*u) / T**2                       # acceleration along the route
        return acc * direction(t)

    return q_of_t, vq_of_t, vvq_of_t

# copied from Tutorial 6 (6_convex_optimisation.ipynb, ediadvancedrobotics/tutorials)
def quadprog_solve_qp(H, q, G=None, h=None, C=None, d=None, verbose=False):
    """
    min (1/2)x' P x + q' x
    subject to  G x <= h
    subject to  C x  = d
    """
    qp_G = 0.5 * (H + H.T)  # make sure P is symmetric
    qp_a = -q
    qp_C = None
    qp_b = None
    meq = 0
    if C is not None:
        if G is not None:
            qp_C = -np.vstack([C, G]).T
            qp_b = -np.hstack([d, h])
        else:
            qp_C = -C.transpose()
            qp_b = -d
        meq = C.shape[0]
    elif G is not None:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
    res = quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)
    if verbose:
        return res
    return res[0]



def weights(t, n):
    '''One row of A: the n+1 Bezier weights at slider value t (0..1).'''
    # TODO 1: comb(n, i) * t**i * (1-t)**(n-i) for i = 0..n, as a numpy array
    return np.array([comb(n, i) * t**i * (1-t)**(n-i) for i in range(n+1)])


def maketrajbezier(path, T, robot, n=15):
    '''Level 1: a single Bezier curve (n+1 control points) fitted to the path with a QP.'''
    path = np.array(path)                    # shape (number of path points, 15)

    # TODO 2: times of the path points, 0..1, from arc length
    #         (the same steps / cum / total as in maketrajpath, then cum / total)
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)   # length of each piece
    cum = np.concatenate(([0], np.cumsum(steps)))           # distance from the start to each point
    total = cum[-1]                                         # length of the whole route
    times = cum / total                                      # fraction of the route at each point, 0..1

    # TODO 3: build A: one row per path point, row k = weights(time_k, n)
    rows = [weights(t, n) for t in times]
    A = np.array(rows)                       # shape (number of path points, n+1

    # TODO 4: C (6 x (n+1)): rows picking out P0, P1, P2, P(n-2), P(n-1), P(n)
    #         G (2(n+1) x (n+1)): identity on top, minus identity below
    #         (C and G are the same for every joint, so build them once, here)
    C = np.zeros((6, n + 1))
    C[0, 0] = 1  # P0 for joint 0
    C[1, 1] = 1  # P1 for joint 0
    C[2, 2] = 1  # P2 for joint 0
    C[3, n-2] = 1  # P(n-2) for joint 0
    C[4, n-1] = 1  # P(n-1) for joint 0
    C[5, n] = 1  # P(n) for joint 0
    d = np.zeros(6)
   


    X = np.zeros((n + 1, 15))                # the control points, filled one joint at a time
    for j in range(15):
        y = path[:, j]                       # this joint's path values
        d[0] = path[0, j]  # P0 for joint 0
        d[1] = path[0, j]  # P1 for joint 0
        d[2] = path[0, j]  # P2 for joint 0
        d[3] = path[-1, j]  # P(n-2) for joint 0
        d[4] = path[-1, j]  # P(n-1) for joint 0
        d[5] = path[-1, j]  # P(n) for joint 0
        # TODO 5: H, q from A and y
        #         d: this joint's q0 three times, then its qe three times (path[0, j], path[-1, j])
        #         h: upper limit (n+1 times), then minus lower limit (n+1 times)
        #            limits: robot.model.upperPositionLimit[j], robot.model.lowerPositionLimit[j]
        #         X[:, j] = quadprog_solve_qp(H, q, G, h, C, d)
        H = A.T @ A
        q = -A.T @ y
        G = np.vstack((np.eye(n + 1), -np.eye(n + 1)))
        h = np.hstack((np.full(n + 1, robot.model.upperPositionLimit[j]), -np.full(n + 1, robot.model.lowerPositionLimit[j])))
        X[:, j] = quadprog_solve_qp(H, q, G, h, C, d)

    curve = Bezier(list(X), t_max=T)         # list(X) = list of n+1 rows, each 15 numbers
    return curve, curve.derivative(1), curve.derivative(2)


if __name__ == "__main__":

    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, LEFT_HAND, RIGHT_HAND
    from inverse_geometry import computeqgrasppose
    from path import computepath

    robot, sim, cube = setupwithpybullet()

    # pybullet gives the two hand-tip links (no mass data in the URDF) a default mass of 1 kg that
    # the pinocchio model does not have: make them (almost) massless
    for i in range(sim.p.getNumJoints(sim.robot)):
        if sim.p.getJointInfo(sim.robot, i)[12].decode() in ("LARM_EFF_Link", "RARM_EFF_Link"):
            sim.p.changeDynamics(sim.robot, i, mass=1e-6, localInertiaDiagonal=[1e-9] * 3)
    # pybullet also damps every joint: read those coefficients so the controller can add them back
    damping[:] = [sim.p.getJointInfo(sim.robot, i)[6] for i in sim.bulletCtrlJointsInPinOrder]

    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    #setting initial configuration
    sim.setqsim(q0)
    

    total_time = 6.
    trajs = maketrajbezier(path, total_time, robot, n=15)  # level 1: a single Bezier curve fitted to the path with a QP
    tcur = 0.

    while tcur < total_time:
        rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        tcur += DT

    # how close did the hands get to where they were asked to go (the end of the path, qe)?
    q, vq = sim.getpybulletstate()
    for name in (LEFT_HAND, RIGHT_HAND):
        frame = robot.model.getFrameId(name)
        pin.framesForwardKinematics(robot.model, robot.data, qe)
        wanted = robot.data.oMf[frame].translation.copy()
        pin.framesForwardKinematics(robot.model, robot.data, q)
        actual = robot.data.oMf[frame].translation
        print(name, "ended", round(1000 * np.linalg.norm(actual - wanted), 1), "mm from where it should be")
