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
Kp = 300.
Kv = 2 * np.sqrt(Kp)
Ki = 2000.              # integral term removes the steady offset left by model errors
ie = np.zeros(15)       # integral of the joint error
damping = np.zeros(15)  # pybullet joint damping, read from the simulator in __main__
K = 1000                # spring pulling each hand onto its hook (N/m)
S = 50                  # fixed squeeze per hand (N), used when FORCE_FEEDBACK is False

# Extension: force feedback on the grip, PI control of the measured contact force
# (Lecture 11, slide 22).
FORCE_FEEDBACK = True
MU = 0.5                # hand-cube friction in pybullet
SAFETY = 0.6
F_MIN = 30.
KF_P, KF_I = 0.5, 30.
press = friction = press_error_sum = 0.


def gripforce(sim):
    '''Squeeze per hand (N), from the measured contact forces.'''
    global press, friction, press_error_sum

    press_now = friction_now = 0.
    for c in sim.p.getContactPoints(sim.robot, sim.cubeId):
        press_now += c[9]                        # normal force
        friction_now += np.hypot(c[10], c[12])   # friction force
    press = 0.8 * press + 0.2 * press_now / 2    # smoothed, per hand
    friction = 0.8 * friction + 0.2 * friction_now / 2

    target = max(F_MIN, friction / (SAFETY * MU))   # enough press for the friction needed
    error = target - press
    press_error_sum += error * DT
    squeeze = target + KF_P * error + KF_I * press_error_sum
    return min(max(squeeze, 0.), 150.)


def controllaw(sim, robot, trajs, tcurrent, cube):
    global ie
    q, vq = sim.getpybulletstate()
    q_ref = trajs[0](tcurrent)
    vq_ref = trajs[1](tcurrent)
    vvq_ref = trajs[2](tcurrent)

    # hooks: cube pose from pybullet times the hook offsets in the cube model
    pos, quat = sim.p.getBasePositionAndOrientation(sim.cubeId)
    oMcube = pin.XYZQUATToSE3(np.concatenate([pos, quat]))
    hookL = (oMcube * cube.data.oMf[cube.model.getFrameId(LEFT_HOOK)]).translation
    hookR = (oMcube * cube.data.oMf[cube.model.getFrameId(RIGHT_HOOK)]).translation

    pin.framesForwardKinematics(robot.model, robot.data, q)
    handL = robot.data.oMf[robot.model.getFrameId(LEFT_HAND)].translation
    handR = robot.data.oMf[robot.model.getFrameId(RIGHT_HAND)].translation

    # linear part of the hand Jacobians, in world axes
    JL = pin.computeFrameJacobian(robot.model, robot.data, q, robot.model.getFrameId(LEFT_HAND), pin.LOCAL_WORLD_ALIGNED)[:3,:]
    JR = pin.computeFrameJacobian(robot.model, robot.data, q, robot.model.getFrameId(RIGHT_HAND), pin.LOCAL_WORLD_ALIGNED)[:3,:]

    # grasp force: spring onto each hook + squeeze towards the other hand
    inward = (handR - handL) / np.linalg.norm(handR - handL)
    squeeze = gripforce(sim) if FORCE_FEEDBACK else S
    FL = K * (hookL - handL) + squeeze * inward
    FR = K * (hookR - handR) - squeeze * inward

    # inverse dynamics with PID on the joint error
    e = q_ref - q
    de = vq_ref - vq
    ie = ie + e * DT
    a_des = vvq_ref + Kp * e + Kv * de + Ki * ie
    M = pin.crba(robot.model, robot.data, q)
    M = np.triu(M) + np.triu(M, 1).T          # older pinocchio fills only the upper half
    nle = pin.nle(robot.model, robot.data, q, vq)
    torques = M @ a_des + nle + damping * vq + JL.T @ FL + JR.T @ FR

    sim.step(torques)


# level 0: straight line in q with a smoothstep time law (zero velocity at both ends)
def maketraj(q0, q1, T):
    dq = q1 - q0

    def q_of_t(t):
        u = t / T
        s = 3*u**2 - 2*u**3
        return q0 + s * dq

    def vq_of_t(t):
        u = t / T
        return (6*u - 6*u**2) * dq / T

    def vvq_of_t(t):
        u = t / T
        return (6 - 12*u) * dq / T**2

    return q_of_t, vq_of_t, vvq_of_t


# level 0 along the planned path: the smoothstep sets the distance travelled,
# linear interpolation between path points gives the configuration
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

    def direction(t):             # unit direction of the path segment we are on
        u = t / T
        dist = (3*u**2 - 2*u**3) * total
        k = np.searchsorted(cum, dist, side='right') - 1
        k = min(max(k, 0), len(steps) - 1)
        return (path[k+1] - path[k]) / max(steps[k], 1e-12)

    def vq_of_t(t):
        u = t / T
        speed = total * (6*u - 6*u**2) / T
        return speed * direction(t)

    def vvq_of_t(t):
        u = t / T
        acc = total * (6 - 12*u) / T**2
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
    '''Bezier (Bernstein) weights of the n+1 control points at t in [0, 1].'''
    return np.array([comb(n, i) * t**i * (1-t)**(n-i) for i in range(n+1)])


def maketrajbezier(path, T, robot, n=15):
    '''Level 1: one Bezier curve (n+1 control points) fitted to the path with a QP.'''
    path = np.array(path)                    # (number of path points, 15)

    # time of each path point in [0, 1], proportional to the distance along the path
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate(([0], np.cumsum(steps)))
    total = cum[-1]
    times = cum / total

    # A @ X = curve at every path time (X = control points)
    A = np.array([weights(t, n) for t in times])

    # first/last three control points = q0/qe: zero velocity and acceleration at both ends
    C = np.zeros((6, n + 1))
    C[0, 0] = 1
    C[1, 1] = 1
    C[2, 2] = 1
    C[3, n-2] = 1
    C[4, n-1] = 1
    C[5, n] = 1
    d = np.zeros(6)

    # joint limits on every control point (the curve stays in their convex hull)
    G = np.vstack((np.eye(n + 1), -np.eye(n + 1)))

    # one QP per joint: min ||A x - y||^2 subject to C x = d, G x <= h
    X = np.zeros((n + 1, 15))
    for j in range(15):
        y = path[:, j]
        d[:3] = path[0, j]
        d[3:] = path[-1, j]
        h = np.hstack((np.full(n + 1, robot.model.upperPositionLimit[j]), -np.full(n + 1, robot.model.lowerPositionLimit[j])))
        H = A.T @ A + 1e-6 * np.eye(n + 1)   # small ridge keeps H invertible when the path has only 2 points
        q = -A.T @ y
        X[:, j] = quadprog_solve_qp(H, q, G, h, C, d)

    curve = Bezier(list(X), t_max=T)
    return curve, curve.derivative(1), curve.derivative(2)


if __name__ == "__main__":

    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, LEFT_HAND, RIGHT_HAND
    from inverse_geometry import computeqgrasppose
    from path import computepath

    robot, sim, cube = setupwithpybullet()

    # pybullet gives the hand-tip links (no inertial data in the URDF) a default mass of 1 kg:
    # add the same 1 kg point mass to the pinocchio model so inverse dynamics compensates it
    for name in (LEFT_HAND, RIGHT_HAND):
        frame = robot.model.frames[robot.model.getFrameId(name)]
        joint = frame.parentJoint if hasattr(frame, "parentJoint") else frame.parent   # pinocchio 3 / 2
        robot.model.inertias[joint] += frame.placement.act(pin.Inertia(1., np.zeros(3), np.zeros((3, 3))))
    robot.data = robot.model.createData()
    # pybullet also damps every joint: the controller adds these torques back
    damping[:] = [sim.p.getJointInfo(sim.robot, i)[6] for i in sim.bulletCtrlJointsInPinOrder]

    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    #setting initial configuration
    sim.setqsim(q0)

    total_time = 6.
    trajs = maketrajbezier(path, total_time, robot, n=15)
    tcur = 0.

    while tcur < total_time:
        rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        tcur += DT

    # final hand error with respect to qe
    q, vq = sim.getpybulletstate()
    for name in (LEFT_HAND, RIGHT_HAND):
        frame = robot.model.getFrameId(name)
        pin.framesForwardKinematics(robot.model, robot.data, qe)
        wanted = robot.data.oMf[frame].translation.copy()
        pin.framesForwardKinematics(robot.model, robot.data, q)
        actual = robot.data.oMf[frame].translation
        print(name, "ended", round(1000 * np.linalg.norm(actual - wanted), 1), "mm from where it should be")
