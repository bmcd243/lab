#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits, jointlimitsviolated
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET


def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    # Iterative inverse kinematics on the placement error of both hands.
    # Adapted from the 6D placement loop and the two-task null-space formula of
    # Tutorial 3 (ediadvancedrobotics/tutorials, 3_invkine.ipynb).
    setcubeplacement(robot, cube, cubetarget)
    oMhookL = getcubeplacement(cube, LEFT_HOOK)      # where the left hand must go
    oMhookR = getcubeplacement(cube, RIGHT_HOOK)     # where the right hand must go
    idL = robot.model.getFrameId(LEFT_HAND)          # ID number of the left-hand frame
    idR = robot.model.getFrameId(RIGHT_HAND)         # ID number of the right-hand frame
    q = qcurrent.copy()                              # our first guess

    for i in range(100):
        pin.framesForwardKinematics(robot.model, robot.data, q)   # work out where every frame is
        oMhandL = robot.data.oMf[idL]                # where the left hand is
        oMhandR = robot.data.oMf[idR]                # where the right hand is

        # 6D error of each hand, expressed in the hand frame
        eL = pin.log(oMhandL.inverse() * oMhookL).vector
        eR = pin.log(oMhandR.inverse() * oMhookR).vector
        e = np.hstack([eL, eR])
        if norm(e) < EPSILON:
            break

        # Jacobians in the hand frame (LOCAL), matching the frame of the errors
        JL = pin.computeFrameJacobian(robot.model, robot.data, q, idL)   # how joints move the LEFT hand
        JR = pin.computeFrameJacobian(robot.model, robot.data, q, idR)   # how joints move the RIGHT hand

        # stacked version (both hands equal priority) gives identical results:
        # J = np.vstack([JL, JR])
        # vq = pinv(J) @ e

        # null-space version (left hand first, right hand in its null space), as in Tutorial 3
        JLpinv = pinv(JL)
        vqL = JLpinv @ eL                            # joint motion that fixes the left hand
        PL = np.eye(robot.nv) - JLpinv @ JL          # joint motions that don't move the left hand
        vq = vqL + pinv(JR @ PL) @ (eR - JR @ vqL)   # fix the right hand using only those motions

        q = q + vq                                   # full step: only the final q matters here
        q = projecttojointlimits(robot, q)           # never leave the joint limits

    # success only if all three hold: hands on the hooks, no collision, within joint limits
    converged = norm(e) < EPSILON
    nocollision = not collision(robot, q)
    withinlimits = not jointlimitsviolated(robot, q)
    success = converged and nocollision and withinlimits
    return q, success


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()

    q = robot.q0.copy()

    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    print("success q0:", successinit, " success qe:", successend)

    # the second call left the cube at the target and robot.data at qe:
    # put both back to the q0 situation before displaying q0
    setcubeplacement(robot, cube, CUBE_PLACEMENT)
    pin.framesForwardKinematics(robot.model, robot.data, q0)
    updatevisuals(viz, robot, cube, q0)
