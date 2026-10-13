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

from tools import setcubeplacement

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)
    oMhookL = getcubeplacement(cube, LEFT_HOOK) #placement of the left hand hook
    oMhookR = getcubeplacement(cube, RIGHT_HOOK) #placement of the right hand hook
    q = qcurrent.copy()                              # our first guess
    idL = robot.model.getFrameId(LEFT_HAND)          # ID number of the left-hand frame
    idR = robot.model.getFrameId(RIGHT_HAND)         # ID number of the right-hand frame
    for i in range(100):
        
        pin.framesForwardKinematics(robot.model, robot.data, q)   # work out where every frame is
        oMhandL = robot.data.oMf[idL]                    # where the left hand is
        oMhandR = robot.data.oMf[idR]                    # where the right hand is

        eL = pin.log(oMhandL.inverse() * oMhookL).vector
        eR = pin.log(oMhandR.inverse() * oMhookR).vector

        e = np.hstack([eL, eR])
        print(i, norm(e))
        if norm(e) < EPSILON: break
        JL = pin.computeFrameJacobian(robot.model, robot.data, q, idL)   # how joints move the LEFT hand
        JR = pin.computeFrameJacobian(robot.model, robot.data, q, idR)   # how joints move the RIGHT hand

        # stacked version (both hands equal priority):
        # J = np.vstack([JL, JR])  # how joints move both hands
        # vq = pinv(J) @ e  # how to move the joints to move the hands to the hooks

        # null-space version (left hand first, right hand in its null space), as in tutorial 3
        vqL = pinv(JL) @ eL                          # joint motion that fixes the left hand
        PL = np.eye(robot.nv) - pinv(JL) @ JL        # joint motions that don't move the left hand
        vq = vqL + pinv(JR @ PL) @ (eR - JR @ vqL)   # fix the right hand using only those motions
        q += vq  # move the joints
        q = projecttojointlimits(robot, q)  # make sure we don't violate joint limits

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
    
    
    
