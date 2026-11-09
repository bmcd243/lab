#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np
import pinocchio as pin
from bezier import Bezier
from config import DT

# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 300.               # proportional gain (P of PD)
Kv = 2 * np.sqrt(Kp)   # derivative gain (D of PD)
Ki = 2000.              # integral gain (I of PID): removes the steady offset left by model errors
ie = np.zeros(15)       # running total of the error over time (the e_i of Tutorial 5)

def controllaw(sim, robot, trajs, tcurrent, cube):
    global ie                                 # ie must survive from one call to the next
    q, vq = sim.getpybulletstate()
    q_ref = trajs[0](tcurrent)
    vq_ref = trajs[1](tcurrent)
    vvq_ref = trajs[2](tcurrent)
    e = q_ref - q
    de = vq_ref - vq
    ie = ie + e * DT                          # add this tick's error to the running total
    a_des = vvq_ref + Kp * e + Kv * de + Ki * ie
    M = pin.crba(robot.model, robot.data, q)
    M = np.triu(M) + np.triu(M, 1).T          # rebuild the full symmetric M (older pinocchio fills half)
    nle = pin.nle(robot.model, robot.data, q, vq)
    torques = M @ a_des + nle

    sim.step(torques)

if __name__ == "__main__":
        
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import DT
    
    robot, sim, cube = setupwithpybullet()
    
    
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET    
    from inverse_geometry import computeqgrasppose
    from path import computepath, projectedpath

    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)   # not used yet: needed for the actual task

    # Task 0: lift both hands 10 cm above the start, with no cube.
    # projectedpath moves the cube placement up in small steps and re-solves the grasp at each
    # step, so qs[-1] holds the joint angles for "cube 10 cm higher".
    lifted = pin.SE3(np.eye(3), CUBE_PLACEMENT.translation + np.array([0., 0., 0.10]))
    qs, Ps, okay = projectedpath(robot, cube, q0, CUBE_PLACEMENT, lifted)
    if not okay:
        print("warning: the lifted pose could not be reached, using the last valid one")
    qlift = qs[-1]


    #setting initial configuration
    sim.setqsim(q0)
    # task 0 is "control without the cube": move the simulated cube far away from the hands
    sim.p.resetBasePositionAndOrientation(sim.cubeId, [3., 3., 0.5], [0, 0, 0, 1])
    
    
    #TODO this is just an example, you are free to do as you please.
    #In any case this trajectory does not follow the path 
    #0 init and end velocities
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

    
    
    total_time = 3.
    trajs = maketraj(q0, qlift, total_time)      # task 0: q0 -> the lifted pose (the actual task will use the path)   
    
    tcur = 0.
    
    
    while tcur < total_time:
        rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        tcur += DT

    # how close did the hands get to where they were asked to go?
    from config import LEFT_HAND, RIGHT_HAND
    q, vq = sim.getpybulletstate()
    for name in (LEFT_HAND, RIGHT_HAND):
        frame = robot.model.getFrameId(name)
        pin.framesForwardKinematics(robot.model, robot.data, qlift)
        wanted = robot.data.oMf[frame].translation.copy()
        pin.framesForwardKinematics(robot.model, robot.data, q)
        actual = robot.data.oMf[frame].translation
        print(name, "ended", round(1000 * np.linalg.norm(actual - wanted), 1), "mm from where it should be")
    
    
    