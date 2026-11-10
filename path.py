#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv, norm
from inverse_geometry import computeqgrasppose
from setup_pinocchio import setuppinocchio
from config import LEFT_HAND, RIGHT_HAND
import time

# Plan: an RRT (adapted from Tutorial 4, ediadvancedrobotics/tutorials 4_rrt.ipynb)
# that plans for the CUBE, not for q directly. Every configuration is obtained from a
# cube placement with computeqgrasppose, so the cube is held along the whole path.

# Planning parameters: starting values, tune them once everything runs
# box where cube positions are sampled (x, y, z). ~53% of samples give a valid grasp.
# z goes above the wall top (1.00 m) so the cube can be lifted over it.
CUBE_LOW  = np.array([0.30, -0.40, 0.93])
CUBE_HIGH = np.array([0.45,  0.20, 1.20])
DELTA = 0.10   # max distance (m) the cube moves along one new edge (like delta_q in Tutorial 4)
STEP  = 0.01   # distance (m) between two re-solved grasps along an edge
K     = 500    # max number of RRT iterations


def samplegrasp(robot, cube, qguess):
    '''II.a  Return (q, placement): a random cube placement inside the sampling box and a
    valid grasp configuration for it.'''
    # TODO 1:
    #   loop until success:
    #     - random cube position: np.random.uniform(CUBE_LOW, CUBE_HIGH)
    #     - placement with no rotation: pin.SE3(np.eye(3), position)
    #     - q, success = computeqgrasppose(robot, qguess, cube, placement)
    #     - if success: return q, placement
    #   (import computeqgrasppose from inverse_geometry at the top of the file)
    while True:
        position = np.random.uniform(CUBE_LOW, CUBE_HIGH)
        placement = pin.SE3(np.eye(3), position)
        q, success = computeqgrasppose(robot, qguess, cube, placement)
        if success:
            return q, placement
    


def lerpplacement(P0, P1, t):
    '''Return the cube placement a fraction t (0..1) of the way from P0 to P1.'''
    # TODO 2:
    #   the cube never rotates, so only the translation changes:
    #   position = (1 - t) * P0.translation + t * P1.translation  (same idea as Tutorial 4's lerp)
    #   return pin.SE3(P0.rotation, position)
    position = P0.translation * (1- t) + P1.translation * t
    return pin.SE3(P0.rotation, position)

    


def projectedpath(robot, cube, qstart, Pstart, Pend):
    '''II.b  Move the cube from Pstart to Pend in small steps, re-solving the grasp at
    each step. Return (qs, success): qs = list of grasp configurations along the way
    (qstart excluded), success = False as soon as one step fails.'''
    # TODO 3:
    #   - number of steps: n = max(1, int(np.ceil(distance between the two positions / STEP)))
    #   - q = qstart
    #   - for i in 1..n:
    #       P = lerpplacement(Pstart, Pend, i / n)
    #       q, success = computeqgrasppose(robot, q, cube, P)    # start from the PREVIOUS q
    #       if not success: return qs, False                     # keep the valid part
    #       qs.append(q)
    #   - return qs, True
    #   Why: a straight line in q between two grasps does NOT keep the cube grasped
    #   (hands 7.2 cm apart instead of 10 cm halfway between q0 and qe).
    n = max(1, int(np.ceil(cubedistance(Pstart,Pend) / STEP)))
    q = qstart
    qs = []
    Ps = []
    for i in range(1, n+1):
        P = lerpplacement(Pstart,Pend, i/n)
        q, success = computeqgrasppose(robot,q, cube, P)
        if not success: return qs,Ps, False
        qs.append(q)
        Ps.append(P)
    return qs,Ps, True



def cubedistance(P1, P2):
    '''Distance between two cube placements (positions only, the cube never rotates).'''
    # TODO 4: norm of the difference of the two translations
    return np.linalg.norm(P2.translation - P1.translation)


def nearestvertex(G, P):
    '''Index of the node of G whose cube placement is closest to P (as in Tutorial 4).'''
    # TODO 5: same as your Tutorial 4 NEAREST_VERTEX, using cubedistance on node[2]
    #   node layout: G[i] = (parent index, q, cube placement, qs from parent to this node)
    min_dist = 10e4
    idx=-1
    for (i,node) in enumerate(G):
        dist = cubedistance(node[2],P) 
        if dist < min_dist:
            min_dist = dist
            idx = i
    return idx


def newconf(robot, cube, node, Prand):
    '''Grow from node towards Prand by at most DELTA. Return (q, placement, qs) of the
    furthest valid point reached, like NEW_CONF in Tutorial 4.'''
    # TODO 6:
    #   - Ptarget = Prand, or the point DELTA along the way if Prand is further than DELTA
    #     (lerpplacement with t = DELTA / distance)
    #   - qs, success = projectedpath(robot, cube, node q, node placement, Ptarget)
    #   - if qs is empty: nothing valid, return None
    #   - otherwise the new vertex is the LAST valid q in qs, with the matching placement
    #     (if success is False you need the placement of the last valid step: think about
    #      how to get it, e.g. have projectedpath also return the placements)
    Ptarget = Prand
    dist = cubedistance(node[2], Prand)
    if dist > DELTA:
        Ptarget = lerpplacement(node[2], Prand, DELTA / dist)
    qs, Ps, success = projectedpath(robot, cube, node[1], node[2], Ptarget)
    if len(qs) == 0:
        return None
    else:
        return qs[-1], Ps[-1], qs
    


def rrt(robot, cube, qinit, qgoal, Pinit, Pgoal):
    '''II.c  RRT in cube-placement space. Return (G, found).'''
    # TODO 7 (same loop as your Tutorial 4 rrt):
    #   G = [(None, qinit, Pinit, [])]
    #   for _ in range(K):
    #       - sample: q_rand, P_rand = samplegrasp(...)   (only P_rand is really needed)
    #       - i_near = nearestvertex(G, P_rand)
    #       - new = newconf(robot, cube, G[i_near], P_rand); skip if None
    #       - append (i_near, q_new, P_new, qs) to G
    #       - try to connect the new node to the goal: projectedpath(..., P_new, Pgoal)
    #         if it succeeds, append the goal node and return G, True
    #   return G, False
    #   Optional: every few iterations sample Pgoal itself ("goal bias") to finish faster.
    G = [(None, qinit, Pinit, [])]
    for _ in range(K):
        q_rand, P_rand = samplegrasp(robot, cube, G[-1][1])
        i_near = nearestvertex(G, P_rand)
        new = newconf(robot,cube, G[i_near], P_rand)
        if new is None:
            continue
        q_new, P_new, qs = new
        G.append((i_near, q_new, P_new, qs))
        qs_goal, _, success = projectedpath(robot, cube, q_new, P_new, Pgoal)
        if success:
            G.append((len(G)-1, qs_goal[-1], Pgoal, qs_goal))
            return G, True
    return G, False


def getpath(G):
    '''Return the full list of configurations from the start node to the last node of G.'''
    # Adapted from Tutorial 4's getpath. Each node is (parent index, q, cube placement, qs),
    # where qs = every grasp configuration from the parent to this node (ending with q).
    # A straight line in q between two nodes would NOT keep the cube grasped, so the
    # path must contain all these in-between grasps, not only the nodes.
    path = []                      # empty to start with
    node = G[-1]                   # start at the LAST node (the goal) and walk back to the start
    while node[0] is not None:     # node[0] is the parent index; the start node has None
        path = node[3] + path      # put this node's in-between grasps (parent -> node) at the FRONT
        node = G[node[0]]          # jump to the parent node
    path = [G[0][1]] + path        # the start node has no in-between grasps, so add its q by hand
    return path


#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(qinit,qgoal,cubeplacementq0, cubeplacementqgoal):
    robot, table, obstacle, cube = setuppinocchio()

    # 1. RRT in cube-placement space
    G, found = rrt(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal)
    if not found:
        print("warning: no path found after", K, "RRT iterations, returning [qinit, qgoal]")
        return [qinit, qgoal]

    # 2. all grasp configurations from qinit to the goal cube placement
    path = getpath(G)

    # 3. the RRT reaches the goal cube with a slightly different arm pose than qgoal
    qlast = path[-1]               
    nslide = 20
    for k in range(1, nslide + 1):
        guess = (1 - k / nslide) * qlast + (k / nslide) * qgoal
        q, success = computeqgrasppose(robot, guess, cube, cubeplacementqgoal)
        if not success:
            print("warning: could not slide onto qgoal, the path ends next to it")
            return path
        path.append(q)
    return path


def displaypath(robot,path,dt,viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)


def cubefromhand(robot, cube, q):
    '''Cube placement held by the robot in configuration q.
    The left hand sits on the left hook, so: cube = left hand * (cube -> left hook)^-1.'''
    from config import LEFT_HOOK
    # cube -> left hook offset, read from the cube model (constant). Do NOT derive it from the
    # current cube placement: that feeds rounding errors back in and they grow at every call.
    cMhook = cube.data.oMf[cube.model.getFrameId(LEFT_HOOK)]
    pin.framesForwardKinematics(robot.model, robot.data, q)
    oMhand = robot.data.oMf[robot.model.getFrameId(LEFT_HAND)]
    return oMhand * cMhook.inverse()


def displaypathwithcube(robot, cube, path, dt, viz):
    '''Like displaypath, but the cube moves with the hands.'''
    from setup_meshcat import updatevisuals
    from tools import setcubeplacement
    for q in path:
        setcubeplacement(robot, cube, cubefromhand(robot, cube, q))
        updatevisuals(viz, robot, cube, q)
        time.sleep(dt)


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose

    robot, cube, viz = setupwithmeshcat()


    q = robot.q0.copy()
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)

    if not(successinit and successend):
        print ("error: invalid initial or end configuration")

    path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    displaypathwithcube(robot, cube, path, dt=0.05, viz=viz)
