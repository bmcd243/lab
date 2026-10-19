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
    pass


def cubedistance(P1, P2):
    '''Distance between two cube placements (positions only, the cube never rotates).'''
    # TODO 4: norm of the difference of the two translations
    return np.linalg.norm(P2.translation - P1.translation)


def nearestvertex(G, P):
    '''Index of the node of G whose cube placement is closest to P (as in Tutorial 4).'''
    # TODO 5: same as your Tutorial 4 NEAREST_VERTEX, using cubedistance on node[2]
    #   node layout: G[i] = (parent index, q, cube placement, qs from parent to this node)
    


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
    pass


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
    pass


def getpath(G):
    '''Return the full list of configurations from the start node to the last node of G.'''
    # TODO 8: like your Tutorial 4 getpath, but walk back from G[-1] and prepend each
    #   node's qs (the dense in-between grasps), not just its q. Finish with the start q.
    pass


#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(qinit,qgoal,cubeplacementq0, cubeplacementqgoal):
    # TODO 9:
    #   - add robot and cube as extra parameters (allowed by the marking scheme) and update
    #     the two calls: __main__ below and control.py
    #   - G, found = rrt(...); if found: return getpath(G)
    #   - if not found: still return something that runs (e.g. [qinit, qgoal]) and print a warning
    #   - optional later: shortcut the path (Tutorial 4) using projectedpath for each shortcut
    return [qinit, qgoal]


def displaypath(robot,path,dt,viz):
    for q in path:
        viz.display(q)
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

    # TODO 10: check the path before trusting it: for every q, collision(robot, q) is False,
    #   jointlimitsviolated(robot, q) is False, and both hands are on the hooks.
    #   Note: viz.display(q) moves the robot but not the cube. To see the cube travel, store the
    #   placements too and call setcubeplacement + updatevisuals for each step.

    displaypath(robot,path,dt=0.5,viz=viz) #you ll probably want to lower dt
