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
MAXJUMP = 0.3  # max change in q (rad) between two consecutive grasps along an edge

# NEW (wall margin): the robot and cube must stay WALLMARGIN away from the wall (obstacle),
# so that tracking errors and the Bezier smoothing in control.py do not touch it.
# The given start and goal grasps are themselves close to the wall (goal: ~2 mm), so the
# margin fades to 0 when the cube is within FADE of the start or the goal.
WALLMARGIN = 0.015  # m
FADE = 0.05         # m


def wallclearance(robot, q):
    '''NEW (wall margin): smallest distance (m) between the wall and the robot or the cube,
    for configuration q (the cube must already be placed, e.g. by computeqgrasppose).'''
    cm, cd = robot.collision_model, robot.collision_data
    wall = cm.getGeometryId('obstaclebase_0')
    pairs = [k for k, p in enumerate(cm.collisionPairs) if wall in (p.first, p.second)]
    pin.updateGeometryPlacements(robot.model, robot.data, cm, cd, q)
    return min(pin.computeDistance(cm, cd, k).min_distance for k in pairs)


def requiredclearance(P, ends):
    '''NEW (wall margin): clearance required when the cube is at placement P.
    ends = (start placement, goal placement): full WALLMARGIN far from both,
    fading linearly to 0 within FADE of either one.'''
    nearest = min(cubedistance(P, E) for E in ends)
    return WALLMARGIN * min(1.0, nearest / FADE)


def samplegrasp(robot, cube, qguess):
    '''II.a  Return (q, placement): a random cube placement inside the sampling box and a
    valid grasp configuration for it.'''
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
    #   the cube never rotates, so only the translation changes:
    #   position = (1 - t) * P0.translation + t * P1.translation  (same idea as Tutorial 4's lerp)
    #   return pin.SE3(P0.rotation, position)
    # used to move cube from A to B in 1cm steps
    position = P0.translation * (1- t) + P1.translation * t
    return pin.SE3(P0.rotation, position)

    


def projectedpath(robot, cube, qstart, Pstart, Pend, ends):   # NEW: ends, for the wall margin
    '''II.b  Move the cube from Pstart to Pend in small steps, re-solving the grasp at
    each step. Return (qs, Ps, success): qs = list of grasp configurations along the way
    (qstart excluded), Ps = the matching cube placements, success = False as soon as one
    step fails (qs and Ps then hold the valid part only).'''
    # move the cube from A to B

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
    n = max(1, int(np.ceil(cubedistance(Pstart, Pend) / STEP)))
    q = qstart
    qs, Ps = [], []
    for i in range(1, n + 1):
        P = lerpplacement(Pstart, Pend, i / n)
        qnext, success = computeqgrasppose(robot, q, cube, P)   # warm start from the previous q
        # a big change in q means IG jumped to a different arm solution: not a smooth path
        if not success or norm(qnext - q) > MAXJUMP:
            return qs, Ps, False                                  # keep the valid part
        # NEW (wall margin): too close to the wall counts as a failed step too
        if wallclearance(robot, qnext) < requiredclearance(P, ends):
            return qs, Ps, False
        q = qnext
        qs.append(q)
        Ps.append(P)
    return qs, Ps, True


def cubedistance(P1, P2):
    '''Distance between two cube placements (positions only, the cube never rotates).'''
    # norm of the difference of the two translations
    return np.linalg.norm(P2.translation - P1.translation)


def nearestvertex(G, P):
    '''Index of the node of G whose cube placement is closest to P (as in Tutorial 4).'''
    # same as your Tutorial 4 NEAREST_VERTEX, using cubedistance on node[2]
    #   node layout: G[i] = (parent index, q, cube placement, qs from parent to this node)
    distances = [cubedistance(node[2], P) for node in G]
    return int(np.argmin(distances))


def newconf(robot, cube, node, Prand, ends):   # NEW: ends, passed on to projectedpath
    # node is from nearestvertex
    '''Grow from node towards Prand by at most DELTA. Return (q, placement, qs) of the
    furthest valid point reached, like NEW_CONF in Tutorial 4.'''
    #   - Ptarget = Prand, or the point DELTA along the way if Prand is further than DELTA
    #     (lerpplacement with t = DELTA / distance)
    #   - qs, success = projectedpath(robot, cube, node q, node placement, Ptarget)
    #   - if qs is empty: nothing valid, return None
    #   - otherwise the new vertex is the LAST valid q in qs, with the matching placement
    #     (if success is False you need the placement of the last valid step: think about
    #      how to get it, e.g. have projectedpath also return the placements)

    q_near, P_near = node[1], node[2]  # P_near = the cube placement at that node. q_near = the robot configuration at that node: the joint angles that hold the cube there
    distance = cubedistance(P_near, Prand)
    if distance <= DELTA:
        Ptarget = Prand
    else:
        Ptarget = lerpplacement(P_near, Prand, DELTA / distance)

    # try to move there, 1cm at a time
    qs, Ps, _ = projectedpath(robot, cube, q_near, P_near, Ptarget, ends)

    if not qs: # not even the first step was valid
        return None
    
    return qs[-1], Ps[-1], qs   # robot config at the furthest point reached, cube placement there, every config along this edge
    

def rrt(robot, cube, qinit, qgoal, Pinit, Pgoal):
    '''II.c  RRT in cube-placement space. Return (G, found).'''
    #   G = [(None, qinit, Pinit, [])]
    #   for _ in range(K):
    #       - sample: q_rand, P_rand = samplegrasp(...)   (only P_rand is really needed)
    #       - i_near = nearestvertex(G, P_rand)
    #       - new = newconf(robot, cube, G[i_near], P_rand); skip if None
    #       - append (i_near, q_new, P_new, qs) to G
    #       - try to connect the new node to the goal: projectedpath(..., P_new, Pgoal)
    #         if it succeeds, append the goal node and return G, True
    #   return G, False
    
    G = [(None, qinit, Pinit, [])]   # start node: no parent, no edge
    ends = (Pinit, Pgoal)            # NEW (wall margin): the margin fades out near these two
    for _ in range(K):
        # direction to grow in: a random cube placement.
        # only the placement is needed (newconf re-solves the grasps along the edge),
        # so no IG here, unlike samplegrasp
        Prand = pin.SE3(np.eye(3), np.random.uniform(CUBE_LOW, CUBE_HIGH))  # (rotation, translation) of where the cube is

        # grow from the nearest node towards it
        i_near = nearestvertex(G, Prand)
        new = newconf(robot, cube, G[i_near], Prand, ends)
        if new is None:   # could not move at all: try another direction
            continue
        q_new, P_new, qs = new
        G.append((i_near, q_new, P_new, qs))

        # try to reach the goal in a straight line from the new node
        qs_goal, _, success = projectedpath(robot, cube, q_new, P_new, Pgoal, ends)
        if success:
            # goal node q is the grasp actually reached, not qgoal (avoids a jump at the end)
            G.append((len(G) - 1, qs_goal[-1], Pgoal, qs_goal))
            return G, True
    return G, False


def getpath(G):
    '''Return the full list of configurations from the start node to the last node of G.'''
    # like your Tutorial 4 getpath, but walk back from G[-1] and prepend each
    #   node's qs (the dense in-between grasps), not just its q. Finish with the start q.
    # start at the goal and follows the parents backwards until you reach the start (the node whose parent is None)

    path = []
    i = len(G) - 1                    # start at the goal node
    while G[i][0] is not None:        # until we reach the start node
        path = G[i][3] + path         # this edge's configs go on the front
        i = G[i][0]                   # step back to the parent
    path = [G[i][1]] + path           # finish with the start configuration (qinit)
    return path


#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(qinit,qgoal,cubeplacementq0, cubeplacementqgoal):
    # 1. own robot + cube for planning (signature must stay as these 4 arguments)
    robot, table, obstacle, cube = setuppinocchio()

    # 2. plan
    G, found = rrt(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal)

    # 3. return a path either way
    if not found:
        print("warning: computepath found no path, returning [qinit, qgoal] (not a valid path)")
        return [qinit, qgoal]
    path = getpath(G)

    # 4. NEW (from the control-wip version): the RRT reaches the goal cube placement with a
    # slightly different arm pose than qgoal. Slide onto qgoal by re-solving the grasp at the
    # goal placement from guesses moving from the last configuration to qgoal.
    qlast = path[-1]
    nslide = 20
    for k in range(1, nslide + 1):
        guess = (1 - k / nslide) * qlast + (k / nslide) * qgoal
        q, success = computeqgrasppose(robot, guess, cube, cubeplacementqgoal)
        if not success or norm(q - path[-1]) > MAXJUMP:
            print("warning: could not slide onto qgoal, the path ends next to it")
            return path
        path.append(q)
    return path


def displaypath(robot,path,dt,viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)


def cubebetweenhands(robot, q):
    '''Cube placement held by configuration q: the cube centre sits halfway between the hands.'''
    idL = robot.model.getFrameId(LEFT_HAND)
    idR = robot.model.getFrameId(RIGHT_HAND)
    pin.framesForwardKinematics(robot.model, robot.data, q)
    mid = (robot.data.oMf[idL].translation + robot.data.oMf[idR].translation) / 2
    return pin.SE3(np.eye(3), mid)


def checkpath(robot, cube, path):
    '''Print whether every configuration in path is a valid grasp: no collision,
    within joint limits, hands holding the cube, and no big jumps. Return True if all pass.'''
    from tools import collision, jointlimitsviolated, setcubeplacement
    idL = robot.model.getFrameId(LEFT_HAND)
    idR = robot.model.getFrameId(RIGHT_HAND)
    collisions, limits, gaps = 0, 0, []
    for q in path:
        # put the cube between the hands first, or collision() checks a stale cube position
        setcubeplacement(robot, cube, cubebetweenhands(robot, q))
        collisions += collision(robot, q)
        limits += jointlimitsviolated(robot, q)
        gaps.append(norm(robot.data.oMf[idL].translation - robot.data.oMf[idR].translation))
    jumps = max(norm(a - b) for a, b in zip(path[:-1], path[1:]))
    print(f"{len(path)} configs | collisions: {collisions} | joint limit violations: {limits}")
    print(f"hand gap: {min(gaps):.3f}-{max(gaps):.3f} m (should be 0.100) | biggest jump: {jumps:.3f} rad")
    ok = len(path) > 2 and collisions == 0 and limits == 0 \
        and abs(min(gaps) - 0.1) < 1e-3 and abs(max(gaps) - 0.1) < 1e-3 and jumps <= MAXJUMP
    print("PATH VALID" if ok else "PATH INVALID")
    return ok


def displaypathwithcube(robot, cube, path, dt, viz):
    '''Like displaypath, but moves the cube with the hands so you can see it being carried.'''
    from setup_meshcat import updatevisuals
    from tools import setcubeplacement
    for q in path:
        setcubeplacement(robot, cube, cubebetweenhands(robot, q))
        updatevisuals(viz, robot, cube, q)
        time.sleep(dt)


if __name__ == "__main__":
    # python path.py         -> plan and check the path (no meshcat needed)
    # python path.py viz     -> also play it in meshcat (start meshcat-server first)
    # python path.py seeds   -> plan and check with 10 different random seeds
    import sys
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "viz":
        from tools import setupwithmeshcat
        robot, cube, viz = setupwithmeshcat()
    else:
        robot, table, obstacle, cube = setuppinocchio()

    q = robot.q0.copy()
    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT)
    qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET)
    if not (successinit and successend):
        print("error: invalid initial or end configuration")

    if mode == "seeds":
        results = []
        for seed in range(10):
            np.random.seed(seed)
            print(f"\nseed {seed}:")
            path = computepath(q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
            results.append(checkpath(robot, cube, path))
        print(f"\n{sum(results)}/10 valid paths")
    else:
        path = computepath(q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
        checkpath(robot, cube, path)
        if mode == "viz":
            displaypathwithcube(robot, cube, path, dt=0.05, viz=viz)