#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv, norm
from inverse_geometry import computeqgrasppose, graspik
from setup_pinocchio import setuppinocchio
from config import LEFT_HAND, RIGHT_HAND
import time

# RRT adapted from Tutorial 4 (ediadvancedrobotics/tutorials, 4_rrt.ipynb), planning for the
# CUBE position: every configuration comes from computeqgrasppose, so the cube stays grasped.

# box where cube positions are sampled (x, y, z); z goes above the wall top (1.00 m)
CUBE_LOW  = np.array([0.30, -0.40, 0.93])
CUBE_HIGH = np.array([0.45,  0.20, 1.20])
DELTA = 0.10   # max distance (m) the cube moves along one new edge
STEP  = 0.01   # distance (m) between two re-solved grasps along an edge
K     = 500    # max number of RRT iterations
MAXJUMP = 0.3  # max change in q (rad) between two consecutive grasps

# keep the robot and cube WALLMARGIN away from the wall, so tracking errors and the Bezier
# smoothing do not touch it; the start and goal grasps are close to the wall (goal ~2 mm),
# so the margin fades to 0 within FADE of either
WALLMARGIN = 0.015  # m
FADE = 0.05         # m


def wallclearance(robot, q):
    '''Smallest distance (m) between the wall and the robot or the (already placed) cube.'''
    cm, cd = robot.collision_model, robot.collision_data
    wall = cm.getGeometryId('obstaclebase_0')
    pairs = [k for k, p in enumerate(cm.collisionPairs) if wall in (p.first, p.second)]
    pin.updateGeometryPlacements(robot.model, robot.data, cm, cd, q)
    return min(pin.computeDistance(cm, cd, k).min_distance for k in pairs)


def requiredclearance(P, ends):
    '''Wall clearance required with the cube at P; ends = (start placement, goal placement).'''
    nearest = min(cubedistance(P, E) for E in ends)
    return WALLMARGIN * min(1.0, nearest / FADE)


def samplegrasp(robot, cube, qguess):
    '''II.a  A random cube placement inside the sampling box and a valid grasp for it.'''
    while True:
        position = np.random.uniform(CUBE_LOW, CUBE_HIGH)
        placement = pin.SE3(np.eye(3), position)
        q, success = computeqgrasppose(robot, qguess, cube, placement)
        if success:
            return q, placement


def lerpplacement(P0, P1, t):
    '''Cube placement a fraction t (0..1) of the way from P0 to P1: straight line for the
    position, constant-speed turn for the rotation.'''
    position = P0.translation * (1- t) + P1.translation * t
    rotation = P0.rotation @ pin.exp3(t * pin.log3(P0.rotation.T @ P1.rotation))
    return pin.SE3(rotation, position)


def randomplacement(Pinit, Pgoal):
    '''Random cube placement in the sampling box, turned about the vertical axis by an angle
    between the start and goal angles.'''
    yaws = [pin.rpy.matrixToRpy(P.rotation)[2] for P in (Pinit, Pgoal)]
    yaw = np.random.uniform(min(yaws), max(yaws))
    return pin.SE3(pin.rpy.rpyToMatrix(0., 0., yaw), np.random.uniform(CUBE_LOW, CUBE_HIGH))


def projectedpath(robot, cube, qstart, Pstart, Pend, ends):
    '''II.b  Move the cube from Pstart to Pend in STEP increments, re-solving the grasp each
    time. Return (qs, Ps, success); on failure qs and Ps hold the valid part only.
    (A straight line in q between two grasps does not keep the cube grasped.)'''
    n = max(1, int(np.ceil(cubedistance(Pstart, Pend) / STEP)))
    q = qstart
    qs, Ps = [], []
    for i in range(1, n + 1):
        P = lerpplacement(Pstart, Pend, i / n)
        qnext, success = graspik(robot, q, cube, P)   # one IK attempt, warm-started from the previous q
        # a big jump in q means IK switched to a different arm pose
        if not success or norm(qnext - q) > MAXJUMP:
            return qs, Ps, False
        if wallclearance(robot, qnext) < requiredclearance(P, ends):
            return qs, Ps, False
        q = qnext
        qs.append(q)
        Ps.append(P)
    return qs, Ps, True


def cubedistance(P1, P2):
    '''Distance between two cube placements: position (m) + 0.1 m per radian of rotation.'''
    angle = np.linalg.norm(pin.log3(P1.rotation.T @ P2.rotation))
    return np.linalg.norm(P2.translation - P1.translation) + 0.1 * angle


def nearestvertex(G, P):
    '''Index of the node whose cube placement is closest to P.
    Node layout: (parent index, q, cube placement, qs from the parent to this node).'''
    distances = [cubedistance(node[2], P) for node in G]
    return int(np.argmin(distances))


def newconf(robot, cube, node, Prand, ends):
    '''Grow from node towards Prand by at most DELTA. Return (q, placement, qs) of the
    furthest valid point reached, or None, like NEW_CONF in Tutorial 4.'''
    q_near, P_near = node[1], node[2]
    distance = cubedistance(P_near, Prand)
    if distance <= DELTA:
        Ptarget = Prand
    else:
        Ptarget = lerpplacement(P_near, Prand, DELTA / distance)

    qs, Ps, _ = projectedpath(robot, cube, q_near, P_near, Ptarget, ends)
    if not qs:
        return None
    return qs[-1], Ps[-1], qs


def rrt(robot, cube, qinit, qgoal, Pinit, Pgoal):
    '''II.c  RRT in cube-placement space. Return (G, found).'''
    G = [(None, qinit, Pinit, [])]
    ends = (Pinit, Pgoal)
    for _ in range(K):
        # random direction to grow in (no IK needed here: newconf solves the grasps)
        Prand = randomplacement(Pinit, Pgoal)

        i_near = nearestvertex(G, Prand)
        new = newconf(robot, cube, G[i_near], Prand, ends)
        if new is None:
            continue
        q_new, P_new, qs = new
        G.append((i_near, q_new, P_new, qs))

        # try to reach the goal in a straight line from the new node
        qs_goal, _, success = projectedpath(robot, cube, q_new, P_new, Pgoal, ends)
        if success:
            G.append((len(G) - 1, qs_goal[-1], Pgoal, qs_goal))
            return G, True
    return G, False


def rrtconnect(robot, cube, qinit, qgoal, Pinit, Pgoal):
    '''Extension: RRT-Connect (Lecture 6). One tree grows from the start and one from the goal;
    after each extension the other tree tries to reach the new node in a straight line.
    When they meet, the cube's route (start branch + goal branch) is re-solved from qinit in one
    direction, so the arm poses join without jumps. Returns (G, found) in the same format as rrt.
    Not used by computepath: on this scene it was slower than rrt (see README).'''
    ends = (Pinit, Pgoal)
    A, B = [(None, qinit, Pinit, [])], [(None, qgoal, Pgoal, [])]
    for _ in range(K):
        Prand = randomplacement(Pinit, Pgoal)
        i = nearestvertex(A, Prand)
        new = newconf(robot, cube, A[i], Prand, ends)
        if new is not None:
            A.append((i, new[0], new[1], new[2]))
            j = nearestvertex(B, new[1])
            qs, Ps, ok = projectedpath(robot, cube, B[j][1], B[j][2], new[1], ends)
            if not ok and qs:                       # keep the part of the connection that worked
                B.append((j, qs[-1], Ps[-1], qs))
            if ok:
                start, goal = (A, B) if A[0][2] is Pinit else (B, A)
                a, b = (len(A) - 1, j) if start is A else (j, len(A) - 1)
                route = []                          # cube placements from Pinit to Pgoal
                while a is not None:
                    route.insert(0, start[a][2]); a = start[a][0]
                while b is not None:
                    route.append(goal[b][2]); b = goal[b][0]
                G, q = [(None, qinit, Pinit, [])], qinit
                for P0, P1 in zip(route[:-1], route[1:]):
                    qs, _, ok = projectedpath(robot, cube, q, P0, P1, ends)
                    if not ok:
                        break
                    G.append((len(G) - 1, qs[-1], P1, qs)); q = qs[-1]
                if ok:
                    return G, True
        A, B = B, A                                 # alternate which tree grows
    return [(None, qinit, Pinit, [])], False


def getpath(G):
    '''All configurations from the start node to the last node of G (as in Tutorial 4,
    but each edge contributes its intermediate grasps, not just its end).'''
    path = []
    i = len(G) - 1
    while G[i][0] is not None:
        path = G[i][3] + path
        i = G[i][0]
    path = [G[i][1]] + path
    return path


def shortcut(robot, cube, G, ends):
    '''Extension: path shortcutting. From each tree node on the path, move the cube straight to
    the furthest later node that is still reachable (projectedpath: grasp, collisions, wall
    margin), skipping the RRT's detours. Returns the new configurations, or None if a segment
    fails when re-projected (the caller then keeps the original path).'''
    nodes = []                                  # tree nodes from the start to the goal
    i = len(G) - 1
    while i is not None:
        nodes.insert(0, G[i])
        i = G[i][0]

    q, path, i = nodes[0][1], [nodes[0][1]], 0
    while i < len(nodes) - 1:
        for j in range(len(nodes) - 1, i, -1):  # furthest node first
            qs, _, ok = projectedpath(robot, cube, q, nodes[i][2], nodes[j][2], ends)
            if ok:
                break
        if not ok:
            return None
        path += qs
        q, i = qs[-1], j
    return path


#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(qinit,qgoal,cubeplacementq0, cubeplacementqgoal):
    robot, table, obstacle, cube = setuppinocchio()

    G, found = rrt(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal)
    if not found:
        print("warning: computepath found no path, returning [qinit, qgoal] (not a valid path)")
        return [qinit, qgoal]
    path = shortcut(robot, cube, G, (cubeplacementq0, cubeplacementqgoal)) or getpath(G)

    # the RRT reaches the goal placement with a slightly different arm pose than qgoal:
    # slide onto qgoal by re-solving the grasp from guesses moving towards qgoal
    qlast = path[-1]
    nslide = 20
    for k in range(1, nslide + 1):
        guess = (1 - k / nslide) * qlast + (k / nslide) * qgoal
        q, success = graspik(robot, guess, cube, cubeplacementqgoal)
        if not success or norm(q - path[-1]) > MAXJUMP:
            print("warning: could not slide onto qgoal, the path ends next to it")
            return path
        path.append(q)
    return path


def displaypath(robot,path,dt,viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)


def cubebetweenhands(robot, cube, q):
    '''Cube placement held by configuration q: halfway between the hands, turned like the hands.'''
    from config import LEFT_HOOK
    idL = robot.model.getFrameId(LEFT_HAND)
    idR = robot.model.getFrameId(RIGHT_HAND)
    pin.framesForwardKinematics(robot.model, robot.data, q)
    pin.framesForwardKinematics(cube.model, cube.data, cube.q0)       # hook offsets in the cube frame
    cMhook = cube.data.oMf[cube.model.getFrameId(LEFT_HOOK)]
    mid = (robot.data.oMf[idL].translation + robot.data.oMf[idR].translation) / 2
    return pin.SE3((robot.data.oMf[idL] * cMhook.inverse()).rotation, mid)


def checkpath(robot, cube, path):
    '''Print whether every configuration is a valid grasp: no collision, within joint limits,
    hands holding the cube, no big jumps. Return True if all pass.'''
    from tools import collision, jointlimitsviolated, setcubeplacement
    idL = robot.model.getFrameId(LEFT_HAND)
    idR = robot.model.getFrameId(RIGHT_HAND)
    collisions, limits, gaps = 0, 0, []
    for q in path:
        setcubeplacement(robot, cube, cubebetweenhands(robot, cube, q))   # else collision() uses a stale cube
        collisions += collision(robot, q)
        limits += jointlimitsviolated(robot, q)
        gaps.append(norm(robot.data.oMf[idL].translation - robot.data.oMf[idR].translation))
    jumps = max(norm(a - b) for a, b in zip(path[:-1], path[1:]))
    print(f"{len(path)} configs | collisions: {collisions} | joint limit violations: {limits}")
    print(f"hand gap: {min(gaps):.3f}-{max(gaps):.3f} m (should be 0.100) | biggest jump: {jumps:.3f} rad")
    ok = len(path) > 2 and collisions == 0 and limits == 0 \
        and abs(min(gaps) - 0.1) < 2e-3 and abs(max(gaps) - 0.1) < 2e-3 and jumps <= MAXJUMP
    print("PATH VALID" if ok else "PATH INVALID")
    return ok


def displaypathwithcube(robot, cube, path, dt, viz):
    '''Like displaypath, but the cube moves with the hands.'''
    from setup_meshcat import updatevisuals
    from tools import setcubeplacement
    for q in path:
        setcubeplacement(robot, cube, cubebetweenhands(robot, cube, q))
        updatevisuals(viz, robot, cube, q)
        time.sleep(dt)


if __name__ == "__main__":
    # python path.py         -> plan, check and play the path in meshcat (start meshcat-server first)
    # python path.py check   -> plan and check only (no meshcat needed)
    # python path.py seeds   -> plan and check with 10 different random seeds
    import sys
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

    mode = sys.argv[1] if len(sys.argv) > 1 else "viz"
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
