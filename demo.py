#!/usr/bin/env python3
'''Demo scenes for the extensions video. Not needed for marking: control.py is the main run.

usage:  python demo.py rotation        [seed]   cube turned at the start and at the goal
        python demo.py shortcut        [seed]   RRT path (grey) vs shortcut path (orange), then the run
        python demo.py force-fixed     [seed]   4 kg cube with a fixed squeeze
        python demo.py force-feedback  [seed]   4 kg cube with force feedback

Uses the same controller, trajectory and planner as control.py and path.py. The only additions
are a larger window, a green box at the target and (shortcut) the two paths; results are printed
in the terminal.'''

import sys
import numpy as np
import pinocchio as pin
import pybullet as pyb

import control
import path as planner
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, DT, LEFT_HAND, RIGHT_HAND
from tools import setupwithpybullet, rununtil
from inverse_geometry import computeqgrasppose, graspik
from setup_pinocchio import setuppinocchio

T = 6.            # motion time, as in control.py
SETTLE = 1.5      # seconds shown after the motion ends
HEAVY_MASS = 4.   # kg, for the force scenes (the normal cube is 0.14 kg)
WINDOW = (1600, 1000)   # pixels

GREY, ORANGE, GREEN = [.6, .6, .6], [1., .5, 0.], [0., .8, 0.]


# ---------------------------------------------------------------- set-up and drawing

def setup():
    '''Robot, simulator and cube, with the same model corrections as control.py's main.'''
    connect = pyb.connect                       # the template opens the window with pyb.connect(pyb.GUI):
    def bigwindow(mode, *args, **kwargs):       # pass it a window size, without editing the template
        if mode == pyb.GUI:
            kwargs.setdefault("options", f"--width={WINDOW[0]} --height={WINDOW[1]}")
        return connect(mode, *args, **kwargs)
    pyb.connect = bigwindow
    robot, sim, cube = setupwithpybullet()
    pyb.connect = connect
    for name in (LEFT_HAND, RIGHT_HAND):   # 1 kg hand tips that pybullet adds (see control.py)
        frame = robot.model.frames[robot.model.getFrameId(name)]
        joint = frame.parentJoint if hasattr(frame, "parentJoint") else frame.parent
        robot.model.inertias[joint] += frame.placement.act(pin.Inertia(1., np.zeros(3), np.zeros((3, 3))))
    robot.data = robot.model.createData()
    control.damping[:] = [sim.p.getJointInfo(sim.robot, i)[6] for i in sim.bulletCtrlJointsInPinOrder]
    return robot, sim, cube


def placement(position, yaw=0.):
    return pin.SE3(pin.rpy.rpyToMatrix(0., 0., yaw), np.array(position, dtype=float))


def drawbox(P, colour, half=0.05):
    '''Wireframe cube of side 2*half at placement P (used for the target).'''
    corners = [P.act(np.array([sx, sy, sz]) * half) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    for a in range(8):
        for b in range(a + 1, 8):
            if bin(a ^ b).count("1") == 1:          # corners that differ in one coordinate
                pyb.addUserDebugLine(corners[a], corners[b], colour, lineWidth=2)


def drawtrace(points, colour, radius):
    '''A path drawn as a chain of small spheres (pybullet on macOS ignores line widths, so debug
    lines are always 1 pixel). Visual only: no mass and no collision shape.'''
    dots = [points[0]]
    for a, b in zip(points[:-1], points[1:]):     # a sphere every radius along each segment
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / radius)))
        dots += [a + (b - a) * k / n for k in range(1, n + 1)]
    shape = pyb.createVisualShape(pyb.GEOM_SPHERE, radius=radius, rgbaColor=list(colour) + [1.])
    pyb.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, batchPositions=[list(p) for p in dots])


def cubetrace(robot, cube, path):
    '''Cube centre along a path (between the hands).'''
    return [planner.cubebetweenhands(robot, cube, q).translation for q in path]


def length(points):
    return sum(np.linalg.norm(b - a) for a, b in zip(points[:-1], points[1:]))


def cubepose(sim):
    pos, quat = pyb.getBasePositionAndOrientation(sim.cubeId)
    return np.array(pos), pyb.getEulerFromQuaternion(quat)[2]


def grip(sim):
    '''Measured squeeze and friction per hand (N), summed over the hand-cube contacts.'''
    press = friction = 0.
    for c in pyb.getContactPoints(sim.robot, sim.cubeId):
        press += c[9]
        friction += np.hypot(c[10], c[12])
    return press / 2, friction / 2


# ---------------------------------------------------------------- planning and running

def grasps(robot, cube, Pstart, Pgoal):
    q0, ok0 = computeqgrasppose(robot, robot.q0, cube, Pstart)
    qe, oke = computeqgrasppose(robot, robot.q0, cube, Pgoal)
    if not (ok0 and oke):
        sys.exit("no valid grasp for the start or goal placement")
    return q0, qe


def plancompare(q0, qe, Pstart, Pgoal):
    '''The steps of path.computepath, also returning the RRT's own path for comparison.'''
    robot, _, _, cube = setuppinocchio()
    G, found = planner.rrt(robot, cube, q0, qe, Pstart, Pgoal)
    if not found:
        sys.exit("planning failed: try another seed")
    original = planner.getpath(G)
    short = planner.shortcut(robot, cube, G, (Pstart, Pgoal)) or original
    final = list(short)
    for k in range(1, 21):                         # slide onto qe, as in path.computepath
        q, ok = graspik(robot, (1 - k / 20) * final[-1] + (k / 20) * qe, cube, Pgoal)
        if not ok or np.linalg.norm(q - final[-1]) > planner.MAXJUMP:
            break
        final.append(q)
    return robot, cube, original, short, final


def run(sim, robot, cube, q0, path, Pstart, ontick=None):
    '''Reset the robot, cube and controller, then track the path. Returns the cube's end pose.'''
    sim.setqsim(q0)
    xyzquat = pin.SE3ToXYZQUAT(Pstart)
    pyb.resetBasePositionAndOrientation(sim.cubeId, xyzquat[:3], xyzquat[3:])
    pyb.resetBaseVelocity(sim.cubeId, [0, 0, 0], [0, 0, 0])
    control.ie = np.zeros(15)
    control.press = control.friction = control.press_error_sum = 0.

    q_t, v_t, a_t = control.maketrajbezier(path, T, robot, n=15)
    trajs = (lambda t: q_t(min(t, T)), lambda t: v_t(min(t, T)), lambda t: a_t(min(t, T)))   # hold the end
    t, k = 0., 0
    while t < T + SETTLE:
        rununtil(control.controllaw, DT, sim, robot, trajs, t, cube)
        if ontick and k % 50 == 0:                 # every 50 ms
            ontick(t)
        t += DT
        k += 1
    return cubepose(sim)


def result(sim, Pgoal):
    pos, yaw = cubepose(sim)
    err = np.linalg.norm(pos - Pgoal.translation) * 100
    ok = err < 5. and pos[2] > 0.9
    print(f"Cube {'delivered' if ok else 'NOT delivered'}: {err:.1f} cm from the target")
    return ok


# ---------------------------------------------------------------- scenes

def rotation():
    Pstart = placement(CUBE_PLACEMENT.translation, yaw=0.3)
    Pgoal = placement(CUBE_PLACEMENT_TARGET.translation, yaw=-0.3)
    robot, sim, cube = setup()
    q0, qe = grasps(robot, cube, Pstart, Pgoal)
    path = planner.computepath(q0, qe, Pstart, Pgoal)
    drawbox(Pgoal, GREEN)
    run(sim, robot, cube, q0, path, Pstart)
    print(f"cube angle: start {np.degrees(0.3):+.1f} deg, goal {np.degrees(-0.3):+.1f} deg, "
          f"end {np.degrees(cubepose(sim)[1]):+.1f} deg")
    result(sim, Pgoal)
    input("press Enter to close")


def shortcut():
    robot, sim, cube = setup()
    q0, qe = grasps(robot, cube, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    rp, cp, original, short, final = plancompare(q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    before, after = cubetrace(rp, cp, original), cubetrace(rp, cp, short)
    print(f"cube path: RRT (grey) {length(before)*100:.0f} cm, shortcut (orange) {length(after)*100:.0f} cm, "
          f"{(1 - length(after) / length(before)) * 100:.0f}% shorter")
    drawbox(CUBE_PLACEMENT_TARGET, GREEN)
    drawtrace(before, GREY, 0.005)
    drawtrace(after, ORANGE, 0.007)
    run(sim, robot, cube, q0, final, CUBE_PLACEMENT)
    result(sim, CUBE_PLACEMENT_TARGET)
    input("press Enter to close")


def force(feedback):
    '''One run per process: a run after a dropped heavy cube in the same simulation is unreliable.'''
    robot, sim, cube = setup()
    q0, qe = grasps(robot, cube, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    path = planner.computepath(q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    pyb.changeDynamics(sim.cubeId, -1, mass=HEAVY_MASS)
    drawbox(CUBE_PLACEMENT_TARGET, GREEN)
    control.FORCE_FEEDBACK = feedback

    peak = {"press": 0., "used": 0.}
    def track(t):
        press, friction = grip(sim)
        peak["press"] = max(peak["press"], press)
        if press > 1e-6:
            peak["used"] = max(peak["used"], friction / (control.MU * press) * 100)

    run(sim, robot, cube, q0, path, CUBE_PLACEMENT, ontick=track)
    mode = "force feedback" if feedback else f"fixed squeeze ({control.S:.0f} N)"
    print(f"{HEAVY_MASS:.0f} kg cube, {mode}: peak squeeze {peak['press']:.0f} N per hand, "
          f"peak friction used {peak['used']:.0f}% of the limit")
    result(sim, CUBE_PLACEMENT_TARGET)
    input("press Enter to close")


if __name__ == "__main__":
    scenes = {"rotation": rotation, "shortcut": shortcut,
              "force-fixed": lambda: force(False), "force-feedback": lambda: force(True)}
    if len(sys.argv) < 2 or sys.argv[1] not in scenes:
        sys.exit(__doc__)
    np.random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    scenes[sys.argv[1]]()
