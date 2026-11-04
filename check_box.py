'''Sanity checks for the cube sampling box (CUBE_LOW / CUBE_HIGH in path.py).
Run from the lab folder:  python check_box.py   (no meshcat needed)'''

import numpy as np
import pinocchio as pin
from numpy.linalg import norm

from setup_pinocchio import setuppinocchio
from inverse_geometry import computeqgrasppose
from tools import collision, jointlimitsviolated
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
from path import CUBE_LOW, CUBE_HIGH

robot, table, obstacle, cube = setuppinocchio()
q0, ok0 = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT)
print("box low :", CUBE_LOW, "\nbox high:", CUBE_HIGH)
print("start grasp ok:", ok0)


def placement(p):
    return pin.SE3(np.eye(3), np.asarray(p, dtype=float))


def why(q, ok, robot):
    '''Short reason for a grasp result.'''
    if ok: return "ok"
    if collision(robot, q): return "collision"
    if jointlimitsviolated(robot, q): return "joint limits"
    return "IG not converged (unreachable?)"


# ---- 1. are start and goal inside the box?
print("\n1. start / goal inside the box")
for name, P in [("start", CUBE_PLACEMENT), ("goal", CUBE_PLACEMENT_TARGET)]:
    p = P.translation
    inside = np.all(p >= CUBE_LOW) and np.all(p <= CUBE_HIGH)
    print(f"  {name} {p} inside: {inside}")

# ---- 2. success rate per height layer, plus reasons for failures
print("\n2. random samples per height layer (warm start = start grasp)")
np.random.seed(0)
zs = np.linspace(CUBE_LOW[2], CUBE_HIGH[2], 7)
for zlo, zhi in zip(zs[:-1], zs[1:]):
    reasons = {}
    for _ in range(40):
        p = np.random.uniform([CUBE_LOW[0], CUBE_LOW[1], zlo], [CUBE_HIGH[0], CUBE_HIGH[1], zhi])
        q, ok = computeqgrasppose(robot, q0, cube, placement(p))
        r = why(q, ok, robot)
        reasons[r] = reasons.get(r, 0) + 1
    print(f"  z {zlo:.3f}-{zhi:.3f}: ok {reasons.get('ok', 0)}/40   {reasons}")

# ---- 3. does the box contain an actual route? hand-made path:
#      lift at start -> move across over the wall at height z -> lower at goal
print("\n3. hand-made route start -> up -> across -> down -> goal, for each height")
STEP, MAXJUMP = 0.01, 0.3
ps, pg = CUBE_PLACEMENT.translation, CUBE_PLACEMENT_TARGET.translation


def follow(waypoints):
    q = q0.copy()
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        n = max(1, int(np.ceil(norm(b - a) / STEP)))
        for i in range(1, n + 1):
            p = a + (b - a) * i / n
            qn, ok = computeqgrasppose(robot, q, cube, placement(p))
            if not ok:
                return False, f"fails at {np.round(p, 3)}: {why(qn, ok, robot)}"
            if norm(qn - q) > MAXJUMP:
                return False, f"IG jump at {np.round(p, 3)} ({norm(qn - q):.2f} rad)"
            q = qn
    return True, "ok"


for z in np.arange(1.00, CUBE_HIGH[2] + 1e-9, 0.04):
    route = [ps, np.array([ps[0], ps[1], z]), np.array([pg[0], pg[1], z]), pg]
    ok, msg = follow(route)
    print(f"  over at z={z:.2f}: {msg}")
