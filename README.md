# ARO lab submission

**Software:** the lab's `requirements.txt` only. `inverse_geometry.py` and `path.py` need `meshcat-server`; `python path.py check` validates a path without it.

## Inverse geometry (`computeqgrasppose`)
Iterative IK on both hands' 6D error, left hand first and right hand in its null space (Tutorial 3); stacking both hands gave the same poses. IK finds the grasp nearest its start, which can hit the wall even when another grasp exists, so we retry from up to 20 other starting guesses (16 -> 21 of 32 tested table positions grasped). The planner uses the single-attempt `graspik`, since its steps are meant to fail at the wall.

## Motion planning (`path.py`)
An RRT (Tutorial 4) over **cube placements**, not q: interpolating q between two grasps loses the cube (hands 7.2 cm apart instead of 10). Edges move the cube in 1 cm steps, re-solving the grasp each time; joint jumps over 0.3 rad are rejected (IK switched arm pose). The path ends by sliding onto qe.

## Trajectory (`maketrajpath`, `maketrajbezier`)
We started with level 0 (smoothstep, linear interpolation) and moved to level 1 because every path corner was a velocity jump. Level 1 fits one Bezier curve to the arc-length-timed path with a QP: first/last three control points fixed to q0/qe (rest to rest), all control points within joint limits. We chose 16 control points: with 11 control points the curve missed the path by up to 0.21 rad, with 16 by 0.09 rad, and with 31 it oscillated into the chest.

## Control (`controllaw`)
Inverse dynamics with PID plus pybullet's joint damping. For the grasp, each hand also adds the torque Jᵀf, where J is the hand's Jacobian and f is a spring pulling the hand onto its hook plus a squeeze towards the other hand.

**Hand masses:** the hand-tip links have no `<inertial>` tag in the URDF. Pinocchio treats them as massless, but pybullet gives them 1 kg each, so inverse dynamics underestimated gravity: with PD alone the hands ended 19–28 cm low. We first set these masses to zero in pybullet; we now add them to the pinocchio model so the controller compensates them. The integral term would also remove the sag, but would only hide the model error.

## Extensions
Each fixes a problem we measured (10 seeds unless stated).
1. **Wall clearance** (`requiredclearance`). RRT paths passed 0.1-4.6 mm from the wall and the arm touched it in 4/10 runs. Steps now keep 15 mm, fading to 0 near the start and goal because the goal grasp is 2 mm from the wall. Result: 10/10 delivered, no contact.
2. **Path shortcutting** (`shortcut`). RRT paths wander and the Bezier fit cuts their corners. Jumping to the furthest reachable node gives 18% shorter paths and a closer fit (largest miss 0.18 -> 0.12 rad). It needs the wall clearance (without it 2/10 touched).
3. **Cube rotation** (`lerpplacement`, `randomplacement`). The planner kept the start orientation: a rotated goal was reached 17 degrees off, a rotated start found no path. Placements now interpolate yaw and samples draw a yaw between start and goal: six combinations (up to 0.5 rad) give 10/10 valid paths and the cube arrives with 0 degrees error.
4. **Force feedback** (`gripforce`, Lecture 11). With a fixed squeeze, friction reached its limit during the lift. A PI loop on the measured contact forces aims to keep it at 60% of the limit. In a test with a 4 kg cube (about 29× the original), the fixed squeeze let it slip out of the hands, while the feedback loop held it all the way to the target.
5. **RRT-Connect, tried and dropped** (`rrtconnect`, Lecture 6): slower than our RRT (10 s vs 2.4 s mean over 20 seeds, one failure), as our RRT already tries the goal from every node.

**Limits:** only yaw rotations are handled; some rotated goals (e.g. +0.3 rad) have no collision-free grasp.

**Videos:** `main_run.mp4` is `python control.py`. `extensions.mp4` was recorded with `demo.py` (`python demo.py rotation | shortcut | force-fixed | force-feedback`), which uses the same planner and controller.
