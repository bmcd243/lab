# ARO lab submission

**Software:** only the lab's `requirements.txt` (uses `quadprog` and `pybullet`).

**Inverse geometry.** Iterative IK on both hands' 6D error: left hand first, right hand in its null space (Tutorial 3), projected onto joint limits each step. Success = converged, collision-free, within limits.

**Motion planning.** RRT over cube positions (Tutorial 4). Edges move the cube up to 10 cm in 1 cm steps, re-solving the grasp from the previous configuration. A step fails on collision, a joint jump above 0.3 rad (IK flipping arm pose), or under 15 mm from the wall; this margin fades out near the start and goal, which sit close to the wall. The path ends by sliding onto qe.

**Control.**
- Trajectory (level 1): one Bézier curve (16 control points) fitted to the arc-length-timed path by a QP; first/last three control points fixed to q0/qe (rest to rest), all within joint limits (convex hull).
- Tracking: inverse dynamics, tau = M(q)a* + h(q,v), with a* = reference acceleration + PID.
- Grasp: force on each hand through J^T f: a spring onto its hook (cube pose from pybullet) plus an inward squeeze.
- Model matching: pybullet gives the hand-tip links (no `<inertial>` in the URDF) 1 kg each and damps every joint; we zero those masses and add the damping to the torques.
- `bezier.py`: fixed an off-by-one in `size_` that dropped a control point.

**Extension: force feedback.** The squeeze follows the measured contact forces: target press N* = max(30 N, F_t / (0.6 mu)), tracked by a PI loop (Lecture 11). With a 3x heavier cube, the fixed squeeze drops it; force feedback delivers it.

**Robustness.** Delivered in all tested start/goal positions where a grasp exists, on both sides of the wall. Cube rotations are not handled.
