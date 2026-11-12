# README – ARO Lab Submission

## Software requirements
- Course environment (`aro2026`, Python 3.11) – no additional packages required.

## Algorithms used

### Part 1 – Inverse geometry (`inverse_geometry.py`)
<!-- ~60 words. Method in computeqgrasppose: solver type, cost terms, joint limits, collision handling. -->

The algorithm uses an iterative inverse kinematics solver on the 6D placement error of both hands. At each iteration we compute each hand's error to its hook and use the Jacobian pseudo-inverse to find a joint update that reduces it. We used a null-space hierarchy: the left hand is solved first, then the right hand is solved within the left hand's null space so it doesn't undo that correction. After each step the configuration is projected back within the joint limits. To succeed, the hands must be on the hooks, with no collisions and within the joint limits.

### Part 2 – Motion planning (`path.py`)
<!-- ~70 words. rrt + projectedpath + samplegrasp: sampling in cube-placement space,
     projecting each sample to a grasp config, edge discretisation, wall clearance. -->

On part 2, we used RRT.

We additionally added a wall-clearance margin to the planner so the path stays clear of obstacles even after Bézier smoothing and tracking error in Part 3.




### Part 3 – Trajectory & control (`control.py`, `trajectory.py`)
<!-- ~70 words. Bezier fitting (QP via quadprog / Bernstein least squares), time parametrisation,
     controllaw: e.g. PD / computed torque + grip force. -->

**Extra:**
<!-- e.g. QP-fitted high-degree Bezier, robustness to other cube targets. -->
