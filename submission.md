# README – ARO Lab Submission

## Software requirements
Course environment only (`aro2026`, Python 3.11); no additional packages required.

## Algorithms used

### Part 1.I – Inverse geometry (`inverse_geometry.py`)
Iterative inverse kinematics on both hands' 6D placement error. A null-space hierarchy solves the left hand first, then the right hand within its null space. Each step is projected within joint limits. Success requires the hands on the hooks, no collisions and joint limits respected.

### Part 1.II – Motion planning (`path.py`)

An RRT over cube positions. Each iteration moves the cube up to 10 cm towards a random sample in 1 cm steps, solving IG (Part 1.I) at each step from the previous configuration, then tries a straight line to the target.

**Extra:** a 1.5 cm wall-clearance margin, so the smoothed Bézier trajectory and tracking error in Part 2 don't cause collisions.

### Part 2 – Trajectory & control (`control.py`)

**Trajectory (level 1):** per joint, a degree-15 Bézier curve (control points $P$) fitted by a QP minimising squared distance to the arc-length-timed waypoints, subject to:
- **equality constraints:** first/last three control points = q₀/q_e, giving zero velocity and acceleration at both ends;
- **inequality constraints:** $q_{\min} \le P_i \le q_{\max}$ for every control point, so the whole curve stays within the joint limits.


**Control:** at each timestep the control law compares the reference with the measured state $(q, \dot q)$ and converts the error into torques using inverse dynamics with PID.


#### Extension: force feedback (`gripforce` in `control.py`)
The squeeze is set from measured hand–cube contact forces (normal $N$, friction $F_t$). The required normal force is

$$
N^* = \max\left(F_{\min},\ \frac{F_t}{s\,\mu}\right)
$$

($\mu$: friction coefficient, $s$: safety factor), and a PI loop drives $N$ towards $N^*$, so the hands squeeze harder only when more friction is needed.
