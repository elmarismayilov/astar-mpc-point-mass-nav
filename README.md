# 2D Point-Mass Navigation using A* Global Planning and Convex MPC

A modular 2D motion planning and control framework for point-mass systems, applicable to drone/quadrotor position control and ground-robot trajectory tracking. The pipeline combines global A* pathfinding on a discrete grid with real-time Receding Horizon Control (RHC) solved via Convex Model Predictive Control (MPC) with linear obstacle-avoidance constraints.

---

## System Architecture

The project is structured into three main modules:

* **`a_star.py`**: Discretizes the 2D environment, executes 8-connected grid search to find the shortest collision-free path, and interpolates it into equidistant global waypoints.
* **`solver.py`**: Solves a local Convex Optimal Control Problem over a finite horizon ($N$). Minimizes control effort and tracking error while enforcing double-integrator state dynamics, control input (acceleration) limits, velocity constraints, and linear obstacle-avoidance constraints.
* **`simulation.py`**: Handles environment visualization, mouse-driven interactive obstacle placement, closed-loop state feedback simulation, and animation.

---

## Mathematical Formulation

### 1. Kinematic Model (Double-Integrator)

The planar dynamics are modeled in discrete time ($\Delta t$) as:

$$\mathbf{x}_{k+1} = \mathbf{A}\mathbf{x}_k + \mathbf{B}\mathbf{u}_k$$

$$\mathbf{x}_k = \begin{bmatrix} x \\ y \\ v_x \\ v_y \end{bmatrix}_k, \quad \mathbf{u}_k = \begin{bmatrix} a_x \\ a_y \end{bmatrix}_k$$

$$\mathbf{A} = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix} \frac{1}{2}\Delta t^2 & 0 \\ 0 & \frac{1}{2}\Delta t^2 \\ \Delta t & 0 \\ 0 & \Delta t \end{bmatrix}$$

### 2. Convex Optimization Problem

At each time step, the solver optimizes over state trajectory $\mathbf{X}$ and control inputs $\mathbf{U}$:

$$\min_{\mathbf{X}, \mathbf{U}} \sum_{k=0}^{N-1} \left( r \|\mathbf{u}_k\|_2^2 + q \|\mathbf{p}_k - \mathbf{p}_{\text{ref}, k}\|_2^2 \right) + q \|\mathbf{p}_N - \mathbf{p}_{\text{ref}, N}\|_2^2$$

**Subject to:**

* Initial state constraint: $\mathbf{x}_0 = \mathbf{x}_{\text{start}}$
* Dynamics constraints: $\mathbf{x}_{k+1} = \mathbf{A}\mathbf{x}_k + \mathbf{B}\mathbf{u}_k, \quad \forall k \in [0, N-1]$
* Control input limits: $\|\mathbf{u}_k\|_\infty \le a_{\max}$
* Speed bounds: $\|\mathbf{v}_k\|_2 \le v_{\max}$
* **Linear Obstacle-Avoidance Constraints:** For each horizon step $k$, a separating half-space constraint is constructed tangent to the nearest obstacle boundary:

  $$\mathbf{a}_k^T \mathbf{p}_k \le b_k$$

  Where $\mathbf{a}_k = \frac{\mathbf{o}_{\text{near}} - \mathbf{p}_{\text{ref}, k}}{\|\mathbf{o}_{\text{near}} - \mathbf{p}_{\text{ref}, k}\|_2}$ and $b_k = \mathbf{a}_k^T (\mathbf{o}_{\text{near}} - \mathbf{a}_k \cdot r_{\text{obs}})$.

---

## Dependencies & Installation

### Requirements

* Python 3.8+
* `numpy`
* `cvxpy`
* `clarabel` (Convex QP/SOCP solver)
* `matplotlib`

### Setup

```bash
git clone https://github.com/elmarismayilov/astar-mpc-point-mass-nav.git
cd astar-mpc-point-mass-nav
python3 -m venv venv
source venv/bin/activate
pip install numpy cvxpy clarabel matplotlib
```
