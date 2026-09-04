import cvxpy as cp
import numpy as np

def get_closest_obstacle(p, obstacles):
    o0 = np.array([obstacles[0][0], obstacles[0][1]])
    closest = o0
    for obs in obstacles:
        o = np.array([obs[0], obs[1]])
        if np.linalg.norm(o - p) < np.linalg.norm(closest - p):
            closest = o
    return closest


def solve_mpc(N, dt, x_start, a_max, v_max, obstacles, waypoints, r_obs=0.5):
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1]
    ])

    B = np.array([
        [0.5 * dt**2, 0],
        [0, 0.5 * dt**2],
        [dt, 0],
        [0, dt]
    ])

    X = cp.Variable((4, N + 1))
    U = cp.Variable((2, N))

    cost = 0.0
    r = 1.0
    q = 100.0

    constraints = [X[:, 0] == x_start]

    for k in range(N):
        constraints += [X[:, k+1] == A @ X[:, k] + B @ U[:, k]]
        constraints += [cp.abs(U[0, k]) <= a_max]
        constraints += [cp.abs(U[1, k]) <= a_max]
        constraints += [cp.norm(X[2:4, k], 2) <= v_max]

        cost += r * cp.sum_squares(U[:, k])

    constraints += [cp.norm(X[2:4, N], 2) <= v_max]

    for k in range(N + 1):
        p_ref = waypoints[:, k]

        cost += q * cp.sum_squares(X[0:2, k] - p_ref)

        n_obs = get_closest_obstacle(p_ref, obstacles)
        diff = n_obs - p_ref
        dist = np.linalg.norm(diff)

        if dist > 0.001:
            a = diff / dist
            p_wall = n_obs - a * r_obs
            b = np.dot(a, p_wall)
            constraints.append(a @ X[0:2, k] <= b)

    objective = cp.Minimize(cost)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    return prob.status, X.value, U.value