import numpy as np
import heapq


def plan_a_star(grid_map, start_pos, goal_pos, grid_res=0.1, N_horizon=30):
    start_idx = (int(np.round(start_pos[0] / grid_res)), int(np.round(start_pos[1] / grid_res)))
    goal_idx = (int(np.round(goal_pos[0] / grid_res)), int(np.round(goal_pos[1] / grid_res)))

    rows, cols = grid_map.shape

    actions = [
        (0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
        (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)),
        (-1, 1, np.sqrt(2)), (-1, -1, np.sqrt(2))
    ]

    def heuristic(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0.0 + heuristic(start_idx, goal_idx), start_idx))

    came_from = {}
    g_score = {start_idx: 0.0}

    path_found = False

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal_idx:
            path_found = True
            break

        for dx, dy, step_cost in actions:
            neighbor = (current[0] + dx, current[1] + dy)

            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                continue

            if grid_map[neighbor[0], neighbor[1]] == 1:
                continue

            tentative_g = g_score[current] + step_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal_idx)
                heapq.heappush(open_set, (f_score, neighbor))

    if not path_found:
        raise ValueError("A* failed to find a valid path to the goal!")

    curr = goal_idx
    raw_path_grid = [curr]
    while curr in came_from:
        curr = came_from[curr]
        raw_path_grid.append(curr)
    raw_path_grid.reverse()

    raw_path_world = np.array(raw_path_grid, dtype=float) * grid_res

    distances = np.cumsum(np.hypot(
        np.diff(raw_path_world[:, 0], prepend=raw_path_world[0, 0]),
        np.diff(raw_path_world[:, 1], prepend=raw_path_world[0, 1])
    ))
    total_dist = distances[-1]
    interp_distances = np.linspace(0, total_dist, N_horizon + 1)

    ref_x = np.interp(interp_distances, distances, raw_path_world[:, 0])
    ref_y = np.interp(interp_distances, distances, raw_path_world[:, 1])

    waypoints = np.vstack((ref_x, ref_y))
    return waypoints