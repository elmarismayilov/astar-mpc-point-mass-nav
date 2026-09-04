import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

from a_star import plan_a_star
from solver import solve_mpc

if __name__ == "__main__":
    dt = 0.1
    N_mpc = 20
    a_max = 2.0
    v_max = 2.5
    grid_res = 0.1
    r_obs = 0.5
    map_size = 13.0

    x_start = np.array([0.5, 0.5, 0.0, 0.0])
    x_goal  = np.array([11.5, 11.5, 0.0, 0.0])

    obstacles = []
    obstacle_patches = []

    curr_state = x_start.copy()
    trajectory_history = [curr_state[0:2].copy()]
    global_waypoints = None
    sim_running = False
    anim = None

    A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    B = np.array([[0.5 * dt**2, 0], [0, 0.5 * dt**2], [dt, 0], [0, dt]])

    fig, ax = plt.subplots(figsize=(9, 9))
    plt.subplots_adjust(bottom=0.15)

    ax.plot(x_start[0], x_start[1], 'go', markersize=10, label="Start")
    ax.plot(x_goal[0], x_goal[1], 'ro', markersize=10, label="Goal")

    line_a_star, = ax.plot([], [], 'g--', alpha=0.5, label="Global A* Path")
    line_trail, = ax.plot([], [], 'b-', linewidth=2, label="Executed Trajectory")
    line_horizon, = ax.plot([], [], 'c-o', markersize=3, label="MPC Horizon Vector")
    drone_dot, = ax.plot([], [], 'bo', markersize=10, label="Drone")

    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Click to Add Obstacles, Then Press 'Start Navigation'")
    ax.grid(True)
    ax.legend(loc="upper left")

    def on_click(event):
        if sim_running:
            return
        
        if event.inaxes == ax and event.button == 1:
            ox, oy = event.xdata, event.ydata
            obstacles.append((ox, oy))
            
            marker, = ax.plot(ox, oy, 'kx', markersize=8)
            circle = plt.Circle((ox, oy), r_obs, color='red', fill=True, alpha=0.3)
            ax.add_patch(circle)
            
            obstacle_patches.extend([marker, circle])
            fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect('button_press_event', on_click)

    def update(frame):
        global curr_state

        if not sim_running or global_waypoints is None:
            return line_trail, line_horizon, drone_dot, line_a_star

        dist_to_goal = np.linalg.norm(curr_state[0:2] - x_goal[0:2])
        if dist_to_goal < 0.2:
            return line_trail, line_horizon, drone_dot, line_a_star

        dists = np.hypot(global_waypoints[0, :] - curr_state[0], global_waypoints[1, :] - curr_state[1])
        closest_idx = np.argmin(dists)

        horizon_indices = np.minimum(closest_idx + np.arange(N_mpc + 1), global_waypoints.shape[1] - 1)
        local_waypoints = global_waypoints[:, horizon_indices]

        status, X_opt, U_opt = solve_mpc(
            N=N_mpc,
            dt=dt,
            x_start=curr_state,
            waypoints=local_waypoints,
            obstacles=obstacles,
            a_max=a_max,
            v_max=v_max,
            r_obs=r_obs
        )

        if X_opt is not None and U_opt is not None:
            u_0 = U_opt[:, 0]
            curr_state = A @ curr_state + B @ u_0
            trajectory_history.append(curr_state[0:2].copy())

            traj_arr = np.array(trajectory_history)
            line_trail.set_data(traj_arr[:, 0], traj_arr[:, 1])
            line_horizon.set_data(X_opt[0, :], X_opt[1, :])
            drone_dot.set_data([curr_state[0]], [curr_state[1]])

        return line_trail, line_horizon, drone_dot, line_a_star

    def start_sim(event):
        global sim_running, global_waypoints, anim

        if sim_running:
            return

        if len(obstacles) == 0:
            print("Please click on the grid to add at least one obstacle first!")
            return

        print(f"Generating A* Path for {len(obstacles)} obstacles...")
        
        grid_dim = int(map_size / grid_res)
        grid_map = np.zeros((grid_dim, grid_dim), dtype=int)
        margin_cells = int(np.ceil(r_obs / grid_res))

        for ox, oy in obstacles:
            ix = int(np.round(ox / grid_res))
            iy = int(np.round(oy / grid_res))
            row_min, row_max = max(0, ix - margin_cells), min(grid_dim, ix + margin_cells + 1)
            col_min, col_max = max(0, iy - margin_cells), min(grid_dim, iy + margin_cells + 1)
            grid_map[row_min:row_max, col_min:col_max] = 1

        try:
            global_waypoints = plan_a_star(grid_map, x_start[0:2], x_goal[0:2], grid_res, N_horizon=250)
            line_a_star.set_data(global_waypoints[0, :], global_waypoints[1, :])
            
            sim_running = True
            ax.set_title("Receding Horizon MPC Navigation Running...")
            
            anim = FuncAnimation(fig, update, frames=300, interval=50, blit=True)
            fig.canvas.draw_idle()
            
        except ValueError as e:
            print(f"Pathfinding Error: {e}")
            ax.set_title("A* Failed: Obstacles block all possible paths!")
            fig.canvas.draw_idle()

    ax_button = plt.axes([0.38, 0.03, 0.25, 0.05])
    btn_start = Button(ax_button, 'Start Navigation', color='lightgreen', hovercolor='0.9')
    btn_start.on_clicked(start_sim)

    plt.show()