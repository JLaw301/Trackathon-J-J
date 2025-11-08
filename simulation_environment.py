
# /// script
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
# ///

"""
Serial-chain 7-DOF demo robot with classic DH forward kinematics and 3D animation.

Key points
----------
- Classic DH: each link i has (a_i, alpha_i, d_i, theta_i)
- FK composes T = T_01 * T_12 * ... * T_{n-1,n}
- Demo IK: base yaw + 2-link planar shoulder/elbow to reach (x,y,z) approximately.
  The wrist joints default to zero to keep the example simple.

Angles are in radians.
"""

from typing import Optional, Sequence, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation




def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Classic DH homogeneous transform."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1],
    ], dtype=float)


class SerialRobotSim:
    def __init__(
        self,
        a_list: Optional[Sequence[float]] = None,
        alpha_list: Optional[Sequence[float]] = None,
        d_list: Optional[Sequence[float]] = None,
        dof: int = 7,
    ):
        """
        Construct a serial-chain robot using classic DH params.

        Defaults are a simple, non-degenerate 7-link chain:
        - a: link lengths (roughly like your original)
        - alpha: alternating twists to create a spatial chain
        - d: zero offsets (all-revolute about each link's z-axis)
        """
        if a_list is None:
            a_list = [7, 6, 4, 5, 3, 2, 1.5]
        if alpha_list is None:
            # A simple alternating pattern to avoid colinear axes
            alpha_list = [np.pi / 2, 0, -np.pi / 2, np.pi / 2, -np.pi / 2,
                          np.pi / 2, 0]
        if d_list is None:
            d_list = [0.0] * dof

        self.a: List[float] = list(a_list)
        self.alpha: List[float] = list(alpha_list)
        self.d: List[float] = list(d_list)

        if not (len(self.a) == len(self.alpha) == len(self.d) == dof):
            raise ValueError("DH parameter lists must all have length == dof")

        self.dof = dof

        self.fig = None
        self.ax = None
        self.joint_lines = None
        self.trajectory_line = None

        # Animation data
        self.joint_angles_time_series: Optional[List[List[float]]] = None
        self.ee_traj: List[Tuple[float, float, float]] = []

    # -------------------------
    # Forward kinematics (FK)
    # -------------------------
    def forward_kinematics(self, joint_angles: Sequence[float]):
        """
        Compute cumulative transforms and joint origins.

        Returns:
            positions: list of 3D points for joint i origins in world frame,
                       including the end-effector origin as the last element.
            transforms: list of 4x4 cumulative transforms T_0i
        """
        if len(joint_angles) != self.dof:
            raise ValueError("Expected %d joint angles" % self.dof)

        T = np.eye(4)
        positions = [(0.0, 0.0, 0.0)]
        transforms = [T.copy()]

        for i in range(self.dof):
            Ti = dh_transform(self.a[i], self.alpha[i], self.d[i],
                              joint_angles[i])
            T = T @ Ti
            positions.append(tuple(T[:3, 3]))
            transforms.append(T.copy())

        # positions has length dof+1 (base + each joint/end)
        return positions, transforms

    # -------------------------
    # Demo inverse kinematics
    # -------------------------
    def demo_inverse_kinematics(self, x: float, y: float, z: float) -> List[
        float]:
        """
        Toy IK for demonstration (not a general solver):
        - q0 (base yaw) aims the arm toward (x, y)
        - q1, q2 solve a planar 2-link problem in the r-z plane (r = horizontal reach)
        - remaining joints are set to 0 to form a neutral wrist

        It uses the first two link lengths (a0, a1) for the planar reach.
        """
        q = [0.0] * self.dof

        # Base yaw
        q[0] = np.arctan2(y, x)

        # Project target into the base-aligned plane
        r = np.hypot(x, y)
        zt = z

        L1 = abs(self.a[1]) if self.dof > 1 else 0.0
        L0 = abs(self.a[0]) if self.dof > 0 else 0.0

        # Clamp to reachable workspace of the 2-link subset
        R = np.hypot(r, zt)        
        R = max(min(R, L0 + L1 - 1e-9), 1e-9)

        # Law of cosines for elbow
        cos_elbow = (R ** 2 - L0 ** 2 - L1 ** 2) / (2 * L0 * L1)
        cos_elbow = np.clip(cos_elbow, -1.0, 1.0)
        elbow = np.arccos(cos_elbow)  # choose "elbow-down" (0..pi)

        # Shoulder from triangle geometry
        gamma = np.arctan2(zt, r)
        phi = np.arctan2(L1 * np.sin(elbow), L0 + L1 * np.cos(elbow))
        shoulder = gamma - phi

        # Map to q1, q2 (assumes revolute joints about z with DH twists providing plane)
        q[1] = shoulder
        q[2] = elbow

        # Wrist neutral
        # q[3:] remain 0.0

        return q

    def setup_axes_3d(self, title: Optional[str] = None):
        reach = sum(abs(a) for a in self.a) + sum(abs(d) for d in self.d)
        lim = (-reach, reach) if reach > 0 else (-10, 10)

        self.fig = plt.figure(figsize=(16, 9), dpi=80)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(*lim)
        self.ax.set_ylim(*lim)
        self.ax.set_zlim(*lim)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        if title:
            self.ax.set_title(title)

        # One segment per link
        self.joint_lines = [
            self.ax.plot([], [], [], 'o-', linewidth=2, markersize=4)[0]
            for _ in range(self.dof)
        ]
        (self.trajectory_line,) = self.ax.plot([], [], [], 'r--', linewidth=1.5)

    def draw_pose(self, joint_angles: Sequence[float]):
        positions, _ = self.forward_kinematics(joint_angles)

        # draw links
        for j, line in enumerate(self.joint_lines):
            a = positions[j]
            b = positions[j + 1]
            xs, ys, zs = zip(a, b)
            line.set_data(xs, ys)
            line.set_3d_properties(zs)

        # update trajectory with end-effector
        ee = positions[-1]
        self.ee_traj.append(ee)
        tx, ty, tz = (np.array(t) for t in zip(*self.ee_traj))
        self.trajectory_line.set_data(tx, ty)
        self.trajectory_line.set_3d_properties(tz)

        return self.joint_lines + [self.trajectory_line]

    def animate_3d(self, frame: int):
        if self.joint_angles_time_series is None or frame >= len(
            self.joint_angles_time_series):
            return []
        return self.draw_pose(self.joint_angles_time_series[frame])

    # -------------------------
    # Trajectories / utilities
    # -------------------------
    @staticmethod
    def create_time_series(start: float, end: float, step: float):
        """Inclusive end for convenience in demos."""
        return np.arange(start, end + step, step)


def simple_static_pose_demo():
    """
    Demo of plotting a static pose from a joint command with the robot.
    """
    sim = SerialRobotSim()
    sim.setup_axes_3d(title="simple_static_pose_demo")
    q_static = [-1.0, 0.6, 0.8, -0.3, 0.9, -0.5, 0.2]
    sim.draw_pose(q_static)
    plt.show()


def simple_joint_trajectory_demo():
    """
    Demo of passing in a series of joint commands to the robot and animating
    the result.
    """
    sim = SerialRobotSim()

    # Time grid
    t0, t_end, dt = 0.0, 9.0, 0.02
    T = sim.create_time_series(t0, t_end, dt)

    # Waypoints: (time, q) with q shape == (dof,)
    q_rest = np.zeros(sim.dof)
    q_pose1 = np.array([-0.6, 0.8, 0.9, -0.4, 0.6, -0.3, 0.2])
    q_pose2 = np.array([0.8, -0.5, 0.7, 0.6, -0.4, 0.3, -0.2])
    q_pose3 = np.array([0.0, 0.3, -0.6, 0.4, 0.2, -0.4, 0.5])

    waypoints = [
        (0.0, q_rest),
        (3.0, q_pose1),
        (6.0, q_pose2),
        (9.0, q_pose3),
    ]

    # Stack waypoint times and values
    t_wp = np.array([t for t, _ in waypoints], dtype=float)
    q_wp = np.vstack([q for _, q in waypoints])  # shape: (num_wp, dof)

    # Linear interpolation per joint
    Q = np.column_stack(
        [np.interp(T, t_wp, q_wp[:, j]) for j in range(sim.dof)])

    # Animate with these joint angles
    sim.joint_angles_time_series = [q.tolist() for q in Q]
    sim.ee_traj = []

    sim.setup_axes_3d(title="simple_joint_trajectory_demo")
    sim.anim = animation.FuncAnimation(
        sim.fig, sim.animate_3d,
        frames=len(T),
        interval=25,
        blit=False
    )

    plt.show()


def simple_ik_trajectory_demo():
    """
    This is really intended to just be a demo to get you started. Highly
    encouraged to implement your own IK and/or use an existing library, as well
    as come up with your own trajectories and motion planning.
    """

    sim = SerialRobotSim()

    t0, t_end, dt = 0.0, 10.0, 0.02  # ~500 frames, smooth & fast
    time_series = sim.create_time_series(t0, t_end, dt)

    def figure_eight_trajectory_3d(t: float, scale: float = 5.0):
        """3D figure-eight trajectory for the end-effector."""
        x = scale * np.sin(2 * np.pi * t)
        y = scale * np.sin(2 * np.pi * t) * np.cos(2 * np.pi * t)
        z = scale * np.cos(4 * np.pi * t)
        return x, y, z

    joint_angles_time_series: List[List[float]] = []
    for t in time_series:
        x, y, z = figure_eight_trajectory_3d(t, scale=8.0)
        q = sim.demo_inverse_kinematics(x, y, z)
        joint_angles_time_series.append(q)

    sim.joint_angles_time_series = joint_angles_time_series
    sim.ee_traj = []  # reset trajectory for the new animation

    sim.setup_axes_3d(title="simple_ik_trajectory_demo")
    sim.anim = animation.FuncAnimation(
        sim.fig, sim.animate_3d,
        frames=len(time_series),
        interval=50,
        blit=False
    )

    plt.show()


def simple_control_loop_demo():
    """
    Tick-based joint-trajectory demo:
    - Build a linear joint-space trajectory from waypoints
    - At fixed ticks (50 ms), compute q_cmd(t) and "apply" it
    - Optionally simulate 1-tick latency (sample-and-hold) for q_act
    - Redraw robot each tick and plot inputs/outputs afterward
    """
    import time

    sim = SerialRobotSim()
    sim.setup_axes_3d(title="simple_control_loop_demo")
    sim.ee_traj = []

    # trajectory definition
    t0, t_end, dt_traj = 0.0, 9.0, 0.02
    T_traj = sim.create_time_series(t0, t_end, dt_traj)

    q_rest = np.zeros(sim.dof)
    q_pose1 = np.array([-0.6, 0.8, 0.9, -0.4, 0.6, -0.3, 0.2])
    q_pose2 = np.array([ 0.8,-0.5, 0.7, 0.6,-0.4, 0.3,-0.2])
    q_pose3 = np.array([ 0.0, 0.3,-0.6, 0.4, 0.2,-0.4, 0.5])

    waypoints = [
        (0.0, q_rest),
        (3.0, q_pose1),
        (6.0, q_pose2),
        (9.0, q_pose3),
    ]
    t_wp = np.array([t for t, _ in waypoints], dtype=float)
    q_wp = np.vstack([q for _, q in waypoints])  # (num_wp, dof)

    # Precompute a simple linear trajectory q_ref(t) on T_traj for convenience
    Q_ref = np.column_stack([np.interp(T_traj, t_wp, q_wp[:, j]) for j in range(sim.dof)])

    # Helper to grab q_ref for any time t via interpolation on the precomputed arrays
    def q_ref_at(t: float) -> np.ndarray:
        t_clamped = np.clip(t, T_traj[0], T_traj[-1])
        # Find index and linear interpolate between neighboring samples
        idx = np.searchsorted(T_traj, t_clamped)
        if idx == 0:
            return Q_ref[0].copy()
        if idx >= len(T_traj):
            return Q_ref[-1].copy()
        t0, t1 = T_traj[idx-1], T_traj[idx]
        a = (t_clamped - t0) / (t1 - t0)
        return (1 - a) * Q_ref[idx-1] + a * Q_ref[idx]

    # tick loop
    dt_tick = 0.050 # 50 ms tick, 20 Hz
    latency_ticks = 1 # set to 0 for no latency; 1 to simulate 1-tick delay

    # Simple FIFO to model actuator latency
    cmd_fifo: List[np.ndarray] = []

    # telemetry log of timestamps
    t_log: List[float] = []

    # telemetry log of commanded joint position at time t
    q_cmd_log: List[np.ndarray] = []
    # telemetry log of actual joint position at time t
    q_act_log: List[np.ndarray] = []

    # Timing
    t_start = time.perf_counter()
    next_tick = t_start

    # Run until a small hold after t_end
    while True:
        if sim.fig is None or not plt.fignum_exists(sim.fig.number):
            break

        now = time.perf_counter()
        t = now - t_start
        if t >= t_end + 0.25:
            break

        # Input: commanded joints from trajectory
        q_cmd = q_ref_at(t)

        # Push into FIFO; pop for actual applied joints
        cmd_fifo.append(q_cmd.copy())
        if len(cmd_fifo) > latency_ticks:
            q_act = cmd_fifo.pop(0)
        else:
            q_act = q_cmd  # not enough items yet; act immediately

        # Draw applied joints
        sim.draw_pose(q_act)
        plt.pause(0.001)

        # Log
        t_log.append(t)
        q_cmd_log.append(q_cmd.copy())
        q_act_log.append(q_act.copy())

        # Sleep to next tick
        next_tick += dt_tick
        sleep_time = next_tick - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            # Overrun: resync to avoid drift
            next_tick = time.perf_counter()

    # quick visualization of input vs output for one joint J1
    t_arr = np.array(t_log)
    q_cmd_arr = np.vstack(q_cmd_log)
    q_act_arr = np.vstack(q_act_log)

    plt.figure(figsize=(10, 4))
    plt.plot(t_arr, q_cmd_arr[:, 0], label="cmd J1")
    plt.plot(t_arr, q_act_arr[:, 0], label="act J1", linestyle="--")
    plt.title("simple_control_loop_demo: commanded vs applied (Joint 1)")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [rad]")
    plt.legend()
    plt.grid(True)

    plt.show()

def main():

    # These are all "offline" demos without a notion of a control loop, just to
    # show how you can interact with the simulator
    simple_static_pose_demo()
    simple_joint_trajectory_demo()
    simple_ik_trajectory_demo()

    # This is an "online" demo, where the robot is commanded "live" at a certain
    # control loop frequency in the same trajectory as
    # simple_joint_trajectory_demo. You might want to look at:
    #   * q_ref_at: input; this is the input command at time t
    #   * t_log: output; this is a telemetry log of all the timestamps
    #   * q_cmd_log: output; this is a telemetry log of all the commanded joint
    #     positions at time t
    #   * q_act_log: output; this is a telemetry log of all the actual joint
    #     positions at time t
    simple_control_loop_demo()

if __name__ == "__main__":
    main()
