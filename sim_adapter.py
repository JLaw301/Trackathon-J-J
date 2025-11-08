
"""
sim_adapter.py
--------------
Adapter between SerialRobotSim (7-DOF) and the Tkinter GUI.

This version ensures the Matplotlib 3D window opens in non-blocking mode
and is safely redrawn from the Tkinter main thread.
"""

from __future__ import annotations
from typing import List, Dict
import time
import numpy as np

# Use TkAgg explicitly for Tkinter integration BEFORE importing pyplot
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from simulation_environment import SerialRobotSim


class SimulationEnvironment:
    def __init__(self):
        self.sim = SerialRobotSim()
        self.dof = self.sim.dof
        self.time0 = time.time()
        self.q = np.zeros(self.dof, dtype=float)
        self.q_prev = self.q.copy()
        self.qdot = np.zeros(self.dof, dtype=float)
        self.q_tgt = self.q.copy()
        self.alpha = 0.12
        self.ee_cmd = [0.0, 0.0, 300.0, 0.0, 0.0, 0.0]

        self._fig_inited = False

    # ---- main-thread safe 3-D viewer ----
    def init_figure(self):
        """Create (or re-create) the 3D view on the main thread, non-blocking."""
        # If closed by user, re-init is allowed
        if (not self._fig_inited) or (self.sim.fig is None) or (not plt.fignum_exists(self.sim.fig.number)):
            self.sim.setup_axes_3d(title="Serial 7-DOF — Live View")
            # Non-blocking show so Tkinter app continues
            try:
                plt.ion()
            except Exception:
                pass
            try:
                plt.show(block=False)
            except Exception:
                pass
            self._fig_inited = True

    def redraw(self):
        """Redraw current pose in the 3D view. Call ONLY from main thread."""
        if self._fig_inited and self.sim.fig is not None and plt.fignum_exists(self.sim.fig.number):
            self.sim.draw_pose(self.q.tolist())
            try:
                self.sim.fig.canvas.draw_idle()
                self.sim.fig.canvas.flush_events()
            except Exception:
                pass
    

    # ---- control API used by GUI ----
    def set_target_ee_pose(self, pose6: List[float]):
        x, y, z, roll, pitch, yaw = pose6
        self.ee_cmd = pose6[:]

        q_ik = np.array(self.sim.demo_inverse_kinematics(x, y, z), dtype=float)
        rpy = np.radians([roll, pitch, yaw])
        if self.dof >= 7:
            q_ik[4:7] = 0.25 * rpy[:3]
        self.q_tgt = q_ik

    def step(self, dt: float):
        dt = max(dt, 1e-3)
        self.q_prev = self.q.copy()
        self.q = self.q + self.alpha * (self.q_tgt - self.q)
        self.qdot = (self.q - self.q_prev) / dt

    def get_telemetry(self) -> Dict[str, object]:
        positions, _ = self.sim.forward_kinematics(self.q.tolist())
        x, y, z = positions[-1]
        roll, pitch, yaw = self.ee_cmd[3:]
        return {
            "time": time.time() - self.time0,
            "joint_positions": self.q.tolist(),
            "joint_velocities": self.qdot.tolist(),
            "joint_torques": [0.0] * self.dof,
            "ee_pose": [float(x), float(y), float(z), float(roll), float(pitch), float(yaw)],
        }
    
    def clear_trajectory(self):
        """Clear the red EE path and refresh the figure."""
        try:
            # wipe stored trajectory
            if hasattr(self.sim, "ee_traj"):
                self.sim.ee_traj = []
            # clear the plotted line if present
            line = getattr(self.sim, "trajectory_line", None)
            if line is not None:
                line.set_data([], [])
                try:
                    # 3D needs this as well
                    line.set_3d_properties([])
                except Exception:
                    pass
            # refresh the figure if it's open
            if getattr(self, "_fig_inited", False) and getattr(self.sim, "fig", None):
                try:
                    self.sim.fig.canvas.draw_idle()
                    self.sim.fig.canvas.flush_events()
                except Exception:
                    pass
        except Exception:
            pass
