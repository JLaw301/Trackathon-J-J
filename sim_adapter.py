#!/usr/bin/env python3
"""
sim_adapter.py
--------------
Adapter between SerialRobotSim (7-DOF) and the Tkinter GUI.

Features:
- Non-blocking Matplotlib 3D window on the main thread (TkAgg + plt.ion()).
- redraw() only called from the Tk main thread.
- clear_trajectory() to wipe the red EE trail.
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
        self.sim = SerialRobotSim()          # 7-DOF by default
        self.dof = self.sim.dof
        self.time0 = time.time()

        # State
        self.q      = np.zeros(self.dof, dtype=float)
        self.q_prev = self.q.copy()
        self.qdot   = np.zeros(self.dof, dtype=float)
        self.q_tgt  = self.q.copy()

        # Smoothing toward target
        self.alpha = 0.12

        # Last commanded pose (for telemetry roll/pitch/yaw)
        self.ee_cmd = [0.0, 0.0, 300.0, 0.0, 0.0, 0.0]

        # Matplotlib fig initialized flag
        self._fig_inited = False

    # ---------- Matplotlib (main thread only) ----------
    def init_figure(self):
        """Create (or re-create) the 3D view on the main thread, non-blocking."""
        needs_new = (not self._fig_inited) or (self.sim.fig is None) or (not plt.fignum_exists(self.sim.fig.number))
        if needs_new:
            self.sim.setup_axes_3d(title="Serial 7-DOF — Live View")
            try:
                plt.ion()
                plt.show(block=False)
            except Exception:
                pass
            self._fig_inited = True

    def redraw(self):
        """Redraw current pose in the 3D view. Call ONLY from Tk main thread."""
        if self._fig_inited and self.sim.fig is not None and plt.fignum_exists(self.sim.fig.number):
            self.sim.draw_pose(self.q.tolist())
            try:
                self.sim.fig.canvas.draw_idle()
                self.sim.fig.canvas.flush_events()
            except Exception:
                pass

    def clear_trajectory(self):
        """Clear the red EE path and refresh the figure."""
        try:
            if hasattr(self.sim, "ee_traj"):
                self.sim.ee_traj = []
            line = getattr(self.sim, "trajectory_line", None)
            if line is not None:
                line.set_data([], [])
                try:
                    line.set_3d_properties([])
                except Exception:
                    pass
            if self._fig_inited and self.sim.fig is not None:
                self.sim.fig.canvas.draw_idle()
                self.sim.fig.canvas.flush_events()
        except Exception:
            pass

    # ---------- Control API used by GUI ----------
    def set_target_ee_pose(self, pose6: List[float]):
        """Accept [x,y,z,roll,pitch,yaw] in mm/deg; compute a joint target."""
        x, y, z, roll, pitch, yaw = pose6
        self.ee_cmd = pose6[:]

        # Use demo IK for XYZ
        q_ik = np.array(self.sim.demo_inverse_kinematics(x, y, z), dtype=float)
        # Map small orientation hints to wrist joints (indices 4..6)
        if self.dof >= 7:
            rpy = np.radians([roll, pitch, yaw])
            q_ik[4:7] = 0.25 * rpy[:3]
        self.q_tgt = q_ik

    def step(self, dt: float):
        """Advance internal state toward q_tgt (no GUI calls here)."""
        dt = max(dt, 1e-3)
        self.q_prev = self.q.copy()
        self.q = self.q + self.alpha * (self.q_tgt - self.q)
        self.qdot = (self.q - self.q_prev) / dt

    def get_telemetry(self) -> Dict[str, object]:
        """Telemetry dict with joints and EE pose (RPY echoed from command)."""
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
