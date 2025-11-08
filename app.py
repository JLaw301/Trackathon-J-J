
"""
Teleoperation Panel (Tkinter) — 7-DOF control for SerialRobotSim + CSV + image-control stub.

Usage:
    python app.py

Dependencies:
    pip install opencv-python matplotlib numpy
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import csv
import os
from datetime import datetime

# Optional — only used for image control demo
try:
    import cv2
except Exception:
    cv2 = None

# Import adapter that wraps SerialRobotSim
from sim_adapter import SimulationEnvironment  # uses SerialRobotSim internally

APP_NAME = "TeleOp 7-DOF — Hackathon Prototype"
UPDATE_HZ = 30.0
DT = 1.0 / UPDATE_HZ


class TeleopApp:
    def __init__(self, root):
        self.root = root
        root.title(APP_NAME)
        root.geometry("980x640")

        self.sim = SimulationEnvironment()
        self.dof = self.sim.dof
        self.running = False
        self.recording = False
        self.record_buffer = []
        self.update_thread = None

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        self.frame_control = ttk.Frame(nb)
        self.frame_replay = ttk.Frame(nb)
        self.frame_help = ttk.Frame(nb)

        nb.add(self.frame_control, text="Teleoperation")
        nb.add(self.frame_replay, text="Replay / Data")
        nb.add(self.frame_help, text="Help")

        self._build_control_tab()
        self._build_replay_tab()
        self._build_help_tab()

    def _build_control_tab(self):
        f = self.frame_control

        # ---- left: sliders ----
        left = ttk.LabelFrame(f, text="End Effector Target (Task Space)")
        left.pack(side="left", fill="y", padx=8, pady=8)

        self.scales = {}
        self.scales["X"] = self._make_scale(left, "X (mm)", -300, 300, 0)
        self.scales["Y"] = self._make_scale(left, "Y (mm)", -300, 300, 0)
        self.scales["Z"] = self._make_scale(left, "Z (mm)", -50, 550, 250)
        self.scales["Roll"] = self._make_scale(left, "Roll (deg)", -180, 180, 0)
        self.scales["Pitch"] = self._make_scale(left, "Pitch (deg)", -180, 180, 0)
        self.scales["Yaw"] = self._make_scale(left, "Yaw (deg)", -180, 180, 0)

        # ---- right: session info ----
        right = ttk.LabelFrame(f, text="Session")
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        btn_row = ttk.Frame(right)
        btn_row.pack(fill="x", pady=(6, 6))

        self.btn_start = ttk.Button(btn_row, text="Start", command=self.start)
        self.btn_stop = ttk.Button(btn_row, text="Stop", command=self.stop, state="disabled")
        self.btn_start.pack(side="left", padx=4)
        self.btn_stop.pack(side="left", padx=4)

        self.btn_record = ttk.Button(
            btn_row, text="Start Recording", command=self.toggle_record, state="disabled"
        )
        self.btn_record.pack(side="left", padx=10)

        self.btn_save = ttk.Button(
            btn_row, text="Save CSV", command=self.save_csv, state="disabled"
        )
        self.btn_save.pack(side="left", padx=4)

        self.btn_image = ttk.Button(
            btn_row, text="Image Control (stub)", command=self.image_control_stub
        )
        self.btn_image.pack(side="left", padx=10)

        # NEW: button to explicitly open the 3D view
        self.btn_open3d = ttk.Button(btn_row, text="Open 3D View", command=self.sim.init_figure)
        self.btn_open3d.pack(side="left", padx=10)

        self.btn_clear_path = ttk.Button(btn_row, text="Clear Path", command=self.sim.clear_trajectory)
        self.btn_clear_path.pack(side="left", padx=10)


        telem = ttk.LabelFrame(right, text="Telemetry (live)")
        telem.pack(fill="both", expand=True, padx=6, pady=6)

        self.lbl_ee = ttk.Label(telem, text="EE Pose: x=0 y=0 z=0 roll=0 pitch=0 yaw=0")
        self.lbl_ee.pack(anchor="w", padx=6, pady=2)
        self.lbl_joints = ttk.Label(telem, text=f"Joints: {[0]*self.dof}")
        self.lbl_joints.pack(anchor="w", padx=6, pady=2)
        self.lbl_vel = ttk.Label(telem, text=f"Joint Vel: {[0]*self.dof}")
        self.lbl_vel.pack(anchor="w", padx=6, pady=2)
        self.lbl_torque = ttk.Label(telem, text=f"Torque: {[0]*self.dof}")
        self.lbl_torque.pack(anchor="w", padx=6, pady=2)

        self.lbl_status = ttk.Label(right, text="Status: Idle")
        self.lbl_status.pack(anchor="w", padx=6, pady=6)

    def _make_scale(self, parent, label, mn, mx, val):
        frame = ttk.Frame(parent)
        frame.pack(padx=6, pady=6, fill="x")
        ttk.Label(frame, text=label).pack(anchor="w")
        s = ttk.Scale(frame, from_=mn, to=mx, orient="horizontal")
        s.set(val)
        s.pack(fill="x")
        v = tk.StringVar(value=f"{val:.1f}")
        ttk.Label(frame, textvariable=v, width=8).pack(side="right", padx=4)

        def on_move(evt):
            v.set(f"{s.get():.1f}")

        s.bind("<B1-Motion>", on_move)
        s.bind("<ButtonRelease-1>", on_move)
        return s

    def _build_replay_tab(self):
        f = self.frame_replay
        row = ttk.Frame(f)
        row.pack(fill="x", padx=8, pady=8)

        self.btn_load_csv = ttk.Button(row, text="Load CSV", command=self.load_csv)
        self.btn_play_csv = ttk.Button(
            row, text="Replay CSV", command=self.replay_csv, state="disabled"
        )
        self.btn_load_csv.pack(side="left", padx=6)
        self.btn_play_csv.pack(side="left", padx=6)

        self.txt_preview = tk.Text(f, height=20)
        self.txt_preview.pack(fill="both", expand=True, padx=8, pady=8)

    def _build_help_tab(self):
        f = self.frame_help
        info = (
            "Teleoperation Panel — Quick Help\n\n"
            "• Move sliders (XYZ mm, RPY deg) to set an EE target.\n"
            "• Start: begin streaming to 7-DOF SerialRobotSim; Stop to halt.\n"
            "• Start Recording to capture telemetry; Save CSV to export.\n"
            "• Load CSV on Replay tab then Replay CSV to drive the sim by file.\n"
            "• Image Control (stub): webcam opens; nudges X/Y toward a bright target.\n\n"
            "Integration:\n"
            "• The GUI talks to sim_adapter.SimulationEnvironment (wraps SerialRobotSim).\n"
            "• Replace the adapter with a real SDK later; keep same 3 methods.\n"
        )
        ttk.Label(f, text=info, justify="left").pack(anchor="w", padx=8, pady=8)

    # ---------------- Control ----------------
    def start(self):
        if self.running:
            return
        self.running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_record.config(state="normal")
        self.btn_save.config(state="disabled")
        self.lbl_status.config(text="Status: Running")

        # Initialize Matplotlib figure safely on main thread
        try:
            self.sim.init_figure()
        except Exception:
            pass

        self.update_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.update_thread.start()
        self._schedule_redraw()  # main-thread redraw loop

    def _schedule_redraw(self):
        try:
            self.sim.redraw()
        except Exception:
            pass
        if self.running:
            self.root.after(33, self._schedule_redraw)  # ~30 FPS

    def stop(self):
        self.running = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_record.config(state="disabled")
        self.lbl_status.config(text="Status: Stopped")
        if self.recording:
            self.toggle_record(force_off=True)

    def toggle_record(self, force_off=False):
        if not self.running and not self.recording:
            messagebox.showinfo("Info", "Start the session before recording.")
            return
        if force_off or self.recording:
            self.recording = False
            self.btn_record.config(text="Start Recording")
            self.btn_save.config(
                state="normal" if self.record_buffer else "disabled"
            )
            self.lbl_status.config(text="Status: Running (record stopped)")
        else:
            self.record_buffer = []
            self.recording = True
            self.btn_record.config(text="Stop Recording")
            self.btn_save.config(state="disabled")
            self.lbl_status.config(text="Status: Running (recording)")

    def _get_slider_pose(self):
        return [
            float(self.scales["X"].get()),
            float(self.scales["Y"].get()),
            float(self.scales["Z"].get()),
            float(self.scales["Roll"].get()),
            float(self.scales["Pitch"].get()),
            float(self.scales["Yaw"].get()),
        ]

    def _run_loop(self):
        last = time.time()
        while self.running:
            now = time.time()
            dt = now - last
            last = now

            pose = self._get_slider_pose()
            self.sim.set_target_ee_pose(pose)
            self.sim.step(dt)

            telem = self.sim.get_telemetry()
            ee = telem["ee_pose"]
            self.lbl_ee.config(
                text=f"EE Pose: x={ee[0]:.1f} y={ee[1]:.1f} z={ee[2]:.1f} "
                f"roll={ee[3]:.1f} pitch={ee[4]:.1f} yaw={ee[5]:.1f}"
            )
            self.lbl_joints.config(
                text=f"Joints: {['%.2f' % j for j in telem['joint_positions']]}"
            )
            self.lbl_vel.config(
                text=f"Joint Vel: {['%.2f' % v for v in telem['joint_velocities']]}"
            )
            self.lbl_torque.config(
                text=f"Torque: {['%.2f' % t for t in telem['joint_torques']]}"
            )

            if self.recording:
                row = {
                    "t": telem["time"],
                    **{f"j{i+1}": telem["joint_positions"][i] for i in range(self.dof)},
                    **{f"vel{i+1}": telem["joint_velocities"][i] for i in range(self.dof)},
                    **{f"tau{i+1}": telem["joint_torques"][i] for i in range(self.dof)},
                    "x": ee[0],
                    "y": ee[1],
                    "z": ee[2],
                    "roll": ee[3],
                    "pitch": ee[4],
                    "yaw": ee[5],
                }
                self.record_buffer.append(row)

            time.sleep(max(0.0, DT - (time.time() - now)))

    # ---------------- CSV ----------------
    def save_csv(self):
        if not self.record_buffer:
            messagebox.showwarning("No data", "Nothing recorded yet.")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default = f"telemetry_{ts}.csv"
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default,
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return

        fields = (
            ["t"]
            + [f"j{i+1}" for i in range(self.dof)]
            + [f"vel{i+1}" for i in range(self.dof)]
            + [f"tau{i+1}" for i in range(self.dof)]
            + ["x", "y", "z", "roll", "pitch", "yaw"]
        )

        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.record_buffer:
                w.writerow(r)

        messagebox.showinfo("Saved", f"CSV saved to:\n{path}")
        self.btn_save.config(state="disabled")

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        try:
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.loaded_csv = rows
            self.loaded_csv_path = path
            self.txt_preview.delete("1.0", "end")
            self.txt_preview.insert("end", f"Loaded: {os.path.basename(path)}\n\n")
            with open(path, "r") as f2:
                for i, line in enumerate(f2):
                    if i > 20:
                        self.txt_preview.insert("end", "...\n")
                        break
                    self.txt_preview.insert("end", line)
            self.btn_play_csv.config(state="normal")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n{e}")

    def replay_csv(self):
        if not hasattr(self, "loaded_csv") or not self.loaded_csv:
            messagebox.showwarning("No CSV", "Load a CSV first.")
            return

        def _replay():
            self.lbl_status.config(text="Status: Replaying CSV")
            for row in self.loaded_csv:
                pose = [
                    float(row.get(k, 0.0))
                    for k in ["x", "y", "z", "roll", "pitch", "yaw"]
                ]
                self.sim.set_target_ee_pose(pose)
                self.sim.step(1.0 / 60.0)
                time.sleep(1.0 / 60.0)

                telem = self.sim.get_telemetry()
                ee = telem["ee_pose"]
                self.lbl_ee.config(
                    text=f"EE Pose: x={ee[0]:.1f} y={ee[1]:.1f} z={ee[2]:.1f} "
                    f"roll={ee[3]:.1f} pitch={ee[4]:.1f} yaw={ee[5]:.1f}"
                )
                self.root.update_idletasks()
            self.lbl_status.config(text="Status: Replay finished")

        threading.Thread(target=_replay, daemon=True).start()

    # ---------------- Image control ----------------
    def image_control_stub(self):
        if cv2 is None:
            messagebox.showwarning(
                "OpenCV missing", "Install OpenCV:\n\npip install opencv-python"
            )
            return

        def _run_cam():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Camera", "Could not open camera.")
                return

            k_gain = 0.15
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, _, _, maxLoc = cv2.minMaxLoc(gray)
                tx, ty = maxLoc

                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                cv2.circle(frame, (tx, ty), 6, (0, 255, 255), 2)
                cv2.line(frame, (cx, cy), (tx, ty), (0, 255, 0), 2)

                dx, dy = tx - cx, ty - cy
                cur = self._get_slider_pose()
                cur[0] += k_gain * dx
                cur[1] += k_gain * (-dy)
                self.scales["X"].set(cur[0])
                self.scales["Y"].set(cur[1])

                cv2.putText(
                    frame,
                    f"dx={dx}px dy={dy}px -> dX={k_gain*dx:.1f}mm dY={-k_gain*dy:.1f}mm",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("Image Control (press q to close)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()

        threading.Thread(target=_run_cam, daemon=True).start()


def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.2)
    except Exception:
        pass
    app = TeleopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
