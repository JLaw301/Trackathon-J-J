"""
Teleoperation Panel (Tkinter) — 7-DOF control for SerialRobotSim + CSV + image-control (ORB) + thread-safe UI.

Run:
    pip install opencv-python matplotlib numpy pillow
    python app.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk  # pip install pillow
import threading
import time
import csv
import os
import sys
from datetime import datetime

# ---------- Packaging-safe asset resolver ----------
def resource_path(rel_path: str) -> str:
    """
    Return absolute path to resource, works for dev and for PyInstaller bundle.
    """
    if hasattr(sys, "_MEIPASS"):  # set by PyInstaller in onefile mode
        return os.path.join(sys._MEIPASS, rel_path)
    return os.path.join(os.path.abspath("."), rel_path)

# Import adapter that wraps SerialRobotSim
from sim_adapter import SimulationEnvironment  # uses SerialRobotSim internally


APP_NAME = "Aggie Trackathon App"
UPDATE_HZ = 30.0
DT = 1.0 / UPDATE_HZ

# --- Optional: Splash / Loading screen (1–2 sec) ---
def show_splash(root, logo_path=None, duration_ms=1500):
    """
    Show a small centered splash for duration_ms, then reveal the main window.
    Safe: no threads started; does not touch your existing logic.
    """
    # Start hidden; we reveal after splash
    root.withdraw()

    splash = tk.Toplevel(root)
    splash.overrideredirect(True)  # borderless
    splash.configure(bg="#0f1116")

    # Try to load an image; fall back to text
    frame = tk.Frame(splash, bg="#0f1116")
    frame.pack(padx=28, pady=22)

    used_image = False
    if logo_path:
        try:
            img = Image.open(logo_path)
            max_w, max_h = 360, 360
            w, h = img.size
            scale = min(max_w / w, max_h / h, 1.0)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            tkimg = ImageTk.PhotoImage(img)
            lbl = tk.Label(frame, image=tkimg, bg="#0f1116")
            lbl.image = tkimg  # keep ref
            lbl.pack()
            used_image = True
        except Exception:
            used_image = False

    if not used_image:
        tk.Label(frame, text="Trackathon", fg="white", bg="#0f1116",
                 font=("Segoe UI", 20, "bold")).pack(pady=(6, 0))

    tk.Label(frame, text="Loading…", fg="#cbd5e1", bg="#0f1116",
             font=("Segoe UI", 11)).pack(pady=(10, 6))

    pb = ttk.Progressbar(frame, mode="indeterminate", length=240)
    pb.pack()
    pb.start(12)  # speed

    # Center on screen
    splash.update_idletasks()
    sw = splash.winfo_screenwidth()
    sh = splash.winfo_screenheight()
    ww = splash.winfo_width()
    wh = splash.winfo_height()
    x = (sw // 2) - (ww // 2)
    y = (sh // 2) - (wh // 2)
    splash.geometry(f"+{x}+{y}")

    # After delay, close splash and show main
    def _close():
        try:
            pb.stop()
        except Exception:
            pass
        splash.destroy()
        root.deiconify()
        root.lift()
        root.focus_force()

    root.after(duration_ms, _close)


class TeleopApp:
    def __init__(self, root):
        self.root = root
        root.title(APP_NAME)
        root.geometry("980x640")

        # Safe window icon (optional). If missing, silently continue.
        try:
            icon_path = resource_path("assets/T-logo.png")  # <— place file here or change name
            if os.path.exists(icon_path):
                _icon_img = Image.open(icon_path)
                _icon_photo = ImageTk.PhotoImage(_icon_img)
                root.iconphoto(True, _icon_photo)
        except Exception:
            pass

        self.sim = SimulationEnvironment()
        self.dof = self.sim.dof

        # ---- Thread-safe pose buffer ----
        self.keys = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
        self.pose_lock = threading.Lock()
        self.pose_cmd = [0.0, 0.0, 250.0, 0.0, 0.0, 0.0]  # default slider pose
        self.defaults = {"X": 0.0, "Y": 0.0, "Z": 250.0, "Roll": 0.0, "Pitch": 0.0, "Yaw": 0.0}

        # after() callback ids (for safe cancel)
        self._redraw_after_id = None
        self._poll_after_id = None

        self.running = False
        self.recording = False
        self.record_buffer = []
        self.update_thread = None

        # ORB template cache (set by image_control)
        self._orb_template = None

        # Camera thread management
        self._cam_thread = None
        self._cam_stop = threading.Event()

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

    def _make_scale(self, parent, label, mn, mx, val):
        frame = ttk.Frame(parent)
        frame.pack(padx=6, pady=6, fill="x")
        ttk.Label(frame, text=label).pack(anchor="w")
        s = ttk.Scale(frame, from_=mn, to=mx, orient="horizontal")
        s.set(val)
        s.pack(fill="x")
        v = tk.StringVar(value=f"{val:.1f}")
        ttk.Label(frame, textvariable=v, width=8).pack(side="right", padx=4)
        # Keep reference so we can update the label programmatically
        s._valvar = v

        def on_move(evt=None):
            v.set(f"{s.get():.1f}")

        s.bind("<B1-Motion>", on_move)
        s.bind("<ButtonRelease-1>", on_move)
        return s

    def reset_sliders(self):
        """Return all sliders to default positions and sync the pose buffer."""
        # 1) Update the shared pose buffer first (thread-safe)
        with self.pose_lock:
            self.pose_cmd = [self.defaults[k] for k in self.keys]
        # 2) Reflect on sliders (Tk main thread)
        for k, v in self.defaults.items():
            s = self.scales[k]
            s.set(v)
            if hasattr(s, "_valvar"):
                s._valvar.set(f"{v:.1f}")

    def _build_control_tab(self):
        f = self.frame_control

        # ---- left: sliders ----
        left = ttk.LabelFrame(f, text="Control Panel")
        left.pack(side="left", fill="y", padx=8, pady=8)

        self.scales = {}
        self.scales["X"]    = self._make_scale(left, "X (mm)", -300, 300, self.pose_cmd[0])
        self.scales["Y"]    = self._make_scale(left, "Y (mm)", -300, 300, self.pose_cmd[1])
        self.scales["Z"]    = self._make_scale(left, "Z (mm)",  -50, 550, self.pose_cmd[2])
        self.scales["Roll"] = self._make_scale(left, "Roll (deg)",  -180, 180, self.pose_cmd[3])
        self.scales["Pitch"]= self._make_scale(left, "Pitch (deg)", -180, 180, self.pose_cmd[4])
        self.scales["Yaw"]  = self._make_scale(left, "Yaw (deg)",   -180, 180, self.pose_cmd[5])

        ttk.Separator(left, orient="horizontal").pack(fill="x", padx=6, pady=(6, 2))
        ttk.Button(left, text="Reset Sliders", command=self.reset_sliders).pack(fill="x", padx=6, pady=6)

        # ---- right: session info ----
        right = ttk.LabelFrame(f, text="Session")
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        btn_row = ttk.Frame(right)
        btn_row.pack(fill="x", pady=(6, 6))

        self.btn_start = ttk.Button(btn_row, text="Start", command=self.start)
        self.btn_stop  = ttk.Button(btn_row, text="Stop", command=self.stop, state="disabled")
        self.btn_start.pack(side="left", padx=4)
        self.btn_stop.pack(side="left", padx=4)

        self.btn_record = ttk.Button(btn_row, text="Start Recording", command=self.toggle_record, state="disabled")
        self.btn_record.pack(side="left", padx=10)

        self.btn_save = ttk.Button(btn_row, text="Save CSV", command=self.save_csv, state="disabled")
        self.btn_save.pack(side="left", padx=4)

        self.btn_image = ttk.Button(btn_row, text="Image Recognition", command=self.image_control_stub)
        self.btn_image.pack(side="left", padx=10)

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

    def _build_replay_tab(self):
        f = self.frame_replay
        row = ttk.Frame(f)
        row.pack(fill="x", padx=8, pady=8)

        self.btn_load_csv = ttk.Button(row, text="Load CSV", command=self.load_csv)
        self.btn_play_csv = ttk.Button(row, text="Replay CSV", command=self.replay_csv, state="disabled")
        self.btn_load_csv.pack(side="left", padx=6)
        self.btn_play_csv.pack(side="left", padx=6)

        self.txt_preview = tk.Text(f, height=20)
        self.txt_preview.pack(fill="both", expand=True, padx=8, pady=8)

    def _build_help_tab(self):
        f = self.frame_help
        info = (
            "Teleoperation Panel — Quick Help\n\n"
            "• Open 3D view: Opens the visual arm simulation\n"
            "• Move sliders (XYZ mm, RPY deg) to set an EE target.\n"
            "• Start: begin streaming to 7-DOF SerialRobotSim; Stop to halt.\n"
            "• Start Recording to capture telemetry; Save CSV to export.\n"
            "• Load CSV on Replay tab then Replay CSV to drive the sim by file.\n"
            "• Image Control: pick a target image; arm nudges X/Y toward it.\n\n"
            "\n"
            "Threading Model:\n"
            "• Worker thread: Runs the physics and updates the arm pose.\n"
            "• Camera thread: Tracks an image and sends movement commands.\n"
            "• Main thread: Handles sliders, buttons, live telemetry. Also draws the 3D arm.\n"
        )
        ttk.Label(f, text=info, justify="left").pack(anchor="w", padx=8, pady=8)

    # ---------------- Thread-safe helpers ----------------
    def _get_pose_cmd_copy(self):
        with self.pose_lock:
            return list(self.pose_cmd)

    def _apply_pose_delta_threadsafe(self, dX_mm: float, dY_mm: float):
        """Apply deltas to the pose buffer, then reflect on sliders in main thread."""
        with self.pose_lock:
            new_pose = list(self.pose_cmd)
            new_pose[0] += dX_mm
            new_pose[1] += dY_mm
            self.pose_cmd = new_pose

        def _ui():
            self.scales["X"].set(new_pose[0])
            self.scales["Y"].set(new_pose[1])
            if hasattr(self.scales["X"], "_valvar"):
                self.scales["X"]._valvar.set(f"{new_pose[0]:.1f}")
            if hasattr(self.scales["Y"], "_valvar"):
                self.scales["Y"]._valvar.set(f"{new_pose[1]:.1f}")
        self.root.after(0, _ui)

    # Poll sliders from main thread and copy into buffer
    def _poll_sliders(self):
        try:
            pose = [float(self.scales[k].get()) for k in self.keys]
            with self.pose_lock:
                self.pose_cmd = pose
        except Exception:
            pass
        if self.running:
            self._poll_after_id = self.root.after(33, self._poll_sliders)  # ~30 Hz

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

        # start worker thread
        self.update_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.update_thread.start()

        # start main-thread redraw + slider polling
        self._schedule_redraw()
        self._poll_sliders()

    def _schedule_redraw(self):
        try:
            self.sim.redraw()
        except Exception:
            pass
        if self.running:
            self._redraw_after_id = self.root.after(33, self._schedule_redraw)  # ~30 FPS

    def stop(self):
        self.running = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.btn_record.config(state="disabled")
        self.lbl_status.config(text="Status: Stopped")
        if self.recording:
            self.toggle_record(force_off=True)

        # stop camera thread if running
        self._stop_camera_thread()

        # cancel scheduled callbacks safely
        if self._redraw_after_id:
            try: self.root.after_cancel(self._redraw_after_id)
            except Exception: pass
            self._redraw_after_id = None
        if self._poll_after_id:
            try: self.root.after_cancel(self._poll_after_id)
            except Exception: pass
            self._poll_after_id = None

    def toggle_record(self, force_off=False):
        if not self.running and not self.recording:
            messagebox.showinfo("Info", "Start the session before recording.")
            return
        if force_off or self.recording:
            self.recording = False
            self.btn_record.config(text="Start Recording")
            self.btn_save.config(state="normal" if self.record_buffer else "disabled")
            self.lbl_status.config(text="Status: Running (record stopped)")
        else:
            self.record_buffer = []
            self.recording = True
            self.btn_record.config(text="Stop Recording")
            self.btn_save.config(state="disabled")
            self.lbl_status.config(text="Status: Running (recording)")

    def _run_loop(self):
        last = time.time()
        while self.running:
            now = time.time()
            dt = now - last
            last = now

            pose = self._get_pose_cmd_copy()          # thread-safe read
            self.sim.set_target_ee_pose(pose)
            self.sim.step(dt)

            telem = self.sim.get_telemetry()
            ee = telem["ee_pose"]

            # schedule UI label updates on main thread
            def _ui():
                self.lbl_ee.config(
                    text=f"EE Pose: x={ee[0]:.1f} y={ee[1]:.1f} z={ee[2]:.1f} roll={ee[3]:.1f} pitch={ee[4]:.1f} yaw={ee[5]:.1f}"
                )
                self.lbl_joints.config(text=f"Joints: {['%.2f' % j for j in telem['joint_positions']]}")
                self.lbl_vel.config(text=f"Joint Vel: {['%.2f' % v for v in telem['joint_velocities']]}")
                self.lbl_torque.config(text=f"Torque: {['%.2f' % t for t in telem['joint_torques']]}")
            self.root.after(0, _ui)

            if self.recording:
                row = {
                    "t": telem["time"],
                    **{f"j{i+1}": telem["joint_positions"][i] for i in range(self.dof)},
                    **{f"vel{i+1}": telem["joint_velocities"][i] for i in range(self.dof)},
                    **{f"tau{i+1}": telem["joint_torques"][i] for i in range(self.dof)},
                    "x": ee[0], "y": ee[1], "z": ee[2],
                    "roll": ee[3], "pitch": ee[4], "yaw": ee[5],
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
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=default,
                                            filetypes=[("CSV files", "*.csv")])
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
                pose = [float(row.get(k, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]]
                # write pose buffer; reflect to sliders on main thread
                with self.pose_lock:
                    self.pose_cmd = pose

                def _ui():
                    vals = dict(zip(self.keys, pose))
                    for k, v in vals.items():
                        self.scales[k].set(v)
                        if hasattr(self.scales[k], "_valvar"):
                            self.scales[k]._valvar.set(f"{v:.1f}")
                self.root.after(0, _ui)

                self.sim.set_target_ee_pose(pose)
                self.sim.step(1.0 / 60.0)
                time.sleep(1.0 / 60.0)

                telem = self.sim.get_telemetry()
                ee = telem["ee_pose"]
                def _ui2():
                    self.lbl_ee.config(
                        text=f"EE Pose: x={ee[0]:.1f} y={ee[1]:.1f} z={ee[2]:.1f} roll={ee[3]:.1f} pitch={ee[4]:.1f} yaw={ee[5]:.1f}"
                    )
                self.root.after(0, _ui2)

            self.lbl_status.config(text="Status: Replay finished")

        threading.Thread(target=_replay, daemon=True).start()

    # ---------------- Image Control (ORB + Homography, thread-safe) ----------------
    def image_control_stub(self):
        """
        Image-based control: follow a specific planar image shown to the camera.
        Runs in a dedicated thread; never blocks Tk main loop; cleans up on Stop/close.
        """
        try:
            import cv2
            import numpy as np
        except Exception:
            messagebox.showwarning("OpenCV missing", "Install OpenCV:\n\npip install opencv-python")
            return

        # If already running, ignore
        if self._cam_thread and self._cam_thread.is_alive():
            messagebox.showinfo("Camera", "Image control is already running.")
            return

        # Pick the template once (on main thread)
        if self._orb_template is None:
            path = filedialog.askopenfilename(
                title="Choose target image (png/jpg)",
                filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")]
            )
            if not path:
                return
            tmpl_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if tmpl_gray is None:
                messagebox.showerror("Error", f"Couldn't load image:\n{path}")
                return
            orb = cv2.ORB_create(nfeatures=1200)
            kpT, desT = orb.detectAndCompute(tmpl_gray, None)
            if desT is None or len(kpT) < 8:
                messagebox.showerror("Error", "Not enough features in the chosen image. Pick a more textured image.")
                return
            self._orb_template = {
                "path": path,
                "img": tmpl_gray,
                "orb": orb,
                "kpT": kpT,
                "desT": desT,
                "bf": cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
                "size": tmpl_gray.shape[:2]  # (h, w)
            }

        # Start camera thread
        self._cam_stop.clear()
        self.btn_image.config(state="disabled")
        self._cam_thread = threading.Thread(
            target=self._run_camera_loop, args=(self._orb_template,), daemon=True
        )
        self._cam_thread.start()

    def _run_camera_loop(self, data):
        """Runs in a daemon thread. No Tk calls here (except via root.after in helper)."""
        import cv2
        import numpy as np

        orb, bf = data["orb"], data["bf"]
        kpT, desT = data["kpT"], data["desT"]
        tmpl_gray = data["img"]
        h_t, w_t = data["size"]

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            # Re-enable button on main thread
            self.root.after(0, lambda: self.btn_image.config(state="normal"))
            return

        # Tuning
        k_gain = 0.12        # px -> mm
        deadband = 6         # px
        min_good = 12
        ratio = 0.75

        try:
            while not self._cam_stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                kpF, desF = orb.detectAndCompute(gray, None)

                h, w = gray.shape[:2]
                cx, cy = w // 2, h // 2

                dx = dy = 0
                found = False

                if desF is not None:
                    matches = bf.knnMatch(desT, desF, k=2)
                    good = [m for m, n in matches if m.distance < ratio * n.distance]

                    if len(good) >= min_good:
                        src = np.float32([kpT[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        dst = np.float32([kpF[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

                        if H is not None:
                            corners = np.float32([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]]).reshape(-1, 1, 2)
                            proj = cv2.perspectiveTransform(corners, H).astype(int)
                            cv2.polylines(frame, [proj.reshape(-1, 2)], True, (0, 255, 255), 2)

                            pts = proj.reshape(-1, 2)
                            tx, ty = int(pts[:, 0].mean()), int(pts[:, 1].mean())

                            cv2.circle(frame, (tx, ty), 6, (0, 255, 255), -1)
                            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                            cv2.line(frame, (cx, cy), (tx, ty), (0, 255, 0), 2)

                            dx, dy = tx - cx, ty - cy
                            found = True

                # deadband
                if abs(dx) < deadband: dx = 0
                if abs(dy) < deadband: dy = 0

                # Apply thread-safe pose delta (invert image Y)
                if found:
                    self._apply_pose_delta_threadsafe(k_gain * dx, k_gain * (-dy))

                label = "FOUND" if found else "SEARCHING..."
                cv2.putText(frame, f"{label}  dx={dx}px dy={dy}px",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                cv2.imshow("Image Control (press q to close)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            try:
                cap.release()
            except Exception:
                pass
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            self.root.after(0, lambda: self.btn_image.config(state="normal"))

    def _stop_camera_thread(self):
        """Signal the camera thread to exit and wait briefly."""
        if self._cam_thread and self._cam_thread.is_alive():
            self._cam_stop.set()
            self._cam_thread.join(timeout=1.0)
        self._cam_thread = None
        self._cam_stop.clear()


def main():
    root = tk.Tk()

    # nice HiDPI scaling (best effort)
    try:
        root.tk.call("tk", "scaling", 1.2)
    except Exception:
        pass

    # SHOW SPLASH (~1.5s). Pass bundled logo if present; otherwise fallback text.
    splash_logo = resource_path("assets/FullTrackathonLogo.png")
    if not os.path.exists(splash_logo):
        splash_logo = None
    show_splash(root, logo_path=splash_logo, duration_ms=1500)

    app = TeleopApp(root)

    # Clean close: stop threads & cancel after() before destroying Tk
    def _on_close():
        try:
            app.stop()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()


if __name__ == "__main__":
    # Crash logger so a frozen EXE shows you the reason if anything fails
    try:
        main()
    except Exception:
        import traceback
        log = traceback.format_exc()
        try:
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            with open(os.path.join(desktop, "Trackathon_error.log"), "w", encoding="utf-8") as f:
                f.write(log)
        except Exception:
            pass
        raise
