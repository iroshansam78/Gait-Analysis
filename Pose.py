#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 14:13:49 2025

@author: iroshanpathirannahalage
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import torch
from ultralytics import YOLO
import time
import math

# -----------------------------------------------------------------------------
# 1) File picker via Tkinter
# -----------------------------------------------------------------------------
def select_video_file():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )

# -----------------------------------------------------------------------------
# 2) COCO-17 skeleton connectivity
# -----------------------------------------------------------------------------
SKELETON = [
    (15, 13), (13, 11),      # left ankle → knee → hip
    (16, 14), (14, 12),      # right ankle → knee → hip
    (11, 12),                # hips
    (5, 11), (6, 12),        # shoulders → corresponding hip
    (5, 6),                  # shoulders
    (5, 7), (7, 9),          # left shoulder → elbow → wrist
    (6, 8), (8, 10)          # right shoulder → elbow → wrist
]

# -----------------------------------------------------------------------------
# 3) Utility: angle between two vectors
# -----------------------------------------------------------------------------
def angle_between(v1, v2):
    dot = np.dot(v1, v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 * n2 == 0:
        return 0.0
    cosang = np.clip(dot / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

# -----------------------------------------------------------------------------
# 4) Main processing function
# -----------------------------------------------------------------------------
def process_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"❌ Could not open {path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    model = YOLO("yolov8n-pose.pt")

    # state for temporal metrics
    prev_mid_hip = None
    prev_step_len = None
    prev_diff = None
    prev_left_ankle_y = None
    prev_right_ankle_y = None
    step_count = 0
    double_support_frames = 0
    frame_idx = 0

    cv2.namedWindow("Gait Features", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # run pose inference
        results = model.predict(frame, device="cpu", verbose=False)[0]
        kpts = results.keypoints.data  # Tensor[N_people, 17, 3]

        annotated = frame.copy()
        if kpts is not None and kpts.shape[0] > 0:
            # process only the first detected person
            person = kpts[0].cpu().numpy()  # shape (17,3)
            if person.shape[0] == 17:
                pts = person[:, :2].astype(int)   # (17,2)
                conf = person[:, 2]               # (17,)

                # draw skeleton with confidence threshold
                for a, b in SKELETON:
                    if conf[a] > 0.3 and conf[b] > 0.3:
                        cv2.line(
                            annotated,
                            tuple(pts[a]),
                            tuple(pts[b]),
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA
                        )
                # draw joints
                for i, (x, y) in enumerate(pts):
                    if conf[i] > 0.3:
                        cv2.circle(
                            annotated,
                            (x, y),
                            4,
                            (0, 0, 255),
                            -1,
                            cv2.LINE_AA
                        )

                # compute gait features
                hip_mid = (pts[11] + pts[12]) / 2.0
                shoulder_mid = (pts[5] + pts[6]) / 2.0

                # trunk angle vs vertical
                trunk_vec = shoulder_mid - hip_mid
                vertical = np.array([0, -1])
                trunk_angle = angle_between(trunk_vec, vertical)

                # hip flexion
                thigh_left = pts[13] - pts[11]
                thigh_right = pts[14] - pts[12]
                hip_flex = (angle_between(trunk_vec, thigh_left) +
                            angle_between(trunk_vec, thigh_right)) / 2.0

                # knee flexion
                shank_left = pts[15] - pts[13]
                shank_right = pts[16] - pts[14]
                knee_flex = (angle_between(thigh_left, shank_left) +
                             angle_between(thigh_right, shank_right)) / 2.0

                # ankle dorsiflexion
                ankle_dorsi = (angle_between(shank_left, vertical) +
                               angle_between(shank_right, vertical)) / 2.0

                # step length (pixel distance between ankles)
                step_len = np.linalg.norm(pts[15] - pts[16])
                if prev_step_len is not None:
                    diff = step_len - prev_step_len
                    if prev_diff is not None and prev_diff > 0 and diff < 0:
                        step_count += 1
                    prev_diff = diff
                prev_step_len = step_len

                # cadence (steps per minute)
                elapsed = frame_idx / fps
                cadence = (step_count / elapsed * 60.0) if elapsed > 0 else 0.0

                # double support time
                if prev_left_ankle_y is not None and prev_right_ankle_y is not None:
                    v_l = abs(pts[15][1] - prev_left_ankle_y) * fps
                    v_r = abs(pts[16][1] - prev_right_ankle_y) * fps
                    if v_l < 30 and v_r < 30:
                        double_support_frames += 1
                prev_left_ankle_y = pts[15][1]
                prev_right_ankle_y = pts[16][1]
                double_support_time = double_support_frames / fps

                # walking speed (pixel/s) from hip midpoint displacement
                if prev_mid_hip is not None:
                    disp = np.linalg.norm(hip_mid - prev_mid_hip)
                    walking_speed = disp * fps
                else:
                    walking_speed = 0.0
                prev_mid_hip = hip_mid

                # overlay text
                stats = [
                    f"StepLen: {step_len:.1f}px",
                    f"Cadence: {cadence:.1f} spm",
                    f"DoubleSup: {double_support_time:.2f}s",
                    f"TrunkAng: {trunk_angle:.1f}°",
                    f"HipFlex: {hip_flex:.1f}°",
                    f"KneeFlex: {knee_flex:.1f}°",
                    f"AnkleDors: {ankle_dorsi:.1f}°",
                    f"Speed: {walking_speed:.1f}px/s"
                ]
                for i, txt in enumerate(stats):
                    cv2.putText(
                        annotated,
                        txt,
                        (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

        cv2.imshow("Gait Features", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------------------------------------------------------
# 5) Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    video_path = select_video_file()
    if video_path:
        process_video(video_path)
    else:
        print("No video selected.")