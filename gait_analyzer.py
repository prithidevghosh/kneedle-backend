"""
MediaPipe Pose pipeline for knee OA gait analysis.
Extracts biomechanical measurements from walking video.

Clinical basis:
- Knee angle measurement: 0.941 correlation with lab-grade equipment
  (Sato et al., 2019, Journal of Biomechanics)
- Toe-out angle modification reduces knee OA pain comparably to NSAIDs
  (Shull et al., 2013, Journal of Orthopaedic Research)
"""

import cv2
import numpy as np
import mediapipe as mp
import math
import base64
from models import GaitMetrics

mp_pose = mp.solutions.pose


def calculate_angle(a, b, c) -> float:
    """
    Calculate angle at point B given three 3D landmark points A, B, C.
    Returns angle in degrees (0-180).
    """
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = math.degrees(math.acos(cosine))
    return round(angle, 1)


def calculate_toe_out_angle(ankle, heel) -> float:
    """
    Calculate toe-out/in angle from ankle and heel landmarks.
    Positive = toe out, negative = toe in.
    """
    dx = ankle.x - heel.x
    dy = ankle.y - heel.y
    angle = math.degrees(math.atan2(dx, dy))
    return round(angle, 1)


def calculate_trunk_lean(left_shoulder, right_shoulder, left_hip, right_hip):
    """
    Calculate trunk lateral lean angle and direction.
    """
    shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
    hip_mid_x = (left_hip.x + right_hip.x) / 2
    shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
    hip_mid_y = (left_hip.y + right_hip.y) / 2

    dx = shoulder_mid_x - hip_mid_x
    dy = abs(shoulder_mid_y - hip_mid_y) + 1e-8
    angle = abs(math.degrees(math.atan(dx / dy)))

    direction = "neutral"
    if angle > 3:
        direction = "right" if dx > 0 else "left"

    return round(angle, 1), direction


def extract_key_frames(video_path: str, num_frames: int = 5) -> list[str]:
    """
    Extract evenly spaced key frames from video.
    Returns list of base64-encoded JPEG strings for Gemma 4.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0 or fps <= 0:
        cap.release()
        return []

    # Skip first and last 10% to avoid static start/stop
    start = int(total_frames * 0.1)
    end = int(total_frames * 0.9)
    indices = np.linspace(start, end, num_frames, dtype=int)

    frames_b64 = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize to reduce token count while preserving detail
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buffer).decode('utf-8')
        frames_b64.append(b64)

    cap.release()
    return frames_b64


def analyse_gait(video_path: str) -> tuple[GaitMetrics, list[str]]:
    """
    Main pipeline: run MediaPipe Pose on video frames.
    Returns GaitMetrics (measurements) and key frames (base64) for Gemma 4.

    MediaPipe landmark indices used:
    - 11: left shoulder, 12: right shoulder
    - 23: left hip, 24: right hip
    - 25: left knee, 26: right knee
    - 27: left ankle, 28: right ankle
    - 29: left heel, 30: right heel
    - 31: left foot index, 32: right foot index
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample every 5th frame to balance accuracy and speed
    sample_interval = max(1, int(fps / 6))

    knee_angles_right = []
    knee_angles_left = []
    trunk_leans = []
    trunk_directions = []
    toe_out_right = []
    toe_out_left = []
    confidences = []
    step_count = 0
    prev_ankle_y = None

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if not result.pose_landmarks:
                frame_idx += 1
                continue

            lm = result.pose_landmarks.landmark

            conf = np.mean([
                lm[25].visibility, lm[26].visibility,
                lm[27].visibility, lm[28].visibility
            ])
            confidences.append(conf)

            if conf < 0.4:
                frame_idx += 1
                continue

            # Right knee angle (hip → knee → ankle)
            r_knee = calculate_angle(lm[24], lm[26], lm[28])
            knee_angles_right.append(r_knee)

            # Left knee angle
            l_knee = calculate_angle(lm[23], lm[25], lm[27])
            knee_angles_left.append(l_knee)

            # Trunk lean
            lean_angle, lean_dir = calculate_trunk_lean(
                lm[11], lm[12], lm[23], lm[24]
            )
            trunk_leans.append(lean_angle)
            trunk_directions.append(lean_dir)

            # Toe-out angles
            if lm[28].visibility > 0.4 and lm[30].visibility > 0.4:
                toe_out_right.append(calculate_toe_out_angle(lm[28], lm[30]))
            if lm[27].visibility > 0.4 and lm[29].visibility > 0.4:
                toe_out_left.append(calculate_toe_out_angle(lm[27], lm[29]))

            # Simple step counter via ankle vertical movement
            current_ankle_y = (lm[27].y + lm[28].y) / 2
            if prev_ankle_y is not None:
                if abs(current_ankle_y - prev_ankle_y) > 0.01:
                    step_count += 1
            prev_ankle_y = current_ankle_y

            frame_idx += 1

    cap.release()

    def safe_mean(lst): return round(float(np.mean(lst)), 1) if lst else None

    r_angle = safe_mean(knee_angles_right)
    l_angle = safe_mean(knee_angles_left)

    symmetry = None
    if r_angle and l_angle:
        diff = abs(r_angle - l_angle)
        symmetry = round(max(0, 100 - (diff * 2.5)), 1)

    trunk_dir = "neutral"
    if trunk_directions:
        from collections import Counter
        trunk_dir = Counter(trunk_directions).most_common(1)[0][0]

    duration_sec = total_frames / fps if fps > 0 else 1
    cadence = round((step_count / duration_sec) * 60, 1) if duration_sec > 0 else None

    metrics = GaitMetrics(
        knee_angle_right=r_angle,
        knee_angle_left=l_angle,
        knee_angle_diff=round(abs(r_angle - l_angle), 1) if r_angle and l_angle else None,
        symmetry_score=symmetry,
        trunk_lean_angle=safe_mean(trunk_leans),
        trunk_lean_direction=trunk_dir,
        toe_out_angle_right=safe_mean(toe_out_right),
        toe_out_angle_left=safe_mean(toe_out_left),
        cadence=cadence,
        frames_analysed=len(knee_angles_right),
        confidence=round(float(np.mean(confidences)), 2) if confidences else 0.0
    )

    key_frames = extract_key_frames(video_path, num_frames=2)

    return metrics, key_frames
