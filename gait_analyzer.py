"""
Dual-video MediaPipe Pose pipeline for knee OA gait analysis.

Frontal view  → varus/valgus thrust, pelvic obliquity, FPPA, trunk lateral lean,
                step width.
Sagittal view → phase-isolated knee angles, hip extension, ankle dorsiflexion,
                trunk anterior lean, gait speed proxy.
Both views    → heel-strike segmentation, temporal parameters, KL-proxy scoring.

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
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from scipy.signal import find_peaks
from models import GaitMetrics

mp_pose = mp.solutions.pose

# ─── Pipeline tuning constants ────────────────────────────────────────────────
# Effective per-second sample rate after frame skipping. 30 fps gives ~30 frames
# per gait cycle, enough for the 10%-wide loading_response window to land 2-3
# samples per cycle.
_TARGET_SAMPLE_FPS = 30.0

# Minimum mean lower-body landmark visibility for a frame to count as analysed.
# Below this MediaPipe is too unsure to feed clinical metrics.
_MIN_FRAME_CONFIDENCE = 0.6

# Minimum per-landmark visibility for that landmark to participate in an angle
# calculation. Lower than the frame floor since one bad landmark hurts less than
# a frame full of bad landmarks.
_MIN_LANDMARK_VIS = 0.5


# ─── Gait phase boundaries (% of stride from heel strike) ─────────────────────
_PHASES = [
    (0,   10,  "loading_response"),
    (10,  30,  "mid_stance"),
    (30,  50,  "terminal_stance"),
    (50,  60,  "pre_swing"),
    (60,  73,  "initial_swing"),
    (73,  87,  "mid_swing"),
    (87, 100,  "terminal_swing"),
]


def _phase_for_pct(pct: float) -> str:
    for lo, hi, name in _PHASES:
        if lo <= pct < hi:
            return name
    return "terminal_swing"


# ─── Internal data structures ──────────────────────────────────────────────────

@dataclass
class KneePhaseAngles:
    loading_response_peak: Optional[float]
    mid_stance_angle: Optional[float]
    peak_swing_flexion: Optional[float]
    rom_delta: Optional[float]
    extension_lag: Optional[float]
    avg_full_cycle: float   # backward-compat: mean across all frames


@dataclass
class GaitParams:
    knee: dict              # {'right': KneePhaseAngles, 'left': KneePhaseAngles}
    right_varus_valgus_thrust: float = 0.0
    left_varus_valgus_thrust: float = 0.0
    pelvic_obliquity_deg: float = 0.0
    trendelenburg_flag: bool = False
    trunk_lateral_lean_deg: float = 0.0
    trunk_lean_direction: str = "neutral"
    step_width_proxy: float = 0.0
    fppa_right: float = 0.0
    fppa_left: float = 0.0
    hip_extension_terminal_stance: Optional[float] = None
    ankle_dorsiflexion: Optional[float] = None
    trunk_anterior_lean_deg: float = 0.0
    cadence: float = 0.0
    stride_time_asymmetry: float = 0.0
    double_support_ratio: float = 20.0
    gait_speed_proxy: float = 0.0
    symmetry_score: float = 0.0
    frontal_confidence: float = 0.0
    sagittal_confidence: float = 0.0
    frontal_frames_analyzed: int = 0
    frontal_frames_skipped: int = 0
    sagittal_frames_analyzed: int = 0
    sagittal_frames_skipped: int = 0
    heel_strike_events_right: int = 0
    heel_strike_events_left: int = 0
    gait_cycles_detected: int = 0
    fallback_mode: bool = False


# ─── Geometry helpers ──────────────────────────────────────────────────────────

def calculate_angle(a, b, c) -> float:
    """Angle at B in degrees (0-180) given three landmarks.

    Uses 2D image-plane projection (x, y only). MediaPipe's z estimate is
    derived from a single camera and is too noisy for clinical angle math —
    including it produced impossible joint angles (e.g. 122° in stance).
    """
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return round(math.degrees(math.acos(np.clip(cos, -1.0, 1.0))), 1)


def calculate_toe_out_angle(ankle, heel) -> float:
    return round(math.degrees(math.atan2(ankle.x - heel.x, ankle.y - heel.y)), 1)


def _safe_mean(lst) -> Optional[float]:
    return round(float(np.mean(lst)), 1) if lst else None


def _smooth(series: list, window: int = 5) -> list:
    if len(series) < window:
        return series
    arr = np.array(series, dtype=float)
    padded = np.pad(arr, (window // 2, window // 2), mode='edge')
    return np.convolve(padded, np.ones(window) / window, mode='valid')[:len(series)].tolist()


# ─── Gait cycle segmentation ───────────────────────────────────────────────────

def detect_heel_strikes(ankle_y: list, visibility: list, sample_fps: float = _TARGET_SAMPLE_FPS) -> list[int]:
    """Find heel-strike events as peaks in ankle-Y series (image coords, Y↓).

    `sample_fps` is the effective per-second rate of the input series so the
    minimum-distance constraint scales correctly. 0.35 s ≈ ~170 strikes/min/foot
    upper bound, which is well above human cadence.
    """
    if len(ankle_y) < 3:
        return []
    smoothed = _smooth(ankle_y)
    min_distance = max(3, int(round(sample_fps * 0.35)))
    # Prominence 0.03 (3% of frame height) discards MediaPipe tracking jitter
    # while keeping real heel-strike excursions (5-10% of frame). At 0.01 the
    # detector was emitting 2-3 phantom strikes per real one, which inflated
    # cadence into the 200+ range and broke every temporal metric downstream.
    peaks, _ = find_peaks(smoothed, distance=min_distance, prominence=0.03)
    return [int(p) for p in peaks if visibility[p] > _MIN_LANDMARK_VIS]


def _label_phases(n_frames: int, hs_indices: list[int]) -> dict[int, str]:
    """Map sampled-frame index → gait phase label for one side."""
    labels: dict[int, str] = {}
    if len(hs_indices) < 2:
        return labels
    for i in range(len(hs_indices) - 1):
        hs, next_hs = hs_indices[i], hs_indices[i + 1]
        cycle_len = next_hs - hs
        if cycle_len <= 0:
            continue
        for f in range(hs, next_hs):
            labels[f] = _phase_for_pct((f - hs) / cycle_len * 100)
    return labels


# ─── MediaPipe runner ──────────────────────────────────────────────────────────

def _run_mediapipe(video_path: str) -> dict:
    """
    Run MediaPipe Pose on all sampled frames of a video.
    Returns a dict with per-frame data and aggregate metadata.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, int(round(fps / _TARGET_SAMPLE_FPS)))
    effective_fps = fps / sample_interval

    frames_data: list[dict] = []
    confidences: list[float] = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
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
            time_sec = frame_idx / fps

            if not result.pose_landmarks:
                frames_data.append({
                    "frame_idx": frame_idx,
                    "sampled_idx": len(frames_data),
                    "time": time_sec,
                    "landmarks": None,
                    "confidence": 0.0,
                })
                frame_idx += 1
                continue

            lm = result.pose_landmarks.landmark
            # Per-side mean visibility (hip + knee + ankle). In a sagittal view
            # the far side is occluded, so taking the global mean would reject
            # frames where the camera-side leg is perfectly tracked. Use the
            # better-visible side as the frame confidence instead.
            right_vis = float(np.mean([lm[24].visibility, lm[26].visibility, lm[28].visibility]))
            left_vis  = float(np.mean([lm[23].visibility, lm[25].visibility, lm[27].visibility]))
            conf = max(right_vis, left_vis)
            confidences.append(conf)
            frames_data.append({
                "frame_idx": frame_idx,
                "sampled_idx": len(frames_data),
                "time": time_sec,
                "landmarks": lm,
                "confidence": conf,
            })
            frame_idx += 1

    cap.release()

    frames_analyzed = sum(1 for f in frames_data if f["landmarks"] and f["confidence"] >= _MIN_FRAME_CONFIDENCE)
    frames_skipped = len(frames_data) - frames_analyzed
    mean_conf = float(np.mean(confidences)) if confidences else 0.0

    return {
        "frames": frames_data,
        "fps": fps,
        "effective_fps": effective_fps,
        "total_frames": total_frames,
        "duration_sec": total_frames / fps if fps > 0 else 0.0,
        "frames_analyzed": frames_analyzed,
        "frames_skipped": frames_skipped,
        "confidence": round(mean_conf, 3),
    }


# ─── Sagittal-view analysis ────────────────────────────────────────────────────

def _extract_sagittal(sag: dict) -> dict:
    frames = sag["frames"]
    sample_fps = sag.get("effective_fps", _TARGET_SAMPLE_FPS)

    def _ankle_series(lm_idx: int):
        ys, vis = [], []
        for f in frames:
            lm = f["landmarks"]
            if (lm and f["confidence"] >= _MIN_FRAME_CONFIDENCE
                    and lm[lm_idx].visibility > _MIN_LANDMARK_VIS):
                ys.append(lm[lm_idx].y)
                vis.append(lm[lm_idx].visibility)
            else:
                ys.append(0.0)
                vis.append(0.0)
        return ys, vis

    hs_right = detect_heel_strikes(*_ankle_series(28), sample_fps=sample_fps)
    hs_left  = detect_heel_strikes(*_ankle_series(27), sample_fps=sample_fps)
    phase_r  = _label_phases(len(frames), hs_right)
    phase_l  = _label_phases(len(frames), hs_left)

    knee_by_phase = {
        side: {p: [] for _, _, p in _PHASES}
        for side in ("right", "left")
    }
    knee_all: dict[str, list[float]] = {"right": [], "left": []}

    hip_ext_vals:  list[float] = []
    ankle_dors_vals: list[float] = []
    trunk_ant_vals: list[float] = []
    hip_x_series: list[tuple] = []   # (time, hip_x, hip_ankle_dist)

    for si, f in enumerate(frames):
        lm = f["landmarks"]
        if not lm or f["confidence"] < _MIN_FRAME_CONFIDENCE:
            continue

        # Store knee values as clinical FLEXION (0° = full extension, larger =
        # more bend), so KL thresholds and "extension lag" semantics match
        # standard biomechanics references.
        # Per-side visibility gating: in a sagittal video only one leg is
        # facing the camera; recording the occluded side would feed garbage
        # into phase angles and KL scoring.
        r_visible = (lm[24].visibility > _MIN_LANDMARK_VIS
                     and lm[26].visibility > _MIN_LANDMARK_VIS
                     and lm[28].visibility > _MIN_LANDMARK_VIS)
        l_visible = (lm[23].visibility > _MIN_LANDMARK_VIS
                     and lm[25].visibility > _MIN_LANDMARK_VIS
                     and lm[27].visibility > _MIN_LANDMARK_VIS)

        if r_visible:
            r_knee = 180.0 - calculate_angle(lm[24], lm[26], lm[28])
            knee_all["right"].append(r_knee)
            if si in phase_r:
                knee_by_phase["right"][phase_r[si]].append(r_knee)

        if l_visible:
            l_knee = 180.0 - calculate_angle(lm[23], lm[25], lm[27])
            knee_all["left"].append(l_knee)
            if si in phase_l:
                knee_by_phase["left"][phase_l[si]].append(l_knee)

        # Hip extension (terminal stance, right side as primary)
        if phase_r.get(si) == "terminal_stance":
            if lm[24].visibility > _MIN_LANDMARK_VIS and lm[26].visibility > _MIN_LANDMARK_VIS:
                dx = lm[26].x - lm[24].x
                dy = abs(lm[26].y - lm[24].y) + 1e-8
                hip_ext_vals.append(math.degrees(math.atan2(abs(dx), dy)))

        # Ankle dorsiflexion (mid_stance, right side)
        if phase_r.get(si) == "mid_stance":
            if (lm[26].visibility > _MIN_LANDMARK_VIS and lm[28].visibility > _MIN_LANDMARK_VIS
                    and lm[32].visibility > _MIN_LANDMARK_VIS):
                ankle_dors_vals.append(calculate_angle(lm[26], lm[28], lm[32]))

        # Trunk anterior lean (sagittal, all frames)
        sh_x = (lm[11].x + lm[12].x) / 2
        hp_x = (lm[23].x + lm[24].x) / 2
        sh_y = (lm[11].y + lm[12].y) / 2
        hp_y = (lm[23].y + lm[24].y) / 2
        trunk_ant_vals.append(abs(math.degrees(math.atan(
            (sh_x - hp_x) / (abs(sh_y - hp_y) + 1e-8)
        ))))

        # Gait speed proxy: hip X displacement / duration / body-height-proxy
        hip_x = (lm[23].x + lm[24].x) / 2
        ankle_y_mid = (lm[27].y + lm[28].y) / 2
        hip_ankle_dist = abs(ankle_y_mid - hp_y) + 1e-6
        hip_x_series.append((f["time"], hip_x, hip_ankle_dist))

    def _build_kpa(side: str) -> KneePhaseAngles:
        # All values in flexion units (0° = straight leg, larger = more bend).
        kp = knee_by_phase[side]
        lr = _safe_mean(kp["loading_response"])
        ms = _safe_mean(kp["mid_stance"])
        swing_vals = kp["initial_swing"] + kp["mid_swing"]
        # Peak swing flexion = most-bent frame in swing → max(flexion).
        sw = round(float(max(swing_vals)), 1) if swing_vals else None
        all_v = knee_all[side]
        # Extension lag = least-bent frame across the cycle. Healthy ≈ 0-2°;
        # >10° indicates the knee never fully straightens (flexion contracture).
        ext_lag = round(float(min(all_v)), 1) if all_v else None
        rd = round(sw - ms, 1) if sw is not None and ms is not None else None
        avg = round(float(np.mean(all_v)), 1) if all_v else 0.0
        return KneePhaseAngles(
            loading_response_peak=lr,
            mid_stance_angle=ms,
            peak_swing_flexion=sw,
            rom_delta=rd,
            extension_lag=ext_lag,
            avg_full_cycle=avg,
        )

    gait_speed_proxy = 0.0
    if len(hip_x_series) >= 2:
        t0, x0, _ = hip_x_series[0]
        t1, x1, _ = hip_x_series[-1]
        dur = t1 - t0
        disp = abs(x1 - x0)
        mean_bh = float(np.mean([h for _, _, h in hip_x_series])) + 1e-6
        if dur > 0:
            gait_speed_proxy = round(disp / dur / mean_bh, 3)

    return {
        "knee_right": _build_kpa("right"),
        "knee_left":  _build_kpa("left"),
        "hip_extension_terminal_stance": _safe_mean(hip_ext_vals),
        "ankle_dorsiflexion": _safe_mean(ankle_dors_vals),
        "trunk_anterior_lean_deg": round(float(np.mean(trunk_ant_vals)), 1) if trunk_ant_vals else 0.0,
        "hs_right": hs_right,
        "hs_left":  hs_left,
        "gait_speed_proxy": gait_speed_proxy,
        "phase_right": phase_r,
    }


# ─── Frontal-view analysis ─────────────────────────────────────────────────────

def _compute_vvt(hip, knee, ankle) -> float:
    """Signed lateral deviation of knee from hip-ankle line, expressed as
    percent of leg length so the value is camera-distance invariant."""
    h = np.array([hip.x, hip.y])
    k = np.array([knee.x, knee.y])
    a = np.array([ankle.x, ankle.y])
    ha = a - h
    hk = k - h
    leg_len_sq = float(np.dot(ha, ha))
    if leg_len_sq < 1e-10:
        return 0.0
    leg_len = math.sqrt(leg_len_sq)
    proj = (np.dot(hk, ha) / leg_len_sq) * ha
    dev = hk - proj
    sign = float(np.sign(float(np.cross(ha, hk))))
    return round(sign * float(np.linalg.norm(dev)) / leg_len * 100, 2)


def _pelvic_obliquity(left_hip, right_hip) -> float:
    """Pelvic tilt in degrees: 0 = level, ±90 = vertical pelvis.

    Uses |dx| in the denominator so the sign of dx (which depends on whether
    the subject is facing the camera or walking away) does not flip the angle
    into the ±180° quadrant. Result range is [-90°, +90°].
    """
    dy = right_hip.y - left_hip.y
    dx = abs(right_hip.x - left_hip.x) + 1e-8
    return float(np.degrees(np.arctan2(dy, dx)))


def _extract_frontal(fro: dict) -> dict:
    frames = fro["frames"]
    sample_fps = fro.get("effective_fps", _TARGET_SAMPLE_FPS)

    def _ankle_series(lm_idx: int):
        ys, vis = [], []
        for f in frames:
            lm = f["landmarks"]
            if (lm and f["confidence"] >= _MIN_FRAME_CONFIDENCE
                    and lm[lm_idx].visibility > _MIN_LANDMARK_VIS):
                ys.append(lm[lm_idx].y)
                vis.append(lm[lm_idx].visibility)
            else:
                ys.append(0.0)
                vis.append(0.0)
        return ys, vis

    hs_right_f = detect_heel_strikes(*_ankle_series(28), sample_fps=sample_fps)
    hs_left_f  = detect_heel_strikes(*_ankle_series(27), sample_fps=sample_fps)
    phase_r_f  = _label_phases(len(frames), hs_right_f)
    phase_l_f  = _label_phases(len(frames), hs_left_f)

    vvt_r, vvt_l = [], []
    fppa_r, fppa_l = [], []
    pelvic_r_ms, pelvic_l_ms = [], []
    trunk_lat, trunk_dirs = [], []
    step_widths: list[float] = []

    for hs_idx in hs_right_f + hs_left_f:
        if hs_idx < len(frames):
            f = frames[hs_idx]
            if f["landmarks"]:
                lm = f["landmarks"]
                step_widths.append(abs(lm[28].x - lm[27].x))

    for si, f in enumerate(frames):
        lm = f["landmarks"]
        if not lm or f["confidence"] < _MIN_FRAME_CONFIDENCE:
            continue

        # FPPA / VVT are only meaningful when the subject is squarely facing
        # (or directly away from) the camera. If they're walking diagonally,
        # sagittal-plane motion projects into the frontal image and inflates
        # both metrics. Hip-line width in image space is a cheap proxy for
        # frontal alignment; <0.08 (≈8% of frame width) means the pelvis is
        # heavily rotated and we skip frontal-plane angle collection.
        hip_line_width = abs(lm[24].x - lm[23].x)
        frontal_aligned = hip_line_width >= 0.08

        if frontal_aligned and phase_r_f.get(si) == "loading_response":
            if all(lm[i].visibility > _MIN_LANDMARK_VIS for i in (24, 26, 28)):
                vvt_r.append(_compute_vvt(lm[24], lm[26], lm[28]))
                fppa_r.append(calculate_angle(lm[24], lm[26], lm[28]))
        if frontal_aligned and phase_l_f.get(si) == "loading_response":
            if all(lm[i].visibility > _MIN_LANDMARK_VIS for i in (23, 25, 27)):
                vvt_l.append(_compute_vvt(lm[23], lm[25], lm[27]))
                fppa_l.append(calculate_angle(lm[23], lm[25], lm[27]))

        # Pelvic obliquity ALSO requires frontal alignment — when the pelvis
        # rotates toward the depth axis, |dx| collapses and arctan2(dy, |dx|)
        # explodes toward 90°, falsely tripping trendelenburg.
        if frontal_aligned and phase_r_f.get(si) == "mid_stance":
            if lm[23].visibility > _MIN_LANDMARK_VIS and lm[24].visibility > _MIN_LANDMARK_VIS:
                pelvic_r_ms.append(_pelvic_obliquity(lm[23], lm[24]))

        if frontal_aligned and phase_l_f.get(si) == "mid_stance":
            if lm[23].visibility > _MIN_LANDMARK_VIS and lm[24].visibility > _MIN_LANDMARK_VIS:
                pelvic_l_ms.append(_pelvic_obliquity(lm[23], lm[24]))

        # Trunk lateral lean — mid_stance frames, frontal-aligned only.
        if frontal_aligned and (phase_r_f.get(si) == "mid_stance"
                                or phase_l_f.get(si) == "mid_stance"):
            sh_x = (lm[11].x + lm[12].x) / 2
            hp_x = (lm[23].x + lm[24].x) / 2
            sh_y = (lm[11].y + lm[12].y) / 2
            hp_y = (lm[23].y + lm[24].y) / 2
            dx = sh_x - hp_x
            dy = abs(sh_y - hp_y) + 1e-8
            trunk_lat.append(abs(math.degrees(math.atan(dx / dy))))
            trunk_dirs.append("right" if dx > 0.01 else ("left" if dx < -0.01 else "neutral"))

    all_obliq = pelvic_r_ms + pelvic_l_ms
    # Use the 75th percentile of |obliquity| rather than max — a single
    # mistracked frame should not override an otherwise level pelvis.
    if all_obliq:
        abs_obliq = sorted(abs(a) for a in all_obliq)
        pelvic_obliq = round(float(np.percentile(abs_obliq, 75)), 1)
    else:
        pelvic_obliq = 0.0
    trendelenburg = pelvic_obliq > 10.0

    trunk_lean = round(float(np.mean(trunk_lat)), 1) if trunk_lat else 0.0
    trunk_dir = Counter(trunk_dirs).most_common(1)[0][0] if trunk_dirs else "neutral"

    def _fppa_dev(vals):
        return round(abs(180.0 - float(np.mean(vals))), 1) if vals else 0.0

    return {
        "right_vvt": round(float(np.mean(vvt_r)), 2) if vvt_r else 0.0,
        "left_vvt":  round(float(np.mean(vvt_l)), 2) if vvt_l else 0.0,
        "pelvic_obliquity_deg": pelvic_obliq,
        "trendelenburg_flag": trendelenburg,
        "trunk_lateral_lean_deg": trunk_lean,
        "trunk_lean_direction": trunk_dir,
        "step_width_proxy": round(float(np.mean(step_widths)), 3) if step_widths else 0.0,
        "fppa_right": _fppa_dev(fppa_r),
        "fppa_left":  _fppa_dev(fppa_l),
        "hs_right_frontal": hs_right_f,
        "hs_left_frontal":  hs_left_f,
        "phase_right_frontal": phase_r_f,
    }


# ─── Temporal parameters ───────────────────────────────────────────────────────

def _extract_temporal(primary: dict, fallback: dict, primary_res: dict, fallback_res: dict) -> dict:
    """
    Compute cadence, stride asymmetry, double-support ratio.
    Uses sagittal as primary; falls back to frontal if <3 heel strikes detected.
    """
    frames_p = primary["frames"]
    hs_r_key = "hs_right"
    hs_l_key = "hs_left"

    hs_r = primary_res.get(hs_r_key, [])
    hs_l = primary_res.get(hs_l_key, [])

    if len(hs_r) + len(hs_l) < 3:
        frames_p = fallback["frames"]
        hs_r = fallback_res.get("hs_right_frontal", [])
        hs_l = fallback_res.get("hs_left_frontal", [])

    def _times(hs_indices):
        return [frames_p[i]["time"] for i in hs_indices if i < len(frames_p)]

    r_times = _times(hs_r)
    l_times = _times(hs_l)

    stride_r = [r_times[i+1] - r_times[i] for i in range(len(r_times) - 1)]
    stride_l = [l_times[i+1] - l_times[i] for i in range(len(l_times) - 1)]

    cadence = 0.0
    all_strikes = sorted(r_times + l_times)
    if len(all_strikes) >= 2:
        dur = all_strikes[-1] - all_strikes[0]
        if dur > 0:
            cadence = round((len(all_strikes) - 1) / dur * 60, 1)

    stride_asym = 0.0
    if stride_r and stride_l:
        mr, ml = float(np.mean(stride_r)), float(np.mean(stride_l))
        mean_all = (mr + ml) / 2
        if mean_all > 0:
            stride_asym = round(abs(mr - ml) / mean_all * 100, 1)

    # Double support: approximate as overlap between right and left stance windows
    double_support = 20.0
    if stride_r and stride_l and r_times and l_times:
        r_stance = [(r_times[i], r_times[i] + 0.6 * stride_r[i])
                    for i in range(min(len(r_times) - 1, len(stride_r)))]
        l_stance = [(l_times[i], l_times[i] + 0.6 * stride_l[i])
                    for i in range(min(len(l_times) - 1, len(stride_l)))]
        overlaps = []
        for rs in r_stance:
            for ls in l_stance:
                ov_s, ov_e = max(rs[0], ls[0]), min(rs[1], ls[1])
                if ov_e > ov_s and (rs[1] - rs[0]) > 0:
                    overlaps.append((ov_e - ov_s) / (rs[1] - rs[0]) * 100)
        if overlaps:
            double_support = round(float(np.mean(overlaps)), 1)

    gait_cycles = max(len(stride_r), len(stride_l))
    return {
        "cadence": cadence,
        "stride_time_asymmetry": stride_asym,
        "double_support_ratio": double_support,
        "gait_cycles_detected": gait_cycles,
        "heel_strike_events_right": len(hs_r),
        "heel_strike_events_left": len(hs_l),
    }


# ─── KL-proxy scoring ──────────────────────────────────────────────────────────

def _compute_kl_proxy(params: GaitParams) -> tuple[float, str, list[str]]:
    score = 0.0
    flags: list[str] = []

    for side in ("right", "left"):
        knee = params.knee[side]
        if knee.loading_response_peak is not None:
            if knee.loading_response_peak < 5:
                score += 2; flags.append(f"{side}_loading_response_absent")
            elif knee.loading_response_peak < 10:
                score += 1; flags.append(f"{side}_loading_response_reduced")
        # Swing flexion thresholds calibrated to MediaPipe single-camera 2D,
        # not to lab marker-based 3D. Validation studies on healthy subjects
        # show ~25-40° peak flexion via MediaPipe vs ~60° via marker capture,
        # so the lab-derived thresholds (35/45) generated bilateral severe
        # flags on every healthy walker.
        if knee.peak_swing_flexion is not None:
            if knee.peak_swing_flexion < 18:
                score += 2; flags.append(f"{side}_swing_flexion_severe")
            elif knee.peak_swing_flexion < 28:
                score += 1; flags.append(f"{side}_swing_flexion_reduced")
        if knee.extension_lag is not None and knee.extension_lag > 10:
            score += 1; flags.append(f"{side}_flexion_contracture")

    # VVT is in % of leg length. Q-angle anatomy and small camera-perspective
    # error easily produce 3-5% deviation in healthy subjects, so the band
    # has to sit above that.
    if abs(params.right_varus_valgus_thrust) > 8 or abs(params.left_varus_valgus_thrust) > 8:
        score += 2; flags.append("significant_varus_valgus_thrust")
    elif abs(params.right_varus_valgus_thrust) > 5 or abs(params.left_varus_valgus_thrust) > 5:
        score += 1; flags.append("mild_varus_valgus_thrust")

    if params.trendelenburg_flag:
        score += 1; flags.append("trendelenburg_positive")

    if params.trunk_lateral_lean_deg > 8:
        score += 1; flags.append("significant_trunk_lean")

    # Threshold accounts for anatomical Q-angle (12-20° in healthy adults)
    # plus camera-perspective error. Only flag deviations clearly outside that.
    if abs(params.fppa_right) > 15 or abs(params.fppa_left) > 15:
        score += 1; flags.append("fppa_deviation")

    # Temporal metrics need MIN strikes per side, not max. The previous
    # `gait_cycles_detected = max(len(stride_r), len(stride_l))` gate let
    # asymmetry fire when one side had 3 strikes and the other only 1
    # (i.e. 1 vs 3 stride comparison — pure noise). Require ≥3 strikes
    # per side ⇒ ≥2 strides per side for a meaningful comparison. Also
    # cap on physiological plausibility — cadence>180 or double-support>50%
    # is almost certainly false-peak heel-strike detection, not pathology.
    min_strikes = min(params.heel_strike_events_right, params.heel_strike_events_left)
    cadence_plausible = 40 <= params.cadence <= 180
    ds_plausible = params.double_support_ratio <= 50

    if min_strikes >= 3 and cadence_plausible and ds_plausible:
        if params.double_support_ratio > 35:
            score += 2; flags.append("high_double_support")
        elif params.double_support_ratio > 28:
            score += 1; flags.append("elevated_double_support")

        if params.stride_time_asymmetry > 15:
            score += 1; flags.append("high_stride_asymmetry")

        if params.cadence < 70:
            score += 1; flags.append("low_cadence")

    if params.hip_extension_terminal_stance is not None and params.hip_extension_terminal_stance < 5:
        score += 1; flags.append("reduced_hip_extension")

    if params.ankle_dorsiflexion is not None and params.ankle_dorsiflexion < 5:
        score += 1; flags.append("reduced_ankle_dorsiflexion")

    kl_map = [(3, "kl_0"), (5, "kl_1"), (8, "kl_2"), (11, "kl_3"), (16, "kl_4")]
    kl_grade = next(label for threshold, label in kl_map if score <= threshold)
    return float(score), kl_grade, flags


def _check_bilateral(params: GaitParams) -> bool:
    r, l = params.knee["right"], params.knee["left"]
    both_lr = (
        r.loading_response_peak is not None and l.loading_response_peak is not None
        and r.loading_response_peak < 12 and l.loading_response_peak < 12
    )
    both_sw = (
        r.peak_swing_flexion is not None and l.peak_swing_flexion is not None
        and r.peak_swing_flexion < 50 and l.peak_swing_flexion < 50
    )
    return both_lr and both_sw


# ─── Key frame extraction ───────────────────────────────────────────────────────

def _extract_phase_frame(video_path: str, frames_data: list[dict], phase_labels: dict[int, str],
                          target_phase: str) -> list[str]:
    """Extract one base64 JPEG from the first available frame of a given gait phase."""
    candidates = sorted(si for si, ph in phase_labels.items() if ph == target_phase)
    if not candidates:
        return []
    # Pick the middle candidate for robustness
    si = candidates[len(candidates) // 2]
    actual_idx = frames_data[si]["frame_idx"]
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, actual_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return []
    h, w = frame.shape[:2]
    if w > 640:
        frame = cv2.resize(frame, (640, int(h * 640 / w)))
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return [base64.b64encode(buf).decode('utf-8')]


def _extract_mid_frame(video_path: str) -> list[str]:
    """One overview frame from the middle of the video (skips first/last 10%)."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start, end = int(total * 0.1), int(total * 0.9)
    cap.set(cv2.CAP_PROP_POS_FRAMES, (start + end) // 2)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return []
    h, w = frame.shape[:2]
    if w > 640:
        frame = cv2.resize(frame, (640, int(h * 640 / w)))
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return [base64.b64encode(buf).decode('utf-8')]


# ─── Main dual-video entry point ───────────────────────────────────────────────

def analyse_gait_dual(frontal_path: str, sagittal_path: str) -> tuple[GaitMetrics, list[str], dict]:
    """
    Full dual-video biomechanical pipeline.
    Returns (GaitMetrics, key_frames_b64, extra_dict).
    extra_dict contains: severity, kl_score, kl_grade, clinical_flags,
                         bilateral_pattern_detected.
    """
    sag, fro = None, None
    sag_results, fro_results = {}, {}
    fallback_mode = False

    try:
        sag = _run_mediapipe(sagittal_path)
        sag_results = _extract_sagittal(sag)
    except Exception:
        fallback_mode = True

    try:
        fro = _run_mediapipe(frontal_path)
        fro_results = _extract_frontal(fro)
    except Exception:
        fallback_mode = True

    # Need at least one working view
    primary_sag = sag if sag else fro
    primary_sag_res = sag_results if sag_results else {}
    primary_fro = fro if fro else sag
    primary_fro_res = fro_results if fro_results else {}

    temporal = _extract_temporal(primary_sag, primary_fro, primary_sag_res, primary_fro_res)

    knee_r = sag_results.get("knee_right", KneePhaseAngles(None, None, None, None, None, 0.0))
    knee_l = sag_results.get("knee_left",  KneePhaseAngles(None, None, None, None, None, 0.0))

    frontal_conf  = fro["confidence"] if fro else 0.0
    sagittal_conf = sag["confidence"] if sag else 0.0

    r_avg, l_avg = knee_r.avg_full_cycle, knee_l.avg_full_cycle
    symmetry = round(max(0.0, 100.0 - abs(r_avg - l_avg) * 2.5), 1) if r_avg and l_avg else 0.0

    params = GaitParams(
        knee={"right": knee_r, "left": knee_l},
        right_varus_valgus_thrust=fro_results.get("right_vvt", 0.0),
        left_varus_valgus_thrust=fro_results.get("left_vvt", 0.0),
        pelvic_obliquity_deg=fro_results.get("pelvic_obliquity_deg", 0.0),
        trendelenburg_flag=fro_results.get("trendelenburg_flag", False),
        trunk_lateral_lean_deg=fro_results.get("trunk_lateral_lean_deg", 0.0),
        trunk_lean_direction=fro_results.get("trunk_lean_direction", "neutral"),
        step_width_proxy=fro_results.get("step_width_proxy", 0.0),
        fppa_right=fro_results.get("fppa_right", 0.0),
        fppa_left=fro_results.get("fppa_left", 0.0),
        hip_extension_terminal_stance=sag_results.get("hip_extension_terminal_stance"),
        ankle_dorsiflexion=sag_results.get("ankle_dorsiflexion"),
        trunk_anterior_lean_deg=sag_results.get("trunk_anterior_lean_deg", 0.0),
        cadence=temporal["cadence"],
        stride_time_asymmetry=temporal["stride_time_asymmetry"],
        double_support_ratio=temporal["double_support_ratio"],
        gait_speed_proxy=sag_results.get("gait_speed_proxy", 0.0),
        symmetry_score=symmetry,
        frontal_confidence=frontal_conf,
        sagittal_confidence=sagittal_conf,
        frontal_frames_analyzed=fro["frames_analyzed"] if fro else 0,
        frontal_frames_skipped=fro["frames_skipped"] if fro else 0,
        sagittal_frames_analyzed=sag["frames_analyzed"] if sag else 0,
        sagittal_frames_skipped=sag["frames_skipped"] if sag else 0,
        heel_strike_events_right=temporal["heel_strike_events_right"],
        heel_strike_events_left=temporal["heel_strike_events_left"],
        gait_cycles_detected=temporal["gait_cycles_detected"],
        fallback_mode=fallback_mode,
    )

    kl_score, kl_grade, clinical_flags = _compute_kl_proxy(params)

    bilateral = _check_bilateral(params)
    if bilateral:
        clinical_flags.append("bilateral_oa_pattern")

    _kl_to_sev = {"kl_0": "mild", "kl_1": "mild", "kl_2": "moderate", "kl_3": "severe", "kl_4": "severe"}
    severity = _kl_to_sev[kl_grade]
    if bilateral and severity == "mild":
        severity = "moderate"

    metrics = GaitMetrics(
        # Backward-compat fields
        knee_angle_right=r_avg or None,
        knee_angle_left=l_avg or None,
        knee_angle_diff=round(abs(r_avg - l_avg), 1) if r_avg and l_avg else None,
        symmetry_score=symmetry,
        trunk_lean_angle=fro_results.get("trunk_lateral_lean_deg"),
        trunk_lean_direction=fro_results.get("trunk_lean_direction", "neutral"),
        cadence=params.cadence,
        frames_analysed=params.frontal_frames_analyzed + params.sagittal_frames_analyzed,
        confidence=round(min(frontal_conf, sagittal_conf), 3) if (fro and sag) else max(frontal_conf, sagittal_conf),
        # Per-view counts
        frontal_frames_analyzed=params.frontal_frames_analyzed,
        frontal_frames_skipped=params.frontal_frames_skipped,
        sagittal_frames_analyzed=params.sagittal_frames_analyzed,
        sagittal_frames_skipped=params.sagittal_frames_skipped,
        # Gait cycle
        heel_strike_events_right=params.heel_strike_events_right,
        heel_strike_events_left=params.heel_strike_events_left,
        gait_cycles_detected=params.gait_cycles_detected,
        # KL scoring
        kl_proxy_score=kl_score,
        kl_proxy_grade=kl_grade,
        clinical_flags=clinical_flags,
        bilateral_pattern_detected=bilateral,
        # Sagittal phase angles
        right_loading_response_peak=knee_r.loading_response_peak,
        left_loading_response_peak=knee_l.loading_response_peak,
        right_mid_stance_angle=knee_r.mid_stance_angle,
        left_mid_stance_angle=knee_l.mid_stance_angle,
        right_peak_swing_flexion=knee_r.peak_swing_flexion,
        left_peak_swing_flexion=knee_l.peak_swing_flexion,
        right_rom_delta=knee_r.rom_delta,
        left_rom_delta=knee_l.rom_delta,
        right_extension_lag=knee_r.extension_lag,
        left_extension_lag=knee_l.extension_lag,
        hip_extension_terminal_stance=params.hip_extension_terminal_stance,
        ankle_dorsiflexion=params.ankle_dorsiflexion,
        trunk_anterior_lean_deg=params.trunk_anterior_lean_deg,
        # Frontal-view
        right_varus_valgus_thrust=params.right_varus_valgus_thrust,
        left_varus_valgus_thrust=params.left_varus_valgus_thrust,
        pelvic_obliquity_deg=params.pelvic_obliquity_deg,
        trendelenburg_flag=params.trendelenburg_flag,
        step_width_proxy=params.step_width_proxy,
        fppa_right=params.fppa_right,
        fppa_left=params.fppa_left,
        # Temporal
        stride_time_asymmetry=params.stride_time_asymmetry,
        double_support_ratio=params.double_support_ratio,
        gait_speed_proxy=params.gait_speed_proxy,
        fallback_mode=fallback_mode,
    )

    # Key frames: sagittal loading_response, sagittal mid_swing,
    #             frontal loading_response, frontal overview
    key_frames: list[str] = []
    if sag and sag_results.get("hs_right"):
        phase_sag = sag_results.get("phase_right", _label_phases(len(sag["frames"]), sag_results["hs_right"]))
        key_frames += _extract_phase_frame(sagittal_path, sag["frames"], phase_sag, "loading_response")
        key_frames += _extract_phase_frame(sagittal_path, sag["frames"], phase_sag, "mid_swing")
    if fro and fro_results.get("hs_right_frontal"):
        phase_fro = fro_results.get("phase_right_frontal",
                                     _label_phases(len(fro["frames"]), fro_results["hs_right_frontal"]))
        key_frames += _extract_phase_frame(frontal_path, fro["frames"], phase_fro, "loading_response")
    if fro:
        key_frames += _extract_mid_frame(frontal_path)
    elif sag:
        key_frames += _extract_mid_frame(sagittal_path)

    key_frames = key_frames[:4]
    if not key_frames:
        # Last-resort: mid-frame from whichever video worked
        key_frames = _extract_mid_frame(sagittal_path) or _extract_mid_frame(frontal_path)

    extra = {
        "severity": severity,
        "kl_score": kl_score,
        "kl_grade": kl_grade,
        "clinical_flags": clinical_flags,
        "bilateral_pattern_detected": bilateral,
        "primary_view_confidence": metrics.confidence,
    }

    return metrics, key_frames, extra
