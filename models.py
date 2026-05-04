from pydantic import BaseModel, Field
from typing import Optional


class GaitMetrics(BaseModel):
    """Raw biomechanical measurements from MediaPipe — dual-video pipeline."""
    # ── Backward-compatible fields (names unchanged) ──────────────────────────
    knee_angle_right: Optional[float] = None    # avg full-cycle, degrees
    knee_angle_left: Optional[float] = None
    knee_angle_diff: Optional[float] = None
    symmetry_score: Optional[float] = None      # 0-100
    trunk_lean_angle: Optional[float] = None    # lateral lean, degrees
    trunk_lean_direction: Optional[str] = None  # "left" | "right" | "neutral"
    toe_out_angle_right: Optional[float] = None
    toe_out_angle_left: Optional[float] = None
    cadence: Optional[float] = None             # steps/min
    frames_analysed: int = 0
    confidence: float = 0.0                     # min(frontal, sagittal)

    # ── Per-view frame counts ─────────────────────────────────────────────────
    frontal_frames_analyzed: int = 0
    frontal_frames_skipped: int = 0
    sagittal_frames_analyzed: int = 0
    sagittal_frames_skipped: int = 0

    # ── Gait cycle metadata ───────────────────────────────────────────────────
    heel_strike_events_right: int = 0
    heel_strike_events_left: int = 0
    gait_cycles_detected: int = 0

    # ── KL-proxy scoring ──────────────────────────────────────────────────────
    kl_proxy_score: float = 0.0
    kl_proxy_grade: str = "kl_0"               # "kl_0" … "kl_4"
    clinical_flags: list[str] = Field(default_factory=list)
    bilateral_pattern_detected: bool = False

    # ── Sagittal-view knee phase angles ───────────────────────────────────────
    right_loading_response_peak: Optional[float] = None   # norm 15-20°
    left_loading_response_peak: Optional[float] = None
    right_mid_stance_angle: Optional[float] = None        # norm 5-10°
    left_mid_stance_angle: Optional[float] = None
    right_peak_swing_flexion: Optional[float] = None      # norm 60-70°
    left_peak_swing_flexion: Optional[float] = None
    right_rom_delta: Optional[float] = None               # norm >50°
    left_rom_delta: Optional[float] = None
    right_extension_lag: Optional[float] = None           # norm ≤5°
    left_extension_lag: Optional[float] = None

    # ── Sagittal-view other ───────────────────────────────────────────────────
    hip_extension_terminal_stance: Optional[float] = None # norm 10-15°
    ankle_dorsiflexion: Optional[float] = None            # norm ~10°
    trunk_anterior_lean_deg: float = 0.0                  # norm <5°

    # ── Frontal-view parameters ───────────────────────────────────────────────
    right_varus_valgus_thrust: float = 0.0    # signed: pos=varus, neg=valgus
    left_varus_valgus_thrust: float = 0.0
    pelvic_obliquity_deg: float = 0.0         # norm <5°
    trendelenburg_flag: bool = False
    step_width_proxy: float = 0.0             # norm 0.10-0.13 normalized units
    fppa_right: float = 0.0                   # frontal plane projection angle deviation
    fppa_left: float = 0.0

    # ── Temporal parameters ───────────────────────────────────────────────────
    stride_time_asymmetry: float = 0.0        # %, norm <5%
    double_support_ratio: float = 20.0        # %, norm ~20%
    gait_speed_proxy: float = 0.0             # normalized, norm >0.8

    # ── Pipeline metadata ─────────────────────────────────────────────────────
    fallback_mode: bool = False


class Exercise(BaseModel):
    name: str
    reps: str
    name_en: str
    reps_en: str
    description: str
    reason: str = ""
    video_url: str = ""


class AnalysisResponse(BaseModel):
    observation: str
    observation_en: str
    fix_title: str
    fix_desc: str
    fix_title_en: str
    fix_desc_en: str
    exercises: list[Exercise]
    active_joint: str
    symmetry_score: float
    session_number: int
    thinking: Optional[str]
    metrics: GaitMetrics

    # Severity-aware patient-facing fields
    severity: str = "moderate"
    symmetry_band: str = "fair"
    symmetry_meaning: str = ""
    symmetry_meaning_en: str = ""
    empathy_line: str = ""
    empathy_line_en: str = ""
    frequency: str = ""
    frequency_en: str = ""
    pain_rule: str = ""
    pain_rule_en: str = ""
    red_flags: str = ""
    red_flags_en: str = ""
    referral_recommended: bool = False
    referral_text: str = ""
    referral_text_en: str = ""
    complementary_actions: str = ""
    complementary_actions_en: str = ""

    # ── New top-level fields (Step 10) ────────────────────────────────────────
    kl_proxy_grade: str = "kl_0"
    clinical_flags: list[str] = Field(default_factory=list)
    bilateral_pattern_detected: bool = False
    primary_view_confidence: float = 0.0


class AnalyseRequest(BaseModel):
    knee: str = "both"
    age: str = "60"
    lang: str = "bn"
    session_number: int = 1
