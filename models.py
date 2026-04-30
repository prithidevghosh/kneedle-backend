from pydantic import BaseModel
from typing import Optional


class GaitMetrics(BaseModel):
    """Raw biomechanical measurements from MediaPipe"""
    knee_angle_right: Optional[float] = None   # degrees, 0-180
    knee_angle_left: Optional[float] = None    # degrees, 0-180
    knee_angle_diff: Optional[float] = None    # asymmetry in degrees
    symmetry_score: Optional[float] = None     # 0-100, 100 = perfect symmetry
    trunk_lean_angle: Optional[float] = None   # degrees from vertical
    trunk_lean_direction: Optional[str] = None # "left", "right", "neutral"
    toe_out_angle_right: Optional[float] = None
    toe_out_angle_left: Optional[float] = None
    cadence: Optional[float] = None            # steps per minute
    frames_analysed: int = 0
    confidence: float = 0.0                    # 0-1, mediapipe detection confidence


class Exercise(BaseModel):
    name: str           # Bengali
    reps: str           # Bengali e.g. "১০×৩"
    name_en: str        # English
    reps_en: str        # English e.g. "10×3"
    description: str    # Bengali instructions
    reason: str = ""    # why Gemma 4 chose this exercise for this patient


class AnalysisResponse(BaseModel):
    observation: str           # Bengali — what was observed
    observation_en: str        # English version
    fix_title: str             # Bengali — title of today's correction
    fix_desc: str              # Bengali — how to do the correction
    fix_title_en: str          # English
    fix_desc_en: str           # English
    exercises: list[Exercise]  # 2 exercises max
    active_joint: str          # "right_knee" | "left_knee" | "hips" | "ankles"
    symmetry_score: float      # from MediaPipe
    session_number: int        # passed in from app
    thinking: Optional[str]    # Gemma 4 reasoning chain — logged, not shown to user
    metrics: GaitMetrics       # raw measurements for writeup/debugging


class AnalyseRequest(BaseModel):
    knee: str = "both"         # "left" | "right" | "both"
    age: str = "60"
    lang: str = "bn"           # "bn" | "hi" | "en"
    session_number: int = 1
