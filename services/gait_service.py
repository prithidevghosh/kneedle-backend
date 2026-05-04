from gait_analyzer import analyse_gait_dual
from models import GaitMetrics


def run_gait_analysis(frontal_path: str, sagittal_path: str) -> tuple[GaitMetrics, list[str], dict]:
    """Run dual-video MediaPipe pipeline. Returns (metrics, key_frames, extra)."""
    return analyse_gait_dual(frontal_path, sagittal_path)
