from gait_analyzer import analyse_gait
from models import GaitMetrics


def run_gait_analysis(video_path: str) -> tuple[GaitMetrics, list[str]]:
    """Run MediaPipe pose pipeline on the video. Returns metrics + key frames."""
    return analyse_gait(video_path)
