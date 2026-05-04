import os
import tempfile
import logging
from fastapi import UploadFile, HTTPException

from models import AnalysisResponse
from services.gait_service import run_gait_analysis
from services.gemma_service import run_gemma_analysis

logger = logging.getLogger("kneedle")


async def _save_video(upload: UploadFile) -> tuple[str, int]:
    """Write uploaded video to a temp file. Returns (path, size_bytes)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        content = await upload.read()
        tmp.write(content)
        return tmp.name, len(content)


async def handle_analyse(
    video_frontal: UploadFile | None,
    video_sagittal: UploadFile | None,
    video_deprecated: UploadFile | None,
    knee: str,
    age: str,
    lang: str,
    session_number: int,
) -> AnalysisResponse:
    # Deprecated single-video field check
    if video_deprecated is not None:
        raise HTTPException(
            status_code=400,
            detail="video field deprecated. Send video_frontal and video_sagittal instead.",
        )

    if video_frontal is None or video_sagittal is None:
        raise HTTPException(
            status_code=400,
            detail="Both video_frontal and video_sagittal are required.",
        )

    if not video_frontal.content_type or "video" not in video_frontal.content_type:
        raise HTTPException(status_code=400, detail="video_frontal must be a video file (MP4).")
    if not video_sagittal.content_type or "video" not in video_sagittal.content_type:
        raise HTTPException(status_code=400, detail="video_sagittal must be a video file (MP4).")

    frontal_path = sagittal_path = None

    try:
        frontal_path, frontal_kb = await _save_video(video_frontal)
        sagittal_path, sagittal_kb = await _save_video(video_sagittal)

        logger.info(
            f"Videos received — frontal: {frontal_kb/1024:.1f}KB, "
            f"sagittal: {sagittal_kb/1024:.1f}KB, lang={lang}, knee={knee}"
        )

        # Layer 1 — dual-video MediaPipe pipeline
        logger.info("Running dual-video MediaPipe gait analysis...")
        metrics, key_frames, extra = run_gait_analysis(frontal_path, sagittal_path)

        logger.info(
            f"MediaPipe done — "
            f"sagittal frames: {metrics.sagittal_frames_analyzed}, "
            f"frontal frames: {metrics.frontal_frames_analyzed}, "
            f"confidence: {metrics.confidence:.2f}, "
            f"gait_cycles: {metrics.gait_cycles_detected}, "
            f"KL grade: {metrics.kl_proxy_grade}, "
            f"key frames: {len(key_frames)}, "
            f"fallback_mode: {metrics.fallback_mode}"
        )

        # Quality gates — only when both views ran (not fallback mode)
        if not metrics.fallback_mode:
            if metrics.sagittal_frames_analyzed == 0:
                raise HTTPException(status_code=400, detail=(
                    "The sagittal video yielded no usable frames. "
                    "Record with full body visible from the side for at least 5 seconds."
                ))
            if metrics.frontal_frames_analyzed == 0:
                raise HTTPException(status_code=400, detail=(
                    "The frontal video yielded no usable frames. "
                    "Record walking toward/away from camera with full body visible."
                ))
            if metrics.confidence < 0.4:
                raise HTTPException(status_code=400, detail=(
                    "Video quality too low for reliable analysis (confidence "
                    f"{metrics.confidence:.0%}). Re-record in good lighting with "
                    "full body visible from head to toe."
                ))
            if metrics.gait_cycles_detected < 1:
                raise HTTPException(status_code=400, detail=(
                    "No complete gait cycles detected. Walk continuously for at "
                    "least 5 seconds in each video."
                ))

        # Layer 2 — Gemma clinical reasoning
        logger.info("Calling Gemma for clinical reasoning...")
        result = run_gemma_analysis(metrics, key_frames, age, knee, lang, session_number)

        # Patch in new top-level fields from gait pipeline
        result.kl_proxy_grade = extra["kl_grade"]
        result.clinical_flags = extra["clinical_flags"]
        result.bilateral_pattern_detected = extra["bilateral_pattern_detected"]
        result.primary_view_confidence = extra["primary_view_confidence"]

        # KL-proxy severity takes precedence over Gemma's assess_severity heuristic
        result.severity = extra["severity"]
        # Bilateral pattern forces minimum "moderate"
        if extra["bilateral_pattern_detected"] and result.severity == "mild":
            result.severity = "moderate"

        if result.thinking:
            logger.info(f"Gemma thinking chain:\n{result.thinking}")

        logger.info(
            f"Analysis complete — active_joint: {result.active_joint}, "
            f"severity: {result.severity}, flags: {result.clinical_flags}"
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    finally:
        for path in (frontal_path, sagittal_path):
            if path and os.path.exists(path):
                os.unlink(path)
        logger.info("Temp video files deleted")
