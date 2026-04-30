import os
import tempfile
import logging
from fastapi import UploadFile, HTTPException

from models import AnalysisResponse
from services.gait_service import run_gait_analysis
from services.gemma_service import run_gemma_analysis

logger = logging.getLogger("kneedle")


async def handle_analyse(
    video: UploadFile,
    knee: str,
    age: str,
    lang: str,
    session_number: int
) -> AnalysisResponse:
    if not video.content_type or "video" not in video.content_type:
        raise HTTPException(
            status_code=400,
            detail="Please upload a video file (MP4)"
        )

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await video.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"Video received: {len(content) / 1024:.1f}KB, lang={lang}, knee={knee}")

        # Layer 1 — MediaPipe: hard biomechanical measurements
        logger.info("Running MediaPipe gait analysis...")
        metrics, key_frames = run_gait_analysis(tmp_path)

        logger.info(
            f"MediaPipe done — "
            f"R knee: {metrics.knee_angle_right}°, "
            f"L knee: {metrics.knee_angle_left}°, "
            f"Symmetry: {metrics.symmetry_score}, "
            f"Frames: {metrics.frames_analysed}, "
            f"Key frames extracted: {len(key_frames)}"
        )

        if metrics.confidence < 0.4:
            logger.warning(
                f"Low MediaPipe confidence: {metrics.confidence}. "
                "Patient may have been too close/far, or lighting was poor."
            )

        # Layer 2 — Gemma 4: clinical reasoning over measurements + frames
        logger.info("Calling Gemma 4 12b for clinical reasoning...")
        result = run_gemma_analysis(metrics, key_frames, age, knee, lang, session_number)

        if result.thinking:
            logger.info(f"Gemma 4 thinking chain:\n{result.thinking}")

        logger.info(
            f"Analysis complete — "
            f"active_joint: {result.active_joint}, "
            f"symmetry: {result.symmetry_score}"
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            logger.info("Temp video file deleted")
