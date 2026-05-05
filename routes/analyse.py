from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional
import logging
from models import AnalysisResponse
from handlers.analyse_handler import handle_analyse


logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/analyse", response_model=AnalysisResponse)
async def analyse_walk(
    video_frontal: Optional[UploadFile] = File(None),
    video_sagittal: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),   # deprecated — returns 400 with message
    knee: str = Form("both"),
    age: str = Form("60"),
    lang: str = Form("bn"),
    session_number: int = Form(1),
):
    """
    Core endpoint. Receives two walking videos + patient profile,
    runs dual-video MediaPipe + Gemma, returns the full gait report.

    Form fields:
    - video_frontal:  MP4 — patient walking toward/away from camera
    - video_sagittal: MP4 — patient walking across frame, full side profile
    - knee:           "left" | "right" | "both"
    - age:            patient age as string
    - lang:           "bn" (Bengali) | "hi" (Hindi) | "en" (English)
    - session_number: session index shown in the app

    Deprecated:
    - video: returns HTTP 400 with deprecation message
    """
    analysis_result = await handle_analyse(
        video_frontal=video_frontal,
        video_sagittal=video_sagittal,
        video_deprecated=video,
        knee=knee,
        age=age,
        lang=lang,
        session_number=session_number,
    )
    logger.info("Gait analysis completed. complete response: %s", analysis_result)
    return analysis_result