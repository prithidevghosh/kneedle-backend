from fastapi import APIRouter, UploadFile, File, Form

from models import AnalysisResponse
from handlers.analyse_handler import handle_analyse

router = APIRouter()


@router.post("/analyse", response_model=AnalysisResponse)
async def analyse_walk(
    video: UploadFile = File(...),
    knee: str = Form("both"),
    age: str = Form("60"),
    lang: str = Form("bn"),
    session_number: int = Form(1)
):
    """
    Core endpoint. Receives the walking video + patient profile,
    runs MediaPipe + Gemma 4, returns the full gait report.

    Form fields:
    - video: MP4 video of the patient walking
    - knee: "left" | "right" | "both"
    - age: patient age as string
    - lang: "bn" (Bengali) | "hi" (Hindi) | "en" (English)
    - session_number: session index shown in the app
    """
    return await handle_analyse(video, knee, age, lang, session_number)
