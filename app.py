"""
Kneedle FastAPI application.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

--host 0.0.0.0 makes the server reachable from the phone on the same WiFi.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.analyse import router as analyse_router
from routes.diagnostics import router as diagnostics_router
from routes.voice_chat import router as voice_chat_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

app = FastAPI(
    title="Kneedle API",
    description="Knee OA gait analysis — MediaPipe + Gemma 4",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your laptop IP in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyse_router)
app.include_router(diagnostics_router)
app.include_router(voice_chat_router)
