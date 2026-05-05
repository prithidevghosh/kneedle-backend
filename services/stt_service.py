"""
Speech-to-Text service for the voice assistant.

Uses faster-whisper (CTranslate2-optimised Whisper) — typically 4-5x faster
than openai-whisper on CPU and supports automatic language detection across
99 languages, including Bengali and Hindi.

Model is loaded lazily on first call and cached as a module-level singleton.
"""

import io
import logging
import tempfile
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Model size trade-off:
#   "tiny"  — ~1s for 5s audio, weakest on Bengali/Hindi
#   "base"  — ~1.5s for 5s audio, good multilingual baseline (DEFAULT)
#   "small" — ~3s for 5s audio, best quality, still under our latency budget
WHISPER_MODEL_SIZE = "base"

# Whisper language codes — map our app codes to Whisper's expected codes.
# Whisper uses ISO 639-1 codes which already match our "bn", "hi", "en".
SUPPORTED_LANGS = {"bn", "hi", "en"}

_model = None


def _get_model():
    """Lazy-load the Whisper model once and reuse."""
    global _model
    if _model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper not installed. Run: pip install faster-whisper"
            ) from e

        logger.info(f"Loading Whisper model '{WHISPER_MODEL_SIZE}' (first call)...")
        _model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8",
        )
        logger.info("Whisper model loaded.")
    return _model


def transcribe(audio_bytes: bytes, lang_hint: Optional[str] = None) -> tuple[str, str]:
    """
    Transcribe a complete audio utterance.

    Args:
        audio_bytes: raw audio (any container ffmpeg can decode — webm, wav, mp3, m4a)
        lang_hint:   optional ISO code ("bn"/"hi"/"en") to skip auto-detect.
                     None → auto-detect.

    Returns:
        (transcript_text, detected_lang_code)
    """
    model = _get_model()

    # faster-whisper accepts a file path, bytes-like, or numpy array.
    # Writing to a tmp file is the most reliable across container formats.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        language = lang_hint if lang_hint in SUPPORTED_LANGS else None

        segments, info = model.transcribe(
            tmp_path,
            language=language,
            beam_size=1,           # greedy decode — much faster, fine for short utterances
            vad_filter=True,       # drop silence
            vad_parameters={"min_silence_duration_ms": 300},
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()
        detected = info.language if info.language in SUPPORTED_LANGS else "en"

        logger.info(
            f"STT: lang={detected} (prob={info.language_probability:.2f}) "
            f"text={text!r}"
        )
        return text, detected
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
