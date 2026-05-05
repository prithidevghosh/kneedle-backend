"""
Text-to-Speech service for the voice assistant.

Uses Microsoft Edge's online TTS via the `edge-tts` package — free, no API key,
high-quality neural voices for Bengali, Hindi, and English. Streams audio
chunks as MP3 so the frontend can begin playback before generation completes.
"""

import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)

# Voice picked per language. All are female / warm clinical tone — appropriate
# for a physiotherapy assistant. Swap to *Neural male voices if preferred.
VOICE_BY_LANG = {
    "bn": "bn-IN-TanishaaNeural",   # Bengali (India), female, warm
    "hi": "hi-IN-SwaraNeural",      # Hindi, female, friendly
    "en": "en-IN-NeerjaNeural",     # English (Indian accent — patient familiarity)
}

DEFAULT_VOICE = VOICE_BY_LANG["en"]


def _voice_for(lang: str) -> str:
    return VOICE_BY_LANG.get(lang, DEFAULT_VOICE)


async def synthesize_stream(text: str, lang: str) -> AsyncIterator[bytes]:
    """
    Stream MP3 audio chunks for `text` in the given language.

    Yields raw MP3 bytes as they arrive from the Edge TTS service. The first
    chunk typically arrives in ~300ms — significantly faster than waiting for
    the entire utterance to render.

    Args:
        text: text to speak
        lang: ISO code ("bn"/"hi"/"en"). Unknown codes fall back to English.
    """
    try:
        import edge_tts
    except ImportError as e:
        raise RuntimeError("edge-tts not installed. Run: pip install edge-tts") from e

    voice = _voice_for(lang)
    logger.info(f"TTS: lang={lang} voice={voice} chars={len(text)}")

    communicate = edge_tts.Communicate(text=text, voice=voice)

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]


async def synthesize_full(text: str, lang: str) -> bytes:
    """Convenience wrapper: collect the full MP3 into a single bytes object."""
    buf = bytearray()
    async for chunk in synthesize_stream(text, lang):
        buf.extend(chunk)
    return bytes(buf)
