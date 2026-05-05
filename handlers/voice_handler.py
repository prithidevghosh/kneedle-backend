"""
WebSocket orchestration for the voice assistant.

Protocol — every message is a JSON envelope on the WebSocket text channel.

Client → server:
  {"type": "context", "gait": {...}}
      Optional. Send once at the start of the session to give the assistant
      access to the patient's last gait analysis. Keys forwarded to the LLM:
      severity, symmetry_score, active_joint, kl_proxy_grade, exercise_names.

  {"type": "audio", "data": "<base64 audio>", "lang": "auto"|"bn"|"hi"|"en"}
      A complete user utterance. `lang="auto"` triggers Whisper auto-detect
      (recommended). Audio container can be webm/wav/mp3/m4a — anything
      ffmpeg can decode.

  {"type": "text", "text": "...", "lang": "bn"|"hi"|"en"}
      Bypass STT (useful for testing or text-only mode).

  {"type": "reset"}
      Clear conversation history without closing the socket.

Server → client (in order, per turn):
  {"type": "transcript", "text": "...", "lang": "bn"}
      What the assistant heard the user say.

  {"type": "reply_text", "text": "...", "lang": "bn"}
      The assistant's text reply (sent before audio so the UI can show
      captions immediately).

  {"type": "audio_chunk", "data": "<base64 mp3 chunk>", "seq": 0}
      Streamed TTS audio. Multiple chunks per turn, sequence numbered.

  {"type": "audio_end"}
      Marks the end of audio for this turn. Client may now record again.

  {"type": "error", "message": "...", "code": "stt_failed"|"llm_failed"|"tts_failed"|"bad_request"}
      Any failure. The session stays open; the client can send another turn.
"""

import asyncio
import base64
import logging
from dataclasses import dataclass, field

from fastapi import WebSocket, WebSocketDisconnect

from services import stt_service, tts_service, voice_chat_service

logger = logging.getLogger(__name__)


@dataclass
class VoiceSession:
    """Per-connection state. One instance per WebSocket."""
    history: list[dict] = field(default_factory=list)
    gait_context: dict | None = None
    default_lang: str = "en"  # used if STT can't detect

    def remember(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        self.history = voice_chat_service.trim_history(self.history)

    def reset(self) -> None:
        self.history = []


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _send_error(ws: WebSocket, message: str, code: str) -> None:
    logger.warning(f"voice error [{code}]: {message}")
    try:
        await ws.send_json({"type": "error", "message": message, "code": code})
    except Exception:
        pass  # socket already gone


async def _stream_tts(ws: WebSocket, text: str, lang: str) -> None:
    """Stream TTS audio chunks to the client, then send audio_end."""
    seq = 0
    try:
        async for chunk in tts_service.synthesize_stream(text, lang):
            await ws.send_json({
                "type": "audio_chunk",
                "data": base64.b64encode(chunk).decode("ascii"),
                "seq": seq,
            })
            seq += 1
    except Exception as e:
        await _send_error(ws, f"TTS failed: {e}", "tts_failed")
        return
    finally:
        await ws.send_json({"type": "audio_end"})


# ── Per-message handlers ─────────────────────────────────────────────────────

async def _handle_context(session: VoiceSession, msg: dict) -> None:
    gait = msg.get("gait") or {}
    if not isinstance(gait, dict):
        return
    session.gait_context = {
        "severity": gait.get("severity"),
        "symmetry_score": gait.get("symmetry_score"),
        "active_joint": gait.get("active_joint"),
        "kl_proxy_grade": gait.get("kl_proxy_grade"),
        "exercise_names": gait.get("exercise_names", []),
    }
    logger.info(f"voice session context set: {session.gait_context}")


async def _handle_audio(ws: WebSocket, session: VoiceSession, msg: dict) -> None:
    b64 = msg.get("data")
    if not b64:
        await _send_error(ws, "missing 'data' on audio message", "bad_request")
        return

    try:
        audio_bytes = base64.b64decode(b64)
    except Exception:
        await _send_error(ws, "invalid base64 audio", "bad_request")
        return

    requested = msg.get("lang", "auto")
    lang_hint = requested if requested in {"bn", "hi", "en"} else None

    # STT
    try:
        transcript, detected_lang = await asyncio.to_thread(
            stt_service.transcribe, audio_bytes, lang_hint
        )
    except Exception as e:
        await _send_error(ws, f"STT failed: {e}", "stt_failed")
        return

    if not transcript:
        # Silent / unintelligible audio — let the client know and stop.
        await ws.send_json({"type": "transcript", "text": "", "lang": detected_lang})
        await ws.send_json({"type": "audio_end"})
        return

    await ws.send_json({"type": "transcript", "text": transcript, "lang": detected_lang})

    await _run_turn(ws, session, transcript, detected_lang)


async def _handle_text(ws: WebSocket, session: VoiceSession, msg: dict) -> None:
    text = (msg.get("text") or "").strip()
    if not text:
        await _send_error(ws, "missing 'text' on text message", "bad_request")
        return
    lang = msg.get("lang") if msg.get("lang") in {"bn", "hi", "en"} else session.default_lang
    await _run_turn(ws, session, text, lang)


async def _run_turn(
    ws: WebSocket, session: VoiceSession, user_text: str, lang: str
) -> None:
    """Shared LLM + TTS pipeline for a single conversational turn."""
    # LLM
    try:
        reply = await asyncio.to_thread(
            voice_chat_service.chat,
            user_text,
            lang,
            list(session.history),  # copy — service must not mutate
            session.gait_context,
        )
    except Exception as e:
        logger.error(f"LLM failed: {e}", exc_info=True)
        reply = voice_chat_service.fallback_reply(lang)
        await _send_error(ws, f"LLM failed, using fallback: {e}", "llm_failed")

    # Persist turn into history (only successful, non-fallback turns to avoid
    # poisoning context with apology messages). Fallback case still streams
    # audio so the user hears something.
    session.remember("user", user_text)
    session.remember("assistant", reply)

    await ws.send_json({"type": "reply_text", "text": reply, "lang": lang})
    await _stream_tts(ws, reply, lang)


# ── Entry point ──────────────────────────────────────────────────────────────

async def handle_voice_chat(ws: WebSocket) -> None:
    """
    Accept a WebSocket connection and run the voice chat loop until the
    client disconnects. Caller (route) is responsible for `await ws.accept()`.
    """
    session = VoiceSession()
    logger.info("voice session started")

    try:
        while True:
            try:
                msg = await ws.receive_json()
            except WebSocketDisconnect:
                break
            except Exception as e:
                await _send_error(ws, f"invalid JSON: {e}", "bad_request")
                continue

            mtype = msg.get("type")

            if mtype == "context":
                await _handle_context(session, msg)
            elif mtype == "audio":
                await _handle_audio(ws, session, msg)
            elif mtype == "text":
                await _handle_text(ws, session, msg)
            elif mtype == "reset":
                session.reset()
                await ws.send_json({"type": "reset_ack"})
            else:
                await _send_error(ws, f"unknown message type: {mtype!r}", "bad_request")

    finally:
        logger.info(f"voice session ended (history len={len(session.history)})")
