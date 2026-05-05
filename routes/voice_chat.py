"""
WebSocket route for the bidirectional voice assistant.

Endpoint:  ws://<host>:<port>/voice/chat

See handlers/voice_handler.py for the full message protocol.
"""

import logging
from fastapi import APIRouter, WebSocket

from handlers.voice_handler import handle_voice_chat

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/voice/chat")
async def voice_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        await handle_voice_chat(websocket)
    except Exception as e:
        logger.error(f"voice chat handler crashed: {e}", exc_info=True)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
