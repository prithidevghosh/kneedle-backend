"""
Conversational LLM service backed by Gemma 4 E2B (lightweight 2B variant).

Why a separate service from `gemma_client.py`:
- The clinical reasoning client (E4B) is multimodal, JSON-structured, and
  optimised for one-shot deep analysis. Voice conversation has different
  needs: short turns, no images, plain text out, conversation history,
  sub-2s latency.
- Using E2B keeps voice replies fast even when E4B is mid-analysis.

Conversation history is held in memory per-session, scoped to the WebSocket
connection (handler is responsible for passing/storing it). This keeps the
service stateless and easy to test.
"""

import logging
import concurrent.futures
import ollama

logger = logging.getLogger(__name__)

# Lightweight Gemma 4 variant — ~2x faster than E4B, sufficient for chat.
OLLAMA_MODEL = "gemma4:e2b"

# Cap conversation memory so prompts stay small and latency stays bounded.
# 8 messages = 4 user + 4 assistant turns.
MAX_HISTORY_MESSAGES = 8

# Hard cap on tokens generated per turn — keeps replies voice-friendly
# (2-3 sentences) and bounds TTS rendering time.
MAX_REPLY_TOKENS = 180


# ── Language handling ────────────────────────────────────────────────────────

LANG_INSTRUCTION = {
    "bn": "You MUST respond in Bengali (বাংলা). Use Bengali script only.",
    "hi": "You MUST respond in Hindi (हिन्दी). Use Devanagari script only.",
    "en": "Respond in clear, simple English.",
}


def _system_prompt(lang: str, gait_context: dict | None) -> str:
    """
    Build the system prompt. Includes the patient's last gait analysis if
    available, so the assistant can reference real numbers instead of giving
    generic advice.
    """
    lang_rule = LANG_INSTRUCTION.get(lang, LANG_INSTRUCTION["en"])

    context_block = ""
    if gait_context:
        context_block = f"""

PATIENT'S LAST GAIT ANALYSIS (reference these when relevant):
- Severity: {gait_context.get('severity', 'unknown')}
- Symmetry score: {gait_context.get('symmetry_score', 'unknown')}/100
- Active joint: {gait_context.get('active_joint', 'unknown')}
- KL grade: {gait_context.get('kl_proxy_grade', 'unknown')}
- Prescribed exercises: {', '.join(gait_context.get('exercise_names', [])) or 'none yet'}
"""

    return f"""You are a warm, encouraging physiotherapy assistant for elderly patients
with knee osteoarthritis. You speak with patients about their gait, prescribed
exercises, pain management, and recovery progress.

{lang_rule}

CRITICAL RULES:
- Keep replies SHORT — 1 to 3 sentences maximum. This is a voice conversation.
- Speak naturally, like a kind nurse — no medical jargon, no markdown, no lists.
- If the patient asks about something outside knee health / exercise / gait,
  gently redirect: "I can best help with your knee and exercises."
- If the patient describes severe pain, sudden swelling, fever, or inability
  to bear weight, tell them to see a doctor immediately.
- Never invent measurements or prescribe new exercises beyond what's already
  in their plan. You may explain or motivate, not diagnose.
- Be encouraging — recovery is slow and patients lose hope easily.
{context_block}"""


# ── Public API ───────────────────────────────────────────────────────────────

def trim_history(history: list[dict]) -> list[dict]:
    """Keep only the last MAX_HISTORY_MESSAGES messages."""
    if len(history) <= MAX_HISTORY_MESSAGES:
        return history
    return history[-MAX_HISTORY_MESSAGES:]


def chat(
    user_text: str,
    lang: str,
    history: list[dict],
    gait_context: dict | None = None,
    timeout_seconds: int = 30,
) -> str:
    """
    Run one chat turn against Gemma E2B.

    Args:
        user_text:    the user's transcribed utterance
        lang:         ISO code for the target reply language ("bn"/"hi"/"en")
        history:      prior turns as [{role, content}, ...]. Will NOT be mutated.
        gait_context: optional dict from the patient's last AnalysisResponse.
                      Keys used: severity, symmetry_score, active_joint,
                      kl_proxy_grade, exercise_names.
        timeout_seconds: hard cap on Ollama call.

    Returns:
        Assistant reply text (already in `lang`).

    Raises:
        RuntimeError on timeout or backend failure. Caller decides fallback.
    """
    messages = [
        {"role": "system", "content": _system_prompt(lang, gait_context)},
        *trim_history(history),
        {"role": "user", "content": user_text},
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            ollama.chat,
            model=OLLAMA_MODEL,
            messages=messages,
            options={
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": MAX_REPLY_TOKENS,
                "num_ctx": 4096,
                "num_gpu": 99,
            },
        )
        try:
            response = future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise RuntimeError(f"Gemma E2B timed out after {timeout_seconds}s")

    reply = response["message"]["content"].strip()
    logger.info(f"Voice chat reply ({lang}, {len(reply)} chars): {reply!r}")
    return reply


def fallback_reply(lang: str) -> str:
    """Static reply used when the LLM backend is unreachable."""
    return {
        "bn": "দুঃখিত, আমি এখন উত্তর দিতে পারছি না। অনুগ্রহ করে আবার চেষ্টা করুন।",
        "hi": "क्षमा करें, मैं अभी जवाब नहीं दे पा रहा हूँ। कृपया फिर से प्रयास करें।",
        "en": "Sorry, I can't respond right now. Please try again in a moment.",
    }.get(lang, "Sorry, I can't respond right now. Please try again.")
