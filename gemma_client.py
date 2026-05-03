"""
Gemma 4 E4B — multimodal gait reasoning.

Architecture:
- MediaPipe provides hard measurements (what the numbers are)
- Gemma 4 provides clinical reasoning (what the numbers mean)
- Severity tiering and exercise filtering are deterministic in code,
  not delegated to the LLM, so unsafe exercises cannot be prescribed
  for severe cases even if the model hallucinates.
- Thinking mode enabled — reasoning chain logged for writeup

Backends:
  "local"    — Ollama on this machine  (ollama serve must be running)
  "hf_space" — Hugging Face Space      (set HF_SPACE_URL below)
"""

import ollama
import json
import re
import logging
from models import GaitMetrics, AnalysisResponse, Exercise
from exercises import get_exercise_by_id, EXERCISE_LIBRARY

logger = logging.getLogger(__name__)

# ── Backend selector ──────────────────────────────────────────────────────────
# Change this one variable to switch inference backends:
#   "local"    → Ollama (ollama serve must be running)
#   "hf_space" → Hugging Face Space
INFERENCE_BACKEND = "local"

# Only used when INFERENCE_BACKEND = "hf_space"
# Format: "your-hf-username/your-space-name"  e.g. "johndoe/kneedle-gemma"
HF_SPACE_URL = "prithvigg/kneedle-gemma"
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_MODEL = "gemma4:e4b"

# Exercises that require good ambulation, balance, or pain-free deep knee flexion.
# Excluded for severe cases (see assess_severity below).
SEVERE_EXCLUDE_IDS = {"wall_sit", "mini_squat", "step_up", "toe_out_walk"}

# Default safety / educational text per language.
# These are clinically conservative defaults; the LLM may override with more
# patient-specific phrasing, but if it omits a field we fall back to these.
SAFETY_DEFAULTS = {
    "bn": {
        "frequency": "প্রতিদিন একবার, ২ সপ্তাহ ধরে অনুশীলন করুন।",
        "pain_rule": "ব্যথা ৫/১০-এর বেশি হলে থামুন। হালকা অস্বস্তি স্বাভাবিক।",
        "red_flags": "হঠাৎ হাঁটু ফুলে গেলে, পা একদম দিতে না পারলে, বা জ্বর হলে অবিলম্বে ডাক্তার দেখান।",
        "complementary": "ওজন কমানো, হাঁটুতে গরম সেঁক (১৫ মিনিট), এবং প্রয়োজনে নী-ক্যাপ ব্যবহার করুন।",
        "referral_severe": "তীব্র উপসর্গের কারণে অনুগ্রহ করে একজন অর্থোপেডিক বা ফিজিওথেরাপিস্টের সাথে সরাসরি দেখা করুন।",
        "empathy": "হাঁটতে কষ্ট হচ্ছে দেখে আমি বুঝতে পারছি — চলুন, ছোট ছোট ধাপে শুরু করি।",
        "sym_good": "৮০-এর উপরে স্বাভাবিক — আপনার হাঁটার ভারসাম্য ভালো আছে।",
        "sym_fair": "৮০-এর উপরে স্বাভাবিক — আপনার স্কোর কিছুটা কম, উন্নতির সুযোগ আছে।",
        "sym_poor": "৮০-এর উপরে স্বাভাবিক — আপনার স্কোর অনেক কম, এই ব্যায়ামগুলি সাহায্য করবে।",
    },
    "hi": {
        "frequency": "रोज़ एक बार, २ सप्ताह तक अभ्यास करें।",
        "pain_rule": "यदि दर्द ५/१० से अधिक हो तो रुकें। हल्की असुविधा सामान्य है।",
        "red_flags": "अचानक घुटना सूज जाए, पैर पर खड़े न हो पाएं, या बुखार हो तो तुरंत डॉक्टर से मिलें।",
        "complementary": "वज़न कम करें, घुटने पर गर्म सिकाई (१५ मिनट) करें, और आवश्यकता हो तो नी-कैप का प्रयोग करें।",
        "referral_severe": "गंभीर लक्षणों के कारण कृपया हड्डी रोग विशेषज्ञ या फिज़ियोथेरेपिस्ट से सीधी मुलाकात करें।",
        "empathy": "चलने में तकलीफ़ हो रही है यह मैं समझ सकता हूँ — आइए, छोटे क़दमों से शुरुआत करें।",
        "sym_good": "८० से ऊपर सामान्य — आपका चाल संतुलन अच्छा है।",
        "sym_fair": "८० से ऊपर सामान्य — आपका स्कोर थोड़ा कम है, सुधार की गुंजाइश है।",
        "sym_poor": "८० से ऊपर सामान्य — आपका स्कोर काफी कम है, ये व्यायाम मदद करेंगे।",
    },
    "en": {
        "frequency": "Daily, once a day, for 2 weeks.",
        "pain_rule": "Stop if pain exceeds 5/10. Mild discomfort is normal.",
        "red_flags": "See a doctor immediately if your knee swells suddenly, you cannot bear weight, or you develop a fever.",
        "complementary": "Lose excess weight, apply a warm compress to the knee for 15 minutes, and consider a knee sleeve.",
        "referral_severe": "Due to severe symptoms, please see an orthopedist or physiotherapist in person.",
        "empathy": "I can see walking is difficult — let's start with small, gentle steps.",
        "sym_good": "Normal is above 80 — your gait balance is good.",
        "sym_fair": "Normal is above 80 — your score is a little low, with room to improve.",
        "sym_poor": "Normal is above 80 — your score is well below normal, these exercises will help.",
    },
}


def assess_severity(metrics: GaitMetrics) -> str:
    """
    Classify gait severity from MediaPipe metrics.

    Severe triggers (any one is enough):
      - symmetry_score < 65
      - trunk_lean_angle > 8°
      - knee_angle_diff > 15°
      - cadence < 70 steps/min  (very slow / antalgic gait)

    Moderate: symmetry 65-79, trunk lean 4-8°, or knee_angle_diff 8-15°
    Mild:     everything else
    """
    sym = metrics.symmetry_score
    lean = metrics.trunk_lean_angle or 0
    diff = metrics.knee_angle_diff or 0
    cad = metrics.cadence

    if (sym is not None and sym < 65) \
       or lean > 8 \
       or diff > 15 \
       or (cad is not None and cad < 70):
        return "severe"

    if (sym is not None and sym < 80) \
       or lean > 4 \
       or diff > 8:
        return "moderate"

    return "mild"


def compute_symmetry_band(score: float | None) -> str:
    """Map raw symmetry_score (0-100) to a patient-facing band."""
    if score is None:
        return "fair"
    if score >= 80:
        return "good"
    if score >= 65:
        return "fair"
    return "poor"


def filter_library_by_severity(severity: str) -> list[dict]:
    """Return the subset of EXERCISE_LIBRARY safe for this severity tier."""
    if severity == "severe":
        return [e for e in EXERCISE_LIBRARY if e["id"] not in SEVERE_EXCLUDE_IDS]
    return list(EXERCISE_LIBRARY)


def _format_library(library: list[dict]) -> str:
    lines = []
    for ex in library:
        lines.append(
            f"[{ex['id']}]\n"
            f"  English: {ex['name_en']} ({ex['reps_en']})\n"
            f"  When to use: {ex['clinical_indication']}\n"
            f"  Do NOT use if: {ex['contraindication']}\n"
            f"  Evidence: {ex['evidence']}\n"
        )
    return "\n".join(lines)


def _safety_for(lang: str) -> dict:
    return SAFETY_DEFAULTS.get(lang, SAFETY_DEFAULTS["bn"])


def build_system_prompt(lang: str) -> str:
    """
    System prompt instructs Gemma 4 to act as a physiotherapist and respond
    in the correct language with a specific JSON format. The full safety
    field set is required so the patient receives complete guidance.
    """
    lang_instruction = {
        "bn": "You MUST respond entirely in Bengali (বাংলা). All localized text fields must be in Bengali script.",
        "hi": "You MUST respond entirely in Hindi (हिन्दी). All localized text fields must be in Hindi script.",
        "en": "Respond in clear, simple English."
    }.get(lang, "You MUST respond entirely in Bengali (বাংলা).")

    return f"""<|think|>
You are a compassionate physiotherapist specialising in knee osteoarthritis.
You are analysing the walking pattern of an elderly patient who cannot afford
in-person physiotherapy and depends on this app.

{lang_instruction}

You will be given:
1. Several images from the patient's walking video
2. Precise biomechanical measurements extracted by MediaPipe Pose
3. A pre-computed severity tier (mild/moderate/severe) — TRUST IT
4. A SEVERITY-FILTERED exercise library — you may ONLY pick from this list

Your job is to identify the single primary clinical finding from the
measurements first, then explain it to the patient in warm, simple language,
anchored to specific numbers — not generic advice.

CRITICAL RULES:
- Open the observation with empathy or what you observed — never with filler
- Speak directly to the patient in warm, simple language — not medical jargon
- Reference at least one specific measurement in the observation, AND interpret
  it ("your symmetry score is 58 — normal is above 80")
- ONE specific correction only in fix_title/fix_desc — not multiple things
- Pick exactly 3 exercises from the provided library (NOT 2)
- AT LEAST ONE of the 3 exercises MUST have contraindication = "None"
  (a safe fallback the patient can always do)
- For severe cases, prefer non-weight-bearing or seated exercises
- Never invent exercises or modify rep counts
- Never make up measurements — only use the numbers provided
- Be encouraging — this patient is in pain and trying hard

OUTPUT FORMAT — respond with ONLY this JSON, no other text:
{{
  "empathy_line": "ONE warm sentence acknowledging difficulty (localized)",
  "empathy_line_en": "Same in English",
  "observation": "3-4 sentence observation tying numbers to plain meaning (localized)",
  "observation_en": "Same in English",
  "symmetry_meaning": "ONE sentence interpreting the symmetry score for the patient (localized)",
  "symmetry_meaning_en": "Same in English",
  "fix_title": "Localized title of the single correction",
  "fix_desc": "1-2 sentence localized description",
  "fix_title_en": "English title",
  "fix_desc_en": "English description",
  "active_joint": "right_knee OR left_knee OR hips OR ankles",
  "thinking_summary": "2-3 sentence summary of clinical reasoning",
  "selected_exercise_ids": ["id_1", "id_2", "id_3"],
  "exercise_reasons": ["1-sentence localized reason for ex1", "for ex2", "for ex3"],
  "frequency": "How often / for how long (localized)",
  "frequency_en": "Same in English",
  "pain_rule": "When to stop (localized)",
  "pain_rule_en": "Same in English",
  "red_flags": "When to seek medical care (localized)",
  "red_flags_en": "Same in English",
  "complementary_actions": "Non-exercise advice — weight, heat, sleeve (localized)",
  "complementary_actions_en": "Same in English"
}}
"""


def build_user_prompt(
    metrics: GaitMetrics,
    age: str,
    knee: str,
    severity: str,
    library: list[dict],
) -> str:
    """Build the clinical context prompt with measurements + filtered library."""
    knee_desc = {"left": "left knee", "right": "right knee", "both": "both knees"}

    primary_finding = "general gait pattern"
    if metrics.symmetry_score and metrics.symmetry_score < 70:
        worse_side = "right" if (metrics.knee_angle_right or 0) < (metrics.knee_angle_left or 0) else "left"
        primary_finding = f"significant gait asymmetry favouring the {worse_side} side"
    elif metrics.trunk_lean_angle and metrics.trunk_lean_angle > 6:
        primary_finding = f"trunk leaning {metrics.trunk_lean_direction} during walking"
    elif metrics.toe_out_angle_right and abs(metrics.toe_out_angle_right) < 5:
        primary_finding = "insufficient toe-out angle (feet too straight)"

    severity_guidance = {
        "severe": "SEVERE: Patient has marked deformity / asymmetry / very slow gait. "
                  "Prefer seated, supine, or aquatic exercises. Avoid deep knee bends. "
                  "Always include quad_set or seated_marching as a safe foundation.",
        "moderate": "MODERATE: Patient has noticeable but functional gait deficit. "
                    "Mix of supine and standing exercises. Avoid only deep loaded squats.",
        "mild": "MILD: Gait is largely preserved. Full library is available, "
                "focus on whichever exercise targets the specific finding.",
    }[severity]

    return f"""PATIENT PROFILE:
- Age: {age} years old
- Diagnosed condition: Knee osteoarthritis ({knee_desc.get(knee, 'both knees')})
- Context: Cannot afford physiotherapy, using this app for daily guidance

MEDIAPIPE BIOMECHANICAL MEASUREMENTS (clinically precise):
- Right knee flexion angle: {metrics.knee_angle_right or 'not detected'}°
- Left knee flexion angle: {metrics.knee_angle_left or 'not detected'}°
- Knee angle difference (asymmetry): {metrics.knee_angle_diff or 'not detected'}°
- Gait symmetry score: {metrics.symmetry_score or 'not detected'}/100 (100 = perfect symmetry, normal >80)
- Trunk lateral lean: {metrics.trunk_lean_angle or 'not detected'}° toward {metrics.trunk_lean_direction}
- Right toe-out angle: {metrics.toe_out_angle_right or 'not detected'}°
- Left toe-out angle: {metrics.toe_out_angle_left or 'not detected'}°
- Walking cadence: {metrics.cadence or 'not detected'} steps/minute
- Frames successfully analysed: {metrics.frames_analysed}
- Detection confidence: {metrics.confidence * 100:.0f}%

PRIMARY CLINICAL FINDING: {primary_finding}
SEVERITY TIER (computed deterministically): {severity.upper()}
SEVERITY GUIDANCE: {severity_guidance}

The walking video frames are attached. Look carefully at:
1. How the patient's weight shifts between legs
2. The trunk position during the stance phase
3. The foot placement angle
4. Any visible compensatory movements

Reason over both the measurements AND the visual frames together.

SEVERITY-FILTERED EXERCISE LIBRARY:
You MUST select exactly 3 exercises from THIS LIST ONLY. Do not invent
exercises. Do not modify rep counts. At least one selection MUST have
contraindication = "None".

{_format_library(library)}

Provide your assessment in the exact JSON format specified."""


def get_active_joint(metrics: GaitMetrics) -> str:
    """Determine which joint to highlight based on primary gait finding."""
    if metrics.symmetry_score and metrics.symmetry_score < 70:
        if metrics.knee_angle_right and metrics.knee_angle_left:
            return "right_knee" if metrics.knee_angle_right < metrics.knee_angle_left else "left_knee"
        return "right_knee"
    elif metrics.trunk_lean_angle and metrics.trunk_lean_angle > 6:
        return "hips"
    elif metrics.toe_out_angle_right and abs(metrics.toe_out_angle_right) < 5:
        return "ankles"
    return "right_knee"


def _build_exercise_obj(ex_def: dict, reason: str) -> Exercise:
    return Exercise(
        name=ex_def["name"],
        reps=ex_def["reps"],
        name_en=ex_def["name_en"],
        reps_en=ex_def["reps_en"],
        description=ex_def["description"],
        reason=reason,
        video_url=ex_def.get("video_url", ""),
    )


def _ensure_safe_fallback(exercises: list[Exercise], chosen_ids: set[str]) -> list[Exercise]:
    """
    Guarantee at least one exercise with contraindication == "None" is in the
    selection. This protects severe-case patients even if the LLM's pick is
    too aggressive.
    """
    if any(get_exercise_by_id(getattr(e, "_id", "") or "")
           and get_exercise_by_id(e._id)["contraindication"] == "None"
           for e in exercises if hasattr(e, "_id")):
        return exercises

    has_safe = any(
        (ex_def := get_exercise_by_id(ex_id)) and ex_def["contraindication"] == "None"
        for ex_id in chosen_ids
    )
    if has_safe:
        return exercises

    safe_pool = [e for e in EXERCISE_LIBRARY
                 if e["contraindication"] == "None" and e["id"] not in chosen_ids]
    if safe_pool:
        exercises.append(_build_exercise_obj(safe_pool[0], ""))
    return exercises


def _localized(data: dict, key: str, fallback: str) -> str:
    val = data.get(key)
    return val if isinstance(val, str) and val.strip() else fallback


def _call_via_ollama(
    system_prompt: str,
    user_prompt: str,
    frames_b64: list[str],
) -> tuple[str, str]:
    """Run inference locally via Ollama. Returns (raw_content, thinking)."""
    import concurrent.futures
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt, "images": frames_b64},
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            ollama.chat,
            model=OLLAMA_MODEL,
            messages=messages,
            format="json",
            options={
                "temperature": 0.4,
                "top_p": 0.9,
                "top_k": 40,
                "seed": 42,
                "num_predict": 2500,
                "num_ctx": 16384,
                "num_gpu": 99,
            },
        )
        try:
            response = future.result(timeout=180)
        except concurrent.futures.TimeoutError:
            raise RuntimeError("Ollama timed out after 180s")

    if response.get("done_reason", "") == "length":
        logger.warning("Gemma hit num_predict limit — response may be truncated")

    thinking = response["message"].get("thinking", "")
    raw_content = response["message"]["content"]
    logger.info(f"Ollama raw content (first 300 chars): {raw_content[:300]!r}")
    return raw_content, thinking


def _call_via_hf_space(
    system_prompt: str,
    user_prompt: str,
    frames_b64: list[str],
) -> tuple[str, str]:
    """Run inference via Hugging Face Space. Returns (raw_content, thinking)."""
    try:
        from gradio_client import Client
    except ImportError:
        raise RuntimeError("gradio_client not installed. Run: pip install gradio_client")

    # Strip <|think|> — raw transformers can't separate thinking from JSON the way
    # Ollama does, so thinking would consume the token budget and truncate the JSON.
    sys_prompt_no_think = system_prompt.replace("<|think|>", "").lstrip()

    import concurrent.futures
    client = Client(HF_SPACE_URL)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            client.predict,
            system_prompt=sys_prompt_no_think,
            user_prompt=user_prompt,
            images_json=json.dumps(frames_b64),
            api_name="/generate",
        )
        try:
            raw_content = future.result(timeout=180)
        except concurrent.futures.TimeoutError:
            raise RuntimeError("HF Space timed out after 180s")

    logger.info(f"HF Space raw content (first 300 chars): {raw_content[:300]!r}")
    return raw_content, ""


def call_gemma4(
    metrics: GaitMetrics,
    frames_b64: list[str],
    age: str,
    knee: str,
    lang: str,
    session_number: int
) -> AnalysisResponse:
    """
    Call Gemma 4 E4B with frames + metrics.
    Routes to Ollama or HF Space based on INFERENCE_BACKEND.
    Severity is computed deterministically and used to filter the exercise
    library before the LLM ever sees it.
    """
    severity = assess_severity(metrics)
    sym_band = compute_symmetry_band(metrics.symmetry_score)
    library = filter_library_by_severity(severity)
    safety = _safety_for(lang)

    logger.info(
        f"Backend: {INFERENCE_BACKEND} | Severity: {severity} | "
        f"Symmetry band: {sym_band} | Library size: {len(library)}"
    )

    system_prompt = build_system_prompt(lang)
    user_prompt = build_user_prompt(metrics, age, knee, severity, library)

    try:
        if INFERENCE_BACKEND == "hf_space":
            raw_content, thinking = _call_via_hf_space(system_prompt, user_prompt, frames_b64)
        else:
            raw_content, thinking = _call_via_ollama(system_prompt, user_prompt, frames_b64)

        # Prefer the LAST ```json ... ``` block (skips any thinking that leaked through),
        # then fall back to the widest {...} span.
        code_blocks = re.findall(r'```json\s*(\{.*?\})\s*```', raw_content, re.DOTALL)
        if code_blocks:
            json_str = code_blocks[-1]
        else:
            clean = re.sub(r'```json\s*|\s*```', '', raw_content).strip()
            m = re.search(r'\{.*\}', clean, re.DOTALL)
            if not m:
                logger.error(f"No JSON found in Gemma response. Full response: {raw_content!r}")
                raise ValueError("No JSON found in response")
            json_str = m.group()

        data = json.loads(json_str)

        # ----- Exercise selection -----
        # Restrict the LLM's selection to the severity-filtered library so it
        # cannot smuggle a contraindicated exercise back in.
        allowed_ids = {e["id"] for e in library}
        selected_ids = [eid for eid in data.get("selected_exercise_ids", []) if eid in allowed_ids]
        exercise_reasons = data.get("exercise_reasons", [])

        exercises: list[Exercise] = []
        for i, ex_id in enumerate(selected_ids[:4]):
            ex_def = get_exercise_by_id(ex_id)
            if ex_def:
                exercises.append(_build_exercise_obj(
                    ex_def,
                    exercise_reasons[i] if i < len(exercise_reasons) else "",
                ))

        # Ensure at least one safe (contraindication == "None") fallback is present
        chosen_ids = {ex_id for ex_id in selected_ids[:4]}
        exercises = _ensure_safe_fallback(exercises, chosen_ids)

        # If the LLM gave us too few, top up from severity-appropriate library
        if len(exercises) < 3:
            for cand in library:
                if cand["id"] in chosen_ids:
                    continue
                exercises.append(_build_exercise_obj(cand, ""))
                chosen_ids.add(cand["id"])
                if len(exercises) >= 3:
                    break

        # ----- Symmetry meaning -----
        sym_default = {
            "good": safety["sym_good"],
            "fair": safety["sym_fair"],
            "poor": safety["sym_poor"],
        }[sym_band]
        sym_default_en = {
            "good": SAFETY_DEFAULTS["en"]["sym_good"],
            "fair": SAFETY_DEFAULTS["en"]["sym_fair"],
            "poor": SAFETY_DEFAULTS["en"]["sym_poor"],
        }[sym_band]

        # ----- Referral logic — severe always gets one -----
        referral_recommended = severity == "severe"
        referral_text = safety["referral_severe"] if referral_recommended else ""
        referral_text_en = SAFETY_DEFAULTS["en"]["referral_severe"] if referral_recommended else ""

        return AnalysisResponse(
            observation=data.get("observation", ""),
            observation_en=data.get("observation_en", ""),
            fix_title=data.get("fix_title", ""),
            fix_desc=data.get("fix_desc", ""),
            fix_title_en=data.get("fix_title_en", ""),
            fix_desc_en=data.get("fix_desc_en", ""),
            exercises=exercises,
            active_joint=data.get("active_joint", get_active_joint(metrics)),
            symmetry_score=metrics.symmetry_score or 0.0,
            session_number=session_number,
            thinking=thinking,
            metrics=metrics,
            severity=severity,
            symmetry_band=sym_band,
            symmetry_meaning=_localized(data, "symmetry_meaning", sym_default),
            symmetry_meaning_en=_localized(data, "symmetry_meaning_en", sym_default_en),
            empathy_line=_localized(data, "empathy_line", safety["empathy"]),
            empathy_line_en=_localized(data, "empathy_line_en", SAFETY_DEFAULTS["en"]["empathy"]),
            frequency=_localized(data, "frequency", safety["frequency"]),
            frequency_en=_localized(data, "frequency_en", SAFETY_DEFAULTS["en"]["frequency"]),
            pain_rule=_localized(data, "pain_rule", safety["pain_rule"]),
            pain_rule_en=_localized(data, "pain_rule_en", SAFETY_DEFAULTS["en"]["pain_rule"]),
            red_flags=_localized(data, "red_flags", safety["red_flags"]),
            red_flags_en=_localized(data, "red_flags_en", SAFETY_DEFAULTS["en"]["red_flags"]),
            referral_recommended=referral_recommended,
            referral_text=referral_text,
            referral_text_en=referral_text_en,
            complementary_actions=_localized(data, "complementary_actions", safety["complementary"]),
            complementary_actions_en=_localized(data, "complementary_actions_en", SAFETY_DEFAULTS["en"]["complementary"]),
        )

    except Exception as e:
        return _fallback_response(metrics, lang, session_number, str(e))


def _fallback_response(
    metrics: GaitMetrics,
    lang: str,
    session_number: int,
    error: str,
) -> AnalysisResponse:
    """Hardcoded fallback if Gemma 4 is unavailable. Severity-aware."""
    severity = assess_severity(metrics)
    sym_band = compute_symmetry_band(metrics.symmetry_score)
    safety = _safety_for(lang)
    en = SAFETY_DEFAULTS["en"]

    if lang == "bn":
        obs = "আপনার হাঁটার ধরন বিশ্লেষণ করা হয়েছে। ডান হাঁটুতে কিছুটা বেশি চাপ পড়ছে।"
        obs_en = "Your gait has been analysed. Your right knee is taking slightly more load."
        fix_t = "পায়ের আঙুল সামান্য বাইরে রাখুন"
        fix_d = "হাঁটার সময় পায়ের আঙুল ১০° বাইরের দিকে রাখুন। এতে হাঁটুর চাপ কমবে।"
        fix_t_en = "Point toes slightly outward"
        fix_d_en = "While walking, point your toes 10° outward. This reduces knee load."
    elif lang == "hi":
        obs = "आपकी चाल का विश्लेषण किया गया है। दाएं घुटने पर थोड़ा अधिक दबाव है।"
        obs_en = "Your gait has been analysed. Your right knee is taking slightly more load."
        fix_t = "पैर की उंगलियां थोड़ी बाहर रखें"
        fix_d = "चलते समय पैर की उंगलियां १०° बाहर की तरफ रखें। इससे घुटने का दबाव कम होगा।"
        fix_t_en = "Point toes slightly outward"
        fix_d_en = "While walking, point your toes 10° outward."
    else:
        obs = "Your gait has been analysed. Your right knee is taking slightly more load than the left."
        obs_en = obs
        fix_t = "Point toes slightly outward"
        fix_d = "While walking, point your toes 10° outward. This reduces the load on your knee."
        fix_t_en = fix_t
        fix_d_en = fix_d

    # Severity-tiered safe fallback exercises
    if severity == "severe":
        fallback_ids = ["quad_set", "seated_marching", "heel_slide"]
    elif severity == "moderate":
        fallback_ids = ["seated_marching", "straight_leg_raise", "side_lying_hip_abduction"]
    else:
        fallback_ids = ["seated_marching", "straight_leg_raise", "calf_raise"]

    exercises = [
        _build_exercise_obj(ex, "")
        for fid in fallback_ids
        if (ex := get_exercise_by_id(fid))
    ]

    sym_meaning = {
        "good": safety["sym_good"], "fair": safety["sym_fair"], "poor": safety["sym_poor"],
    }[sym_band]
    sym_meaning_en = {
        "good": en["sym_good"], "fair": en["sym_fair"], "poor": en["sym_poor"],
    }[sym_band]

    referral_recommended = severity == "severe"

    return AnalysisResponse(
        observation=obs,
        observation_en=obs_en,
        fix_title=fix_t,
        fix_desc=fix_d,
        fix_title_en=fix_t_en,
        fix_desc_en=fix_d_en,
        exercises=exercises,
        active_joint=get_active_joint(metrics),
        symmetry_score=metrics.symmetry_score or 50.0,
        session_number=session_number,
        thinking=f"Fallback used due to: {error}",
        metrics=metrics,
        severity=severity,
        symmetry_band=sym_band,
        symmetry_meaning=sym_meaning,
        symmetry_meaning_en=sym_meaning_en,
        empathy_line=safety["empathy"],
        empathy_line_en=en["empathy"],
        frequency=safety["frequency"],
        frequency_en=en["frequency"],
        pain_rule=safety["pain_rule"],
        pain_rule_en=en["pain_rule"],
        red_flags=safety["red_flags"],
        red_flags_en=en["red_flags"],
        referral_recommended=referral_recommended,
        referral_text=safety["referral_severe"] if referral_recommended else "",
        referral_text_en=en["referral_severe"] if referral_recommended else "",
        complementary_actions=safety["complementary"],
        complementary_actions_en=en["complementary"],
    )
