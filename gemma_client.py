"""
Gemma 4 12b via Ollama — multimodal gait reasoning.

Architecture:
- MediaPipe provides hard measurements (what the numbers are)
- Gemma 4 provides clinical reasoning (what the numbers mean)
- Thinking mode enabled — reasoning chain logged for writeup

Gemma 4 model used: gemma4:12b
Ollama must be running: ollama serve
"""

import ollama
import json
import re
from models import GaitMetrics, AnalysisResponse, Exercise
from exercises import get_library_for_prompt, get_exercise_by_id, EXERCISE_LIBRARY

OLLAMA_MODEL = "gemma4:12b"


def build_system_prompt(lang: str) -> str:
    """
    System prompt instructs Gemma 4 to act as a physiotherapist
    and respond in the correct language with a specific JSON format.
    Thinking mode enabled via <|think|> token.
    """
    lang_instruction = {
        "bn": "You MUST respond entirely in Bengali (বাংলা). All text fields must be in Bengali script.",
        "hi": "You MUST respond entirely in Hindi (हिन्दी). All text fields must be in Hindi script.",
        "en": "Respond in clear, simple English."
    }.get(lang, "You MUST respond entirely in Bengali (বাংলা).")

    return f"""<|think|>
You are a compassionate physiotherapist specialising in knee osteoarthritis.
You are analysing the walking pattern of an elderly patient.

{lang_instruction}

You will be given:
1. Several images from the patient's walking video
2. Precise biomechanical measurements extracted by MediaPipe Pose

Your job is to reason carefully over both the visual evidence and the measurements, then provide a clear, actionable report.

CRITICAL RULES:
- Speak directly to the patient in warm, simple language — not medical jargon
- Maximum 2 sentences for observation, 2 sentences for fix
- ONE specific correction only — not multiple things to change
- TWO exercises maximum
- Never say "consult a doctor" — give a real, specific recommendation
- Never make up measurements — only use the numbers provided
- Be encouraging and warm — this patient is in pain and trying hard

OUTPUT FORMAT — respond with ONLY this JSON, no other text:
{{
  "observation": "Bengali observation text",
  "observation_en": "English observation text",
  "fix_title": "Bengali title of correction",
  "fix_desc": "Bengali description",
  "fix_title_en": "English title",
  "fix_desc_en": "English description",
  "active_joint": "right_knee OR left_knee OR hips OR ankles",
  "thinking_summary": "2-3 sentence summary of your clinical reasoning",
  "selected_exercise_ids": ["id_1", "id_2"],
  "exercise_reasons": ["1-sentence reason in {lang} why this exercise suits this patient", "same for exercise 2"]
}}
"""


def build_user_prompt(metrics: GaitMetrics, age: str, knee: str, lang: str) -> str:
    """
    Build the clinical context prompt with all MediaPipe measurements.
    This is what Gemma 4 reasons over.
    """
    knee_desc = {"left": "left knee", "right": "right knee", "both": "both knees"}

    primary_finding = "general gait pattern"
    if metrics.symmetry_score and metrics.symmetry_score < 70:
        worse_side = "right" if (metrics.knee_angle_right or 0) < (metrics.knee_angle_left or 0) else "left"
        primary_finding = f"significant gait asymmetry favouring the {worse_side} side"
    elif metrics.trunk_lean_angle and metrics.trunk_lean_angle > 6:
        primary_finding = f"trunk leaning {metrics.trunk_lean_direction} during walking"
    elif metrics.toe_out_angle_right and abs(metrics.toe_out_angle_right) < 5:
        primary_finding = "insufficient toe-out angle (feet too straight)"

    prompt = f"""PATIENT PROFILE:
- Age: {age} years old
- Diagnosed condition: Knee osteoarthritis ({knee_desc.get(knee, 'both knees')})
- Context: Cannot afford physiotherapy, using this app for daily guidance

MEDIAPIPE BIOMECHANICAL MEASUREMENTS (clinically precise):
- Right knee flexion angle: {metrics.knee_angle_right or 'not detected'}°
- Left knee flexion angle: {metrics.knee_angle_left or 'not detected'}°
- Knee angle difference (asymmetry): {metrics.knee_angle_diff or 'not detected'}°
- Gait symmetry score: {metrics.symmetry_score or 'not detected'}/100 (100 = perfect symmetry)
- Trunk lateral lean: {metrics.trunk_lean_angle or 'not detected'}° toward {metrics.trunk_lean_direction}
- Right toe-out angle: {metrics.toe_out_angle_right or 'not detected'}°
- Left toe-out angle: {metrics.toe_out_angle_left or 'not detected'}°
- Walking cadence: {metrics.cadence or 'not detected'} steps/minute
- Frames successfully analysed: {metrics.frames_analysed}
- Detection confidence: {metrics.confidence * 100:.0f}%

PRIMARY CLINICAL FINDING: {primary_finding}

The walking video frames are attached. Look carefully at:
1. How the patient's weight shifts between legs
2. The trunk position during the stance phase
3. The foot placement angle
4. Any visible compensatory movements

Reason over both the measurements AND the visual frames together.

EXERCISE LIBRARY (ACR/Arthritis Foundation 2019 Guidelines):
You MUST select exactly 2 exercises from this library only.
Do not invent exercises. Do not modify rep counts.
Select based on the patient's specific gait findings above.

{get_library_for_prompt()}

In your JSON response include:
"selected_exercise_ids": ["id_1", "id_2"]
"exercise_reasons": ["1-sentence reason in {lang} why this exercise suits this patient", "same for exercise 2"]

Provide your assessment in the exact JSON format specified."""

    return prompt


def get_active_joint(metrics: GaitMetrics) -> str:
    """
    Determine which joint to highlight in the app's skeleton overlay
    based on primary gait finding.
    """
    if metrics.symmetry_score and metrics.symmetry_score < 70:
        if metrics.knee_angle_right and metrics.knee_angle_left:
            return "right_knee" if metrics.knee_angle_right < metrics.knee_angle_left else "left_knee"
        return "right_knee"
    elif metrics.trunk_lean_angle and metrics.trunk_lean_angle > 6:
        return "hips"
    elif metrics.toe_out_angle_right and abs(metrics.toe_out_angle_right) < 5:
        return "ankles"
    return "right_knee"


def call_gemma4(
    metrics: GaitMetrics,
    frames_b64: list[str],
    age: str,
    knee: str,
    lang: str,
    session_number: int
) -> AnalysisResponse:
    """
    Call Gemma 4 12b via Ollama with frames + metrics.
    Uses multimodal input (images + text in same prompt).
    Returns structured AnalysisResponse.
    """
    system_prompt = build_system_prompt(lang)
    user_prompt = build_user_prompt(metrics, age, knee, lang)

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt,
            "images": frames_b64  # Ollama handles base64 image array
        }
    ]

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 600,
                "num_ctx": 8192,
            }
        )

        raw_content = response["message"]["content"]

        # Extract thinking chain (between <think> tags) for logging
        thinking = ""
        think_match = re.search(r'<think>(.*?)</think>', raw_content, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            raw_content = raw_content.replace(think_match.group(0), "").strip()

        # Strip markdown code blocks if present
        clean = re.sub(r'```json\s*|\s*```', '', raw_content).strip()

        json_match = re.search(r'\{.*\}', clean, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in response")

        data = json.loads(json_match.group())

        selected_ids = data.get("selected_exercise_ids", [])
        exercise_reasons = data.get("exercise_reasons", [])

        exercises = []
        for i, ex_id in enumerate(selected_ids[:2]):
            ex = get_exercise_by_id(ex_id)
            if ex:
                exercises.append(Exercise(
                    name=ex["name"],
                    reps=ex["reps"],
                    name_en=ex["name_en"],
                    reps_en=ex["reps_en"],
                    description=ex["description"],
                    reason=exercise_reasons[i] if i < len(exercise_reasons) else ""
                ))

        # Fallback if LLM returned invalid or missing IDs
        if len(exercises) < 2:
            safe = [e for e in EXERCISE_LIBRARY if e["contraindication"] == "None"]
            for ex in safe[:2 - len(exercises)]:
                exercises.append(Exercise(
                    name=ex["name"],
                    reps=ex["reps"],
                    name_en=ex["name_en"],
                    reps_en=ex["reps_en"],
                    description=ex["description"],
                    reason=""
                ))

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
            metrics=metrics
        )

    except Exception as e:
        return _fallback_response(metrics, lang, session_number, str(e))


def _fallback_response(
    metrics: GaitMetrics,
    lang: str,
    session_number: int,
    error: str
) -> AnalysisResponse:
    """
    Hardcoded fallback if Gemma 4 is unavailable.
    Ensures the app works even if Ollama is down.
    """
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

    fallback_ids = ["seated_marching", "straight_leg_raise"]
    exercises = [
        Exercise(
            name=ex["name"],
            reps=ex["reps"],
            name_en=ex["name_en"],
            reps_en=ex["reps_en"],
            description=ex["description"],
            reason=""
        )
        for fid in fallback_ids
        if (ex := get_exercise_by_id(fid))
    ]

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
        metrics=metrics
    )
