"""
Knee OA exercise library.
Source: 2019 ACR/Arthritis Foundation Guideline for Management of
Osteoarthritis of the Hand, Hip, and Knee.
Kolasinski et al., Arthritis Care & Research, 2020.
DOI: 10.1002/acr.24131

Each exercise includes:
- Clinical indication (which gait finding it addresses)
- Contraindication (when NOT to prescribe it)
- Evidence level from the guideline
"""

EXERCISE_LIBRARY = [
    {
        "id": "straight_leg_raise",
        "name": "পা সোজা তোলা",
        "name_en": "Straight Leg Raise",
        "reps": "১৫×৩",
        "reps_en": "15×3",
        "description": "পিঠে শুয়ে এক পা সোজা রেখে ৪৫° উপরে তুলুন, ৫ সেকেন্ড ধরে রাখুন তারপর নামান।",
        "description_en": "Lie on back, raise one straight leg to 45°, hold 5 seconds, lower slowly.",
        "clinical_indication": "Quadriceps weakness — primary cause of knee instability in OA. Addresses reduced knee flexion angle and asymmetry.",
        "contraindication": "Acute lower back pain",
        "evidence": "Strong recommendation — ACR/AF 2019"
    },
    {
        "id": "supported_knee_bend",
        "name": "চেয়ারে হাঁটু মোড়ানো",
        "name_en": "Seated Knee Flexion",
        "reps": "১০×৩",
        "reps_en": "10×3",
        "description": "চেয়ারে বসে ধীরে ধীরে হাঁটু যতটুকু সম্ভব ভাঁজ করুন এবং আবার সোজা করুন।",
        "description_en": "Sit in chair, slowly bend knee as far as comfortable, then straighten.",
        "clinical_indication": "Reduced knee range of motion. Directly addresses limited flexion angle measured by MediaPipe.",
        "contraindication": "None for seated version",
        "evidence": "Strong recommendation — ACR/AF 2019"
    },
    {
        "id": "toe_out_walk",
        "name": "আঙুল বাইরে রেখে হাঁটা",
        "name_en": "Toe-Out Gait Training",
        "reps": "১০ মিনিট",
        "reps_en": "10 minutes",
        "description": "ঘরের মধ্যে পায়ের আঙুল ১০° বাইরে রেখে ধীরে হাঁটুন। এটি হাঁটুর ভেতরের চাপ কমায়।",
        "description_en": "Walk indoors with toes pointed 10° outward. Clinically proven to reduce medial knee load.",
        "clinical_indication": "Low toe-out angle. Toe-out modification reduces knee adduction moment — equivalent to NSAIDs for pain reduction (Shull et al., 2013, J Orthop Res).",
        "contraindication": "Hip pain",
        "evidence": "Conditional recommendation — supported by biomechanical evidence"
    },
    {
        "id": "wall_sit",
        "name": "দেওয়ালে হেলান বসা",
        "name_en": "Supported Wall Sit",
        "reps": "৩০ সেকেন্ড×৩",
        "reps_en": "30 seconds×3",
        "description": "দেওয়ালে পিঠ দিয়ে ধীরে সামান্য বসার ভঙ্গিতে যান (৩০° কোণ), ৩০ সেকেন্ড থাকুন।",
        "description_en": "Back against wall, slide down to 30° bend, hold 30 seconds. Strengthens without joint impact.",
        "clinical_indication": "Trunk lean and gait asymmetry — builds hip and quad strength to correct compensatory lean.",
        "contraindication": "Severe pain at any knee bend",
        "evidence": "Strong recommendation — ACR/AF 2019 (strengthening exercise)"
    },
    {
        "id": "calf_raise",
        "name": "গোড়ালি তোলা",
        "name_en": "Standing Calf Raise",
        "reps": "১২×৩",
        "reps_en": "12×3",
        "description": "চেয়ার ধরে দাঁড়িয়ে ধীরে গোড়ালি তুলুন এবং নামান। ভারসাম্য ও হাঁটার গতি উন্নত করে।",
        "description_en": "Hold chair, rise on toes slowly, lower. Improves balance and walking cadence.",
        "clinical_indication": "Low cadence and poor balance. Addresses ankle stability which affects knee loading.",
        "contraindication": "Severe ankle arthritis",
        "evidence": "Conditional recommendation — ACR/AF 2019 (balance training)"
    },
    {
        "id": "side_lying_hip_abduction",
        "name": "পাশে শুয়ে পা তোলা",
        "name_en": "Side-Lying Hip Abduction",
        "reps": "১০×২",
        "reps_en": "10×2",
        "description": "পাশে শুয়ে উপরের পা সোজা রেখে ৩০° তুলুন, ৩ সেকেন্ড ধরুন তারপর নামান।",
        "description_en": "Lie on side, raise top leg straight to 30°, hold 3 seconds, lower slowly.",
        "clinical_indication": "Trunk lean and gait asymmetry — hip abductor weakness is the primary cause of lateral trunk lean in knee OA.",
        "contraindication": "Hip replacement",
        "evidence": "Strong recommendation — ACR/AF 2019 (hip strengthening)"
    },
    {
        "id": "seated_marching",
        "name": "চেয়ারে বসে পা তোলা",
        "name_en": "Seated Marching",
        "reps": "২০×২",
        "reps_en": "20×2",
        "description": "চেয়ারে বসে একবার ডান পা, একবার বাম পা উঁচু করুন — হাঁটার মতো। ধীরে করুন।",
        "description_en": "Sit in chair, alternately lift each knee as if marching. Improves hip flexor strength and gait rhythm.",
        "clinical_indication": "Low cadence and general deconditioning. Safe for all severity levels.",
        "contraindication": "None",
        "evidence": "Strong recommendation — ACR/AF 2019 (aerobic/functional exercise)"
    },
    {
        "id": "tai_chi_step",
        "name": "ধীর পদক্ষেপ অনুশীলন",
        "name_en": "Slow Step Practice",
        "reps": "৫ মিনিট",
        "reps_en": "5 minutes",
        "description": "খুব ধীরে, সচেতনভাবে হাঁটুন। প্রতিটি পদক্ষেপে মাটিতে পা রাখার অনুভূতিতে মনোযোগ দিন।",
        "description_en": "Walk very slowly and mindfully. Focus on each foot placement. Based on Tai Chi principles.",
        "clinical_indication": "General knee OA, poor balance, high pain. Tai Chi is strongly recommended by ACR/AF 2019 specifically for knee OA.",
        "contraindication": "None",
        "evidence": "Strong recommendation — ACR/AF 2019 (Tai Chi specifically for knee OA)"
    }
]

# Pass to Gemma 4 as a formatted string
def get_library_for_prompt() -> str:
    lines = []
    for ex in EXERCISE_LIBRARY:
        lines.append(
            f"[{ex['id']}]\n"
            f"  Bengali: {ex['name']} ({ex['reps']})\n"
            f"  English: {ex['name_en']} ({ex['reps_en']})\n"
            f"  When to use: {ex['clinical_indication']}\n"
            f"  Do NOT use if: {ex['contraindication']}\n"
            f"  Evidence: {ex['evidence']}\n"
        )
    return "\n".join(lines)

def get_exercise_by_id(exercise_id: str) -> dict | None:
    for ex in EXERCISE_LIBRARY:
        if ex["id"] == exercise_id:
            return ex
    return None