"""
Kneedle — Gemma 4 E4B inference Space.

Accepts system_prompt + user_prompt + images (base64 JSON list) and returns
the model's raw JSON string. The backend (gemma_client.py) does all parsing.

For persistent GPU hardware (T4, L4, etc.) — no @spaces.GPU needed.
Set INFERENCE_BACKEND = "hf_space" and HF_SPACE_URL in gemma_client.py.
"""

import json
import base64
import io
import os
import sys
import time
import traceback
import torch
import gradio as gr
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# Force unbuffered stdout/stderr so prints appear immediately in HF Space logs.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

MODEL_ID = os.getenv("MODEL_ID", "google/gemma-4-E4B-it")

print(f"[boot] Loading processor for {MODEL_ID}", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, padding_side="left")
print(f"[boot] Loading model {MODEL_ID}", flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
model.eval()
print(f"[boot] Model ready on device: {model.device}", flush=True)


def generate(system_prompt: str, user_prompt: str, images_json: str) -> str:
    t0 = time.time()
    print(f"[generate] called — sys_len={len(system_prompt)} usr_len={len(user_prompt)} images_json_len={len(images_json)}", flush=True)
    try:
        raw_images: list[str] = json.loads(images_json)
        print(f"[generate] decoded {len(raw_images)} base64 images", flush=True)

        pil_images: list[Image.Image] = []
        for b64 in raw_images:
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            img_bytes = base64.b64decode(b64)
            pil_images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        print(f"[generate] loaded {len(pil_images)} PIL images, sizes={[img.size for img in pil_images]}", flush=True)

        # Gemma 4 supports a native system role — keep system and user separate.
        # System prompt may begin with <|think|> to activate thinking mode.
        # Per Gemma 4 docs, image content should come BEFORE text in the user turn.
        image_content = [{"type": "image", "image": img} for img in pil_images]
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": user_prompt}],
            },
        ]

        print(f"[generate] running apply_chat_template...", flush=True)
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        input_len = inputs["input_ids"].shape[-1]
        print(f"[generate] inputs ready — input_len={input_len} keys={list(inputs.keys())}", flush=True)

        print(f"[generate] starting model.generate (greedy, max_new_tokens=1500)...", flush=True)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1500,
                do_sample=False,
                use_cache=True,
            )
        print(f"[generate] generation done — total tokens={outputs.shape[-1]} elapsed={time.time()-t0:.1f}s", flush=True)

        generated_ids = outputs[0][input_len:]
        result = processor.decode(generated_ids, skip_special_tokens=True)
        print(f"[generate] returning {len(result)} chars", flush=True)
        return result
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[generate] EXCEPTION: {e}\n{tb}", flush=True)
        raise


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="System Prompt", lines=10),
        gr.Textbox(label="User Prompt", lines=10),
        gr.Textbox(label="Images JSON (base64 list)"),
    ],
    outputs=gr.Textbox(label="Model Response"),
    title="Kneedle — Gemma 4 E4B",
    description="Gait analysis inference endpoint. Called programmatically by the Kneedle backend.",
    api_name="generate",
).queue(max_size=4, default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(show_error=True, max_threads=2)
