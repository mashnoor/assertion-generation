# Quick test: verify AutoProcessor + AutoModelForImageTextToText loads and generates
import sys
try:
    import pysqlite3; sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model_name = "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"
print(f"Loading processor...", flush=True)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

print(f"Loading model...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
)
model.eval()
print(f"Model loaded on {model.device}", flush=True)

messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Say hello in 5 words."},
]
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)
inputs = processor(text=text, return_tensors="pt").to(model.device)
print(f"Input shape: {inputs['input_ids'].shape}", flush=True)

import time
t0 = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.7, do_sample=True)
elapsed = time.time() - t0

new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
result = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
print(f"Response ({elapsed:.1f}s): {result}", flush=True)
