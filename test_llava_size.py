#!/usr/bin/env python
import os
import csv
import re
import numpy as np
import ipdb
from pathlib import Path
from collections import defaultdict
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from PIL import Image

# ─── Helpers ────────────────────────────────────────────────────────────────────

def extract_number(text):
    m = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", text)
    if not m:
        return None
    try:
        num = float(m.group())
        return int(num) if num.is_integer() else num
    except ValueError:
        return None

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start=0.0, end=0.9, interval=0.1):
    if pred is None or target is None or target == 0:
        return 0.0
    confs = np.arange(start, end + interval, interval)
    return float(np.mean([abs_dist_norm(pred, target) <= 1 - c for c in confs]))

# ─── LLAVA MODEL SETUP ─────────────────────────────────────────────────────────

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16"
)
processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    quantization_config=quantization_config,
    device_map="auto"
)

def query_llava(prompt, videos=None):
    # support video inputs: videos is a numpy array (N, H, W, 3)
    video_list = videos if isinstance(videos, list) else [videos]
    # prepend one <image> token per video clip
    prompt = "USER: <video>\n" * len(video_list) + prompt + "ASSISTANT:"
    inputs = processor(text=[prompt], videos=video_list, padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # generate
    generate_kwargs = {"max_new_tokens": 100, "do_sample": True, "top_p": 0.9}
    outputs = model.generate(**inputs, **generate_kwargs)
    # decode and return first result
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]
  

# ─── Main test harness ─────────────────────────────────────────────────────────

BASE_DATA_DIR      = Path("/home/zifan/ARKitScenes/data/3dod")
DOWNLOAD_LIST_CSV  = Path("/home/zifan/ARKitScenes/counting_download_list.csv")
COUNTING_CSV       = Path("/home/zifan/SpatialLM/counting_questions.csv")
SIZE_CSV           = Path("/home/zifan/SpatialLM/size_questions.csv")
SANITY_BASE        = Path("/home/zifan/SpatialLM/sanity_check")

def load_questions(*csv_paths):
    qm = defaultdict(list)
    for path in csv_paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scene = row["scene_name"]
                gt = float(row["ground_truth"])
                qm[scene].append({
                    "question": row["question"].strip(),
                    "ground_truth": gt,
                    "type": row.get("question_type", "counting")
                })
    return qm

def main():
    # 1) read download list
    dl = list(csv.DictReader(open(DOWNLOAD_LIST_CSV, newline="")))
    if not dl:
        print("No videos found in download list.")
        return

    first = dl[0]
    vid  = first["video_id"]
    fold = first["fold"]
    print(f"→ Testing video {vid} (fold={fold})")

    # 2) load all questions
    questions_map = load_questions( SIZE_CSV)
    qlist = questions_map.get(vid, [])
    if not qlist:
        print(f"No questions for video {vid}")
        return

    # 3) load & subsample video frames (once)
    frames_dir = BASE_DATA_DIR / fold / vid / f"{vid}_frames" / "lowres_wide"
    frame_paths = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        print(f"No frame images found in {frames_dir}")
        return

    indices = np.linspace(0, len(frame_paths) - 1, 64, dtype=int)
    video_frames = [Image.open(frame_paths[i]).convert("RGB") for i in indices]

    # save for sanity
    sanity_dir = SANITY_BASE / vid
    sanity_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(video_frames):
        frame.save(sanity_dir / f"{vid}_frame_{i}.png")
    video_arr = np.stack([np.array(f) for f in video_frames], axis=0)
    np.save(sanity_dir / f"{vid}_frames.npy", video_arr)

    # 4) run LLAVA for each question
    for qinfo in qlist:
        qtype = qinfo["type"]
        prompt = qinfo["question"]
        # ensure single-number answer
        prompt = f"{prompt} Answer only a single number."
        gt     = qinfo["ground_truth"]
        print(f"\n→ Question ({qtype}): {prompt}  (gt={gt})")
        resp = query_llava(prompt, videos=video_arr)
        pred = extract_number(resp)
        mra  = mean_relative_accuracy(pred, gt)
        correct = "✓" if pred == gt else "✗"
        
        print("────────────────────────────────────────")
        print(f"Response text: {resp}")
        print(f"Predicted: {pred}, Ground truth: {gt}, Exact: {correct}, MRA: {mra:.3f}")
        print("────────────────────────────────────────")

if __name__ == "__main__":
    main()