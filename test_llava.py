#!/usr/bin/env python
import os
import csv
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from PIL import Image
import ipdb


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

def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    if pred is None or target is None or target == 0:
        return 0.0
    confs = np.arange(start, end + interval, interval)
    return float(np.mean([abs_dist_norm(pred, target) <= 1 - c for c in confs]))

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

    prompt = "USER: <video>\n" * len(video_list) + prompt + "ASSISTANT:"
    inputs = processor(text=[prompt], videos=video_list, padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generate_kwargs = {"max_new_tokens": 100, "do_sample": True, "top_p": 0.9}
    outputs = model.generate(**inputs, **generate_kwargs)
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]
  

BASE_DATA_DIR     = Path("/home/zifan/ARKitScenes/data/3dod")
DOWNLOAD_LIST_CSV = Path("/home/zifan/ARKitScenes/counting_download_list.csv")
COUNTING_CSV      = Path("/home/zifan/SpatialLM/counting_questions.csv")



def main():
    with open(DOWNLOAD_LIST_CSV, newline="") as df:
        dl = list(csv.DictReader(df))
    if not dl:
        print("No videos found in download list.")
        return

    first = dl[0]
    vid  = first["video_id"]
    fold = first["fold"]
    print(f"→ Testing video {vid} (fold={fold})")

    questions_map = defaultdict(list)
    with open(COUNTING_CSV, newline="") as cf:
        for row in csv.DictReader(cf):
            questions_map[row["scene_name"]].append({
                "question":     row["question"],
                "ground_truth": float(row["ground_truth"])
            })

    qlist = questions_map.get(vid)
    if not qlist:
        print(f"No counting question for video {vid}")
        return

    qinfo = qlist[0]
    question = qinfo["question"]
    question = f"Count the number of the target object in the above scene. {question} Answer only a single number "
    gt       = qinfo["ground_truth"]
    print(f"→ Using question: “{question}”  (gt={gt})")

    BASE_DATA_DIR = Path("/home/zifan/ARKitScenes/data/3dod")
    frames_dir = BASE_DATA_DIR / fold / vid / f"{vid}_frames" / "lowres_wide"
    frame_paths = sorted([str(p) for p in frames_dir.glob("*.jpg")] + [str(p) for p in frames_dir.glob("*.png")])
    if not frame_paths:
        print(f"No frame images found in {frames_dir}")
        return
    
    print("video length:",len(frame_paths)-1)
    indices = np.linspace(0, len(frame_paths)-1,64, dtype=int)
    video_frames = [Image.open(frame_paths[i]).convert("RGB") for i in indices]
    if not video_frames:
        print(f"Warning: No frames loaded from {frames_dir}")
        return

    sanity_dir = Path("/home/zifan/SpatialLM/sanity_check") / vid
    sanity_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(video_frames):
        frame_path = sanity_dir / f"{vid}_frame_{i}.png"
        frame.save(frame_path)
    # stack sampled PIL frames into numpy array (N, H, W, 3) and save

    video_arr = np.stack([np.array(f) for f in video_frames], axis=0)
    np.save(sanity_dir / f"{vid}_frames.npy", video_arr)

    resp = query_llava(question, videos=video_arr)
    pred = extract_number(resp)
    mra  = mean_relative_accuracy(pred, gt)
    match = "✓" if pred == gt else "✗"

    print("────────────────────────────────────────")
    print(f"Response text: {resp}")
    print(f"Predicted: {pred}, Ground truth: {gt}, Exact: {match}, MRA: {mra:.3f}")
    print("────────────────────────────────────────")

if __name__ == "__main__":
    main()