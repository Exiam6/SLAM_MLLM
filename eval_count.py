import os
import csv
import json
from pathlib import Path
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from PIL import Image
import torch
import ipdb

import re
import numpy as np
from collections import defaultdict

def extract_number(text):
    m = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", text)
    if not m:
        return None
    try:
        num = float(m.group())
        return int(num) if num.is_integer() else num
    except ValueError:
        return None

def exact_match(pred, target):
    return 1.0 if (pred is not None and pred == target) else 0.0

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    if pred is None or target is None or target == 0:
        return 0.0
    conf_intervals = np.arange(start, end + interval, interval)
    acc = [(abs_dist_norm(pred, target) <= 1 - ci) for ci in conf_intervals]
    return float(np.mean(acc))

COUNTING_CSV = "/home/zifan/SpatialLM/counting_questions.csv"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    quantization_config=quantization_config,
    device_map='auto'
)

def query_llava(prompt, images=None, videos=None):
    img_list = images if images is not None else []
    vid_list = videos if videos is not None else []


    prefix = ""
    if img_list:
        prefix += "USER: " + "<image>\n" * len(img_list)
    if vid_list:
        prefix += "USER: " + "<video>\n" * len(vid_list)

    prompt_full = prefix + prompt + " ASSISTANT:"

    proc_kwargs = {"text": [prompt_full], "return_tensors": "pt", "padding": True}
    if img_list:
        proc_kwargs["images"] = img_list
    if vid_list:
        proc_kwargs["videos"] = vid_list

    inputs = processor(**proc_kwargs)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=64)
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]

def main():
    total_nc = 0
    sum_exact_nc = 0.0
    sum_mra_nc = 0.0

    total_wc = 0
    sum_exact_wc = 0.0
    sum_mra_wc = 0.0

    base_data_dir      = Path("/home/zifan/ARKitScenes/data/3dod")
    download_list_csv  = Path("/home/zifan/ARKitScenes/counting_download_list.csv")
    scene_maps_dir     = Path("/home/zifan/SpatialLM/scene_maps")
    output_dir         = Path("frame_evals")
    output_dir.mkdir(exist_ok=True)

    with open(download_list_csv, newline="") as f:
        reader = csv.DictReader(f)
        entries = list(reader)
    questions_map = defaultdict(list)
    with open(COUNTING_CSV, newline="") as qf:
        qreader = csv.DictReader(qf)
        for qrow in qreader:
            vid_q = qrow["scene_name"]
            try:
                gt = float(qrow["ground_truth"])
                if gt.is_integer():
                    gt = int(gt)
            except:
                continue
            questions_map[vid_q].append({
                "question": qrow["question"],
                "ground_truth": gt
            })

    for entry in entries:
        vid    = entry["video_id"]
        fold   = entry["fold"]
        frames_dir = base_data_dir / fold /f"{vid}"/ f"{vid}_frames" / "lowres_wide"
        if not frames_dir.is_dir():
            print(f"[WARN] frames dir missing: {frames_dir}")
            continue

        frame_paths = sorted(
            [fp for fp in frames_dir.iterdir() if fp.suffix.lower() in {".jpg", ".png"}]
        )
        if not frame_paths:
            print(f"[WARN] no frames for video {vid}, skipping")
            continue
        indices = np.linspace(0, len(frame_paths) - 1, 64, dtype=int)
        video_frames = [Image.open(frame_paths[i]).convert("RGB") for i in indices]

        json_path = scene_maps_dir / f"{vid}.json"
        map_path  = scene_maps_dir / f"{vid}.png"
        has_context = json_path.is_file() and map_path.is_file()
        if has_context:
            context_json = json.dumps(json.load(open(json_path)), indent=2)

        out_csv = output_dir / f"frame_results.csv"
        with open(out_csv, "w", newline="") as outf:
            writer = csv.writer(outf)
            writer.writerow([
                "video_id",
                "question",
                "ground_truth",
                "pred_nc",
                "mra_nc",
                "pred_wc" if has_context else "skipped",
                "mra_wc" if has_context else ""
            ])
            for qinfo in questions_map.get(vid, []):
                question = qinfo["question"] + 'Answer only a single number'
                gt = qinfo["ground_truth"]

                # 1) Without context on full video
                video_arr = [np.array(f) for f in video_frames]
                resp_nc = query_llava(question, images=None, videos=[video_arr])
                pred_nc = extract_number(resp_nc)
                total_nc += 1
                mra_nc = mean_relative_accuracy(pred_nc, gt)
                sum_mra_nc += mra_nc

                # 2) With SLAM-MLLM on full video
                if has_context:
                    map_img = Image.open(map_path).convert("RGB")
                    prompt = (
                        "Below is the 2D scene map and the parsed objects/Walls JSON:\n\n"
                        f"{context_json}\n\n"
                        "Scene map image follows, then the video frames.\n\n"
                        f"Question: {question}."
                    )
                    resp_wc = query_llava(prompt, images=[map_img], videos=[video_arr])
                    total_wc += 1
                    pred_wc = extract_number(resp_wc)
                    mra_wc = mean_relative_accuracy(pred_wc, gt)
                    sum_mra_wc += mra_wc
                else:
                    pred_wc = ""
                    mra_wc = ""

                writer.writerow([
                    vid,
                    question,
                    gt,
                    pred_nc,
                    mra_nc,
                    pred_wc,
                    mra_wc
                ])
                
    print("\n===== Overall Summary =====")
    if total_nc > 0:
        print(f"No-context Exact Match: {sum_exact_nc/total_nc:.3f}")
        print(f"No-context MRA: {sum_mra_nc/total_nc:.3f}")
    if total_wc > 0:
        print(f"With-context Exact Match: {sum_exact_wc/total_wc:.3f}")
        print(f"With-context MRA: {sum_mra_wc/total_wc:.3f}")
    print("===========================\n")

if __name__ == "__main__":
    main()