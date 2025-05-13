import os
import csv
import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from PIL import Image
import torch

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
    bnb_4bit_compute_dtype=torch.float16
)
processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    quantization_config=quantization_config,
    device_map="auto"
)

def query_llava(prompt, images=None, videos=None):
    img_list = images if images is not None else []
    vid_list = videos if videos is not None else []
    prefix = ""
    if img_list:
        prefix += "USER: <image>\n" * len(img_list)
    if vid_list:
        prefix += "USER: <video>\n" * len(vid_list)
    prompt_full = prefix + prompt + " ASSISTANT:"
    proc_args = {"text": [prompt_full], "padding": True, "return_tensors": "pt"}
    if img_list:
        proc_args["images"] = img_list
    if vid_list:
        proc_args["videos"] = vid_list
    inputs = processor(**proc_args)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_p=0.9)
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]

def main():
    base_data_dir     = Path("/home/zifan/ARKitScenes/data/3dod")
    download_list_csv = Path("/home/zifan/ARKitScenes/counting_download_list.csv")
    size_csv          = Path("/home/zifan/SpatialLM/size_questions.csv")
    scene_maps_dir    = Path("/home/zifan/SpatialLM/scene_maps")
    output_dir        = Path("/home/zifan/SpatialLM/frame_evals")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(download_list_csv, newline="") as f:
        entries = list(csv.DictReader(f))

    questions_map = defaultdict(list)
    with open(size_csv, newline="") as qf:
        for row in csv.DictReader(qf):
            try:
                gt = float(row["ground_truth"])
                gt = int(gt) if gt.is_integer() else gt
            except:
                continue
            questions_map[row["scene_name"]].append({
                "question": row["question"].strip(),
                "ground_truth": gt
            })

    total_nc = sum_mra_nc = 0
    total_wc = sum_mra_wc = 0

    out_csv = output_dir / "frame_results.csv"
    with open(out_csv, "w", newline="") as outf:
        writer = csv.writer(outf)
        writer.writerow([
            "video_id",
            "question",
            "ground_truth",
            "pred_no_context",
            "mra_no_context",
            "pred_with_context",
            "mra_with_context"
        ])

        for entry in entries:
            vid  = entry["video_id"]
            fold = entry["fold"]
            frames_dir = base_data_dir / fold / vid / f"{vid}_frames" / "lowres_wide"
            if not frames_dir.is_dir():
                print(f"[WARN] Missing frames dir: {frames_dir}")
                continue

            frame_paths = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg",".png"}])
            if not frame_paths:
                print(f"[WARN] No frames for video {vid}")
                continue
            indices = np.linspace(0, len(frame_paths)-1, 64, dtype=int)
            video_frames = [Image.open(frame_paths[i]).convert("RGB") for i in indices]

            json_path = scene_maps_dir / f"{vid}.json"
            map_path  = scene_maps_dir / f"{vid}.png"
            has_context = json_path.is_file() and map_path.is_file()
            context_json = None
            if has_context:
                context_json = json.dumps(json.load(open(json_path)), indent=2)
                
            for qinfo in questions_map.get(vid, []):
                question = qinfo["question"] + " Answer only a single number."
                gt       = qinfo["ground_truth"]

                resp_nc = query_llava(question, videos=[np.stack([np.array(f) for f in video_frames],0)])
                pred_nc = extract_number(resp_nc)
                mra_nc  = mean_relative_accuracy(pred_nc, gt)
                total_nc += 1
                sum_mra_nc += mra_nc

                if has_context:
                    map_img = Image.open(map_path).convert("RGB")
                    prompt = (
                        "Below is scene objects in JSON:\n\n"
                        f"{context_json}\n\n"
                        "Map image follows, then video frames:\n\n"
                        f"Question: {question}"
                    )
                    resp_wc = query_llava(prompt, images=[map_img], videos=[np.stack([np.array(f) for f in video_frames],0)])
                    pred_wc = extract_number(resp_wc)
                    mra_wc  = mean_relative_accuracy(pred_wc, gt)
                    total_wc += 1
                    sum_mra_wc += mra_wc
                else:
                    pred_wc = None
                    mra_wc  = None

                writer.writerow([
                    vid,
                    question,
                    gt,
                    pred_nc,
                    f"{mra_nc:.3f}",
                    pred_wc if pred_wc is not None else "",
                    f"{mra_wc:.3f}" if mra_wc is not None else ""
                ])

    print("===== Overall Summary =====")
    if total_nc:
        print(f"No-context average MRA: {sum_mra_nc/total_nc:.3f}")
    if total_wc:
        print(f"With-context average MRA: {sum_mra_wc/total_wc:.3f}")
    print("===========================")

if __name__ == "__main__":
    main()