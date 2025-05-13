import os
import re
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
import time
from pathlib import Path
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
model     = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    quantization_config=quantization_config,
    device_map="auto"
)

def query_llava_multimodal(prompt: str, scene_map: Image.Image=None, video_frames=None) -> str:
    """
    Sends prompt plus optional scene_map (one PIL Image) and/or
    video_frames (list of PIL Images) to LLaVA.
    """
    prefix = ""
    visuals = []
    if scene_map is not None:
        prefix += "USER: <image>\n"
        visuals.append(scene_map)
    if video_frames:
        prefix += "USER: <video>\n"
        visuals.extend(video_frames)

    prompt_full = prefix + prompt + " ASSISTANT:"
    proc_kwargs = {"text": [prompt_full], "return_tensors":"pt", "padding":True}
    if scene_map is not None:
        proc_kwargs["images"] = [scene_map]
    if video_frames:
        proc_kwargs["videos"] = [[np.array(f) for f in video_frames]]

    inputs = processor(**proc_kwargs)
    inputs = {k: v.to(model.device) for k,v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens=64)
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]

def extract_number(text: str):
    m = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", text)
    if not m: 
        return None
    num = float(m.group())
    return int(num) if num.is_integer() else num

def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    if target == 0:
        return 1.0 if pred == 0 else 0.0
    rel_err = abs(pred - target) / target
    thresholds = np.arange(start, end + 1e-8, interval)
    return float((rel_err <= (1.0 - thresholds)).mean())

def process_scene_info(scene_info: str) -> str:
    lines = [l.strip() for l in scene_info.splitlines() if l.strip()]
    non_wall = [l for l in lines if not l.startswith("wall_")]
    if not non_wall:
        return None
    out = []
    for line in lines:
        if "=" in line:
            key, val = line.split("=", 1)
            val = val.strip().rstrip(")")
            val = val.replace("(", ": ", 1)
            out.append(f"{key}: {val})")
        else:
            out.append(line)
    return "\n".join(out)

def main_room_size_estimation():
    SCENE_TXT_DIR   = Path("/home/zifan/SpatialLM/ARKitScenes_Outputs")
    SCENE_MAPS_DIR  = Path("/home/zifan/SpatialLM/scene_maps")
    BASE_DATA_DIR   = Path("/home/zifan/ARKitScenes/data/3dod")
    ROOM_SIZE_CSV   = "room_size.csv"

    df = pd.read_csv(ROOM_SIZE_CSV)
    df = df[(df.dataset=="arkitscenes") & (df.question_type=="room_size_estimation")]
    
    total_nc    = correct_nc  = 0
    mra_nc_list = []
    total_wc    = correct_wc  = 0
    mra_wc_list = []

    for _, row in df.iterrows():
        scene_id = str(row.scene_name)
        question = row.question
        try:
            gt = float(row.ground_truth)
            gt = int(gt) if gt.is_integer() else gt
        except:
            continue

        map_path   = SCENE_MAPS_DIR / f"{scene_id}.png"
        frames_dir = BASE_DATA_DIR / row.dataset / scene_id / f"{scene_id}_frames" / "lowres_wide"
        if map_path.is_file():
            scene_map = Image.open(map_path).convert("RGB")
        else:
            scene_map = None

        if frames_dir.is_dir():
            all_frames = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
            indices    = np.linspace(0, len(all_frames)-1, 64, dtype=int)
            video_frames = [Image.open(all_frames[i]).convert("RGB") for i in indices]
        else:
            video_frames = None

        txt_file = SCENE_TXT_DIR / f"{scene_id}.txt"
        if not txt_file.is_file():
            print(f"[WARN] missing scene txt for {scene_id}, skipping")
            continue
        info = txt_file.read_text().strip()
        proc = process_scene_info(info)
        if proc is None:
            continue 
        
        prompt_nc = (
            "Estimate the total area of the room in square meters. "
            "If multiple rooms are present, estimate the combined area. "
            "Provide only a single number.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        resp_nc = query_llava_multimodal(prompt_nc, video_frames=video_frames)
        pred_nc = extract_number(resp_nc)
        if pred_nc is not None:
            total_nc += 1
            if pred_nc == gt:
                correct_nc += 1
            mra_nc_list.append(mean_relative_accuracy(pred_nc, gt))
        else:
            print(f"[{scene_id}] No-Ctx parse fail: “{resp_nc}”")

        prompt_wc = (
            "Here is the 2D scene map and a short video clip of the space.\n\n"
            "Use them to improve your estimate.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        resp_wc = query_llava_multimodal(prompt_wc, scene_map=scene_map, video_frames=video_frames)
        pred_wc = extract_number(resp_wc)
        if pred_wc is not None:
            total_wc += 1
            if pred_wc == gt:
                correct_wc += 1
            mra_wc_list.append(mean_relative_accuracy(pred_wc, gt))
        else:
            print(f"[{scene_id}] W-Ctx parse fail: “{resp_wc}”")

    acc_nc = correct_nc/total_nc*100 if total_nc else 0.0
    mra_nc = np.mean(mra_nc_list)*100 if mra_nc_list else 0.0
    acc_wc = correct_wc/total_wc*100 if total_wc else 0.0
    mra_wc = np.mean(mra_wc_list)*100 if mra_wc_list else 0.0

    print("\n" + "="*40)
    print("Room Size Estimation Results:")
    print(f"  No-Context    : {correct_nc}/{total_nc} = {acc_nc:.2f}%,  MRA = {mra_nc:.2f}%")
    print(f"  With-Context  : {correct_wc}/{total_wc} = {acc_wc:.2f}%,  MRA = {mra_wc:.2f}%")
    print("="*40)

if __name__=="__main__":
    main_room_size_estimation()