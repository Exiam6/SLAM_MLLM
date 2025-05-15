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
    num = float(m.group())
    return int(num) if num.is_integer() else num

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
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

def query_llava(prompt, images=None, videos=None):
    img_list = images or []
    vid_list = videos or []
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

# ─── MAIN EVALUATION ──────────────────────────────────────────────────────────

BASE_DATA_DIR     = Path("/home/zifan/ARKitScenes/data/3dod")
DOWNLOAD_LIST_CSV = Path("/home/zifan/ARKitScenes/counting_download_list.csv")
ROOM_SIZE_CSV     = Path("/home/zifan/SpatialLM/room_size.csv")
SCENE_MAPS_DIR    = Path("/home/zifan/SpatialLM/scene_maps")

def main():
    downloads = list(csv.DictReader(open(DOWNLOAD_LIST_CSV, newline="")))

    questions_map = defaultdict(list)
    for row in csv.DictReader(open(ROOM_SIZE_CSV, newline="")):
        if row["question_type"] != "room_size_estimation":
            continue
        try:
            gt = float(row["ground_truth"])
            gt = int(gt) if gt.is_integer() else gt
        except:
            continue
        questions_map[row["scene_name"]].append({
            "question": row["question"].strip(),
            "ground_truth": gt
        })

    total_nc = correct_nc = 0
    mra_nc_list = []
    total_wc = correct_wc = 0
    mra_wc_list = []

    for entry in downloads:
        vid, fold = entry["video_id"], entry["fold"]
        qlist = questions_map.get(vid, [])
        if not qlist:
            continue

        frames_dir = BASE_DATA_DIR / fold / vid / f"{vid}_frames" / "lowres_wide"
        paths = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
        if not paths:
            print(f"[WARN] no frames for {vid}")
            continue
        idxs = np.linspace(0, len(paths)-1, 64, dtype=int)
        video_arr = np.stack([np.array(Image.open(paths[i]).convert("RGB")) for i in idxs], axis=0)

        # load optional 2D map
        map_path = SCENE_MAPS_DIR / f"{vid}.png"
        scene_map = Image.open(map_path).convert("RGB") if map_path.is_file() else None

        for qinfo in qlist:
            question = qinfo["question"] + " Answer only a single number."
            gt = qinfo["ground_truth"]

            # no-context
            resp_nc = query_llava(question, videos=[video_arr])
            pred_nc = extract_number(resp_nc)
            mra_nc = mean_relative_accuracy(pred_nc, gt)
            total_nc += 1
            correct_nc += (pred_nc == gt)
            mra_nc_list.append(mra_nc)

            print(f"\n[No-Context] Video={vid} | Resp: {resp_nc}")
            print(f"  Predicted={pred_nc}, GT={gt}, Exact={'✓' if pred_nc==gt else '✗'}, MRA={mra_nc:.3f}")

            # with-context
            question = qinfo["question"] + " Answer only a single number."
            resp_wc = query_llava(question, images=[scene_map] if scene_map else None)
            pred_wc = extract_number(resp_wc)
            mra_wc = mean_relative_accuracy(pred_wc, gt)
            total_wc += 1
            correct_wc += (pred_wc == gt)
            mra_wc_list.append(mra_wc)

            print(f"[With-Context] Video={vid} | Resp: {resp_wc}")
            print(f"  Predicted={pred_wc}, GT={gt}, Exact={'✓' if pred_wc==gt else '✗'}, MRA={mra_wc:.3f}")

    # summary
    acc_nc = 100 * correct_nc / total_nc if total_nc else 0.0
    acc_wc = 100 * correct_wc / total_wc if total_wc else 0.0
    mra_nc = 100 * np.mean(mra_nc_list) if mra_nc_list else 0.0
    mra_wc = 100 * np.mean(mra_wc_list) if mra_wc_list else 0.0

    print("\n" + "="*40)
    print("Room-Size Estimation Results")
    print(f"No-Context    : {correct_nc}/{total_nc} = {acc_nc:.2f}%, MRA = {mra_nc:.2f}%")
    print(f"With-Context  : {correct_wc}/{total_wc} = {acc_wc:.2f}%, MRA = {mra_wc:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()