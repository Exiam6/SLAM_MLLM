# On Improving Spatial Reasoning of MLLMs


SLAM-MLLM is a spatial reasoning pipeline with explicit 3D spatial memory module for multimodal large language models. It reconstructs a dense, loop-closed point cloud from video via SLAM 3D reconstruction, encodes it with point-cloud encoder, integrates those features into an LLM for structured scene information and enhance the spatial reasoning ability of current Multimodal Large Language models


## Installation

Tested on:

- Python 3.11  
- PyTorch 2.4.1  
- CUDA 12.4  

```bash
# 1. Clone the repo
git clone https://github.com/Exiam6/slam_mllm.git
cd slam_mllm

# 2. Create Conda env
conda create -n slam_mllm python=3.11 -y
conda activate slam_mllm
conda install -y nvidia/label/cuda-12.4.0::cuda-toolkit conda-forge::sparsehash

# 3. Install dependencies
pip install poetry
poetry config virtualenvs.create false --local
poetry install
poe install-torchsparse  # may take a while to build torchsparse
```

---

## Steps to start

1. **Reconstruct** your point clouds with MASt3R-SLAM 
2. **Encode + Inference** with `inference.py` to produce JSON layouts.  
3. **Visualize** with `visualize.py` (outputs Rerun `.rrd` files).  

---

## Evaluation

We provide three evaluation scripts to measure performance on standard spatial reasoning tasks:

- **Object Counting**:  
  ```bash
  python eval_count.py \\
    --pred_dir outputs/predictions/counting \\
    --gt_dir data/ground_truth/counting \\
    --output results/counting_results.csv
  ```

- **Object Size Estimation**:  
  ```bash
  python eval_obj_size.py \\
    --pred_dir outputs/predictions/size \\
    --gt_dir data/ground_truth/size \\
    --output results/size_results.csv
  ```

- **Room Size Estimation**:  
  ```bash
  python eval_room_size.py \\
    --pred_dir outputs/predictions/room_size \\
    --gt_dir data/ground_truth/room_size \\
    --output results/room_size_results.csv
  ```

Each script will print a brief summary to stdout and write a detailed CSV with per-scene metrics.

---

## Repository Structure

```
slam_mllm/
├── data/                      # raw and ground-truth data
│   ├── ground_truth/
│   │   ├── counting/
│   │   ├── size/
│   │   └── room_size/
│   └── videos/
├── outputs/                   # inference outputs & evaluation results
│   ├── predictions/
│   │   ├── counting/
│   │   ├── size/
│   │   └── room_size/
│   └── results/
├── scripts/                   # helper scripts (reconstruct, inference, visualize)
├── eval_count.py              # counting evaluation
├── eval_obj_size.py           # object-size evaluation
├── eval_room_size.py          # room-size evaluation
├── inference.py
├── visualize.py
├── reconstruct.py
├── pyproject.toml
└── README.md
```
