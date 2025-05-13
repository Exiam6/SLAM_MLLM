# Recreating README.md in /mnt/data via python_user_visible
readme_content = """# SLAM-MLLM

SLAM-MLLM is an explicit 3D spatial memory module for multimodal large language models. It reconstructs a dense, loop-closed point cloud from video via MASt3R-SLAM, encodes it with a SpatialLLM-style point-cloud encoder, and integrates those features into an LLM (e.g. LLaVA-NeXT-Video-7B) for structured spatial reasoning (object counting, size estimation, room layout).

---

## ⚙️ Installation

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

## 🚀 Quick Start

1. **Reconstruct** your point clouds with MASt3R-SLAM 
2. **Encode + Inference** with `inference.py` to produce JSON layouts.  
3. **Visualize** with `visualize.py` (outputs Rerun `.rrd` files).  

---

## 🧪 Evaluation

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

## 📂 Repository Structure

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

<<<<<<< HEAD
=======
Run inference:

```bash
python inference.py --point_cloud pcd/scene0000_00.ply --output scene0000_00.txt --model_path manycore-research/SpatialLM-Llama-1B
```

### Visualization

Use `rerun` to visualize the point cloud and the predicted structured 3D layout output:

```bash
# Convert the predicted layout to Rerun format
python visualize.py --point_cloud pcd/scene0000_00.ply --layout scene0000_00.txt --save scene0000_00.rrd

# Visualize the point cloud and the predicted layout
rerun scene0000_00.rrd
```

### Evaluation

To evaluate the performance of SpatialLM, we provide `eval.py` script that reports the benchmark results on the SpatialLM-Testset in the table below in section [Benchmark Results](#benchmark-results).

Download the testset:

```bash
huggingface-cli download manycore-research/SpatialLM-Testset --repo-type dataset --local-dir SpatialLM-Testset
```

Run evaluation:

```bash
# Run inference on the PLY point clouds in folder SpatialLM-Testset/pcd with SpatialLM-Llama-1B model
python inference.py --point_cloud SpatialLM-Testset/pcd --output SpatialLM-Testset/pred --model_path manycore-research/SpatialLM-Llama-1B

# Evaluate the predicted layouts
python eval.py --metadata SpatialLM-Testset/test.csv --gt_dir SpatialLM-Testset/layout --pred_dir SpatialLM-Testset/pred --label_mapping SpatialLM-Testset/benchmark_categories.tsv
```

### Example using a custom video

We provide an example of how to use our model to estimate scene layout starting from a RGB video with the newly released [SLAM3R](https://github.com/PKU-VCL-3DV/SLAM3R) in [EXAMPLE.md](EXAMPLE.md). These steps work for MASt3R-SLAM, and other reconstruction methods as well.

## SpatialLM Testset

We provide a test set of 107 preprocessed point clouds, reconstructed from RGB videos using [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM). SpatialLM-Testset is quite challenging compared to prior clean RGBD scans datasets due to the noises and occlusions in the point clouds reconstructed from monocular RGB videos.

<div align="center">

|    **Dataset**    | **Download**                                                                       |
| :---------------: | ---------------------------------------------------------------------------------- |
| SpatialLM-Testset | [🤗 Datasets](https://huggingface.co/datasets/manycore-research/SpatialLM-TestSet) |

</div>

## Benchmark Results

Benchmark results on the challenging SpatialLM-Testset are reported in the following table:

<div align="center">

| **Method**       | **SpatialLM-Llama-1B** | **SpatialLM-Qwen-0.5B** |
| ---------------- | ---------------------- | ----------------------- |
| **Floorplan**    | **mean IoU**           |                         |
| wall             | 78.62                  | 74.81                   |
|                  |                        |                         |
| **Objects**      | **F1 @.25 IoU (3D)**   |                         |
| curtain          | 27.35                  | 28.59                   |
| nightstand       | 57.47                  | 54.39                   |
| chandelier       | 38.92                  | 40.12                   |
| wardrobe         | 23.33                  | 30.60                   |
| bed              | 95.24                  | 93.75                   |
| sofa             | 65.50                  | 66.15                   |
| chair            | 21.26                  | 14.94                   |
| cabinet          | 8.47                   | 8.44                    |
| dining table     | 54.26                  | 56.10                   |
| plants           | 20.68                  | 26.46                   |
| tv cabinet       | 33.33                  | 10.26                   |
| coffee table     | 50.00                  | 55.56                   |
| side table       | 7.60                   | 2.17                    |
| air conditioner  | 20.00                  | 13.04                   |
| dresser          | 46.67                  | 23.53                   |
|                  |                        |                         |
| **Thin Objects** | **F1 @.25 IoU (2D)**   |                         |
| painting         | 50.04                  | 53.81                   |
| carpet           | 31.76                  | 45.31                   |
| tv               | 67.31                  | 52.29                   |
| door             | 50.35                  | 42.15                   |
| window           | 45.4                   | 45.9                    |

</div>

## License

SpatialLM-Llama-1B is derived from Llama3.2-1B-Instruct, which is licensed under the Llama3.2 license.
SpatialLM-Qwen-0.5B is derived from the Qwen-2.5 series, originally licensed under the Apache 2.0 License.

All models are built upon the SceneScript point cloud encoder, licensed under the CC-BY-NC-4.0 License. TorchSparse, utilized in this project, is licensed under the MIT License.

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{spatiallm,
  title        = {SpatialLM: Large Language Model for Spatial Understanding},
  author       = {ManyCore Research Team},
  howpublished = {\url{https://github.com/manycore-research/SpatialLM}},
  year         = {2025}
}
```

## Acknowledgements

We would like to thank the following projects that made this work possible:

[Llama3.2](https://github.com/meta-llama) | [Qwen2.5](https://github.com/QwenLM/Qwen2.5) | [Transformers](https://github.com/huggingface/transformers) | [SceneScript](https://github.com/facebookresearch/scenescript) | [TorchSparse](https://github.com/mit-han-lab/torchsparse)
>>>>>>> d52a6a5165490ff13fe430e4e8e572c9033099b8
