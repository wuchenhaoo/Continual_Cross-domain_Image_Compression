# Continual Cross-domain Image Compression via Entropy Prior Guided Knowledge Distillation and Scalable Decoding

Pytorch Code for the paper "Continual Cross-domain Image Compression via Entropy Prior Guided Knowledge Distillation and Scalable Decoding". 

This repository is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI).

## Installation

```bash
conda create -n ccic python=3.7
conda activate ccic
pip install compressai
pip install tqdm
```

## Usage

**Training**

```bash
CUDA_VISIBLE_DEVICES='1' python train.py --model mbt2018 --filters 128 192 --lambda 0.032 --checkpoints_dir /path/to/checkpoint/folder/ --epochs 10000 10000 10000 10000 10000 --save_interval 200 -lr 1e-4
```

**Datasets**

- CLIC [https://www.compression.cc/]
- OP [[Unsplash Dataset | The world’s largest open library dataset](https://unsplash.com/data)]
- GTA5 [https://download.visinf.tu-darmstadt.de/data/from_games/]
- DOTA [https://captain-whu.github.io/DOTA/index.html]
- COVID19 [https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=89096912#89096912aaf792e54ffd4c16ad8dd5b4b117ab3f]
