## Techinical Report

[Enhancing Small Object Detection with Lightweight Attention: Structural Optimization of SimAM in RT-DETR-L](./report.md)

## Reproduction Instructions

CUDA needed for training. Tested on CUDA 13 with Python 12 on a NVIDIA RTX 4070.

```bash
uv venv
uv pip install -e .
```

From [bccd.yaml](./ultralytics/cfg/datasets/bccd.yaml), download the BCCD dataset and place it in the `datasets/bccd/` directory. Remember to adjust the paths in `bccd.yaml`.

```bash
# remember to adjust the logs path and batch size as needed
python train-baseline.py # for baseline RT-DETR-L
python train-simam.py # for RT-DETR-L with SimAM at Stage 4
python train-simam-hgblock.py # for RT-DETR-L with SimAM at HGBlock
python train-siman-pre_s2.py # for RT-DETR-L with SimAM before Stage 2
python train-siman-post_s2.py # for RT-DETR-L with SimAM after Stage 2
```
