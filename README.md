# ACFG BEV Fusion
This repository contains my custom modifications for robust LiDAR-camera BEV fusion built on top of OpenPCDet / BEVFusion.
## Overview
The main contribution is an Adaptive Confidence Fusion Gate (ACFG) that replaces the standard ConvFuser with an explicit spatially varying fusion module. I also include a GT-guided / density-guided supervision version, LiDAR corruption processing for robustness experiments, and BEV visualization scripts for qualitative analysis.
## What is Included
- `configs/bevfusion_acfg.yaml`: config for the end-to-end ACFG model
- `configs/bevfusion_acfg_gt.yaml`: config for the GT-guided / supervised ACFG model
- `pcdet/models/backbones_2d/fuser/acfg_fuser.py`: ACFG fusion head
- `pcdet/models/backbones_2d/fuser/acfg_fuser_gt.py`: supervised ACFG fusion head
- `pcdet/models/backbones_2d/fuser/__init__.py`: fuser module registration
- `pcdet/datasets/processor/data_processor.py`: modified data processor with corruption hook
- `pcdet/datasets/processor/lidar_corruption.py`: LiDAR corruption functions
- `tools/demo_acfg_gate_bev.py`: gate heatmap visualization
- `tools/demo_bevfusion_camratio_bev.py`: baseline camera-ratio BEV visualization

- ## Report
A short project report is available in `docs/AER1515_ACFG_report.pdf`. It summarizes the method, experimental setup, robustness evaluation, and qualitative visualization results.

## How to Use
This is not a full standalone detection framework. It is a compact extension repository that contains only the files I modified for this project.
To use it:
1. Prepare a working OpenPCDet / BEVFusion codebase.
2. Copy the files in this repository into the matching locations of the base project.
3. Use `bevfusion_acfg.yaml` or `bevfusion_acfg_gt.yaml` as the model config.
4. Train or evaluate within the original OpenPCDet workflow.
## Example
Typical visualization commands look like:
```bash
python tools/demo_acfg_gate_bev.py --cfg_file configs/bevfusion_acfg.yaml --ckpt /path/to/checkpoint.pth --out_dir ./vis_results/acfg
python tools/demo_bevfusion_camratio_bev.py --cfg_file configs/bevfusion_acfg.yaml --ckpt /path/to/checkpoint.pth --out_dir ./vis_results/baseline
