import argparse
import os
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # 使用无窗口后端，避免 Tk 报错
import matplotlib.pyplot as plt

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from pcdet.models import build_network


def draw_bev_boxes(ax, boxes, scores, labels, score_thresh=0.3):
    """
    在现有的 ax 上画 3D box 的 BEV 投影（简单画一个矩形）

    boxes: (N, 7) [x, y, z, dx, dy, dz, yaw]
    scores: (N,)
    labels: (N,)
    """
    if boxes is None or len(boxes) == 0:
        return

    for box, score, cls_id in zip(boxes, scores, labels):
        if score < score_thresh:
            continue

        # NuScenes box 通常是 [x, y, z, dx, dy, dz, yaw, vx, vy]
        # 我们只取前 7 个，用 [:7] 兼容 (7,) 或 (9,)
        x, y, z, dx, dy, dz, yaw = box[:7]

        # 先在 box 局部坐标系里定义四个角点
        corners = np.array([
            [ dx / 2,  dy / 2],
            [ dx / 2, -dy / 2],
            [-dx / 2, -dy / 2],
            [-dx / 2,  dy / 2],
        ])  # (4, 2)

        # 绕 z 轴旋转 yaw
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        R = np.array([[cos_yaw, -sin_yaw],
                      [sin_yaw,  cos_yaw]])  # (2, 2)

        corners_world = corners @ R.T  # (4, 2)
        corners_world[:, 0] += x
        corners_world[:, 1] += y

        # 闭合多边形
        xs = np.append(corners_world[:, 0], corners_world[0, 0])
        ys = np.append(corners_world[:, 1], corners_world[0, 1])

        ax.plot(xs, ys, linewidth=1.0)


def bev_draw_one_frame_with_gate(points_xy,
                                 pred_boxes,
                                 pred_scores,
                                 pred_labels,
                                 gate_map,
                                 out_path,
                                 score_thresh=0.3):
    """
    绘制一帧 BEV：
      - 背景点云（LiDAR）
      - ACFG 相机 gate 热力图（w_cam）
      - 检测框
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    # 1) gate 热力图
    if gate_map is not None:
        if isinstance(gate_map, torch.Tensor):
            gate_np = gate_map.numpy()
        else:
            gate_np = np.asarray(gate_map)

        H, W = gate_np.shape
        extent = [-54.0, 54.0, -54.0, 54.0]

        im = ax.imshow(
            gate_np,
            origin="lower",
            extent=extent,
            cmap="viridis",
            alpha=0.6,
            vmin=0.0,
            vmax=1.0,
        )
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("ACFG gate (w_cam)", fontsize=8)

    # 2) 点云散点
    if points_xy is not None and len(points_xy) > 0:
        pts = np.asarray(points_xy)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=0.1,
            alpha=0.5,
        )

    # 3) 检测框
    if pred_boxes is not None and len(pred_boxes) > 0:
        draw_bev_boxes(ax, pred_boxes, pred_scores, pred_labels, score_thresh=score_thresh)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("ACFG BEV + gate heatmap")
    ax.set_aspect("equal", "box")
    ax.set_xlim(-54.0, 54.0)
    ax.set_ylim(-54.0, 54.0)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_model_and_dataloader(args, logger):
    """
    和 demo_static_bev.py 类似：
      - 读 cfg
      - 构建 dataset / dataloader
      - 构建并加载模型
    """
    cfg_from_yaml_file(args.cfg_file, cfg)

    dataset, dataloader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )
    model.cuda(args.gpu)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.eval()

    logger.info(f"Loaded ckpt from {args.ckpt}")
    return model, dataloader, dataset


def parse_args():
    parser = argparse.ArgumentParser(description="NuScenes BEV ACFG gate demo")
    parser.add_argument("--cfg_file", type=str, required=True, help="config file, e.g. bevfusion_acfg.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint, e.g. checkpoint_epoch_12.pth")
    parser.add_argument("--out_dir", type=str, required=True, help="output directory for frames")
    parser.add_argument("--max_frames", type=int, default=80, help="how many frames to dump")
    parser.add_argument("--score_thresh", type=float, default=0.3, help="detection score threshold")
    parser.add_argument("--workers", type=int, default=2, help="num workers for dataloader")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    log_file = os.path.join(args.out_dir, "demo_log.txt")
    logger = common_utils.create_logger(log_file, rank=0)
    logger.info("===== NuScenes BEV demo for ACFG gate =====")
    logger.info(f"cfg_file: {args.cfg_file}")
    logger.info(f"ckpt: {args.ckpt}")
    logger.info(f"out_dir: {args.out_dir}")

    # 1) 建模型 + dataloader
    model, dataloader, dataset = build_model_and_dataloader(args, logger)
    model.eval()
    module_root = model.module if hasattr(model, "module") else model

    from pcdet.models import load_data_to_gpu

    # 2) 在整个网络里找“带 last_w_cam 属性”的模块，当成 ACFG gate
    acfg_gate_module = None
    for name, m in module_root.named_modules():
        if hasattr(m, "last_w_cam"):
            acfg_gate_module = m
            logger.info(f"Found ACFG-like gate module at: {name}")
            break

    if acfg_gate_module is None:
        logger.warning(
            "未在模型中找到带 last_w_cam 属性的模块，"
            "后续只会画点云 + 检测框，不画 gate 热力图。"
        )
    else:
        logger.info("ACFG gate 模块查找成功，将从 gate_module.last_w_cam 读取热力图。")

    # 3) 循环 dataloader，取前 max_frames 帧画出来
    frame_count = 0
    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(dataloader):
            if frame_count >= args.max_frames:
                break

            load_data_to_gpu(batch_dict)
            pred_dicts, _ = model(batch_dict)

            points = batch_dict["points"]
            if isinstance(points, torch.Tensor):
                pts_np = points.cpu().numpy()
            else:
                pts_np = np.asarray(points)

            if pts_np.ndim == 3:
                batch_points = pts_np[0]
            elif pts_np.ndim == 2:
                batch_points = pts_np
            else:
                logger.warning(
                    f"Frame {frame_count}: unexpected points shape {pts_np.shape}, skip."
                )
                continue

            if batch_points.shape[1] < 3:
                logger.warning(
                    f"Frame {frame_count}: points dim < 3 ({batch_points.shape}), skip."
                )
                continue

            # OpenPCDet points 通常是 [batch_idx, x, y, z, ...]，保持和 static demo 一致用 [:,1:3]
            points_xy = batch_points[:, 1:3]

            pred_boxes = pred_dicts[0]["pred_boxes"].cpu().numpy()
            pred_scores = pred_dicts[0]["pred_scores"].cpu().numpy()
            pred_labels = pred_dicts[0]["pred_labels"].cpu().numpy()

            # 4) 从 gate 模块取 gate_map
            gate_np = None
            if acfg_gate_module is not None:
                gate = getattr(acfg_gate_module, "last_w_cam", None)

                if gate is None:
                    logger.warning(
                        f"Frame {frame_count}: gate_module.last_w_cam is None，"
                        "本帧不画 gate 热力图。"
                    )
                else:
                    if isinstance(gate, torch.Tensor):
                        gate_np = gate.detach().cpu().numpy()
                    else:
                        gate_np = np.asarray(gate)

                    # 期望形状为 [B, 1, H, W] / [1, H, W] / [H, W]
                    if gate_np.ndim == 4:
                        gate_np = gate_np[0, 0]
                    elif gate_np.ndim == 3:
                        gate_np = gate_np[0]
                    elif gate_np.ndim == 2:
                        pass
                    else:
                        logger.warning(
                            f"Frame {frame_count}: unexpected gate map shape "
                            f"{gate_np.shape}，本帧跳过 gate 热力图。"
                        )
                        gate_np = None

                    if gate_np is not None:
                        gate_np = np.clip(gate_np, 0.0, 1.0)

            out_path = os.path.join(args.out_dir, f"frame_{frame_count:04d}.png")

            bev_draw_one_frame_with_gate(
                points_xy=points_xy,
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_labels=pred_labels,
                gate_map=gate_np,
                out_path=out_path,
                score_thresh=args.score_thresh,
            )

            logger.info(
                f"Saved frame {frame_count} (dataset idx {batch_idx}) -> {out_path}"
            )
            frame_count += 1

    logger.info(f"Done. Total frames saved: {frame_count}")


if __name__ == "__main__":
    main()
