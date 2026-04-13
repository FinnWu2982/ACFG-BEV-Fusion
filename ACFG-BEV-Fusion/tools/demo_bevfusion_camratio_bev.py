import argparse
import os
import numpy as np
import torch
import matplotlib

# 使用无窗口后端，避免服务器/WSL 没有显示时报错
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils


def parse_args():
    parser = argparse.ArgumentParser("NuScenes BEVFusion camera-ratio BEV demo")
    parser.add_argument("--cfg_file", type=str, required=True, help="config yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint path")
    parser.add_argument("--out_dir", type=str, required=True, help="output dir for images")
    parser.add_argument("--max_frames", type=int, default=1, help="how many frames to save")
    parser.add_argument("--score_thresh", type=float, default=0.3, help="min score to draw a box")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def build_model_and_dataloader(args, logger):
    # 载入配置
    cfg_from_yaml_file(args.cfg_file, cfg)
    logger.info(f"cfg file: {args.cfg_file}")

    # dataloader: 基本照抄 tools/test.py 的用法
    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=2,
        logger=logger,
        training=False,
    )

    # build model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.cuda(args.gpu)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.eval()
    logger.info(f"Loaded ckpt from {args.ckpt}")

    # ====== 在模型里找到 ConvFuser 模块，并挂 forward hook ======
    fusion_module = None
    for m in model.modules():
        if m.__class__.__name__ == "ConvFuser":
            fusion_module = m
            break

    assert fusion_module is not None, "ConvFuser module not found in model (class name 'ConvFuser')."

    def convfuser_hook(module, inputs, output):
        """
        inputs: tuple，只包含一个 batch_dict
        """
        batch_dict = inputs[0]
        img_bev = batch_dict["spatial_features_img"]   # [B, C_cam, H, W]
        lidar_bev = batch_dict["spatial_features"]     # [B, C_lidar, H, W]

        with torch.no_grad():
            cam_norm = img_bev.norm(dim=1)        # [B, H, W]
            lidar_norm = lidar_bev.norm(dim=1)    # [B, H, W]
            ratio = cam_norm / (cam_norm + lidar_norm + 1e-6)  # camera 能量占比

            module.last_cam_ratio = ratio.detach()

    fusion_module.register_forward_hook(convfuser_hook)

    return model, dataloader, fusion_module


def bev_draw_one_frame_acfg_style(
    points_xy,
    pred_boxes,
    pred_scores,
    pred_labels,
    cam_ratio,
    out_path,
    score_thresh=0.3,
):
    """
    画一帧 BEV：
      - 背景：camera 能量比例热力图（和 ACFG gate 图同风格）
      - 前景：点云散点 + 预测框
    坐标范围固定为 [-54, 54] x [-54, 54]，和 demo_acfg_gate_bev 一致。
    """
    # 和 ACFG demo 保持完全一致的画布设置
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    # 1) camera ratio 热力图
    if isinstance(cam_ratio, torch.Tensor):
        ratio_np = cam_ratio.cpu().numpy()
    else:
        ratio_np = np.asarray(cam_ratio)

    # 期望形状 [H, W]
    if ratio_np.ndim == 3:
        ratio_np = ratio_np[0]
    elif ratio_np.ndim == 4:
        ratio_np = ratio_np[0, 0]

    extent = [-54.0, 54.0, -54.0, 54.0]

    im = ax.imshow(
        ratio_np,
        origin="lower",
        extent=extent,
        cmap="viridis",
        alpha=0.6,
        vmin=0.0,
        vmax=1.0,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("camera energy ratio  E_cam / (E_cam + E_lidar)", fontsize=8)

    # 2) 点云散点
    if points_xy is not None and len(points_xy) > 0:
        pts = np.asarray(points_xy)
        if pts.shape[0] > 60000:
            idx = np.random.choice(pts.shape[0], 60000, replace=False)
            pts = pts[idx]
        ax.scatter(pts[:, 0], pts[:, 1], s=0.1, alpha=0.5)

    # 3) 画预测框（和 ACFG demo 同样的矩形投影）
    if pred_boxes is not None and len(pred_boxes) > 0:
        for box, score, cls_id in zip(pred_boxes, pred_scores, pred_labels):
            if score < score_thresh:
                continue

            box = np.asarray(box)
            x, y, z, dx, dy, dz, yaw = box[:7]

            corners = np.array(
                [
                    [dx / 2, dy / 2],
                    [dx / 2, -dy / 2],
                    [-dx / 2, -dy / 2],
                    [-dx / 2, dy / 2],
                ]
            )

            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            R = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
            corners_world = corners @ R.T
            corners_world[:, 0] += x
            corners_world[:, 1] += y

            xs = np.append(corners_world[:, 0], corners_world[0, 0])
            ys = np.append(corners_world[:, 1], corners_world[0, 1])
            ax.plot(xs, ys, linewidth=1.0)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("BEVFusion BEV + camera energy ratio")
    ax.set_aspect("equal", "box")
    ax.set_xlim(-54.0, 54.0)
    ax.set_ylim(-54.0, 54.0)
    ax.grid(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    log_file = os.path.join(args.out_dir, "demo_log.txt")
    logger = common_utils.create_logger(log_file, rank=0)
    logger.info("===== NuScenes BEVFusion camera-ratio BEV demo =====")
    logger.info(f"cfg_file: {args.cfg_file}")
    logger.info(f"ckpt: {args.ckpt}")
    logger.info(f"out_dir: {args.out_dir}")

    model, dataloader, fusion_module = build_model_and_dataloader(args, logger)

    frame_count = 0
    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(dataloader):
            if frame_count >= args.max_frames:
                break

            load_data_to_gpu(batch_dict)
            pred_dicts, _ = model(batch_dict)

            # points: (N, 5) -> [batch_idx, x, y, z, intensity]
            points = batch_dict["points"].cpu().numpy()
            mask = points[:, 0] == 0  # batch_idx == 0
            points_xy = points[mask, 1:3]

            pred_boxes = pred_dicts[0]["pred_boxes"].cpu().numpy()
            pred_scores = pred_dicts[0]["pred_scores"].cpu().numpy()
            pred_labels = pred_dicts[0]["pred_labels"].cpu().numpy()

            # 从 ConvFuser hook 里拿 camera ratio
            cam_ratio = fusion_module.last_cam_ratio[0]  # [H, W]

            out_path = os.path.join(args.out_dir, f"frame_camratio_{frame_count:04d}.png")
            bev_draw_one_frame_acfg_style(
                points_xy=points_xy,
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
                pred_labels=pred_labels,
                cam_ratio=cam_ratio,
                out_path=out_path,
                score_thresh=args.score_thresh,
            )

            logger.info(f"Saved frame {frame_count} (dataset idx {batch_idx}) -> {out_path}")
            frame_count += 1

    logger.info(f"Done. Total frames saved: {frame_count}")


if __name__ == "__main__":
    main()
