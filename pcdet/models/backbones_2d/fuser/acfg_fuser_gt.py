import math
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ACFGGateGT(nn.Module):
    """
    ACFG fusion head with GT-guided gating supervision.

    - 输入:
        batch_dict["spatial_features"]      : LiDAR BEV   [B, C_l, H, W]
        batch_dict["spatial_features_img"]  : Camera BEV  [B, C_c, H, W]
        batch_dict["points"]                : 点云 [N, 5]  (b, x, y, z, ...)
        batch_dict["gt_boxes"]              : GT boxes [B, M, 8+] (x, y, z, dx, dy, dz, yaw, cls)

    - 输出:
        batch_dict["spatial_features"]      : fused BEV   [B, C_out, H, W]
        （训练时）batch_dict["loss_acfg_gate"] : scalar gate loss (未乘 λ)

    说明:
    - 结构上与原来的 ACFGGate 基本一致：1x1 投影 + 3x3 上下文门控 + 1x1 输出 + 残差。
    - 额外加入 build_gate_target_from_gt()，利用 GT box + LiDAR 点密度
      生成 gate 的监督信号，并在前景 cell 上施加 CE loss。
    """

    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        # === 1. 配置读取 ===
        # 和你 yaml 里保持一致
        self.in_channels_lidar = getattr(model_cfg, "IN_CHANNELS_LIDAR", 256)
        self.in_channels_cam = getattr(model_cfg, "IN_CHANNELS_CAM", 80)
        self.out_dim = getattr(model_cfg, "OUT_DIM", 128)
        self.out_channels = getattr(model_cfg, "OUT_CHANNEL", 256)
        self.use_residual = getattr(model_cfg, "USE_RESIDUAL", True)

        # 传感器几何 & gate 监督相关超参
        pc_range = getattr(
            model_cfg,
            "POINT_CLOUD_RANGE",
            [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],  # 默认 nuScenes，可在 yaml 里覆盖
        )
        self.register_buffer(
            "point_cloud_range", torch.as_tensor(pc_range, dtype=torch.float32)
        )

        # LiDAR 点密度 -> gate target 的超参
        self.density_alpha = getattr(model_cfg, "DENSITY_ALPHA", 10.0)
        self.density_high = getattr(model_cfg, "DENSITY_HIGH", 0.7)
        self.density_low = getattr(model_cfg, "DENSITY_LOW", 0.3)

        # Ablation 开关（和原版 ACFG 保持一致）
        self.zero_lidar_by_env = os.environ.get("ACFG_ZERO_LIDAR", "0") == "1"
        self.zero_cam_by_env = os.environ.get("ACFG_ZERO_CAM", "0") == "1"

        # === 2. LiDAR / Camera 侧 1x1 投影到统一维度 ===
        self.lidar_proj = nn.Sequential(
            nn.Conv2d(self.in_channels_lidar, self.out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU(inplace=True),
        )
        self.cam_proj = nn.Sequential(
            nn.Conv2d(self.in_channels_cam, self.out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU(inplace=True),
        )

        # === 3. 上下文门控网络（3x3） ===
        self.gate_net = nn.Sequential(
            nn.Conv2d(self.out_dim * 2, self.out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_dim, 2, kernel_size=1, bias=True),  # 输出 [w_l, w_c] logits
        )

        # === 4. 输出映射到最终通道数，并且支持残差 ===
        self.out_conv = nn.Conv2d(self.out_dim, self.out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.act = nn.ReLU(inplace=True)

        # debug 信息只打印一次
        self._printed_debug = False

        # 供可视化使用
        self.last_w_lidar = None
        self.last_w_cam = None

    @staticmethod
    def _l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
        norm = torch.sqrt(torch.clamp((x ** 2).sum(dim=dim, keepdim=True), min=eps))
        return x / norm

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                spatial_features      : LiDAR BEV   [B, C_l, H, W]
                spatial_features_img  : Camera BEV  [B, C_c, H, W]
        Returns:
            batch_dict:
                spatial_features      : fused BEV   [B, C_out, H, W]
                (训练时) loss_acfg_gate : scalar gate loss（未乘 λ）
        """
        lidar_bev = batch_dict["spatial_features"]           # [B, C_l, H, W]
        cam_bev = batch_dict["spatial_features_img"]         # [B, C_c, H, W]
        B, _, H, W = lidar_bev.shape

        # === 1. Ablation: 可选置零某一模态 ===
        if self.zero_lidar_by_env:
            lidar_bev = torch.zeros_like(lidar_bev)
        if self.zero_cam_by_env:
            cam_bev = torch.zeros_like(cam_bev)

        # === 2. 统一投影到 out_dim，并做 L2 Normalization ===
        x_l = self.lidar_proj(lidar_bev)    # [B, D, H, W]
        x_c = self.cam_proj(cam_bev)        # [B, D, H, W]

        x_l_norm = self._l2_normalize(x_l, dim=1)
        x_c_norm = self._l2_normalize(x_c, dim=1)

        # === 3. 上下文门控网络，输出 softmax gate ===
        x_cat = torch.cat([x_l_norm, x_c_norm], dim=1)   # [B, 2D, H, W]
        gate_logits = self.gate_net(x_cat)               # [B, 2, H, W]
        gate = torch.softmax(gate_logits, dim=1)

        w_l = gate[:, 0:1, :, :]   # [B,1,H,W]
        w_c = gate[:, 1:2, :, :]   # [B,1,H,W]

        # 保存最近一帧的权重供 demo 可视化
        self.last_w_lidar = w_l.detach()
        self.last_w_cam = w_c.detach()

        # === 4. 加权融合 & 输出映射 ===
        fused = w_l * x_l + w_c * x_c         # [B, D, H, W]
        out = self.out_conv(fused)            # [B, C_out, H, W]

        if self.use_residual and lidar_bev.shape[1] == self.out_channels:
            out = out + lidar_bev

        out = self.bn(out)
        out = self.act(out)

        batch_dict["spatial_features"] = out

        # === 5. 构建 GT 引导的 gate loss（只在训练阶段） ===
        if self.training and ("gt_boxes" in batch_dict) and ("points" in batch_dict):
            with torch.no_grad():
                target, mask = self.build_gate_target_from_gt(
                    batch_dict=batch_dict, H=H, W=W, device=out.device
                )  # target: [B,2,H,W], mask: [B,1,H,W]

            gate_loss = self.compute_gate_loss(w_l, w_c, target, mask)
            batch_dict["loss_acfg_gate"] = gate_loss

        # 打印一次 debug 信息
        if not self._printed_debug:
            print(f"[ACFG-GT] Input Shapes - LiDAR: {lidar_bev.shape}, Cam: {cam_bev.shape}")
            print(f"[ACFG-GT] Gate Mean (no GT loss yet) - LiDAR: {w_l.mean():.3f}, Cam: {w_c.mean():.3f}")
            self._printed_debug = True

        return batch_dict

    # ----------------------------------------------------------------------
    # GT-supervised gate target & loss
    # ----------------------------------------------------------------------

    def build_gate_target_from_gt(self, batch_dict, H: int, W: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        利用 GT boxes + LiDAR 点密度 构造 gate target。

        步骤:
        1. 对每个 batch / 每个 GT box 统计 LiDAR 点数 N_k。
        2. 归一化为密度 d_k = N_k / (N_k + alpha)。
        3. 根据 d_k -> [y_L, y_C] (hard / soft label)。
        4. 把 [y_L, y_C] rasterize 到该 box 对应的 BEV 栅格区域，形成 target & mask。

        Returns:
            target: [B, 2, H, W]  (y_L, y_C)
            mask  : [B, 1, H, W]  (1 表示前景 cell，有 gate 监督)
        """
        points = batch_dict["points"]              # [N,5] (b,x,y,z,intensity)
        gt_boxes = batch_dict["gt_boxes"]          # [B,M,7/8/9]
        B = gt_boxes.shape[0]

        target = torch.zeros((B, 2, H, W), device=device)
        mask = torch.zeros((B, 1, H, W), device=device)

        pc_range = self.point_cloud_range
        x_min, y_min, _, x_max, y_max, _ = pc_range.tolist()
        x_size = (x_max - x_min) / float(W)
        y_size = (y_max - y_min) / float(H)

        alpha = float(self.density_alpha)
        d_high = float(self.density_high)
        d_low = float(self.density_low)

        for b in range(B):
            # 当前 batch 的点
            batch_mask = points[:, 0].long() == b
            pts_b = points[batch_mask][:, 1:4]  # [n_b, 3]  (x,y,z)

            # 当前 batch 有效 GT（类别 > 0）
            cur_gt = gt_boxes[b]  # [M,7/8/9]
            if cur_gt.shape[-1] >= 8:
                cls_ids = cur_gt[:, -1]
                valid_gt = cur_gt[cls_ids > 0]
            else:
                # 没有类别维度时，用 all-zero 过滤
                valid_mask = (cur_gt.abs().sum(dim=-1) > 0)
                valid_gt = cur_gt[valid_mask]

            if valid_gt.numel() == 0:
                continue

            for k in range(valid_gt.shape[0]):
                box = valid_gt[k]  # (x,y,z,dx,dy,dz,yaw[,cls])
                cx, cy, cz, dx, dy, dz, yaw = box[:7].tolist()

                if pts_b.numel() == 0:
                    N_k = 0
                else:
                    # 只在 box 周围一个 margin 内考虑点，减少计算
                    margin = 2.0  # meters
                    x_cond = (pts_b[:, 0] > cx - dx / 2 - margin) & (pts_b[:, 0] < cx + dx / 2 + margin)
                    y_cond = (pts_b[:, 1] > cy - dy / 2 - margin) & (pts_b[:, 1] < cy + dy / 2 + margin)
                    cand_mask = x_cond & y_cond
                    pts_cand = pts_b[cand_mask]

                    if pts_cand.numel() == 0:
                        N_k = 0
                    else:
                        # 坐标旋转到 box 坐标系里做精确 in-box 判断
                        cosa = math.cos(-yaw)
                        sina = math.sin(-yaw)
                        x_shift = pts_cand[:, 0] - cx
                        y_shift = pts_cand[:, 1] - cy
                        x_rot = x_shift * cosa - y_shift * sina
                        y_rot = x_shift * sina + y_shift * cosa

                        half_dx = dx / 2.0
                        half_dy = dy / 2.0
                        inside_mask = (
                            (x_rot >= -half_dx) & (x_rot <= half_dx) &
                            (y_rot >= -half_dy) & (y_rot <= half_dy)
                        )
                        N_k = inside_mask.long().sum().item()

                # LiDAR 点密度 (0~1)
                d_k = N_k / (N_k + alpha)

                # 根据密度构造 target gate [y_L, y_C]
                if d_k >= d_high:
                    # LiDAR 优势明显
                    y_L, y_C = 1.0, 0.0
                elif d_k <= d_low:
                    # Camera 优势明显
                    y_L, y_C = 0.0, 1.0
                else:
                    # 中间区域给个平滑过渡 (线性插值)
                    t = (d_k - d_low) / max(d_high - d_low, 1e-6)
                    # 中间区间让 y_L 约在 [0.25, 0.75] 内平滑变化
                    y_L = d_low + t * (d_high - d_low)
                    y_L = float(max(0.0, min(1.0, y_L)))
                    y_C = 1.0 - y_L

                # 把这个 object 的 target 写到对应 BEV 区域
                # 简化起见，用 axis-aligned 的包围盒
                x_min_box = cx - dx / 2.0
                x_max_box = cx + dx / 2.0
                y_min_box = cy - dy / 2.0
                y_max_box = cy + dy / 2.0

                ix_min = int((x_min_box - x_min) / x_size)
                ix_max = int((x_max_box - x_min) / x_size)
                iy_min = int((y_min_box - y_min) / y_size)
                iy_max = int((y_max_box - y_min) / y_size)

                ix_min = max(ix_min, 0)
                iy_min = max(iy_min, 0)
                ix_max = min(ix_max, W - 1)
                iy_max = min(iy_max, H - 1)

                if ix_min > ix_max or iy_min > iy_max:
                    continue

                # 如果多个 box 重叠，则简单平均（极少出现）
                cur_region = target[b, :, iy_min : iy_max + 1, ix_min : ix_max + 1]
                cur_mask = mask[b, :, iy_min : iy_max + 1, ix_min : ix_max + 1]

                new_target = torch.tensor([y_L, y_C], device=device).view(2, 1, 1)

                if cur_mask.sum() == 0:
                    target[b, :, iy_min : iy_max + 1, ix_min : ix_max + 1] = new_target
                    mask[b, :, iy_min : iy_max + 1, ix_min : ix_max + 1] = 1.0
                else:
                    # 平均已有的 target 和新的 target
                    target[b, :, iy_min : iy_max + 1, ix_min : ix_max + 1] = (
                        cur_region + new_target
                    ) / 2.0
                    mask[b, :, iy_min : iy_max + 1, ix_min : ix_max + 1] = 1.0

        return target, mask

    def compute_gate_loss(
        self,
        w_l: torch.Tensor,
        w_c: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        对前景区域的 gate 做交叉熵损失:
            y_L * log(p_L) + y_C * log(p_C)

        Args:
            w_l: [B,1,H,W]
            w_c: [B,1,H,W]
            target: [B,2,H,W]
            mask: [B,1,H,W]

        Returns:
            scalar tensor (未乘 λ)
        """
        B, _, H, W = w_l.shape
        eps = 1e-6

        pred = torch.cat([w_l, w_c], dim=1).clamp(min=eps, max=1.0 - eps)  # [B,2,H,W]

        # CE per-cell: - sum_c y_c log p_c
        ce_per_cell = - (target * pred.log()).sum(dim=1, keepdim=True)  # [B,1,H,W]

        valid = (mask > 0.5).float()
        if valid.sum() == 0:
            # 没有前景，返回 0，不影响训练
            return pred.new_tensor(0.0)

        gate_loss = (ce_per_cell * valid).sum() / (valid.sum() + eps)
        return gate_loss
