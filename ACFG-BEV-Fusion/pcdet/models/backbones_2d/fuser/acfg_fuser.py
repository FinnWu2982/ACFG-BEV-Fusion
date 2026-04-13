import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ACFGGate(nn.Module):
    """
    [Upgraded & Fixed] Context-Aware ACFG Fuser
    1. Fixed Key Mismatch: uses 'spatial_features_img' to match OpenPCDet standard.
    2. Added Ablation Flags: ACFG_ZERO_LIDAR / ACFG_ZERO_CAM for controlled testing.
    3. Context-Aware Gating: uses 3x3 convolution.
    """

    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        
        # === 1. 安全读取配置 ===
        self.in_channels_lidar = self.model_cfg.get('IN_CHANNELS_LIDAR', 256)
        self.in_channels_cam = self.model_cfg.get('IN_CHANNELS_CAM', 80)
        self.out_dim = self.model_cfg.get('OUT_DIM', 128)
        self.use_residual = self.model_cfg.get('USE_RESIDUAL', True)

        # === 2. 特征投影层 ===
        self.to_out_lidar = nn.Conv2d(self.in_channels_lidar, self.out_dim, kernel_size=1, bias=False)
        self.to_out_cam = nn.Conv2d(self.in_channels_cam, self.out_dim, kernel_size=1, bias=False)

        # === 3. 门控网络 (3x3 Context-Aware) ===
        self.gate = nn.Sequential(
            nn.Conv2d(self.out_dim * 2, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1, bias=True) # 输出 [B, 2, H, W]
        )

        # === 4. 输出投影层 ===
        self.proj = nn.Conv2d(self.out_dim, self.in_channels_lidar, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(self.in_channels_lidar)
        self.act = nn.ReLU(inplace=True)

        # === 5. 消融实验控制 (Ablation Control) ===
        # 默认为 "0" (关闭消融，正常运行)
        # 设置为 "1" 则强制将对应模态置零
        self.zero_cam_by_env = os.environ.get("ACFG_ZERO_CAM", "0") == "1"
        self.zero_lidar_by_env = os.environ.get("ACFG_ZERO_LIDAR", "0") == "1"
        
        # 调试打印控制
        self._printed_debug = False

    def _maybe_zero_cam(self, cam_bev: torch.Tensor):
        if cam_bev is None:
            return None
        if self.zero_cam_by_env:
            return torch.zeros_like(cam_bev)
        return cam_bev

    def _maybe_zero_lidar(self, lidar_bev: torch.Tensor):
        if self.zero_lidar_by_env:
            return torch.zeros_like(lidar_bev)
        return lidar_bev

    def forward(self, batch_dict):
        # === 1. 获取特征 (Key 已修正) ===
        # 官方 ConvFuser 使用 'spatial_features_img'，必须对齐！
        lidar_bev = batch_dict["spatial_features"]          # [B, 256, H, W]
        cam_bev = batch_dict.get("spatial_features_img", None) 
        
        # 获取 GT Boxes (为后续 GT-Guided Supervision 预留)
        # gt_boxes = batch_dict.get('gt_boxes', None)

        B, _, H_l, W_l = lidar_bev.shape

        # === 2. 处理输入特征 (对齐 & 消融) ===
        # LiDAR 消融
        lidar_bev = self._maybe_zero_lidar(lidar_bev)

        # Camera 处理
        if cam_bev is None:
            cam_bev = lidar_bev.new_zeros(B, self.in_channels_cam, H_l, W_l)
        else:
            cam_bev = self._maybe_zero_cam(cam_bev)
            # 尺寸对齐
            if cam_bev.shape[-2:] != (H_l, W_l):
                cam_bev = F.interpolate(
                    cam_bev, size=(H_l, W_l), mode="bilinear", align_corners=False
                )

        # === 3. 投影 & 强制归一化 (Normalization) ===
        x_l = self.to_out_lidar(lidar_bev)
        x_c = self.to_out_cam(cam_bev)
        
        # 这里的 *5.0 是为了防止归一化后数值过小导致梯度消失
        x_l_norm = F.normalize(x_l, dim=1) * 5.0
        x_c_norm = F.normalize(x_c, dim=1) * 5.0

        # === 4. 计算 Gate (Context-Aware) ===
        x_cat = torch.cat([x_l_norm, x_c_norm], dim=1)
        gate_logits = self.gate(x_cat)
        gate = torch.softmax(gate_logits, dim=1) # Softmax 互斥竞争

        w_l = gate[:, 0:1, :, :]
        w_c = gate[:, 1:2, :, :]
        
        self.last_w_lidar = w_l.detach()
        self.last_w_cam = w_c.detach()
        # 保存权重 (Visualize / Debug)
        # batch_dict['acfg_w_lidar'] = w_l
        # batch_dict['acfg_w_cam'] = w_c

        # === 5. 加权融合 (Weighted Fusion) ===
        # 使用【原始】投影特征进行加权，保留强度信息
        fused = w_l * x_l + w_c * x_c

        # === 6. 输出投影 & 残差 ===
        out = self.proj(fused)
        
        if self.use_residual:
            if out.shape == lidar_bev.shape:
                out = out + lidar_bev

        out = self.bn(out)
        out = self.act(out)

        batch_dict["spatial_features"] = out

        # === Debug 信息 ===
        if not self._printed_debug:
            print(f"\n[ACFG Info] Key Fixed. Input Shapes - Lidar: {lidar_bev.shape}, Cam: {cam_bev.shape}")
            print(f"[ACFG Info] Ablation Status - ZeroLidar: {self.zero_lidar_by_env}, ZeroCam: {self.zero_cam_by_env}")
            print(f"[ACFG Info] Gate Mean Weights - Lidar: {w_l.mean():.3f}, Cam: {w_c.mean():.3f}")
            self._printed_debug = True

        return batch_dict