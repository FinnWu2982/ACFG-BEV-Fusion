# pcdet/datasets/processor/lidar_corruption.py
import numpy as np

# ====== 1. 把你原来 data_processor 里的退化函数搬过来 ======
# 比如：
# def lidar_dropout(points, drop_ratio): ...
# def lidar_fog(points, ...): ...
# def lidar_sector_cut(points, ...): ...
# def lidar_smear(points, ...): ...
# 具体实现和参数，直接从
#   pcdet/datasets/processor/data_processor_with_corruption_backup.py
# 里 copy 过来，不要改数值。
def _ensure_min_points(pts: np.ndarray, keep_mask=None, min_pts: int = 50):
    if pts.shape[0] >= min_pts:
        return pts
    if pts.shape[0] == 0:
        return pts
    need = max(0, min_pts - pts.shape[0])
    extra = np.random.choice(pts.shape[0], need, replace=True)
    return np.vstack([pts, pts[extra]]) if need > 0 else pts


def _lidar_dropout(points: np.ndarray, level: str):
    cfg = {
        'light': dict(base_drop=0.20, extra_far=0.10, r_far=40.0),
        'heavy': dict(base_drop=0.30, extra_far=0.20, r_far=40.0),
    }
    if level not in cfg:
        return points
    p = cfg[level]
    xy = np.linalg.norm(points[:, :2], axis=1)
    drop_prob = np.full(points.shape[0], p['base_drop'], dtype=np.float32)
    far_mask = (xy > p['r_far'])
    drop_prob[far_mask] = np.minimum(1.0, p['base_drop'] + p['extra_far'])
    rng = np.random.default_rng()
    keep = rng.random(points.shape[0]) >= drop_prob
    pts = points[keep]
    return _ensure_min_points(pts, keep, 50)


def _lidar_fog(points: np.ndarray, level: str):
    beta = 0.015 if level == 'light' else 0.020
    xy = np.linalg.norm(points[:, :2], axis=1)
    keep_prob = np.exp(-beta * xy)
    rng = np.random.default_rng()
    keep = rng.random(points.shape[0]) < keep_prob
    if points.shape[1] > 3:
        pts_mod = points.copy()
        pts_mod[:, 3] = pts_mod[:, 3] * np.exp(-beta * xy)
    else:
        pts_mod = points
    pts = pts_mod[keep]
    return _ensure_min_points(pts, keep, 50)


def _lidar_sector_drop(points: np.ndarray, level: str):
    n_regions = 1 if level == 'light' else 2
    deg_min, deg_max = (20, 35) if level == 'light' else (30, 45)
    rng = np.random.default_rng()
    ang = np.arctan2(points[:, 1], points[:, 0])  # [-pi, pi]
    drop = np.zeros(points.shape[0], dtype=bool)
    for _ in range(n_regions):
        width = np.deg2rad(rng.integers(deg_min, deg_max + 1))
        center = rng.uniform(-np.pi, np.pi)
        a = (ang - center + np.pi) % (2 * np.pi) - np.pi
        drop |= (np.abs(a) <= width / 2)
    keep = ~drop
    pts = points[keep]
    return _ensure_min_points(pts, keep, 50)


def _lidar_motion_smear(points: np.ndarray, level: str):
    has_ts = points.shape[1] >= 5
    if level == 'light':
        v_std, w_std_deg = 0.3, 0.4
    else:
        v_std, w_std_deg = 0.6, 0.8
    rng = np.random.default_rng()
    pts = points.copy()
    if has_ts:
        ts = pts[:, 4]
        mask = np.abs(ts) > 1e-6
        if not np.any(mask):
            return pts
        vx, vy = rng.normal(0.0, v_std, size=2)
        wz = np.deg2rad(rng.normal(0.0, w_std_deg))
        dt = ts[mask]
        theta = wz * dt
        c, s = np.cos(theta), np.sin(theta)
        x = pts[mask, 0]; y = pts[mask, 1]
        x_rot = c * x - s * y
        y_rot = s * x + c * y
        x_rot += vx * dt; y_rot += vy * dt
        pts[mask, 0] = x_rot; pts[mask, 1] = y_rot
        return pts
    else:
        frac = 0.5 if level == 'light' else 0.8
        m = rng.random(pts.shape[0]) < frac
        dt = (np.random.uniform(0.03, 0.08) if level == 'light'
              else np.random.uniform(0.08, 0.12))
        vx, vy = rng.normal(0.0, v_std, size=2)
        wz = np.deg2rad(rng.normal(0.0, w_std_deg))
        theta = wz * dt
        c, s = np.cos(theta), np.sin(theta)
        x = pts[m, 0]; y = pts[m, 1]
        x_rot = c * x - s * y
        y_rot = s * x + c * y
        x_rot += vx * dt; y_rot += vy * dt
        pts[m, 0] = x_rot; pts[m, 1] = y_rot
        return pts

# ====================================================


def apply_lidar_corruption(points: np.ndarray, mode: str, ops: list):
    """
    points: (N, C) numpy array
    mode: "clean" / "light" / "heavy"
    ops: ["dropout","fog","sector","smear"] 的子集
    """
    if mode == "clean":
        return points

    # 注意：您上面定义的函数包含内置参数逻辑，只需要传入 level (即 mode)
    # 因此这里不需要再定义复杂的 cfg 字典
    
    out = points
    for op in ops:
        if op == "dropout":
            # 调用上面定义的 _lidar_dropout
            out = _lidar_dropout(out, level=mode) 
        elif op == "fog":
            # 调用上面定义的 _lidar_fog
            out = _lidar_fog(out, level=mode)
        elif op == "sector":
            # 调用上面定义的 _lidar_sector_drop
            out = _lidar_sector_drop(out, level=mode)
        elif op == "smear":
            # 调用上面定义的 _lidar_motion_smear
            out = _lidar_motion_smear(out, level=mode)
        else:
            continue

    return out
    # ====================================================

    out = points
    for op in ops:
        if op == "dropout":
            out = lidar_dropout(out, drop_ratio=cfg["dropout_ratio"])
        elif op == "fog":
            out = lidar_fog(out, **cfg["fog"])
        elif op == "sector":
            out = lidar_sector_cut(out, **cfg["sector"])
        elif op == "smear":
            out = lidar_smear(out, **cfg["smear"])
        else:
            continue

    return out
