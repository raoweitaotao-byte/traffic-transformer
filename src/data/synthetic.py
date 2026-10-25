"""虚拟路网交通流数据生成器。

模拟一条环形快速路 + 若干放射线上的多检测器交通流/速度数据，
尽量贴近真实交通数据的统计特征（早晚高峰、周末效应、空间相关、突发事故）。
"""
import argparse
import os

import numpy as np
import yaml


def _build_network(n_sensors, rng):
    """构造虚拟路网几何并生成邻接矩阵。

    前 2/3 检测器均匀分布在环形快速路上（单位圆），
    其余分布在 2~3 条放射线（半径 0.5~1.5）上。
    邻接权重用检测器间欧氏距离的高斯核：w_ij = exp(-d_ij^2 / (2*sigma^2))，
    自身权重置 0，最后按行归一化。
    """
    n_ring = int(round(n_sensors * 0.6))
    n_radial = n_sensors - n_ring

    # 环形部分：均匀分布在单位圆上，加轻微半径扰动模拟真实线圈位置
    angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    radii = 1.0 + rng.uniform(-0.05, 0.05, n_ring)
    ring_pos = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)

    # 放射线部分：随机角度、半径 0.4~1.6
    rad_angles = rng.uniform(0, 2 * np.pi, n_radial)
    rad_radii = rng.uniform(0.4, 1.6, n_radial)
    radial_pos = np.stack([rad_radii * np.cos(rad_angles), rad_radii * np.sin(rad_angles)], axis=1)

    positions = np.concatenate([ring_pos, radial_pos], axis=0)  # (N, 2)

    # 高斯核邻接矩阵
    dist = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)  # (N,N)
    sigma = np.median(dist[dist > 0])  # 用中位数距离做核宽，自适应网络密度
    adjacency = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(adjacency, 0.0)
    adjacency = adjacency / adjacency.sum(axis=1, keepdims=True)  # 行归一化

    return adjacency.astype(np.float32), positions


def _gaussian_peak(steps_in_day, center_hour, width_hours, interval_minutes):
    """生成一天内以 center_hour 为中心的高斯峰曲线（幅值为 1）。"""
    t = np.arange(steps_in_day) * interval_minutes / 60.0  # 一天内的小时刻度
    return np.exp(-((t - center_hour) ** 2) / (2 * width_hours ** 2))


def generate_synthetic(n_sensors, n_days, interval_minutes, seed=42):
    """生成虚拟交通数据。

    Returns:
        dict: {
            "flow": (steps, N) float32,   每个 5min 间隔通过的车辆数
            "speed": (steps, N) float32,  km/h
            "adjacency": (N, N) float32,
            "meta": dict,
        }
    """
    rng = np.random.default_rng(seed)
    steps_in_day = (24 * 60) // interval_minutes  # 每天的时间步数，5min -> 288
    total_steps = steps_in_day * n_days

    adjacency, positions = _build_network(n_sensors, rng)

    # ---------- 流量 ----------
    # 每个检测器的随机基础流量（夜间低流量水平，veh/5min）
    base_flow = rng.uniform(15, 45, n_sensors)

    # 早晚高峰：早高峰中心 8:00（宽约 1h），晚高峰中心 18:00（更高更宽，约 1.5h）
    morning = _gaussian_peak(steps_in_day, 8.0, 1.0, interval_minutes)
    evening = _gaussian_peak(steps_in_day, 18.0, 1.5, interval_minutes)

    # 每个检测器的高峰幅度不同（与基础流量正相关，再加随机扰动）
    morning_amp = base_flow * rng.uniform(1.5, 2.5, n_sensors)
    evening_amp = base_flow * rng.uniform(2.5, 4.0, n_sensors)  # 晚高峰更高

    flow = np.zeros((total_steps, n_sensors), dtype=np.float64)
    is_weekend = np.zeros(n_days, dtype=bool)
    for d in range(n_days):
        # 以 30 天为周期近似自然周：第 5、6 天（周六、周日）为周末
        weekend = (d % 7) in (5, 6)
        is_weekend[d] = weekend
        s = slice(d * steps_in_day, (d + 1) * steps_in_day)
        if weekend:
            # 周末：整体流量约降 30%，高峰幅度减半且更宽（峰形平缓）
            day_curve = (0.7 * base_flow
                         + 0.5 * morning_amp * _gaussian_peak(steps_in_day, 9.0, 1.6, interval_minutes)[:, None]
                         + 0.5 * evening_amp * _gaussian_peak(steps_in_day, 17.0, 2.0, interval_minutes)[:, None])
        else:
            day_curve = (base_flow
                         + morning_amp * morning[:, None]
                         + evening_amp * evening[:, None])
        flow[s] = day_curve

    # 空间相关：相邻检测器流量互相影响，用邻接矩阵做一次空间平滑混合
    # flow <- (1-alpha)*flow + alpha * flow @ A.T
    # 其中 (flow @ A.T)[t,i] = sum_j A[i,j] * flow[t,j]，即 i 的邻居加权平均
    alpha = 0.3
    flow = (1 - alpha) * flow + alpha * (flow @ adjacency.T)

    # 突发事故：随机 2~3 次，某检测器某时段流量骤降 40%~60%，并波及邻居
    n_incidents = rng.integers(2, 4)
    incidents = []
    for _ in range(n_incidents):
        sensor = int(rng.integers(0, n_sensors))
        start = int(rng.integers(steps_in_day, total_steps - steps_in_day))
        duration = int(rng.integers(6, 19))  # 30~90 分钟
        drop = float(rng.uniform(0.4, 0.6))
        flow[start:start + duration, sensor] *= (1 - drop)
        # 波及邻居：按邻接权重衰减（拥堵向上游传播）
        neighbor_effect = adjacency[sensor] * drop * 0.5
        flow[start:start + duration] *= (1 - neighbor_effect[None, :])
        incidents.append({"sensor": sensor, "start_step": start,
                          "duration": duration, "drop_ratio": round(drop, 3)})

    # 高斯观测噪声（异方差：噪声水平与流量成正比）
    flow = flow + rng.normal(0, 0.05, flow.shape) * flow
    flow = np.clip(flow, 0, None)

    # ---------- 速度 ----------
    # Greenshields 流量-速度关系：speed = free_flow * (1 - flow / capacity)，并裁剪下限
    free_flow = rng.uniform(50, 70, n_sensors)             # 每检测器自由流速度 km/h
    capacity = flow.max(axis=0) * rng.uniform(1.3, 1.8, n_sensors)  # 通行能力略高于峰值流量
    speed = free_flow[None, :] * (1 - flow / capacity[None, :])
    speed = np.clip(speed, 5.0, None)                      # 拥堵时速度下限 5 km/h
    speed = speed + rng.normal(0, 1.0, speed.shape)        # 小观测噪声
    speed = np.clip(speed, 1.0, None)

    meta = {
        "n_sensors": n_sensors,
        "n_days": n_days,
        "interval_minutes": interval_minutes,
        "steps_in_day": steps_in_day,
        "total_steps": total_steps,
        "seed": seed,
        "is_weekend": is_weekend.tolist(),
        "incidents": incidents,
        "sensor_positions": positions.round(4).tolist(),
    }

    return {
        "flow": flow.astype(np.float32),
        "speed": speed.astype(np.float32),
        "adjacency": adjacency,
        "meta": meta,
    }


def main():
    parser = argparse.ArgumentParser(description="生成虚拟交通流数据并保存为 npy")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg["data"]

    data = generate_synthetic(
        n_sensors=dcfg["n_sensors"],
        n_days=dcfg["n_days"],
        interval_minutes=dcfg["interval_minutes"],
        seed=cfg.get("training", {}).get("seed", 42),
    )

    os.makedirs(dcfg["data_dir"], exist_ok=True)
    flow_path = os.path.join(dcfg["data_dir"], "synthetic_flow.npy")
    speed_path = os.path.join(dcfg["data_dir"], "synthetic_speed.npy")
    adj_path = os.path.join(dcfg["data_dir"], "adjacency.npy")
    np.save(flow_path, data["flow"])
    np.save(speed_path, data["speed"])
    np.save(adj_path, data["adjacency"])

    print(f"flow      : {data['flow'].shape}  -> {flow_path}")
    print(f"speed     : {data['speed'].shape}  -> {speed_path}")
    print(f"adjacency : {data['adjacency'].shape}  -> {adj_path}")
    print(f"事故注入  : {len(data['meta']['incidents'])} 次, {data['meta']['incidents']}")


if __name__ == "__main__":
    main()
