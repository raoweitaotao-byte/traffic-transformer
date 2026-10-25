"""数据集与 DataLoader 构建。

滑动窗口样本：x:(T, N, F)，y:(H, N)（预测目标默认为流量通道）。
按时间顺序划分 train/val/test，仅用训练段统计量做 z-score 标准化。
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import synthetic

_REAL_DATA_HELP = (
    "真实数据文件不存在: {path}\n"
    "请下载并处理为 npy（形状 (steps, N) 或 (steps, N, F)）：\n"
    "  - METR-LA: https://github.com/liyaguang/DCRNN （下载 metr-la.h5，"
    "取 df 值转成 (steps, 207) 的速度矩阵后保存为 npy）\n"
    "  - PeMS04 : 加州 PeMS 官方导出 / DCRNN 风格预处理脚本，"
    "整理为 (steps, 307) 后保存为 npy\n"
    "然后在配置文件中设置 data.real_data_path 指向该文件。"
)


def _to_float_tensor(array):
    """numpy -> float32 tensor。

    优先走零拷贝的 torch.from_numpy；当 torch 与 numpy 大版本不兼容
    （如 torch 2.0.x 配 numpy 2.x）时回退到 frombuffer 路径。
    """
    array = np.ascontiguousarray(array, dtype=np.float32)
    try:
        return torch.from_numpy(array).float()
    except RuntimeError:
        # bytearray 可写，避免 frombuffer 的 non-writable 警告
        buf = bytearray(array.tobytes())
        return torch.frombuffer(buf, dtype=torch.float32).reshape(array.shape)


class TrafficDataset(Dataset):
    """滑动窗口交通数据集。

    Args:
        array: (steps, N, F) numpy 数组
        input_window: 历史窗口长度 T
        pred_horizon: 预测步长 H
        target_feature_idx: 预测目标所在特征通道（默认 0 = 流量）
    """

    def __init__(self, array, input_window, pred_horizon, target_feature_idx=0):
        self.data = _to_float_tensor(array)
        self.T = input_window
        self.H = pred_horizon
        self.target_feature_idx = target_feature_idx
        self.n_samples = self.data.shape[0] - self.T - self.H + 1

    def __len__(self):
        return max(self.n_samples, 0)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.T]                        # (T, N, F)
        y = self.data[idx + self.T: idx + self.T + self.H,
                      :, self.target_feature_idx]               # (H, N)
        return x, y


def load_array(config):
    """按配置加载 (steps, N, F) 的特征数组。

    synthetic 数据集读取/自动生成本地 npy；真实数据集从 real_data_path 加载。
    """
    dcfg = config["data"]
    dataset = dcfg["dataset"]
    data_dir = dcfg["data_dir"]

    if dataset == "synthetic":
        flow_path = os.path.join(data_dir, "synthetic_flow.npy")
        speed_path = os.path.join(data_dir, "synthetic_speed.npy")
        if not (os.path.exists(flow_path) and os.path.exists(speed_path)):
            # 文件不存在则自动生成并保存
            print(f"[load_array] 未找到合成数据，正在生成: {data_dir}")
            data = synthetic.generate_synthetic(
                n_sensors=dcfg["n_sensors"],
                n_days=dcfg["n_days"],
                interval_minutes=dcfg["interval_minutes"],
                seed=config.get("training", {}).get("seed", 42),
            )
            os.makedirs(data_dir, exist_ok=True)
            np.save(flow_path, data["flow"])
            np.save(speed_path, data["speed"])
            np.save(os.path.join(data_dir, "adjacency.npy"), data["adjacency"])
        flow = np.load(flow_path)      # (steps, N)
        speed = np.load(speed_path)    # (steps, N)
        return np.stack([flow, speed], axis=-1)  # (steps, N, 2)

    if dataset in ("metr-la", "pems04"):
        path = dcfg.get("real_data_path") or ""
        if not path or not os.path.exists(path):
            raise FileNotFoundError(_REAL_DATA_HELP.format(path=path or "<未配置>"))
        arr = np.load(path)
        if arr.ndim == 2:
            # 单通道（仅速度或仅流量）：复制成 2 通道以匹配模型输入
            print(f"[load_array] {path} 为单通道 {arr.shape}，已复制为 2 通道")
            arr = np.stack([arr, arr], axis=-1)
        elif arr.ndim != 3:
            raise ValueError(f"真实数据形状应为 (steps,N) 或 (steps,N,F)，实际: {arr.shape}")
        return arr.astype(np.float32)

    raise ValueError(f"未知数据集: {dataset}（可选: synthetic | metr-la | pems04）")


def build_dataloaders(config):
    """构建 train/val/test DataLoader。

    按时间顺序划分（不打乱时序），仅用训练段均值/方差做 z-score 标准化。
    返回 dict，含各 DataLoader、target 反标准化统计量与邻接矩阵。
    """
    dcfg = config["data"]
    array = load_array(config)  # (steps, N, F)
    steps = array.shape[0]
    T, H = dcfg["input_window"], dcfg["pred_horizon"]

    # 时序划分：前 70% 训练，接下来 10% 验证，最后 20% 测试
    n_train = int(steps * dcfg["train_ratio"])
    n_val = int(steps * dcfg["val_ratio"])

    # 仅用训练段计算标准化统计量，避免信息泄露
    train_seg = array[:n_train]
    mean = train_seg.mean(axis=(0, 1), keepdims=True)
    std = train_seg.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    array_norm = ((array - mean) / std).astype(np.float32)

    # 窗口可跨界：val 从 (n_train - T - H) 起截取，保证 val 首样本可见完整历史
    val_start = max(n_train - T - H + 1, 0)
    test_start = max(n_train + n_val - T - H + 1, 0)

    ds_train = TrafficDataset(array_norm[:n_train], T, H)
    ds_val = TrafficDataset(array_norm[val_start: n_train + n_val], T, H)
    ds_test = TrafficDataset(array_norm[test_start:], T, H)

    batch_size = dcfg["batch_size"]
    loaders = {
        "train": DataLoader(ds_train, batch_size=batch_size, shuffle=True),
        "val": DataLoader(ds_val, batch_size=batch_size, shuffle=False),
        "test": DataLoader(ds_test, batch_size=batch_size, shuffle=False),
        # 预测目标（流量通道）的反标准化统计量
        "target_mean": float(mean[0, 0, 0]),
        "target_std": float(std[0, 0, 0]),
        "adjacency": None,
    }

    adj_path = os.path.join(dcfg["data_dir"], "adjacency.npy")
    if os.path.exists(adj_path):
        loaders["adjacency"] = np.load(adj_path)

    return loaders


if __name__ == "__main__":
    import yaml

    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    loaders = build_dataloaders(config)
    for split in ("train", "val", "test"):
        print(f"{split}: {len(loaders[split])} batches")
    x, y = next(iter(loaders["train"]))
    print(f"batch x: {tuple(x.shape)}, y: {tuple(y.shape)}")
    print(f"target_mean={loaders['target_mean']:.2f}, target_std={loaders['target_std']:.2f}, "
          f"adjacency={None if loaders['adjacency'] is None else loaders['adjacency'].shape}")
