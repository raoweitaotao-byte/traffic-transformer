# -*- coding: utf-8 -*-
"""TrafficTransformer 可视化脚本。

生成四张图到 results/:
    1. prediction_curve.png          单窗口: 历史输入 + 未来真实 vs 预测
    2. multi_horizon_curve.png       长时间跨度: 滚动预测 vs 真实
    3. temporal_attention_heatmap.png 时间注意力热力图
    4. spatial_attention_heatmap.png  空间注意力 vs 路网邻接矩阵

用法:
    python -m src.visualize --config configs/default.yaml \
        --checkpoint results/checkpoints/best_model.pt

若 checkpoint 不存在，回退到随机初始化模型，并在图标题注明 "(random weights)"。
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")  # 无显示环境

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from .data.dataset import build_dataloaders
from .models.transformer import TrafficTransformer

RESULTS_DIR = "results"


def to_np(t):
    """torch tensor -> numpy。

    torch 2.0.x 与 numpy 2.x 不兼容时 tensor.numpy() 会抛 RuntimeError，
    回退走 tolist 路径。
    """
    t = t.detach().cpu()
    try:
        return t.numpy()
    except RuntimeError:
        return np.asarray(t.tolist(), dtype=np.float32)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(config):
    mcfg = config["model"]
    dcfg = config["data"]
    return TrafficTransformer(
        n_features=mcfg["n_features"],
        n_sensors=dcfg["n_sensors"],
        input_window=dcfg["input_window"],
        pred_horizon=dcfg["pred_horizon"],
        d_model=mcfg["d_model"],
        n_heads=mcfg["n_heads"],
        n_layers=mcfg["n_layers"],
        d_ff=mcfg["d_ff"],
        dropout=mcfg["dropout"],
    )


def maybe_load_checkpoint(model, ckpt_path, device):
    """加载 checkpoint；返回 (model, loaded, target_mean, target_std 或 None)。"""
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        mean = ckpt.get("target_mean")
        std = ckpt.get("target_std")
        print(f"[visualize] 已加载 checkpoint: {ckpt_path}")
        return model, True, mean, std
    print(f"[visualize] checkpoint 不存在 ({ckpt_path})，使用随机初始化权重")
    return model, False, None, None


def denorm(arr, mean, std):
    """反标准化回原始流量量纲。"""
    return arr * std + mean


def pick_representative_sensor(x_norm):
    """挑选历史流量方差最大的检测器作为代表性检测器。

    x_norm: (B, T, N, F) tensor，取 batch 中第 0 个样本。
    """
    flow = to_np(x_norm[0, :, :, 0])  # (T, N)
    return int(flow.std(axis=0).argmax())


def plot_prediction_curve(history, future_true, future_pred, sensor, tag, out_path):
    """单窗口预测曲线: 历史 T 步 + 未来 H 步 真实 vs 预测。"""
    T = len(history)
    H = len(future_true)
    t_hist = np.arange(-T, 0)
    t_fut = np.arange(0, H)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t_hist, history, color="#4878CF", marker="o", ms=4,
            label="History (input)")
    ax.plot(t_fut, future_true, color="#6ACC65", marker="o", ms=4,
            label="Ground truth")
    ax.plot(t_fut, future_pred, color="#D65F5F", marker="s", ms=4, ls="--",
            label="Prediction")
    # 用连接线把历史末尾与未来起点衔接起来
    ax.plot([-1, 0], [history[-1], future_true[0]], color="#6ACC65", lw=1)
    ax.axvline(x=-0.5, color="gray", ls=":", lw=1.2)
    ax.set_xlabel("Time (5-min steps)")
    ax.set_ylabel("Flow (vehicles per 5 min)")
    ax.set_title(f"Traffic Flow Prediction - Sensor {sensor} {tag}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_multi_horizon(model, test_loader, sensor, mean, std, horizon,
                       total_steps, tag, out_path, device):
    """长时间跨度滚动预测: 每隔 H 步取一个预测起点，拼接各窗口预测。"""
    n_windows = int(np.ceil(total_steps / horizon))
    start_idx = [i * horizon for i in range(n_windows)]
    need = start_idx[-1] + 1

    preds, trues = [], []
    n_seen = 0
    for x, y in test_loader:
        x = x.to(device)
        y_hat = model(x)
        preds.append(to_np(y_hat))
        trues.append(to_np(y))
        n_seen += x.shape[0]
        if n_seen >= need:
            break
    preds = np.concatenate(preds, axis=0)  # (n_samples, H, N)
    trues = np.concatenate(trues, axis=0)

    pred_curve, true_curve = [], []
    for s in start_idx:
        pred_curve.append(denorm(preds[s][:, sensor], mean, std))
        true_curve.append(denorm(trues[s][:, sensor], mean, std))
    pred_curve = np.concatenate(pred_curve)[:total_steps]
    true_curve = np.concatenate(true_curve)[:total_steps]
    t = np.arange(len(true_curve))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(t, true_curve, color="#6ACC65", lw=1.8, label="Ground truth")
    ax.plot(t, pred_curve, color="#D65F5F", lw=1.4, ls="--", label="Rolling prediction")
    for i in range(1, n_windows):
        ax.axvline(x=i * horizon - 0.5, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("Time (5-min steps)")
    ax.set_ylabel("Flow (vehicles per 5 min)")
    ax.set_title(f"Multi-Horizon Rolling Prediction ({len(t)} steps, ~"
                 f"{len(t) * 5 / 60:.0f} hours) - Sensor {sensor} {tag}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_temporal_attention(attn, tag, out_path):
    """时间注意力热力图 (T, T)。"""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attn, cmap="viridis", aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Key time step")
    ax.set_ylabel("Query time step")
    ax.set_title(f"Temporal Attention (last layer, head-averaged) {tag}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_spatial_attention(attn, adjacency, tag, out_path):
    """空间注意力热力图；如有邻接矩阵则并排对比。"""
    if adjacency is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        im0 = axes[0].imshow(attn, cmap="viridis", aspect="equal")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        axes[0].set_title(f"Learned Spatial Attention {tag}")
        im1 = axes[1].imshow(adjacency, cmap="viridis", aspect="equal")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].set_title("Road Network Adjacency")
        for ax in axes:
            ax.set_xlabel("Sensor index")
            ax.set_ylabel("Sensor index")
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(attn, cmap="viridis", aspect="equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("Sensor index")
        ax.set_ylabel("Sensor index")
        ax.set_title(f"Learned Spatial Attention {tag}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="TrafficTransformer 可视化")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="results/checkpoints/best_model.pt")
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    parser.add_argument("--total-steps", type=int, default=96,
                        help="multi_horizon 图覆盖的连续时间步数")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("training", {}).get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(config.get("training", {}).get("device", "cpu"))

    loaders = build_dataloaders(config)
    model = build_model(config).to(device)
    model, loaded, ckpt_mean, ckpt_std = maybe_load_checkpoint(
        model, args.checkpoint, device)
    model.eval()

    # 反标准化统计量: 优先 checkpoint，否则用数据模块基于训练段的统计量
    mean = ckpt_mean if ckpt_mean is not None else loaders["target_mean"]
    std = ckpt_std if ckpt_std is not None else loaders["target_std"]
    tag = "" if loaded else "(random weights)"

    os.makedirs(args.out_dir, exist_ok=True)
    saved = []

    # ---- 取一个测试 batch，前向 + 注意力 ----
    x, y = next(iter(loaders["test"]))
    with torch.no_grad():
        y_hat, attn = model(x.to(device), return_attn=True)
    y_hat = to_np(y_hat)
    y_np = to_np(y)
    x_np = to_np(x)

    sensor = pick_representative_sensor(x)
    print(f"[visualize] 代表性检测器: Sensor {sensor}")

    # ---- 图1: 单窗口预测曲线 ----
    p1 = os.path.join(args.out_dir, "prediction_curve.png")
    history = denorm(x_np[0, :, sensor, 0], mean, std)
    future_true = denorm(y_np[0, :, sensor], mean, std)
    future_pred = denorm(y_hat[0, :, sensor], mean, std)
    plot_prediction_curve(history, future_true, future_pred, sensor, tag, p1)
    saved.append(p1)

    # ---- 图2: 多窗口滚动预测 ----
    p2 = os.path.join(args.out_dir, "multi_horizon_curve.png")
    horizon = config["data"]["pred_horizon"]
    plot_multi_horizon(model, loaders["test"], sensor, mean, std, horizon,
                       args.total_steps, tag, p2, device)
    saved.append(p2)

    # ---- 图3: 时间注意力热力图 ----
    p3 = os.path.join(args.out_dir, "temporal_attention_heatmap.png")
    plot_temporal_attention(to_np(attn["temporal"][0]), tag, p3)
    saved.append(p3)

    # ---- 图4: 空间注意力 vs 邻接矩阵 ----
    p4 = os.path.join(args.out_dir, "spatial_attention_heatmap.png")
    adjacency = loaders.get("adjacency")
    plot_spatial_attention(to_np(attn["spatial"][0]), adjacency, tag, p4)
    saved.append(p4)

    print("\n生成的图片:")
    for p in saved:
        print(f"  - {os.path.abspath(p)}")


if __name__ == "__main__":
    main()
