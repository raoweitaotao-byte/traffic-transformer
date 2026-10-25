# -*- coding: utf-8 -*-
"""TrafficTransformer 测试集评估脚本。

用法:
    python -m src.evaluate --config configs/default.yaml \
        --checkpoint results/checkpoints/best_model.pt

功能:
    - 加载训练保存的 checkpoint (含 config 与 target mean/std)
    - 在测试集上预测，反标准化回原始流量量纲后计算 MAE / RMSE / MAPE
      (MAPE 对 |真值| < 5.0 的点做掩膜，避免除零爆炸)
    - 与持续法基线（输入窗口最后一步作为未来全部 H 步预测）对比
    - 指标打印（保留 2 位小数）并追加写入 results/metrics.json

注意: 本环境 torch 2.0.1 与 numpy 2.x 的 tensor.numpy() 不兼容，
    因此全程用 torch 张量计算指标，仅在最终转 Python float。
"""
import argparse
import json
import os
from datetime import datetime

import torch
import yaml

from .data.dataset import build_dataloaders
from .train import build_model

MAPE_THRESHOLD = 5.0  # |真值| 低于该阈值不参与 MAPE 计算


def compute_metrics(pred, truth):
    """pred/truth: 原始量纲的 torch 张量 (n, H, N)。返回 MAE/RMSE/MAPE dict。"""
    mae = torch.mean(torch.abs(pred - truth)).item()
    rmse = torch.sqrt(torch.mean((pred - truth) ** 2)).item()
    mask = torch.abs(truth) >= MAPE_THRESHOLD
    mape = torch.mean(
        torch.abs((pred[mask] - truth[mask]) / truth[mask])
    ).item() * 100.0
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


@torch.no_grad()
def predict(model, loader, device):
    """在测试集上推理。

    返回 (pred, truth, last_input) 三个标准化空间上的 torch 张量，
    last_input 为输入窗口最后一步的流量通道 (n, N)，用于持续法基线。
    """
    model.eval()
    preds, trues, lasts = [], [], []
    for x, y in loader:
        x = x.to(device)
        preds.append(model(x).cpu())          # (B, H, N)
        trues.append(y)                        # (B, H, N)
        lasts.append(x[:, -1, :, 0].cpu())     # (B, N) 流量通道
    return torch.cat(preds), torch.cat(trues), torch.cat(lasts)


def append_metrics(metrics_path, record):
    """把一次评估结果追加写入 JSON 文件（list 结构）。"""
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        if not isinstance(history, list):
            history = [history]
    else:
        history = []
    history.append(record)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="TrafficTransformer 测试集评估")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="配置文件路径")
    parser.add_argument("--checkpoint", default="results/checkpoints/best_model.pt",
                        help="训练保存的 checkpoint 路径")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", config)  # 优先使用 checkpoint 内保存的配置
    target_mean = float(ckpt["target_mean"])
    target_std = float(ckpt["target_std"])

    device = torch.device(cfg["training"].get("device", "cpu"))
    loaders = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"已加载 checkpoint: {args.checkpoint} "
          f"(epoch {ckpt.get('epoch', '?')}, val loss {ckpt.get('val_loss', float('nan')):.4f})")

    pred_n, true_n, last_n = predict(model, loaders["test"], device)

    # 反标准化回原始流量量纲
    pred = pred_n * target_std + target_mean
    truth = true_n * target_std + target_mean

    # 持续法基线: 输入窗口最后一步 -> 未来全部 H 步
    H = cfg["data"]["pred_horizon"]
    baseline = last_n.unsqueeze(1).expand(-1, H, -1) * target_std + target_mean

    model_metrics = compute_metrics(pred, truth)
    base_metrics = compute_metrics(baseline, truth)

    print("=" * 58)
    print(f"测试集样本数: {pred.shape[0]}  (H={H}, N={pred.shape[2]})")
    print(f"{'method':<26}{'MAE':>10}{'RMSE':>10}{'MAPE(%)':>10}")
    print("-" * 58)
    print(f"{'TrafficTransformer':<26}{model_metrics['MAE']:>10.2f}"
          f"{model_metrics['RMSE']:>10.2f}{model_metrics['MAPE']:>10.2f}")
    print(f"{'persistence baseline':<26}{base_metrics['MAE']:>10.2f}"
          f"{base_metrics['RMSE']:>10.2f}{base_metrics['MAPE']:>10.2f}")
    print("=" * 58)

    results_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
    metrics_path = os.path.join(results_dir, "metrics.json")
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": cfg["data"]["dataset"],
        "checkpoint": args.checkpoint,
        "n_test_samples": int(pred.shape[0]),
        "model": {k: round(v, 4) for k, v in model_metrics.items()},
        "baseline_persistence": {k: round(v, 4) for k, v in base_metrics.items()},
    }
    append_metrics(metrics_path, record)
    print(f"指标已追加写入: {metrics_path}")


if __name__ == "__main__":
    main()
