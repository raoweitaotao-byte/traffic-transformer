# -*- coding: utf-8 -*-
"""TrafficTransformer 训练脚本。

用法:
    python -m src.train --config configs/default.yaml [--epochs N]

特性:
    - 固定随机种子 (torch / numpy / random)
    - 标准化空间上的 MSELoss + AdamW 优化器
    - 每轮验证集评估 + 早停 (early_stop_patience)
    - 最优验证 loss 权重保存至 results/checkpoints/best_model.pt
      (checkpoint 内含 state_dict / config / target_mean / target_std)
    - tqdm 进度条 + 每轮打印 train/val loss
    - 训练曲线保存至 results/loss_curve.png (matplotlib Agg 后端)
"""
import argparse
import os
import random
import time

import matplotlib

matplotlib.use("Agg")  # 无显示环境下保存图片
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from .data.dataset import build_dataloaders
from .models.transformer import TrafficTransformer, count_parameters


def set_seed(seed):
    """固定随机种子，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(config):
    """按配置构建 TrafficTransformer。"""
    dcfg = config["data"]
    mcfg = config["model"]
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


def run_epoch(model, loader, criterion, device, optimizer=None):
    """跑一个 epoch；optimizer 为 None 时做验证（关闭梯度）。

    返回该 epoch 的平均 MSE loss。
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_count = 0.0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x = x.to(device)  # (B, T, N, F)
            y = y.to(device)  # (B, H, N)
            pred = model(x)
            loss = criterion(pred, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_count += x.size(0)
    return total_loss / max(total_count, 1)


def save_loss_curve(train_losses, val_losses, best_epoch, save_path):
    """绘制并保存 train/val loss 曲线。"""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", ms=3, label="train loss")
    plt.plot(epochs, val_losses, marker="s", ms=3, label="val loss")
    plt.axvline(best_epoch, ls="--", c="gray", alpha=0.6,
                label=f"best epoch {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (normalized)")
    plt.title("TrafficTransformer Training Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="TrafficTransformer 训练")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="配置文件路径")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖配置中的训练轮数（冒烟测试用）")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tcfg = config["training"]
    set_seed(tcfg.get("seed", 42))

    device = torch.device(tcfg.get("device", "cpu"))
    checkpoint_dir = tcfg.get("checkpoint_dir", "results/checkpoints")
    results_dir = os.path.dirname(checkpoint_dir) or "results"
    os.makedirs(checkpoint_dir, exist_ok=True)

    loaders = build_dataloaders(config)
    model = build_model(config).to(device)
    print(f"设备: {device} | 可训练参数: {count_parameters(model):,}")
    print(f"train/val/test batches: {len(loaders['train'])}/"
          f"{len(loaders['val'])}/{len(loaders['test'])}")

    criterion = nn.MSELoss()  # 标准化空间上的 MSE
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )

    epochs = args.epochs if args.epochs is not None else tcfg["epochs"]
    patience = tcfg.get("early_stop_patience", 6)

    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    train_losses, val_losses = [], []
    ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")

    start = time.time()
    for epoch in tqdm(range(1, epochs + 1), desc="训练进度", ncols=90):
        train_loss = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss = run_epoch(model, loaders["val"], criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val, best_epoch = val_loss, epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "target_mean": loaders["target_mean"],
                    "target_std": loaders["target_std"],
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                ckpt_path,
            )
        else:
            bad_epochs += 1

        tqdm.write(f"Epoch {epoch:03d}/{epochs} | train loss: {train_loss:.4f} | "
                   f"val loss: {val_loss:.4f}" + ("  *best" if improved else ""))

        if bad_epochs >= patience:
            print(f"[早停] 验证集 loss 连续 {patience} 轮未改善，于 epoch {epoch} 停止训练。")
            break

    elapsed = time.time() - start
    print(f"训练结束 | 最佳验证 loss: {best_val:.4f} (epoch {best_epoch}) | "
          f"总耗时: {elapsed:.1f}s")
    print(f"最优权重已保存: {ckpt_path}")

    curve_path = os.path.join(results_dir, "loss_curve.png")
    save_loss_curve(train_losses, val_losses, best_epoch, curve_path)
    print(f"训练曲线已保存: {curve_path}")


if __name__ == "__main__":
    main()
