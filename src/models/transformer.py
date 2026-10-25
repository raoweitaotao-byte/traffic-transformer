# -*- coding: utf-8 -*-
"""
TrafficTransformer: 基于时空 Transformer 的交通流预测模型。

输入: x (B, T, N, F)  —— 过去 T 步、N 个检测器、F 个特征(flow, speed)
输出: y (B, H, N)     —— 未来 H 步各检测器的流量预测

结构:
    1. 输入嵌入: F -> d_model 线性投影
    2. 时间位置编码 (论文第 3.5 节正弦编码)
    3. n_layers 个 (TemporalBlock, SpatialBlock) 交替堆叠:
       时间注意力按检测器分批 (B*N, T, d), 空间注意力按时间步分批 (B*T, N, d)
    4. 预测头: 每个检测器 flatten T 步表示, 经 MLP 输出 H 维
"""
import math

import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention


class PositionalEncoding(nn.Module):
    """《Attention Is All You Need》第 3.5 节的正弦位置编码:

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    不同频率的正余弦波使模型能感知序列中各时间步的相对位置。
    """

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 注册为 buffer, 随模型移动设备但不参与训练
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, L, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalBlock(nn.Module):
    """时间维 Transformer 块: pre-LN 的多头自注意力 + 前馈网络。

    pre-LN 结构 (先 LayerNorm 再进子层, 残差直连):
        x = x + MHA(LN(x));  x = x + FFN(LN(x))
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attn=False):
        """x: (B, L, d_model); return_attn=True 时额外返回注意力权重"""
        if return_attn:
            attn_out, attn = self.attn(self.norm1(x), return_attn=True)
        else:
            attn_out, attn = self.attn(self.norm1(x)), None
        x = x + attn_out  # 残差连接 1
        x = x + self.ffn(self.norm2(x))  # 残差连接 2
        if return_attn:
            return x, attn
        return x


class SpatialBlock(nn.Module):
    """空间(检测器)维 Transformer 块: pre-LN 的多头自注意力 + 前馈网络。

    结构与 TemporalBlock 相同, 但作用于检测器维, 用于捕捉检测器之间
    (路网空间) 的相关性。
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attn=False):
        """x: (B, L, d_model); return_attn=True 时额外返回注意力权重"""
        if return_attn:
            attn_out, attn = self.attn(self.norm1(x), return_attn=True)
        else:
            attn_out, attn = self.attn(self.norm1(x)), None
        x = x + attn_out  # 残差连接 1
        x = x + self.ffn(self.norm2(x))  # 残差连接 2
        if return_attn:
            return x, attn
        return x


class TrafficTransformer(nn.Module):
    """时空 Transformer 交通流预测模型。

    参数:
        n_features:   输入特征数 F (flow, speed) -> 2
        n_sensors:    检测器数 N -> 20
        input_window: 历史时间窗 T -> 12
        pred_horizon: 预测步长 H -> 12
        d_model:      嵌入维度 -> 64
        n_heads:      注意力头数 -> 4
        n_layers:     时空块层数 -> 2
        d_ff:         前馈网络隐层维度 -> 128
        dropout:      dropout 概率 -> 0.1
    """

    def __init__(
        self,
        n_features=2,
        n_sensors=20,
        input_window=12,
        pred_horizon=12,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.input_window = input_window
        self.pred_horizon = pred_horizon
        self.d_model = d_model
        self.n_layers = n_layers

        # 1. 输入嵌入: F -> d_model
        self.input_proj = nn.Linear(n_features, d_model)
        # 2. 时间位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len=input_window, dropout=dropout)

        # 3. 交替的时间 / 空间 Transformer 块
        self.temporal_blocks = nn.ModuleList(
            [TemporalBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.spatial_blocks = nn.ModuleList(
            [SpatialBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # 4. 预测头: 每个检测器 flatten T 步表示 -> MLP -> H 维输出
        self.head_norm = nn.LayerNorm(d_model)
        self.pred_head = nn.Sequential(
            nn.Linear(input_window * d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, pred_horizon),
        )

    def forward(self, x, return_attn=False):
        """
        参数:
            x: (B, T, N, F) float32
            return_attn: 为 True 时返回 (y, attn_dict),
                         attn_dict = {"temporal": (B, T, T), "spatial": (B, N, N)}
        返回:
            y: (B, H, N) float32
        """
        B, T, N, F = x.shape

        # 输入嵌入: (B, T, N, F) -> (B, T, N, d_model)
        h = self.input_proj(x)

        # 加时间位置编码: 对每个检测器的时间序列施加 -> (B*N, T, d)
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, self.d_model)
        h = self.pos_encoding(h)
        h = h.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3)  # (B, T, N, d)

        last_temporal_attn = None
        last_spatial_attn = None

        for i in range(self.n_layers):
            # 时间注意力: 按检测器分批 (B*N, T, d)
            h_t = h.permute(0, 2, 1, 3).reshape(B * N, T, self.d_model)
            if return_attn and i == self.n_layers - 1:
                h_t, last_temporal_attn = self.temporal_blocks[i](h_t, return_attn=True)
            else:
                h_t = self.temporal_blocks[i](h_t)
            h = h_t.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3)

            # 空间注意力: 按时间步分批 (B*T, N, d)
            h_s = h.reshape(B * T, N, self.d_model)
            if return_attn and i == self.n_layers - 1:
                h_s, last_spatial_attn = self.spatial_blocks[i](h_s, return_attn=True)
            else:
                h_s = self.spatial_blocks[i](h_s)
            h = h_s.reshape(B, T, N, self.d_model)

        # 预测头: (B, T, N, d) -> (B, N, T*d) -> (B, N, H) -> (B, H, N)
        h = self.head_norm(h)
        h = h.permute(0, 2, 1, 3).reshape(B, N, T * self.d_model)
        y = self.pred_head(h)
        y = y.permute(0, 2, 1)

        if return_attn:
            # 时间注意力 (B*N, n_heads, T, T) -> 按检测器/头取均值 -> (B, T, T)
            t_attn = last_temporal_attn.reshape(B, N, -1, T, T).mean(dim=(1, 2))
            # 空间注意力 (B*T, n_heads, N, N) -> 按时间步/头取均值 -> (B, N, N)
            s_attn = last_spatial_attn.reshape(B, T, -1, N, N).mean(dim=(1, 2))
            attn_dict = {"temporal": t_attn, "spatial": s_attn}
            return y, attn_dict
        return y


def count_parameters(model):
    """统计模型可训练参数总量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(42)

    # 默认超参构造模型
    model = TrafficTransformer()
    x = torch.randn(2, 12, 20, 2)

    # 测试 return_attn=False
    y = model(x)
    print(f"[return_attn=False] 输出 y 形状: {tuple(y.shape)}")
    assert y.shape == (2, 12, 20), f"输出形状错误: {tuple(y.shape)}"

    # 测试 return_attn=True
    y, attn_dict = model(x, return_attn=True)
    print(f"[return_attn=True]  输出 y 形状: {tuple(y.shape)}")
    print(f"  temporal 注意力形状: {tuple(attn_dict['temporal'].shape)}")
    print(f"  spatial 注意力形状: {tuple(attn_dict['spatial'].shape)}")
    assert y.shape == (2, 12, 20), f"输出形状错误: {tuple(y.shape)}"
    assert attn_dict["temporal"].shape == (2, 12, 12), "temporal 注意力形状错误"
    assert attn_dict["spatial"].shape == (2, 20, 20), "spatial 注意力形状错误"

    # 参数总量
    n_params = count_parameters(model)
    print(f"可训练参数总量: {n_params:,}")

    # 梯度反传冒烟测试
    model.zero_grad()
    y = model(x)
    loss = y.mean()
    loss.backward()
    grad = model.input_proj.weight.grad
    assert grad is not None, "输入嵌入层权重梯度为 None"
    assert torch.isfinite(grad).all(), "梯度包含 NaN/Inf"
    print(f"梯度反传正常, input_proj.weight.grad 形状: {tuple(grad.shape)}")
    print("全部冒烟测试通过 ✓")
