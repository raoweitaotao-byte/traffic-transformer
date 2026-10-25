# -*- coding: utf-8 -*-
"""
多头自注意力模块。

实现《Attention Is All You Need》(Vaswani et al., 2017) 第 3.2 节的
Scaled Dot-Product Attention 与 Multi-Head Attention:

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V          (式 3.2.1)
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O      (式 3.2.2)
    其中 head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

除以 sqrt(d_k) 是为了防止点积过大导致 softmax 梯度过小。
"""
import math

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """标准多头自注意力。

    流程: QKV 线性投影 -> 分头 -> softmax(QK^T / sqrt(d_k)) -> 加权求和
          -> 拼接各头 -> 输出投影 + dropout

    参数:
        d_model: 输入/输出特征维度
        n_heads: 注意力头数, 必须整除 d_model
        dropout: 输出投影后的 dropout 概率
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # Q / K / V 及输出的线性投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attn=False):
        """
        参数:
            x: (B, L, d_model) 输入序列
            mask: 可选掩码, 可广播到 (B, n_heads, L, L), 0 表示屏蔽
            return_attn: 为 True 时同时返回注意力权重 (B, n_heads, L, L)
        返回:
            out: (B, L, d_model); 若 return_attn=True 则为 (out, attn)
        """
        B, L, _ = x.shape

        # 线性投影并分头: (B, L, d_model) -> (B, n_heads, L, d_k)
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention (论文式 3.2.1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)  # (B, n_heads, L, L)

        # 加权求和 -> 拼接各头 -> 输出投影 + dropout
        out = torch.matmul(attn, v)  # (B, n_heads, L, d_k)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.dropout(self.out_proj(out))

        if return_attn:
            return out, attn
        return out
