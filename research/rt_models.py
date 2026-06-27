# -*- coding: utf-8 -*-
"""关系张量 v2 编码器网络 (RT-003 核心) — per-anchor CNN + 沿 N 轴 TCN + 融合 head.

设计见 research/DESIGN_relation_tensor_v2.md §2。三段:

  1. AnchorEncoder — per-anchor 小 CNN: 价格关系 5×4×4×2 (深度 d × OCHL_i × OCHL_j × {soft,hard})
     重排成 (10 通道 × 4×4 空间) 过 Conv2d 给"跨阶×跨价位"局部性归纳偏置,
     再拼 量(5)+形态(5) 标量 → 锚点 embedding (ANCHOR_EMB=32)。
  2. SeqEncoder — 沿 N=40 锚点时间轴的 TCN (dilated Conv1d), 因果方向已由数据正序保证 →
     序列 embedding (SEQ_EMB=64)。
  3. FusionHead — 序列 embedding ⊕ A0 标量因子 → MLP → 标量分数 (回归 r20 / 启动子)。

两条 head (设计 §2, 两条都跑对比):
  (a) NN end-to-end: AnchorEncoder→SeqEncoder→FusionHead 整体训练。
  (b) GBDT-on-embedding: 训好的 SeqEncoder 冻结, 抽序列 embedding ⊕ A0 喂 LightGBM (更稳/便宜, 先跑)。

本模块只定义网络与张量重排, 不含训练/IO (在 rt003_train.py)。纯前向, 无未来信息 (因果由特征保证)。
"""
from __future__ import annotations
import torch
import torch.nn as nn

# ── 维度 (设计 §2 锁定) ──
N_DEPTH = 5            # 深度 d=1..5
N_PRICE = 4            # OCHL
N_CHAN = 2             # soft / hard 双通道
N_PRICE_FLAT = N_DEPTH * N_PRICE * N_PRICE * N_CHAN   # 160
N_VOL = 5
N_SHAPE = 5
N_ANCHOR_FEAT = N_PRICE_FLAT + N_VOL + N_SHAPE        # 170
CONV_CHAN = N_DEPTH * N_CHAN                          # 10 (d×通道) 作 Conv2d 输入通道

ANCHOR_EMB = 32
SEQ_EMB = 64
N_SEQ = 40             # N 锚点


def split_anchor(x: torch.Tensor):
    """(..., 170) → (price (...,10,4,4), vol (...,5), shape (...,5)).

    price 前 160 列序为 feature_columns(): d 外, OCHL_i, OCHL_j, 内层 (soft,hard) →
    reshape (5,4,4,2)=[d,i,j,c]; permute 成 (d,c,i,j) 再并 (d,c)=10 通道, (i,j)=4×4 空间。
    """
    price = x[..., :N_PRICE_FLAT]
    vol = x[..., N_PRICE_FLAT:N_PRICE_FLAT + N_VOL]
    shape = x[..., N_PRICE_FLAT + N_VOL:]
    lead = price.shape[:-1]
    price = price.reshape(*lead, N_DEPTH, N_PRICE, N_PRICE, N_CHAN)
    # [d,i,j,c] → [d,c,i,j] → [(d c), i, j]
    price = price.permute(*range(len(lead)), len(lead), len(lead) + 3,
                          len(lead) + 1, len(lead) + 2).contiguous()
    price = price.reshape(*lead, CONV_CHAN, N_PRICE, N_PRICE)
    return price, vol, shape


class AnchorEncoder(nn.Module):
    """per-anchor: 价格 (10,4,4) CNN + 量(5)+形态(5) → 锚点 embedding (32)。"""

    def __init__(self, emb: int = ANCHOR_EMB):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(CONV_CHAN, 24, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=2), nn.ReLU(),         # 4×4 → 3×3
            nn.AdaptiveAvgPool2d(1),                              # → (24,1,1)
        )
        self.head = nn.Sequential(
            nn.Linear(24 + N_VOL + N_SHAPE, emb), nn.ReLU(),
            nn.Linear(emb, emb),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, 170) → (B, N, ANCHOR_EMB)。"""
        B, N, _ = x.shape
        price, vol, shape = split_anchor(x)
        price = price.reshape(B * N, CONV_CHAN, N_PRICE, N_PRICE)
        cv = self.conv(price).reshape(B * N, 24)
        scal = torch.cat([vol, shape], dim=-1).reshape(B * N, N_VOL + N_SHAPE)
        emb = self.head(torch.cat([cv, scal], dim=-1))
        return emb.reshape(B, N, -1)


class _TCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, dilation: int, k: int = 3):
        super().__init__()
        pad = (k - 1) * dilation                       # 因果 padding (左侧), 见 forward 裁剪
        self.conv = nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation)
        self.pad = pad
        self.relu = nn.ReLU()
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x):
        out = self.conv(x)
        if self.pad:
            out = out[..., :-self.pad]                 # 裁右侧 → 因果 (只看过去锚点)
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)


class SeqEncoder(nn.Module):
    """沿 N=40 锚点轴的因果 TCN → 序列 embedding (64)。取末锚点 (最新) 表征。"""

    def __init__(self, c_in: int = ANCHOR_EMB, emb: int = SEQ_EMB):
        super().__init__()
        self.net = nn.Sequential(
            _TCNBlock(c_in, emb, dilation=1),
            _TCNBlock(emb, emb, dilation=2),
            _TCNBlock(emb, emb, dilation=4),
        )

    def forward(self, anchor_emb: torch.Tensor) -> torch.Tensor:
        """anchor_emb: (B, N, ANCHOR_EMB) → (B, SEQ_EMB)。"""
        x = anchor_emb.transpose(1, 2)                 # (B, C, N)
        h = self.net(x)
        return h[..., -1]                              # 末锚点 (因果, 最新决策时刻)


class FusionModel(nn.Module):
    """AnchorEncoder → SeqEncoder → (序列 emb ⊕ A0 标量) → MLP head → 标量分数。

    expose_embedding=True 时 forward 额外返回序列 embedding (供 GBDT-on-embedding head 抽取)。
    """

    def __init__(self, n_a0: int, anchor_emb: int = ANCHOR_EMB, seq_emb: int = SEQ_EMB):
        super().__init__()
        self.anchor = AnchorEncoder(anchor_emb)
        self.seq = SeqEncoder(anchor_emb, seq_emb)
        self.head = nn.Sequential(
            nn.Linear(seq_emb + n_a0, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def embed(self, x_seq: torch.Tensor) -> torch.Tensor:
        return self.seq(self.anchor(x_seq))

    def forward(self, x_seq, a0, return_embedding: bool = False):
        emb = self.embed(x_seq)
        score = self.head(torch.cat([emb, a0], dim=-1)).squeeze(-1)
        if return_embedding:
            return score, emb
        return score
