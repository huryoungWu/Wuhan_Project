import math

import torch
import torch.nn as nn

# ============================================================================
# transformer_model.py — Transformer 流量预测模型结构 (训练/推理共用)
#
# 2026-08-08 从 train_transformer.py 原样抽取; train_transformer.py 与
# inference_transformer.py 均从此文件导入模型定义, 保证训练与推理用的是
# 同一个模型结构 (同一份代码)。
#
# 结构: 特征投影 + 正弦位置编码 + N 层 Transformer Encoder + 线性输出头,
# 直接多步输出 (无自回归 / 无 teacher forcing), 训练可并行。
#
# 前向接口与 LSTM 版一致 (src, target_len, tgt, teacher_forcing_ratio),
# 以便共用 evaluate / 训练循环; tgt 与 teacher_forcing 不使用。
# ============================================================================


def build_sinusoid_pe(seq_len, d_model):
    """正弦位置编码 (Vaswani et al. 2017): 偶维 sin / 奇维 cos, 频率按 10000^(-2k/d) 递减。

    与窗口长度解耦: 任意 seq_len 即时生成, 换 lookback 窗口长度无需重训
    (可学习位置编码的长度与训练窗口绑定, 做不到这一点)。
    """
    assert d_model % 2 == 0, "d_model 需为偶数 (正弦编码按偶/奇维拆分)"
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)   # (L, 1)
    k = torch.arange(d_model // 2, dtype=torch.float32)             # (d/2,)
    freq = torch.exp(k * (-math.log(10000.0) / (d_model / 2)))      # 10000^(-2k/d)
    ang = pos * freq                                                # (L, d/2)
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang)
    return pe


class TimeSeriesTransformer(nn.Module):
    """Transformer Encoder 多步预测模型: 输入投影 + 正弦位置编码 + 多头自注意力 + 线性输出头。

    结构:
      Linear(input_dim → d_model) 特征投影
      + 正弦位置编码 (与窗口长度解耦, 换 lookback 无需重训)
      + N 层 Transformer Encoder (自注意力聚合全窗口信息, 长程依赖)
      + 最后时刻表示 → Linear(d_model → horizon * output_dim) 直接多步输出
    """
    def __init__(self, input_dim, output_dim, horizon, input_len,
                 d_model=64, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"
        self.input_proj = nn.Linear(input_dim, d_model)
        # input_len 仅用于显式声明支持的最大输入长度, 实际位置编码按输入长度即时生成

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="relu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_dim = output_dim
        self.horizon = horizon
        self.head = nn.Linear(d_model, horizon * output_dim)

    def forward(self, src, target_len, tgt=None, teacher_forcing_ratio=0.0):
        pos = build_sinusoid_pe(src.size(1), self.input_proj.out_features).to(src.device)
        h = self.input_proj(src) + pos
        h = self.encoder(h)
        h = h[:, -1]                                   # 最后时刻 (自注意力已聚合全窗口)
        out = self.head(h).view(h.size(0), -1, self.output_dim)
        return out[:, :target_len]
