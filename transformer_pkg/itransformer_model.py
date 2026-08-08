import torch
import torch.nn as nn

# ============================================================================
# itransformer_model.py — iTransformer 流量预测模型结构 (训练/推理共用)
#
# 参考: Liu et al., "iTransformer: Inverted Transformers Are Effective for Time
# Series Forecasting", ICLR 2024.
#
# 与 transformer_model.py 的 TimeSeriesTransformer 对比:
#   vanilla Transformer 把"时间步"当 token, 对时间做 attention;
#   iTransformer 反过来——把"每个变量的一条序列"当 token, 对变量做 attention,
#   时间依赖交给逐变量的 MLP 嵌入 (Linear(L → d_model)), 预测时每变量独立出整个
#   horizon。本项目输入是 ~96 维特征 (流量 + 时间特征 + 滞后/滚动统计), 跨变量
#   attention 天然适配多特征结构。
#
# 接口与 TimeSeriesTransformer 完全一致 (src, target_len, tgt, teacher_forcing_ratio),
# 以便共用 train_transformer.py / inference_transformer.py 的训练与评估循环;
# tgt 与 teacher_forcing 不使用 (直接多步输出)。
#
# 前置约束:
#   - RevIN 对窗口做逐通道实例归一化, 与管线已有的全局 StandardScaler 叠加;
#     完全常量通道 (如 is_weekend 在 1 天窗口内) 会被归一化到 ≈0, 属可接受的
#     信息丢失 (无数值爆炸), 不处理。
#   - 要求 target_transform=None: 反归一化后目标通道处于 feature_scaler 域,
#     与 target_scaler 域仅在 target_transform=None (二者标定同一原始列) 时一致。
# ============================================================================


class RevIN(nn.Module):
    """可逆实例归一化 (Reversible Instance Normalization, Kim et al. 2021)。

    官方 iTransformer 实现的逐通道实例归一化:
      - norm:   逐样本、逐通道对 lookback 轴求 mean/var (均 detach, 统计量不参与
                梯度), 归一化后做逐通道 affine (weight=1, bias=0 初始化);
      - denorm: 用同一份统计量 + affine 逆变换还原。
    统计量只在单次 forward 内有效 (norm 后紧跟 denorm), 批大小任意 — batch=1
    推理与训练等价。
    """

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode="norm"):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise ValueError(f"mode 只支持 norm/denorm, 收到 {mode}")
        return x

    def _get_statistics(self, x):
        # x: (B, L, C) → 沿 L 轴统计, 逐样本逐通道
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / (self.stdev + self.eps)
        if self.affine:
            x = x * self.weight + self.bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.bias
            x = x / (self.weight + self.eps)
        x = x * self.stdev + self.mean
        return x


class iTransformer(nn.Module):
    """iTransformer: 变量即 token 的 Transformer Encoder 多步预测模型。

    结构:
      输入 (B, L, C) → RevIN 实例归一化
      → 转置 (B, C, L): 每个变量的一条序列 = 一个 token
      → 共享 Linear(L → d_model) 嵌入 + 可学习变量位置编码 (1, C, d_model)
      → N 层 Transformer Encoder (跨变量 self-attention)
      → LayerNorm → 共享 Linear(d_model → horizon) 逐变量出完整预测
      → 转置 (B, horizon, C) → RevIN 反归一化
      → 取目标变量通道 → (B, target_len, output_dim)

    超参含义与 TimeSeriesTransformer 对齐 (d_model/nhead/num_layers/
    dim_feedforward/dropout 从同一份 cfg 传入), 与基线同规模公平对比。

    target_idx: 目标变量在 feature_cols 中的下标 (本管线中 Total_Flow 恒为 0,
    但用 feature_cols.index(target_cols[0]) 解析, 不硬编码)。
    """

    def __init__(self, input_dim, output_dim, horizon, input_len,
                 d_model=64, nhead=4, num_layers=3, dim_feedforward=256,
                 dropout=0.1, target_idx=0):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"
        self.output_dim = output_dim
        self.horizon = horizon
        self.target_idx = target_idx

        self.revin = RevIN(input_dim)
        self.embedding = nn.Linear(input_len, d_model)        # 共享嵌入: 序列 → token
        self.var_pos_emb = nn.Parameter(torch.zeros(1, input_dim, d_model))  # 变量位置编码

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="relu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, horizon)               # 共享输出头: 逐变量出整个 horizon

    def forward(self, src, target_len, tgt=None, teacher_forcing_ratio=0.0):
        # src: (B, L, C)
        x = self.revin(src, "norm")                           # (B, L, C) 实例归一化
        x = x.permute(0, 2, 1)                                # (B, C, L) 变量 = token
        x = self.embedding(x) + self.var_pos_emb              # (B, C, d_model)
        x = self.encoder(x)                                   # 跨变量 attention
        x = self.head(self.layer_norm(x))                     # (B, C, horizon)
        x = x.permute(0, 2, 1)                                # (B, horizon, C)
        x = self.revin(x, "denorm")                           # 反归一化还原
        x = x[..., self.target_idx:self.target_idx + self.output_dim]  # (B, H, output_dim)
        return x[:, :target_len]
