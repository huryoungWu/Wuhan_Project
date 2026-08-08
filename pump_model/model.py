import numpy as np
import torch
import torch.nn as nn

# ============================================================================
# model.py — 水泵预测模型结构 (训练/推理共用)
#
# 2026-08-08 从 train.py / pump_inference.py 抽取合并: 原两处各有一份
# DeepWaterPlantModelWithEmbedding 定义 (推理版 dropout=0 以"推理时关闭"),
# 现统一为本文件定义 (训练版)。推理时 model.eval() 使 dropout 不生效,
# 与原推理版行为完全一致; 网络层结构与 state_dict 完全相同, 旧权重
# (models/model_v2_combo_split.pt) 可直接加载。
#
# 本文件同时包含 PINN 物理约束 (physics_loss_pump) 与 7 台泵额定参数 —
# 物理损失与模型架构/额定参数强耦合, 随模型一起维护。
# ============================================================================

# ============================================================================
# 泵额定参数与 PINN 物理约束 (相似定律)
# ============================================================================

# 7台泵额定参数 (顺序: P1~P7), 相似定律: Q∝f, H∝f², P∝f³
RATED_Q = [2020, 670, 2020, 1260, 670, 1260, 670]   # 额定流量 (m3/h)
RATED_P = [220, 132, 220, 220, 132, 220, 90]        # 额定功率 (kW)
RATED_H = [32, 43, 32, 43, 43, 43, 33]              # 额定扬程 (m)
RATED_F = 50.0                                       # 额定频率 (Hz)

# 物理约束权重
W_GHOST = 0.01    # 幽灵流量 (泵关→对应管路流量必须为0)
W_FLOW  = 0.01    # 总管流量上界 ≤ Σ理论流量
W_POWER = 1e-4    # 效率反推电功率 ≈ Σ理论功率
W_HEAD  = 0.05    # 系统扬程 ≤ 运行泵最大理论扬程×1.2
SLACK   = 1.15    # 物理上界允许超出15% (泵曲线裕度/传感器误差)


def physics_loss_pump(discrete_x, continuous_x, y_pred_scaled,
                      cont_mean, cont_scale, out_mean, out_scale,
                      rated_q, rated_p, rated_h, rated_f=RATED_F,
                      w_ghost=W_GHOST, w_flow=W_FLOW, w_power=W_POWER, w_head=W_HEAD,
                      slack=SLACK):
    """PINN 物理约束损失 — 基于水泵相似定律, 在物理空间计算 (可微, 梯度可回传)

    y_pred_scaled 经输出标准化器反变换回物理空间后施加约束:
      1. 幽灵流量: 泵1~6全停 → 170:1/170:2 流量必须为0; 泵7停 → 70:3 为0 (硬约束)
      2. 流量上界: 总管流量 ≤ slack × Σ 运行泵额定流量×(f/f₀)   (Q∝f)
      3. 功率一致: 效率反推电功率 ≈ Σ 运行泵额定功率×(f/f₀)³  (P∝f³, 上界×slack)
      4. 扬程约束: 系统扬程 ≤ slack × 1.2 × 运行泵最大理论扬程    (H∝f²)
      slack=1.15 表示物理上界允许超出约15% (泵曲线裕度/传感器误差)
    """
    y = y_pred_scaled.float() * out_scale + out_mean    # (B,4) 物理空间
    f1, f2, f3, eff = y[:, 0], y[:, 1], y[:, 2], y[:, 3]

    freqs = continuous_x[:, :7] * cont_scale[:7] + cont_mean[:7]   # Hz
    press = continuous_x[:, 7] * cont_scale[7] + cont_mean[7]      # MPa
    states = discrete_x.float()                                     # (B,7) 0/1

    fr = freqs / rated_f
    q_theory = states * rated_q * fr               # 理论流量 (m3/h)
    p_theory = states * rated_p * fr ** 3          # 理论功率 (kW)
    h_theory = states * rated_h * fr ** 2          # 理论扬程 (m)

    relu = torch.relu

    # 1) 幽灵流量: 泵关 → 对应管路流量必须为0
    main_off = (states[:, :6].sum(dim=1) == 0).float()
    p7_off = (states[:, 6] == 0).float()
    L_ghost = (main_off * (relu(f1) + relu(f2))).mean() + (p7_off * relu(f3)).mean()

    # 2) 流量上界: 总管流量 ≤ slack × Σ 理论流量 (允许超出15%)
    total_q = f1 + f2 + f3
    L_flow = relu(total_q - slack * q_theory.sum(dim=1)).mean()

    # 3) 功率一致性: 水力功率/效率 → 电功率 ≈ Σ 理论功率 (上界×slack)
    head_m = press * 102.0                          # MPa → 米水柱
    ph_kw = 9.81 * total_q * head_m / 3600.0        # 水力功率 (kW)
    pe = ph_kw / (eff / 100.0 + 1e-6)               # 效率反推电功率 (kW)
    L_power = (relu(pe - 1.5 * slack * p_theory.sum(dim=1))
               + relu(0.33 * p_theory.sum(dim=1) - pe)).mean()

    # 4) 扬程约束: 系统扬程 ≤ slack × 1.2 × 最大理论扬程
    h_max = h_theory.max(dim=1).values
    L_head = relu(head_m - slack * 1.2 * h_max).mean()

    return w_ghost * L_ghost + w_flow * L_flow + w_power * L_power + w_head * L_head


# ============================================================================
# 模型定义
# ============================================================================

class DeepWaterPlantModelWithEmbedding(nn.Module):
    """
    改进的多任务水泵预测模型 v6 — 纯流量+效率 (无功率)

    架构:
      输入 → 共享编码器 → ┬─ 170系统头 → [170:1流量, 170:2流量]
                          ├─ 70:3专用头 → [70:3流量] (P7频率捷径)
                          └─ 效率头     → [总管效率]
      输出: 4维 [170:1流量, 170:2流量, 70:3流量, 总管效率]

    训练/推理共用此定义: 推理时调用 model.eval() 关闭 dropout,
    与推理专用定义 (dropout=0) 行为一致。
    """
    def __init__(self, num_pumps, continuous_dim, output_dim=4, embed_dim=8,
                 pump_embedding_sizes=None):
        super(DeepWaterPlantModelWithEmbedding, self).__init__()
        self.num_pumps = num_pumps
        self.continuous_dim = continuous_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        if pump_embedding_sizes is None:
            pump_embedding_sizes = [2] * num_pumps
        self.pump_embedding_sizes = pump_embedding_sizes

        # ── Embedding: 7泵离散状态 ──
        self.pump_embeddings = nn.ModuleList([
            nn.Embedding(self.pump_embedding_sizes[i], embed_dim)
            for i in range(num_pumps)
        ])
        total_discrete_dim = num_pumps * embed_dim
        total_input_dim = total_discrete_dim + continuous_dim

        # ── 共享编码器 (带残差) ──
        self.enc_fc1 = nn.Linear(total_input_dim, 512)
        self.enc_bn1 = nn.BatchNorm1d(512)
        self.enc_fc2 = nn.Linear(512, 256)
        self.enc_bn2 = nn.BatchNorm1d(256)
        self.enc_fc3 = nn.Linear(256, 128)
        self.enc_bn3 = nn.BatchNorm1d(128)

        self.enc_proj1 = nn.Linear(total_input_dim, 512)
        self.enc_proj2 = nn.Linear(512, 256)

        # ── 泵组互斥注意力 ──
        self.pair_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=4, batch_first=True, dropout=0.1
        )

        # ── 170系统头: 输出 [170:1流量, 170:2流量] ──
        self.head_170_fc1 = nn.Linear(128, 64)
        self.head_170_bn1 = nn.BatchNorm1d(64)
        self.head_170_fc2 = nn.Linear(64, 32)
        self.head_170_bn2 = nn.BatchNorm1d(32)
        self.head_170_out = nn.Linear(32, 2)

        # ── 70:3专用头: P7频率物理捷径 → [70:3流量] ──
        self.head_703_linear = nn.Linear(1, 16)
        self.head_703_fc1 = nn.Linear(128 + 16, 48)
        self.head_703_bn1 = nn.BatchNorm1d(48)
        self.head_703_fc2 = nn.Linear(48, 24)
        self.head_703_bn2 = nn.BatchNorm1d(24)
        self.head_703_out = nn.Linear(24, 1)

        # ── 效率头: 输出 [总管效率] ──
        self.head_eff_fc1 = nn.Linear(128, 48)
        self.head_eff_bn1 = nn.BatchNorm1d(48)
        self.head_eff_fc2 = nn.Linear(48, 24)
        self.head_eff_bn2 = nn.BatchNorm1d(24)
        self.head_eff_out = nn.Linear(24, 1)

        # ── 通用 ──
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(0.25)

    def forward(self, discrete_x, continuous_x):
        batch_size = discrete_x.shape[0]

        # ── 1. Embedding ──
        embedded_pumps = []
        for i in range(self.num_pumps):
            idx = discrete_x[:, i].clamp(0, self.pump_embedding_sizes[i] - 1)
            emb = self.pump_embeddings[i](idx)
            embedded_pumps.append(emb)
        pump_emb_stack = torch.stack(embedded_pumps, dim=1)  # [B, 7, embed_dim]

        # ── 2. 泵组互斥注意力 ──
        attn_out, _ = self.pair_attention(pump_emb_stack, pump_emb_stack, pump_emb_stack)
        discrete_embedded = attn_out.reshape(batch_size, -1)  # [B, 7*embed_dim]

        # ── 3. 共享编码器 (带残差) ──
        x = torch.cat([discrete_embedded, continuous_x], dim=1)

        h1 = self.enc_fc1(x)
        h1 = self.enc_bn1(h1)
        h1 = self.leaky_relu(h1)
        h1 = self.dropout(h1)
        h1 = h1 + self.enc_proj1(x)

        h2 = self.enc_fc2(h1)
        h2 = self.enc_bn2(h2)
        h2 = self.leaky_relu(h2)
        h2 = self.dropout(h2)
        h2 = h2 + self.enc_proj2(h1)

        shared = self.enc_fc3(h2)
        shared = self.enc_bn3(shared)
        shared = self.leaky_relu(shared)
        shared = self.dropout(shared)

        # ── 4a. 170系统头: [170:1流量, 170:2流量] ──
        h_170 = self.head_170_fc1(shared)
        h_170 = self.head_170_bn1(h_170)
        h_170 = self.leaky_relu(h_170)
        h_170 = self.dropout(h_170)
        h_170 = self.head_170_fc2(h_170)
        h_170 = self.head_170_bn2(h_170)
        h_170 = self.leaky_relu(h_170)
        out_170 = self.head_170_out(h_170)  # [B, 2]

        # ── 4b. 70:3专用头: [70:3流量] (P7频率捷径) ──
        p7_freq = continuous_x[:, 6:7]
        p7_feat = self.head_703_linear(p7_freq)
        p7_feat = self.leaky_relu(p7_feat)

        h_703 = torch.cat([shared, p7_feat], dim=1)
        h_703 = self.head_703_fc1(h_703)
        h_703 = self.head_703_bn1(h_703)
        h_703 = self.leaky_relu(h_703)
        h_703 = self.dropout(h_703)
        h_703 = self.head_703_fc2(h_703)
        h_703 = self.head_703_bn2(h_703)
        h_703 = self.leaky_relu(h_703)
        out_703 = self.head_703_out(h_703)  # [B, 1]

        # ── 4c. 效率头: [总管效率] ──
        h_eff = self.head_eff_fc1(shared)
        h_eff = self.head_eff_bn1(h_eff)
        h_eff = self.leaky_relu(h_eff)
        h_eff = self.dropout(h_eff)
        h_eff = self.head_eff_fc2(h_eff)
        h_eff = self.head_eff_bn2(h_eff)
        h_eff = self.leaky_relu(h_eff)
        out_eff = self.head_eff_out(h_eff)  # [B, 1]

        # ── 5. 拼接: [170:1流量, 170:2流量, 70:3流量, 总管效率] ──
        output = torch.cat([out_170, out_703, out_eff], dim=1)
        return output
