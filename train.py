
"""
水厂水泵运行预测模型 — v6 纯流量+效率版 (总功率由累计电度差分计算)
改进:
  1. 清洗 NaN 行、全停状态
  1b. 总功率由累计电度差分计算 (compute_total_power_from_meters)，p1~6用172电度Δ/Δh，p7用70:7总有功
  2. 按泵组组合划分训练/测试集
  3. 移除功率预测, 仅输出 3管流量 + 1效率 = 4维
  4. NN容量全部用于流量和效率
  5. 压力修正: 修正后压力 = 原压力 - (吸水井液位-3.58)/102 (MPa)
     液位读取自原始列 '170:吸水井液位', 修正后压力参与训练, 液位本身不进入特征
  6. PINN物理约束: 基于水泵相似定律 (Q∝f, H∝f², P∝f³) 施加物理损失
     - 泵关闭 → 对应管路流量必须为0 (消除幽灵流量)
     - 总管流量 ≤ Σ 理论流量; 效率反推电功率 ≈ Σ 理论功率; 系统扬程 ≤ 最大理论扬程
"""
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
import time


# ============================================================================
# 0. 泵功率计算
# ============================================================================

def compute_pump_powers(df):
    """从三相电压/电流数据计算7个泵的功率。"""
    for pn in ['1', '2', '3', '4', '5', '6']:
        va_col = f'172:{pn}_泵A相电压'
        vb_col = f'172:{pn}_泵B相电压'
        vc_col = f'172:{pn}_泵C相电压'
        ia_col = f'172:{pn}_泵A相电流'
        ib_col = f'172:{pn}_泵B相电流'
        ic_col = f'172:{pn}_泵C相电流'

        missing = [c for c in [va_col, vb_col, vc_col, ia_col, ib_col, ic_col]
                   if c not in df.columns]
        if missing:
            print(f"  警告: 泵{pn}缺少列 {missing}, 功率设为0")
            df[f'泵{pn}_功率_kVA'] = 0.0
            continue

        df[f'泵{pn}_功率_kVA'] = (
            df[va_col].fillna(0) * df[ia_col].fillna(0) +
            df[vb_col].fillna(0) * df[ib_col].fillna(0) +
            df[vc_col].fillna(0) * df[ic_col].fillna(0)
        ) / 1000.0

    if '70:7_总有功' in df.columns:
        df['泵7_功率_kW'] = df['70:7_总有功'].fillna(0) / 1000.0
    else:
        print("  警告: 缺少 '70:7_总有功' 列, 泵7功率设为0")
        df['泵7_功率_kW'] = 0.0

    return df


def clean_pump_power(df, freq_cols, power_cols,
                     mad_multiplier=5.0, min_samples_per_bin=20):
    """清洗每个泵的单泵功率数据 (频率分段MAD离群检测) — 向量化加速版。"""
    # 不 copy 全量 df（避免 OOM），直接在原 df 上标记 NaN
    total_before = len(df)
    stats = {}

    for i, (freq_col, power_col) in enumerate(zip(freq_cols, power_cols)):
        pump_name = f'泵{i+1}'
        unit = 'kW' if 'kW' in power_col else 'kVA'
        removed_reasons = {}

        # 1. 运行但功率<=0
        bad_power = (df[freq_col] > 0) & (df[power_col] <= 0)
        removed_reasons['运行但功率<=0'] = int(bad_power.sum())
        df.loc[bad_power, power_col] = np.nan

        running_mask = (df[freq_col] > 0) & (df[power_col] > 0)
        if running_mask.sum() < 10:
            stats[pump_name] = {'unit': unit, 'before': total_before,
                                'removed': removed_reasons, 'total_removed': int(bad_power.sum())}
            df = df[~(bad_power & ~running_mask)] if bad_power.sum() else df
            continue

        pump_mean_power = df.loc[running_mask, power_col].mean()

        # 2. 高频低功率
        high_freq_low = (df[freq_col] > 20) & (df[power_col] > 0) & (df[power_col] < pump_mean_power * 0.05)
        removed_reasons['高频(>20Hz)但功率<5%均值'] = int(high_freq_low.sum())
        df.loc[high_freq_low, power_col] = np.nan
        running_mask = running_mask & ~high_freq_low

        # 3. 频率分段MAD — 用 groupby.transform 向量化替换 Python for 循环
        tmp = df.loc[running_mask, [freq_col, power_col]].copy()
        tmp['freq_bin'] = tmp[freq_col].round().astype(int)

        def _mad_outlier(grp):
            if len(grp) < min_samples_per_bin:
                return pd.Series(False, index=grp.index)
            med = grp[power_col].median()
            mad = (grp[power_col] - med).abs().median()
            if mad < 1e-9:
                return pd.Series(False, index=grp.index)
            return (grp[power_col] - med).abs() / mad > mad_multiplier

        mad_flags = tmp.groupby('freq_bin', group_keys=False).apply(_mad_outlier)
        mad_outlier_count = int(mad_flags.sum())
        removed_reasons['频率段MAD离群'] = mad_outlier_count

        outlier_indices = mad_flags[mad_flags].index
        df.loc[outlier_indices, power_col] = np.nan

        total_removed = int(bad_power.sum() + high_freq_low.sum() + mad_outlier_count)
        stats[pump_name] = {
            'unit': unit,
            'running_before': int((df[freq_col] > 0).sum()),
            'running_after': int(running_mask.sum() - mad_outlier_count),
            'mean_power': pump_mean_power,
            'removed': removed_reasons,
            'total_removed': total_removed
        }

    # 一次性删除所有标记为 NaN 的行
    before_drop = len(df)
    df = df.dropna(subset=power_cols)
    stats['total_before'] = total_before
    stats['total_after'] = len(df)
    stats['total_removed'] = total_before - len(df)
    return df, stats


def compute_total_power_from_meters(df):
    """从累计电度表读数差分计算7泵总功率 (kW)，写入 df['总功率'] 列。

    泵1~6: 172:1~6_泵电度 (累计kWh) → ΔkWh / Δ小时 → kW
    泵7:   70:7_总有功 (W) → /1000 → kW
    总功率 = 泵1+...+泵6 + 泵7

    分钟级差分噪声的处理：
      - 零值 → ffill(limit=15)（电表非每分钟刷新）
      - 5分钟滚动均值平滑尖峰
    """
    meter_cols = ['172:1_泵电度', '172:2_泵电度', '172:3_泵电度',
                  '172:4_泵电度', '172:5_泵电度', '172:6_泵电度']

    # 时间差 (小时)
    if 'F_DateTime' in df.columns:
        time_series = pd.to_datetime(df['F_DateTime'])
    else:
        time_series = pd.to_datetime(df.index)
    dt_hours = time_series.diff().dt.total_seconds() / 3600.0

    # 泵1~6: 累计kWh差分 → kW
    pump_powers = {}
    for col in meter_cols:
        name = col.replace('172:', '').replace('_泵电度', '')
        dkwh = df[col].diff()
        pump_powers[f'泵{name}_功率_kW'] = (dkwh / dt_hours).clip(lower=0, upper=500)

    # 泵7: 瞬时有功 W → kW
    if '70:7_总有功' in df.columns:
        pump_powers['泵7_功率_kW'] = (df['70:7_总有功'].fillna(0) / 1000).clip(lower=0, upper=200)
    else:
        pump_powers['泵7_功率_kW'] = 0.0

    # 总功率 (kW) → 平滑处理
    raw_total = sum(pump_powers.values())
    df['总功率'] = (raw_total
                    .replace(0, np.nan)
                    .ffill(limit=15)
                    .rolling(window=5, center=True, min_periods=1)
                    .mean()
                    .fillna(0))

    return df


def compute_efficiency(df, power_cols, flow_cols, pressure_col='170:总管压力',
                       pf_estimate=0.88, save_csv=True):
    """计算系统总管效率并写入CSV。"""
    df['总流量_m3h'] = df[flow_cols[0]].fillna(0) + df[flow_cols[1]].fillna(0) + df[flow_cols[2]].fillna(0)
    df['水力功率_kW'] = 9.81 * df['总流量_m3h'] * df[pressure_col].fillna(0) * 102 / 3600
    # total_kva = sum(df[c].fillna(0) for c in power_cols[:6])
    # pump7_kw = df[power_cols[6]].fillna(0)
    # 总功率已在 compute_total_power_from_meters() 中完成差分+平滑
    df['总有功功率_kW'] = df['总功率'].fillna(0)
    df['总管效率_pct'] = (df['水力功率_kW'] / (df['总有功功率_kW'] + 1e-6) * 100.0).clip(0, 90)
    # df['视在效率_pct'] = (df['水力功率_kW'] / (total_kva + pump7_kw + 1e-6) * 100.0).clip(0, 90)
    valid_mask = (df['总流量_m3h'] > 100) & (df['总有功功率_kW'] > 30)
    df['总管效率_valid'] = np.where(valid_mask, df['总管效率_pct'], np.nan)
    # df['视在效率_valid'] = np.where(valid_mask, df['视在效率_pct'], np.nan)

    if save_csv:
        csv_path = r"D:\Wuhan_Project\new_data\merged_minute_all_with_efficiency.csv"
        print(f"\n保存含效率的CSV到: {csv_path}")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    return df


# ============================================================================
# 0b. 压力修正
# ============================================================================

def correct_pressure(df, level_baseline=3.58, level_divisor=102.0):
    """压力修正: 修正后压力 = 原压力 - (吸水井液位 - 3.58) / 102 (MPa)

    吸水井液位读取自原始列 '170:吸水井液位' (m)。修正后的压力直接覆盖
    df['170:总管压力'] 供下游清洗/效率计算/训练使用 (液位本身不进入特征)，
    并保留一列 '170:总管压力_修正' 作为缓存标记。
    """
    level_col = '170:吸水井液位'
    if level_col not in df.columns:
        print(f"  [WARN] 缺少列 '{level_col}', 无法修正压力 (修正量=0)")
        df['170:总管压力_修正'] = df['170:总管压力']
        return df

    level = df[level_col]
    n_missing = int(level.isna().sum())
    if n_missing > 0:
        print(f"  [WARN] '{level_col}' 缺失 {n_missing} 行, 这些行按修正量=0处理")

    correction = (level.fillna(level_baseline) - level_baseline) / level_divisor
    df['170:总管压力_修正'] = df['170:总管压力'] - correction
    df['170:总管压力'] = df['170:总管压力_修正']
    print(f"  压力修正: 修正后压力 = 原压力 - (吸水井液位-{level_baseline})/{level_divisor}")
    print(f"    修正后总管压力范围: [{df['170:总管压力'].min():.4f}, {df['170:总管压力'].max():.4f}] MPa")
    return df


# ============================================================================
# 0c. 泵额定参数与 PINN 物理约束 (相似定律)
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
# 1. 模型定义
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        elif isinstance(obj, (np.floating,)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, np.bool_): return bool(obj)
        else: return super().default(obj)


class FastTensorDataLoader:
    """高效GPU数据加载器"""
    def __init__(self, *tensors, batch_size=4096, shuffle=True):
        self.tensors = tensors
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len, device=self.tensors[0].device)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i : self.i + self.batch_size]
            batch = tuple(t[indices] for t in self.tensors)
        else:
            batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return (self.dataset_len + self.batch_size - 1) // self.batch_size


class DeepWaterPlantModelWithEmbedding(nn.Module):
    """
    改进的多任务水泵预测模型 v6 — 纯流量+效率 (无功率)

    架构:
      输入 → 共享编码器 → ┬─ 170系统头 → [170:1流量, 170:2流量]
                          ├─ 70:3专用头 → [70:3流量] (P7频率捷径)
                          └─ 效率头     → [总管效率]
      输出: 4维 [170:1流量, 170:2流量, 70:3流量, 总管效率]
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


# ============================================================================
# 2. 训练函数
# ============================================================================

def train_model(train_df, test_df, discrete_cols, continuous_cols,
                flow_cols, eff_col, engineered_cols,
                num_epochs=250, batch_size=16384, patience=60):
    """单模型多输出训练: output_dim=4 (3管流量 + 1效率)"""
    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    all_output_cols = list(flow_cols) + [eff_col]
    output_dim = len(all_output_cols)  # 4

    X_discrete_train = np.where(train_df[discrete_cols].values > 0, 1, 0)
    X_discrete_test = np.where(test_df[discrete_cols].values > 0, 1, 0)

    all_continuous_cols = continuous_cols
    X_continuous_train = train_df[all_continuous_cols].values
    X_continuous_test = test_df[all_continuous_cols].values

    y_train = train_df[all_output_cols].values
    y_test = test_df[all_output_cols].values

    def clean(arr):
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    X_continuous_train = clean(X_continuous_train)
    X_continuous_test = clean(X_continuous_test)
    y_train = clean(y_train)
    y_test = clean(y_test)

    continuous_scaler = StandardScaler()
    X_continuous_train_scaled = continuous_scaler.fit_transform(X_continuous_train)
    X_continuous_test_scaled = continuous_scaler.transform(X_continuous_test)

    output_scaler = StandardScaler()
    y_train_scaled = output_scaler.fit_transform(y_train)
    y_test_scaled = output_scaler.transform(y_test)

    # 安全保护: scale_过小会导致除零/inf, 用 1e-4 兜底
    eps = 1e-4
    for sc in [continuous_scaler, output_scaler]:
        bad = np.abs(sc.scale_) < eps
        if bad.any():
            print(f"  [WARN] {bad.sum()} 个特征的 scale_<{eps}, 已用 {eps} 兜底")
            sc.scale_ = np.where(bad, eps, sc.scale_)

    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    X_discrete_train_t = torch.tensor(X_discrete_train, dtype=torch.long, device=device)
    X_continuous_train_t = torch.tensor(X_continuous_train_scaled, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32, device=device)
    X_discrete_test_t = torch.tensor(X_discrete_test, dtype=torch.long, device=device)
    X_continuous_test_t = torch.tensor(X_continuous_test_scaled, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32, device=device)

    # PINN 物理约束所需张量 (标准化器仿射参数 + 泵额定参数)
    cont_mean_t = torch.tensor(continuous_scaler.mean_, dtype=torch.float32, device=device)
    cont_scale_t = torch.tensor(continuous_scaler.scale_, dtype=torch.float32, device=device)
    out_mean_t = torch.tensor(output_scaler.mean_, dtype=torch.float32, device=device)
    out_scale_t = torch.tensor(output_scaler.scale_, dtype=torch.float32, device=device)
    rated_q_t = torch.tensor(RATED_Q, dtype=torch.float32, device=device)
    rated_p_t = torch.tensor(RATED_P, dtype=torch.float32, device=device)
    rated_h_t = torch.tensor(RATED_H, dtype=torch.float32, device=device)

    train_loader = FastTensorDataLoader(
        X_discrete_train_t, X_continuous_train_t, y_train_t,
        batch_size=batch_size, shuffle=True
    )
    test_loader = FastTensorDataLoader(
        X_discrete_test_t, X_continuous_test_t, y_test_t,
        batch_size=batch_size, shuffle=False
    )

    model = DeepWaterPlantModelWithEmbedding(
        num_pumps=len(discrete_cols),
        continuous_dim=X_continuous_train_scaled.shape[1],
        output_dim=output_dim,
        embed_dim=8
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    print(f"使用设备: {device}")
    print(f"离散特征: {len(discrete_cols)} 个泵运行状态")
    print(f"连续特征维度: {X_continuous_train_scaled.shape[1]} (无工程特征)")
    print(f"输出维度: {output_dim} (3管流量 + 1效率)")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    use_amp = use_gpu and device.type == 'cuda'
    try:
        from torch.amp import GradScaler as AmpGradScaler
        scaler = AmpGradScaler('cuda', enabled=use_amp)
    except ImportError:
        try:
            from torch.cuda.amp import GradScaler as AmpGradScaler
            scaler = AmpGradScaler(enabled=use_amp)
        except ImportError:
            use_amp = False; scaler = None

    best_val_loss = float('inf')
    best_model_state = None
    trigger_times = 0
    train_losses = []
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for discrete_batch, continuous_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if use_amp and scaler is not None:
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(discrete_batch, continuous_batch)
                    loss = criterion(outputs, y_batch) + physics_loss_pump(
                        discrete_batch, continuous_batch, outputs,
                        cont_mean_t, cont_scale_t, out_mean_t, out_scale_t,
                        rated_q_t, rated_p_t, rated_h_t)
            else:
                outputs = model(discrete_batch, continuous_batch)
                loss = criterion(outputs, y_batch) + physics_loss_pump(
                    discrete_batch, continuous_batch, outputs,
                    cont_mean_t, cont_scale_t, out_mean_t, out_scale_t,
                    rated_q_t, rated_p_t, rated_h_t)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for discrete_batch, continuous_batch, y_batch in test_loader:
                if use_amp and scaler is not None:
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        outputs = model(discrete_batch, continuous_batch)
                else:
                    outputs = model(discrete_batch, continuous_batch)
                val_loss += (criterion(outputs, y_batch) + physics_loss_pump(
                    discrete_batch, continuous_batch, outputs,
                    cont_mean_t, cont_scale_t, out_mean_t, out_scale_t,
                    rated_q_t, rated_p_t, rated_h_t)).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        train_losses.append(avg_train_loss)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            trigger_times = 0
        else:
            trigger_times += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

        if trigger_times >= patience:
            print(f"早停触发，第 {epoch+1} 轮停止")
            break

    training_time = time.time() - start_time

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

    # ======== 评估 ========
    model.eval()
    y_pred_list = []; y_true_list = []
    with torch.no_grad():
        for discrete_batch, continuous_batch, y_batch in test_loader:
            if use_amp and scaler is not None:
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(discrete_batch, continuous_batch)
            else:
                outputs = model(discrete_batch, continuous_batch)
            y_pred_list.append(outputs.cpu())
            y_true_list.append(y_batch.cpu())

    y_pred_scaled = torch.cat(y_pred_list, dim=0).numpy()
    y_true_scaled = torch.cat(y_true_list, dim=0).numpy()

    # ── DEBUG: 定位 pipe 1 inf 来源 ──
    print(f"\n  [DEBUG] output_scaler scale_: {output_scaler.scale_[:5]}")
    print(f"  [DEBUG] output_scaler mean_:  {output_scaler.mean_[:5]}")
    print(f"  [DEBUG] y_pred_scaled[:,0] range: [{y_pred_scaled[:,0].min():.6f}, {y_pred_scaled[:,0].max():.6f}]")
    print(f"  [DEBUG] y_true_scaled[:,0] range: [{y_true_scaled[:,0].min():.6f}, {y_true_scaled[:,0].max():.6f}]")
    has_inf_pred = np.isinf(y_pred_scaled).any(axis=0)
    has_inf_true = np.isinf(y_true_scaled).any(axis=0)
    print(f"  [DEBUG] y_pred_scaled 含inf的维度: {np.where(has_inf_pred)[0].tolist()}")
    print(f"  [DEBUG] y_true_scaled 含inf的维度: {np.where(has_inf_true)[0].tolist()}")

    y_pred_scaled = np.clip(y_pred_scaled, -50, 50)
    y_pred = output_scaler.inverse_transform(y_pred_scaled)
    y_true = output_scaler.inverse_transform(y_true_scaled)

    # DEBUG: 反标准化后
    print(f"  [DEBUG] y_pred[:,0] after inv: [{y_pred[:,0].min():.2f}, {y_pred[:,0].max():.2f}]")
    print(f"  [DEBUG] y_true[:,0] after inv: [{y_true[:,0].min():.2f}, {y_true[:,0].max():.2f}]")
    print(f"  [DEBUG] y_true[:,0] has inf: {np.isinf(y_true[:,0]).any()}, nan: {np.isnan(y_true[:,0]).any()}")
    print(f"  [DEBUG] y_pred[:,0] has inf: {np.isinf(y_pred[:,0]).any()}, nan: {np.isnan(y_pred[:,0]).any()}")

    y_pred = np.maximum(y_pred, 0)
    y_pred[:, 3] = np.clip(y_pred[:, 3], 0, 100)  # 效率在索引3

    metrics = {}
    for i, col in enumerate(all_output_cols):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        if '效率' in col: th = 10.0
        else: th = 10.0
        mask = y_true[:, i] > th
        mape = np.mean(np.abs(y_true[mask, i] - y_pred[mask, i]) / y_true[mask, i]) * 100 if mask.sum() > 0 else np.nan
        metrics[col] = {
            'MAE': round(mae, 2), 'RMSE': round(rmse, 2),
            'R2': round(r2, 4), 'MAPE(%)': round(mape, 2)
        }

    # flowtotal
    ft_true = y_true[:, :3].sum(axis=1); ft_pred = y_pred[:, :3].sum(axis=1)
    ft_rmse = np.sqrt(mean_squared_error(ft_true, ft_pred))
    ft_mae = mean_absolute_error(ft_true, ft_pred)
    ft_r2 = r2_score(ft_true, ft_pred)
    mask_ft = ft_true > 10
    ft_mape = np.mean(np.abs(ft_true[mask_ft] - ft_pred[mask_ft]) / ft_true[mask_ft]) * 100 if mask_ft.sum() > 0 else np.nan
    metrics['flowtotal(总和)'] = {
        'MAE': round(ft_mae, 2), 'RMSE': round(ft_rmse, 2),
        'R2': round(ft_r2, 4), 'MAPE(%)': round(ft_mape, 2)
    }

    return {
        'model': model,
        'metrics': metrics,
        'continuous_scaler': continuous_scaler,
        'output_scaler': output_scaler,
        'y_true': y_true,
        'y_pred': y_pred,
        'all_output_cols': all_output_cols,
        'training_time': training_time,
        'epochs_trained': len(train_losses),
        'final_train_loss': train_losses[-1] if train_losses else 0,
        'best_train_loss': min(train_losses) if train_losses else 0
    }


# ============================================================================
# 3. 按泵组组合评估
# ============================================================================

def evaluate_by_combination(test_df, y_true, y_pred, output_cols):
    results = []
    test_df = test_df.reset_index(drop=True)
    for combo in sorted(test_df['泵组状态'].unique()):
        mask = test_df['泵组状态'] == combo
        group_true = y_true[mask]; group_pred = y_pred[mask]
        if len(group_true) == 0:
            continue
        for i, col in enumerate(output_cols):
            rmse = np.sqrt(mean_squared_error(group_true[:, i], group_pred[:, i]))
            mae = mean_absolute_error(group_true[:, i], group_pred[:, i])
            r2 = r2_score(group_true[:, i], group_pred[:, i])
            valid_mask = group_true[:, i] > 10
            mape = np.mean(np.abs(group_true[valid_mask, i] - group_pred[valid_mask, i]) /
                          group_true[valid_mask, i]) * 100 if np.sum(valid_mask) > 0 else np.nan
            results.append({
                '泵组状态': combo, '输出列': col,
                '样本数': len(group_true), 'MAE': round(mae, 2),
                'RMSE': round(rmse, 2), 'R2': round(r2, 4),
                'MAPE(%)': round(mape, 2)
            })
    return pd.DataFrame(results)


# ============================================================================
# 4. 指标保存
# ============================================================================

def save_metrics_to_txt(metrics, train_combos, test_combos, train_samples, test_samples,
                         all_output_cols, output_dir=None):
    """将指标结果保存到本地txt文件，文件名含时间戳。

    Parameters
    ----------
    metrics : dict
        训练返回的 metrics 字典，包含各输出列的 MAE/RMSE/MAPE/R2
    train_combos : list or set
        训练集泵组组合列表
    test_combos : list or set
        测试集泵组组合列表
    train_samples : int
        训练集样本数
    test_samples : int
        测试集样本数
    all_output_cols : list
        输出列名列表 (前3为流量, 第4为效率)
    output_dir : str or None
        输出目录，默认为脚本所在目录
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    filename = f"metrics_result_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("水泵预测模型 — 指标评估结果\n")
        f.write(f"保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        # ── 数据集信息 ──
        f.write(f"训练集样本数: {train_samples:,}\n")
        f.write(f"测试集样本数: {test_samples:,}\n")

        # ── 训练集泵组组合 ──
        f.write("\n" + "-" * 40 + "\n")
        f.write(f"训练集泵组组合 (共 {len(train_combos)} 个):\n")
        f.write("-" * 40 + "\n")
        for combo in sorted(train_combos):
            f.write(f"  {combo}\n")

        # ── 测试集泵组组合 ──
        f.write("\n" + "-" * 40 + "\n")
        f.write(f"测试集泵组组合 (共 {len(test_combos)} 个):\n")
        f.write("-" * 40 + "\n")
        for combo in sorted(test_combos):
            f.write(f"  {combo}\n")

        # ── 流量指标 ──
        f.write("\n" + "=" * 60 + "\n")
        f.write("管道流量 — 评估指标\n")
        f.write("=" * 60 + "\n")
        for col in all_output_cols[:3]:
            if col in metrics:
                m = metrics[col]
                f.write(f"\n  [{col}]\n")
                f.write(f"    MAE  = {m['MAE']:.2f} m³/h\n")
                f.write(f"    RMSE = {m['RMSE']:.2f} m³/h\n")
                f.write(f"    MAPE = {m['MAPE(%)']:.2f} %\n")
                f.write(f"    R²   = {m['R2']:.4f}\n")

        # ── 总流量指标 ──
        if 'flowtotal(总和)' in metrics:
            m = metrics['flowtotal(总和)']
            f.write(f"\n  [flowtotal(总和)]\n")
            f.write(f"    MAE  = {m['MAE']:.2f} m³/h\n")
            f.write(f"    RMSE = {m['RMSE']:.2f} m³/h\n")
            f.write(f"    MAPE = {m['MAPE(%)']:.2f} %\n")
            f.write(f"    R²   = {m['R2']:.4f}\n")

        # ── 效率指标 ──
        if len(all_output_cols) > 3 and all_output_cols[3] in metrics:
            m = metrics[all_output_cols[3]]
            f.write("\n" + "=" * 60 + "\n")
            f.write("总管效率 — 评估指标\n")
            f.write("=" * 60 + "\n")
            f.write(f"\n  [{all_output_cols[3]}]\n")
            f.write(f"    MAE  = {m['MAE']:.2f} %\n")
            f.write(f"    RMSE = {m['RMSE']:.2f} %\n")
            f.write(f"    MAPE = {m['MAPE(%)']:.2f} %\n")
            f.write(f"    R²   = {m['R2']:.4f}\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"\n指标结果已保存到: {filepath}")
    return filepath


# ============================================================================
# 5. 主函数
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("水厂水泵运行预测模型")
    print("=" * 60)

    # ── 列定义 (无论缓存与否都相同) ──
    discrete_cols = [
        '170:1_泵运行', '170:2_泵运行', '170:3_泵运行',
        '170:4_泵运行', '170:5_泵运行', '170:6_泵运行',
        '70:7_泵运行'
    ]
    continuous_cols = [
        '170:1_运行频率', '170:2_运行频率', '170:3_运行频率',
        '170:4_运行频率', '170:5_运行频率', '170:6_运行频率',
        '70:7_运行频率',
        '170:总管压力',
    ]
    flow_cols = ['170:1_瞬时流量', '170:2_瞬时流量', '70:3_瞬时流量']
    power_output_cols = [
        '泵1_功率_kVA', '泵2_功率_kVA', '泵3_功率_kVA',
        '泵4_功率_kVA', '泵5_功率_kVA', '泵6_功率_kVA',
        '泵7_功率_kW'
    ]
    power_freq_cols = [
        '170:1_运行频率', '170:2_运行频率', '170:3_运行频率',
        '170:4_运行频率', '170:5_运行频率', '170:6_运行频率',
        '70:7_运行频率'
    ]
    eff_col = '总管效率_pct'

    # ── Parquet 缓存：跳过重复的 CSV 加载和清洗 ──
    CACHE_PATH = r"D:\Wuhan_Project\new_data\processed_cache.parquet"
    TRAIN_SAMPLE_SIZE = None  # 训练集最多采样数，None=不采样

    cache_usable = False
    if os.path.exists(CACHE_PATH):
        print(f"\n从缓存加载已处理数据: {CACHE_PATH}")
        t0 = time.time()
        df = pd.read_parquet(CACHE_PATH)
        print(f"数据形状: {df.shape}  加载耗时 {time.time()-t0:.1f}s")
        if '170:总管压力_修正' in df.columns:
            cache_usable = True
        else:
            print("  [WARN] 缓存为旧版(未含压力修正), 重新从CSV生成")

    if cache_usable:
        # 缓存已含修正压力: 修正值覆盖 170:总管压力
        df['170:总管压力'] = df['170:总管压力_修正']
        df = df.drop(columns=['170:总管压力_修正'])
        if 'F_DateTime' in df.columns:
            df['F_DateTime'] = pd.to_datetime(df['F_DateTime'])
    else:
        DATA_PATH = r"D:\Wuhan_Project\new_data\merged_minute_all.csv"
        print(f"\n加载数据: {DATA_PATH}")
        t0 = time.time()
        df = pd.read_csv(DATA_PATH)
        print(f"数据形状: {df.shape}  加载耗时 {time.time()-t0:.1f}s")
        print(f"时间范围: {df['F_DateTime'].min()} ~ {df['F_DateTime'].max()}")

        # 压力修正: 修正后压力 = 原压力 - (吸水井液位-3.58)/102
        df = correct_pressure(df)

        # ================================================================
        # 数据清洗
        # ================================================================
        n_before = len(df)
        print(f"\n数据清洗:")

        # 剔除传感器全NaN行
        data_cols = [c for c in df.columns if c not in ['F_TimeStamp', 'F_DateTime', 'flowtotal', 'LJflowtotal']]
        nan_mask = df[data_cols].isna().all(axis=1)
        if nan_mask.sum() > 0:
            print(f"  剔除传感器全NaN行: {nan_mask.sum()} 条")
            df = df[~nan_mask]

        # 管2极端野点
        mask_bad = df['170:2_瞬时流量'] > 10000
        if mask_bad.sum() > 0:
            print(f"  剔除管2极端野点 (>10000): {mask_bad.sum()} 条")
            df = df[~mask_bad]

        # 管1极端野点 (>10000)
        mask_bad_p1 = df['170:1_瞬时流量'] > 10000
        if mask_bad_p1.sum() > 0:
            print(f"  剔除管1极端野点 (>10000): {mask_bad_p1.sum()} 条")
            df = df[~mask_bad_p1]

        # 频率超范围
        freq_cols_raw = [c for c in df.columns if '运行频率' in c]
        for fc in freq_cols_raw:
            mask_freq = df[fc] > 55
            if mask_freq.sum() > 0:
                print(f"  剔除 {fc} 超范围 (>55Hz): {mask_freq.sum()} 条")
                df = df[~mask_freq]

        # 总管压力异常
        mask_p = (df['170:总管压力'] < 0.1) | (df['170:总管压力'] > 0.5)
        if mask_p.sum() > 0:
            print(f"  剔除总管压力异常: {mask_p.sum()} 条")
            df = df[~mask_p]

        # 全泵停止
        all_pumps_off = (df[discrete_cols] == 0).all(axis=1)
        if all_pumps_off.sum() > 0:
            print(f"  剔除全泵停止: {all_pumps_off.sum()} 条")
            df = df[~all_pumps_off]

        # 泵7停但管3>10
        mask_p7off_pipe3 = (df['70:7_泵运行'] == 0) & (df['70:3_瞬时流量'] > 10)
        if mask_p7off_pipe3.sum() > 0:
            print(f"  剔除泵7停但管3>10: {mask_p7off_pipe3.sum()} 条")
            df = df[~mask_p7off_pipe3]

        # 3σ原则清洗管1流量离群点 (仅泵1运行时)
        running_p1 = df['170:1_泵运行'] > 0
        flow_data = df.loc[running_p1, '170:1_瞬时流量']
        if len(flow_data) > 100:
            mean_f = flow_data.mean()
            std_f = flow_data.std()
            lo, hi = mean_f - 3 * std_f, mean_f + 3 * std_f
            outliers = running_p1 & ((df['170:1_瞬时流量'] < lo) | (df['170:1_瞬时流量'] > hi))
            if outliers.sum() > 0:
                print(f"  剔除 170:1_瞬时流量 3σ离群点 ({lo:.1f}~{hi:.1f}): {outliers.sum()} 条")
                df = df[~outliers]

        # 物理约束: 6台主泵全停时，管1和管2流量应≈0
        main_pumps_off = (df[['170:1_泵运行','170:2_泵运行','170:3_泵运行',
                              '170:4_泵运行','170:5_泵运行','170:6_泵运行']] == 0).all(axis=1)
        ghost_flow_mask = main_pumps_off & ((df['170:1_瞬时流量'] > 5) | (df['170:2_瞬时流量'] > 5))
        if ghost_flow_mask.sum() > 0:
            print(f"  剔除主泵全停但管1/管2有流量: {ghost_flow_mask.sum()} 条")
            df = df[~ghost_flow_mask]

        # 物理约束: 泵7停时管3流量应≈0
        ghost_pipe3 = (df['70:7_泵运行'] == 0) & (df['70:3_瞬时流量'] > 5)
        if ghost_pipe3.sum() > 0:
            print(f"  剔除泵7停但管3有流量: {ghost_pipe3.sum()} 条")
            df = df[~ghost_pipe3]

        # 流量突变检测: 按时间排序后，相邻秒变化率>500 m³/h/s 视为传感器跳变
        df = df.sort_values('F_TimeStamp')
        for fc in flow_cols:
            delta = df[fc].diff().abs()
            spike_mask = delta > 500
            if spike_mask.sum() > 0:
                # 删除跳变行及其后一行（跳变回位）
                spike_idx = spike_mask[spike_mask].index
                remove_idx = set(spike_idx) | set(i + 1 for i in spike_idx if i + 1 in df.index)
                print(f"  剔除 {fc} 流量突变 (>{500}): {len(remove_idx)} 条")
                df = df.drop(index=list(remove_idx))

        # 关键列为NaN
        essential_cols = (list(flow_cols) + list(discrete_cols)
                          + freq_cols_raw + ['170:总管压力'])
        nan_essential = df[essential_cols].isna().any(axis=1)
        if nan_essential.sum() > 0:
            print(f"  剔除关键列为NaN的行: {nan_essential.sum()} 条")
            df = df[~nan_essential]

        n_after = len(df)
        print(f"  清洗前: {n_before:,}, 清洗后: {n_after:,}, 剔除: {n_before-n_after:,} ({100*(n_before-n_after)/n_before:.2f}%)")

        # 计算功率 + 效率
        df = compute_pump_powers(df)
        df = compute_total_power_from_meters(df)        # 从累计电度差分计算总功率
        df = compute_efficiency(df, power_output_cols, flow_cols,
                                pf_estimate=0.88, save_csv=True)

        # 单泵功率清洗 (向量化版)
        print("\n清洗单泵功率 (频率分段MAD离群检测)...")
        t_clean = time.time()
        df, clean_stats = clean_pump_power(df, power_freq_cols, power_output_cols,
                                            mad_multiplier=3.5)
        print(f"清洗: {clean_stats['total_before']:,} -> {clean_stats['total_after']:,} "
              f"(剔除 {clean_stats['total_removed']:,}, {clean_stats['total_removed']/clean_stats['total_before']*100:.2f}%) "
              f"耗时 {time.time()-t_clean:.1f}s")

        # 保存缓存
        print(f"\n保存处理缓存: {CACHE_PATH}")
        df.to_parquet(CACHE_PATH, index=False)

    print(f"\n{'泵号':<6} {'运行点':<10} {'范围':<24} {'均值':<10} {'单位'}")
    for pn, pcol in zip(['1','2','3','4','5','6','7'], power_output_cols):
        fc = power_freq_cols[int(pn)-1]
        mask = (df[fc] > 0) & (df[pcol] > 0)
        d = df.loc[mask, pcol]
        unit = 'kW' if 'kW' in pcol else 'kVA'
        if len(d) > 10:
            print(f"  泵{pn}   {len(d):<10,} {d.min():.1f}~{d.max():.1f} {unit:<8} {d.mean():.1f}     {unit}")

    # ======== 泵组状态 ========
    df['泵组状态'] = df[discrete_cols].apply(
        lambda row: ''.join(['1' if x > 0 else '0' for x in row]), axis=1
    )

    # 统计泵组组合
    combo_counts = df['泵组状态'].value_counts()
    print(f"\n泵组组合总数: {len(combo_counts)}")
    print("Top 15 泵组组合:")
    for combo, cnt in combo_counts.head(15).items():
        print(f"  {combo}: {cnt:>10,} ({100*cnt/len(df):.1f}%)")

    engineered_cols = []

    # ================================================================
    # ★ 核心改进: 按泵组组合划分训练/测试集
    #   测试集出现的泵组组合不会出现在训练集中
    # ================================================================
    df['F_DateTime'] = pd.to_datetime(df['F_DateTime'])
    df = df.sort_values('F_TimeStamp')

    all_combos = sorted(combo_counts.index.tolist())
    n_combos = len(all_combos)

    # 过滤掉样本数太少的组合 (无法有效评估)
    min_combo_samples = 100
    valid_combos = [c for c in all_combos if combo_counts[c] >= min_combo_samples]
    too_small = [c for c in all_combos if combo_counts[c] < min_combo_samples]
    if too_small:
        print(f"\n样本数<{min_combo_samples}的组合 (归入训练集): {len(too_small)} 个 {too_small}")

    print(f"\n有效泵组组合: {len(valid_combos)} 个")

    # 随机划分: 80% 训练, 20% 测试
    SEED = 42
    USE_FIXED_SEED = True
    random.seed(SEED) if USE_FIXED_SEED else random.seed()
    shuffled_combos = random.sample(valid_combos, len(valid_combos))
    n_train_combos = max(1, int(len(valid_combos) * 0.8))
    train_combos = set(shuffled_combos[:n_train_combos] + too_small)
    test_combos = set(shuffled_combos[n_train_combos:])

    print(f"  训练集组合数: {len(train_combos)}")
    print(f"  测试集组合数: {len(test_combos)}")
    print(f"  测试集组合: {sorted(test_combos)}")

    train_mask = df['泵组状态'].isin(train_combos)
    test_mask = df['泵组状态'].isin(test_combos)

    train_df = df[train_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    print(f"\n训练集: {len(train_df):,} 行 ({100*len(train_df)/len(df):.1f}%)")
    print(f"测试集: {len(test_df):,} 行 ({100*len(test_df)/len(df):.1f}%)")

    # ── 训练集分层采样加速 ──
    if TRAIN_SAMPLE_SIZE and len(train_df) > TRAIN_SAMPLE_SIZE:
        print(f"\n训练集采样: {len(train_df):,} -> {TRAIN_SAMPLE_SIZE:,} (按泵组组合分层)")
        train_df = train_df.groupby('泵组状态', group_keys=False).apply(
            lambda g: g.sample(n=max(1, int(len(g) * TRAIN_SAMPLE_SIZE / len(train_df))),
                               random_state=SEED)
        ).reset_index(drop=True)
        print(f"  采样后训练集: {len(train_df):,} 行")
        # 测试集也限制上限
        max_test = TRAIN_SAMPLE_SIZE // 2
        if len(test_df) > max_test:
            test_df = test_df.sample(n=max_test, random_state=SEED).reset_index(drop=True)
            print(f"  采样后测试集: {len(test_df):,} 行")

    # 验证: 测试集的泵组组合不应出现在训练集中
    train_combos_actual = set(train_df['泵组状态'].unique())
    test_combos_actual = set(test_df['泵组状态'].unique())
    overlap = test_combos_actual & train_combos_actual
    if overlap:
        print(f"  ⚠ 警告: {len(overlap)} 个组合同时出现在训练和测试集中! {overlap}")
    else:
        print(f"  [OK] 验证通过: 测试集泵组组合均未出现在训练集中")

    # 打印测试集泵组组合详情
    print(f"\n测试集泵组组合详情:")
    for combo in sorted(test_combos_actual):
        cnt = (test_df['泵组状态'] == combo).sum()
        print(f"  {combo}: {cnt:,} 行")

    # ======== 训练 ========
    result = train_model(
        train_df, test_df,
        discrete_cols, continuous_cols,
        flow_cols, eff_col,
        engineered_cols
    )

    y_true = result['y_true']
    y_pred = result['y_pred']
    all_cols = result['all_output_cols']

    # ======== 评估: 流量 ========
    print("\n" + "=" * 60)
    print("整体评估 — 管道流量")
    print("=" * 60)
    for i in range(3):
        m = result['metrics'][all_cols[i]]
        print(f"\n【{all_cols[i]}】:")
        print(f"  MAE={m['MAE']:.2f} m3/h  RMSE={m['RMSE']:.2f}  MAPE={m['MAPE(%)']:.2f}%  R2={m['R2']:.4f}")

    m = result['metrics'].get('flowtotal(总和)', {})
    if m:
        print(f"\n【flowtotal(总和)】:")
        print(f"  MAE={m['MAE']:.2f} m3/h  RMSE={m['RMSE']:.2f}  MAPE={m['MAPE(%)']:.2f}%  R2={m['R2']:.4f}")

    # ======== 评估: 效率 ========
    print("\n" + "=" * 60)
    print("整体评估 — 总管效率")
    print("=" * 60)
    m = result['metrics'][all_cols[3]]
    print(f"\n【{all_cols[3]}】:")
    print(f"  MAE={m['MAE']:.2f} %  RMSE={m['RMSE']:.2f}  MAPE={m['MAPE(%)']:.2f}%  R2={m['R2']:.4f}")

    # ======== 按泵组组合评估 (流量) ========
    combo_metrics_flow = evaluate_by_combination(test_df, y_true[:, :3], y_pred[:, :3], list(flow_cols))

    print("\n" + "=" * 60)
    print("各泵组组合评估 — 流量 (样本数不限)")
    print("=" * 60)
    print(combo_metrics_flow.to_string(index=False))

    # ======== 保存模型 ========
    model_dir = r"D:\Wuhan_Project\models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model_v2_combo_split.pt")
    torch.save({
        'model_state_dict': result['model'].state_dict(),
        'continuous_scaler': result['continuous_scaler'],
        'output_scaler': result['output_scaler'],
        'discrete_cols': discrete_cols,
        'continuous_cols': continuous_cols,
        'flow_cols': list(flow_cols),
        'eff_col': eff_col,
        'engineered_cols': engineered_cols,
        'all_output_cols': all_cols,
        'output_dim': 4,
        'pressure_correction': {'level_baseline': 3.58, 'level_divisor': 102.0},
        'rated_specs': {'Q_m3h': RATED_Q, 'P_kW': RATED_P, 'H_m': RATED_H, 'f_Hz': RATED_F},
        'train_combos': sorted(train_combos_actual),
        'test_combos': sorted(test_combos_actual),
    }, model_path)
    print(f"\n模型权重已保存: {model_path}")

    # ======== JSON ========
    output_result = {
        'model_name': 'v6 — 纯流量+效率版 (4维输出)',
        'output_dim': 4,
        'outputs': [f'[{i}] {c}' for i, c in enumerate(all_cols)],
        'train_combos_count': len(train_combos_actual),
        'test_combos_count': len(test_combos_actual),
        'test_combos': sorted(test_combos_actual),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'training_time': result['training_time'],
        'epochs_trained': result['epochs_trained'],
        'metrics': result['metrics']
    }

    print("\n" + "=" * 60)
    print("JSON输出")
    print("=" * 60)
    print(json.dumps(output_result, ensure_ascii=False, indent=2, cls=NumpyEncoder))

    # ======== 保存指标到本地txt ========
    save_metrics_to_txt(
        metrics=result['metrics'],
        train_combos=train_combos_actual,
        test_combos=test_combos_actual,
        train_samples=len(train_df),
        test_samples=len(test_df),
        all_output_cols=all_cols,
    )
