import os
import math
import pickle
import random
from copy import deepcopy
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================================
# train_transformer_patch.py — 基于 train_transformer.py 的 PatchTST 风格升级
#
# 数据清洗 / 特征工程 / 评估管线与 train_transformer.py 完全一致,
# 仅替换模型编码端:
#   1. Patch 化 (PatchTST 思路): 输入序列先沿时间维切成不重叠的 patch
#      (48 步 → 8 个 6 步 patch), 每个 patch 展平后线性投影到 d_model 再进
#      Transformer。token 数 48 → 8, 自注意力只做 patch 间的聚合, 计算量下降;
#      patch 内局部信息由投影层一次编码, 对流量这类平滑信号相当于隐式局部平滑,
#      通常比逐时刻 token 更稳。
#   2. Pre-LN (norm_first=True): 层归一化放在自注意力/FFN 子层之前,
#      梯度流更平稳, 深层堆叠时训练更稳定, 可承受更大学习率。
#
# 输出侧仍取最后一个 patch 的表示直接映射到整个预测窗 (与 train_transformer.py
# 的 h[:, -1] 行为一致)。位置编码按 patch 数即时生成, 换 lookback 窗口无需重训。
# ============================================================================

# ==================== 异常值清洗参数 (Hampel 滤波) ====================
HAMPEL_WINDOW     = 600     # 滚动窗口 (秒, 10 分钟; 数据为秒级, 重采样之前执行)
HAMPEL_K          = 10.0    # 阈值: |x - 局部中位数| > k * scale 判为异常
MAD_FLOOR_RATIO   = 0.02    # scale 下限 = 局部中位数的 2% (防 MAD≈0 误报)
MAX_IMPUTE_ROWS   = 900     # 插值最多跨越的行数 (= 秒, 即 15 分钟); 更长的空洞不补, 保持 NaN
PUMP_GUARD_SECONDS = 300    # 泵切换前后 5 分钟内不做 Hampel 判定 (真实阶跃保护)

BASE_CONFIG = {
    "file_path": r"D:\Wuhan_Project\new_data\merged_minute_all.csv",
    "encoding": "utf-8-sig",
    "resample_freq": "30min",
    "stride": 1,
    # Total_Flow 会在清洗阶段展开为三根管逐管清洗; Target_Pressure 也补做 Hampel
    "hampel_cols": ["Total_Flow", "Target_Pressure"],

    "lookback_days": 1,                      # 回看窗口 (天)
    "predict_days": 1.0,                     # 预测窗口 (天)
    "label": "L7_P24H_30min_transformer_patch",   # 结果子目录名

    "test_days": 10,  # 测试集取最后 N 天: 必须 ≥ 回看+预测天数, 否则测试序列数为 0

    "mape_floor_ratio": 0.1,  # MAPE 过滤: 排除 |true| < 该比例 * max|true| 的点 (夜间近零流量)

    # "target_transform": "log1p" 时目标做 log1p 变换后归一化训练, 评估时 expm1 反变换回原始单位
    # (流量右偏, log 空间训练与相对误差/MAPE 对齐, 通常有改善); 默认 None = 原版行为
    "target_transform": None,

    # ── Transformer 架构超参 (Encoder 全自注意力 + PatchTST 式 Patch 嵌入) ──
    "d_model": 128,             # 嵌入 / 注意力维度
    "nhead": 4,                # 注意力头数 (需整除 d_model)
    "num_layers": 3,           # Encoder 层数
    "dim_feedforward": 256,    # 前馈层隐层维度
    "transformer_dropout": 0.2,
    "patch_len": 6,            # PatchTST: 每个 patch 的步数 (48 步 → 8 个 patch)
    "patch_stride": 6,         # patch 滑窗步长 (= patch_len → 不重叠; < patch_len → 重叠)

    # 训练超参 (Transformer 对 lr 更敏感, 降到 1e-3)
    "batch_size": 32,
    "epochs": 500,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "patience": 30,
    "min_delta": 1e-4,

    "T_0": 30,
    "T_mult": 2,

    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "base_result_dir": r"D:\Wuhan_Project\results_transformer",
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def detect_outliers(s, window=HAMPEL_WINDOW, k=HAMPEL_K, floor_ratio=MAD_FLOOR_RATIO, guard=None):
    """Hampel 滤波: 基于滚动中位数 + MAD 的稳健离群检测。

    缺陷3修复: 窗口改为 center=False, 判定 t 时刻只用 t 及之前的数据,
    不再使用未来数据 (原版 center=True 引入训练/测试边界泄漏)。
    scale = max(1.4826*MAD, floor_ratio*中位数),
    底部分数保证信号极稳定 (MAD≈0) 时仍不会把正常波动误判为异常。

    guard: 布尔 Series (True = 受保护不判异常), 用于泵切换等真实阶跃时刻。
    """
    med = s.rolling(window, center=False, min_periods=window // 2).median()
    res = (s - med).abs()
    mad = res.rolling(window, center=False, min_periods=window // 2).median()
    scale = np.maximum(1.4826 * mad, floor_ratio * med)
    flag = (res > k * scale).fillna(False)
    if guard is not None:
        flag = flag & ~guard
    return flag


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.target_cols = ["Total_Flow"]
        self.feature_cols = None

    def load_raw(self):
        file_path = self.config["file_path"]
        if file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path, encoding=self.config.get("encoding", "utf-8-sig"))
        else:
            df = pd.read_excel(file_path)

        # 武汉数据: 时间列名为 F_DateTime (秒级)
        for ts_col in ("F_DateTime", "时间"):
            if ts_col in df.columns and "timestamp" not in df.columns:
                df.rename(columns={ts_col: "timestamp"}, inplace=True)
                break

        if "timestamp" not in df.columns:
            raise ValueError("数据中必须包含 timestamp / F_DateTime / 时间 列")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        df = df.set_index("timestamp")
        return df

    def build_base_features(self, df):
        """只做列挑选/类型转换, 不求和。

        缺陷2修复: 三根管的原始流量列原样保留 (含 NaN), 由 clean_and_resample
        逐管清洗后再求和——避免单管掉线被 sum(skipna=True) 静默吞掉。
        """
        # 武汉数据: 总流量 = 170:1 + 170:2 + 70:3 (无总管流量列, 3 管求和, 不含 70:7)
        flow_cols = ["170:1_瞬时流量", "170:2_瞬时流量", "70:3_瞬时流量"]
        missing = [c for c in flow_cols if c not in df.columns]
        if missing:
            raise ValueError(f"缺少流量列: {missing}")

        # 剔除"3 管流量全 NaN"的行 (尾部 2026-07-16 晚起 ~58.7 万行整段无数据,
        # 不剔除会污染按时间的训练/测试划分——最后 10% 时间恰好落在 NaN 段上)
        all_nan = df[flow_cols].isna().all(axis=1)
        if all_nan.sum() > 0:
            df = df[~all_nan].copy()
            print(f"  剔除 3 管流量全 NaN 行: {all_nan.sum()} 条 (剩 {len(df)})")

        df = df.copy()
        for c in flow_cols:
            df[c] = df[c].astype(float)

        if "170:总管压力" in df.columns:
            df["Target_Pressure"] = df["170:总管压力"].astype(float)
        elif "总管压力1" in df.columns:
            df["Target_Pressure"] = df["总管压力1"].astype(float)
        else:
            pressure_candidates = [c for c in df.columns if "压力" in c]
            if len(pressure_candidates) == 0:
                raise ValueError("未找到压力列")
            df["Target_Pressure"] = df[pressure_candidates[0]].astype(float)

        # 武汉数据: 泵运行列为 170:1~6_泵运行 + 70:7_泵运行, 前缀不固定 → 用 "泵运行" 子串匹配
        pump_run_cols = [c for c in df.columns if "泵运行" in c]
        if len(pump_run_cols) > 0:
            pump_sum = df[pump_run_cols].sum(axis=1)
            pump_valid = df[pump_run_cols].notna().sum(axis=1)
            # 缺陷5修复: 泵列全部缺失 → NaN (缺失 ≠ 停泵), 不再静默当作 0 台泵
            df["运行泵数量"] = pump_sum.where(pump_valid > 0, np.nan)
        else:
            df["运行泵数量"] = 0.0

        out = df[flow_cols + ["Target_Pressure", "运行泵数量"]].copy()
        return out

    def clean_and_resample(self, df):
        """逐管清洗 → 求和 → 重采样。

        缺陷2/4/6/7修复: Hampel + 物理界限逐管执行 (泵切换保护), 压力补做 Hampel;
        插值仅对连续量 (流量/压力) 且设 15 分钟上限; 泵数量是状态量不做插值;
        重采样后先 dropna 删空 bin (缺陷1修复), 空洞不再被常数填充。
        """
        out = df.copy()
        flow_cols = [c for c in out.columns if "瞬时流量" in c]

        # 泵切换保护掩码: 运行泵数量变化的 ±PUMP_GUARD_SECONDS 内不判异常
        # (泵切换导致流量真实阶跃, 不应被 Hampel 抹平; 用未来泵状态只是为了"不误伤", 不向特征注入信息)
        pump_guard = pd.Series(False, index=out.index)
        if out["运行泵数量"].notna().any():
            switch = out["运行泵数量"].diff().fillna(0).ne(0)
            g = 2 * PUMP_GUARD_SECONDS + 1
            pump_guard = switch.rolling(g, center=True, min_periods=1).max().astype(bool)

        # 展开 hampel_cols: "Total_Flow" → 三根管; 其余 (如 Target_Pressure) 原样
        hampel_targets = []
        for c in self.config.get("hampel_cols", []):
            if c == "Total_Flow":
                hampel_targets += flow_cols
            elif c in out.columns:
                hampel_targets.append(c)
        hampel_targets = list(dict.fromkeys(hampel_targets))  # 去重保序

        # ── Hampel 清洗: 检出后置 NaN, 与物理界限越界值一起在下方按时间插值填补 ──
        for col in hampel_targets:
            if col not in out.columns:
                continue
            flag = detect_outliers(out[col], guard=pump_guard)
            n_out = int(flag.sum())
            if n_out > 0:
                out.loc[flag, col] = np.nan
                print(f"  {col} Hampel 离群点 {n_out} 条 ({n_out / len(out):.3%}) 置 NaN 待插值")

        # ── 物理界限裁剪 (逐管执行; 压力 0.21~0.37 MPa 实测, 界限仅作兜底) ──
        BOUNDS = {
            "Target_Pressure": (0.1, 0.5),     # MPa, 同 train.py (实测 0.21~0.37); 越界 → 插值
        }
        for col, (lo, hi) in BOUNDS.items():
            if col not in out.columns:
                continue
            s = out[col].copy()
            s[(s < lo) | (s > hi)] = np.nan
            out[col] = s
        for col in flow_cols:
            # 负值为 170:1 的垃圾读数 (最小 -6e23); 170:2 有 47710208 的尖峰 → 逐管裁剪
            s = out[col].copy()
            s[(s < 0.0) | (s > 10000.0)] = np.nan
            out[col] = s

        # ── 插值 (缺陷6/7修复): 只对连续量插值, 且设上限, 不 ffill/bfill ──
        # 泵数量是状态量, 时间线性插值会产生小数泵 (如 1.37 台), 排除在插值之外;
        # 超过 MAX_IMPUTE_ROWS 的空洞保持 NaN → 重采样后整个 bin 被删除,
        # 不会被线性插值"平滑斜坡"或常数平台伪造数据。
        impute_cols = flow_cols + ["Target_Pressure"]
        out[impute_cols] = out[impute_cols].interpolate(method="time", limit=MAX_IMPUTE_ROWS)

        # ── 逐管清洗后求和 (缺陷2核心): min_count=3 要求三管全部有效 ──
        # 任一管仍为 NaN (长空洞/无法插值) → 总量置 NaN → 该 bin 不参与训练,
        # 而不是把缺失管当 0 静默低估总流量。
        out["Total_Flow"] = out[flow_cols].sum(axis=1, min_count=len(flow_cols))

        freq = self.config["resample_freq"]
        res = pd.DataFrame(index=out.resample(freq).mean().index)

        if "Total_Flow" in out.columns:
            res["Total_Flow"] = out["Total_Flow"].resample(freq).mean()
        if "Target_Pressure" in out.columns:
            res["Target_Pressure"] = out["Target_Pressure"].resample(freq).mean()
        if "运行泵数量" in out.columns:
            # 状态量取 bin 末值 (last() 跳过 NaN); bin 内切换细节丢失是重采样固有限制
            res["运行泵数量"] = out["运行泵数量"].resample(freq).last()

        # 缺陷1修复: 先删除无数据的空 bin, 不再 ffill/bfill 常数填充。
        # (原版 ffill().bfill().dropna() 中 dropna 是死代码——空洞会被填成常数平台。
        #  时间轴留下空洞是刻意的: 滚动特征可能跨洞计算, 但优于虚构数据。)
        res = res.dropna()
        return res

    def validate_cumulative_flow(self, df_raw, df_base):
        """交叉校验: 用累计流量列 LJflowtotal 的增量核对瞬时流量总和的量级。

        只打印统计, 不改数据。假设: 累计流量单位 m³, 瞬时流量单位 m³/h,
        则 1 小时累计增量 ≈ 小时均流量 × 3600。累计表可能翻转归零 (负增量)
        或含垃圾值, 一律跳过, 只统计"偏差 > 25% 的窗口"占比。
        """
        if "LJflowtotal" not in df_raw.columns or "Total_Flow" not in df_base.columns:
            return
        c = df_raw["LJflowtotal"].astype(float)
        c = c.where((c > 0) & (c < 1e9))                    # 垃圾值: 负值/超大值 → NaN
        inc = c.resample("1h").last().diff()                # 每小时累计增量 (m³)
        vol = df_base["Total_Flow"].resample("1h").mean() * 3600   # 小时量 (m³/h × h)
        cmp = pd.concat([inc.rename("inc"), vol.rename("vol")], axis=1).dropna()
        cmp = cmp[(cmp["inc"] > 0) & (cmp["inc"] < 5e5) & (cmp["vol"] > 0)]
        if len(cmp) == 0:
            print("  [交叉校验] 无可比窗口 (累计流量单位可能不匹配, 忽略)")
            return
        ratio = (cmp["inc"] - cmp["vol"]).abs() / cmp["vol"]
        n_bad = int((ratio > 0.25).sum())
        print(f"  [交叉校验] 累计流量增量 vs 瞬时流量小时量: {len(cmp)} 窗口, "
              f"偏差>25% 的 {n_bad} 个 ({n_bad / len(cmp):.2%})")

    def add_time_features(self, df):
        out = df.copy()
        out["hour"] = out.index.hour
        out["dayofweek"] = out.index.dayofweek
        out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
        out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
        out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
        out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
        out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
        return out

    def add_lag_rolling_features(self, df):
        out = df.copy()

        base_vars = ["Total_Flow", "Target_Pressure"]

        # 物理时间窗口 (分钟) → 按当前 resample_freq 折算步数:
        # 原 15min 口径 lags=[1,6,12,24,48,96] 即 15min~24h、rolls=[6,12,24,48,96] 即 1.5h~24h,
        # trend_6/24 即 90min/6h。改为按分钟定义, 换重采样频率不改变特征物理含义。
        freq_minutes = int(self.config["resample_freq"].replace("min", ""))
        ppd = (24 * 60) // freq_minutes                     # 每24小时点数 (288 @5min)
        lags = sorted({max(1, m // freq_minutes) for m in [15, 90, 180, 360, 720, 1440]})
        rolls = sorted({max(1, m // freq_minutes) for m in [90, 180, 360, 720, 1440]})
        trend_mins = {"trend_90m": 90, "trend_6h": 360}     # 原 trend_6 / trend_24

        for col in base_vars:
            if col not in out.columns:
                continue
            for lag in lags:
                out[f"{col}_lag_{lag}"] = out[col].shift(lag)
            for d in range(1, 7):
                out[f"{col}_diff_{d}"] = out[col].diff(d)
            out[f"{col}_acc"] = out[col].diff(1).diff(1)
            for w in rolls:
                out[f"{col}_roll_mean_{w}"] = out[col].rolling(w).mean()
                out[f"{col}_roll_std_{w}"] = out[col].rolling(w).std()
                out[f"{col}_roll_min_{w}"] = out[col].rolling(w).min()
                out[f"{col}_roll_max_{w}"] = out[col].rolling(w).max()
            for name, win_min in trend_mins.items():
                win = max(1, win_min // freq_minutes)
                out[f"{col}_{name}"] = out[col] - out[col].rolling(win).mean()

        if "Total_Flow" in out.columns:
            out["flow_lag_1day"] = out["Total_Flow"].shift(ppd)      # 原硬编码 48/96 是 15min 专属(12h/24h), 现按真实 1d/2d
            out["flow_lag_2day"] = out["Total_Flow"].shift(2 * ppd)
        if "Target_Pressure" in out.columns:
            out["pressure_lag_1day"] = out["Target_Pressure"].shift(ppd)
            out["pressure_lag_2day"] = out["Target_Pressure"].shift(2 * ppd)
        if "Total_Flow" in out.columns and "Target_Pressure" in out.columns:
            out["flow_pressure_ratio"] = out["Total_Flow"] / (out["Target_Pressure"] + 1e-3)
            out["flow_pressure_diff"] = out["Total_Flow"] - out["Target_Pressure"]
        if "运行泵数量" in out.columns:
            # 缺陷8修复: 泵 lag 按分钟折算 (原 shift(1)/shift(6) 是步数, 换频率后物理含义漂移)
            for m in [15, 90]:
                out[f"运行泵数量_lag_{m}min"] = out["运行泵数量"].shift(max(1, m // freq_minutes))
            out["pump_change"] = out["运行泵数量"].diff(1)
            if "Total_Flow" in out.columns:
                out["flow_per_pump"] = out["Total_Flow"] / (out["运行泵数量"] + 1e-3)
                out["flow_mul_pump"] = out["Total_Flow"] * out["运行泵数量"]
        if "Total_Flow" in out.columns:
            # 原 flow_volatility_6/24 = 90min/6h 波动率
            out["flow_volatility_90m"] = out["Total_Flow"].rolling(max(1, 90 // freq_minutes)).std()
            out["flow_volatility_6h"] = out["Total_Flow"].rolling(max(1, 360 // freq_minutes)).std()
        if "hour" in out.columns:
            out["is_morning_peak"] = out["hour"].isin([6, 7, 8, 9]).astype(int)
            out["is_evening_peak"] = out["hour"].isin([17, 18, 19, 20]).astype(int)
            out["is_night"] = out["hour"].isin([0, 1, 2, 3, 4]).astype(int)

        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.dropna()
        return out

    def build_feature_table(self):
        df_raw = self.load_raw()
        df_base = self.build_base_features(df_raw)
        self.validate_cumulative_flow(df_raw, df_base)

        # 缺陷3修复: 先按时间切分, 再分别清洗/插值 → 插值不跨越训练/测试边界。
        # 滞后/滚动特征仍在拼接后的全量表上计算 (因果特征, 无泄漏)。
        test_days = self.config.get("test_days", 15)
        test_start = df_base.index[-1] - pd.Timedelta(days=test_days)
        df_base_train = df_base.loc[df_base.index <= test_start].copy()
        df_base_test = df_base.loc[df_base.index > test_start].copy()
        print(f"  清洗分段: 训练段 {df_base_train.index.min()} ~ {df_base_train.index.max()} "
              f"({len(df_base_train)} 行), 测试段 {df_base_test.index.min()} ~ {df_base_test.index.max()} "
              f"({len(df_base_test)} 行), 分段清洗互不跨界")

        df_clean = pd.concat([self.clean_and_resample(df_base_train),
                              self.clean_and_resample(df_base_test)])

        df_time = self.add_time_features(df_clean)
        df_feat = self.add_lag_rolling_features(df_time)
        self.feature_cols = df_feat.columns.tolist()
        return df_feat

    def split_by_time(self, df):
        """按时间划分训练/测试集: 测试集 = 最后 test_days 天。

        不用固定比例的原因: 原 90/10 切分下测试段仅 ~3.3 天 (956 行@5min),
        而最长序列窗口 (L5_P1D = 1440+288 步 = 6 天) 比测试段还长 → 测试序列数为 0。
        按天数切分保证测试段长度确定、所有实验都能生成测试序列。
        """
        test_days = self.config.get("test_days", 7)
        test_start = df.index[-1] - pd.Timedelta(days=test_days)
        df_test = df.loc[df.index > test_start].copy()
        df_train = df.loc[df.index <= test_start].copy()
        return df_train, df_test

    def fit_scalers(self, df_train):
        self.feature_scaler.fit(df_train[self.feature_cols].values)
        target_raw = df_train[self.target_cols].values.astype(np.float64)
        if self.config.get("target_transform") == "log1p":
            target_raw = np.log1p(target_raw)
        self.target_scaler.fit(target_raw)

    def transform_df(self, df):
        X = self.feature_scaler.transform(df[self.feature_cols].values)
        target_raw = df[self.target_cols].values.astype(np.float64)
        if self.config.get("target_transform") == "log1p":
            target_raw = np.log1p(target_raw)
        Y = self.target_scaler.transform(target_raw)
        return X, Y

    def inverse_transform_targets(self, arr):
        arr = np.asarray(arr)
        if arr.size == 0:
            return arr
        if arr.ndim == 2:
            inv = self.target_scaler.inverse_transform(arr)
        elif arr.ndim == 3:
            shape = arr.shape
            flat = arr.reshape(-1, shape[-1])
            inv = self.target_scaler.inverse_transform(flat).reshape(shape)
        else:
            raise ValueError("只支持2维或3维数组反归一化")
        if self.config.get("target_transform") == "log1p":
            inv = np.expm1(inv)   # 反变换回原始流量单位 (评估指标口径与 baseline 一致)
        return inv

    def make_sequences(self, x_array, y_array, lookback_days, predict_days):
        """支持自定义 lookback/predict 天数的序列生成"""
        freq_minutes = int(self.config["resample_freq"].replace("min", ""))
        points_per_day = (24 * 60) // freq_minutes

        lookback_steps = int(lookback_days * points_per_day)
        horizon_steps = int(predict_days * points_per_day)
        stride = self.config["stride"]

        total_len = len(x_array)
        n_samples = (total_len - lookback_steps - horizon_steps) // stride + 1
        if n_samples <= 0:
            return np.empty((0, lookback_steps, x_array.shape[1]), dtype=np.float32), \
                   np.empty((0, horizon_steps, y_array.shape[1]), dtype=np.float32)

        # 预分配 + 直填: 原"列表收集→np.array 复制"在构建期会有两份序列数组并存,
        # 5min 序列可达 3~4GB/份, 13.9GB 内存机上峰值翻倍有 OOM 风险
        X = np.empty((n_samples, lookback_steps, x_array.shape[1]), dtype=np.float32)
        Y = np.empty((n_samples, horizon_steps, y_array.shape[1]), dtype=np.float32)

        idx = 0
        for i in range(0, total_len - lookback_steps - horizon_steps + 1, stride):
            X[idx] = x_array[i:i + lookback_steps]
            Y[idx] = y_array[i + lookback_steps:i + lookback_steps + horizon_steps]
            idx += 1

        return X, Y


# ==================== 模型定义 ====================

class SeqDataset(Dataset):
    def __init__(self, X, Y):
        # from_numpy 共享内存, 不做双份拷贝——5min 序列数组可达 3~4GB, 13.9GB 内存机必须省
        # (make_sequences 输出已为 float32 连续数组)
        self.X = torch.from_numpy(np.ascontiguousarray(X))
        self.Y = torch.from_numpy(np.ascontiguousarray(Y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def build_sinusoid_pe(seq_len, d_model):
    """正弦位置编码 (Vaswani et al. 2017): 偶维 sin / 奇维 cos, 频率按 10000^(-2k/d) 递减。

    与窗口长度解耦: 任意 seq_len 即时生成, 换 lookback 窗口长度无需重训
    (可学习位置编码的长度与训练窗口绑定, 做不到这一点)。Patch 化后 seq_len
    为 patch 个数 (48 步 / 6 步每 patch = 8), 位置编码加在 patch 位置而非时刻上。
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
    """PatchTST 风格 Transformer Encoder 多步预测模型。

    前向接口与 LSTM 版一致 (src, target_len, tgt, teacher_forcing_ratio),
    以便共用 evaluate / 训练循环; tgt 与 teacher_forcing 不使用
    (直接多步输出, 无自回归)。

    结构 (相对 train_transformer.py 的改动点):
      沿时间维切 patch (patch_len 步/patch) → 展平后 Linear 投影到 d_model
      + 正弦位置编码 (加在 patch 位置, 与窗口长度解耦)
      + N 层 Pre-LN Transformer Encoder (norm_first=True, 归一化在子层前,
        梯度更稳) 自注意力聚合 patch 间信息
      + 最后一个 patch 的表示 → Linear(d_model → horizon * output_dim) 直接多步输出

    Patch 化收益: token 数从 L 降到 L/patch_len (48 → 8), 注意力计算量下降;
    patch 内局部模式 (流量平滑段) 由投影层一次性编码, 无需逐时刻注意力。
    """
    def __init__(self, input_dim, output_dim, horizon, input_len,
                 d_model=64, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1,
                 patch_len=6, patch_stride=6):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        # patch 嵌入: 每个 patch 的 patch_len*input_dim 维扁平向量 → d_model
        # (input_len 仅用于显式声明支持的最大输入长度, 实际按输入即时切 patch)
        self.patch_proj = nn.Linear(patch_len * input_dim, d_model)

        # Pre-LN: norm_first=True → 归一化在多头注意力/FFN 之前
        # (默认 post-LN 的梯度在深层堆叠下更易发散, Pre-LN 更稳、可承受更大 lr)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="relu", norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_dim = output_dim
        self.horizon = horizon
        self.head = nn.Linear(d_model, horizon * output_dim)

    def _patchify(self, x):
        """(B, L, D) → (B, n_patches, patch_len*D): 沿时间维滑窗切 patch。

        长度不足时尾部补零 (仅影响最后一个不完整 patch); 不重叠时
        n_patches = ceil(L / patch_len)。unfold 实现, 无显式循环。
        """
        B, L, D = x.shape
        pad = (self.patch_len - (L - self.patch_len) % self.patch_stride) % self.patch_stride
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))          # 时间维 (倒数第2维) 尾部补零
        x = x.contiguous().unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        x = x.permute(0, 1, 3, 2).contiguous()    # (B, n, patch_len, D)
        return x.reshape(B, x.size(1), self.patch_len * D)

    def forward(self, src, target_len, tgt=None, teacher_forcing_ratio=0.0):
        x = self._patchify(src)                              # (B, n_patches, patch_len*input_dim)
        pos = build_sinusoid_pe(x.size(1), self.patch_proj.out_features).to(src.device)
        h = self.patch_proj(x) + pos                         # (B, n_patches, d_model)
        h = self.encoder(h)
        h = h[:, -1]                                   # 最后一个 patch (最接近预测起点, 自注意力已聚合全窗口)
        out = self.head(h).view(h.size(0), -1, self.output_dim)
        return out[:, :target_len]


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.stop = False

    def step(self, val):
        if self.best is None:
            self.best = val
            return False
        if val < self.best - self.min_delta:
            self.best = val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ==================== 工具函数 ====================

def weighted_mse_loss(pred, target, flow_weight=1.0):
    flow_loss = ((pred[:, :, 0] - target[:, :, 0]) ** 2).mean()
    return flow_weight * flow_loss


def compute_mape(y_true, y_pred, floor_ratio=0.05):
    """MAPE (%), 排除 |true| < floor_ratio * max|true| 的点。

    夜间近零流量 (几 m³/h) 使分母趋近 0, 相对误差虚高并主导整体 MAPE;
    按最大流量的比例设下限, 只统计有意义的点。返回 (mape, n_total, n_used)。
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    n_total = len(y_true)
    if n_total == 0:
        return 0.0, 0, 0
    thr = floor_ratio * np.abs(y_true).max()
    mask = np.abs(y_true) >= thr
    n_used = int(mask.sum())
    if n_used == 0:
        return 0.0, n_total, 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + 1e-8))) * 100
    return mape, n_total, n_used


def evaluate(model, loader, device, processor):
    model.eval()
    total_loss = 0.0
    all_preds, all_trues = [], []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x, target_len=batch_y.size(1), tgt=None, teacher_forcing_ratio=0.0)
            loss = weighted_mse_loss(pred, batch_y)
            total_loss += loss.item() * len(batch_x)
            all_preds.append(pred.cpu().numpy())
            all_trues.append(batch_y.cpu().numpy())

    if len(all_preds) == 0:
        return {"loss": 0, "flow_mae": 0, "flow_rmse": 0, "flow_mape": 0,
                "y_pred_inv": np.array([]), "y_true_inv": np.array([])}

    avg_loss = total_loss / len(loader.dataset)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_trues, axis=0)
    y_pred_inv = processor.inverse_transform_targets(y_pred)
    y_true_inv = processor.inverse_transform_targets(y_true)

    y_true_flat = y_true_inv[:, :, 0].reshape(-1)
    y_pred_flat = y_pred_inv[:, :, 0].reshape(-1)
    flow_mae = mean_absolute_error(y_true_flat, y_pred_flat)
    flow_rmse = math.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    # MAPE 过滤夜间近零流量点: |true| < floor_ratio * max|true| 不参与 (见 compute_mape)
    flow_mape, mape_n_total, mape_n_used = compute_mape(
        y_true_flat, y_pred_flat, processor.config.get("mape_floor_ratio", 0.05))

    return {
        "loss": avg_loss, "flow_mae": flow_mae, "flow_rmse": flow_rmse,
        "flow_mape": flow_mape, "mape_n_used": mape_n_used, "mape_n_total": mape_n_total,
        "y_pred_inv": y_pred_inv, "y_true_inv": y_true_inv
    }


def plot_best_worst_cases(y_true_inv, y_pred_inv, save_dir, num_best=30, num_worst=30, title_prefix="Test"):
    n_samples = len(y_true_inv)
    if n_samples == 0:
        return

    per_sample_mae = np.mean(np.abs(y_true_inv - y_pred_inv), axis=(1, 2))
    sorted_idx = np.argsort(per_sample_mae)
    best_idx = sorted_idx[:min(num_best, n_samples)]
    worst_idx = sorted_idx[-min(num_worst, n_samples):][::-1]

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    ncols = 5

    def draw_grid(indices, nrows, tag, save_filename):
        nrows = max(nrows, 1)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.2))
        axes = np.atleast_2d(axes)
        for i, idx in enumerate(indices):
            r, c = divmod(i, ncols)
            ax = axes[r, c]
            ax.plot(y_true_inv[idx, :, 0], color='#2c3e50', linewidth=1.2, label='True')
            ax.plot(y_pred_inv[idx, :, 0], color='#e74c3c', linewidth=1.2, linestyle='--', label='Pred')
            ax.set_title(f"#{i+1} MAE={per_sample_mae[idx]:.0f}", fontsize=9)
            ax.grid(alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc='upper right')
        for j in range(len(indices), nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r, c].set_visible(False)
        fig.suptitle(f"{title_prefix} - {tag} ({len(indices)} cases)", fontsize=16, fontweight='bold', y=1.01)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, save_filename), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"    {tag}图已保存: {os.path.join(save_dir, save_filename)}")

    nrows_best = int(np.ceil(len(best_idx) / ncols))
    nrows_worst = int(np.ceil(len(worst_idx) / ncols))
    draw_grid(best_idx, nrows_best, "Best", f"{title_prefix}_best_cases.png")
    draw_grid(worst_idx, nrows_worst, "Worst", f"{title_prefix}_worst_cases.png")


def plot_error_distribution(y_true_inv, y_pred_inv, save_path, title_prefix="Test"):
    if len(y_true_inv) == 0:
        return
    y_true_flat = y_true_inv[:, :, 0].reshape(-1)
    y_pred_flat = y_pred_inv[:, :, 0].reshape(-1)
    abs_errors = np.abs(y_true_flat - y_pred_flat)
    relative_errors = np.where(y_true_flat != 0, abs_errors / np.abs(y_true_flat), 0)

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].hist(abs_errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(abs_errors), color='red', linestyle='--', label=f'Mean AE: {np.mean(abs_errors):.2f}')
    axes[0].axvline(np.median(abs_errors), color='green', linestyle='--', label=f'Median AE: {np.median(abs_errors):.2f}')
    axes[0].set_title(f"{title_prefix} Absolute Error Distribution")
    axes[0].set_xlabel("Absolute Error")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    mask = relative_errors < 0.5
    rel_err_display = relative_errors[mask]
    axes[1].hist(rel_err_display * 100, bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(rel_err_display) * 100, color='red', linestyle='--', label=f'Mean RE: {np.mean(rel_err_display)*100:.2f}%')
    axes[1].axvline(np.median(rel_err_display) * 100, color='green', linestyle='--', label=f'Median RE: {np.mean(rel_err_display)*100:.2f}%')
    axes[1].set_title(f"{title_prefix} Relative Error Distribution (<50%)")
    axes[1].set_xlabel("Relative Error (%)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    误差分布图已保存: {save_path}")


def stat_by_start_time(y_true_inv, y_pred_inv, start_times, save_dir, floor_ratio=0.05):
    """按预测起点时刻 (0:00~23:30, 30 分钟间隔) 统计测试集 MAE/RMSE/MAPE。

    start_times: 与 y_true_inv 样本一一对应的预测起点时间 (DatetimeIndex 子集)。
    15min 数据的起点落在 :00/:15/:30/:45, 归入最近半小时槽。
    MAPE 用整个测试集的全局阈值过滤近零流量点 (|true| < floor_ratio * max|true|),
    保证各槽位过滤口径一致。
    """
    n = len(y_true_inv)
    if n == 0 or start_times is None or len(start_times) != n:
        print("    ⚠ 样本数与起点时间不匹配, 跳过按起点时刻统计")
        return

    # 起点 → 30 分钟槽 (0~47, 对应 0:00~23:30)
    slots = np.array([t.hour * 2 + (1 if t.minute >= 30 else 0) for t in start_times])
    y_true = y_true_inv[:, :, 0]
    y_pred = y_pred_inv[:, :, 0]

    # 全局阈值 (整个测试集), 各槽位过滤口径一致
    thr = floor_ratio * np.abs(y_true).max()

    rows = []
    for s in range(48):
        mask = slots == s
        n_slot = int(mask.sum())
        if n_slot == 0:
            continue
        tt = y_true[mask].reshape(-1)
        pp = y_pred[mask].reshape(-1)
        mae = mean_absolute_error(tt, pp)
        rmse = math.sqrt(mean_squared_error(tt, pp))
        m_map = np.abs(tt) >= thr
        mape = (np.mean(np.abs((tt[m_map] - pp[m_map]) / (tt[m_map] + 1e-8))) * 100
                if m_map.sum() > 0 else 0.0)
        rows.append({
            "start_time": f"{s // 2:02d}:{30 if s % 2 else 0:02d}",
            "n_samples": n_slot, "mae": mae, "rmse": rmse, "mape": mape,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, "test_start_time_metrics.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"    按起点时刻精度统计已保存: {csv_path}")
    print(f"    (MAPE 过滤阈值 {thr:.2f}, 排除 |true| < 阈值 的点)")
    print("    起点时刻  样本数      MAE      RMSE     MAPE%")
    for r in rows:
        print(f"    {r['start_time']}  {r['n_samples']:<6}  "
              f"{r['mae']:<8.2f}  {r['rmse']:<8.2f}  {r['mape']:<8.2f}")


# ==================== 单次实验运行 ====================

def run_experiment(cfg, x_train_all, y_train_all, x_test_all, y_test_all, processor, device, test_index=None):
    """
    按合并后的 cfg (BASE_CONFIG 被实验配置覆盖) 运行一次完整的训练+评估
    返回 metrics 字典
    test_index: 测试段时间索引 (DatetimeIndex), 用于按预测起点时刻统计精度
    """
    processor.config = cfg   # 序列构建/时间划分按本实验参数
    lookback = cfg["lookback_days"]
    predict = cfg["predict_days"]
    label = cfg["label"]

    # 结果子目录
    result_dir = os.path.join(cfg["base_result_dir"], label)
    ensure_dir(result_dir)

    print(f"\n{'='*80}")
    print(f" 实验: {label}  |  lookback={lookback}d  |  predict={predict}d"
          f"  |  freq={cfg['resample_freq']}  |  test_days={cfg['test_days']}"
          f"  |  d_model={cfg['d_model']}  |  nhead={cfg['nhead']}"
          f"  |  layers={cfg['num_layers']}  |  patch_len={cfg.get('patch_len', 6)}"
          f"  |  lr={cfg['learning_rate']}")
    print(f" 结果目录: {result_dir}")
    print(f"{'='*80}")

    # 构建序列 (不同 lookback/predict → 不同步数)
    X_train, Y_train = processor.make_sequences(x_train_all, y_train_all, lookback, predict)
    X_test, Y_test = processor.make_sequences(x_test_all, y_test_all, lookback, predict)

    # 计算 predict steps 用于显示
    freq_minutes = int(cfg["resample_freq"].replace("min", ""))
    points_per_day = (24 * 60) // freq_minutes
    predict_steps = int(predict * points_per_day)

    # 每个测试样本的预测起点时刻 = 测试段第 lookback_steps + idx*stride 行的时间
    # (样本 idx 的回看窗口从 idx*stride 行开始, 预测从 +lookback_steps 行开始)
    stride = cfg.get("stride", 1)
    lookback_steps = int(lookback * points_per_day)
    if test_index is not None:
        test_starts = test_index[lookback_steps:len(X_test) * stride + lookback_steps:stride]
    else:
        test_starts = None

    print(f"  {cfg['resample_freq']}频率: lookback={int(lookback * points_per_day)}步, predict={predict_steps}步")
    print(f"  X_train={X_train.shape}, Y_train={Y_train.shape}")
    print(f"  X_test={X_test.shape}, Y_test={Y_test.shape}")

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"  ⚠ 样本数为0，跳过此配置")
        return None

    train_loader = DataLoader(SeqDataset(X_train, Y_train), batch_size=cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(SeqDataset(X_test, Y_test), batch_size=cfg["batch_size"], shuffle=False)

    model = TimeSeriesTransformer(
        input_dim=X_train.shape[2],
        output_dim=1,
        horizon=predict_steps,
        input_len=lookback_steps,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["transformer_dropout"],
        patch_len=cfg.get("patch_len", 6),
        patch_stride=cfg.get("patch_stride", 6),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg["T_0"], T_mult=cfg["T_mult"])
    early_stopper = EarlyStopping(cfg["patience"], cfg["min_delta"])

    best_state = None
    best_test_loss = float("inf")
    history = []

    print(f"\n  Training: {label}")
    print(f"  {'Epoch':<8}{'TrainLoss':<15}{'TestLoss':<15}{'FlowMAE':<12}{'FlowRMSE':<12}{'FlowMAPE':<12}{'LR':<12}")
    print(f"  {'-'*90}")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_loss_sum = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x, target_len=batch_y.size(1))   # 直接多步输出, 无需 teacher forcing
            loss = weighted_mse_loss(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(batch_x)

        train_loss = train_loss_sum / len(train_loader.dataset)
        test_metrics = evaluate(model, test_loader, device, processor)
        test_loss = test_metrics["loss"]
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch, "train_loss": train_loss, "test_loss": test_loss,
            "flow_mae": test_metrics["flow_mae"], "flow_rmse": test_metrics["flow_rmse"],
            "flow_mape": test_metrics["flow_mape"], "lr": lr_now
        })

        print(f"  {epoch:<8}{train_loss:<15.6f}{test_loss:<15.6f}{test_metrics['flow_mae']:<12.4f}{test_metrics['flow_rmse']:<12.4f}{test_metrics['flow_mape']:<12.2f}{lr_now:<12.6f}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(result_dir, "best_seq2seq_model.pth"))

        if early_stopper.step(test_loss):
            print(f"\n  Early stopping at epoch={epoch}")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(result_dir, "train_history.csv"), index=False)

    if best_state is not None:
        model.load_state_dict(best_state)

    # 保存 scaler / 特征列 / 配置, 供推理脚本加载 (推理需要与训练完全一致的特征工程)
    scaler_path = os.path.join(result_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump({
            "config": cfg,
            "feature_scaler": processor.feature_scaler,
            "target_scaler": processor.target_scaler,
            "feature_cols": processor.feature_cols,
            "target_cols": processor.target_cols,
        }, f)
    print(f"  推理用 scaler/特征配置已保存: {scaler_path}")

    # 最终评估
    train_metrics = evaluate(model, train_loader, device, processor)
    test_metrics = evaluate(model, test_loader, device, processor)

    print(f"\n  最终结果:")
    print(f"  Train: Loss={train_metrics['loss']:.6f}, MAE={train_metrics['flow_mae']:.2f}, "
          f"RMSE={train_metrics['flow_rmse']:.2f}, MAPE={train_metrics['flow_mape']:.2f}%")
    print(f"  Test : Loss={test_metrics['loss']:.6f}, MAE={test_metrics['flow_mae']:.2f}, "
          f"RMSE={test_metrics['flow_rmse']:.2f}, MAPE={test_metrics['flow_mape']:.2f}%")
    mape_floor = cfg.get("mape_floor_ratio", 0.05)
    print(f"  (MAPE 已过滤 |true| < {mape_floor:.0%}*max 的近零流量点: "
          f"Train 保留 {train_metrics['mape_n_used']}/{train_metrics['mape_n_total']}, "
          f"Test 保留 {test_metrics['mape_n_used']}/{test_metrics['mape_n_total']})")

    with open(os.path.join(result_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        for name, m in [("Train", train_metrics), ("Test", test_metrics)]:
            f.write(f"{name}: Loss={m['loss']:.6f}, MAE={m['flow_mae']:.2f}, "
                    f"RMSE={m['flow_rmse']:.2f}, MAPE={m['flow_mape']:.2f}%\n")
        f.write(f"MAPE 过滤: 排除 |true| < {mape_floor:.0%} * max|true| 的点 "
                f"(Train 保留 {train_metrics['mape_n_used']}/{train_metrics['mape_n_total']} 点, "
                f"Test 保留 {test_metrics['mape_n_used']}/{test_metrics['mape_n_total']} 点)\n")

    # 画图 (仅测试集)
    y_true_test = test_metrics["y_true_inv"]
    y_pred_test = test_metrics["y_pred_inv"]

    if len(y_true_test) > 0:
        plot_best_worst_cases(y_true_test, y_pred_test, result_dir, num_best=30, num_worst=30, title_prefix="Test")
        plot_error_distribution(y_true_test, y_pred_test,
                                os.path.join(result_dir, "test_error_distribution.png"), title_prefix="Test")

        mae_per_step = np.mean(np.abs(y_true_test - y_pred_test), axis=0)
        step_mae_df = pd.DataFrame({"step": np.arange(len(mae_per_step)), "flow_mae": mae_per_step[:, 0]})
        step_mae_df.to_csv(os.path.join(result_dir, "test_step_mae.csv"), index=False)

        # 按预测起点时刻 (0:00~23:30) 统计测试集精度
        stat_by_start_time(y_true_test, y_pred_test, test_starts, result_dir,
                           floor_ratio=cfg.get("mape_floor_ratio", 0.05))

    # loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["test_loss"], label="Test Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Training Curve — {label}")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "loss_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 返回汇总指标
    return {
        "label": label,
        "resample_freq": cfg["resample_freq"],
        "test_days": cfg["test_days"],
        "d_model": cfg["d_model"],
        "nhead": cfg["nhead"],
        "num_layers": cfg["num_layers"],
        "patch_len": cfg.get("patch_len", 6),
        "patch_stride": cfg.get("patch_stride", 6),
        "learning_rate": cfg["learning_rate"],
        "lookback_days": lookback,
        "predict_days": predict,
        "predict_steps": predict_steps,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "best_epoch": int(history_df.loc[history_df["test_loss"].idxmin(), "epoch"]),
        "train_loss": train_metrics["loss"],
        "test_loss": test_metrics["loss"],
        "train_mae": train_metrics["flow_mae"],
        "test_mae": test_metrics["flow_mae"],
        "train_rmse": train_metrics["flow_rmse"],
        "test_rmse": test_metrics["flow_rmse"],
        "train_mape": train_metrics["flow_mape"],
        "test_mape": test_metrics["flow_mape"],
    }


# ==================== 主入口 ====================

def main():
    config = BASE_CONFIG
    set_seed(config["seed"])

    device = torch.device(config["device"])
    print(f"Device: {device}")
    print(f"结果目录: {os.path.join(config['base_result_dir'], config['label'])}")

    # ============ 第一步: 数据加载 & 特征工程 (按 config 的 resample_freq) ============
    print("\n" + "=" * 80)
    print(f" [Phase 1] 数据加载 & 特征工程 (resample_freq={config['resample_freq']})")
    print("=" * 80)

    processor = DataProcessor(config)
    print(" 正在加载并处理数据...")
    df_all_feat = processor.build_feature_table()
    print(f" 全量特征表: {df_all_feat.shape}")
    print(f" 时间范围: {df_all_feat.index.min()} ~ {df_all_feat.index.max()}")

    # ============ 第二步: 按时间划分训练/测试集 + 归一化 ============
    print("\n" + "=" * 80)
    print(" [Phase 2] 按时间划分训练/测试集")
    print("=" * 80)

    df_train_feat, df_test_feat = processor.split_by_time(df_all_feat)
    print(f" 训练集: {df_train_feat.shape} | {df_train_feat.index.min()} ~ {df_train_feat.index.max()}")
    print(f" 测试集: {df_test_feat.shape} | {df_test_feat.index.min()} ~ {df_test_feat.index.max()}")

    processor.fit_scalers(df_train_feat)
    x_train_all, y_train_all = processor.transform_df(df_train_feat)
    x_test_all, y_test_all = processor.transform_df(df_test_feat)

    # ============ 第三步: 训练 & 评估 ============
    result = run_experiment(config, x_train_all, y_train_all,
                            x_test_all, y_test_all,
                            processor, device, test_index=df_test_feat.index)
    if result is not None:
        print(f"\n 训练完成! 结果保存在: {os.path.join(config['base_result_dir'], config['label'])}")
    else:
        print(f"\n ⚠ 样本数为 0, 未产生结果 (请检查 test_days 是否 ≥ 回看+预测天数)")


if __name__ == "__main__":
    main()
