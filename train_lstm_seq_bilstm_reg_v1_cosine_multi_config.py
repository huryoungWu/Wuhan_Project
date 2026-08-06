import os
import math
import random
from copy import deepcopy
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==================== 异常值清洗参数 (Hampel 滤波, 同 plot_daily_flow_curves.py) ====================
HAMPEL_WINDOW   = 600    # 滚动窗口 (秒, 10 分钟; 在秒级数据上执行, 重采样之前)
HAMPEL_K        = 10.0   # 阈值: |x - 局部中位数| > k * scale 判为异常
MAD_FLOOR_RATIO = 0.02   # scale 下限 = 局部中位数的 2% (防 MAD≈0 误报)

# ==================== 实验配置列表 (20 组) ====================
# 每项覆盖 BASE_CONFIG 的对应键; label 需唯一 (作为结果子目录名)。
# test_days 必须 ≥ 回看+预测天数, 否则测试序列数为 0
# (数据有效跨度约 34.5 天: 41 天去掉尾部 6.8 天 NaN 段)。
EXPERIMENTS = [
    # ── 回看/预测窗口 × 重采样频率 (基准: dropout 0.0, hidden 64, layers 2, test 15) ──
  
    {"lookback_days": 7,  "predict_days": 0.5,  "resample_freq": "5min",  "test_days": 8,  "dropout": 0.0, "hidden_dim": 64, "label": "L7_P12H_5min"},
    {"lookback_days": 7,  "predict_days": 0.5,  "resample_freq": "5min",  "test_days": 8,  "dropout": 0.2, "hidden_dim": 64, "label": "L7_P12H_5min_d02"},
    {"lookback_days": 7,  "predict_days": 0.25, "resample_freq": "5min",  "test_days": 8,  "dropout": 0.0, "hidden_dim": 64, "label": "L7_P6H_5min"},
    {"lookback_days": 7,  "predict_days": 0.5,  "resample_freq": "10min", "test_days": 8,  "dropout": 0.0, "hidden_dim": 64, "label": "L7_P12H_10min"},
    {"lookback_days": 7,  "predict_days": 0.25, "resample_freq": "10min", "test_days": 8,  "dropout": 0.0, "hidden_dim": 64, "label": "L7_P6H_10min"},
    {"lookback_days": 7,  "predict_days": 0.25, "resample_freq": "15min", "test_days": 8,  "dropout": 0.0, "hidden_dim": 64, "label": "L7_P6H_15min"},

    # ── 预测窗口变化 ──
    {"lookback_days": 7,  "predict_days": 0.125, "resample_freq": "5min",  "test_days": 8,  "dropout": 0.0, "hidden_dim": 64, "label": "L7_P3H_5min"},
    {"lookback_days": 7,  "predict_days": 0.75,  "resample_freq": "5min",  "test_days": 8,  "dropout": 0.0, "hidden_dim": 64, "label": "L7_P18H_5min"},
    {"lookback_days": 7,  "predict_days": 1.0,   "resample_freq": "5min",  "test_days": 8,  "dropout": 0.0, "hidden_dim": 64, "label": "L7_P24H_5min"},
    {"lookback_days": 7,  "predict_days": 0.375, "resample_freq": "5min",  "test_days": 8,  "dropout": 0.0, "hidden_dim": 64, "label": "L7_P9H_5min"},
    {"lookback_days": 7,  "predict_days": 0.625, "resample_freq": "5min",  "test_days": 8,  "dropout": 0.0, "hidden_dim": 64, "label": "L7_P15H_5min"},

    # ── 重采样频率变化 ──
    {"lookback_days": 7,  "predict_days": 0.5,  "resample_freq": "1min",  "test_days": 8,  "dropout": 0.0, "hidden_dim": 64, "label": "L7_P12H_1min"},
    {"lookback_days": 7,  "predict_days": 0.5,  "resample_freq": "2min",  "test_days": 8,  "dropout": 0.0, "hidden_dim": 64, "label": "L7_P12H_2min"},
    {"lookback_days": 7,  "predict_days": 0.5,  "resample_freq": "15min", "test_days": 8,  "dropout": 0.0, "hidden_dim": 64, "label": "L7_P12H_15min"},
    {"lookback_days": 7,  "predict_days": 0.5,  "resample_freq": "30min", "test_days": 8,  "dropout": 0.0, "hidden_dim": 64, "label": "L7_P12H_30min"},
    {"lookback_days": 7,  "predict_days": 0.5,  "resample_freq": "60min", "test_days": 8,  "dropout": 0.0, "hidden_dim": 64, "label": "L7_P12H_60min"},

    # ── Dropout 变化 ──
    {"lookback_days": 7,  "predict_days": 0.5,  "resample_freq": "5min",  "test_days": 8,  "dropout": 0.1, "hidden_dim": 64, "label": "L7_P12H_5min_d01"},
    {"lookback_days": 7,  "predict_days": 0.5,  "resample_freq": "5min",  "test_days": 8,  "dropout": 0.3, "hidden_dim": 64, "label": "L7_P12H_5min_d03"},
    {"lookback_days": 7,  "predict_days": 0.5,  "resample_freq": "5min",  "test_days": 8,  "dropout": 0.4, "hidden_dim": 64, "label": "L7_P12H_5min_d04"},
    {"lookback_days": 7,  "predict_days": 0.5,  "resample_freq": "5min",  "test_days": 8,  "dropout": 0.5, "hidden_dim": 64, "label": "L7_P12H_5min_d05"},

    # ── Hidden Dim 变化 ──
    {"lookback_days": 7,  "predict_days": 0.5,  "resample_freq": "5min",  "test_days": 8,  "dropout": 0.0, "hidden_dim": 32,  "label": "L7_P12H_5min_h32"},
    {"lookback_days": 7,  "predict_days": 0.5,  "resample_freq": "5min",  "test_days": 8,  "dropout": 0.0, "hidden_dim": 96,  "label": "L7_P12H_5min_h96"},
    {"lookback_days": 7,  "predict_days": 0.5,  "resample_freq": "5min",  "test_days": 8,  "dropout": 0.0, "hidden_dim": 128, "label": "L7_P12H_5min_h128"},
    {"lookback_days": 7,  "predict_days": 0.5,  "resample_freq": "5min",  "test_days": 8,  "dropout": 0.0, "hidden_dim": 256, "label": "L7_P12H_5min_h256"},

    # ── Test Days 变化 ──
    {"lookback_days": 7,  "predict_days": 0.5,  "resample_freq": "5min",  "test_days": 1,  "dropout": 0.0, "hidden_dim": 64, "label": "L7_P12H_5min_t1"},
    {"lookback_days": 7,  "predict_days": 0.5,  "resample_freq": "5min",  "test_days": 3,  "dropout": 0.0, "hidden_dim": 64, "label": "L7_P12H_5min_t3"},
    {"lookback_days": 7,  "predict_days": 0.5,  "resample_freq": "5min",  "test_days": 7,  "dropout": 0.0, "hidden_dim": 64, "label": "L7_P12H_5min_t7"},
    {"lookback_days": 7,  "predict_days": 0.5,  "resample_freq": "5min",  "test_days": 14, "dropout": 0.0, "hidden_dim": 64, "label": "L7_P12H_5min_t14"},
    {"lookback_days": 7,  "predict_days": 0.5,  "resample_freq": "5min",  "test_days": 20, "dropout": 0.0, "hidden_dim": 64, "label": "L7_P12H_5min_t20"},

    # ── 综合变化 ──
    {"lookback_days": 7,  "predict_days": 0.25, "resample_freq": "10min", "test_days": 5,  "dropout": 0.1, "hidden_dim": 96,  "label": "L7_P6H_10min_t5_d01_h96"},
    {"lookback_days": 7,  "predict_days": 0.5,  "resample_freq": "15min", "test_days": 10, "dropout": 0.2, "hidden_dim": 128, "label": "L7_P12H_15min_t10_d02_h128"},
    {"lookback_days": 7,  "predict_days": 0.75, "resample_freq": "30min", "test_days": 15, "dropout": 0.3, "hidden_dim": 192, "label": "L7_P18H_30min_t15_d03_h192"},
    {"lookback_days": 7,  "predict_days": 1.0,  "resample_freq": "60min", "test_days": 22, "dropout": 0.4, "hidden_dim": 384, "label": "L7_P24H_60min_t22_d04_h384"},
    {"lookback_days": 7,  "predict_days": 0.125,"resample_freq": "2min",  "test_days": 3,  "dropout": 0.0, "hidden_dim": 160, "label": "L7_P3H_2min_t3_h160"},
    {"lookback_days": 7,  "predict_days": 0.875, "resample_freq": "20min", "test_days": 12, "dropout": 0.35,"hidden_dim": 72,  "label": "L7_P21H_20min_t12_d035_h72"},

]

BASE_CONFIG = {
    "file_path": r"D:\Wuhan_Project\new_data\merged_minute_all.csv",
    "encoding": "utf-8-sig",
    "resample_freq": "10min",
    "stride": 1,
    "hampel_cols": ["Total_Flow"],  # 用 Hampel 滤波清洗的列 (总流量; 压力波动是真实工况, 仅物理界限裁剪)

    "test_days": 15,  # 测试集取最后 N 天: 必须 ≥ 最长(回看+预测)窗口, 否则测试序列数为 0
                      # (5min×14天回看+预测 = 4176 步 = 14.5 天; 原 90/10 比例切分测试段仅 ~3.3 天, 装不下)

    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.0,

    "teacher_forcing_ratio": 0.1,

    "batch_size": 32,
    "epochs": 500,
    "learning_rate": 0.002,
    "weight_decay": 5e-5,
    "patience": 30,
    "min_delta": 1e-4,

    "T_0": 15,
    "T_mult": 2,

    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "base_result_dir": r"D:\Wuhan_Project\results_lstm_seq_multi_config",
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


def detect_outliers(s, window=HAMPEL_WINDOW, k=HAMPEL_K, floor_ratio=MAD_FLOOR_RATIO):
    """Hampel 滤波: 基于滚动中位数 + MAD 的稳健离群检测 (同 plot_daily_flow_curves.py)。

    返回布尔 Series (True = 异常点)。scale = max(1.4826*MAD, floor_ratio*中位数),
    底部分数保证信号极稳定 (MAD≈0) 时仍不会把正常波动误判为异常。
    """
    med = s.rolling(window, center=True, min_periods=window // 2).median()
    res = (s - med).abs()
    mad = res.rolling(window, center=True, min_periods=window // 2).median()
    scale = np.maximum(1.4826 * mad, floor_ratio * med)
    return (res > k * scale).fillna(False)


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

        # sum(skipna=True): 单管 NaN 不拖垮该行总流量; 剔除全 NaN 后总和恒有有效分量
        df["Total_Flow"] = df[flow_cols].sum(axis=1).astype(float)

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
            df["运行泵数量"] = df[pump_run_cols].sum(axis=1)
        else:
            df["运行泵数量"] = 0

        out = df[["Total_Flow", "Target_Pressure", "运行泵数量"]].copy()
        return out

    def clean_and_resample(self, df):
        out = df.copy()

        # ── Hampel 清洗 (同 plot_daily_flow_curves.py): 严重偏离的异常值不删除,
        #    检出后置 NaN, 与物理界限越界值一起在下方按时间插值填补 ──
        for col in self.config.get("hampel_cols", []):
            if col not in out.columns:
                continue
            flag = detect_outliers(out[col])
            n_out = int(flag.sum())
            if n_out > 0:
                out.loc[flag, col] = np.nan
                print(f"  {col} Hampel 离群点 {n_out} 条 ({n_out / len(out):.3%}) 置 NaN 待插值")

        # 武汉数据: 用物理界限而非分位数裁剪——
        #   本数据 1% 分位 (170:1≈1312 m³/h, 压力≈0.30 MPa) 会把真实低流量/低压工况当成野点剔除,
        #   分位数清洗会破坏日内负荷的谷值形态, 影响长时预测。
        #   界限裁剪保留作为 Hampel 之后的兜底 (防长段垃圾值污染滚动中位数)
        BOUNDS = {
            "Total_Flow": (0.0, 10000.0),      # 负值为 170:1 的垃圾读数 (最小 -6e23); 上界同 train.py 野点阈值
            "Target_Pressure": (0.1, 0.5),     # MPa, 同 train.py (实测 0.21~0.37); 压力不做 Hampel
        }
        for col, (lo, hi) in BOUNDS.items():
            if col not in out.columns:
                continue
            s = out[col].copy()
            s[(s < lo) | (s > hi)] = np.nan
            out[col] = s

        numeric_cols = out.select_dtypes(include=[np.number]).columns
        out[numeric_cols] = out[numeric_cols].interpolate(method="time")
        out[numeric_cols] = out[numeric_cols].ffill().bfill()

        freq = self.config["resample_freq"]
        res = pd.DataFrame(index=out.resample(freq).mean().index)

        if "Total_Flow" in out.columns:
            res["Total_Flow"] = out["Total_Flow"].resample(freq).mean()
        if "Target_Pressure" in out.columns:
            res["Target_Pressure"] = out["Target_Pressure"].resample(freq).mean()
        if "运行泵数量" in out.columns:
            res["运行泵数量"] = out["运行泵数量"].resample(freq).last()

        res = res.ffill().bfill().dropna()
        return res

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
            out["运行泵数量_lag_1"] = out["运行泵数量"].shift(1)
            out["运行泵数量_lag_6"] = out["运行泵数量"].shift(6)
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
        df_clean = self.clean_and_resample(df_base)
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
        self.target_scaler.fit(df_train[self.target_cols].values)

    def transform_df(self, df):
        X = self.feature_scaler.transform(df[self.feature_cols].values)
        Y = self.target_scaler.transform(df[self.target_cols].values)
        return X, Y

    def inverse_transform_targets(self, arr):
        arr = np.asarray(arr)
        if arr.size == 0:
            return arr
        if arr.ndim == 2:
            return self.target_scaler.inverse_transform(arr)
        elif arr.ndim == 3:
            shape = arr.shape
            flat = arr.reshape(-1, shape[-1])
            inv = self.target_scaler.inverse_transform(flat)
            return inv.reshape(shape)
        else:
            raise ValueError("只支持2维或3维数组反归一化")

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


class Encoder(nn.Module):
    """双向 LSTM Encoder"""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cell_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        num_directions = 2
        num_layers = hidden.size(0) // num_directions
        batch_size = hidden.size(1)
        half_dim = hidden.size(2)

        hidden_cat = hidden.view(num_layers, num_directions, batch_size, half_dim)
        hidden_cat = torch.cat([hidden_cat[:, 0], hidden_cat[:, 1]], dim=2)
        cell_cat = cell.view(num_layers, num_directions, batch_size, half_dim)
        cell_cat = torch.cat([cell_cat[:, 0], cell_cat[:, 1]], dim=2)

        hidden_out = self.hidden_proj(hidden_cat)
        cell_out = self.cell_proj(cell_cat)
        return hidden_out, cell_out


class Decoder(nn.Module):
    """LSTM Decoder（自回归）"""
    def __init__(self, output_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        out, (hidden, cell) = self.lstm(x, (hidden, cell))
        pred = self.fc(out)
        return pred, hidden, cell


class Seq2SeqModel(nn.Module):
    """Seq2Seq: BiLSTM Encoder + LSTM Decoder"""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim=1):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(output_dim, hidden_dim, num_layers, dropout)
        self.output_dim = output_dim

    def forward(self, src, target_len, tgt=None, teacher_forcing_ratio=0.0):
        batch_size = src.size(0)
        hidden, cell = self.encoder(src)
        decoder_input = torch.zeros(batch_size, 1, self.output_dim, device=src.device)
        outputs = []

        for t in range(target_len):
            out, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs.append(out)
            use_teacher_forcing = (tgt is not None) and (random.random() < teacher_forcing_ratio)
            if use_teacher_forcing:
                decoder_input = tgt[:, t:t+1, :]
            else:
                decoder_input = out

        outputs = torch.cat(outputs, dim=1)
        return outputs


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
    flow_mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100

    return {
        "loss": avg_loss, "flow_mae": flow_mae, "flow_rmse": flow_rmse,
        "flow_mape": flow_mape, "y_pred_inv": y_pred_inv, "y_true_inv": y_true_inv
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
    axes[1].axvline(np.median(rel_err_display) * 100, color='green', linestyle='--', label=f'Median RE: {np.median(rel_err_display)*100:.2f}%')
    axes[1].set_title(f"{title_prefix} Relative Error Distribution (<50%)")
    axes[1].set_xlabel("Relative Error (%)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    误差分布图已保存: {save_path}")


# ==================== 单次实验运行 ====================

def run_experiment(cfg, x_train_all, y_train_all, x_test_all, y_test_all, processor, device):
    """
    按合并后的 cfg (BASE_CONFIG 被实验配置覆盖) 运行一次完整的训练+评估
    返回 metrics 字典
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
          f"  |  dropout={cfg['dropout']}  |  hidden={cfg['hidden_dim']}"
          f"  |  layers={cfg['num_layers']}  |  lr={cfg['learning_rate']}")
    print(f" 结果目录: {result_dir}")
    print(f"{'='*80}")

    # 构建序列 (不同 lookback/predict → 不同步数)
    X_train, Y_train = processor.make_sequences(x_train_all, y_train_all, lookback, predict)
    X_test, Y_test = processor.make_sequences(x_test_all, y_test_all, lookback, predict)

    # 计算 predict steps 用于显示
    freq_minutes = int(cfg["resample_freq"].replace("min", ""))
    points_per_day = (24 * 60) // freq_minutes
    predict_steps = int(predict * points_per_day)

    print(f"  {cfg['resample_freq']}频率: lookback={int(lookback * points_per_day)}步, predict={predict_steps}步")
    print(f"  X_train={X_train.shape}, Y_train={Y_train.shape}")
    print(f"  X_test={X_test.shape}, Y_test={Y_test.shape}")

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"  ⚠ 样本数为0，跳过此配置")
        return None

    train_loader = DataLoader(SeqDataset(X_train, Y_train), batch_size=cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(SeqDataset(X_test, Y_test), batch_size=cfg["batch_size"], shuffle=False)

    model = Seq2SeqModel(
        input_dim=X_train.shape[2],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        output_dim=1
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

            pred = model(batch_x, target_len=batch_y.size(1), tgt=batch_y,
                         teacher_forcing_ratio=cfg["teacher_forcing_ratio"])
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

    # 最终评估
    train_metrics = evaluate(model, train_loader, device, processor)
    test_metrics = evaluate(model, test_loader, device, processor)

    print(f"\n  最终结果:")
    print(f"  Train: Loss={train_metrics['loss']:.6f}, MAE={train_metrics['flow_mae']:.2f}, "
          f"RMSE={train_metrics['flow_rmse']:.2f}, MAPE={train_metrics['flow_mape']:.2f}%")
    print(f"  Test : Loss={test_metrics['loss']:.6f}, MAE={test_metrics['flow_mae']:.2f}, "
          f"RMSE={test_metrics['flow_rmse']:.2f}, MAPE={test_metrics['flow_mape']:.2f}%")

    with open(os.path.join(result_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        for name, m in [("Train", train_metrics), ("Test", test_metrics)]:
            f.write(f"{name}: Loss={m['loss']:.6f}, MAE={m['flow_mae']:.2f}, "
                    f"RMSE={m['flow_rmse']:.2f}, MAPE={m['flow_mape']:.2f}%\n")

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
        "dropout": cfg["dropout"],
        "hidden_dim": cfg["hidden_dim"],
        "num_layers": cfg["num_layers"],
        "learning_rate": cfg["learning_rate"],
        "teacher_forcing_ratio": cfg["teacher_forcing_ratio"],
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
    base_config = BASE_CONFIG
    set_seed(base_config["seed"])

    device = torch.device(base_config["device"])
    print(f"Device: {device}")
    print(f"结果根目录: {base_config['base_result_dir']}")

    # ============ 第一/二步: 按 resample_freq 分组, 每组特征工程只做一次 ============
    all_results = []

    # 按 resample_freq 分组 (保持 EXPERIMENTS 定义顺序)
    groups = {}
    for exp_cfg in EXPERIMENTS:
        freq = exp_cfg.get("resample_freq", base_config["resample_freq"])
        groups.setdefault(freq, []).append(exp_cfg)

    for freq, group in groups.items():
        print("\n" + "=" * 80)
        print(f" [Phase 1] 数据加载 & 特征工程 (resample_freq={freq}, 本组 {len(group)} 个实验共用)")
        print("=" * 80)

        group_cfg = {**base_config, "resample_freq": freq}
        processor = DataProcessor(group_cfg)
        print(" 正在加载并处理数据...")
        df_all_feat = processor.build_feature_table()
        print(f" 全量特征表: {df_all_feat.shape}")
        print(f" 时间范围: {df_all_feat.index.min()} ~ {df_all_feat.index.max()}")

        print("\n" + "=" * 80)
        print(" [Phase 2] 实验运行 (本组)")
        print("=" * 80)

        for exp_cfg in group:
            cfg = {**group_cfg, **exp_cfg}
            processor.config = cfg
            df_train_feat, df_test_feat = processor.split_by_time(df_all_feat)
            print(f" 训练集: {df_train_feat.shape} | {df_train_feat.index.min()} ~ {df_train_feat.index.max()}")
            print(f" 测试集: {df_test_feat.shape} | {df_test_feat.index.min()} ~ {df_test_feat.index.max()}")

            processor.fit_scalers(df_train_feat)
            x_train_all, y_train_all = processor.transform_df(df_train_feat)
            x_test_all, y_test_all = processor.transform_df(df_test_feat)

            result = run_experiment(cfg, x_train_all, y_train_all,
                                    x_test_all, y_test_all,
                                    processor, device)
            if result is not None:
                all_results.append(result)

    # ============ 第三步：汇总对比 ============
    print("\n" + "=" * 80)
    print(" [Phase 3] 实验汇总对比")
    print("=" * 80)

    if len(all_results) > 0:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(base_config["base_result_dir"], "experiment_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print(f"\n{'Label':<24}{'Freq':<7}{'Look':<6}{'Pred':<6}{'TestD':<6}{'Drop':<6}"
              f"{'TrMAE':<10}{'TeMAE':<10}{'TrRMSE':<10}{'TeRMSE':<10}"
              f"{'TrMAPE':<9}{'TeMAPE':<9}{'BestEp':<7}")
        print("-" * 140)
        for _, row in summary_df.iterrows():
            print(f"{row['label']:<24}{row['resample_freq']:<7}{row['lookback_days']:<6}{row['predict_days']:<6}"
                  f"{row['test_days']:<6}{row['dropout']:<6}"
                  f"{row['train_mae']:<10.2f}{row['test_mae']:<10.2f}"
                  f"{row['train_rmse']:<10.2f}{row['test_rmse']:<10.2f}"
                  f"{row['train_mape']:<9.2f}{row['test_mape']:<9.2f}"
                  f"{row['best_epoch']:<7}")

        # 汇总柱状图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        labels = summary_df["label"].tolist()
        x = np.arange(len(labels))
        w = 0.35

        axes[0].bar(x - w/2, summary_df["train_mae"], w, label='Train', color='steelblue')
        axes[0].bar(x + w/2, summary_df["test_mae"], w, label='Test', color='coral')
        axes[0].set_title('MAE Comparison')
        axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
        axes[0].legend(); axes[0].grid(alpha=0.3, axis='y')

        axes[1].bar(x - w/2, summary_df["train_rmse"], w, label='Train', color='steelblue')
        axes[1].bar(x + w/2, summary_df["test_rmse"], w, label='Test', color='coral')
        axes[1].set_title('RMSE Comparison')
        axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
        axes[1].legend(); axes[1].grid(alpha=0.3, axis='y')

        axes[2].bar(x - w/2, summary_df["train_mape"], w, label='Train', color='steelblue')
        axes[2].bar(x + w/2, summary_df["test_mape"], w, label='Test', color='coral')
        axes[2].set_title('MAPE (%) Comparison')
        axes[2].set_xticks(x); axes[2].set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
        axes[2].legend(); axes[2].grid(alpha=0.3, axis='y')

        plt.suptitle('Multi-Config Experiment Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(base_config["base_result_dir"], "experiment_comparison.png"), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\n 汇总图已保存: {os.path.join(base_config['base_result_dir'], 'experiment_comparison.png')}")
        print(f" 汇总表已保存: {summary_path}")

    print(f"\n 全部实验完成！结果保存在: {base_config['base_result_dir']}")


if __name__ == "__main__":
    main()
