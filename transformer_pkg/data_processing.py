import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler

# ============================================================================
# data_processing.py — 数据清洗 / 特征工程 / 训练测试集划分 (共用管线)
#
# 2026-08-08 从 train_transformer.py 原样抽取; 训练 (train_transformer.py) 与
# 推理 (inference_transformer.py) 共用, 保证训练与推理特征口径完全一致:
#   1. Hampel 离群清洗 + 物理界限裁剪 + 时间插值 (逐管执行, 泵切换保护)
#   2. 重采样 (30min) + 删除空 bin (空洞不虚构)
#   3. 时间特征 / 滞后滚动特征
#   4. 按时间划分训练/测试集 (split_by_time) 与归一化 (scaler 由训练端拟合并保存)
#   5. 序列窗口生成 (make_sequences) 与 SeqDataset
#
# 注意: 本文件任何修改都会同时改变训练与推理的特征口径 —— 改动后必须重新训练,
# 否则旧模型权重与新特征不匹配 (scaler.pkl 里的 feature_cols 校验只能兜底一部分)。
# ============================================================================

# ==================== 异常值清洗参数 (Hampel 滤波) ====================
HAMPEL_WINDOW     = 600     # 滚动窗口 (秒, 10 分钟; 数据为秒级, 重采样之前执行)
HAMPEL_K          = 10.0    # 阈值: |x - 局部中位数| > k * scale 判为异常
MAD_FLOOR_RATIO   = 0.02    # scale 下限 = 局部中位数的 2% (防 MAD≈0 误报)
MAX_IMPUTE_ROWS   = 900     # 插值最多跨越的行数 (= 秒, 即 15 分钟); 更长的空洞不补, 保持 NaN
PUMP_GUARD_SECONDS = 300    # 泵切换前后 5 分钟内不做 Hampel 判定 (真实阶跃保护)


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
