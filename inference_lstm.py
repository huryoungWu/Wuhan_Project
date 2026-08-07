import os
import sys
import pickle
import argparse

import numpy as np
import pandas as pd
import torch

# 保证能从本目录导入训练脚本 (复用其特征工程, 推理与训练完全一致)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_lstm import BASE_CONFIG, DataProcessor, Seq2SeqModel

DEFAULT_DATA = r'D:\Wuhan_Project\results_lstm_seq\L7_P24H_15min\input_lookback_L7_P24H_15min.csv'
DEFAULT_RESULT_DIR = r"D:\Wuhan_Project\results_lstm_seq\L7_P24H_15min"

# ── 分时压力默认参数 (厂方自行决定, 时段待定; 有需要可自行更改, 不改即默认) ──
# 每项: (起始小时, 结束小时, 目标压力 MPa), 区间左闭右开 [start, end)
DEFAULT_PRESSURE_SCHEDULE = [
    (0, 5, 0.30),    # 0-5点   0.3
    (5, 12, 0.33),   # 5-12点  0.33
    (12, 16, 0.33),  # 12-16点 0.33
    (16, 23, 0.33),  # 16-23点 0.33
    (23, 24, 0.30),  # 23-0点  0.3
]
DEFAULT_PRESSURE_ERROR = 0      # 典型压力误差
DEFAULT_PRESSURE_ERROR_MAX = 0   # 最大压力误差


class FlowPredictor:
    """基于 train_lstm_clean.py 训练结果的流量预测推理接口。

    加载训练产物 (best_seq2seq_model.pth + scaler.pkl), 接受最近
    lookback_days 天的原始数据, 输出未来 predict_days 天的 15min 级
    预测流量序列。

    输入 DataFrame 格式与训练数据一致 (merged_minute_all.csv 同构):
      时间列:  F_DateTime / 时间 / timestamp (任一; 秒级/分钟级均可, 或直接给
               DatetimeIndex 索引)
      必需列:  170:1_瞬时流量, 170:2_瞬时流量, 70:3_瞬时流量
      必需列:  170:总管压力 (或 总管压力1)
      可选列:  泵运行列 (*泵运行; 缺失时按 0 台泵处理)

    时间跨度: 至少 lookback_days + 2 天 (最长滞后特征为 2 天, 见下注),
              建议提供 7 天以上, 前 2 天用于填满滞后特征。
      注: 特征里 flow_lag_2day 需要 2 天前的值, 所以只给恰好 lookback 天的
          数据会导致滞后特征全为 NaN 被删光, 必须多带历史。

    用法 (程序接口, 三种输入模式, 返回完全一致):
      predictor = FlowPredictor()

      # 模式1: 直接给出 DataFrame (已在内存中, 不读 CSV)
      pred = predictor.predict(df_raw)

      # 模式2: 以 CSV 文件路径给出 (接口内部读 CSV)
      pred = predictor.predict("input.csv")

      # 模式3: 以列表/数组直接给出 (列名与训练数据一致)
      rows = [{"F_DateTime": "2026-07-15 06:00:00", "170:1_瞬时流量": 100.0, ...}, ...]
      pred = predictor.predict(rows)                       # list[dict]
      pred = predictor.predict(rows2, columns=["F_DateTime", "170:1_瞬时流量", ...])  # list[list]/ndarray 需列名

      # 输出也可为列表形式: 传入 as_list=True, 返回 list[dict]
      pred_list = predictor.predict(df_raw, as_list=True)  # [{"timestamp": ..., "Total_Flow": ...}, ...]
    """

    def __init__(self, result_dir=DEFAULT_RESULT_DIR, device=None,
                 pressure_schedule=None, pressure_error=None, pressure_error_max=None):
        self.result_dir = result_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ── 加载训练时保存的 scaler / 特征列 / 配置 ──
        # 优先用训练时保存的 scaler.pkl; 旧训练结果 (无此文件) 按训练流程重建
        scaler_path = os.path.join(result_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                saved = pickle.load(f)
            self._apply_saved(saved)
        else:
            self._rebuild_scaler(scaler_path)

        self.lookback_days = self.config["lookback_days"]
        self.predict_days = self.config["predict_days"]
        self.resample_freq = self.config["resample_freq"]
        self.freq_minutes = int(self.resample_freq.replace("min", ""))
        self.points_per_day = (24 * 60) // self.freq_minutes
        self.lookback_steps = int(self.lookback_days * self.points_per_day)
        self.predict_steps = int(self.predict_days * self.points_per_day)

        # ── 加载模型权重 ──
        model_path = os.path.join(result_dir, "best_seq2seq_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到模型权重: {model_path}")
        self.model = Seq2SeqModel(
            input_dim=len(self.feature_cols),
            hidden_dim=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            output_dim=1,
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # ── 复用训练的特征工程对象 (清洗/重采样/时间/滞后特征完全一致) ──
        self.processor = DataProcessor(self.config)
        self.processor.feature_scaler = self.feature_scaler
        self.processor.target_scaler = self.target_scaler
        self.processor.feature_cols = self.feature_cols

        # ── 分时压力参数 (默认见模块顶部, 有需要可自行更改) ──
        self.pressure_schedule = (list(DEFAULT_PRESSURE_SCHEDULE)
                                  if pressure_schedule is None else pressure_schedule)
        self.pressure_error = DEFAULT_PRESSURE_ERROR if pressure_error is None else pressure_error
        self.pressure_error_max = (DEFAULT_PRESSURE_ERROR_MAX
                                   if pressure_error_max is None else pressure_error_max)

        print(f"[FlowPredictor] 模型已加载: {os.path.basename(model_path)}")
        print(f"[FlowPredictor] lookback={self.lookback_days}d ({self.lookback_steps}步), "
              f"predict={self.predict_days}d ({self.predict_steps}步), freq={self.resample_freq}, "
              f"特征数={len(self.feature_cols)}, device={self.device}")

    def _apply_saved(self, saved):
        self.config = saved["config"]
        self.feature_scaler = saved["feature_scaler"]
        self.target_scaler = saved["target_scaler"]
        self.feature_cols = saved["feature_cols"]
        self.target_cols = saved["target_cols"]

    def _rebuild_scaler(self, scaler_path):
        """旧训练结果 (无 scaler.pkl) 兼容: 按训练脚本相同流程从原始数据重建。

        StandardScaler 拟合是确定性的, 相同数据 + 相同配置 → 与训练时统计量
        完全一致。注意: 若 BASE_CONFIG 已修改且与旧模型超参不一致, 结果可能
        不匹配, 此时应重新训练生成 scaler.pkl。
        """
        print("[FlowPredictor] 未找到 scaler.pkl, 按训练流程从原始数据重建"
              "(若 BASE_CONFIG 与旧模型超参不一致, 请重训一次)...")
        config = BASE_CONFIG
        processor = DataProcessor(config)
        df_all_feat = processor.build_feature_table()
        df_train_feat, _ = processor.split_by_time(df_all_feat)
        processor.fit_scalers(df_train_feat)
        saved = {
            "config": config,
            "feature_scaler": processor.feature_scaler,
            "target_scaler": processor.target_scaler,
            "feature_cols": processor.feature_cols,
            "target_cols": processor.target_cols,
        }
        with open(scaler_path, "wb") as f:
            pickle.dump(saved, f)
        print(f"[FlowPredictor] scaler 已重建并保存: {scaler_path}")
        self._apply_saved(saved)

    def predict(self, data, columns=None, encoding="utf-8-sig", as_list=False):
        """统一推理接口, 支持三种输入模式 (返回结果完全一致):

        模式1 (直接给出): data 为 pd.DataFrame, 原始数据已在内存中, 不读 CSV
        模式2 (CSV 给出): data 为 CSV 文件路径 (str / os.PathLike)
        模式3 (列表给出): data 为 list[dict] / list[list] / list[tuple] /
                          np.ndarray, 直接以 Python 数据结构给出输入参数

        Parameters
        ----------
        data : pd.DataFrame | str | os.PathLike | list | tuple | np.ndarray
            原始数据或其 CSV 路径; 列格式见类说明, 时间列可作索引或普通列。
        columns : list[str] | None
            仅模式3的位置型数据 (list[list] / ndarray) 需要: 原始数据列名,
            顺序与每行一致 (与训练数据列名相同); list[dict] 时可不传。
        encoding : str
            仅模式2的 CSV 文件编码 (默认 utf-8-sig)。
        as_list : bool
            为 True 时返回 list[dict] (每项含 timestamp / Total_Flow, 与
            模式3的 list[dict] 输入格式对称); 默认 False 返回 DataFrame。

        Returns
        -------
        pd.DataFrame 或 list[dict]
            DataFrame: index = 预测时刻 (15min 分辨率), 列 Total_Flow = 预测流量。
            list[dict]: [{"timestamp": "YYYY-MM-DD HH:MM:SS", "Total_Flow": 值}, ...]
        """
        if isinstance(data, pd.DataFrame):
            result = self._predict_df(data)
        elif isinstance(data, (str, os.PathLike)):
            result = self._predict_from_csv(data, encoding)
        elif isinstance(data, np.ndarray):
            result = self._predict_from_list(data, columns)
        elif isinstance(data, (list, tuple)):
            result = self._predict_from_list(data, columns)
        else:
            raise TypeError(
                f"不支持的数据类型: {type(data).__name__}; 请传入 pd.DataFrame / "
                f"CSV 路径 / 列表或数组")

        if as_list:
            return self._to_list(result)
        return result

    @staticmethod
    def _to_list(result):
        """将预测结果 DataFrame 转为 list[dict] (含 timestamp 与 Total_Flow)。"""
        return [
            {"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "Total_Flow": float(v)}
            for ts, v in result["Total_Flow"].items()
        ]

    @staticmethod
    def _pressure_target(ts, schedule):
        """按小时查找时刻 ts 对应的分时压力目标值 (时段表左闭右开)。"""
        hour = ts.hour
        for start, end, target in schedule:
            if start <= hour < end:
                return target
        raise ValueError(
            f"时刻 {ts} (小时 {hour}) 不在分时压力时段内, "
            f"请检查 pressure_schedule: {schedule}")

    def predict_pressure(self, pred, pressure_schedule=None, pressure_error=None,
                         pressure_error_max=None, as_list=False):
        """基于 predict() 的结果生成分时压力预测, 返回新变量 (不覆盖原 pred)。

        默认分时压力时段 (厂方待定, 可自行更改):
            0-5点 0.3 | 5-12点 0.33 | 12-16点 0.33 | 16-23点 0.33 | 23-0点 0.3
        默认压力误差: 典型 0.02, 最大 0.03 (误差幅值在 [0.02, 0.03] 内随机取值,
        方向随机; 传 pressure_error=0 可使误差在 [0, 0.03] 内取值)。

        Parameters
        ----------
        pred : pd.DataFrame | list[dict]
            predict() 的输出 (DataFrame: DatetimeIndex + Total_Flow;
            list[dict]: timestamp + Total_Flow)。
        pressure_schedule : list[(int, int, float)] | None
            分时压力时段表 [(起始小时, 结束小时, 目标压力), ...], 默认模块级
            DEFAULT_PRESSURE_SCHEDULE。
        pressure_error / pressure_error_max : float | None
            典型 / 最大压力误差幅值, 默认 DEFAULT_PRESSURE_ERROR / _MAX。
        as_list : bool
            与 predict() 一致: True 返回 list[dict], False 返回 DataFrame。

        Returns
        -------
        pd.DataFrame | list[dict]
            新变量 (不改动传入的 pred): 按时间顺序排列, 与 pred 时刻一一对应,
            在原有 Total_Flow 基础上新增 Pressure 列/键。
        """
        schedule = self.pressure_schedule if pressure_schedule is None else pressure_schedule
        err = self.pressure_error if pressure_error is None else pressure_error
        err_max = self.pressure_error_max if pressure_error_max is None else pressure_error_max
        if err > err_max:
            raise ValueError(f"pressure_error ({err}) 不应大于 pressure_error_max ({err_max})")

        # 统一为按时间排序的 DataFrame (copy, 不修改传入的 pred)
        if isinstance(pred, pd.DataFrame):
            result = pred.copy()
        else:
            result = pd.DataFrame(list(pred))
            result["timestamp"] = pd.to_datetime(result["timestamp"])
            result = result.set_index("timestamp")
        result = result.sort_index()

        pressures = []
        for ts in result.index:
            target = self._pressure_target(ts, schedule)
            err_mag = np.random.uniform(err, err_max)          # 误差幅值 [典型, 最大]
            pressures.append(round(target + (err_mag if np.random.rand() < 0.5 else -err_mag), 3))
        result["Pressure"] = pressures

        if as_list:
            return [
                {"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                 "Total_Flow": float(row["Total_Flow"]),
                 "Pressure": float(row["Pressure"])}
                for ts, row in result.iterrows()
            ]
        return result

    def _predict_from_csv(self, csv_path, encoding="utf-8-sig"):
        """模式2: 以 CSV 文件路径给出输入, 读取后交给 _predict_df 执行。"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"输入 CSV 不存在: {csv_path}")
        df_raw = pd.read_csv(csv_path, encoding=encoding)
        print(f"[predict] 模式2: 从 CSV 读取原始数据: {csv_path} ({len(df_raw)} 行)")
        return self._predict_df(df_raw)

    def _predict_from_list(self, data, columns=None):
        """模式3: 直接以列表/数组给出输入参数, 不读 CSV。

        支持形式:
          list[dict] / tuple[dict]   → 键为列名 (列名与训练数据一致)
          list[list] / list[tuple]   → 位置型行数据, 必须配合 columns 给列名
          np.ndarray (1D/2D)         → 单行/多行位置型数据, 必须配合 columns
          扁平 list[标量]             → 单行位置型数据, 必须配合 columns
        """
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                return self._predict_from_positional([list(data)], columns)
            if data.ndim == 2:
                return self._predict_from_positional(data.tolist(), columns)
            raise ValueError(f"数组维度过高: {data.ndim}D, 只支持 1D(单行) / 2D(多行)")
        if len(data) == 0:
            raise ValueError("输入列表为空")
        if isinstance(data[0], dict):
            # 列名来自 dict 键 (与训练数据列名一致)
            return self._predict_df(pd.DataFrame(list(data)))
        return self._predict_from_positional(list(data), columns)

    def _predict_from_positional(self, rows, columns):
        """位置型数据 (list[list] 等) → DataFrame; 必须提供原始数据列名。"""
        if columns is None:
            raise ValueError(
                "位置型列表/数组输入必须提供 columns 参数 (原始数据列名, 与训练数据列一致)")
        columns = list(columns)
        if len(rows) > 0 and len(columns) != len(rows[0]):
            raise ValueError(
                f"columns 长度 ({len(columns)}) 与数据宽度 ({len(rows[0])}) 不一致")
        return self._predict_df(pd.DataFrame(rows, columns=columns))

    def _predict_df(self, raw_df):
        """模式1: 直接给出内存中的 DataFrame, 不读 CSV (核心实现)。"""
        df = raw_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            for ts_col in ("F_DateTime", "时间", "timestamp"):
                if ts_col in df.columns:
                    df[ts_col] = pd.to_datetime(df[ts_col])
                    df = df.set_index(ts_col)
                    break
            else:
                raise ValueError("输入数据必须包含时间列 (F_DateTime / 时间 / timestamp) 或 DatetimeIndex 索引")
        df = df.sort_index()
        print(f"[predict] 输入数据: {df.index.min()} ~ {df.index.max()}, {len(df)} 行")

        # ── ① 清洗 + 特征工程 (与训练完全相同; 单段整体清洗, 无 train/test 切分) ──
        df_base = self.processor.build_base_features(df)
        df_clean = self.processor.clean_and_resample(df_base)
        df_time = self.processor.add_time_features(df_clean)
        df_feat = self.processor.add_lag_rolling_features(df_time)   # 内含 dropna

        # 校验特征列与训练一致 (输入格式不同会导致特征集合不同)
        missing = [c for c in self.feature_cols if c not in df_feat.columns]
        if missing:
            raise ValueError(
                f"特征列与训练不一致, 缺失: {missing[:5]}...\n"
                f"请检查输入数据列与训练数据 (merged_minute_all.csv) 是否同构")
        df_feat = df_feat[self.feature_cols]

        if len(df_feat) < self.lookback_steps:
            raise ValueError(
                f"输入历史不足: 清洗+特征工程后仅剩 {len(df_feat)} 行, "
                f"模型需要 ≥ {self.lookback_steps} 行 (lookback={self.lookback_days}天)。\n"
                f"最长滞后特征为 2 天, 建议提供 ≥ {self.lookback_days + 2} 天的原始数据。")
        print(f"[predict] 清洗+特征工程后: {len(df_feat)} 行 ({df_feat.index.min()} ~ {df_feat.index.max()})")

        # ── ② 取最后 lookback_steps 行作为模型输入窗口 ──
        window = df_feat.iloc[-self.lookback_steps:]
        X = window.values.astype(np.float32)
        X = self.feature_scaler.transform(X)
        x_tensor = torch.from_numpy(X).unsqueeze(0).to(self.device)   # (1, lookback, n_feat)

        # ── ③ 自回归解码 (teacher_forcing=0, 与测试评估一致) ──
        with torch.no_grad():
            pred = self.model(x_tensor, target_len=self.predict_steps,
                              tgt=None, teacher_forcing_ratio=0.0)    # (1, pred, 1)
        y_inv = self.target_scaler.inverse_transform(pred.cpu().numpy()[0])   # (pred, 1)

        # ── ④ 预测时刻: 输入窗口末尾之后, 按 resample_freq 逐点外推 ──
        last_ts = window.index[-1]
        future_idx = pd.date_range(
            start=last_ts + pd.Timedelta(minutes=self.freq_minutes),
            periods=self.predict_steps, freq=self.resample_freq)
        result = pd.DataFrame({"Total_Flow": y_inv[:, 0]}, index=future_idx)
        result.index.name = "timestamp"
        print(f"[predict] 输出: {result.index[0]} ~ {result.index[-1]}, {len(result)} 行 (15min)")
        return result


def main():
    parser = argparse.ArgumentParser(
        description="LSTM 流量预测推理 (基于 train_lstm_clean.py 训练结果)")
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="输入原始数据 CSV 路径 (与训练数据格式一致)")
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR,
                        help="训练结果目录 (含 best_seq2seq_model.pth 与 scaler.pkl)")
    parser.add_argument("--out", default="prediction.csv",
                        help="预测结果 CSV 输出路径 (默认 prediction.csv)")
    args = parser.parse_args()

    predictor = FlowPredictor(args.result_dir)
    pred = predictor.predict(args.data,as_list=True)   # 统一接口: 自动识别 CSV 路径
    print(f'pred:{pred}')

    # 新增: 基于 pred 按分时压力时段生成压力预测 → 新变量 pred_pressure,
    # 不覆盖原 pred (默认时段/误差见模块顶部, 有需要可自行更改)
    pred_pressure = predictor.predict_pressure(pred, as_list=True)
    print(f'pred_pressure:{pred_pressure}')

    # pred.to_csv(args.out, index=True, encoding="utf-8-sig")

    # print(f"\n预测结果已保存: {args.out}")
    # print(pred.head(10))
    # print(f"... (共 {len(pred)} 行, {pred.index[0]} ~ {pred.index[-1]})")


if __name__ == "__main__":
    main()
