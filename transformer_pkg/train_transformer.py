import os
import argparse
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
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── 共享模块 (2026-08-08 重构拆分, 与推理脚本同目录) ──
# 模型结构 → transformer_model.py (TimeSeriesTransformer, 推理共用同一份代码)
#          → itransformer_model.py (iTransformer, 经 --model 切换, 默认不变)
# 数据管线 → data_processing.py   (清洗/特征/训练测试集划分, 推理共用)
from transformer_model import TimeSeriesTransformer
from itransformer_model import iTransformer
from data_processing import DataProcessor, SeqDataset

# ============================================================================
# train_transformer.py — 基于 train_lstm.py 的模型架构升级版 (Transformer)
#
# 2026-08-08 重构: 模型结构与数据管线已拆分为同目录下的
#   transformer_model.py / data_processing.py, 本文件只保留配置/训练/评估。
#   原 D:\Wuhan_Project\train_transformer.py 保持原样, 未做修改。
#
# 数据清洗 / 特征工程 / 评估管线与 train_lstm.py 完全一致 (含按预测起点时刻统计),
# 仅替换模型: 原 BiLSTM Seq2Seq → Transformer Encoder (多头自注意力)。
#
# Transformer 特性:
#   - 自注意力直接建模回看窗口内任意两时刻间的依赖, 长期依赖捕捉强于循环网络
#     (流量存在 24h 日周期 + 多日滞后特征, 正是注意力擅长的模式)
#   - 直接多步输出 (无自回归 / 无 teacher forcing), 训练可并行
#   - 可学习位置编码保留时序位置信息
#
# 相对 train_lstm.py 的修改点:
#   1. BASE_CONFIG: 模型超参改为 d_model / nhead / num_layers / dim_feedforward /
#      transformer_dropout; lr 降到 1e-3 (Transformer 对学习率更敏感)
#   2. Encoder/Decoder/Seq2SeqModel → TimeSeriesTransformer (定义见 transformer_model.py)
#   3. 训练循环去掉 teacher forcing (直接多步输出用不到)
# ============================================================================

BASE_CONFIG = {
    "file_path": r"D:\Wuhan_Project\new_data\merged_minute_all.csv",
    "encoding": "utf-8-sig",
    "resample_freq": "30min",
    "stride": 1,
    # Total_Flow 会在清洗阶段展开为三根管逐管清洗; Target_Pressure 也补做 Hampel
    "hampel_cols": ["Total_Flow", "Target_Pressure"],

    "lookback_days": 1,                      # 回看窗口 (天)
    "predict_days": 1.0,                     # 预测窗口 (天)
    "label": "L7_P24H_30min_itransformer_test",         # 结果子目录名

    "test_days": 12,  # 测试集取最后 N 天: 必须 ≥ 回看+预测天数, 否则测试序列数为 0

    "mape_floor_ratio": 0.1,  # MAPE 过滤: 排除 |true| < 该比例 * max|true| 的点 (夜间近零流量)

    # "target_transform": "log1p" 时目标做 log1p 变换后归一化训练, 评估时 expm1 反变换回原始单位
    # (流量右偏, log 空间训练与相对误差/MAPE 对齐, 通常有改善); 默认 None = 原版行为
    "target_transform": None,

    # ── Transformer 架构超参 (Encoder 全自注意力) ──
    "d_model": 128,             # 嵌入 / 注意力维度
    "nhead": 4,                # 注意力头数 (需整除 d_model)
    "num_layers": 3,           # Encoder 层数
    "dim_feedforward": 256,    # 前馈层隐层维度
    "transformer_dropout": 0.2,

    "model_type": "itransformer",  # 模型类型: transformer (默认, 原版) | itransformer

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
    "base_result_dir": r"D:\Wuhan_Project\transformer_pkg\results",
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


# ==================== 训练工具 ====================

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
    print(f" 实验: {label}  |  model_type={cfg.get('model_type', 'transformer')}"
          f"  |  lookback={lookback}d  |  predict={predict}d"
          f"  |  freq={cfg['resample_freq']}  |  test_days={cfg['test_days']}"
          f"  |  d_model={cfg['d_model']}  |  nhead={cfg['nhead']}"
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

    # ── 模型工厂: 按 cfg["model_type"] 构造 (默认 transformer, 与原版一致) ──
    model_type = cfg.get("model_type", "transformer")
    common_model_kwargs = dict(
        input_dim=X_train.shape[2],
        output_dim=1,
        horizon=predict_steps,
        input_len=lookback_steps,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["transformer_dropout"],
    )
    print(model_type,model_type == "itransformer")
    if model_type == "itransformer":
        # iTransformer 反归一化后目标通道处于 feature_scaler 域, 仅当
        # target_transform=None (两 scaler 标定同一原始列) 时与 target_scaler 域一致
        assert cfg.get("target_transform") is None, \
            "iTransformer 要求 target_transform=None (RevIN 反归一化与 log1p 目标域不兼容)"
        model = iTransformer(
            **common_model_kwargs,
            target_idx=processor.feature_cols.index(processor.target_cols[0]),
        ).to(device)
    else:
        model = TimeSeriesTransformer(**common_model_kwargs).to(device)

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
    parser = argparse.ArgumentParser(description="训练 Transformer / iTransformer 流量预测模型")
    # default=None: 不传 --model/--label 时用 BASE_CONFIG 里的值 (改配置即生效),
    # 显式传参才覆盖, 避免 argparse 默认值把 BASE_CONFIG 的修改顶掉
    parser.add_argument("--model", choices=["transformer", "itransformer"], default=None,
                        help="模型类型 (默认 None = 用 BASE_CONFIG['model_type'])")
    parser.add_argument("--label", default=None,
                        help="结果子目录名 (默认 None = 用 BASE_CONFIG['label'])")
    args = parser.parse_args()

    config = dict(BASE_CONFIG)            # 浅拷贝, 不修改全局 BASE_CONFIG (被 _rebuild_scaler 导入)
    if args.model is not None:
        config["model_type"] = args.model
    if args.label is not None:
        config["label"] = args.label
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
