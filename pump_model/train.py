"""
水厂水泵运行预测模型 — v6 纯流量+效率版 (总功率由累计电度差分计算)
改进:
  1. 清洗 NaN 行、全停状态
  1b. 总功率由累计电度差分计算 (compute_total_power_from_meters)，p1~6用172电度Δ/Δh，p7用70:7总有功
     泵级0值填充: 差分功率=0 且泵运行=1 (电表未刷新) → 最近有效读数填充; 泵停的0不填充
  2. 按泵组组合划分训练/测试集
  3. 移除功率预测, 仅输出 3管流量 + 1效率 = 4维
  4. NN容量全部用于流量和效率
  5. 压力修正: 修正后压力 = 原压力 - (吸水井液位-3.58)/102 (MPa)
     液位读取自原始列 '170:吸水井液位', 修正后压力参与训练, 液位本身不进入特征
  6. PINN物理约束: 基于水泵相似定律 (Q∝f, H∝f², P∝f³) 施加物理损失
     - 泵关闭 → 对应管路流量必须为0 (消除幽灵流量)
     - 总管流量 ≤ Σ 理论流量; 效率反推电功率 ≈ Σ 理论功率; 系统扬程 ≤ 最大理论扬程

2026-08-08 重构: 模型结构与数据管线已拆分为同目录下的
  model.py / data_processing.py, 本文件只保留训练/评估流程。
  原 D:\Wuhan_Project\train.py 保持原样, 未做修改。
"""
import json
import os
import sys
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

# ── 共享模块 (2026-08-08 重构拆分, 与推理脚本同目录) ──
# 模型结构 → model.py            (DeepWaterPlantModelWithEmbedding + PINN 物理约束 + 额定参数)
# 数据管线 → data_processing.py  (清洗 / 功率效率 / 训练测试集划分)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (RATED_Q, RATED_P, RATED_H, RATED_F,
                   physics_loss_pump, DeepWaterPlantModelWithEmbedding)
from data_processing import load_and_clean_data, add_combo_state, split_by_combo


# ============================================================================
# 1. 训练工具
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
                         all_output_cols, output_dir="results_v2"):
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
    meter_power_cols = [
        '泵1_功率_kW', '泵2_功率_kW', '泵3_功率_kW',
        '泵4_功率_kW', '泵5_功率_kW', '泵6_功率_kW',
        '泵7_功率_kW'
    ]   # 电度分钟差分功率列 (compute_total_power_from_meters 写入)
    power_freq_cols = [
        '170:1_运行频率', '170:2_运行频率', '170:3_运行频率',
        '170:4_运行频率', '170:5_运行频率', '170:6_运行频率',
        '70:7_运行频率'
    ]
    eff_col = '总管效率_pct'

    # ── 数据加载/清洗/功率效率/缓存 由 data_processing.load_and_clean_data 处理 ──
    TRAIN_SAMPLE_SIZE = None  # 训练集最多采样数，None=不采样

    # 加载 + 清洗 + 功率/效率 (含 Parquet 缓存加载/保存, 与原 train.py 一致)
    df = load_and_clean_data(discrete_cols, flow_cols, meter_power_cols, power_freq_cols)

    # 泵组状态编码 + 组合统计
    df, combo_counts = add_combo_state(df, discrete_cols)

    engineered_cols = []

    # 按泵组组合划分训练/测试集 (80/20, 组合不重叠) + 分层采样 + 重叠验证
    train_df, test_df, train_combos_actual, test_combos_actual = split_by_combo(
        df, combo_counts, train_sample_size=TRAIN_SAMPLE_SIZE)

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
    model_dir = r"D:\Wuhan_Project\pump_model\models"
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
