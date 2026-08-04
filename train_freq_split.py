# -*- coding: utf-8 -*-
"""
train_freq_split.py — 按「泵组组合 × 频率组合」划分训练/测试集 (v6 模型)

背景
----
train.py 的划分是泵组组合级: 测试集的 7 位泵组组合 (如 1100010) 不出现在训练集中。
本脚本改为频率组合级, 对每个泵组组合单独做一次实验:

  1. 每个泵组组合 (如 1100010) 只用它自己的行 —— 其他泵组组合不参与;
  2. 该组合每行的「频率组合」= 7 台泵的运行频率向量 (0.1Hz 取整, 停泵=0);
  3. 该组合出现过的频率组合按 80% / 20% 随机划分: 80% 进训练集, 20% 进测试集
     (测试集的频率组合在训练集中没有出现过);
  4. 用 v6 模型 (train.train_model, 4 维输出: 3 管流量 + 总管效率) 训练并评估,
     给出该组合的各项指标。

数据说明
--------
优先加载 new_data/processed_cache.parquet (train.py 生成的缓存, 已含压力修正
和功率/效率列); 缓存不存在时自动走与 train.py 相同的完整 CSV 清洗流程。

输出: new_results/ 文件夹 (不存在则自动创建)
  summary.csv / summary.json        每个泵组组合的整体指标汇总
  detail_{组合}.csv                  每个组合按频率组合细分的指标
  metrics_{组合}.json               每个组合的完整指标 (JSON)
  details_all.csv                   全部组合的频率组合级指标合并表

用法
----
  python train_freq_split.py                     # 全部组合 (~60 个)
  python train_freq_split.py --only 1100010      # 只跑指定组合
  python train_freq_split.py --epochs 150 --patience 40
  python train_freq_split.py --max-combos 3      # 先跑前 3 个组合试运行
"""

import os
import sys
import json
import time
import random
import zlib
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 复用 train.py 中的模型训练 / 数据处理函数 (train.py 的主体在 __main__ 中,
# import 不会触发训练流程)
from train import (train_model, NumpyEncoder,
                   correct_pressure,
                   compute_total_power_from_meters, compute_efficiency,
                   clean_pump_power)   # evaluate_by_combination 未用, 不引入

# ============================================================================
# 0. 常量配置
# ============================================================================

RESULTS_DIR = os.path.join(BASE_DIR, "new_results")   # 输出目录 (自动创建)
CACHE_PATH = r"D:\Wuhan_Project\new_data\processed_cache.parquet"
DATA_PATH = r"D:\Wuhan_Project\new_data\merged_minute_all.csv"
CACHE_VERSION = 'power_clean_v1'          # 与 train.py 相同的缓存版本标记

FREQ_DECIMALS = 1                        # 频率组合取整精度 (0.1Hz)
FREQ_SPLIT_RATIO = 0.2                   # 测试集频率组合占比 20%
MIN_FREQ_COMBO_SAMPLES = 50              # 频率组合样本数少于该值的只进训练集
MIN_FREQ_COMBOS = 10                     # 有效频率组合数少于该值的组合跳过
SEED = 42

# 与 train.py 主流程相同的列定义
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


# ============================================================================
# 1. 数据加载 (与 train.py 主流程一致: 缓存优先, 否则完整清洗)
# ============================================================================

def load_data():
    """加载数据: 优先 processed_cache.parquet; 缺失则按 train.py 的完整流程从 CSV 生成。"""
    if os.path.exists(CACHE_PATH):
        print(f"\n从缓存加载已处理数据: {CACHE_PATH}")
        t0 = time.time()
        df = pd.read_parquet(CACHE_PATH)
        print(f"数据形状: {df.shape}  加载耗时 {time.time()-t0:.1f}s")
        if ('170:总管压力_修正' in df.columns
                and '_cache_version' in df.columns
                and (df['_cache_version'] == CACHE_VERSION).all()):
            # 缓存已含修正压力: 修正值覆盖 170:总管压力
            df['170:总管压力'] = df['170:总管压力_修正']
            df = df.drop(columns=['170:总管压力_修正', '_cache_version'])
            return df
        print(f"  [WARN] 缓存为旧版, 重新从CSV生成")
        del df

    # ── 完整 CSV 清洗流程 (与 train.py 主流程相同) ──
    print(f"\n加载数据: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"数据形状: {df.shape}")

    df = correct_pressure(df)

    n_before = len(df)
    print(f"\n数据清洗:")
    data_cols = [c for c in df.columns if c not in ['F_TimeStamp', 'F_DateTime', 'flowtotal', 'LJflowtotal']]
    df = df[~df[data_cols].isna().all(axis=1)]                      # 传感器全NaN行
    df = df[~(df['170:2_瞬时流量'] > 10000)]                        # 管2极端野点
    df = df[~(df['170:1_瞬时流量'] > 10000)]                        # 管1极端野点
    for fc in [c for c in df.columns if '运行频率' in c]:           # 频率超范围
        df = df[~(df[fc] > 55)]
    df = df[~((df['170:总管压力'] < 0.1) | (df['170:总管压力'] > 0.5))]   # 总管压力异常
    df = df[~(df[discrete_cols] == 0).all(axis=1)]                  # 全泵停止
    df = df[~((df['70:7_泵运行'] == 0) & (df['70:3_瞬时流量'] > 10))]     # 泵7停但管3>10

    # 3σ 清洗管1流量离群点 (仅泵1运行时)
    running_p1 = df['170:1_泵运行'] > 0
    flow_data = df.loc[running_p1, '170:1_瞬时流量']
    if len(flow_data) > 100:
        mean_f, std_f = flow_data.mean(), flow_data.std()
        lo, hi = mean_f - 3 * std_f, mean_f + 3 * std_f
        df = df[~(running_p1 & ((df['170:1_瞬时流量'] < lo) | (df['170:1_瞬时流量'] > hi)))]

    # 物理约束: 主泵全停时管1/管2流量应≈0; 泵7停时管3流量应≈0
    main_pumps_off = (df[['170:1_泵运行', '170:2_泵运行', '170:3_泵运行',
                          '170:4_泵运行', '170:5_泵运行', '170:6_泵运行']] == 0).all(axis=1)
    df = df[~(main_pumps_off & ((df['170:1_瞬时流量'] > 5) | (df['170:2_瞬时流量'] > 5)))]
    df = df[~((df['70:7_泵运行'] == 0) & (df['70:3_瞬时流量'] > 5))]

    # 流量突变检测
    df = df.sort_values('F_TimeStamp')
    for fc in flow_cols:
        spike_idx = df[df[fc].diff().abs() > 500].index
        remove_idx = set(spike_idx) | set(i + 1 for i in spike_idx if i + 1 in df.index)
        df = df.drop(index=list(remove_idx)) if remove_idx else df

    # 关键列为NaN
    essential_cols = (list(flow_cols) + list(discrete_cols)
                      + [c for c in df.columns if '运行频率' in c] + ['170:总管压力'])
    df = df[~df[essential_cols].isna().any(axis=1)]

    print(f"  清洗前: {n_before:,}, 清洗后: {len(df):,}, 剔除: {n_before-len(df):,}")

    df = compute_total_power_from_meters(df)
    df = compute_efficiency(df, meter_power_cols, flow_cols, pf_estimate=0.88, save_csv=True)
    df, clean_stats = clean_pump_power(df, power_freq_cols, meter_power_cols, mad_multiplier=3.5)
    print(f"单泵功率清洗: {clean_stats['total_before']:,} -> {clean_stats['total_after']:,}")

    df['_cache_version'] = CACHE_VERSION
    print(f"\n保存处理缓存: {CACHE_PATH}")
    df.to_parquet(CACHE_PATH, index=False)
    return df


# ============================================================================
# 2. 频率组合划分
# ============================================================================

def add_freq_combo_column(df):
    """为每行添加「频率组合」列: 7台泵运行频率 (0.1Hz取整, 停泵=0) 拼接, 如 45.2|46.1|0|0|0|50.0|0。

    取整到 0.1Hz 是变频泵的实际设定精度; 停泵位强制为 0, 避免电表回零噪声
    造成同一工况被拆成多个频率组合。
    """
    keys = []
    for i in range(7):
        v = np.round(df[continuous_cols[i]].to_numpy(dtype=np.float64), FREQ_DECIMALS)
        v = np.where(df[discrete_cols[i]].to_numpy(dtype=np.float64) > 0, v, 0.0)
        keys.append(np.rint(v * 10 ** FREQ_DECIMALS).astype(np.int64))
    disp = pd.DataFrame(np.stack(keys, axis=1) / 10 ** FREQ_DECIMALS, index=df.index).astype(str)
    df['频率组合'] = disp[0].str.cat([disp[i] for i in range(1, 7)], sep='|')
    return df


def split_by_freq_combo(combo_df, seed=SEED):
    """对单个泵组组合: 按频率组合 80% 训练 / 20% 测试 划分。

    返回 (train_df, test_df, stats):
      - 频率组合样本数 < MIN_FREQ_COMBO_SAMPLES 的只进训练集 (无法可靠评估);
      - 剩余频率组合打乱后按 80/20 划分 (按组合划分, 不按行), 保证
        测试集的频率组合在训练集中未出现过;
      - 种子 = crc32(组合串) 使每个组合的划分与处理顺序无关且可复现。
    """
    counts = combo_df['频率组合'].value_counts()
    total_fc = len(counts)
    small = counts[counts < MIN_FREQ_COMBO_SAMPLES].index.tolist()   # 样本过少 → 训练集
    large = counts[counts >= MIN_FREQ_COMBO_SAMPLES].index.tolist()

    if not large:
        # 无任何 ≥最小样本的频率组合 → 测试集为空, 由调用方跳过
        empty = combo_df.iloc[0:0]
        return empty.copy(), empty.copy(), {
            '频率组合总数': total_fc, '有效频率组合': 0,
            '训练频率组合': len(small), '测试频率组合': 0,
            '训练样本': len(combo_df), '测试样本': 0,
        }

    rng = random.Random(zlib.crc32(combo_df['泵组状态'].iloc[0].encode()) ^ seed)
    rng.shuffle(large)
    n_test = max(1, int(len(large) * FREQ_SPLIT_RATIO))
    n_test = min(n_test, len(large) - 1)   # 至少留 1 个组合进训练集
    test_fc = set(large[:n_test])
    train_fc = set(large[n_test:]) | set(small)

    train_df = combo_df[combo_df['频率组合'].isin(train_fc)].reset_index(drop=True)
    test_df = combo_df[combo_df['频率组合'].isin(test_fc)].reset_index(drop=True)

    stats = {
        '频率组合总数': total_fc,
        '有效频率组合': len(large),
        '训练频率组合': len(train_fc),
        '测试频率组合': len(test_fc),
        '训练样本': len(train_df),
        '测试样本': len(test_df),
    }
    return train_df, test_df, stats


# ============================================================================
# 3. 评估: 按频率组合细分 (类比 train.py 的 evaluate_by_combination)
# ============================================================================

def _clean_nan(obj):
    """递归替换 NaN → None, 避免写入 JSON 时出现非法的 NaN 字面量。

    train.py 的 train_model 在真实值全 ≤10 时 (如泵7关时 70:3 恒为0) MAPE 返回
    np.nan; JSON 标准不允许 NaN 字面量, 下游严格解析会报错, 故统一替换为 null。
    """
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj

def evaluate_by_freq_combo(test_df, y_true, y_pred, output_cols):
    """按频率组合计算测试集各项指标 (MAE/RMSE/R2/MAPE)。"""
    results = []
    test_df = test_df.reset_index(drop=True)
    combo_name = test_df['泵组状态'].iloc[0] if '泵组状态' in test_df.columns else ''
    for fc in sorted(test_df['频率组合'].unique()):
        mask = test_df['频率组合'] == fc
        gt, pr = y_true[mask], y_pred[mask]
        if len(gt) == 0:
            continue
        for i, col in enumerate(output_cols):
            rmse = np.sqrt(mean_squared_error(gt[:, i], pr[:, i]))
            mae = mean_absolute_error(gt[:, i], pr[:, i])
            r2 = r2_score(gt[:, i], pr[:, i])
            valid_mask = gt[:, i] > 10
            # 无 >10 的有效样本 (如泵7关时 70:3 恒为0) → MAPE 无定义, 用 None 占位
            # (JSON 输出 null, CSV 输出空; 不用 NaN 避免下游严格 JSON 解析报错)
            mape = (round(float(np.mean(np.abs(gt[valid_mask, i] - pr[valid_mask, i])
                                        / gt[valid_mask, i])) * 100, 2)
                    if np.sum(valid_mask) > 0 else None)
            results.append({
                '泵组状态': combo_name, '频率组合': fc, '输出列': col,
                '样本数': len(gt), 'MAE': round(mae, 2),
                'RMSE': round(rmse, 2), 'R2': round(r2, 4),
                'MAPE(%)': mape
            })
    return pd.DataFrame(results)


# ============================================================================
# 4. 单个泵组组合实验
# ============================================================================

def run_experiment(combo, df, cfg):
    """对单个泵组组合: 划分 → 训练 → 评估 → 保存指标。返回汇总行 (dict)。"""
    t0 = time.time()
    combo_df = df[df['泵组状态'] == combo]
    print(f"\n{'=' * 60}\n泵组组合 {combo}  行数 {len(combo_df):,}\n{'=' * 60}")

    if len(combo_df) == 0:
        return {'泵组状态': combo, '状态': 'skipped', '原因': '无数据行'}

    train_df, test_df, stats = split_by_freq_combo(combo_df, seed=cfg.seed)

    if stats['有效频率组合'] < MIN_FREQ_COMBOS:
        msg = f"有效频率组合 {stats['有效频率组合']} < {MIN_FREQ_COMBOS}"
        print(f"  [SKIP] {msg}")
        return {'泵组状态': combo, '状态': 'skipped', '原因': msg,
                '频率组合总数': stats['频率组合总数'],
                '有效频率组合': stats['有效频率组合'], '样本数': len(combo_df)}
    if len(test_df) == 0:
        msg = "所有频率组合样本过少, 测试集为空"
        print(f"  [SKIP] {msg}")
        return {'泵组状态': combo, '状态': 'skipped', '原因': msg,
                '频率组合总数': stats['频率组合总数'],
                '有效频率组合': stats['有效频率组合'], '样本数': len(combo_df)}

    # 验证: 测试集频率组合不出现在训练集中
    overlap = set(train_df['频率组合']) & set(test_df['频率组合'])
    if overlap:
        print(f"  [WARN] 频率组合重叠 {len(overlap)} 个 (划分失败)")
    else:
        print(f"  [OK] 频率组合不重叠: 训练 {stats['训练频率组合']} 个 / 测试 {stats['测试频率组合']} 个")

    print(f"  训练样本 {len(train_df):,}  测试样本 {len(test_df):,}")

    # ── 训练 (复用 train.py 的 v6 模型) ──
    result = train_model(
        train_df, test_df,
        discrete_cols, continuous_cols,
        flow_cols, eff_col,
        [],
        num_epochs=cfg.epochs, batch_size=cfg.batch_size, patience=cfg.patience
    )

    y_true, y_pred = result['y_true'], result['y_pred']
    all_cols = result['all_output_cols']

    # ── 汇总行 (整体指标) ──
    row = {
        '泵组状态': combo,
        '状态': 'ok',
        '训练样本': len(train_df),
        '测试样本': len(test_df),
        '频率组合总数': stats['频率组合总数'],
        '有效频率组合': stats['有效频率组合'],
        '训练频率组合': stats['训练频率组合'],
        '测试频率组合': stats['测试频率组合'],
        '训练轮数': result['epochs_trained'],
        '训练时长_s': round(result['training_time'], 1),
        '运行时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    for col, m in result['metrics'].items():
        for k in ['MAE', 'RMSE', 'MAPE(%)', 'R2']:
            row[f'{col}_{k}'] = m[k]
    row = _clean_nan(row)   # MAPE 可能为 np.nan (真实值全 ≤10), 统一转 None

    # ── 按频率组合细分指标 ──
    detail = evaluate_by_freq_combo(test_df, y_true, y_pred, all_cols)
    detail = detail[detail['样本数'] > 0]

    # ── 保存 ──
    detail_path = os.path.join(RESULTS_DIR, f"detail_{combo}.csv")
    detail.to_csv(detail_path, index=False, encoding='utf-8-sig')
    metrics_path = os.path.join(RESULTS_DIR, f"metrics_{combo}.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(_clean_nan({
            '泵组状态': combo,
            '频率组合划分': stats,
            '整体指标': result['metrics'],
            '频率组合级指标': detail.to_dict(orient='records'),
        }), f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    if cfg.save_models:
        model_dir = os.path.join(RESULTS_DIR, "models")
        os.makedirs(model_dir, exist_ok=True)
        torch_model_path = os.path.join(model_dir, f"model_{combo}.pt")
        import torch
        torch.save({
            'model_state_dict': result['model'].state_dict(),
            'continuous_scaler': result['continuous_scaler'],
            'output_scaler': result['output_scaler'],
            'discrete_cols': discrete_cols,
            'continuous_cols': continuous_cols,
            'flow_cols': list(flow_cols),
            'eff_col': eff_col,
            'all_output_cols': all_cols,
            'output_dim': 4,
            'train_combos': [combo],
        }, torch_model_path)

    print(f"  指标已保存: {detail_path}")
    return row


# ============================================================================
# 5. 主流程
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="按 泵组组合×频率组合 划分的训练/测试实验")
    parser.add_argument('--only', type=str, default=None, help='只跑指定的泵组组合 (如 1100010)')
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数 (默认 150)')
    parser.add_argument('--patience', type=int, default=40, help='早停耐心 (默认 40)')
    parser.add_argument('--batch-size', type=int, default=16384, help='批大小')
    parser.add_argument('--seed', type=int, default=SEED, help='随机种子')
    parser.add_argument('--min-freq-samples', type=int, default=MIN_FREQ_COMBO_SAMPLES,
                        help='频率组合最小样本数 (低于该值只进训练集)')
    parser.add_argument('--max-combos', type=int, default=0, help='只跑前 N 个组合 (0=全部, 调试用)')
    parser.add_argument('--save-models', action='store_true', help='同时保存每个组合的模型权重')
    return parser.parse_args()


def main():
    cfg = parse_args()
    global MIN_FREQ_COMBO_SAMPLES
    MIN_FREQ_COMBO_SAMPLES = cfg.min_freq_samples

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"结果输出目录: {RESULTS_DIR}")

    # ── 数据加载 ──
    df = load_data()

    # ── 泵组状态 + 频率组合 ──
    df['泵组状态'] = df[discrete_cols].apply(
        lambda row: ''.join(['1' if x > 0 else '0' for x in row]), axis=1
    )
    df = add_freq_combo_column(df)

    combo_counts = df['泵组状态'].value_counts()
    print(f"\n泵组组合总数: {len(combo_counts)}")
    for combo, cnt in combo_counts.head(10).items():
        print(f"  {combo}: {cnt:>10,}")

    combos = sorted(combo_counts.index.tolist())
    if cfg.only:
        if cfg.only not in combo_counts.index:
            print(f"组合 {cfg.only} 在数据中不存在"); sys.exit(1)
        combos = [cfg.only]
    if cfg.max_combos:
        combos = combos[:cfg.max_combos]
    print(f"\n将处理 {len(combos)} 个泵组组合")

    # ── 逐个组合实验 ──
    summary_rows = []
    for combo in combos:
        row = run_experiment(combo, df, cfg)
        summary_rows.append(row)

    # ── 汇总保存 ──
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    summary_json = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary.where(pd.notna(summary), None).to_dict(orient='records'),
                  f, ensure_ascii=False, indent=2)

    # 合并全部频率组合级明细
    # 注意: 泵组状态 形如 0110010, 读回时须用 dtype=str 防止被推断为 int 丢前导0
    # (Excel 直接打开 CSV 也会丢前导0, 以 JSON/summary.json 为准)
    detail_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith('detail_') and f.endswith('.csv')]
    if detail_files:
        all_detail = pd.concat([pd.read_csv(os.path.join(RESULTS_DIR, f), encoding='utf-8-sig',
                                            dtype={'泵组状态': str})
                                for f in sorted(detail_files)], ignore_index=True)
        all_detail.to_csv(os.path.join(RESULTS_DIR, "details_all.csv"),
                          index=False, encoding='utf-8-sig')

    print(f"\n{'=' * 60}\n汇总结果已保存到:\n  {summary_path}\n  {summary_json}")
    ok_rows = summary[summary['状态'] == 'ok'] if '状态' in summary.columns else summary
    print(f"\n成功实验: {len(ok_rows)}  跳过: {len(summary) - len(ok_rows)}")
    if len(ok_rows):
        cols = ['泵组状态', '训练样本', '测试样本', '训练频率组合', '测试频率组合',
                '170:1_瞬时流量_R2', '170:2_瞬时流量_R2', '70:3_瞬时流量_R2',
                'flowtotal(总和)_R2', '总管效率_pct_R2']
        cols = [c for c in cols if c in summary.columns]
        print("\n" + summary[cols].to_string(index=False))
    return summary


if __name__ == '__main__':
    main()
