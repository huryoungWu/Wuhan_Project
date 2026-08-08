import os
import random
import time
import numpy as np
import pandas as pd

# ============================================================================
# data_processing.py — 水泵训练数据清洗 / 功率效率计算 / 训练测试集划分
#
# 2026-08-08 从 train.py 抽取 (原 D:\Wuhan_Project\train.py 保持不变), 由
# pump_model 包内 train.py 调用, 与原 __main__ 中内联逻辑逐行一致:
#   1. 数据清洗: 传感器全NaN/野点/频率超限/压力越界/全停/幽灵流量/突变/关键列NaN
#   2. 功率与效率: 累计电度分钟级差分 (v2 口径) + 单泵功率频率分段MAD清洗 + 总管效率
#   3. 压力修正: 修正后压力 = 原压力 - (吸水井液位-3.58)/102
#   4. 泵组状态编码 (7位组合串) + 组合统计
#   5. 按泵组组合划分训练/测试集 (80/20, 测试组合不参与训练) + 分层采样 + 重叠验证
#
# 注意: 本文件任何修改都会改变训练数据口径, 改动后必须重新训练;
# 若缓存 Parquet 存在, 清洗逻辑不会重新执行 (由 CACHE_VERSION 控制失效)。
# ============================================================================

# ==================== 数据路径与缓存版本 ====================
DATA_PATH     = r"D:\Wuhan_Project\new_data\merged_minute_all.csv"
CACHE_PATH    = r"D:\Wuhan_Project\new_data\processed_cache.parquet"
CACHE_VERSION = 'power_clean_v1'  # 功率口径: 分钟级差分 + 单泵功率清洗基于电度差分功率列


# ============================================================================
# 0. 泵功率计算
# ============================================================================

def compute_pump_powers(df):
    """从三相电压/电流数据计算7个泵的功率 (已弃用, 功率以电度差分为准)。"""
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


def _pump_power_from_meter(kwh_series, state_series, time_index, clip_upper=500.0):
    """累计电度表 → 泵功率 (kW)，按秒级时间轴输出。

    物理背景: 电度表按 0.25 kWh 分度、每 ~5s 才跳一次，而数据是 1s 粒度。
      若在 1s 粒度上直接差分，跳表瞬间 ΔkWh/(1/3600h) = 900 kW 虚假脉冲
      (被 clip 到 500 后 ffill 填满跳表间隙)，导致总功率虚高 ~2.7 倍。
    正确做法: 先按分钟聚合 (每分钟取末值)，在分钟粒度上差分:
      分钟功率 = 该分钟内电量 / 1min，不受跳表节奏影响。
    分钟间无跳表 (Δ=0) 但泵在运行 → 沿用上一分钟功率 (ffill, 限30分钟);
    泵停止 → 功率强制为 0 (真实停机, 不填充)。
    """
    kwh_min = kwh_series.resample('1min').last()
    state_min = state_series.resample('1min').max()
    dt_h = kwh_min.index.to_series().diff().dt.total_seconds() / 3600.0
    p_min = (kwh_min.diff() / dt_h).clip(lower=0, upper=clip_upper)
    running = state_min > 0
    p_min = p_min.mask(p_min == 0)          # 分钟无跳表 → NaN (待沿用)
    p_min = p_min.ffill(limit=30).bfill(limit=30)
    p_min = p_min.where(running, 0.0).fillna(0.0)   # 泵停止 → 0
    p_sec = p_min.reindex(time_index, method='ffill').fillna(0.0)
    p_sec = p_sec.where(state_series.values > 0, 0.0)  # 秒级: 泵停止 → 0
    return p_sec.values


def compute_total_power_from_meters(df):
    """从累计电度表读数差分计算7泵总功率 (kW)，写入 df['总功率'] 列。

    泵1~6: 172:1~6_泵电度 (累计kWh) → 分钟级 ΔkWh / Δ小时 → kW
          (分钟级差分, 避免秒级跳表脉冲; 无跳表分钟沿用上分钟, 泵停=0)
    泵7:   70:7_总有功 (W) → /1000 → kW (瞬时值, 0=泵停, 不填充)
    总功率 = 泵1+...+泵6 + 泵7

    分钟级差分噪声的处理：
      - 泵级: 分钟无跳表+泵运行 → 沿用上分钟功率 (ffill(30) 即30分钟)
      - 总功率层: 零值 → ffill(limit=15)（电表非每分钟刷新）
      - 5秒滚动均值平滑尖峰
    """
    meter_cols = ['172:1_泵电度', '172:2_泵电度', '172:3_泵电度',
                  '172:4_泵电度', '172:5_泵电度', '172:6_泵电度']
    state_cols = ['170:1_泵运行', '170:2_泵运行', '170:3_泵运行',
                  '170:4_泵运行', '170:5_泵运行', '170:6_泵运行']

    # 时间轴 → DatetimeIndex; 差分依赖时间顺序, 未排序则先排序
    if 'F_DateTime' in df.columns:
        time_index = pd.DatetimeIndex(pd.to_datetime(df['F_DateTime']))
    else:
        time_index = pd.DatetimeIndex(pd.to_datetime(df.index))
    if not time_index.is_monotonic_increasing:
        print("  [WARN] 数据未按时间排序, 先排序再差分")
        order = time_index.argsort()
        df = df.iloc[order].reset_index(drop=True)
        time_index = pd.DatetimeIndex(pd.to_datetime(df['F_DateTime']))

    # 泵1~6: 分钟级累计kWh差分 → kW (修正秒级跳表脉冲)
    pump_powers = {}
    for col, state_col in zip(meter_cols, state_cols):
        name = col.replace('172:', '').replace('_泵电度', '')
        kwh_series = pd.Series(df[col].values, index=time_index)
        state_series = pd.Series(df[state_col].values, index=time_index)
        p = _pump_power_from_meter(kwh_series, state_series, time_index)
        pump_powers[f'泵{name}_功率_kW'] = p
        df[f'泵{name}_功率_kW'] = p

    # 泵7: 瞬时有功 W → kW (无电度表, 不差分, 直接取瞬时值)
    if '70:7_总有功' in df.columns:
        p7 = (df['70:7_总有功'].fillna(0) / 1000).clip(lower=0, upper=200)
        df['泵7_功率_kW'] = p7
        pump_powers['泵7_功率_kW'] = p7
    else:
        print("  警告: 缺少 '70:7_总有功' 列, 泵7功率设为0")
        df['泵7_功率_kW'] = 0.0
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
                       pf_estimate=0.88, save_csv=True, csv_path=None):
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
        if csv_path is None:
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
# 5a. 加载 + 清洗 + 功率/效率 (原 train.py __main__ 前半段, 含 Parquet 缓存)
# ============================================================================

def load_and_clean_data(discrete_cols, flow_cols, meter_power_cols, power_freq_cols,
                        data_path=DATA_PATH, cache_path=CACHE_PATH, cache_version=CACHE_VERSION):
    """加载原始数据 → 压力修正 → 清洗 → 功率/效率 → 缓存, 返回清洗后的 df。

    与原 train.py __main__ 中"缓存加载/生成"整段逻辑一致 (含所有打印):
      1. Parquet 缓存可用 (含 '170:总管压力_修正' + 正确功率口径版本) → 直接加载
      2. 否则从 CSV 重建: 压力修正 → 多步清洗 → 电度差分功率 → 效率 → 保存缓存
    返回的 df 已含 泵功率列 / 总功率 / 效率列, 且已按 F_TimeStamp 排序。
    """
    cache_usable = False
    if os.path.exists(cache_path):
        print(f"\n从缓存加载已处理数据: {cache_path}")
        t0 = time.time()
        df = pd.read_parquet(cache_path)
        print(f"数据形状: {df.shape}  加载耗时 {time.time()-t0:.1f}s")
        if ('170:总管压力_修正' in df.columns
                and '_cache_version' in df.columns
                and (df['_cache_version'] == cache_version).all()):
            cache_usable = True
        else:
            print(f"  [WARN] 缓存为旧版(未含压力修正或功率口径非{cache_version}), 重新从CSV生成")

    if cache_usable:
        # 缓存已含修正压力: 修正值覆盖 170:总管压力
        df['170:总管压力'] = df['170:总管压力_修正']
        df = df.drop(columns=['170:总管压力_修正', '_cache_version'])
        if 'F_DateTime' in df.columns:
            df['F_DateTime'] = pd.to_datetime(df['F_DateTime'])
    else:
        print(f"\n加载数据: {data_path}")
        t0 = time.time()
        df = pd.read_csv(data_path)
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

        # 计算功率 + 效率 (单泵功率列由 compute_total_power_from_meters 电度差分写入)
        df = compute_total_power_from_meters(df)        # 从累计电度差分计算总功率
        df = compute_efficiency(df, meter_power_cols, flow_cols,
                                pf_estimate=0.88, save_csv=True)

        # 单泵功率清洗 (向量化版)
        print("\n清洗单泵功率 (频率分段MAD离群检测)...")
        t_clean = time.time()
        df, clean_stats = clean_pump_power(df, power_freq_cols, meter_power_cols,
                                            mad_multiplier=3.5)
        print(f"清洗: {clean_stats['total_before']:,} -> {clean_stats['total_after']:,} "
              f"(剔除 {clean_stats['total_removed']:,}, {clean_stats['total_removed']/clean_stats['total_before']*100:.2f}%) "
              f"耗时 {time.time()-t_clean:.1f}s")

        # 保存缓存 (带功率口径版本标记)
        df['_cache_version'] = cache_version
        print(f"\n保存处理缓存: {cache_path}")
        df.to_parquet(cache_path, index=False)

    print(f"\n{'泵号':<6} {'运行点':<10} {'范围':<24} {'均值':<10} {'单位'}")
    for pn, pcol in zip(['1','2','3','4','5','6','7'], meter_power_cols):
        fc = power_freq_cols[int(pn)-1]
        mask = (df[fc] > 0) & (df[pcol] > 0)
        d = df.loc[mask, pcol]
        unit = 'kW' if 'kW' in pcol else 'kVA'
        if len(d) > 10:
            print(f"  泵{pn}   {len(d):<10,} {d.min():.1f}~{d.max():.1f} {unit:<8} {d.mean():.1f}     {unit}")

    return df


# ============================================================================
# 5b. 泵组状态编码 + 组合统计
# ============================================================================

def add_combo_state(df, discrete_cols):
    """生成泵组状态列 (7位0/1组合串) 并打印组合统计。返回 (df, combo_counts)。"""
    df['泵组状态'] = df[discrete_cols].apply(
        lambda row: ''.join(['1' if x > 0 else '0' for x in row]), axis=1
    )

    # 统计泵组组合
    combo_counts = df['泵组状态'].value_counts()
    print(f"\n泵组组合总数: {len(combo_counts)}")
    print("Top 15 泵组组合:")
    for combo, cnt in combo_counts.head(15).items():
        print(f"  {combo}: {cnt:>10,} ({100*cnt/len(df):.1f}%)")
    return df, combo_counts


# ============================================================================
# 5c. 按泵组组合划分训练/测试集 (含分层采样与组合重叠验证)
# ============================================================================

def split_by_combo(df, combo_counts, min_combo_samples=100, train_ratio=0.8,
                   seed=42, use_fixed_seed=False, train_sample_size=None):
    """按泵组组合划分训练/测试集 (测试集组合不参与训练), 与原 train.py 逻辑一致。

    ★ 核心改进: 测试集出现的泵组组合不会出现在训练集中。
    流程: 时间排序 → 过滤样本过少的组合 (归入训练) → 80/20 随机划分 → 可选
    分层采样 → 组合重叠验证 → 打印测试集组合详情。
    返回 (train_df, test_df, train_combos_actual, test_combos_actual)。
    """
    df['F_DateTime'] = pd.to_datetime(df['F_DateTime'])
    df = df.sort_values('F_TimeStamp')

    all_combos = sorted(combo_counts.index.tolist())
    n_combos = len(all_combos)

    # 过滤掉样本数太少的组合 (无法有效评估)
    valid_combos = [c for c in all_combos if combo_counts[c] >= min_combo_samples]
    too_small = [c for c in all_combos if combo_counts[c] < min_combo_samples]
    if too_small:
        print(f"\n样本数<{min_combo_samples}的组合 (归入训练集): {len(too_small)} 个 {too_small}")

    print(f"\n有效泵组组合: {len(valid_combos)} 个")

    # 随机划分: train_ratio 训练, 其余测试
    random.seed(seed) if use_fixed_seed else random.seed()
    shuffled_combos = random.sample(valid_combos, len(valid_combos))
    n_train_combos = max(1, int(len(valid_combos) * train_ratio))
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
    if train_sample_size and len(train_df) > train_sample_size:
        print(f"\n训练集采样: {len(train_df):,} -> {train_sample_size:,} (按泵组组合分层)")
        train_df = train_df.groupby('泵组状态', group_keys=False).apply(
            lambda g: g.sample(n=max(1, int(len(g) * train_sample_size / len(train_df))),
                               random_state=seed)
        ).reset_index(drop=True)
        print(f"  采样后训练集: {len(train_df):,} 行")
        # 测试集也限制上限
        max_test = train_sample_size // 2
        if len(test_df) > max_test:
            test_df = test_df.sample(n=max_test, random_state=seed).reset_index(drop=True)
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

    return train_df, test_df, train_combos_actual, test_combos_actual
