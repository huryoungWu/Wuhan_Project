"""重新计算功率与效率 (修正电度差分方法) — 独立数据重算脚本

背景:
    原 compute_total_power_from_meters 在 1s 粒度上对累计电度表 (172:N_泵电度) 差分。
    但该表按 0.25 kWh 分度、每 ~5s 才跳一次, 而数据是 1s 粒度 →
    跳表瞬间 ΔkWh/(1/3600h) = 900 kW 虚假脉冲 (被 clip 到 500),
    总功率虚高 ~2.7 倍, 总管效率被低估 ~2.6 倍 (18% → 真实 ~48%)。

    本脚本用修正后的分钟级差分逻辑重算 泵功率/总功率/水力功率/总管效率,
    输出到 merged_minute_all_with_efficiency_v2.csv (不覆盖原文件)。

修正逻辑 (train.py compute_total_power_from_meters):
    先按分钟聚合累计电度 (每分钟取末值), 在分钟粒度上差分:
      分钟功率 = 该分钟内电量 / 1min, 不受跳表节奏影响;
    分钟间无跳表但泵在运行 → 沿用上一分钟功率 (限30分钟);
    泵停止 → 功率强制为 0。

用法:
    python recompute_power_efficiency.py
"""
import os
import time

import pandas as pd

from train import compute_total_power_from_meters, compute_efficiency

SRC = r"D:\Wuhan_Project\new_data\merged_minute_all_with_efficiency.csv"
DST = r"D:\Wuhan_Project\new_data\merged_minute_all_with_efficiency_v2.csv"

flow_cols = ['170:1_瞬时流量', '170:2_瞬时流量', '70:3_瞬时流量']
power_output_cols = [
    '泵1_功率_kVA', '泵2_功率_kVA', '泵3_功率_kVA',
    '泵4_功率_kVA', '泵5_功率_kVA', '泵6_功率_kVA',
    '泵7_功率_kW'
]
meter_cols = ['172:1_泵电度', '172:2_泵电度', '172:3_泵电度',
              '172:4_泵电度', '172:5_泵电度', '172:6_泵电度']
state_cols = ['170:1_泵运行', '170:2_泵运行', '170:3_泵运行',
              '170:4_泵运行', '170:5_泵运行', '170:6_泵运行']


def cross_check_net_gain(df):
    """净增量法交叉校验: 每台泵 累计电度净增 / 总运行时长 → 真实平均功率。

    与分钟级差分应基本一致 (能量守恒), 用于确认重算结果可信。
    """
    print("\n=== 交叉校验: 净增量法 (累计电度净增/运行时长) ===")
    true_powers = {}
    for mcol, scol in zip(meter_cols, state_cols):
        name = mcol.replace('172:', '').replace('_泵电度', '')
        running = df[scol] > 0
        if running.sum() == 0:
            continue
        net = df.loc[running, mcol].iloc[-1] - df.loc[running, mcol].iloc[0]
        hrs = running.sum() / 3600.0
        true_powers[f'泵{name}'] = net / hrs
        print(f"  泵{name}: 净增 {net:9.0f} kWh / {hrs:6.0f} h = {net/hrs:6.1f} kW")
    p7 = df.loc[df['70:7_泵运行'] > 0, '70:7_总有功'].mean() / 1000
    true_powers['泵7'] = p7
    print(f"  泵7(瞬时表): {p7:6.1f} kW")
    return true_powers


def main():
    print("=" * 60)
    print("重新计算功率与效率 (修正电度差分: 秒级→分钟级)")
    print("=" * 60)

    # 1. 加载现有CSV (含清洗后行集与修正压力)
    print(f"\n加载: {SRC}")
    t0 = time.time()
    df = pd.read_csv(SRC, encoding='utf-8-sig')
    print(f"形状: {df.shape}  耗时 {time.time()-t0:.1f}s")
    print(f"时间范围: {df['F_DateTime'].min()} ~ {df['F_DateTime'].max()}")

    # 2. 旧口径基准
    old_total_mean = df['总功率'].mean()
    old_eff = df['总管效率_pct']
    old_eff_mean, old_eff_med = old_eff.mean(), old_eff.median()

    # 3. 修正后的功率 → 效率
    print("\n重算总功率 (分钟级差分)...")
    t0 = time.time()
    df = compute_total_power_from_meters(df)
    print(f"总功率重算完成, 耗时 {time.time()-t0:.1f}s")

    t0 = time.time()
    df = compute_efficiency(df, power_output_cols, flow_cols,
                            pf_estimate=0.88, save_csv=True, csv_path=DST)
    print(f"效率重算完成, 耗时 {time.time()-t0:.1f}s")

    # 4. 交叉校验 (净增量法)
    true_powers = cross_check_net_gain(df)

    # 5. 新旧对比报告
    new_eff = df['总管效率_pct']
    new_total = df['总功率']
    print("\n=== 新旧口径对比 ===")
    print(f"{'指标':<24}{'旧口径':>12}{'新口径':>12}")
    print(f"{'总功率 mean (kW)':<24}{old_total_mean:>12.0f}{new_total.mean():>12.0f}")
    print(f"{'总功率 median (kW)':<24}{'':>12}{new_total.median():>12.0f}")
    print(f"{'总管效率 mean (%)':<24}{old_eff_mean:>12.2f}{new_eff.mean():>12.2f}")
    print(f"{'总管效率 median (%)':<24}{old_eff_med:>12.2f}{new_eff.median():>12.2f}")
    for q in [0.1, 0.25, 0.75, 0.9]:
        print(f"{f'效率 P{int(q*100)} (%)':<24}{'':>12}{new_eff.quantile(q):>12.2f}")
    print(f"{'效率提升倍数':<24}{'':>12}{new_eff.mean()/old_eff_mean:>12.1f}")

    # 6. 差分法 vs 净增量法对比 (确认两种口径一致)
    print("\n=== 分钟级差分 vs 净增量法 (泵运行期间均值) ===")
    for i in range(1, 7):
        name = f'泵{i}'
        running = df[state_cols[i-1]] > 0
        diff_mean = df.loc[running, f'{name}_功率_kW'].mean()
        print(f"  {name}: 差分法 {diff_mean:6.1f} kW | 净增量法 {true_powers[name]:6.1f} kW")

    print(f"\n新CSV已保存: {DST}  (大小 {os.path.getsize(DST)/1e6:.0f} MB)")
    print("\n注意: 旧缓存 processed_cache.parquet 为旧功率口径, 训练前需删除或由版本检查自动重建。")


if __name__ == '__main__':
    main()
