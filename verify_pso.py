# -*- coding: utf-8 -*-
"""验证 PSO 寻优算法的合理性 — 基于真实数据逐行回放比较

对效率 CSV 中的每一行 (秒级真实工况):
  1. 提取真实工况: 三管流量 (作寻优目标), 总管压力, 液位, 实际泵组状态/频率,
     实际效率 (实测: 水力功率/总有功功率)
  2. 用代理模型评估"实际泵组状态" → 模型侧实际效率 (与寻优结果同源, 公平比较)
  3. PSO 寻优: 目标流量 = 真实三管流量, 压力/液位 = 真实值 → 最优方案 (候选 top_k)
  4. 比较三组:
      a. 寻优效率 vs 实测效率   (工程收益, 但混入模型误差)
      b. 寻优效率 vs 模型侧实际效率 (同源比较, 纯寻优收益, 最公平)
      c. 泵组状态一致时: 同状态最佳 vs 模型侧实际效率 (用户关注: 状态一致时是否更优)
  5. 汇总统计: 提升占比/平均幅度/状态一致率/可行性/模型保真度

输出:
  verify_pso_results.csv — 每行回放结果 (--all/--unique_states 时增量追加, 支持断点续跑)
  verify_pso_delta.png   — 提升幅度分布直方图
  控制台汇总统计

用法:
  python verify_pso.py                        # 默认: 按时间采样 200 行
  python verify_pso.py --n 500 --spacing 600  # 自定义采样
  python verify_pso.py --unique_states        # 每个泵组组合取 1 个代表工况 (共 ~61 种)
  python verify_pso.py --unique_condition     # 每个"工况"取 1 个代表行, 工况 = 三管流量×
                                              #  压力×液位×泵组组合 (流量/压力/液位四舍五入
                                              #  后去重, 频率不计入工况); 精度可用
                                              #  --flow_prec/--pressure_prec/--level_prec 调整
  python verify_pso.py --all                  # 跑全部有效工况 (约 300 万行, 耗时很长;
                                              #  结果每 25 行增量落盘, 可 Ctrl+C 后重跑续传)
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pump_inference import PumpInference
from pump_optimize_PSO import optimize_strategy

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

DATA = r"D:\Wuhan_Project\new_data\merged_minute_all_with_efficiency.csv"
MODEL_PATH = r"D:\Wuhan_Project\models\model_v2_combo_split.pt"
OUT_CSV = r"D:\Wuhan_Project\verify_pso_results.csv"
OUT_PNG = r"D:\Wuhan_Project\verify_pso_delta.png"

STATE_COLS = [f"170:{i}_泵运行" for i in range(1, 7)] + ["70:7_泵运行"]
FREQ_COLS = [f"170:{i}_运行频率" for i in range(1, 7)] + ["70:7_运行频率"]
FLOW_COLS = ["170:1_瞬时流量", "170:2_瞬时流量", "70:3_瞬时流量"]
PRESSURE_COL = "170:总管压力"
LEVEL_COL = "170:吸水井液位"
EFF_COL = "总管效率_pct"


def states_str(s):
    return "".join(str(int(x)) for x in s)


def load_valid_df():
    """读取效率 CSV → 过滤无效行 → 返回全部有效工况 (按时间排序)。"""
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df["F_DateTime"] = pd.to_datetime(df["F_DateTime"])
    df = df.sort_values("F_DateTime").reset_index(drop=True)

    # ── 有效性过滤 ──
    df = df[df[EFF_COL].notna()]
    for c in STATE_COLS + FREQ_COLS + FLOW_COLS + [PRESSURE_COL, LEVEL_COL]:
        df = df[df[c].notna()]
    df = df[(df[PRESSURE_COL].astype(float) >= 0.1) & (df[PRESSURE_COL].astype(float) <= 0.5)]  # 压力物理范围
    df = df[(df[LEVEL_COL].astype(float) >= 1.0) & (df[LEVEL_COL].astype(float) <= 5.0)]      # 液位物理范围
    for c in FLOW_COLS:
        df = df[(df[c].astype(float) >= 0) & (df[c].astype(float) <= 10000)]                  # 流量物理范围

    # ── 泵状态/频率有效性 (numpy 显式逻辑, 避免 pandas where 列名对齐陷阱) ──
    st = df[STATE_COLS].astype(float).to_numpy()
    fq = df[FREQ_COLS].astype(float).to_numpy()
    run = st > 0.5
    freq_ok = ((fq >= 30.0) & (fq <= 50.0)) | ~run     # 停泵频率不限, 运行泵须 30~50 Hz
    keep = (st.sum(axis=1) >= 1) & freq_ok.all(axis=1)  # 至少 1 台泵运行
    df = df[keep].reset_index(drop=True)
    print(f"过滤后可用工况: {len(df)}")
    return df


def sample_rows(df, n, spacing_s):
    """按 ≥spacing_s 间隔采样 n 行 (去秒级自相关)。"""
    ts = df["F_DateTime"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    idx, cur = [], 0
    while cur < len(ts) and len(idx) < n:
        idx.append(cur)
        cur = int(np.searchsorted(ts, ts[cur] + spacing_s, side="left"))
    print(f"采样 {len(idx)} 行 (间隔 ≥ {spacing_s}s)")
    return df.iloc[idx]


def first_row_per_state(df):
    """每个泵组组合取第一个出现的行为代表工况 (按时间排序后)。"""
    key = df[STATE_COLS].astype(float).round().astype(int).astype(str).agg("".join, axis=1)
    idx = key.drop_duplicates(keep="first").index
    return df.loc[idx]


def _round_half_up(s, digits):
    """真正的四舍五入 (pandas/numpy 的 round 是银行家舍入: 0.5 舍成偶数)。"""
    scale = 10.0 ** digits
    vals = np.floor(s.astype(float).to_numpy() * scale + 0.5) / scale
    return pd.Series(vals, index=s.index)


def condition_key(df, flow_prec=0, pressure_prec=2, level_prec=2):
    """工况键 = 三管流量(四舍五入) | 压力(四舍五入) | 液位(四舍五入) | 泵组组合。

    频率不算在工况里 (用户口径): 相同流量/压力/液位/泵组、不同频率视为同一工况。
    """
    parts = [_round_half_up(df[c], flow_prec).astype(str) for c in FLOW_COLS]
    parts.append(_round_half_up(df[PRESSURE_COL], pressure_prec).astype(str))
    parts.append(_round_half_up(df[LEVEL_COL], level_prec).astype(str))
    st_key = df[STATE_COLS].astype(float).round().astype(int).astype(str).agg("".join, axis=1)
    parts.append(st_key)
    return pd.concat(parts, axis=1).agg("|".join, axis=1)


def first_row_per_condition(df, flow_prec=0, pressure_prec=2, level_prec=2):
    """每个"工况"取第一个出现的行为代表。

    工况 = 三管流量(四舍五入) × 总管压力(四舍五入) × 液位(四舍五入) × 泵组组合,
    频率不计入。返回 (代表行 DataFrame, 工况总数)。
    """
    key = condition_key(df, flow_prec, pressure_prec, level_prec)
    idx = key.drop_duplicates(keep="first").index
    return df.loc[idx], len(idx)


def main():
    ap = argparse.ArgumentParser(description="PSO 寻优合理性验证")
    ap.add_argument("--n", type=int, default=200, help="验证行数 (默认 200, 仅采样模式)")
    ap.add_argument("--spacing", type=int, default=300, help="采样最小间隔秒 (默认 300)")
    ap.add_argument("--all", action="store_true",
                    help="跑所有有效工况 (不采样; 结果增量落盘, 可中断后重跑续传)")
    ap.add_argument("--unique_states", action="store_true",
                    help="每个泵组组合取一个代表工况 (共 ~61 行)")
    ap.add_argument("--unique_condition", action="store_true",
                    help="每个工况取一个代表行 (工况 = 三管流量×压力×液位×泵组组合, "
                         "流量/压力/液位四舍五入后去重, 频率不计入)")
    ap.add_argument("--flow_prec", type=int, default=0, help="流量四舍五入位数 (默认 0, 取整 m3/h)")
    ap.add_argument("--pressure_prec", type=int, default=2, help="压力四舍五入位数 (默认 2, 0.01 MPa)")
    ap.add_argument("--level_prec", type=int, default=2, help="液位四舍五入位数 (默认 2, 0.01 m)")
    ap.add_argument("--pop", type=int, default=40, help="PSO 种群 (默认 40)")
    ap.add_argument("--gens", type=int, default=200, help="PSO 迭代代数 (默认 200)")
    ap.add_argument("--topk", type=int, default=12, help="PSO 返回候选状态数 (默认 12)")
    ap.add_argument("--seed", type=int, default=42, help="随机种子 (默认 42)")
    args = ap.parse_args()

    model = PumpInference(MODEL_PATH)
    print(f"设备: {model.device}, 模型已加载")

    df_valid = load_valid_df()
    if args.unique_condition:
        rows, n_cond = first_row_per_condition(df_valid, args.flow_prec,
                                               args.pressure_prec, args.level_prec)
        print(f"唯一工况 (流量×压力×液位×泵组, 四舍五入后去重, 频率不计): {n_cond} 行")
        print(f"  (约 {n_cond * 2 / 3600:.1f} 小时 @ 2s/行, 支持断点续跑)")
    elif args.unique_states:
        rows = first_row_per_state(df_valid)
        print(f"唯一泵组组合工况: {len(rows)} 行")
    elif args.all:
        rows = df_valid
        print(f"全部有效工况: {len(rows)} 行 "
              f"(约 {len(rows) * 2 / 3600:.0f} 小时 @ 2s/行, 建议中断分段跑, 重跑自动续传)")
    else:
        rows = sample_rows(df_valid, args.n, args.spacing)
    if len(rows) == 0:
        print("无有效行, 退出")
        sys.exit(1)

    # ── 断点续跑: --all/--unique_states 增量追加, 已处理时间戳跳过; 采样模式重置输出 ──
    resume = args.all or args.unique_states or args.unique_condition
    done = set()
    if resume and os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV, encoding="utf-8-sig")
        done = set(prev["datetime"].astype(str))
        dup = sum(1 for t in rows["F_DateTime"] if str(t) in done)
        print(f"断点续跑: 结果 CSV 已有 {len(done)} 行, 本次跳过其中 {dup} 个重复工况")
    elif not resume and os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)
        print(f"采样模式: 重置输出 {OUT_CSV}")

    REC_COLUMNS = ["datetime", "f1", "f2", "f3", "pressure", "level",
                   "states_actual", "states_opt", "states_same",
                   "eff_actual", "eff_sur_actual", "eff_opt", "eff_same_state",
                   "opt_feasible", "opt_violation", "opt_pred_total",
                   "d_vs_measured", "d_vs_surrogate", "d_same_state"]

    def flush():
        """把已收集的结果追加写入 CSV (文件不存在时写表头)。

        注意: header 判定必须在 open 之前——"a" 模式会先创建文件,
        在 open 之后再 exists() 永远为 True, 表头会被吞掉。
        """
        frame = pd.DataFrame(results, columns=REC_COLUMNS)
        header = not os.path.exists(OUT_CSV)
        with open(OUT_CSV, "a", encoding="utf-8-sig", newline="") as f:
            frame.to_csv(f, index=False, header=header)
        results.clear()

    results, t_start, n_new, skipped = [], time.perf_counter(), 0, 0

    for i in range(len(rows)):
        r = rows.iloc[i]
        if str(r["F_DateTime"]) in done:
            skipped += 1
            continue
        f = [float(r[c]) for c in FLOW_COLS]
        pressure = float(r[PRESSURE_COL])
        level = float(r[LEVEL_COL])
        st_act = np.array([int(round(float(r[c]))) for c in STATE_COLS], dtype=np.int64)
        fq_act = np.array([float(r[c]) if st_act[j] else 0.0
                           for j, c in enumerate(FREQ_COLS)], dtype=np.float64)
        eff_act = float(r[EFF_COL])

        # ① 代理模型评估实际泵组状态 (同源基线)
        _, _, _, _, eff_sur, _ = model.predict(st_act, fq_act, pressure, level)

        # ② PSO 寻优 (验证用较小种群/代数)
        res = optimize_strategy(model, target_flows=f, pressure=pressure, level=level,
                                pop_size=args.pop, n_generations=args.gens,
                                seed=args.seed, top_k=args.topk)
        best = res["candidates"][0]

        # ③ 与实际状态一致的候选里的最优 (用户关注: 状态一致时是否更优)
        same = [c for c in res["candidates"] if tuple(c["states"]) == tuple(st_act)]
        same_best = max(same, key=lambda c: c["efficiency"]) if same else None

        results.append({
            "datetime": r["F_DateTime"],
            "f1": f[0], "f2": f[1], "f3": f[2],
            "pressure": pressure, "level": level,
            "states_actual": states_str(st_act),
            "states_opt": states_str(best["states"]),
            "states_same": states_str(same_best["states"]) if same_best else "",
            "eff_actual": eff_act,
            "eff_sur_actual": eff_sur,
            "eff_opt": best["efficiency"],
            "eff_same_state": same_best["efficiency"] if same_best else np.nan,
            "opt_feasible": int(best["feasible"]),
            "opt_violation": best["violation"],
            "opt_pred_total": best["total_flow"],
            "d_vs_measured": best["efficiency"] - eff_act,
            "d_vs_surrogate": best["efficiency"] - eff_sur,
            "d_same_state": (same_best["efficiency"] - eff_sur) if same_best else np.nan,
        })
        n_new += 1

        # 每 25 行增量落盘一次 (断点续跑的基础), 顺带打印进度
        if n_new % 25 == 0:
            flush()
            el = time.perf_counter() - t_start
            left = len(rows) - i - 1 - skipped
            print(f"  新处理 {n_new} 行 (跳过 {skipped}), 耗时 {el:.0f}s, "
                  f"剩余约 {el / n_new * left:.0f}s")

    if results:
        flush()
    dfr = pd.read_csv(OUT_CSV, encoding="utf-8-sig")
    print(f"\n逐行结果已保存: {OUT_CSV} (累计 {len(dfr)} 行)")
    if "d_vs_measured" not in dfr.columns:
        print(f"错误: 结果 CSV 无数据列 (可能是旧版无表头残留文件), 请删除 {OUT_CSV} 后重跑")
        sys.exit(1)

    # ═══════════════ 汇总统计 ═══════════════
    n = len(dfr)
    d_meas = dfr["d_vs_measured"]
    d_sur = dfr["d_vs_surrogate"]
    d_same = dfr["d_same_state"].dropna()
    eff_sur_err = (dfr["eff_sur_actual"] - dfr["eff_actual"]).abs()

    print("\n" + "=" * 66)
    print(f"  PSO 寻优合理性验证 — 共 {n} 个真实工况")
    print("=" * 66)
    print(f"\n[模型保真度] 代理模型评估实际状态 vs 实测效率")
    print(f"  MAE = {eff_sur_err.mean():.2f} pp, 偏差均值 = {(dfr['eff_sur_actual']-dfr['eff_actual']).mean():+.2f} pp")
    print(f"  (比较 b/c 同为模型输出, 不受此影响; 比较 a 需注意模型误差)")

    print(f"\n[a] 寻优效率 vs 实测效率   (工程收益, 含模型误差)")
    print(f"  平均 Δ = {d_meas.mean():+.2f} pp, 中位数 Δ = {d_meas.median():+.2f} pp")
    print(f"  提升占比: {(d_meas > 0).mean()*100:.1f}%   (Δ > 0 为提升)")

    print(f"\n[b] 寻优效率 vs 模型侧实际效率   (同源比较, 纯寻优收益)")
    print(f"  平均 Δ = {d_sur.mean():+.2f} pp, 中位数 Δ = {d_sur.median():+.2f} pp")
    print(f"  提升占比: {(d_sur > 0).mean()*100:.1f}%")

    print(f"\n[c] 泵组状态一致时: 同状态最佳 vs 模型侧实际效率")
    print(f"  找到同状态候选的行占比: {len(d_same)/n*100:.1f}%")
    print(f"  平均 Δ = {d_same.mean():+.2f} pp, 中位数 Δ = {d_same.median():+.2f} pp")
    print(f"  提升占比: {(d_same > 0).mean()*100:.1f}%")

    print(f"\n[泵组状态] 寻优最佳状态 == 实际状态 的行占比: {(dfr['states_opt']==dfr['states_actual']).mean()*100:.1f}%")
    print(f"[可行性] 寻优方案流量偏差在容差内 (feasible) 占比: {dfr['opt_feasible'].mean()*100:.1f}%")
    print(f"[效率水平] 实测效率均值 {dfr['eff_actual'].mean():.1f}% | 寻优效率均值 {dfr['eff_opt'].mean():.1f}%")

    # 状态一致时的 a 比较 (实测口径)
    mask_same = dfr["states_opt"] == dfr["states_actual"]
    if mask_same.any():
        dm = dfr.loc[mask_same, "d_vs_measured"]
        print(f"\n[状态一致子集 {mask_same.sum()} 行] 寻优效率 vs 实测效率")
        print(f"  平均 Δ = {dm.mean():+.2f} pp, 提升占比 {(dm > 0).mean()*100:.1f}%")

    # ═══════════════ 直方图 ═══════════════
    _, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, title in [
        (axes[0], d_sur, "b. 同源比较 (寻优 vs 模型侧实际)"),
        (axes[1], d_meas, "a. 实测口径 (寻优 vs 实测)"),
    ]:
        ax.hist(data, bins=40, color="steelblue", edgecolor="black", alpha=0.75)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
        ax.axvline(data.mean(), color="green", linestyle="--", linewidth=1.2,
                   label=f"均值 {data.mean():+.2f}")
        ax.set_title(title)
        ax.set_xlabel("效率提升 Δ (百分点)")
        ax.set_ylabel("工况数")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.suptitle(f"PSO 寻优效率提升分布 (n={n})", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n提升分布图已保存: {OUT_PNG}")
    print(f"总耗时 {time.perf_counter()-t_start:.0f}s")


if __name__ == "__main__":
    main()
