"""
每日泵开泵策略 — Transformer 流量预测 + PSO 寻优 融合脚本
=================================================

把 inference_transformer.py 与 pump_optimize_PSO.py 串成完整流水线:

  ① Transformer 流量预测 (inference_transformer.FlowPredictor)
       → 未来 48 个时间点 (30min × 24h) 的 总管流量 Total_Flow + 分时压力 Pressure
  ② 三管分流: Transformer 只预测总管流量, 170:1/170:2/70:3 的比例按最近
       ratio_lookback_days 天历史数据的"同一时段(30min槽位)"均值估计
       (70:3 深夜常为 0, 按时段比例比全天平均更贴近真实)
  ③ 逐点 PSO 寻优 (pump_optimize_PSO.optimize_strategy)
       → 每个时间点输出多条候选策略 (默认 top 5, 不同泵状态组合)
  ④ 全局策略选择 (动态规划): 对 48 个时间点的候选做全局路径寻优,
       代价 = −效率×w_eff + 流量超差×w_viol + 泵状态翻转×w_state
              + 频率变化×w_freq, 选出每个时间点最终采用哪条候选
       (泵状态切换最少 / 频率切换最少 / 效率最高, 权重越大优先级越高)

每点寻优输入 (目标工况):
  - 目标流量: 170:1 / 170:2 / 70:3 (m3/h) = 预测总管流量 × 该时段分流比例
  - 总管压力: 分时压力时段表的目标压力值 (predict_pressure 生成)
  - 液位:     默认 3.58 m (泵站基准液位, --level 可改)。液位取基准时压力
              修正量 = 0, 即给定压力直接按"修正后压力"送入寻优模型
              (与 pump_inference._correct_pressure 的约定一致:
               修正后压力 = 原压力 - (液位 - 3.58) / 102)

输出:
  - daily_pump_schedule.csv           : 48 个时间点逐点策略 (DP 选中的方案)
  - daily_pump_schedule_candidates.csv: 每点 top_k 条候选策略 + 是否被选中
  - daily_pump_schedule_blocks.csv    : 相同 (状态,频率) 的连续时段合并
                                         (便于现场按时间段执行)

时段划分约定:
  每个预测时刻 t 的预测流量/目标压力在窗口 [t-15min, t+15min) 内生效;
  块起点 = 首点时刻 - 15min, 块终点 = 末点时刻 + 15min (如 00:30 点 → 00:15~00:45)。
  相邻时间点的泵组组合与各泵频率完全一致时合并为一段,
  块内流量/压力取各点均值 (逐点值见 daily_pump_schedule.csv)。

注意:
  - 逐点寻优耗时为 pop×gen 批量评估 × 48 点, 建议先 --fast 试跑
  - DP 代价权重 (--w_state/--w_freq/--w_eff/--w_viol) 可调, 默认:
    泵状态翻转 1.0/台 > 频率变化 0.01/Hz > 效率 1.0/%, 流量超差 1.0/(m3/h)
    (权重越大该原则优先级越高)
  - 寻优结果是数据驱动代理模型的建议, 上线前需结合实际工况校核
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

# 保证能从本目录导入两个待融合模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference_transformer import FlowPredictor, DEFAULT_DATA, DEFAULT_RESULT_DIR
from pump_optimize_PSO import optimize_strategy
from pump_inference import PumpInference

# ============================================================================
# 默认参数
# ============================================================================
DEFAULT_PUMP_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "models", "model_v2_combo_split.pt")
LEVEL_DEFAULT = 3.58      # 泵站基准液位 (m): 液位缺失 → 压力修正量=0, 压力按修正后压力处理
FLOW_COLS = ["170:1_瞬时流量", "170:2_瞬时流量", "70:3_瞬时流量"]
FREQ_MINUTES = 30         # 与 Transformer 训练一致的重采样频率
POINTS_PER_DAY = (24 * 60) // FREQ_MINUTES   # 48


# ============================================================================
# ① 三管分流比例估计 (按 30min 时段槽位)
# ============================================================================

def compute_flow_ratios(df_raw, lookback_days, freq_minutes=FREQ_MINUTES):
    """由历史原始数据估计 170:1/170:2/70:3 三管分流比例。

    每根管独立做 freq_minutes 均值重采样, 取最近 lookback_days 天; 按天内的
    时段槽位 (0..points_per_day-1) 分组求多天均值, 得到每个槽位的三管比例。
    槽位缺失或总量为 0 时退回全天平均比例; 全天也为 0 时退回均分。

    返回 (global_ratio, slot_ratios):
      global_ratio : (3,) 全天平均比例
      slot_ratios  : (points_per_day, 3) 各时段槽位的比例 (行 i = 槽位 i)
    """
    df = df_raw.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        for ts_col in ("F_DateTime", "时间", "timestamp"):
            if ts_col in df.columns:
                df[ts_col] = pd.to_datetime(df[ts_col])
                df = df.set_index(ts_col)
                break
        else:
            raise ValueError("输入数据必须包含时间列 (F_DateTime / 时间 / timestamp) "
                             "或 DatetimeIndex 索引")
    df = df.sort_index()

    missing = [c for c in FLOW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少流量列: {missing}, 需要 {FLOW_COLS}")

    points_per_day = (24 * 60) // freq_minutes
    flows = df[FLOW_COLS].resample(f"{freq_minutes}min").mean()
    tail = flows.iloc[-lookback_days * points_per_day:]
    if len(tail) == 0:
        raise ValueError(f"历史数据不足: 需要 ≥ {lookback_days} 天 "
                         f"({lookback_days * points_per_day} 个 {freq_minutes}min 点)")

    # 全天平均比例 (兜底)
    global_ratio = tail.mean().values
    gs = global_ratio.sum()
    if gs <= 0:
        global_ratio = np.ones(3) / 3.0
    else:
        global_ratio /= gs

    # 按天内的时段槽位求多天均值 → 每个槽位的比例
    slot = (tail.index.hour * 60 + tail.index.minute) // freq_minutes
    slot_means = tail.groupby(slot).mean()
    slot_ratios = np.zeros((points_per_day, 3))
    for s in range(points_per_day):
        if s in slot_means.index:
            r = slot_means.loc[s].values
            rs = r.sum()
            slot_ratios[s] = r / rs if rs > 0 else global_ratio
        else:
            slot_ratios[s] = global_ratio
    return global_ratio, slot_ratios


def split_total_flow(totals, slot_ratios, slot_indices):
    """把总管流量按槽位比例拆成三管目标流量 (m3/h): (n,) × (n,3) → (n,3)"""
    totals = np.asarray(totals, dtype=np.float64)
    ratios = slot_ratios[np.asarray(slot_indices)]
    return totals[:, None] * ratios


# ============================================================================
# ② 逐点寻优 + 结果整理
# ============================================================================

def optimize_point(pump_model, target_flows, pressure, level,
                   pop_size, n_generations, top_k, seed=42):
    """单时间点 PSO 寻优, 返回最优候选字典 (口径同 pump_optimize_PSO)。"""
    return optimize_strategy(pump_model, target_flows=target_flows, pressure=pressure,
                             level=level, pop_size=pop_size, n_generations=n_generations,
                             seed=seed, top_k=top_k)


def _close_block(blocks, start_ts, end_ts, n_points, states_str, freqs_str,
                 effs, kwts, feas, flows, pressures):
    blocks.append({
        "start": start_ts,
        "end": end_ts,
        "num_points": n_points,
        "flow_mean": round(float(np.mean(flows)), 1),          # 块内各点预测流量均值 (m³/h)
        "pressure_mean": round(float(np.mean(pressures)), 3),  # 块内各点目标压力均值 (MPa)
        "states": states_str,
        "freqs": freqs_str,
        "efficiency_mean": round(float(np.mean(effs)), 2),
        "kwt_mean": round(float(np.mean(kwts)), 2),
        "all_feasible": all(feas),
    })


# ============================================================================
# ④ 全局策略选择 (动态规划): 泵状态切换/频率切换最少 + 效率最高
# ============================================================================

def transition_cost(prev_cand, curr_cand, w_state, w_freq):
    """相邻两点候选之间的切换代价:
    - 泵状态切换: 每台泵 开/关 翻转计 w_state
    - 频率切换:   仅统计前后两点都在运行的泵, 频率差值 (Hz) 计 w_freq
    """
    s1, s2 = prev_cand["states"], curr_cand["states"]
    f1, f2 = prev_cand["freqs"], curr_cand["freqs"]
    state_cost = w_state * float(np.sum(s1 != s2))
    both_on = (s1 == 1) & (s2 == 1)
    freq_cost = w_freq * float(np.sum(np.abs(f1 - f2)[both_on]))
    return state_cost + freq_cost


def select_global_path(cands_per_point, w_state, w_freq, w_eff, w_viol):
    """动态规划: 从每点的多条候选里选出 48 点全局代价最小的策略路径。

    总代价 = Σ_t (−w_eff×效率 + w_viol×流量超差量) + Σ_t 相邻切换代价。
    返回: 每点选中的候选下标列表 (长度 = 点数)。
    """
    T = len(cands_per_point)
    dp = []   # dp[t][j] = (到 t 点选候选 j 的累计代价, 上一时刻候选下标)
    for t in range(T):
        node = [(-w_eff * c["efficiency"] + w_viol * c["violation"])
                for c in cands_per_point[t]]
        if t == 0:
            dp.append([(nc, -1) for nc in node])
            continue
        cur = []
        for j, nc in enumerate(node):
            best_cost, best_k = float("inf"), -1
            for k in range(len(cands_per_point[t - 1])):
                cost = (dp[t - 1][k][0] + nc
                        + transition_cost(cands_per_point[t - 1][k],
                                          cands_per_point[t][j], w_state, w_freq))
                if cost < best_cost:
                    best_cost, best_k = cost, k
            cur.append((best_cost, best_k))
        dp.append(cur)

    # 回溯
    chosen = [0] * T
    j = int(np.argmin([c[0] for c in dp[T - 1]]))
    for t in range(T - 1, -1, -1):
        chosen[t] = j
        j = dp[t][j][1]
    return chosen


def path_switch_metrics(cands_per_point, chosen):
    """统计选中路径: (泵切换总次数, 频率变化总量 Hz, 平均效率 %)。"""
    n_toggle, freq_delta, effs = 0, 0.0, []
    for t, j in enumerate(chosen):
        c = cands_per_point[t][j]
        effs.append(c["efficiency"])
        if t > 0:
            p = cands_per_point[t - 1][chosen[t - 1]]
            n_toggle += int(np.sum(p["states"] != c["states"]))
            both = (p["states"] == 1) & (c["states"] == 1)
            freq_delta += float(np.sum(np.abs(p["freqs"] - c["freqs"])[both]))
    return n_toggle, freq_delta, float(np.mean(effs))


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="每日泵开泵策略: Transformer 流量预测 + PSO 寻优")
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="原始数据 CSV 路径 (与训练数据格式一致)")
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR,
                        help="Transformer 训练结果目录 (含 best_seq2seq_model.pth 与 scaler.pkl)")
    parser.add_argument("--pump_model", default=DEFAULT_PUMP_MODEL,
                        help="泵站代理模型权重路径")
    parser.add_argument("--level", type=float, default=LEVEL_DEFAULT,
                        help=f"吸水井液位 (m); 默认 {LEVEL_DEFAULT} = 基准液位, "
                             f"压力修正量=0 → 压力直接按修正后压力处理")
    parser.add_argument("--ratio_lookback_days", type=int, default=7,
                        help="三管分流比例估计用的历史天数 (默认 7)")
    parser.add_argument("--pop", type=int, default=None, help="PSO 种群规模 (默认 50)")
    parser.add_argument("--gen", type=int, default=None, help="PSO 迭代代数 (默认 200)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="每点 PSO 返回的候选策略数 (不同泵状态组合, 默认 5)")
    parser.add_argument("--w_state", type=float, default=1.0,
                        help="DP: 泵状态翻转代价 (每台泵, 默认 1.0)")
    parser.add_argument("--w_freq", type=float, default=0.01,
                        help="DP: 频率变化代价 (每 Hz, 仅统计连续运行的泵, 默认 0.01; "
                             "1 台泵满档变频 50Hz=0.5, 低于 1 次状态翻转)")
    parser.add_argument("--w_eff", type=float, default=1.0,
                        help="DP: 效率权重 (每 % 效率, 默认 1.0)")
    parser.add_argument("--w_viol", type=float, default=1.0,
                        help="DP: 流量超差惩罚 (每 m3/h 超出容差, 默认 1.0)")
    parser.add_argument("--fast", action="store_true",
                        help="快速模式 (pop=30, gen=80, 约 1/4 耗时; 显式给定 --pop/--gen 时以显式值为准)")
    parser.add_argument("--out", default="daily_pump_schedule.csv",
                        help="逐点策略 CSV 输出路径")
    args = parser.parse_args()

    # --fast 只作为默认值兜底: 显式给出 --pop/--gen 时不被覆盖
    if args.pop is None:
        args.pop = 30 if args.fast else 50
    if args.gen is None:
        args.gen = 80 if args.fast else 200

    # ── ① Transformer 流量预测 ──
    print("\n" + "=" * 72)
    print("① Transformer 流量预测 (inference_transformer.FlowPredictor)")
    print("=" * 72)
    predictor = FlowPredictor(args.result_dir)
    pred = predictor.predict(args.data)          # DataFrame: Total_Flow (48 行)
    pred = predictor.predict_pressure(pred)      # + Pressure 列 (分时压力目标)
    print(f"   预测点数: {len(pred)}  ({pred.index[0]} ~ {pred.index[-1]})")

    # ── ② 三管分流比例 ──
    print("\n" + "=" * 72)
    print(f"② 三管分流比例估计 (最近 {args.ratio_lookback_days} 天, 按 {FREQ_MINUTES}min 时段)")
    print("=" * 72)
    raw = pd.read_csv(args.data, encoding="utf-8-sig")
    global_ratio, slot_ratios = compute_flow_ratios(raw, args.ratio_lookback_days)
    print(f"   全天平均比例: 170:1={global_ratio[0]:.3f}  "
          f"170:2={global_ratio[1]:.3f}  70:3={global_ratio[2]:.3f}")

    # ── ③ 泵站代理模型 ──
    print("\n" + "=" * 72)
    print("③ 泵站代理模型 (PumpInference)")
    print("=" * 72)
    pump_model = PumpInference(args.pump_model)
    pump_model.info()

    # ── ④ 逐点 PSO 寻优 (每点输出 top_k 条候选) ──
    print("\n" + "=" * 72)
    print(f"④ 逐点 PSO 寻优 (pop={args.pop}, gen={args.gen}, top_k={args.top_k}, "
          f"液位={args.level:.2f} m)")
    print("=" * 72)
    n_points = len(pred)
    cands_per_point = []        # 每点: [{states, freqs, efficiency, kwt, feasible, violation, ...}, ...]
    t_start = time.perf_counter()
    for i, (ts, r) in enumerate(pred.iterrows()):
        total_flow = float(r["Total_Flow"])
        pressure = float(r["Pressure"])
        slot = i % POINTS_PER_DAY
        target_flows = total_flow * slot_ratios[slot]          # (3,) 三管目标流量

        result = optimize_point(pump_model, target_flows=target_flows, pressure=pressure,
                                level=args.level, pop_size=args.pop,
                                n_generations=args.gen, top_k=args.top_k)
        cands = result["candidates"] if result.get("candidates") else [result]
        cands_per_point.append(cands)

        # 进度行显示该点效率最高的候选 (candidates[0])
        best = cands[0]
        states_str = "".join(str(int(s)) for s in best["states"])
        freqs_str = " ".join(f"{f:.0f}" for f in best["freqs"])
        elapsed = time.perf_counter() - t_start
        eta = elapsed / (i + 1) * (n_points - i - 1)
        flag = "OK" if best["feasible"] else "XX"   # GBK 控制台不可用 ✓/✗, 用 ASCII
        print(f"[{i+1:3d}/{n_points}] {ts.strftime('%H:%M')}  "
              f"目标 {target_flows[0]:6.0f}/{target_flows[1]:6.0f}/{target_flows[2]:6.0f}  "
              f"状态 {states_str}  频率 {freqs_str}  "
              f"效率 {best['efficiency']:.1f}%  千吨电耗 {best['kwt']:.1f} {flag}  "
              f"候选 {len(cands)} 条  (累计 {elapsed:.0f}s, ETA {eta:.0f}s)")
        sys.stdout.flush()

    # ── ⑤ DP 全局选择 ──
    print("\n" + "=" * 72)
    print(f"⑤ 全局策略选择 (DP): 泵状态翻转 {args.w_state}/台 + 频率变化 {args.w_freq}/Hz "
          f"- 效率 {args.w_eff}/% + 超差 {args.w_viol}/(m3/h)")
    print("=" * 72)
    chosen = select_global_path(cands_per_point, args.w_state, args.w_freq,
                                args.w_eff, args.w_viol)

    # 对照基线: 逐点只选效率最高的候选 (不关心切换)
    baseline = [int(np.argmax([c["efficiency"] for c in cands]))
                for cands in cands_per_point]
    b_toggle, b_freq, b_eff = path_switch_metrics(cands_per_point, baseline)
    s_toggle, s_freq, s_eff = path_switch_metrics(cands_per_point, chosen)
    print(f"逐点效率最高 (基线): 泵切换 {b_toggle} 次, 频率变化 {b_freq:.0f} Hz, "
          f"平均效率 {b_eff:.1f}%")
    print(f"DP 全局最优       : 泵切换 {s_toggle} 次, 频率变化 {s_freq:.0f} Hz, "
          f"平均效率 {s_eff:.1f}%")
    print(f"节省: 泵切换 {b_toggle - s_toggle} 次, 频率变化 {b_freq - s_freq:.0f} Hz")

    # ── ⑥ 逐点结果 (DP 选中方案) + 全部候选 ──
    rows = []
    cand_rows = []      # 全部候选 (供多策略 CSV)
    for i, (ts, r) in enumerate(pred.iterrows()):
        total_flow = float(r["Total_Flow"])
        pressure = float(r["Pressure"])
        slot = i % POINTS_PER_DAY
        target_flows = total_flow * slot_ratios[slot]
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        for j, c in enumerate(cands_per_point[i]):
            states_str = "".join(str(int(s)) for s in c["states"])
            freqs_str = " ".join(f"{f:.0f}" for f in c["freqs"])
            is_sel = int(j == chosen[i])
            cand_rows.append({
                "timestamp": ts_str,
                "rank": j + 1,
                "selected": is_sel,
                "states": states_str,
                "freqs": freqs_str,
                "efficiency": round(float(c["efficiency"]), 2),
                "kwt": round(float(c["kwt"]), 2),
                "feasible": c["feasible"],
                "pred_170_1": round(float(c["pred_flows"][0]), 1),
                "pred_170_2": round(float(c["pred_flows"][1]), 1),
                "pred_70_3": round(float(c["pred_flows"][2]), 1),
                "deviation": ", ".join(f"{d:+.1f}" for d in c["deviation"]),
                "violation": round(float(c["violation"]), 1),
            })
            if is_sel:
                rows.append({
                    "timestamp": ts_str,
                    "rank": j + 1,
                    "Total_Flow": round(total_flow, 1),
                    "Pressure": round(pressure, 3),
                    "q_170_1": round(float(target_flows[0]), 1),
                    "q_170_2": round(float(target_flows[1]), 1),
                    "q_70_3": round(float(target_flows[2]), 1),
                    "states": states_str,
                    "freqs": freqs_str,
                    "pred_170_1": round(float(c["pred_flows"][0]), 1),
                    "pred_170_2": round(float(c["pred_flows"][1]), 1),
                    "pred_70_3": round(float(c["pred_flows"][2]), 1),
                    "efficiency": round(float(c["efficiency"]), 2),
                    "kwt": round(float(c["kwt"]), 2),
                    "feasible": c["feasible"],
                    "num_candidates": len(cands_per_point[i]),
                })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n[OK] 逐点策略 (DP 选中) 已保存: {args.out} ({len(df_out)} 行)")

    cand_path = args.out.replace(".csv", "_candidates.csv")
    pd.DataFrame(cand_rows).to_csv(cand_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 全部候选策略已保存: {cand_path} ({len(cand_rows)} 行)")

    # ── ⑦ 连续时段合并 ──
    # 时段划分: 每个预测时刻 t 对应执行窗口 [t-15min, t+15min) (该时刻的预测流量/
    # 目标压力在该窗口内生效); 故块起点 = 首点时刻 - 15min, 块终点 = 末点时刻 + 15min。
    # 合并条件: 相邻时间点的泵组组合与各泵频率完全一致。
    half = FREQ_MINUTES // 2
    blocks = []
    if len(df_out) > 0:
        cur_states, cur_freqs = df_out.iloc[0]["states"], df_out.iloc[0]["freqs"]
        cur_start = (pd.to_datetime(df_out.iloc[0]["timestamp"])
                     - pd.Timedelta(minutes=half)).strftime("%Y-%m-%d %H:%M:%S")
        effs, kwts, feas = [], [], []
        flows, pressures = [], []
        prev_ts = pd.to_datetime(df_out.iloc[0]["timestamp"])   # 块内最后点的时刻

        for _, row in df_out.iterrows():
            if row["states"] == cur_states and row["freqs"] == cur_freqs:
                effs.append(row["efficiency"]); kwts.append(row["kwt"])
                feas.append(row["feasible"])
                flows.append(row["Total_Flow"]); pressures.append(row["Pressure"])
                prev_ts = pd.to_datetime(row["timestamp"])
                continue
            # 切换组合: 前一段覆盖到末点 + 15min
            end_ts = (prev_ts + pd.Timedelta(minutes=half)).strftime("%Y-%m-%d %H:%M:%S")
            _close_block(blocks, cur_start, end_ts, len(effs), cur_states, cur_freqs,
                         effs, kwts, feas, flows, pressures)
            cur_states, cur_freqs = row["states"], row["freqs"]
            cur_start = (pd.to_datetime(row["timestamp"])
                         - pd.Timedelta(minutes=half)).strftime("%Y-%m-%d %H:%M:%S")
            effs, kwts, feas = [row["efficiency"]], [row["kwt"]], [row["feasible"]]
            flows, pressures = [row["Total_Flow"]], [row["Pressure"]]
            prev_ts = pd.to_datetime(row["timestamp"])

        if len(effs) > 0:
            _close_block(blocks, cur_start,
                         (prev_ts + pd.Timedelta(minutes=half)).strftime("%Y-%m-%d %H:%M:%S"),
                         len(effs), cur_states, cur_freqs, effs, kwts, feas, flows, pressures)

    df_blocks = pd.DataFrame(blocks)
    if len(df_blocks) > 0:
        blocks_path = args.out.replace(".csv", "_blocks.csv")
        df_blocks.to_csv(blocks_path, index=False, encoding="utf-8-sig")
        print(f"[OK] 连续时段合并已保存: {blocks_path} ({len(df_blocks)} 段)")

        # ── 控制台汇总表 ──
        print("\n" + "=" * 72)
        print("每日泵开泵策略 — 汇总 (按相同泵组合合并)")
        print("=" * 72)
        print(f"{'时段':<26}{'时长':>4}  {'流量':>8}{'压力':>6}  "
              f"{'泵状态':<10}{'频率':<22}{'效率':>6}{'千吨电耗':>8}")
        for b in df_blocks.itertuples(index=False):
            hh1 = b.start[11:16]; hh2 = b.end[11:16]
            dur = b.num_points * FREQ_MINUTES
            print(f"{hh1}-{hh2:<20}{dur:>4}min  {b.flow_mean:>8.1f}{b.pressure_mean:>6.2f}  "
                  f"{b.states:<10}{b.freqs:<22}"
                  f"{b.efficiency_mean:>6.1f}{b.kwt_mean:>8.1f}")

    # ── ⑧ 汇总 ──
    n_feas = int(df_out["feasible"].sum())
    total_elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 72)
    print(f"汇总: {n_points} 点, 可行 {n_feas} 点 (偏差 ≤ ±100 m3/h), "
          f"不可行 {n_points - n_feas} 点")
    print(f"总耗时 {total_elapsed:.0f}s (平均每点 {total_elapsed / n_points:.1f}s)")
    print("=" * 72)


if __name__ == "__main__":
    main()
