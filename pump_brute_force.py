# -*- coding: utf-8 -*-
"""
pump_brute_force.py — 基于 pump_inference.py 推理接口的暴力寻优算法

思路对齐 optimizer.py (暴力枚举 + 向量化批量评估 + 剪枝 + 按目标择优):

  搜索空间:
    - 泵组合  : 2^7 = 128 种, 排除全关/全开, 可按目标流量约束开启台数 (同 optimizer.py)
    - 频率    : 30~50 Hz 步长 1 Hz (即 optimizer.py 中的"转速"; 泵6/泵7 默认工频恒 50 Hz)
    - 压力    : 0.24~0.36 MPa 步长 0.01
    - 液位    : 1~3 m 步长 0.5
  评估: 全部经 PumpInference.predict 批量推理 ((n,7) 批量, 等价 optimizer.py 的 predict_batch)
  剪枝: 流量>0, 效率∈(eff_range), 可选流量上限校验 (相似定律额定流量之和)
  择优: 按目标流量分组 (容差 1.5×流量步长) → 每泵组合取效率最高 → 时间序列相邻点复用

  千吨水电耗: v6 模型只输出效率, 不输出功率 → 按 optimizer.py 口径由效率换算:
    H_eff = 压力×102 − (液位−2.35) (m);  kwt = 272.5 × H_eff / 效率%  (kWh/千吨)

  目标流量范围依据 merged_minute_all.csv 实测 (3.6M 条):
    总管流量 P1=1676, P50=2720, P99=3675, max=4032 m³/h → 寻优范围 1700~3700 步长 100
    注: v6 模型在 0.24~0.36 MPa 下最大总流量约 4400, 超出真实范围的目标不可达。

注意: v6 模型只输出流量+效率, 不输出功率 → 目标函数为效率最大化 (无 kwt 指标)。

用法:
  opt = PumpBruteForceOptimizer()
  # 离线建表
  opt.build_lookup_table(pressures, target_flows, levels, csv_path)
  # 在线查询 (时间点序列, 支持复用)
  solutions = opt.query_optimal_solutions([(0.33, 4000, 3.0), (0.33, 4050, 3.0)])
"""

import os
import csv
import gc
import time
import itertools
import numpy as np
import sys

from pump_inference import PumpInference

# 结果目录: 基于脚本所在位置, 不依赖运行时的当前工作目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


class ProgressBar:
    """简易进度条 (同 optimizer.py)"""
    def __init__(self, total, width=50, description=""):
        self.total = max(total, 1)
        self.width = width
        self.current = 0
        self.start_time = time.time()
        self.description = description

    def update(self, n=1):
        self.current += n
        percent = min(self.current / self.total, 1.0)
        filled = int(self.width * percent)
        elapsed = time.time() - self.start_time
        remaining = (elapsed / self.current) * (self.total - self.current) if self.current > 0 else 0
        bar = '█' * filled + '-' * (self.width - filled)
        sys.stdout.write(f'\r{self.description} |{bar}| {percent:.1%} ({self.current}/{self.total}) 剩余: {remaining:.0f}s')
        sys.stdout.flush()

    def finish(self):
        if self.current < self.total:
            self.update(self.total - self.current)
        sys.stdout.write('\n')


class PumpBruteForceOptimizer:
    """
    暴力寻优器 — 枚举全部 (泵组合 × 频率档位) 并批量推理, 按目标流量取最高效率解。

    参数:
        model_path       : 推理权重路径, None 用默认 models/model_v2_combo_split.pt
        freq_min/max/step: 变频泵频率档位 (默认 30~50 Hz 步长 1)
        fixed_pumps      : 工频泵索引 (默认 {5,6} → P6/P7 恒 50 Hz, 依据真实数据直方图)
        flow_per_pump    : 台数约束系数 num_active = int(target_flow / flow_per_pump) + 1
        eff_range        : 效率过滤区间 (%, 默认 40~90, 对齐 optimizer.py 的 0.4~0.9)
        min/max_active_pumps: 开启台数上下限 (默认 2~6; 全开不可行, 最少两台运行)
        count_band       : 台数搜索带宽 (±band 台, 默认 1)
                           模型实际流量常低于相似定律估计, 精确台数约束易漏解
        chunk_size       : 单批推理行数上限 (防内存爆)
    """
    def __init__(self, model_path=None,
                 freq_min=30, freq_max=50, freq_step=1,
                 fixed_pumps=(5, 6),
                 flow_per_pump=1300.0,   # 模型单泵实际容量 600~1900, 按均值定台数约束
                 eff_range=(40.0, 90.0),
                 min_active_pumps=2, max_active_pumps=6,
                 count_band=1,
                 chunk_size=200000):
        self.model = PumpInference(model_path)

        # ---- 每台泵的频率档位: 变频泵 30~50 步长1; 工频泵恒 50 ----
        self.vfd_levels = np.arange(freq_min, freq_max + 1e-9, freq_step, dtype=np.float32)
        self.speed_ranges = {}
        for i in range(7):
            if i in fixed_pumps:
                self.speed_ranges[i] = np.array([50.0], dtype=np.float32)   # 工频: 开启=满频
            else:
                self.speed_ranges[i] = self.vfd_levels

        self.flow_per_pump = flow_per_pump
        self.eff_range = eff_range
        self.min_active_pumps = min_active_pumps
        self.max_active_pumps = max_active_pumps
        self.count_band = count_band
        self.chunk_size = chunk_size

        # 泵额定流量 (m3/h @ 50Hz, 取自 pump_inference.py 演示的相似定律基准)
        self.rated_flows = np.array([2020, 670, 2020, 1260, 670, 1260, 670], dtype=np.float64)

        # 千吨水电耗换算 (同 optimizer.py): kwt = 272.5 × H_eff / 效率%
        self.pump_level = 2.35   # optimizer.py PUMP_LEVEL 基准液位

        # (pressure, level, num_active) → [(combo, flows, effs, conf, max_rated_flow), ...] 缓存
        self._eval_cache = {}

    # ====================== 泵组合生成 ======================
    def generate_pump_combinations(self, target_flow=None):
        """
        生成有效泵组合 (同 optimizer.py generate_valid_combinations):
          - 排除全关 / 全开 (物理不可行)
          - 可选: 按目标流量约束开启台数 num_active = int(flow / flow_per_pump) + 1
        """
        if target_flow is not None:
            num_active = max(self.min_active_pumps,
                             min(self.max_active_pumps,
                                 int(target_flow / self.flow_per_pump) + 1))
        else:
            num_active = None

        combos = []
        for bits in itertools.product([0, 1], repeat=7):
            s = tuple(bits)
            n = sum(s)
            if n == 0:                    # 全关
                continue
            if n == 7:                    # 全开 (七台不能同时全开)
                continue
            if num_active is not None and n != num_active:
                continue
            combos.append(s)
        return combos

    def _combos_with_count(self, num_active):
        combos = []
        for bits in itertools.product([0, 1], repeat=7):
            s = tuple(bits)
            if sum(s) == num_active:
                combos.append(s)
        return combos

    def _flow_to_counts(self, target_flow):
        """目标流量 → 需要搜索的台数集合 (公式台数 ± count_band, 夹在上下限内)"""
        num = int(target_flow / self.flow_per_pump) + 1
        lo = max(self.min_active_pumps, num - self.count_band)
        hi = min(self.max_active_pumps, num + self.count_band)
        return range(lo, hi + 1)

    # ====================== 转速矩阵生成 (缓存) ======================
    def _build_speed_matrix(self, states):
        """开启泵取各自频率档位做笛卡尔积, 构建 (N,7) 矩阵, 关闭泵填 0"""
        ranges = [self.speed_ranges[i] for i, s in enumerate(states) if s == 1]
        if not ranges:
            return None, 0
        grids = np.meshgrid(*ranges, indexing='ij')
        N = grids[0].size
        mat = np.zeros((N, 7), dtype=np.float32)
        j = 0
        for i, s in enumerate(states):
            if s == 1:
                mat[:, i] = grids[j].ravel()
                j += 1
        return mat, N

    def _max_rated_flow(self, states):
        return float(self.rated_flows[[i for i, s in enumerate(states) if s]].sum())

    # ====================== 批量推理 ======================
    def _predict_chunks(self, states, speed_matrix, pressure, level):
        """分块调用推理接口 (等价 optimizer.py 的 batch_predict)"""
        N = len(speed_matrix)
        states_tile = np.tile(np.asarray(states, dtype=np.int64), (N, 1))
        flows = np.zeros(N, dtype=np.float64)
        effs = np.zeros(N, dtype=np.float64)
        for s in range(0, N, self.chunk_size):
            e = min(s + self.chunk_size, N)
            _, _, _, tot, eff = self.model.predict(
                states_tile[s:e], speed_matrix[s:e], pressure, level)
            flows[s:e] = tot
            effs[s:e] = eff
        return flows, effs

    # ====================== 单条件评估 (压力, 液位, 台数) ======================
    def evaluate_condition(self, pressure, level, num_active, validate_flow_limit=True):
        """
        评估 (压力, 液位) 下所有开启 num_active 台的泵组合 × 频率档位组合。
        返回 [(states, flows, effs, confidence, appear_count, max_rated_flow), ...]
        """
        key = (round(float(pressure), 4), round(float(level), 3), int(num_active))
        if key in self._eval_cache:
            return self._eval_cache[key]

        combos = self._combos_with_count(num_active)
        results = []
        total_rows = 0
        for states in combos:
            speed_matrix, N = self._build_speed_matrix(states)
            if N == 0:
                continue
            total_rows += N

            flows, effs = self._predict_chunks(states, speed_matrix, pressure, level)

            # ---- 过滤: 流量>0, 效率∈(eff_range), 可选流量上限 ----
            mask = (flows > 0) & (effs > self.eff_range[0]) & (effs <= self.eff_range[1])
            if validate_flow_limit:
                mrf = self._max_rated_flow(states)
                mask &= (flows <= mrf) & (flows >= 0.5 * mrf)
            else:
                mrf = self._max_rated_flow(states)

            conf, count = self.model.combo_confidence(states)

            results.append((states, speed_matrix, flows, effs, conf, count, mrf))
            gc.collect()

        print(f"  压力 {pressure} 液位 {level} 台数 {num_active}: "
              f"{len(combos)} 个泵组合 × {total_rows:,} 行推理完成")
        self._eval_cache[key] = results
        return results

    def kwt_from_eff(self, pressure, level, eff_percent):
        """效率(%) → 千吨水电耗 (kWh/千吨): kwt = 272.5 × H_eff / 效率%
        H_eff = 压力×102 − (液位−2.35), 同 optimizer.py 有效扬程口径"""
        H_eff = pressure * 102 - (level - self.pump_level)
        if H_eff <= 0:
            return 0.0
        return 272.5 * H_eff / eff_percent

    # ====================== 按目标流量择优 (对齐 optimizer.py) ======================
    def _best_per_combo_for_flow(self, results, target_flow, tolerance, pressure, level):
        """
        每组结果中取与 target_flow 误差 ≤ tolerance 的效率最高行;
        该流量下无匹配 → 取全局最近一行 (fallback)。
        """
        best_rows = []
        for states, speed_matrix, flows, effs, conf, count, mrf in results:
            diff = np.abs(flows - target_flow)
            within = diff <= tolerance
            if np.any(within):
                idx = np.where(within)[0][int(np.argmax(effs[within]))]
            else:
                idx = int(np.argmin(diff))     # fallback: 最近流量
            eff_val = float(effs[idx])
            best_rows.append({
                'states': list(states),
                'freqs': speed_matrix[idx].tolist(),   # 转速矩阵第 idx 行即最优频率组合
                'total_flow': float(flows[idx]),
                'efficiency': eff_val,
                'kwt': self.kwt_from_eff(pressure, level, eff_val),
                'confidence': float(conf),
                'appear_count': int(count),
                'max_rated_flow': float(mrf),
            })
        return best_rows

    # ====================== 在线查询 (时间序列, 支持复用) ======================
    def query_optimal_solutions(self, points, tolerance=150.0, csv_path=None):
        """
        查询接口: 输入 [(压力, 流量, 液位), ...] 时间点序列。
        复用规则 (对齐 optimizer.py query_optimal_solutions):
          压力/液位相同 且 |Δ流量| < 100 → 直接复用上一时间点最优解。
        返回: 每时间点一个列表, 每个元素为最优解 dict
              {states, freqs, total_flow, efficiency, confidence, appear_count}
        可选: csv_path 传路径则把每时间点最优解写入 CSV (默认写入 results/ 带时间戳)。
        """
        if csv_path is None:
            csv_path = os.path.join(RESULTS_DIR, f"pump_brute_force_query_{_timestamp()}.csv")
        else:
            os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        all_solutions = []
        last_solution = None
        last_pl = None
        last_flow = None

        for t_idx, (pressure, target_flow, level) in enumerate(points):
            print(f"\n时间点 {t_idx + 1}/{len(points)}: "
                  f"压力={pressure} MPa, 流量={target_flow} m³/h, 液位={level} m")

            # ---- 复用上一时间点 ----
            if (last_solution is not None and last_pl == (pressure, level)
                    and abs(target_flow - last_flow) < 100):
                print(f"  流量变化 {abs(target_flow - last_flow):.1f} < 100 且工况相同 → 直接复用")
                all_solutions.append([{**r, 'pressure': pressure, 'target_flow': target_flow,
                                       'level': level, 'reused': 1} for r in last_solution])
                last_flow = target_flow
                continue

            best_rows = []
            for num_active in self._flow_to_counts(target_flow):
                results = self.evaluate_condition(pressure, level, num_active)
                best_rows += self._best_per_combo_for_flow(results, target_flow, tolerance,
                                                           pressure, level)

            if best_rows:
                # 择优: 容差内有解 → 取效率最高; 无解 → 取偏差最小 (避免远端高效行误导)
                in_tol = [r for r in best_rows
                          if abs(r['total_flow'] - target_flow) <= tolerance]
                if in_tol:
                    best = max(in_tol, key=lambda r: r['efficiency'])
                else:
                    best = min(best_rows,
                               key=lambda r: abs(r['total_flow'] - target_flow))
                best['flow_deviation'] = best['total_flow'] - target_flow
                best['pressure'] = pressure
                best['target_flow'] = target_flow
                best['level'] = level
                best['reused'] = 0
                print(f"  最优: 组合={''.join(map(str, best['states']))} "
                      f"频率={[int(f) for f in best['freqs']]}, "
                      f"流量={best['total_flow']:.0f} (偏差 {best['flow_deviation']:+.0f}), "
                      f"效率={best['efficiency']:.1f}%, "
                      f"千吨水电耗={best['kwt']:.1f} kWh, "
                      f"置信度={'高' if best['confidence'] > 0 else '低'} "
                      f"(出现 {best['appear_count']:,} 次)")
                all_solutions.append([best])
                last_solution = [best]
            else:
                if last_solution is not None:
                    print(f"  无有效解 → 复用上一时间点解")
                    all_solutions.append([{**r, 'pressure': pressure,
                                           'target_flow': target_flow,
                                           'level': level, 'reused': 1}
                                          for r in last_solution])
                else:
                    print(f"  无有效解且无历史解 → 返回零解")
                    all_solutions.append([{
                        'states': [0] * 7, 'freqs': [0.0] * 7,
                        'total_flow': 0.0, 'efficiency': 0.0,
                        'confidence': 0.0, 'appear_count': 0,
                        'flow_deviation': -target_flow,
                        'pressure': pressure, 'target_flow': target_flow,
                        'level': level, 'reused': 0,
                    }])

            last_pl = (pressure, level)
            last_flow = target_flow

        # ---- 写入查询结果 CSV ----
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['TimePoint', 'Pressure', 'Target_Flow', 'Liquid_Level',
                             'Pump_Group',
                             *[f'P{i}_Freq' for i in range(1, 8)],
                             'Total_Flow', 'Flow_Deviation', 'Eff', 'PCTW',
                             'Confidence', 'Appear_Count', 'Reused'])
            for i, sol in enumerate(all_solutions):
                s = sol[0]
                row = [i + 1, s.get('pressure', ''), s.get('target_flow', ''),
                       s.get('level', ''),
                       ''.join(map(str, s['states'])),
                       *[int(round(f)) for f in s['freqs']],
                       round(s['total_flow'], 1), round(s.get('flow_deviation', 0), 1),
                       round(s['efficiency'], 2), round(s.get('kwt', 0), 2),
                       s['confidence'], s['appear_count'],
                       s.get('reused', 0)]
                writer.writerow(row)
        print(f"查询结果已写入: {csv_path}")

        return all_solutions

    # ====================== 离线建表 (对齐 optimizer.py 主流程) ======================
    def build_lookup_table(self, pressures, target_flows, levels,
                           csv_path=None, tolerance=None, validate_flow_limit=True):
        """
        全网格暴力寻优并写入 CSV:
          对每个 (压力, 液位): 对每个目标流量 → 按台数约束枚举 → 每泵组合取效率最高行
        CSV 列: Header_Press, Liquid_Level, Flow_Lower/Upper, Pump_Group,
                P1_Freq~P7_Freq, Total_Flow, Flow_Deviation, Eff, PCTW,
                Confidence, Appear_Count
        """
        if csv_path is None:
            csv_path = os.path.join(RESULTS_DIR,
                                    f"pump_brute_force_table_{_timestamp()}.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        target_flows = sorted(target_flows)
        if tolerance is None:
            tolerance = 1.5 * (target_flows[1] - target_flows[0] if len(target_flows) > 1 else 100.0)

        cols = (['Header_Press', 'Liquid_Level', 'Flow_Lower', 'Flow_Upper', 'Pump_Group']
                + [f'P{i}_Freq' for i in range(1, 8)]
                + ['Total_Flow', 'Flow_Deviation', 'Eff', 'PCTW',
                   'Confidence', 'Appear_Count'])
        total_rows = 0
        header_written = False

        # 台数约束去重: 每个 num_active 只评估一次 (同 query, 支持 count_band)
        num_to_flows = {}
        for tf in target_flows:
            for num in self._flow_to_counts(tf):
                num_to_flows.setdefault(num, []).append(tf)

        n_conditions = len(pressures) * len(levels)
        bar = ProgressBar(n_conditions, description="建表进度")
        t0 = time.time()

        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            if not header_written:
                writer.writeheader()
                header_written = True

            for pressure in pressures:
                for level in levels:
                    print(f"\n压力 {pressure} MPa, 液位 {level} m:")
                    for num, tfs in sorted(num_to_flows.items()):
                        results = self.evaluate_condition(pressure, level, num, validate_flow_limit)
                        for tf in tfs:
                            best_rows = self._best_per_combo_for_flow(results, tf, tolerance,
                                                                      pressure, level)
                            for r in best_rows:
                                row = {
                                    'Header_Press': round(pressure, 3),
                                    'Liquid_Level': round(level, 1),
                                    'Flow_Lower': round(tf - tolerance / 1.5, 1),
                                    'Flow_Upper': round(tf + tolerance / 1.5, 1),
                                    'Pump_Group': ''.join(map(str, r['states'])),
                                }
                                for i in range(7):
                                    row[f'P{i + 1}_Freq'] = int(round(r['freqs'][i]))
                                row['Total_Flow'] = round(r['total_flow'], 1)
                                row['Flow_Deviation'] = round(r['total_flow'] - tf, 1)
                                row['Eff'] = round(r['efficiency'], 2)
                                row['PCTW'] = round(r['kwt'], 2)
                                row['Confidence'] = r['confidence']
                                row['Appear_Count'] = r['appear_count']
                                writer.writerow(row)
                                total_rows += 1
                    self._eval_cache.clear()
                    bar.update(1)
            bar.finish()

        print(f"\n建表完成: {total_rows} 行, 耗时 {time.time() - t0:.0f}s")
        print(f"输出文件: {csv_path}")
        return csv_path

    def info(self):
        print("PumpBruteForceOptimizer")
        print(f"  频率档位: 变频泵 {self.vfd_levels[0]:.0f}~{self.vfd_levels[-1]:.0f} Hz 步长 "
              f"{self.vfd_levels[1] - self.vfd_levels[0]:.0f}, 工频泵恒 50 Hz")
        print(f"  台数约束: num_active = int(流量/{self.flow_per_pump:.0f}) + 1, "
              f"范围 [{self.min_active_pumps}, {self.max_active_pumps}]")
        print(f"  效率过滤: {self.eff_range[0]}% < 效率 ≤ {self.eff_range[1]}%")
        print(f"  批量推理分块: {self.chunk_size:,} 行/批")
        self.model.info()


# ====================== 演示 ======================
if __name__ == '__main__':
    opt = PumpBruteForceOptimizer()
    opt.info()

    # 1) 在线查询: 3 个时间点, 第 2 点触发"流量变化<100 复用"
    #    目标流量取真实数据范围 (P1~P99 = 1676~3675 m³/h)
    points = [(0.33, 2700, 3.3), (0.33, 2750, 3.3), (0.35, 3500, 2.0)]
    sols = opt.query_optimal_solutions(points)
    print("\n查询结果汇总:")
    for i, sol in enumerate(sols):
        s = sol[0]
        print(f"  时间点{i + 1}: 组合={''.join(map(str, s['states']))} "
              f"频率={[int(f) for f in s['freqs']]} 流量={s['total_flow']:.0f} "
              f"效率={s['efficiency']:.1f}% 千吨水电耗={s['kwt']:.1f} kWh")

    # 2) 离线建表 (小网格, 全量网格见注释; 默认写入 results/ 带时间戳)
    opt.build_lookup_table(
        pressures=[0.30, 0.33, 0.36],
        target_flows=list(range(2200, 3701, 100)),   # 与全量网格同一步长
        levels=[2.0, 3.0],
    )
    # 全量网格 (覆盖实测流量 P1~P99 = 1676~3675, 步长 100 → 容差 ±150, 约 1~2 小时):
    # opt.build_lookup_table(
    #     pressures=np.arange(0.24, 0.3601, 0.01).tolist(),
    #     target_flows=list(range(1700, 3701, 100)),
    #     levels=[1.0, 1.5, 2.0, 2.5, 3.0],
    # )
