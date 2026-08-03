"""
水泵开启策略寻优 — 粒子群优化 (PSO) (基于 pump_inference.PumpInference 接口)

给定目标工况, 反向推导"效率最优"的水泵开启/频率策略:

输入 (用户给定):
  - 目标流量: 170:1 / 170:2 / 70:3 (m3/h)
  - 总管压力: (MPa) — 寻优时允许在给定值 ±0.01 MPa 内微调
  - 吸水井液位: (m) — 用于压力修正

寻优 (粒子群优化):
  - 约束: 模型预测流量与目标流量偏差 ≤ ±100 m³/h
  - 目标: 满足约束的前提下, 预测总管效率最大化

用法 (接口调用):
  from pump_inference import PumpInference
  from pump_optimizer import optimize_strategy, print_result

  model = PumpInference("models/model_v2_combo_split.pt")
  result = optimize_strategy(model, target_flows=[1500, 1200, 800],
                             pressure=0.42, level=3.42)
  print(result['states'], result['freqs'])
  print_result(result)

注意:
  - 模型是数据驱动的代理模型, 寻优结果作为运行建议, 上线前需结合实际工况校核
  - 种群大小不要设为 7 (推理脚本会把 n==7 误判为单样本而平铺频率)
"""

import sys
import time

import numpy as np

from pump_inference import PumpInference

# ============================================================================
# 寻优参数
# ============================================================================
N_PUMPS = 7
FREQ_MIN = 30.0            # 运行泵最低频率 (Hz)
FREQ_MAX = 50.0            # 运行泵最高频率 (Hz)
FLOW_TOL = 100.0           # 流量容差 (m3/h)
PRESSURE_TOL = 0.01        # 压力容差 (MPa) — 压力在给定值 ±0.01 内微调
PENALTY = 1.0              # 超差惩罚系数 (每个 m3/h 扣 1 个效率点)

# ============================================================================
# PSO 参数 (可调)
# ============================================================================
PSO_POP_SIZE = 100            # 粒子群规模 (建议 30~80)
PSO_N_GENERATIONS = 500      # 迭代代数
PSO_INERTIA = 0.9           # 惯性权重 (0.4~0.9)
PSO_C1 = 1.5                # 个体学习因子
PSO_C2 = 2.0                # 社会学习因子
PSO_SEED = 42                # 随机种子


# ============================================================================
# 编码/解码: 每个个体 = [s1..s7 (0~1), f1..f7 (0~1), p (0~1)]
#   s_i > 0.5 → 泵 i 开启; f_i → [FREQ_MIN, FREQ_MAX] Hz (停泵为0)
#   p       → 压力在 [目标-0.01, 目标+0.01] MPa 内微调
# ============================================================================

def decode(genes, pressure_target):
    """基因 → (泵状态, 频率, 压力). genes: (n, 15)"""
    states = (genes[:, :N_PUMPS] > 0.5).astype(np.int64)
    freqs = np.where(states > 0,
                     FREQ_MIN + genes[:, N_PUMPS:2 * N_PUMPS] * (FREQ_MAX - FREQ_MIN),
                     0.0)
    pressure = pressure_target + (genes[:, -1] * 2 - 1) * PRESSURE_TOL
    return states, freqs, pressure


def evaluate(model, genes, target_flows, pressure_target, level, stats=None):
    """批量评估适应度: fitness = 效率 - PENALTY * 总超差量 (m3/h)

    stats: 推理耗时统计字典 (可选) — {'n_calls', 'n_samples', 'infer_time'}
    """
    states, freqs, pressure = decode(genes, pressure_target)
    t0 = time.perf_counter()
    # v6 返回 6 个值: f1, f2, f3, total_flow, eff, kwt
    f1, f2, f3, total_flow, eff, kwt = model.predict(states, freqs, pressure, level)
    if stats is not None:
        stats['n_calls'] += 1
        stats['n_samples'] += genes.shape[0]
        stats['infer_time'] += time.perf_counter() - t0
    f1, f2, f3, eff = map(np.atleast_1d, (f1, f2, f3, eff))
    pred_flows = np.column_stack([f1, f2, f3])
    violation = np.sum(np.maximum(np.abs(pred_flows - np.asarray(target_flows)) - FLOW_TOL, 0.0),
                       axis=1)
    fitness = eff - PENALTY * violation
    return fitness, pred_flows, total_flow, eff, states, freqs, pressure


# ============================================================================
# 粒子群优化 (PSO) 主函数
# ============================================================================

def optimize_strategy(model, target_flows, pressure, level,
                      pop_size=PSO_POP_SIZE, n_generations=PSO_N_GENERATIONS,
                      seed=PSO_SEED, top_k=3):
    target_flows = np.asarray(target_flows, dtype=np.float64)
    rng = np.random.default_rng(seed)
    dim = 2 * N_PUMPS + 1

    # 推理耗时统计
    stats = {'n_calls': 0, 'n_samples': 0, 'infer_time': 0.0}
    t_start = time.perf_counter()

    # 初始化
    positions = rng.random((pop_size, dim))
    velocities = rng.uniform(-0.1, 0.1, (pop_size, dim))
    fitness, _, _, _, _, _, _ = evaluate(model, positions, target_flows, pressure, level, stats)
    pbest_positions = positions.copy()
    pbest_fitness = fitness.copy()
    gbest_idx = np.argmax(fitness)
    gbest_position = positions[gbest_idx].copy()
    gbest_fitness = fitness[gbest_idx]

    # 记录所有历史最优状态
    state_best = {}  # key: 状态元组, value: (适应度, 位置)

    # 初始化记录
    for i in range(pop_size):
        states = (positions[i][:N_PUMPS] > 0.5).astype(np.int64)
        key = tuple(states)
        if key not in state_best or fitness[i] > state_best[key][0]:
            state_best[key] = (fitness[i], positions[i].copy())

    # 重启频率
    restart_interval = max(1, n_generations // 5)

    for gen in range(n_generations):
        inertia = PSO_INERTIA * (1 - gen / n_generations) + 0.4 * (gen / n_generations)

        # 随机重启
        if gen > 0 and gen % restart_interval == 0:
            n_reset = max(1, pop_size // 5)
            reset_idx = rng.choice(pop_size, n_reset, replace=False)
            positions[reset_idx] = rng.random((n_reset, dim))
            velocities[reset_idx] = rng.uniform(-0.1, 0.1, (n_reset, dim))
            fitness, _, _, _, _, _, _ = evaluate(model, positions, target_flows, pressure, level, stats)

        for i in range(pop_size):
            r1 = rng.random(dim)
            r2 = rng.random(dim)
            velocities[i] = (inertia * velocities[i] +
                             PSO_C1 * r1 * (pbest_positions[i] - positions[i]) +
                             PSO_C2 * r2 * (gbest_position - positions[i]))
            velocities[i] = np.clip(velocities[i], -0.5, 0.5)
            positions[i] = positions[i] + velocities[i]
            positions[i] = np.clip(positions[i], 0.0, 1.0)

        fitness, _, _, _, _, _, _ = evaluate(model, positions, target_flows, pressure, level, stats)

        for i in range(pop_size):
            if fitness[i] > pbest_fitness[i]:
                pbest_positions[i] = positions[i].copy()
                pbest_fitness[i] = fitness[i]
                # 更新历史最优状态
                states = (positions[i][:N_PUMPS] > 0.5).astype(np.int64)
                key = tuple(states)
                if key not in state_best or fitness[i] > state_best[key][0]:
                    state_best[key] = (fitness[i], positions[i].copy())

        current_best_idx = np.argmax(fitness)
        if fitness[current_best_idx] > gbest_fitness:
            gbest_position = positions[current_best_idx].copy()
            gbest_fitness = fitness[current_best_idx]

    # 从 state_best 中按适应度排序，取不同状态
    sorted_states = sorted(state_best.items(), key=lambda x: x[1][0], reverse=True)

    candidates = []
    # 取前 top_k 个不同状态（已经去重）
    for key, (fit, pos) in sorted_states[:top_k]:
        states = np.array(key, dtype=np.int64)
        freqs = np.where(states > 0,
                         FREQ_MIN + pos[N_PUMPS:2 * N_PUMPS] * (FREQ_MAX - FREQ_MIN),
                         0.0)
        pressure_used = pressure + (pos[-1] * 2 - 1) * PRESSURE_TOL
        # v6 返回 6 个值
        t0 = time.perf_counter()
        f1, f2, f3, total_flow, eff, kwt = model.predict(states, freqs, pressure_used, level)
        stats['n_calls'] += 1
        stats['n_samples'] += 1
        stats['infer_time'] += time.perf_counter() - t0
        pred_flows = np.array([f1, f2, f3])
        deviation = np.abs(pred_flows - target_flows)
        violation = float(np.sum(np.maximum(deviation - FLOW_TOL, 0.0)))
        candidates.append({
            'states': states,
            'freqs': freqs,
            'pressure': float(pressure_used),
            'pred_flows': pred_flows,
            'total_flow': float(total_flow),
            'efficiency': float(eff),
            'kwt': float(kwt),
            'deviation': deviation,
            'violation': violation,
            'feasible': violation < 1e-6,
        })

    # 如果仍然少于 top_k，从 pbest 中补充（但一般不会）
    if len(candidates) < top_k:
        # 从 pbest 中补充不同状态
        for idx in np.argsort(-pbest_fitness):
            states = (pbest_positions[idx][:N_PUMPS] > 0.5).astype(np.int64)
            key = tuple(states)
            if key in [tuple(c['states']) for c in candidates]:
                continue
            pos = pbest_positions[idx]
            freqs = np.where(states > 0,
                             FREQ_MIN + pos[N_PUMPS:2 * N_PUMPS] * (FREQ_MAX - FREQ_MIN),
                             0.0)
            pressure_used = pressure + (pos[-1] * 2 - 1) * PRESSURE_TOL
            # v6 返回 6 个值
            t0 = time.perf_counter()
            f1, f2, f3, total_flow, eff, kwt = model.predict(states, freqs, pressure_used, level)
            stats['n_calls'] += 1
            stats['n_samples'] += 1
            stats['infer_time'] += time.perf_counter() - t0
            pred_flows = np.array([f1, f2, f3])
            deviation = np.abs(pred_flows - target_flows)
            violation = float(np.sum(np.maximum(deviation - FLOW_TOL, 0.0)))
            candidates.append({
                'states': states,
                'freqs': freqs,
                'pressure': float(pressure_used),
                'pred_flows': pred_flows,
                'total_flow': float(total_flow),
                'efficiency': float(eff),
                'kwt': float(kwt),
                'deviation': deviation,
                'violation': violation,
                'feasible': violation < 1e-6,
            })
            if len(candidates) >= top_k:
                break

    # 按效率降序排序
    candidates = sorted(candidates, key=lambda x: x['efficiency'], reverse=True)

    best = candidates[0]
    elapsed = time.perf_counter() - t_start
    best.update({
        'target_flows': target_flows,
        'pressure_target': float(pressure),
        'level': float(level),
        'candidates': candidates,
        'num_unique_states': len(candidates),
        # 推理耗时统计
        'timing': {
            'total_s': elapsed,                       # 寻优总耗时 (s)
            'n_calls': stats['n_calls'],              # predict 调用次数
            'n_samples': stats['n_samples'],          # 推理样本总数
            'infer_total_s': stats['infer_time'],     # 模型推理总耗时 (s)
            'infer_per_call_ms': stats['infer_time'] / stats['n_calls'] * 1000.0,
            'infer_per_sample_ms': stats['infer_time'] / stats['n_samples'] * 1000.0,
        },
    })
    return best


# ============================================================================
# 结果打印
# ============================================================================

def print_result(result):
    t = result['target_flows']
    num = result.get('num_unique_states', len(result['candidates']))
    print("\n" + "=" * 72)
    print(f"目标: 170:1={t[0]:.0f}  170:2={t[1]:.0f}  70:3={t[2]:.0f} m3/h | "
          f"压力={result['pressure_target']:.3f}±0.01 MPa | 液位={result['level']:.2f} m")
    print(f"找到 {num} 种不同的泵状态组合 (期望 {len(result['candidates'])} 种)")
    print("=" * 72)

    for rank, cand in enumerate(result['candidates'], 1):
        status = '✓ 偏差可控' if cand['feasible'] else '✗ 偏差较大'
        print(f"\n方案{rank} [{status}] 预测效率 {cand['efficiency']:.1f}%  |  千吨水电耗 {cand['kwt']:.1f} kWh")
        print(f"  泵号:  {' '.join(f'P{i+1}' for i in range(N_PUMPS))}")
        print(f"  状态:  {' '.join('开 ' if s else '关 ' for s in cand['states'])}")
        print(f"  频率:  {' '.join(f'{f:5.1f}' for f in cand['freqs'])} Hz")
        print(f"  压力:  {cand['pressure']:.4f} MPa (修正后送模型)")
        print(f"  预测流量: 170:1={cand['pred_flows'][0]:.1f}  170:2={cand['pred_flows'][1]:.1f}  "
              f"70:3={cand['pred_flows'][2]:.1f} m3/h")
        print(f"  偏差:    {' '.join(f'{d:+.1f}' for d in cand['deviation'])} m3/h (容差 ±{FLOW_TOL:.0f})")
        print(f"  总管流量: {cand['total_flow']:.1f} m3/h")

    # 推理耗时
    timing = result.get('timing')
    if timing:
        print("\n" + "-" * 72)
        print(f"⏱ 推理耗时: 共 {timing['n_calls']} 次推理调用 / {timing['n_samples']} 个样本, "
              f"推理总耗时 {timing['infer_total_s']:.2f}s "
              f"(单次调用平均 {timing['infer_per_call_ms']:.1f} ms, "
              f"单样本平均 {timing['infer_per_sample_ms']:.3f} ms)")
        print(f"⏱ 寻优总耗时: {timing['total_s']:.1f}s")


# ============================================================================
# 单样本推理耗时基准
# ============================================================================

def benchmark_inference(model, n_calls=1000, pressure=0.42, level=3.42, seed=42):
    """
    单样本推理耗时基准: 每次 predict 只测 1 个样本, 重复 n_calls 次。

    返回统计字典:
      mean_ms / min_ms / max_ms / p50_ms / p95_ms — 单次调用耗时 (ms)
      var_ms2 / std_ms          — 方差 (ms²) / 标准差 (ms)
      total_s                   — 总耗时 (s)
    """
    rng = np.random.default_rng(seed)
    states = (rng.random(N_PUMPS) > 0.5).astype(np.int64)
    freqs = np.where(states > 0,
                     FREQ_MIN + rng.random(N_PUMPS) * (FREQ_MAX - FREQ_MIN), 0.0)

    # 预热: 前 20 次调用不参与统计 (CUDA 初始化 / 线程池 / 缓存)
    for _ in range(20):
        model.predict(states, freqs, pressure, level)

    times = np.empty(n_calls)
    for i in range(n_calls):
        t0 = time.perf_counter()
        model.predict(states, freqs, pressure, level)
        times[i] = time.perf_counter() - t0

    ms = times * 1000.0
    return {
        'n_calls': n_calls,
        'mean_ms': float(ms.mean()),
        'min_ms': float(ms.min()),
        'max_ms': float(ms.max()),
        'p50_ms': float(np.percentile(ms, 50)),
        'p95_ms': float(np.percentile(ms, 95)),
        'var_ms2': float(ms.var()),
        'std_ms': float(ms.std()),
        'total_s': float(times.sum()),
    }


def print_benchmark(stats):
    """打印单样本推理耗时基准结果"""
    print("\n" + "=" * 72)
    print(f"单样本推理耗时基准: {stats['n_calls']} 次调用 (每次 1 个样本)")
    print("=" * 72)
    print(f"  平均耗时: {stats['mean_ms']:.2f} ms")
    print(f"  最小耗时: {stats['min_ms']:.2f} ms")
    print(f"  最大耗时: {stats['max_ms']:.2f} ms")
    print(f"  P50 耗时: {stats['p50_ms']:.2f} ms")
    print(f"  P95 耗时: {stats['p95_ms']:.2f} ms")
    print(f"  方差:     {stats['var_ms2']:.4f} ms²")
    print(f"  标准差:   {stats['std_ms']:.2f} ms")
    print(f"  总耗时:   {stats['total_s']:.2f} s")


# ============================================================================
# 交互式用户输入
# ============================================================================

def get_user_input():
    """交互式获取用户输入的目标值"""
    print("\n" + "=" * 56)
    print("  水泵策略寻优 — 请输入目标工况")
    print("=" * 56)

    while True:
        try:
            print("\n【目标流量】")
            q1 = float(input("  170:1 管道目标流量 (m3/h): ").strip())
            q2 = float(input("  170:2 管道目标流量 (m3/h): ").strip())
            q3 = float(input("  70:3 管道目标流量  (m3/h): ").strip())

            if q1 < 0 or q2 < 0 or q3 < 0:
                print("  ❌ 流量不能为负数，请重新输入")
                continue
            break
        except ValueError:
            print("  ❌ 请输入有效数字，请重新输入")

    while True:
        try:
            print("\n【运行工况】")
            pressure = float(input("  总管压力 (MPa): ").strip())
            if pressure <= 0:
                print("  ❌ 压力必须大于0，请重新输入")
                continue
            break
        except ValueError:
            print("  ❌ 请输入有效数字，请重新输入")

    while True:
        try:
            level = float(input("  吸水井液位 (m): ").strip())
            break
        except ValueError:
            print("  ❌ 请输入有效数字，请重新输入")

    print("\n" + "-" * 56)
    print(f"确认输入: 170:1={q1:.0f}  170:2={q2:.0f}  70:3={q3:.0f} m3/h | "
          f"压力={pressure:.3f} MPa | 液位={level:.2f} m")
    confirm = input("确认无误? (y/n，直接回车默认 y): ").strip().lower()
    if confirm in ('n', 'no'):
        print("已取消，请重新运行程序")
        return None

    return [q1, q2, q3], pressure, level


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == '__main__':
    # 加载模型
    MODEL_PATH = "models/model_v2_combo_split.pt"
    try:
        model = PumpInference(MODEL_PATH)
        model.info()
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("请检查模型文件路径是否正确")
        sys.exit(1)

    # --benchmark: 单样本推理耗时基准 (1000 次调用, 每次 1 个样本)
    if '--benchmark' in sys.argv:
        print_benchmark(benchmark_inference(model))
        sys.exit(0)

    # 获取用户输入
    user_input = get_user_input()
    if user_input is None:
        import sys
        sys.exit(0)

    TARGET_FLOWS, PRESSURE, LEVEL = user_input

    # 调用 PSO 优化
    print("\n🔄 正在寻优，请稍候...")
    result = optimize_strategy(model, target_flows=TARGET_FLOWS,
                               pressure=PRESSURE, level=LEVEL)
    print_result(result)
