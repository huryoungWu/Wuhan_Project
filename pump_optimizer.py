"""
水泵开启策略寻优 — 遗传算法 (基于 pump_inference.PumpInference 接口)

给定目标工况, 反向推导"效率最优"的水泵开启/频率策略:

输入 (用户给定):
  - 目标流量: 170:1 / 170:2 / 70:3 (m3/h)
  - 总管压力: (MPa) — 寻优时允许在给定值 ±0.01 MPa 内微调
  - 吸水井液位: (m) — 用于压力修正

寻优 (遗传算法):
  - 约束: 模型预测流量与目标流量偏差 ≤ ±100 m³/h
  - 目标: 满足约束的前提下, 预测总管效率最大化

用法 (命令行):
  python pump_optimizer.py --flow1 1500 --flow2 1200 --flow3 800 \
      --pressure 0.42 --level 3.42

用法 (代码):
  from pump_inference import PumpInference
  from pump_optimizer import optimize_strategy

  model = PumpInference("models/model_v2_combo_split.pt")
  result = optimize_strategy(model, target_flows=[1500, 1200, 800],
                             pressure=0.42, level=3.42)
  print(result['states'], result['freqs'])

注意:
  - 模型是数据驱动的代理模型, 寻优结果作为运行建议, 上线前需结合实际工况校核
  - 种群大小不要设为 7 (推理脚本会把 n==7 误判为单样本而平铺频率)
"""

import argparse

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

POP_SIZE = 100
N_GENERATIONS = 100
TOURNAMENT_SIZE = 3
ELITE_COUNT = 2
MUTATION_RATE = 0.1
MUTATION_SIGMA = 0.12
SEED = 42


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


def evaluate(model, genes, target_flows, pressure_target, level):
    """批量评估适应度: fitness = 效率 - PENALTY * 总超差量 (m3/h)"""
    states, freqs, pressure = decode(genes, pressure_target)
    f1, f2, f3, total_flow, eff = model.predict(states, freqs, pressure, level)
    f1, f2, f3, eff = map(np.atleast_1d, (f1, f2, f3, eff))
    pred_flows = np.column_stack([f1, f2, f3])
    violation = np.sum(np.maximum(np.abs(pred_flows - np.asarray(target_flows)) - FLOW_TOL, 0.0),
                       axis=1)
    fitness = eff - PENALTY * violation
    return fitness, pred_flows, total_flow, eff, states, freqs, pressure


# ============================================================================
# 遗传算子
# ============================================================================

def init_population(rng, n):
    return rng.random((n, 2 * N_PUMPS + 1))


def tournament_select(rng, fitness, n_children):
    """锦标赛选择, 返回选中个体的索引"""
    n = len(fitness)
    chosen = np.empty(n_children, dtype=int)
    for i in range(n_children):
        idx = rng.integers(0, n, TOURNAMENT_SIZE)
        chosen[i] = idx[np.argmax(fitness[idx])]
    return chosen


def uniform_crossover(rng, p1, p2):
    mask = rng.random(p1.shape) < 0.5
    return np.where(mask, p1, p2)


def gaussian_mutate(rng, genes):
    noise = rng.normal(0.0, MUTATION_SIGMA, genes.shape)
    mutated = genes + noise * (rng.random(genes.shape) < MUTATION_RATE)
    return np.clip(mutated, 0.0, 1.0)


# ============================================================================
# 寻优主函数
# ============================================================================

def optimize_strategy(model, target_flows, pressure, level,
                      pop_size=POP_SIZE, n_generations=N_GENERATIONS,
                      seed=SEED, top_k=3):
    """
    遗传算法寻优: 效率最优的水泵开启/频率策略

    参数:
        model:         PumpInference 实例
        target_flows:  [170:1, 170:2, 70:3] 目标流量 (m3/h)
        pressure:      总管压力 (MPa), 寻优时可在 ±0.01 内微调
        level:         吸水井液位 (m)
        pop_size:      种群大小 (勿设为 7)
        n_generations: 迭代代数
        seed:          随机种子
        top_k:         返回前几个不同的开启策略 (按泵开关组合去重)

    返回:
        dict: 最优策略 + 候选列表, 键:
          states / freqs / pressure / pred_flows / total_flow /
          efficiency / violation / feasible / candidates
    """
    target_flows = np.asarray(target_flows, dtype=np.float64)
    rng = np.random.default_rng(seed)
    pop = init_population(rng, pop_size)

    for _ in range(n_generations):
        fitness, *_ = evaluate(model, pop, target_flows, pressure, level)

        # 精英保留
        order = np.argsort(-fitness)
        elites = pop[order[:ELITE_COUNT]].copy()

        # 锦标赛选择 → 交叉 + 变异
        n_offspring = pop_size - ELITE_COUNT
        parent_idx = tournament_select(rng, fitness, n_offspring)
        parents = pop[parent_idx]
        offspring = np.empty_like(parents)
        for i in range(n_offspring):
            j = rng.integers(0, n_offspring)
            child = uniform_crossover(rng, parents[i], parents[j])
            offspring[i] = gaussian_mutate(rng, child)

        pop = np.vstack([elites, offspring])

    # 最终评估
    fitness, pred_flows, _, eff, states, freqs, pressure_used = \
        evaluate(model, pop, target_flows, pressure, level)
    order = np.argsort(-fitness)

    # 按泵开关组合去重, 取前 top_k 个不同策略
    candidates = []
    seen = set()
    for idx in order:
        key = tuple(states[idx])
        if key in seen:
            continue
        seen.add(key)
        deviation = np.abs(pred_flows[idx] - target_flows)
        violation = float(np.sum(np.maximum(deviation - FLOW_TOL, 0.0)))
        candidates.append({
            'states': states[idx],
            'freqs': freqs[idx],
            'pressure': float(pressure_used[idx]),
            'pred_flows': pred_flows[idx],
            'total_flow': float(np.sum(pred_flows[idx])),
            'efficiency': float(eff[idx]),
            'deviation': deviation,
            'violation': violation,
            'feasible': violation < 1e-6,
        })
        if len(candidates) >= top_k:
            break

    best = candidates[0]
    best.update({
        'target_flows': target_flows,
        'pressure_target': float(pressure),
        'level': float(level),
        'candidates': candidates,
    })
    return best


# ============================================================================
# 结果打印
# ============================================================================

def print_result(result):
    t = result['target_flows']
    print("\n" + "=" * 64)
    print(f"目标: 170:1={t[0]:.0f}  170:2={t[1]:.0f}  70:3={t[2]:.0f} m3/h | "
          f"压力={result['pressure_target']:.3f}±0.01 MPa | 液位={result['level']:.2f} m")
    print("=" * 64)

    for rank, cand in enumerate(result['candidates'], 1):
        status = '可行' if cand['feasible'] else '超差'
        print(f"\n方案{rank} [{status}] 预测效率 {cand['efficiency']:.1f}%")
        print(f"  泵号:  {' '.join(f'P{i+1}' for i in range(N_PUMPS))}")
        print(f"  状态:  {' '.join('开 ' if s else '关 ' for s in cand['states'])}")
        print(f"  频率:  {' '.join(f'{f:5.1f}' for f in cand['freqs'])} Hz")
        print(f"  压力:  {cand['pressure']:.4f} MPa (修正后送模型)")
        print(f"  预测流量: 170:1={cand['pred_flows'][0]:.1f}  170:2={cand['pred_flows'][1]:.1f}  "
              f"70:3={cand['pred_flows'][2]:.1f} m3/h")
        print(f"  偏差:  {' '.join(f'{d:+.1f}' for d in cand['deviation'])} m3/h (容差 ±{FLOW_TOL:.0f})")
        print(f"  总管流量: {cand['total_flow']:.1f} m3/h")


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description='水泵开启策略寻优 (遗传算法, 效率最优)')
    ap.add_argument('--flow1', type=float, required=True, help='170:1目标流量 (m3/h)')
    ap.add_argument('--flow2', type=float, required=True, help='170:2目标流量 (m3/h)')
    ap.add_argument('--flow3', type=float, required=True, help='70:3目标流量 (m3/h)')
    ap.add_argument('--pressure', type=float, required=True, help='总管压力 (MPa)')
    ap.add_argument('--level', type=float, default=3.58, help='吸水井液位 (m), 默认3.58 (修正量=0)')
    ap.add_argument('--model', default=None, help='权重文件路径 (默认 models/model_v2_combo_split.pt)')
    ap.add_argument('--pop', type=int, default=POP_SIZE, help='种群大小 (勿设为7)')
    ap.add_argument('--gens', type=int, default=N_GENERATIONS, help='迭代代数')
    ap.add_argument('--seed', type=int, default=SEED, help='随机种子')
    ap.add_argument('--topk', type=int, default=3, help='输出前几个不同开启策略')
    args = ap.parse_args()

    model = PumpInference(args.model)
    result = optimize_strategy(
        model, [args.flow1, args.flow2, args.flow3],
        args.pressure, args.level,
        pop_size=args.pop, n_generations=args.gens,
        seed=args.seed, top_k=args.topk,
    )
    print_result(result)


if __name__ == '__main__':
    main()
