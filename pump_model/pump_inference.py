"""
泵站推理模型 v6 — 加载 CQU_improve_v6 训练的权重，输出分管/总管流量/效率 (不含功率)

输入 (用户提供):
  - 7个泵运行状态 (0/1)
  - 7个泵运行频率 (Hz)，未运行泵填0
  - 1个总管压力 (MPa, 原始值)
  - 1个吸水井液位 (m) — 用于压力修正

压力修正: 修正后压力 = 原压力 - (吸水井液位 - 3.58) / 102，送入模型的是修正后的压力。

输出:
  - 170:1瞬时流量 (m3/h)
  - 170:2瞬时流量 (m3/h)
  - 70:3瞬时流量 (m3/h)
  - 总管流量 (m3/h)    = sum(170:1 + 170:2 + 70:3)
  - 总管效率 (%)
  - 千吨水电耗 (kWh/千吨) = 272.5 × H_eff / 效率%, H_eff = 压力×102 − (液位−2.35)
  - 置信度: 泵组组合在真实数据中出现过→高, 未出现过→低 (predict_with_confidence)

工程特征自动生成，权重文件由 CQU_improve_v6_flow_eff_only.py 训练产生。

2026-08-08 重构: 模型结构改为从同目录 model.py 导入 (与训练共用同一份定义,
训练版 dropout 在 eval() 下不生效, 行为与原推理版一致); 原
D:\Wuhan_Project\pump_inference.py 保持原样, 未做修改。

用法:
  model = PumpInference("models/model_v2_combo_split.pt")
  flow_170_1, flow_170_2, flow_703, total_flow, eff, kwt = model.predict(states, freqs, pressure, level)
"""

import os
import sys
import numpy as np
import torch

# 保证能从本目录导入共享模型结构 (训练/推理共用一个模型定义)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DeepWaterPlantModelWithEmbedding

# ============================================================================
# 千吨水电耗换算常量 (口径同 optimizer.py)
#   H_eff = 压力(MPa)×102 − (液位 − PUMP_LEVEL)  有效扬程 (m)
#   kwt   = 272.5 × H_eff / 效率(%)   (kWh/千吨水)
# ============================================================================
PUMP_LEVEL = 2.35   # 泵安装基准液位 (m)
KWT_COEF = 272.5    # 由 ρ·g/3.6e6 × 1000 × 100 折算 (ρ=1000 kg/m³, g=9.81)


# ============================================================================
# 真实数据中出现的泵组组合统计 (7位编码 P1~P7, 值 = 该组合出现次数)
# 来源: new_data/merged_minute_all_with_efficiency.csv (3,036,418 条, 61 种组合)
# 推理置信度: 组合出现在表中 → 高 (1.0); 未出现 → 低 (0.0)
# ============================================================================
APPEARED_COMBOS = {
    "1000010": 287155,
    "1100010": 267341,
    "1100011": 221713,
    "1101001": 188841,
    "1101000": 188512,
    "0001010": 183088,
    "1001000": 177419,
    "1000011": 160983,
    "0101011": 143190,
    "1001011": 104030,
    "0011000": 98122,
    "0110010": 82146,
    "0111001": 81293,
    "0110011": 75136,
    "0010010": 72227,
    "0101010": 70128,
    "0111000": 60629,
    "1001010": 55145,
    "1000110": 51601,
    "0011001": 43600,
    "0001110": 43074,
    "1001001": 42856,
    "1001100": 42078,
    "0001011": 38890,
    "1001101": 33990,
    "1000111": 32229,
    "0010011": 27394,
    "0001111": 26833,
    "0101110": 25013,
    "0011100": 20018,
    "0011011": 17354,
    "0011010": 16794,
    "1011001": 8236,
    "0011101": 8179,
    "0000110": 7381,
    "1010011": 5887,
    "1011000": 4513,
    "1001110": 4274,
    "1100110": 3717,
    "1100111": 2670,
    "1101010": 1971,
    "1010010": 1849,
    "1101100": 1658,
    "1101011": 1582,
    "0111011": 883,
    "1001111": 792,
    "0010110": 752,
    "0111010": 735,
    "0111101": 452,
    "1111000": 437,
    "1101101": 355,
    "0011110": 258,
    "0110110": 165,
    "1111001": 159,
    "0101111": 139,
    "1110011": 120,
    "1101111": 117,
    "0001100": 107,
    "1110010": 106,
    "1101110": 89,
    "1100000": 13,
}


# ============================================================================
# 推理类 v6 (模型结构定义见 model.py, 与训练共用)
# ============================================================================

class PumpInference:
    """
    泵站推理模型 v6 — 加载 CQU_improve_v6_flow_eff_only.py 训练的权重。

    输出:
      - 170:1瞬时流量 (m3/h)
      - 170:2瞬时流量 (m3/h)
      - 70:3瞬时流量 (m3/h)
      - 总管流量 (m3/h) = 170:1 + 170:2 + 70:3
      - 总管效率 (%)

    压力修正: 修正后压力 = 原压力 - (吸水井液位 - 3.58) / 102 (MPa)

    用法:
        model = PumpInference("models/model_v2_combo_split.pt")
        flow_170_1, flow_170_2, flow_703, total_flow, eff, kwt = model.predict(states, freqs, pressure, level)
    """

    def __init__(self, model_path=None):
        if model_path is None:
            # 权重保存路径仍为项目根目录 models/ (train.py 保存路径不变);
            # 本文件在新目录下, 相对路径会指到 pump_model/models, 故用绝对路径
            model_path = r"D:\Wuhan_Project\pump_model\models\model_v2_combo_split.pt"

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"权重文件不存在: {model_path}\n"
            )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        n_cont_raw = len(ckpt['continuous_cols'])       # 8 (7 freq + 1 pressure)
        n_eng = len(ckpt['engineered_cols'])             # 0 (无工程特征)
        continuous_dim = n_cont_raw + n_eng             # 8
        num_pumps = len(ckpt['discrete_cols'])          # 7
        output_dim = ckpt['output_dim']                 # 4

        self.model = DeepWaterPlantModelWithEmbedding(
            num_pumps=num_pumps,
            continuous_dim=continuous_dim,
            output_dim=output_dim,
            embed_dim=8
        ).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        self.cont_scaler = ckpt['continuous_scaler']
        self.out_scaler = ckpt['output_scaler']
        self.all_output_cols = ckpt['all_output_cols']  # [170:1, 170:2, 70:3, 效率]

        # 压力修正参数 (与 train.py 一致, 随权重保存; 旧权重无此字段时用默认值)
        pc = ckpt.get('pressure_correction', {})
        self.level_baseline = float(pc.get('level_baseline', 3.58))
        self.level_divisor = float(pc.get('level_divisor', 102.0))

    def _preprocess(self, states, freqs, pressure):
        """原始输入 → 模型输入张量"""
        states = np.atleast_2d(np.asarray(states, dtype=np.int64))
        freqs = np.atleast_2d(np.asarray(freqs, dtype=np.float32))
        pressure = np.atleast_1d(np.asarray(pressure, dtype=np.float32))

        n = states.shape[0]
        if freqs.shape[0] == 7 and n > 1:
            freqs = np.tile(freqs, (n, 1))
        if pressure.shape[0] == 1 and n > 1:
            pressure = np.tile(pressure, n)

        discrete = np.clip(states, 0, 1).astype(np.int64)

        # 8 raw continuous features: 7 freqs + 1 pressure
        continuous_raw = np.column_stack([freqs[:, :7], pressure.reshape(-1, 1)])

        cont_s = self.cont_scaler.transform(continuous_raw.astype(np.float64))

        d_t = torch.tensor(discrete, dtype=torch.long, device=self.device)
        c_t = torch.tensor(cont_s, dtype=torch.float32, device=self.device)
        return d_t, c_t

    def _correct_pressure(self, pressure, level):
        """修正后压力 = 原压力 - (吸水井液位 - 3.58) / 102 (MPa)，液位缺失按基准值处理(修正量=0)"""
        pressure = np.atleast_1d(np.asarray(pressure, dtype=np.float32))
        level = np.atleast_1d(np.asarray(level, dtype=np.float32))
        level = np.nan_to_num(level, nan=self.level_baseline)
        return pressure - (level - self.level_baseline) / self.level_divisor

    def predict(self, states, freqs, pressure, level):
        """
        返回 (170:1流量, 170:2流量, 70:3流量, 总管流量, 总管效率)。

        参数:
            states:   (7,) 或 (n,7) — 7泵状态 (0/1)
            freqs:    (7,) 或 (n,7) — 7泵频率 (Hz)
            pressure: float 或 (n,)  — 总管压力 (MPa, 原始值)
            level:    float 或 (n,)  — 吸水井液位 (m), 用于压力修正

        返回:
            flow_170_1: 170:1管道瞬时流量 (m3/h)
            flow_170_2: 170:2管道瞬时流量 (m3/h)
            flow_703:   70:3管道瞬时流量 (m3/h)
            total_flow: 总管流量 (m3/h) = sum(三条管道)
            efficiency: 总管效率 (%)
            kwt:        千吨水电耗 (kWh/千吨) = 272.5 × H_eff / 效率%
        """
        d_t, c_t = self._preprocess(states, freqs, self._correct_pressure(pressure, level))

        with torch.no_grad():
            out_s = self.model(d_t, c_t).cpu().numpy()

        out = self.out_scaler.inverse_transform(out_s)
        out = np.maximum(out, 0)
        out[:, 3] = np.clip(out[:, 3], 0, 100)  # 效率裁剪

        # 三条分管流量
        flow_170_1 = out[:, 0]
        flow_170_2 = out[:, 1]
        flow_703   = out[:, 2]
        total_flow = out[:, 0] + out[:, 1] + out[:, 2]   # 3管流量之和
        efficiency = out[:, 3]                            # 效率

        # 千吨水电耗 (kWh/千吨): H_eff = 压力×102 − (液位−2.35), kwt = 272.5×H_eff/效率
        pressure_arr = np.atleast_1d(np.asarray(pressure, dtype=np.float64))
        level_arr = np.atleast_1d(np.asarray(level, dtype=np.float64))
        H_eff = pressure_arr * 102 - (level_arr - PUMP_LEVEL)
        with np.errstate(divide='ignore', invalid='ignore'):
            kwt = np.where((efficiency > 0) & (H_eff > 0),
                           KWT_COEF * H_eff / efficiency, 0.0)

        if total_flow.shape[0] == 1:
            return (float(flow_170_1[0]), float(flow_170_2[0]),
                    float(flow_703[0]), float(total_flow[0]), float(efficiency[0]),
                    float(kwt[0]))
        return flow_170_1, flow_170_2, flow_703, total_flow, efficiency, kwt

    def combo_confidence(self, states):
        """
        泵组组合置信度: 组合在真实数据中出现过 → 高 (1.0), 未出现过 → 低 (0.0)。

        返回 (confidence, count):
          confidence: 1.0 (出现过) / 0.0 (未出现)
          count:      该组合在真实数据中的出现次数 (未出现为 0)
        """
        states = np.atleast_2d(np.asarray(states, dtype=np.int64))
        results = []
        for row in states:
            combo = ''.join(str(int(s)) for s in row)
            cnt = APPEARED_COMBOS.get(combo, 0)
            results.append((1.0 if cnt > 0 else 0.0, cnt))
        return results[0] if len(results) == 1 else results

    def predict_with_confidence(self, states, freqs, pressure, level):
        """
        同 predict(), 额外返回组合置信度 (出现过=高 1.0 / 未出现=低 0.0)。

        返回:
            flow_170_1, flow_170_2, flow_703, total_flow, efficiency, kwt  (同 predict)
            confidence: 单样本返回 (conf, count); 多样本返回两个列表
        """
        f1, f2, f3, total, eff, kwt = self.predict(states, freqs, pressure, level)
        conf = self.combo_confidence(states)
        return f1, f2, f3, total, eff, kwt, conf

    def predict_detail(self, states, freqs, pressure, level):
        """
        返回全部4维明细 + 总管值 + 千吨水电耗。pressure/level 含义同 predict()。

        返回:
            detail:     dict {col_name: value}  4个明细 + 'kwt' 千吨水电耗
            total_flow: 总管流量 (m3/h)
            efficiency: 总管效率 (%)
        """
        d_t, c_t = self._preprocess(states, freqs, self._correct_pressure(pressure, level))

        with torch.no_grad():
            out_s = self.model(d_t, c_t).cpu().numpy()

        out = self.out_scaler.inverse_transform(out_s)
        out = np.maximum(out, 0)
        out[:, 3] = np.clip(out[:, 3], 0, 100)

        detail = {self.all_output_cols[i]: float(out[0, i]) for i in range(4)}

        total_flow = float(out[0, 0] + out[0, 1] + out[0, 2])
        efficiency = float(out[0, 3])

        # 千吨水电耗 (kWh/千吨): H_eff = 压力×102 − (液位−2.35), kwt = 272.5×H_eff/效率
        H_eff = float(np.atleast_1d(pressure)[0]) * 102 - (float(np.atleast_1d(level)[0]) - PUMP_LEVEL)
        detail['kwt'] = KWT_COEF * H_eff / efficiency if (efficiency > 0 and H_eff > 0) else 0.0

        return detail, total_flow, efficiency

    def info(self):
        """打印模型信息"""
        print(f"PumpInference")
        print(f"  压力修正: 修正后压力 = 原压力 - (吸水井液位-{self.level_baseline})/{self.level_divisor} MPa")
        print(f"  设备: {self.device}")
        print(f"  输出明细 ({len(self.all_output_cols)}维): {self.all_output_cols}")
        print(f"  汇总输出: 170:1流量, 170:2流量, 70:3流量, 总管流量 (m3/h), 总管效率 (%)")
        print(f"  置信度: 组合在真实数据中出现过→高 / 未出现→低 (库内 {len(APPEARED_COMBOS)} 种真实组合)")


# ============================================================================
# 演示
# ============================================================================

if __name__ == '__main__':
    model_path = r"D:\Wuhan_Project\models\model_v2_combo_split.pt"

    if not os.path.exists(model_path):
        print(f"权重文件不存在: {model_path}")
        print("请先运行 CQU_improve_v6_flow_eff_only.py 训练生成权重")
        import sys; sys.exit(1)

    model = PumpInference(model_path)
    model.info()

    # ════════════════════════════════════════════════════════════════════════
    # 第一组: 实际数据中出现过的泵组组合 (15种, 按出现占比降序)
    #   压力/液位/频率 = 该组合在真实数据中的均值。真实范围:
    #   总管压力 P1~P99 = 0.296~0.342 MPa, 吸水井液位 P1~P99 = 2.51~3.98 m,
    #   以下所有用例的压力/液位均落在真实范围内。
    # ════════════════════════════════════════════════════════════════════════
    cases_appeared = [
        ("1000010 P1+P6        (11.6%)", [1, 0, 0, 0, 0, 1, 0], [44.7, 0, 0, 0, 0, 50.0, 0],       0.3106, 3.32),
        ("1100010 P1+P2+P6     (10.6%)", [1, 1, 0, 0, 0, 1, 0], [48.8, 46.2, 0, 0, 0, 50.0, 0],    0.3318, 3.28),
        ("1100011 P1+P2+P6+P7   (8.7%)", [1, 1, 0, 0, 0, 1, 1], [48.3, 45.5, 0, 0, 0, 50.0, 49.9], 0.3327, 3.23),
        ("1101000 P1+P2+P4      (7.5%)", [1, 1, 0, 1, 0, 0, 0], [48.8, 46.5, 0, 50.0, 0, 0, 0],    0.3318, 3.25),
        ("0001010 P4+P6         (7.4%)", [0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 45.1, 0, 50.0, 0],       0.3078, 3.42),
        ("1001000 P1+P4         (7.2%)", [1, 0, 0, 1, 0, 0, 0], [45.2, 0, 0, 50.0, 0, 0, 0],       0.3120, 3.32),
        ("1101001 P1+P2+P4+P7   (6.6%)", [1, 1, 0, 1, 0, 0, 1], [49.1, 47.5, 0, 50.0, 0, 0, 49.9], 0.3328, 3.11),
        ("1000011 P1+P6+P7      (5.7%)", [1, 0, 0, 0, 0, 1, 1], [48.3, 0, 0, 0, 0, 50.0, 50.0],    0.3298, 3.36),
        ("0101011 P2+P4+P6+P7   (5.5%)", [0, 1, 0, 1, 0, 1, 1], [0, 45.6, 0, 49.4, 0, 50.0, 49.9], 0.3306, 3.38),
        ("0110010 P2+P3+P6      (3.3%)", [0, 1, 1, 0, 0, 1, 0], [0, 45.9, 49.3, 0, 0, 50.0, 0],    0.3325, 3.20),
        ("1001011 P1+P4+P6+P7   (2.9%)", [1, 0, 0, 1, 0, 1, 1], [46.8, 0, 0, 50.0, 0, 50.0, 49.9], 0.3352, 3.00),
        ("0010010 P3+P6         (2.8%)", [0, 0, 1, 0, 0, 1, 0], [0, 0, 45.2, 0, 0, 50.0, 0],       0.3120, 3.27),
        ("0101010 P2+P4+P6      (2.6%)", [0, 1, 0, 1, 0, 1, 0], [0, 46.7, 0, 49.6, 0, 50.0, 0],    0.3307, 3.19),
        ("0110011 P2+P3+P6+P7   (2.4%)", [0, 1, 1, 0, 0, 1, 1], [0, 46.4, 49.4, 0, 0, 50.0, 49.8], 0.3344, 2.93),
        ("1000110 P1+P5+P6      (2.1%)", [1, 0, 0, 0, 1, 1, 0], [48.4, 0, 0, 0, 45.6, 50.0, 0],    0.3305, 3.36),
    ]

    # ════════════════════════════════════════════════════════════════════════
    # 第二组: 实际数据中从未出现过的泵组组合 (全站 128 种组合中 67 种未出现)
    #   注意: 泵3/5/7 与 泵1/2/4/6 分属两条出水管线, 七台泵不能同时全开,
    #   因此不列 1111111 (全开) 等物理上不可行的极端工况; 也不列单泵工况
    #   (最少两台泵运行)。
    #   压力/液位取全局中位数 0.3308 MPa / 3.26 m (在真实范围内);
    #   频率取运行泵典型值: P1~P6 = 48/46/48/49/46 Hz, P7 = 50 Hz
    # ════════════════════════════════════════════════════════════════════════
    cases_never = [
        ("0100010 P2+P6 (未出现)",      [0, 1, 0, 0, 0, 1, 0], [0, 46.0, 0, 0, 0, 50.0, 0],        0.3308, 3.26),
        ("0101000 P2+P4 (未出现)",      [0, 1, 0, 1, 0, 0, 0], [0, 46.0, 0, 49.0, 0, 0, 0],        0.3308, 3.26),
        ("0000011 P6+P7 (未出现)",      [0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 50.0, 50.0],       0.3308, 3.26),
        ("1010000 P1+P3 (未出现)",      [1, 0, 1, 0, 0, 0, 0], [48.0, 0, 48.0, 0, 0, 0, 0],       0.3308, 3.26),
        ("0100100 P2+P5 (未出现)",      [0, 1, 0, 0, 1, 0, 0], [0, 46.0, 0, 0, 46.0, 0, 0],       0.3308, 3.26),
    ]
    cases = cases_appeared + cases_never

    # 泵额定流量 (m3/h, 顺序 P1~P7), 相似定律 Q∝f: 理论流量 = Q额定 × f/50
    # 管1+管2 ← 泵1~6 (两者合计), 管3 ← 泵7
    RATED_Q = [2020, 670, 2020, 1260, 670, 1260, 670]

    print("\n" + "=" * 78)
    print("推理测试: 实际出现过的 15 种组合 (真实均值工况) + 从未出现的 5 种组合")
    print("=" * 78)
    for i, (name, states, freqs, pressure, level) in enumerate(cases):
        if i == 0:
            print("\n──── 第一组: 实际数据中出现过的组合 (按占比降序) ────")
        elif i == len(cases_appeared):
            print("\n──── 第二组: 实际数据中从未出现的组合 ────")
        f1, f2, f3, total, eff, kwt, conf = model.predict_with_confidence(states, freqs, pressure, level)
        combo = ''.join('1' if s else '0' for s in states)
        conf_txt = f"置信度: 高 (出现 {conf[1]:,} 次)" if conf[0] > 0 else "置信度: 低 (未出现过)"

        # 理论流量: 管1+管2 ← Σ 泵1~6 Q额定×f/50; 管3 ← 泵7 Q额定×f/50
        t12 = sum(RATED_Q[i] * freqs[i] / 50.0 for i in range(6))
        t3 = RATED_Q[6] * freqs[6] / 50.0
        p12 = f1 + f2

        def pct(diff, t):
            return f"{100 * diff / t:+.0f}%" if t > 0 else "  -"

        print(f"\n[{name}]  开启组合={combo}  压力={pressure}  液位={level}")
        print(f"  频率: {freqs}  |  {conf_txt}")
        print(f"  预测: 管1+管2={p12:7.0f}  管3={f3:6.0f} m3/h | "
              f"总管={total:7.0f} m3/h | 效率={eff:5.1f}% | 千吨水电耗={kwt:6.1f} kWh")
        print(f"  理论: 管1+管2={t12:7.0f}  管3={t3:6.0f} m3/h")
        print(f"  偏差: 管1+管2={p12 - t12:+7.0f} m3/h ({pct(p12 - t12, t12)})  "
              f"管3={f3 - t3:+6.0f} m3/h ({pct(f3 - t3, t3)})")

    # 明细 (以第一个工况为例)
    detail, f, e = model.predict_detail(cases[0][1], cases[0][2], cases[0][3], cases[0][4])
    print(f"\n明细输出 (以 '{cases[0][0]}' 为例):")
    for k, v in detail.items():
        print(f"  {k}: {v:.2f}")
