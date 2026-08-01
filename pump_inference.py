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

工程特征自动生成，权重文件由 CQU_improve_v6_flow_eff_only.py 训练产生。

用法:
  model = PumpInference("models/model_v2_combo_split.pt")
  flow_170_1, flow_170_2, flow_703, total_flow, eff = model.predict(states, freqs, pressure, level)
"""

import os
import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# 模型定义 (与 CQU_improve_v6_flow_eff_only.py 完全一致 — 4维输出)
# ============================================================================

class DeepWaterPlantModelWithEmbedding(nn.Module):
    """v6 纯流量+效率模型: 输出 [170:1流量, 170:2流量, 70:3流量, 总管效率]"""
    def __init__(self, num_pumps, continuous_dim, output_dim=4, embed_dim=8,
                 pump_embedding_sizes=None):
        super().__init__()
        self.num_pumps = num_pumps
        self.continuous_dim = continuous_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        if pump_embedding_sizes is None:
            pump_embedding_sizes = [2] * num_pumps
        self.pump_embedding_sizes = pump_embedding_sizes

        # ── Embedding ──
        self.pump_embeddings = nn.ModuleList([
            nn.Embedding(self.pump_embedding_sizes[i], embed_dim)
            for i in range(num_pumps)
        ])
        total_discrete_dim = num_pumps * embed_dim
        total_input_dim = total_discrete_dim + continuous_dim

        # ── 共享编码器 (带残差) ──
        self.enc_fc1 = nn.Linear(total_input_dim, 512)
        self.enc_bn1 = nn.BatchNorm1d(512)
        self.enc_fc2 = nn.Linear(512, 256)
        self.enc_bn2 = nn.BatchNorm1d(256)
        self.enc_fc3 = nn.Linear(256, 128)
        self.enc_bn3 = nn.BatchNorm1d(128)
        self.enc_proj1 = nn.Linear(total_input_dim, 512)
        self.enc_proj2 = nn.Linear(512, 256)

        # ── 泵组互斥注意力 ──
        self.pair_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=4, batch_first=True, dropout=0.0
        )

        # ── 170系统头 → [170:1流量, 170:2流量] ──
        self.head_170_fc1 = nn.Linear(128, 64)
        self.head_170_bn1 = nn.BatchNorm1d(64)
        self.head_170_fc2 = nn.Linear(64, 32)
        self.head_170_bn2 = nn.BatchNorm1d(32)
        self.head_170_out = nn.Linear(32, 2)

        # ── 70:3专用头 (P7频率捷径) → [70:3流量] ──
        self.head_703_linear = nn.Linear(1, 16)
        self.head_703_fc1 = nn.Linear(128 + 16, 48)
        self.head_703_bn1 = nn.BatchNorm1d(48)
        self.head_703_fc2 = nn.Linear(48, 24)
        self.head_703_bn2 = nn.BatchNorm1d(24)
        self.head_703_out = nn.Linear(24, 1)

        # ── 效率头 → [总管效率] ──
        self.head_eff_fc1 = nn.Linear(128, 48)
        self.head_eff_bn1 = nn.BatchNorm1d(48)
        self.head_eff_fc2 = nn.Linear(48, 24)
        self.head_eff_bn2 = nn.BatchNorm1d(24)
        self.head_eff_out = nn.Linear(24, 1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(0.0)  # 推理时关闭

    def forward(self, discrete_x, continuous_x):
        batch_size = discrete_x.shape[0]

        # 1. Embedding
        embedded_pumps = []
        for i in range(self.num_pumps):
            idx = discrete_x[:, i].clamp(0, self.pump_embedding_sizes[i] - 1)
            emb = self.pump_embeddings[i](idx)
            embedded_pumps.append(emb)
        pump_emb_stack = torch.stack(embedded_pumps, dim=1)

        # 2. 泵组互斥注意力
        attn_out, _ = self.pair_attention(pump_emb_stack, pump_emb_stack, pump_emb_stack)
        discrete_embedded = attn_out.reshape(batch_size, -1)

        # 3. 共享编码器 (带残差)
        x = torch.cat([discrete_embedded, continuous_x], dim=1)

        h1 = self.enc_fc1(x)
        h1 = self.enc_bn1(h1)
        h1 = self.leaky_relu(h1)
        h1 = h1 + self.enc_proj1(x)

        h2 = self.enc_fc2(h1)
        h2 = self.enc_bn2(h2)
        h2 = self.leaky_relu(h2)
        h2 = h2 + self.enc_proj2(h1)

        shared = self.enc_fc3(h2)
        shared = self.enc_bn3(shared)
        shared = self.leaky_relu(shared)

        # 4a. 170系统头
        h_170 = self.head_170_fc1(shared)
        h_170 = self.head_170_bn1(h_170)
        h_170 = self.leaky_relu(h_170)
        h_170 = self.head_170_fc2(h_170)
        h_170 = self.head_170_bn2(h_170)
        h_170 = self.leaky_relu(h_170)
        out_170 = self.head_170_out(h_170)

        # 4b. 70:3专用头 (P7频率捷径)
        p7_freq = continuous_x[:, 6:7]
        p7_feat = self.head_703_linear(p7_freq)
        p7_feat = self.leaky_relu(p7_feat)

        h_703 = torch.cat([shared, p7_feat], dim=1)
        h_703 = self.head_703_fc1(h_703)
        h_703 = self.head_703_bn1(h_703)
        h_703 = self.leaky_relu(h_703)
        h_703 = self.head_703_fc2(h_703)
        h_703 = self.head_703_bn2(h_703)
        h_703 = self.leaky_relu(h_703)
        out_703 = self.head_703_out(h_703)

        # 4c. 效率头
        h_eff = self.head_eff_fc1(shared)
        h_eff = self.head_eff_bn1(h_eff)
        h_eff = self.leaky_relu(h_eff)
        h_eff = self.head_eff_fc2(h_eff)
        h_eff = self.head_eff_bn2(h_eff)
        h_eff = self.leaky_relu(h_eff)
        out_eff = self.head_eff_out(h_eff)

        # 5. 拼接: [170:1流量, 170:2流量, 70:3流量, 总管效率]
        return torch.cat([out_170, out_703, out_eff], dim=1)


# ============================================================================
# ============================================================================
# 推理类 v6
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
        flow_170_1, flow_170_2, flow_703, total_flow, eff = model.predict(states, freqs, pressure, level)
    """

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "models", "model_v2_combo_split.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"权重文件不存在: {model_path}\n"
                f"请先运行 CQU_improve_v6_flow_eff_only.py 训练生成权重"
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

        if total_flow.shape[0] == 1:
            return (float(flow_170_1[0]), float(flow_170_2[0]),
                    float(flow_703[0]), float(total_flow[0]), float(efficiency[0]))
        return flow_170_1, flow_170_2, flow_703, total_flow, efficiency

    def predict_detail(self, states, freqs, pressure, level):
        """
        返回全部4维明细 + 总管值。pressure/level 含义同 predict()。

        返回:
            detail:     dict {col_name: value}  4个明细
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

        return detail, total_flow, efficiency

    def info(self):
        """打印模型信息"""
        print(f"PumpInference")
        print(f"  压力修正: 修正后压力 = 原压力 - (吸水井液位-{self.level_baseline})/{self.level_divisor} MPa")
        print(f"  设备: {self.device}")
        print(f"  输出明细 ({len(self.all_output_cols)}维): {self.all_output_cols}")
        print(f"  汇总输出: 170:1流量, 170:2流量, 70:3流量, 总管流量 (m3/h), 总管效率 (%)")


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

    # 典型工况
    states   = [1, 1, 0, 1, 1, 1, 1]
    freqs    = [48.7, 47.3, 0, 50.0, 41.5, 50.0, 50.0]
    pressure = 0.33
    level    = 3.42  # 吸水井液位 (m)

    flow_170_1, flow_170_2, flow_703, flow, eff = model.predict(states, freqs, pressure, level)
    print(f"\n示例输入: 泵状态={states}, 频率={freqs}, 压力={pressure}, 液位={level}")
    print(f"推理结果: 170:1流量={flow_170_1:.0f} m3/h, 170:2流量={flow_170_2:.0f} m3/h, "
          f"70:3流量={flow_703:.0f} m3/h")
    print(f"          总管流量={flow:.0f} m3/h, 总管效率={eff:.1f}%")

    # 明细
    detail, f, e = model.predict_detail(states, freqs, pressure, level)
    print(f"\n明细输出:")
    for k, v in detail.items():
        print(f"  {k}: {v:.2f}")
