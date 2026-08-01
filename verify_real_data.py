# -*- coding: utf-8 -*-
"""
用真实运行数据回验模型预测 (重点: 70:3 管3流量)
取 CSV 中 P7 满频(50Hz) 运行的记录, 真实状态/频率/压力/液位输入模型,
对比预测管3流量 与 实测 70:3_瞬时流量。
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'D:\Wuhan_Project')
from pump_inference import PumpInference

CSV = r'D:\Wuhan_Project\new_data\merged_minute_all_with_efficiency.csv'

pump_cols = [f'170:{i}_泵运行' for i in range(1, 7)] + ['70:7_泵运行']
freq_cols = [f'170:{i}_运行频率' for i in range(1, 7)] + ['70:7_运行频率']
flow_cols = ['170:1_瞬时流量', '170:2_瞬时流量', '70:3_瞬时流量']
base_cols = ['170:总管压力', '170:吸水井液位']

head = pd.read_csv(CSV, encoding='utf-8-sig', nrows=1)
eff_cols = [c for c in head.columns if '效率' in c]
print('效率列:', eff_cols)

usecols = pump_cols + freq_cols + flow_cols + base_cols + eff_cols
df = pd.read_csv(CSV, encoding='utf-8-sig', usecols=usecols)

# 数值化 + 剔除缺失
for c in pump_cols + freq_cols + flow_cols + base_cols + eff_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=base_cols + flow_cols)
df = df[(df['70:7_泵运行'] == 1) & (df['70:7_运行频率'] >= 49.9)].reset_index(drop=True)
print(f'P7 满频(50Hz)运行样本: {len(df)}')

# 均匀抽样 30 个样本
n = len(df)
idx = np.linspace(0, n - 1, 30).astype(int)
s = df.iloc[idx].copy()

states = s[pump_cols].values.astype(int)
freqs = s[freq_cols].values.astype(float)
pres = s['170:总管压力'].values.astype(float)
level = s['170:吸水井液位'].values.astype(float)

model = PumpInference()
f1, f2, f3, total, eff = model.predict(states, freqs, pres, level)
f1, f2, f3, total, eff = map(np.asarray, (f1, f2, f3, total, eff))

a1 = s['170:1_瞬时流量'].values.astype(float)
a2 = s['170:2_瞬时流量'].values.astype(float)
a3 = s['70:3_瞬时流量'].values.astype(float)
a_total = a1 + a2 + a3

print('\n' + '=' * 104)
print('样本 | 泵组组合 | P7频率 | 压力 | 液位 | 实际管3 | 预测管3 | 管3误差% | 实际总管 | 预测总管 | 总管误差%')
print('=' * 104)
for i in range(len(s)):
    combo = ''.join(str(states[i, j]) for j in range(7))
    e3 = 100 * (f3[i] - a3[i]) / a3[i] if a3[i] > 0 else float('nan')
    et = 100 * (total[i] - a_total[i]) / a_total[i] if a_total[i] > 0 else float('nan')
    print(f'{i+1:>4} | {combo} | {freqs[i,6]:5.1f} | {pres[i]:.3f} | {level[i]:.2f} | '
          f'{a3[i]:7.1f} | {f3[i]:7.1f} | {e3:+7.1f} | {a_total[i]:8.1f} | {total[i]:8.1f} | {et:+7.1f}')

print('=' * 104)
re3 = np.abs(f3 - a3) / a3 * 100
ret = np.abs(total - a_total) / a_total * 100
print(f'管3(70:3):   实际均值 {a3.mean():7.1f} | 预测均值 {f3.mean():7.1f} | MAE {np.mean(np.abs(f3-a3)):6.1f} | '
      f'平均相对误差 {re3.mean():5.1f}% | 最大 {re3.max():5.1f}%')
print(f'总管流量:    实际均值 {a_total.mean():7.1f} | 预测均值 {total.mean():7.1f} | MAE {np.mean(np.abs(total-a_total)):6.1f} | '
      f'平均相对误差 {ret.mean():5.1f}%')

if eff_cols:
    ae = s[eff_cols[0]].values.astype(float)
    print(f'效率({eff_cols[0]}): 实际均值 {ae.mean():6.2f} | 预测均值 {eff.mean():6.2f}')
