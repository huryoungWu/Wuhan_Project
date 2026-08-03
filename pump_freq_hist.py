# -*- coding: utf-8 -*-
"""
统计 merged_minute_all.csv 中 7 台泵运行频率的分布并绘制直方图
7 台泵 = 170:1~170:6 + 70:7
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CSV_PATH = r"D:\Wuhan_Project\new_data\merged_minute_all.csv"
OUT_PATH = r"D:\Wuhan_Project\results\pump_freq_distribution.png"

FREQ_COLS = [
    "170:1_运行频率", "170:2_运行频率", "170:3_运行频率",
    "170:4_运行频率", "170:5_运行频率", "170:6_运行频率",
    "70:7_运行频率",
]
PUMP_NAMES = ["P1 (170:1)", "P2 (170:2)", "P3 (170:3)", "P4 (170:4)",
              "P5 (170:5)", "P6 (170:6)", "P7 (70:7)"]

# ---- dataviz 规范：浅色表面 + 分类色板固定顺序 1~7 ----
SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]

print("读取数据...")
df = pd.read_csv(CSV_PATH, usecols=FREQ_COLS, dtype=np.float32)
n_total = len(df)
print(f"总行数: {n_total}")

BINS = np.arange(0, 52, 0.5)  # 0~52 Hz, 0.5Hz 一档

fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), dpi=110)
fig.patch.set_facecolor(SURFACE)
fig.suptitle("7 台泵运行频率分布直方图（merged_minute_all.csv）",
             fontsize=17, color=PRIMARY_INK, fontweight="bold", y=0.98)

summary_lines = []
for i, (col, name) in enumerate(zip(FREQ_COLS, PUMP_NAMES)):
    ax = axes[i // 4][i % 4]
    ax.set_facecolor(SURFACE)

    s = df[col].to_numpy(dtype=np.float32)
    valid = ~np.isnan(s)
    off_mask = valid & (s <= 1.0)    # 频率 <=1Hz 视为停机
    run = s[valid & (s > 1.0) & (s <= 52.0)]   # 排除 >52Hz 传感器异常尖峰
    n_nan = n_total - valid.sum()
    off_frac = off_mask.sum() / n_total * 100
    run_frac = len(run) / n_total * 100

    if len(run) == 0:
        ax.text(0.5, 0.5, "无运行数据", ha="center", va="center", color=MUTED)
        summary_lines.append(f"{name}: 无运行数据")
        ax.set_title(name, color=PRIMARY_INK, fontsize=12)
        continue

    ax.hist(run, bins=BINS, color=SERIES[i], edgecolor=SURFACE, linewidth=0.4)

    # 边框与网格：弱化
    for spine in ax.spines.values():
        spine.set_color(BASELINE)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)

    ax.set_title(name, color=PRIMARY_INK, fontsize=12)
    ax.set_xlabel("运行频率 (Hz)", color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel("样本数 (分钟)", color=SECONDARY_INK, fontsize=10)
    ax.set_xlim(0, 52)

    # 关键统计标注（含停机占比）
    n_run = len(run)
    stat_txt = (f"运行占比 {run_frac:.1f}%\n"
                f"停机占比 {off_frac:.1f}%\n"
                f"运行均值 {run.mean():.1f} Hz\n"
                f"中位数 {np.median(run):.1f} Hz\n"
                f"区间 [{run.min():.1f}, {run.max():.1f}] Hz")
    ax.text(0.985, 0.985, stat_txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=SECONDARY_INK,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff",
                      edgecolor=BASELINE, alpha=0.85))
    summary_lines.append(
        f"{name}: 运行 {len(run):,} ({run_frac:.1f}%), 停机 {off_frac:.1f}%, "
        f"缺失 {n_nan/n_total*100:.1f}%, 均值 {run.mean():.1f} Hz, "
        f"中位 {np.median(run):.1f} Hz, 范围 [{run.min():.1f}, {run.max():.1f}] Hz")

# 第 8 格：汇总说明
ax = axes[1][3]
ax.set_facecolor(SURFACE)
ax.axis("off")
summary = "\n".join(summary_lines)
ax.text(0.02, 0.98, summary, transform=ax.transAxes, ha="left", va="top",
        fontsize=10.5, color=PRIMARY_INK,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff",
                  edgecolor=BASELINE, alpha=0.9))

fig.text(0.5, 0.015,
         "停机判定：运行频率 <= 1 Hz 视为停机；直方图仅统计运行状态（>1 Hz）的样本",
         ha="center", fontsize=10, color=MUTED)

plt.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig(OUT_PATH, facecolor=SURFACE, bbox_inches="tight")
print(f"\n图表已保存: {OUT_PATH}\n")
print("\n".join(summary_lines))
