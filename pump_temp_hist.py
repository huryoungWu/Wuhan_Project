# -*- coding: utf-8 -*-
"""
统计 merged_minute_all.csv 中 6 台泵 (170:1~6, P7 无温度传感器) 的温度频数分布
并统计温度过高报警标志 (170:i_温度过高) 及高温段数据
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CSV_PATH = r"D:\Wuhan_Project\new_data\merged_minute_all.csv"
OUT_PATH = r"D:\Wuhan_Project\output\pump_temp_distribution.png"

TEMP_COLS = [f"170:{i}_温度" for i in range(1, 7)]
ALARM_COLS = [f"170:{i}_温度过高" for i in range(1, 7)]
FREQ_COLS = [f"170:{i}_运行频率" for i in range(1, 7)]
PUMP_NAMES = [f"P{i}" for i in range(1, 7)]

# ---- dataviz 规范: 浅色表面 + 分类色板 1~6 ----
SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
HIGH_TEMP = 60.0          # 高温段分析阈值 (°C)

print("读取数据...")
df = pd.read_csv(CSV_PATH, usecols=TEMP_COLS + ALARM_COLS + FREQ_COLS, dtype=np.float32)
n_total = len(df)
print(f"总行数: {n_total}")

# ---- 温度过高报警统计 ----
alarm_summary = []
for i, col in enumerate(ALARM_COLS):
    a = df[col].to_numpy(dtype=np.float32)
    v = ~np.isnan(a)
    n_alarm = int((a[v] > 0.5).sum())
    alarm_summary.append(n_alarm)
total_alarm_samples = sum(alarm_summary)
print(f"\n温度过高报警 (标志=1) 样本: 各泵 {alarm_summary}, 合计 {total_alarm_samples:,}")

BINS = np.arange(15, 82, 0.5)
fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), dpi=110)
fig.patch.set_facecolor(SURFACE)
fig.suptitle("6 台泵温度频数分布直方图（merged_minute_all.csv，P7 无温度传感器）",
             fontsize=17, color=PRIMARY_INK, fontweight="bold", y=0.98)

summary_lines = [f"温度过高报警标志=1 的样本: {total_alarm_samples:,} (全部为 0, 无报警事件)"]
for i, name in enumerate(PUMP_NAMES):
    ax = axes[i // 4][i % 4]
    ax.set_facecolor(SURFACE)

    t = df[TEMP_COLS[i]].to_numpy(dtype=np.float32)
    f = df[FREQ_COLS[i]].to_numpy(dtype=np.float32)
    valid = ~np.isnan(t)
    t_valid = t[valid]
    running = valid & (f > 1.0)      # 运行: 频率>1Hz
    n_nan = n_total - valid.sum()

    ax.hist(t_valid, bins=BINS, color=SERIES[i], edgecolor=SURFACE, linewidth=0.4)

    for spine in ax.spines.values():
        spine.set_color(BASELINE)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)

    ax.set_title(name, color=PRIMARY_INK, fontsize=12)
    ax.set_xlabel("温度 (°C)", color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel("样本数 (分钟)", color=SECONDARY_INK, fontsize=10)
    ax.set_xlim(15, 82)

    n_run = running.sum()
    n_high = int((t_valid >= HIGH_TEMP).sum())
    n_high_run = int((t[running] >= HIGH_TEMP).sum())
    t_run_med = np.median(t[running]) if n_run else np.nan
    t_idle_med = np.median(t[valid & ~running]) if (valid & ~running).sum() else np.nan

    stat_txt = (f"运行温度中位 {t_run_med:.1f}°C\n"
                f"停机温度中位 {t_idle_med:.1f}°C\n"
                f"整体中位 {np.median(t_valid):.1f}°C | P99 {np.percentile(t_valid, 99):.1f}°C\n"
                f"最高 {t_valid.max():.1f}°C | ≥{HIGH_TEMP:.0f}°C: {n_high:,} 条 "
                f"({n_high_run:,} 运行)")
    ax.text(0.985, 0.985, stat_txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=SECONDARY_INK,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff",
                      edgecolor=BASELINE, alpha=0.85))

    summary_lines.append(
        f"{name}: 温度中位 {np.median(t_valid):.1f}°C (运行 {t_run_med:.1f} / 停机 {t_idle_med:.1f}), "
        f"P99={np.percentile(t_valid, 99):.1f}°C, 最高 {t_valid.max():.1f}°C, "
        f"≥{HIGH_TEMP:.0f}°C 共 {n_high:,} 条 ({n_high_run:,} 条为运行期), "
        f"缺失 {n_nan / n_total * 100:.1f}%, 报警 {alarm_summary[i]:,}")

# 第 7 格: 汇总面板
ax = axes[1][2]
ax.set_facecolor(SURFACE)
ax.axis("off")
ax.text(0.02, 0.98, "\n".join(summary_lines), transform=ax.transAxes,
        ha="left", va="top", fontsize=10, color=PRIMARY_INK,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff",
                  edgecolor=BASELINE, alpha=0.9))

# 第 8 格: 各泵运行期 ≥60°C 高温样本数条形图
ax = axes[1][3]
ax.set_facecolor(SURFACE)
high_run_counts = []
for i in range(6):
    t = df[TEMP_COLS[i]].to_numpy(dtype=np.float32)
    f = df[FREQ_COLS[i]].to_numpy(dtype=np.float32)
    run = ~np.isnan(t) & (f > 1.0)
    high_run_counts.append(int((t[run] >= HIGH_TEMP).sum()))

bars = ax.bar(PUMP_NAMES, high_run_counts, color=SERIES,
              edgecolor=SURFACE, linewidth=0.5, width=0.62)
for b, v in zip(bars, high_run_counts):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
            ha="center", va="bottom", fontsize=10, color=SECONDARY_INK)
for spine in ax.spines.values():
    spine.set_color(BASELINE)
ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, alpha=0.7)
ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, labelsize=10)
ax.set_title("运行期 ≥60°C 高温分钟数", color=PRIMARY_INK, fontsize=12)
ax.set_ylabel("样本数 (分钟)", color=SECONDARY_INK, fontsize=10)
ax.ticklabel_format(axis="y", style="plain")

fig.text(0.5, 0.015,
         "运行判定: 运行频率 > 1 Hz; 停机温度反映环境温度基线; 温度过高报警标志全周期均为 0",
         ha="center", fontsize=10, color=MUTED)

plt.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig(OUT_PATH, facecolor=SURFACE, bbox_inches="tight")
print(f"\n图表已保存: {OUT_PATH}\n")
print("\n".join(summary_lines))
