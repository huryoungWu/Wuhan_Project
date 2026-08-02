"""总管效率频数分布直方图 (修正后 v2 口径)

读取 merged_minute_all_with_efficiency_v2.csv 的 总管效率_pct 列,
绘制频数分布直方图并保存 PNG。
用法: python plot_efficiency_histogram.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import time

# ── 中文字体 ──
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ── dataviz 规范颜色 (浅色画布) ──
SURFACE      = '#fcfcfb'   # 图表画布
PRIMARY_INK  = '#0b0b0b'   # 主文字
SECONDARY_INK = '#52514e'  # 次文字
MUTED_INK    = '#898781'   # 轴刻度
GRIDLINE     = '#e1e0d9'   # 网格线 (hairline)
AXIS_BASE    = '#c3c2b7'   # 坐标轴
BAR_FILL     = '#2a78d6'   # 柱体 (categorical slot 1 蓝)
MEAN_LINE    = '#104281'   # 均值线 (深蓝)
MEDIAN_LINE  = '#eb6834'   # 中位数线 (橙)

SRC = r"D:\Wuhan_Project\new_data\merged_minute_all_with_efficiency_v2.csv"
OUT = r"D:\Wuhan_Project\效率频数分布直方图.png"
BIN_WIDTH = 1.0            # 分箱宽度 (%)
X_LO, X_HI = 35.0, 90.0    # 绘图范围 (%)


def main():
    print(f"加载: {SRC}")
    t0 = time.time()
    df = pd.read_csv(SRC, encoding='utf-8-sig', usecols=['总管效率_pct'])
    s = df['总管效率_pct']
    n = len(s)
    print(f"N={n:,}  耗时 {time.time()-t0:.1f}s")

    # 分箱
    bins = np.arange(X_LO, X_HI + BIN_WIDTH, BIN_WIDTH)
    counts, edges = np.histogram(s, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    # 统计量
    mean, med, std = s.mean(), s.median(), s.std()
    p5, p95 = s.quantile(0.05), s.quantile(0.95)
    peak_bin = centers[np.argmax(counts)]
    peak_cnt = counts.max()
    clipped = int((s >= 90.0).sum())

    # ── 绘图 ──
    fig, ax = plt.subplots(figsize=(11.5, 6.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # 柱体 (细白边模拟相邻柱间隙)
    ax.bar(centers, counts, width=BIN_WIDTH * 0.94, color=BAR_FILL,
           edgecolor=SURFACE, linewidth=0.6, zorder=3)

    # 均值/中位数参考线
    ax.axvline(mean, color=MEAN_LINE, ls='--', lw=1.4, zorder=4)
    ax.axvline(med, color=MEDIAN_LINE, ls='--', lw=1.4, zorder=4)
    y_max = counts.max()
    ax.text(mean, y_max * 0.97, f'均值 {mean:.2f}%', color=MEAN_LINE,
            ha='center', va='top', fontsize=10)
    ax.text(med, y_max * 0.90, f'中位数 {med:.2f}%', color=MEDIAN_LINE,
            ha='center', va='top', fontsize=10)

    # 峰值标注
    ax.annotate(f'峰值 {peak_bin:.0f}%  (频数 {peak_cnt:,})',
                xy=(peak_bin, peak_cnt), xytext=(peak_bin + 3.2, peak_cnt * 0.82),
                arrowprops=dict(arrowstyle='->', color=MUTED_INK, lw=1.0),
                color=SECONDARY_INK, fontsize=10)

    # 坐标轴样式
    ax.set_xlim(X_LO - 0.8, X_HI)
    ax.set_ylim(0, y_max * 1.06)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d%%'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f'{v:,.0f}'))
    ax.set_xlabel('总管效率 (%)', color=PRIMARY_INK, fontsize=11)
    ax.set_ylabel('频数 (条)', color=PRIMARY_INK, fontsize=11)
    ax.tick_params(colors=MUTED_INK, labelsize=10)
    ax.grid(axis='y', color=GRIDLINE, lw=0.8, zorder=0)
    ax.grid(axis='x', visible=False)
    for spine in ax.spines.values():
        spine.set_color(AXIS_BASE)

    # 标题与统计框
    ax.set_title('总管效率频数分布直方图 (修正后口径 v2)',
                 color=PRIMARY_INK, fontsize=14, pad=14)
    stat_txt = (f'N = {n:,}\n'
                f'均值 {mean:.2f}%  中位数 {med:.2f}%\n'
                f'标准差 {std:.2f}%   P5~P95: {p5:.1f}%~{p95:.1f}%')
    ax.text(0.015, 0.965, stat_txt, transform=ax.transAxes, va='top', ha='left',
            fontsize=9.5, color=SECONDARY_INK,
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#f4f3ef',
                      edgecolor=AXIS_BASE, lw=0.8), zorder=5)

    if clipped > 0:
        ax.text(0.985, 0.02,
                f'注: 效率按 0~90% 裁剪, 90% 处含被裁剪值 {clipped:,} 条',
                transform=ax.transAxes, va='bottom', ha='right',
                fontsize=8.5, color=MUTED_INK)

    fig.tight_layout()
    fig.savefig(OUT, facecolor=SURFACE, bbox_inches='tight')
    print(f"已保存: {OUT}")


if __name__ == '__main__':
    main()
