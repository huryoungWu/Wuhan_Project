"""每天总流量 (flowtotal) 随时间变化曲线

读取 new_data/merged_minute_all.csv:
  0) 清洗: 用 Hampel 滤波 (滚动中位数 + MAD) 检测严重偏离的异常值
     (巨值/负垃圾值/0值/冻结陈旧值), 不删除, 置 NaN 后用时间插值填补;
  1) 按日期分组, 每天绘制一张 "总流量 ~ 一天内时刻" 曲线,
     保存到 results/每日总流量曲线/ 下 (每天一张 PNG, 如 2026-06-11.png);
  2) 所有日期叠加到一张图 (按日期先后由浅蓝渐变到深蓝, 附日期色条),
     保存为 results/总流量_全部日期对比.png。

说明: "总流量" 取 flowtotal 列 (各瞬时流量之和); 单日曲线使用全局统一的
y 轴范围, 便于天与天之间直接比较。

用法: python plot_daily_flow_curves.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import to_rgb, Normalize, LinearSegmentedColormap
from matplotlib import cm
import numpy as np
import pandas as pd
import time
from pathlib import Path

# ── 中文字体 ──
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ── dataviz 规范颜色 (浅色画布) ──
SURFACE       = '#fcfcfb'   # 图表画布
PRIMARY_INK   = '#0b0b0b'   # 主文字
SECONDARY_INK = '#52514e'   # 次文字
MUTED_INK     = '#898781'   # 轴刻度
GRIDLINE      = '#e1e0d9'   # 网格线 (hairline)
AXIS_BASE     = '#c3c2b7'   # 坐标轴
FLOW_BLUE     = '#2a78d6'   # 单日曲线 (categorical slot 1 蓝)
RAMP_LO       = '#8fc0e8'   # 叠加图: 最早日期 (浅蓝)
RAMP_HI       = '#104281'   # 叠加图: 最晚日期 (深蓝)

# ── 异常值检测 (Hampel 滤波) ──
HAMPEL_WINDOW   = 600    # 滚动窗口 (秒, 10 分钟)
HAMPEL_K        = 10.0   # 阈值: |x - 局部中位数| > k * scale 判为异常
MAD_FLOOR_RATIO = 0.02   # scale 下限 = 局部中位数的 2% (防 MAD≈0 误报)

SRC      = r"D:\Wuhan_Project\new_data\merged_minute_all.csv"
OUT_DIR  = Path(r"D:\Wuhan_Project\results\每日总流量曲线")
COMBINED = Path(r"D:\Wuhan_Project\results\总流量_全部日期对比.png")

LINE_LW     = 1.4        # 单日曲线线宽 (≈2px @150dpi)
TICK_EVERY_H = 4         # x 轴刻度间隔 (小时)
MAX_PTS_DAY = 2000       # 叠加图单日降采样点数上限


def lerp_hex(c1, c2, t):
    """在两种颜色间线性插值 (t: 0~1), 返回十六进制颜色串。"""
    r1, g1, b1 = to_rgb(c1)
    r2, g2, b2 = to_rgb(c2)
    rgb = tuple(round(a + (b - a) * t) * 255 for a, b in ((r1, r2), (g1, g2), (b1, b2)))
    return '#%02x%02x%02x' % rgb


def detect_outliers(s, window=HAMPEL_WINDOW, k=HAMPEL_K, floor_ratio=MAD_FLOOR_RATIO):
    """Hampel 滤波: 基于滚动中位数 + MAD 的稳健离群检测。

    返回布尔 Series (True = 异常点)。scale = max(1.4826*MAD, floor_ratio*中位数),
    底部分数保证流量极稳定 (MAD≈0) 时仍不会把正常波动误判为异常。
    """
    med = s.rolling(window, center=True, min_periods=window // 2).median()
    res = (s - med).abs()
    mad = res.rolling(window, center=True, min_periods=window // 2).median()
    scale = np.maximum(1.4826 * mad, floor_ratio * med)
    return (res > k * scale).fillna(False)


def style_axes(ax):
    """统一的浅色画布/网格/轴样式。"""
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED_INK, labelsize=9)
    ax.grid(axis='y', color=GRIDLINE, lw=0.8, zorder=0)
    ax.grid(axis='x', visible=False)
    for sp in ax.spines.values():
        sp.set_color(AXIS_BASE)
    ax.set_xlabel('一天内时刻 (时)', color=PRIMARY_INK, fontsize=11)
    ax.set_ylabel('总流量', color=PRIMARY_INK, fontsize=11)


def main():
    print(f"加载: {SRC}")
    t0 = time.time()
    df = pd.read_csv(SRC, encoding='utf-8-sig', usecols=['F_DateTime', 'flowtotal'])
    df['F_DateTime'] = pd.to_datetime(df['F_DateTime'])
    df['flowtotal'] = pd.to_numeric(df['flowtotal'], errors='coerce')
    df = df.dropna(subset=['flowtotal']).sort_values('F_DateTime').reset_index(drop=True)

    # ── 清洗: 严重偏离的异常值不删除, 置 NaN 后用时间插值填补 ──
    flag = detect_outliers(df['flowtotal'])
    df['is_out'] = flag.values
    n_out = int(flag.sum())
    if n_out:
        df.loc[flag, 'flowtotal'] = np.nan
        filled = df.set_index('F_DateTime')['flowtotal'].interpolate(method='time')
        df['flowtotal'] = filled.to_numpy()
        n_edge = int(df['flowtotal'].isna().sum())   # 首/尾无法插值的点
        if n_edge:
            df = df[df['flowtotal'].notna()].reset_index(drop=True)
            print(f"首尾不可插值, 已删除 {n_edge} 条")
    print(f"有效数据 {len(df):,} 条  异常点插值 {n_out:,} 条 "
          f"({n_out / len(df):.3%})  耗时 {time.time()-t0:.1f}s")

    # 日期 + 一天内时刻 (小时, 0~24)
    df['date'] = df['F_DateTime'].dt.date
    df['hour'] = (df['F_DateTime'].dt.hour
                  + df['F_DateTime'].dt.minute / 60
                  + df['F_DateTime'].dt.second / 3600)

    days = sorted(df['date'].unique())
    n_days = len(days)
    print(f"共 {n_days} 天: {days[0]} ~ {days[-1]}")

    # 全局 y 轴范围 (所有天统一, 便于比较)
    y_lo, y_hi = df['flowtotal'].min(), df['flowtotal'].max()
    y_pad = (y_hi - y_lo) * 0.04

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1) 每天一张图 ──
    for i, d in enumerate(days):
        sub = df[df['date'] == d]
        n = len(sub)
        n_out_day = int(sub['is_out'].sum())
        cov = n / 86400.0  # 当日数据覆盖率 (按秒采样满一天=86400 条)

        fig, ax = plt.subplots(figsize=(11.5, 5), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        ax.plot(sub['hour'], sub['flowtotal'], color=FLOW_BLUE,
                lw=LINE_LW, zorder=3)
        ax.set_title(f'{d} 总流量随时间变化', color=PRIMARY_INK, fontsize=13, pad=10)
        style_axes(ax)
        ax.set_xlim(0, 24)
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(TICK_EVERY_H))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, p: f'{int(v):02d}:00'))
        ax.text(0.985, 0.02,
                f'有效 {n - n_out_day:,} 条 · 插值 {n_out_day:,} 条 · 覆盖 {cov:.0%}',
                transform=ax.transAxes, va='bottom', ha='right',
                fontsize=8.5, color=MUTED_INK)
        fig.tight_layout()
        out = OUT_DIR / f'{d}.png'
        fig.savefig(out, facecolor=SURFACE, bbox_inches='tight')
        plt.close(fig)
        print(f"[{i+1}/{n_days}] 已保存: {out}")

    df = df.drop(columns=['is_out'])

    # ── 2) 全部日期叠加一张图 (浅→深 = 日期由早到晚) ──
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    for i, d in enumerate(days):
        sub = df[df['date'] == d]
        stride = max(1, len(sub) // MAX_PTS_DAY)  # 线图降采样, 加速渲染
        sub = sub.iloc[::stride]
        t = i / (n_days - 1) if n_days > 1 else 0.0
        ax.plot(sub['hour'], sub['flowtotal'], color=lerp_hex(RAMP_LO, RAMP_HI, t),
                lw=1.0, alpha=0.9, zorder=3)

    style_axes(ax)
    ax.set_xlim(0, 24)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(TICK_EVERY_H))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, p: f'{int(v):02d}:00'))
    ax.set_title(f'总流量每日曲线叠加 (共 {n_days} 天, {days[0]} ~ {days[-1]})',
                 color=PRIMARY_INK, fontsize=13, pad=10)

    # 日期色条
    cmap = LinearSegmentedColormap.from_list('flow_ramp', [RAMP_LO, RAMP_HI])
    sm = cm.ScalarMappable(norm=Normalize(0, n_days - 1), cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.015)
    cbar.set_label('日期 (浅→深 = 由早到晚)', color=SECONDARY_INK, fontsize=10)
    cbar.ax.tick_params(colors=MUTED_INK, labelsize=9)
    cbar.ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, p: str(days[int(round(v))]) if 0 <= int(round(v)) < n_days else ''))

    fig.tight_layout()
    fig.savefig(COMBINED, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {COMBINED}")


if __name__ == '__main__':
    main()
