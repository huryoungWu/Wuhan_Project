"""各泵 频率-功率 密度散点图 + 幂律拟合曲线 (v2修正后功率)

百万级样本用 hexbin 显示密度 (顺序蓝), 变频泵叠加 P = a·fⁿ 拟合曲线 (橙)。
恒频泵 (6/7) 频率几乎恒定, 不做幂律拟合, 标注"恒频运行"。
用法: python plot_pump_power_freq.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import time
from scipy import stats

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ── dataviz 规范颜色 (浅色画布) ──
SURFACE       = '#fcfcfb'
PRIMARY_INK   = '#0b0b0b'
SECONDARY_INK = '#52514e'
MUTED_INK     = '#898781'
GRIDLINE      = '#e1e0d9'
AXIS_BASE     = '#c3c2b7'
FIT_LINE      = '#eb6834'   # 拟合曲线 (橙, categorical slot 2)
# 顺序蓝 (密度): 100→700 步
BLUE_STEPS = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec',
              '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab',
              '#184f95', '#104281', '#0d366b']
DENSITY_CMAP = mcolors.LinearSegmentedColormap.from_list('density_blue', BLUE_STEPS)

SRC = r"D:\Wuhan_Project\new_data\merged_minute_all_with_efficiency_v2.csv"
OUT = r"D:\Wuhan_Project\泵功率频率散点图.png"

PUMP_COLS = [(f'170:{i}_运行频率', f'泵{i}_功率_kW') for i in range(1, 7)]
PUMP_COLS += [('70:7_运行频率', '泵7_功率_kW')]
N_PANELS = len(PUMP_COLS)   # 7


def fit_power_law(f, p):
    """P = a·fⁿ 幂律拟合 (log-log 线性回归), 返回 (a, n, r², n_samples)"""
    m = (f > 5) & (p > 5) & (f < 49)
    if m.sum() < 100:
        return None
    lf, lp = np.log(f[m]), np.log(p[m])
    slope, intercept, r, _, _ = stats.linregress(lf, lp)
    return (np.exp(intercept), slope, r ** 2, int(m.sum()))


def main():
    print(f"加载: {SRC}")
    t0 = time.time()
    df = pd.read_csv(SRC, encoding='utf-8-sig',
                     usecols=[c for pair in PUMP_COLS for c in pair])
    print(f"N={len(df):,}  耗时 {time.time()-t0:.1f}s")

    # 预取运行样本 (降采样至每泵最多 80 万点, hexbin 提速)
    data = []
    fits = []
    for fr, pw in PUMP_COLS:
        run = (df[fr] > 0) & (df[pw] > 0)
        f, p = df.loc[run, fr].values, df.loc[run, pw].values
        if len(f) > 800_000:
            idx = np.random.default_rng(42).choice(len(f), 800_000, replace=False)
            f, p = f[idx], p[idx]
        r, _ = stats.pearsonr(f, p)
        data.append((f, p, r))
        fits.append(fit_power_law(f, p))

    # 全局密度色标上下限 (各面板可比)
    vmin_all, vmax_all = 1.0, 1.0
    for f, p, _ in data:
        hb = plt.hexbin(f, p, gridsize=48, mincnt=1, cmap=DENSITY_CMAP)
        vmax_all = max(vmax_all, hb.get_array().max())
        plt.close('all')
    print(f"密度色标: {vmin_all:.0f} ~ {vmax_all:.0f}")

    # ── 2×4 网格 ──
    fig, axes = plt.subplots(2, 4, figsize=(17, 9.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle('各泵 频率-功率 密度散点图与幂律拟合曲线 (功率口径 v2 修正)',
                 color=PRIMARY_INK, fontsize=15, y=0.99)
    axes_flat = axes.ravel()

    for k, ((fr, pw), (f, p, r), fit) in enumerate(zip(PUMP_COLS, data, fits)):
        ax = axes_flat[k]
        ax.set_facecolor(SURFACE)

        hb = ax.hexbin(f, p, gridsize=48, mincnt=1, cmap=DENSITY_CMAP,
                       vmin=vmin_all, vmax=vmax_all, linewidths=0.3,
                       edgecolors=SURFACE, zorder=2)

        pump_no = fr.split(':')[1].split('_')[0]
        if fit is not None:
            a, n, r2, nfit = fit
            f_curve = np.linspace(5, 50, 300)
            p_curve = a * f_curve ** n
            ax.plot(f_curve, p_curve, color=FIT_LINE, lw=2.2, zorder=4,
                    label=f'P = a·f$^{{{n:.2f}}}$  (R²={r2:.2f})')
            ax.legend(loc='lower right', fontsize=9, frameon=False,
                      labelcolor=SECONDARY_INK)
            tag = f'r={r:.2f}   n={n:.2f}   R²={r2:.2f}'
        else:
            tag = f'r={r:.2f}'
            if pump_no in ('6', '7'):
                tag += '  (恒频运行, 无拟合)'
        ax.text(0.985, 0.96, f'泵{pump_no}', transform=ax.transAxes,
                ha='right', va='top', fontsize=13, fontweight='bold',
                color=PRIMARY_INK)
        ax.text(0.985, 0.88, tag, transform=ax.transAxes, ha='right', va='top',
                fontsize=9, color=SECONDARY_INK)

        # 恒频泵: 标注运行集中区
        if pump_no in ('6', '7'):
            p_run = p[f > 49]
            if len(p_run):
                ax.axhline(p_run.mean(), color=MUTED_INK, lw=1.0, ls=':', zorder=3)
                ax.text(0.02, 0.05, f'运行功率均值 {p_run.mean():.0f} kW',
                        transform=ax.transAxes, fontsize=8.5, color=MUTED_INK)

        ax.set_xlim(0, 52)
        y_hi = np.percentile(p, 99.9) * 1.08
        ax.set_ylim(0, y_hi)
        ax.set_xlabel('运行频率 (Hz)', color=PRIMARY_INK, fontsize=10)
        ax.set_ylabel('功率 (kW)', color=PRIMARY_INK, fontsize=10)
        ax.tick_params(colors=MUTED_INK, labelsize=9)
        ax.grid(axis='both', color=GRIDLINE, lw=0.7, zorder=0)
        for spine in ax.spines.values():
            spine.set_color(AXIS_BASE)

    # 空面板 (第8格): 色标说明
    ax = axes_flat[7]
    ax.set_facecolor(SURFACE)
    ax.axis('off')
    sm = plt.cm.ScalarMappable(cmap=DENSITY_CMAP, norm=mcolors.Normalize(vmin=vmin_all, vmax=vmax_all))
    cb = fig.colorbar(sm, ax=ax, fraction=0.3, pad=0.05)
    cb.set_label('样本密度 (个/格)', color=SECONDARY_INK, fontsize=10)
    cb.ax.tick_params(colors=MUTED_INK)
    ax.text(0.5, 0.35, '拟合曲线: P = a·fⁿ (离心泵相似定律, 理论 n=3)\n恒频泵频率几乎恒定, 相关性无意义',
            transform=ax.transAxes, ha='center', fontsize=9.5, color=SECONDARY_INK)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT, facecolor=SURFACE, bbox_inches='tight')
    print(f"已保存: {OUT}")


if __name__ == '__main__':
    main()
