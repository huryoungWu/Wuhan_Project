# -*- coding: utf-8 -*-
"""Parse all metrics_result_*.txt files and compute per-indicator statistics."""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import glob
import re
import statistics
from collections import OrderedDict

files = sorted(glob.glob(r"D:\Wuhan_Project\results\metrics_result_*.txt"))
print(f"共找到 {len(files)} 个文件\n")

# indicator -> {metric -> [values]}
data = OrderedDict()
meta = {}  # indicator -> {"sample_count": ..., "header": ...}

metric_re = re.compile(r'^\s+(MAE|RMSE|MAPE|R²)\s+=\s+([-\d.]+)')
indicator_re = re.compile(r'^\s+\[(.+)\]\s*$')

for f in files:
    with open(f, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    cur_ind = None
    found_any = False
    for ln in lines:
        m = indicator_re.match(ln)
        if m:
            cur_ind = m.group(1)
            data.setdefault(cur_ind, {"MAE": [], "RMSE": [], "MAPE": [], "R²": []})
            continue
        m = metric_re.match(ln)
        if m and cur_ind is not None:
            data[cur_ind][m.group(1)].append(float(m.group(2)))
            found_any = True
    if not found_any:
        print(f"警告: {f} 未解析到任何指标")

# sample counts (from each file's header)
for f in files:
    with open(f, encoding="utf-8") as fh:
        first = fh.read(2000)
    m = re.search(r'训练集样本数:\s*([\d,]+)', first)
    if m:
        meta["train"] = int(m.group(1).replace(",", ""))
    m = re.search(r'测试集样本数:\s*([\d,]+)', first)
    if m:
        meta["test"] = int(m.group(1).replace(",", ""))

if meta:
    print(f"训练集样本数: {meta['train']:,}  测试集样本数: {meta['test']:,}")

# Verify every file has exactly 5 indicators x 4 metrics
for ind, d in data.items():
    for met, vals in d.items():
        if len(vals) != len(files):
            print(f"注意: {ind} / {met} 只有 {len(vals)}/{len(files)} 个值")

def stats(vals):
    n = len(vals)
    mean = statistics.mean(vals)
    med = statistics.median(vals)
    var = statistics.variance(vals) if n > 1 else 0.0
    sd = statistics.stdev(vals) if n > 1 else 0.0
    return n, mean, med, var, sd, min(vals), max(vals)

# Determine unit/decimals per metric
units = {"MAE": "m³/h", "RMSE": "m³/h", "MAPE": "%", "R²": ""}

print("\n" + "=" * 100)
for ind, d in data.items():
    print(f"\n### {ind}  (共 {len(files)} 个文件)")
    print(f"{'指标':<6}{'样本数':>6}{'平均值':>12}{'中位数':>12}{'方差':>14}{'标准差':>12}{'最小值':>12}{'最大值':>12}")
    print("-" * 100)
    for met in ["MAE", "RMSE", "MAPE", "R²"]:
        n, mean, med, var, sd, mn, mx = stats(d[met])
        print(f"{met:<6}{n:>6}{mean:>12.4f}{med:>12.4f}{var:>14.4f}{sd:>12.4f}{mn:>12.4f}{mx:>12.4f}")

print("\n" + "=" * 100)
print("\n全部指标合并统计:")
all_mae = [v for d in data.values() for v in d["MAE"]]
all_rmse = [v for d in data.values() for v in d["RMSE"]]
all_mape = [v for d in data.values() for v in d["MAPE"]]
all_r2 = [v for d in data.values() for v in d["R²"]]
print(f"{'指标':<6}{'样本数':>6}{'平均值':>12}{'中位数':>12}{'方差':>14}{'标准差':>12}{'最小值':>12}{'最大值':>12}")
print("-" * 100)
for met, vals in [("MAE", all_mae), ("RMSE", all_rmse), ("MAPE", all_mape), ("R²", all_r2)]:
    n, mean, med, var, sd, mn, mx = stats(vals)
    print(f"{met:<6}{n:>6}{mean:>12.4f}{med:>12.4f}{var:>14.4f}{sd:>12.4f}{mn:>12.4f}{mx:>12.4f}")

# CSV output (per indicator)
import csv
with open(r"D:\Wuhan_Project\metrics_statistics_per_indicator.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["指标", "模型/目标", "样本数", "均值", "中位数", "方差", "标准差", "最小值", "最大值"])
    for ind, d in data.items():
        for met in ["MAE", "RMSE", "MAPE", "R²"]:
            n, mean, med, var, sd, mn, mx = stats(d[met])
            w.writerow([met, ind, n, round(mean, 4), round(med, 4), round(var, 4), round(sd, 4), round(mn, 4), round(mx, 4)])
print("\n已保存: metrics_statistics_per_indicator.csv")

# ================= 分布直方图 (三个流量 + 一个效率) =================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    target_inds = ["170:1_瞬时流量", "170:2_瞬时流量", "70:3_瞬时流量", "flowtotal(总和)", "总管效率_pct"]
    metrics = ["MAE", "RMSE", "MAPE"]
    units = {"MAE": "m³/h", "RMSE": "m³/h", "MAPE": "%"}

    fig, axes = plt.subplots(len(target_inds), len(metrics),
                             figsize=(4.2 * len(metrics), 3.2 * len(target_inds)))
    for i, ind in enumerate(target_inds):
        for j, met in enumerate(metrics):
            ax = axes[i, j]
            vals = data.get(ind, {}).get(met, [])
            if not vals:
                ax.set_title(f"{ind} · {met} (无数据)", fontsize=9)
                continue
            n, mean, med, var, sd, mn, mx = stats(vals)
            ax.hist(vals, bins=min(20, len(vals)), edgecolor="white", alpha=0.8, color=f"C{j}")
            ax.axvline(mean, color="red", ls="--", lw=1.2, label=f"均值 {mean:.4g}")
            ax.axvline(med, color="blue", ls=":", lw=1.2, label=f"中位数 {med:.4g}")
            ax.set_title(f"{ind} · {met} (n={n}, σ={sd:.4g})", fontsize=9)
            ax.set_xlabel(f"{met} ({units[met]})", fontsize=8)
            if j == 0:
                ax.set_ylabel("频次", fontsize=8)
            ax.legend(fontsize=7)
    fig.tight_layout()
    out_png = r"D:\Wuhan_Project\results\metrics_distribution_histograms.png"
    fig.savefig(out_png, dpi=150)
    print(f"\n已保存直方图: {out_png}")
except ImportError:
    print("\n未安装 matplotlib, 跳过直方图绘制")
