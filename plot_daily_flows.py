# -*- coding: utf-8 -*-
"""按原始数据画三个分支流量 + 总流量的每日变化图, 一天一张。

- 数据: new_data/merged_minute_all.csv (秒级)
- 分支: 170:1 / 170:2 / 70:3 (与训练一致, 不含 70:7)
- 总流量: 三管求和 (与 train_lstm.py 语义一致: 三管全部有效才求和)
- 垃圾值屏蔽: 与训练清洗同款物理界限, 流量 < 0 或 > 10000 → NaN
  (原始 170:1 含 -6e23 级垃圾读数, 不屏蔽会毁掉纵轴量程)
- 展示分辨率: 重采样到 1 分钟均值 (秒级 86400 点/天过重, 1min 1440 点足够看趋势)
- 输出: results_daily_flow/YYYY-MM-DD.png (每图两行: 上行三分支, 下行总流量)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

DATA = r"D:\Wuhan_Project\new_data\merged_minute_all.csv"
OUT_DIR = r"D:\Wuhan_Project\results_daily_flow"

FLOW_COLS = ["170:1_瞬时流量", "170:2_瞬时流量", "70:3_瞬时流量"]
BRANCH_STYLE = {
    "170:1_瞬时流量": dict(color="#2c3e50", label="170:1"),
    "170:2_瞬时流量": dict(color="#16a085", label="170:2"),
    "70:3_瞬时流量":  dict(color="#8e44ad", label="70:3"),
}
TOTAL_STYLE = dict(color="#e74c3c", label="Total")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA, encoding="utf-8-sig")
    ts_col = "F_DateTime" if "F_DateTime" in df.columns else "时间"
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col).sort_index()

    missing = [c for c in FLOW_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少流量列: {missing}")

    # ── 垃圾值屏蔽 (与训练物理界限一致) ──
    base = df[FLOW_COLS].astype(float)
    for c in FLOW_COLS:
        base[c] = base[c].where((base[c] >= 0.0) & (base[c] <= 10000.0))

    # ── 总流量: 三管全部有效才求和 (min_count=3) ──
    base["Total_Flow"] = base[FLOW_COLS].sum(axis=1, min_count=len(FLOW_COLS))

    # ── 展示用: 1 分钟均值重采样 ──
    disp = base.resample("1min").mean()

    days = disp.index.normalize().unique()
    print(f"共 {len(days)} 天: {days[0].date()} ~ {days[-1].date()}")

    for i, day in enumerate(days):
        d = disp[disp.index.normalize() == day]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                                       gridspec_kw={"height_ratios": [2, 1]})

        for c in FLOW_COLS:
            st = BRANCH_STYLE[c]
            ax1.plot(d.index, d[c], color=st["color"], linewidth=0.8,
                     label=f'{st["label"]} (均值 {d[c].mean():.0f})')
        ax1.set_ylabel("分支流量 (m3/h)")
        ax1.grid(alpha=0.3)
        ax1.legend(fontsize=9, loc="upper right")
        ax1.set_title(f"{day.date()}  三分支流量与总流量", fontsize=14, fontweight="bold")

        ax2.plot(d.index, d["Total_Flow"], color=TOTAL_STYLE["color"], linewidth=0.8,
                 label=f'Total (均值 {d["Total_Flow"].mean():.0f})')
        ax2.set_ylabel("总流量 (m3/h)")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=9, loc="upper right")

        save_path = os.path.join(OUT_DIR, f"{day.date()}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        if (i + 1) % 5 == 0 or i == len(days) - 1:
            print(f"  已保存 {i + 1}/{len(days)}: {os.path.basename(save_path)}")

    print(f"完成! 共 {len(days)} 张图 → {OUT_DIR}")


if __name__ == "__main__":
    main()
