# ============================================================================
# tuning_driver.py — Transformer 调参实验驱动: 运行一次实验并记录结果
#
# 用法:
#   python tuning_driver.py '<overrides JSON>' <exp_id>
# 例:
#   python tuning_driver.py '{"d_model": 128, "num_layers": 4, "label": "..."}' E01
#
# 原理: 导入 train_transformer.py 模块, 覆盖其 BASE_CONFIG 后调用 main()。
# 运行后解析结果目录的 metrics.txt / train_history.csv,
# 追加一行到 transformer_tuning_log.csv (每个实验一行, 永不覆盖)。
# ============================================================================
import json
import os
import re
import sys
import time
import traceback
import importlib.util

import pandas as pd

LOG_CSV = r"D:\Wuhan_Project\transformer_tuning_log.csv"


def parse_metrics(path):
    """从 metrics.txt 解析 Train/Test 的 Loss/MAE/RMSE/MAPE。"""
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"(Train|Test): Loss=([0-9.]+), MAE=([0-9.]+), RMSE=([0-9.]+), MAPE=([0-9.]+)%", line)
            if m:
                name, loss, mae, rmse, mape = m.groups()
                d[f"{name.lower()}_loss"] = float(loss)
                d[f"{name.lower()}_mae"] = float(mae)
                d[f"{name.lower()}_rmse"] = float(rmse)
                d[f"{name.lower()}_mape"] = float(mape)
    return d


def main():
    overrides = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    exp_id = sys.argv[2] if len(sys.argv) > 2 else "E??"

    spec = importlib.util.spec_from_file_location("train_transformer", "train_transformer.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for k, v in overrides.items():
        m.BASE_CONFIG[k] = v

    label = m.BASE_CONFIG["label"]
    result_dir = os.path.join(m.BASE_CONFIG["base_result_dir"], label)

    row = {"exp_id": exp_id, "label": label,
           "overrides": json.dumps(overrides, ensure_ascii=False),
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "OK"}
    t0 = time.time()
    try:
        m.main()
        row["runtime_min"] = round((time.time() - t0) / 60, 2)
        row.update(parse_metrics(os.path.join(result_dir, "metrics.txt")))
        hist_path = os.path.join(result_dir, "train_history.csv")
        if os.path.exists(hist_path):
            h = pd.read_csv(hist_path)
            row["n_epochs"] = int(len(h))
            row["best_epoch"] = int(h["test_loss"].idxmin()) + 1
    except Exception:
        row["status"] = "FAILED"
        row["runtime_min"] = round((time.time() - t0) / 60, 2)
        traceback.print_exc()

    cols = ["exp_id", "label", "overrides", "timestamp", "status", "n_epochs", "best_epoch",
            "runtime_min", "train_loss", "train_mae", "train_rmse", "train_mape",
            "test_loss", "test_mae", "test_rmse", "test_mape"]
    df = pd.DataFrame([{c: row.get(c) for c in cols}])
    if os.path.exists(LOG_CSV):
        df = pd.concat([pd.read_csv(LOG_CSV), df], ignore_index=True)
    df.to_csv(LOG_CSV, index=False, encoding="utf-8-sig")

    print(f"\n[record] 已追加到 {LOG_CSV}")
    print(df.tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
