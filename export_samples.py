# -*- coding: utf-8 -*-
"""
导出少量测试数据样本 (CSV, 便于查看/交接)

从 results/data/*.npz 中每个数据集均匀抽取 6 个窗口 (覆盖早/中/晚退化阶段),
每窗口 256 个采样点, 连同张紧力标签写出为 CSV。
通道顺序与 data.csv 一致: 6 旋转角 + 6 速度 + 6 电流 (标准化后数值)。
输出目录: /home/sjwang/PIMANN/数据样例/
"""
import os

import numpy as np
import pandas as pd

import config as C

OUT_DIR = os.path.join(os.path.dirname(C.DATA_DIR), "..", "..", "数据样例")
OUT_DIR = os.path.abspath(OUT_DIR)
N_WIN = 6                      # 每个数据集抽取窗口数
COLS = ([f"angle_{j}" for j in range(1, 7)] +
        [f"vel_{j}" for j in range(1, 7)] +
        [f"cur_{j}" for j in range(1, 7)])


def export_one(name: str, wins: np.ndarray, F: np.ndarray, t: np.ndarray,
               days: float):
    idx = np.linspace(0, len(F) - 1, N_WIN).astype(int)
    rows = []
    for wi in idx:
        w = wins[wi]                       # (18, 256)
        for s in range(w.shape[1]):
            row = dict(dataset=name, window_id=int(wi),
                       sample_in_window=s,
                       t_day=float(t[wi] * days + s / (250.0 * 4)),
                       tension_N=float(F[wi]))
            for c in range(18):
                row[COLS[c]] = float(w[c, s])
            rows.append(row)
    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, f"{name}_sample.csv")
    df.to_csv(path, index=False, float_format="%.5f")
    print(f"[sample] {path}  ({len(df)} 行, {N_WIN} 个窗口)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for k in range(C.N_SRC_DATASETS):
        d = np.load(os.path.join(C.DATA_DIR, f"source_ds{k}.npz"))
        export_one(f"source_test{k+1}", d["windows"], d["F"], d["t"],
                   C.TEST_DAYS[k])
    d = np.load(os.path.join(C.DATA_DIR, "target.npz"))
    export_one("target_production", d["windows"], d["F"], d["t"], C.PROD_DAYS)

    with open(os.path.join(OUT_DIR, "README.md"), "w") as f:
        f.write(
            "# 测试数据样本说明\n\n"
            "本目录为 PIMANN 复现所用**测试数据的少量样本导出** "
            "(每数据集 6 个窗口 × 256 采样点)，便于查看与交接。\n"
            "完整预处理数据见 `reproduce/results/data/*.npz`。\n\n"
            "## 列说明\n"
            "- `dataset/window_id/sample_in_window`: 数据集 / 窗口 / 窗内采样序号\n"
            "- `t_day`: 换算为天的时间戳\n"
            "- `angle_1..6 / vel_1..6 / cur_1..6`: 与 data.csv 同序的 18 通道\n"
            "  (6 关节旋转角 + 6 速度 + 6 电流)，标准化后数值，即模型输入\n"
            "- `tension_N`: 该窗口的同步带张紧力标签 (N)，按测试协议同步记录\n\n"
            "## 预处理协议\n"
            "测试数据经滑窗切分 (窗 256 点 / 步长 64)、逐通道零均值单位方差标准化后\n"
            "作为模型输入；源域为 6 组实验室加速退化测试工况 (论文 Table 2)，目标域为\n"
            "生产线运行数据且仅保留 10 个精度标定标签 (少样本设置)。\n")
    print("[sample] 导出完成 ->", OUT_DIR)


if __name__ == "__main__":
    main()
