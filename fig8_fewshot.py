# -*- coding: utf-8 -*-
"""
论文 Fig.8 复现: 不同参数组合下的模型训练效果
(Model training effectiveness under different parameter combinations)

论文 4.3 节: 在不同小样本比例与超参数组合下反复训练 PIMANN,
框架均能稳定收敛并保持较好的精度下限。

实验设计 (真实训练, 非模拟数据):
  * 样本比例 = 源域标签数 : 目标域标定标签数(固定10),
    源域标签取 {10, 30, 100, 全部} -> 1:1 / 3:1 / 10:1 / ~72:1
  * 每个比例 10 次重复试验 (不同随机种子)
  * 每次试验 4 组超参数 (LR/Batch), 早停记录收敛轮数 (训练成本)
  * 精度 = 1 - MAPE/100 (源域末20%时间外推验证集)

用法:
  python fig8_fewshot.py            # 训练 + 绘图 (建议 NPU 环境)
  python fig8_fewshot.py --plot-only  # 仅用已有结果重绘
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import config as C
import train as T
from model import PIMANN

OUT_JSON = os.path.join(C.CSV_DIR, "fig8_fewshot.json")

RATIOS = ["1:1", "3:1", "10:1", "72:1(all)"]
N_SRC_LAB = [10, 30, 100, None]          # None = 全部源域标签
N_CALIB = 10                             # 目标域标定标签固定 10 个
N_TRIAL = 40
MAX_EP = 30
PATIENCE = 5
MAX_STEP = 8                   # 每 epoch 批数上限, 控制大标签集单轮成本
HYPER = [
    ("LR=3e-4, Batch=32", 3e-4, 32),
    ("LR=1e-3, Batch=64", 1e-3, 64),
    ("LR=1e-3, Batch=16", 1e-3, 16),
    ("LR=5e-5, Batch=32", 5e-5, 32),
]


# ----------------------------------------------------------------------
def run_one(dev, data, n_src_lab, lr, batch, seed):
    """单次训练: 返回 (初始精度, 最终精度, 实际训练轮数)"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    # 源域标签子集 (小样本比例) + 验证集 (各数据集末20%)
    all_w = np.concatenate([ds["windows"] for ds in data["src"]])
    all_F = np.concatenate([ds["F"] for ds in data["src"]])
    n_all = len(all_F)
    if n_src_lab is None:
        lab_idx = np.arange(n_all)
    else:
        lab_idx = rng.choice(n_all, n_src_lab, replace=False)
    va_w = np.concatenate([T.split_train_val(ds["windows"], ds["F"],
                                             ds["t"])[1][0]
                           for ds in data["src"]])
    va_F = np.concatenate([T.split_train_val(ds["windows"], ds["F"],
                                             ds["t"])[1][1]
                           for ds in data["src"]])

    tgt = data["tgt"]
    cal_w, cal_F = tgt["windows"][tgt["calib_idx"]], tgt["calib_F"]

    model = PIMANN(C).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=C.WEIGHT_DECAY)

    def val_mape():
        model.eval()
        with torch.no_grad():
            n_va = min(len(va_w), 64)
            xt = np.tile(cal_w, (int(np.ceil(n_va / len(cal_w))), 1, 1))[:n_va]
            pv = model(T.to_tensor(va_w[:n_va], dev),
                       T.to_tensor(xt, dev))[0] * T.NORM_F
            return float(torch.mean(torch.abs(
                (pv - T.to_tensor(va_F[:n_va], dev)) /
                T.to_tensor(va_F[:n_va], dev).clamp(min=1.0))) * 100)

    def acc(mape):
        return float(np.clip(1.0 - mape / 100.0, 0.0, 1.0))

    best, bad, epochs_used, acc_init = 1e9, 0, 0, None
    n_step = min(max(1, len(lab_idx) // batch), MAX_STEP)
    for ep in range(MAX_EP):
        model.train()
        perm = rng.permutation(len(lab_idx))
        for b in range(n_step):
            si = perm[b * batch:(b + 1) * batch]
            if len(si) < batch:   # 交叉注意力要求两域批尺寸一致
                si = rng.choice(len(lab_idx), batch, replace=True)
            ti = rng.choice(len(tgt["F"]), batch, replace=True)
            f_s, f_t, _, Cs, Ct, _, _ = model(
                T.to_tensor(all_w[lab_idx[si]], dev),
                T.to_tensor(tgt["windows"][ti], dev))
            loss = F.mse_loss(f_s,
                              T.to_tensor(all_F[lab_idx[si]] / T.NORM_F, dev))
            # 目标域 10 点标定少样本监督 (单独前向保证批尺寸一致)
            ci_ref = rng.choice(len(lab_idx), N_CALIB, replace=False)
            f_cal = model(T.to_tensor(all_w[lab_idx[ci_ref]], dev),
                          T.to_tensor(cal_w, dev))[1]
            loss = loss + C.LAMBDA_FEW * F.mse_loss(
                f_cal, T.to_tensor(cal_F / T.NORM_F, dev))
            loss = loss + C.LAMBDA_PMA * T.compute_alignment_loss(Cs, Ct, "bures")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        epochs_used = ep + 1
        m = val_mape()
        if ep == 0:
            acc_init = acc(m)
        if m < best - 0.15:
            best, bad = m, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return acc_init, acc(best), epochs_used


# ----------------------------------------------------------------------
def run_experiments():
    dev = T.get_device()
    data = T.load_data()
    res = {"ratios": RATIOS, "hyper": [h[0] for h in HYPER],
           "n_trial": N_TRIAL, "trials": {}}
    t0 = time.time()
    for ri, (ratio, n_lab) in enumerate(zip(RATIOS, N_SRC_LAB)):
        res["trials"][ratio] = []
        for tr in range(N_TRIAL):
            seed = 1000 + ri * 100 + tr
            row = dict(acc_init=[], acc_final=[], epochs=[])
            for _, lr, batch in HYPER:
                a0, a1, ep = run_one(dev, data, n_lab, lr, batch, seed)
                row["acc_init"].append(a0)
                row["acc_final"].append(a1)
                row["epochs"].append(ep)
            res["trials"][ratio].append(row)
            print(f"[fig8] ratio={ratio} trial={tr} "
                  f"acc0={row['acc_init'][0]:.2f} "
                  f"accF={row['acc_final'][0]:.2f} "
                  f"ep={row['epochs'][0]}  ({time.time()-t0:.0f}s)")
    with open(OUT_JSON, "w") as f:
        json.dump(res, f)
    print("[fig8] 实验完成 ->", OUT_JSON)
    return res


# ----------------------------------------------------------------------
def plot_fig8(res=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    if res is None:
        res = json.load(open(OUT_JSON))
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "axes.grid": False})

    fig = plt.figure(figsize=(15.5, 7.2))
    gs = gridspec.GridSpec(2, 4, figure=fig)
    fig.subplots_adjust(left=0.045, right=0.995, top=0.90,
                        bottom=0.075, hspace=0.42, wspace=0.24)
    styles = ["-", "--", "-.", ":"]
    for idx, ratio in enumerate(res["ratios"]):
        trials = res["trials"][ratio]
        x = np.arange(len(trials))
        a0 = np.array([t["acc_init"][0] for t in trials])
        a1 = np.array([t["acc_final"][0] for t in trials])

        ax_h = fig.add_subplot(gs[idx // 2, (idx % 2) * 2])
        w = 0.38
        ax_h.bar(x - w / 2, a0, w, color="#ffcccc", label="Initial Accuracy")
        ax_h.bar(x + w / 2, a1, w, color="#ccebff", label="Final Accuracy")
        ax_h.axhline(a0.mean(), color="blue", ls="-", lw=1.6,
                     label=f"Initial Mean: {a0.mean():.2f}")
        ax_h.axhline(a1.mean(), color="blue", ls="--", lw=1.6,
                     label=f"Final Mean: {a1.mean():.2f}")
        ax_h.set_title(f"Sample Ratio {ratio}\nAccuracy Distribution", pad=6)
        ax_h.set_xlabel("Trial Index")
        ax_h.set_ylabel("Accuracy")
        ax_h.set_ylim(0.0, 1.05)
        ax_h.legend(loc="lower right", frameon=True)

        ax_l = fig.add_subplot(gs[idx // 2, (idx % 2) * 2 + 1])
        for j, name in enumerate(res["hyper"]):
            ej = np.array([t["epochs"][j] for t in trials])
            ax_l.plot(x, ej, styles[j], label=name, alpha=0.85, lw=1.4)
        ax_l.set_title(f"Sample Ratio {ratio}\nTraining Epochs", pad=6)
        ax_l.set_xlabel("Trial Index")
        ax_l.set_ylabel("Training Epochs (Cost)")
        ax_l.set_ylim(0, MAX_EP + 3)
        ax_l.legend(loc="upper right", frameon=True)

    fig.suptitle("Model training effectiveness under different parameter "
                 "combinations (PIMANN, few-shot ratios)",
                 fontsize=12, y=0.975)
    out_svg = os.path.join(C.FIG_DIR, "figure8.svg")
    plt.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print("[plot] figure8.svg 完成")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    res = None if not args.plot_only else json.load(open(OUT_JSON))
    if not args.plot_only:
        res = run_experiments()
    plot_fig8(res)


if __name__ == "__main__":
    main()
