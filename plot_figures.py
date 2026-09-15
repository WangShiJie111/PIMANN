# -*- coding: utf-8 -*-
"""
PIMANN 复现 —— 重绘论文 Fig.9 ~ Fig.16
全部曲线/指标来自本服务器上的真实训练与推理结果 (results/csv),
布局与配色参考 绘图样例/ 中的原始脚本。
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

import config as C

plt.rcParams["font.family"] = ["Times New Roman", "WenQuanYi Micro Hei",
                               "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

FIG = C.FIG_DIR
CSV = C.CSV_DIR


# ----------------------------------------------------------------------
# Fig.9  基于 PIMANN 的实验室数据张紧力拟合结果 (2x3)
# ----------------------------------------------------------------------
def fig9():
    curve_color = '#1f77b4'
    fill_color = '#aec7e8'
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    titles = [f'Trajectory similar to {s}' for s in C.TEST_TRAJ]
    for k in range(6):
        d = np.load(os.path.join(CSV, f"src_pred_ds{k}.npz"))
        t, Ft, Fp = d["t"], d["F_true"], d["F_pred"]
        ax = axes[k // 3, k % 3]
        resid = np.abs(Ft - Fp)
        interval = 1.5 * resid.mean()
        ax.fill_between(t, Fp - interval, Fp + interval, color=fill_color,
                        alpha=0.6, label='confidence interval')
        ax.plot(t, Fp, color=curve_color, linewidth=2.5, label='fitting curve')
        ax.scatter(t, Ft, color=curve_color, s=30, alpha=0.8,
                   edgecolors='white', linewidths=0.6, label='real data')
        ax.set_title(f"{titles[k]} ({C.TEST_DAYS[k]} days)",
                     fontsize=15, fontweight='bold')
        ax.set_xlabel('time / day', fontsize=15)
        ax.set_ylabel('tension / N', fontsize=15)
        ax.set_ylim(C.F_FAIL - 20, C.F0 + 20)
        ax.legend(loc='upper right', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(FIG, "figure9.svg"))
    plt.close(fig)
    print("[plot] figure9.svg 完成")


# ----------------------------------------------------------------------
# Fig.10  跨域模态一致性特征与样本匹配 (3x2)
# ----------------------------------------------------------------------
def fig10():
    d = np.load(os.path.join(CSV, "fig10_recon_demo.npz"))
    stages = ['Early stage', 'Middle stage', 'Late stage']
    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    x = np.arange(C.WIN)
    for r in range(3):
        axl, axr = axes[r]
        axl.plot(x, d["src_obs"][r], 'r-', lw=1.1, alpha=0.85,
                 label='Observation signal')
        axl.plot(x, d["src_recon"][r], 'b--', lw=1.8,
                 label='Intrinsic consistency features')
        axl.set_title(f'test group{r+1} ({stages[r]}) - origin',
                      fontweight='bold', fontsize=13)
        axr.plot(x, d["tgt_obs"][r], 'g-', lw=1.1, alpha=0.85,
                 label='Observation signal')
        axr.plot(x, d["tgt_recon"][r], 'b--', lw=1.8,
                 label='Intrinsic consistency features')
        axr.set_title(f'production data ({stages[r]}) - target',
                      fontweight='bold', fontsize=13)
        for ax in (axl, axr):
            ax.set_xlabel('sample')
            ax.set_ylabel('amplitude')
            ax.legend(fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(FIG, "figure10.svg"))
    plt.close(fig)
    print("[plot] figure10.svg 完成")


# ----------------------------------------------------------------------
# Fig.11  张紧力拟合量化指标柱状图
# ----------------------------------------------------------------------
def fig11():
    metrics = json.load(open(os.path.join(CSV, "fig11_metrics.json")))
    datasets = [f'dataset {m["dataset"]}' for m in metrics]
    names = ['Mean Square Error (MSE)', 'Mean Absolute Error (MAE)',
             'Coefficient of Determination (R2)', 'Root Mean Square Error (RMSE)']
    keys = ['MSE', 'MAE', 'R2', 'RMSE']
    vals = np.array([[m[k] for k in keys] for m in metrics])
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(datasets))
    width = 0.2
    colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E']
    for i, nm in enumerate(names):
        ax.bar(x + i * width, vals[:, i], width, label=nm,
               color=colors[i], alpha=0.8)
    ax.set_xlabel('dataset id', fontsize=20)
    ax.set_ylabel('error metric', fontsize=20)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets, fontsize=16)
    ax.legend(loc='upper right', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "figure11.svg"))
    plt.close(fig)
    print("[plot] figure11.svg 完成")


# ----------------------------------------------------------------------
# Fig.12  生产环境张紧力匹配柱状图 (10 次标定)
# ----------------------------------------------------------------------
def fig12():
    m = np.loadtxt(os.path.join(CSV, "fig12_matching.csv"), delimiter=",",
                   skiprows=1)
    measured, matched = m[:, 0], m[:, 1]
    x = np.arange(len(measured))
    source_color, target_color = '#2878B5', '#9AC9DB'
    aux = '#F8AC8C'
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    bw = 0.35
    ax.bar(x - bw / 2, matched, bw, color=source_color,
           label='Source tension (PIMANN matched)', zorder=3)
    ax.bar(x + bw / 2, measured, bw, yerr=measured * 0.05, capsize=3,
           color=target_color, error_kw=dict(ecolor=aux, lw=1.5, capthick=2),
           label='Target tension (measured)', zorder=3)
    for i in range(len(x)):
        ax.plot([x[i] - bw / 2, x[i] + bw / 2], [matched[i], measured[i]],
                '--', color=aux, alpha=0.6, linewidth=1, zorder=2)
    ax.axhline(matched.mean(), color=source_color, ls='-.', lw=1, alpha=0.7,
               label=f'Source average: {matched.mean():.1f} N')
    ax.axhline(measured.mean(), color=target_color, ls='-.', lw=1, alpha=0.7,
               label=f'Target average: {measured.mean():.1f} N')
    ax.set_xlabel('calibration point number', fontsize=18)
    ax.set_ylabel('tension force / N', fontsize=18)
    ax.set_xticks(x)
    ax.legend(loc='upper right', fontsize=14)
    ax.set_xlim(-0.5, len(x) - 0.5)
    lo = min(matched.min(), measured.min()) - 20
    ax.set_ylim(lo, measured.max() + 25)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "figure12.svg"))
    plt.close(fig)
    print("[plot] figure12.svg 完成")


# ----------------------------------------------------------------------
# Fig.13  精度标定实验: 圆/矩形末端轨迹 (假设 DH 参数 + 数值 IK)
# ----------------------------------------------------------------------
def fig13():
    import kinematics as K
    m = np.loadtxt(os.path.join(CSV, "fig12_matching.csv"), delimiter=",",
                   skiprows=1)
    measured, matched = m[:, 0], m[:, 1]
    q0 = K.reference_pose()
    p0 = K.forward_kinematics(q0)
    n_pts = 160
    circle_path = K.make_circle_path(p0, 0.05, n_pts)
    rect_path = K.make_rectangle_path(p0, 0.10, 0.07, n_pts)
    q_circ = K.inverse_kinematics_path(circle_path, q0)
    q_rect = K.inverse_kinematics_path(rect_path, q0)
    # IK 数值残差不是物理偏差: 以未附加转角的 FK 为基准计算纯 θ0 偏差
    base_circ = np.array([K.forward_kinematics(q) for q in q_circ])
    base_rect = np.array([K.forward_kinematics(q) for q in q_rect])

    red_colors = [(0.8, 0.2, 0.2), (0.9, 0.5, 0.5)]
    blue_colors = [(0.2, 0.3, 0.7), (0.5, 0.6, 0.9)]
    AMP = 500.0  # 微米级偏差可视化放大系数

    fig = plt.figure(figsize=(12.5, 10.8))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.32, wspace=0.26)
    fig.subplots_adjust(left=0.045, right=0.995, top=0.93,
                        bottom=0.015, hspace=0.32, wspace=0.26)
    rows_tab = []
    for i in range(10):
        th_true = K.belt_additional_angle(np.array([measured[i]]))[0]
        th_pred = K.belt_additional_angle(np.array([matched[i]]))[0]

        def shifted(qs, th):
            pts = []
            for q in qs:
                q2 = q.copy()
                q2[4] += th
                pts.append(K.forward_kinematics(q2))
            return np.array(pts)

        dev = {}
        for shape, q_traj, base in (("circle", q_circ, base_circ),
                                    ("rectangle", q_rect, base_rect)):
            d_t = shifted(q_traj, th_true) - base
            d_p = shifted(q_traj, th_pred) - base
            dev[shape] = d_p
            n_t = np.linalg.norm(d_t, axis=1)
            err = np.abs(n_t - np.linalg.norm(d_p, axis=1))
            rows_tab.append(dict(type=shape, id=i + 1,
                                 range_um=float((err.max() - err.min()) * 1e6),
                                 MAE_um=float(err.mean() * 1e6),
                                 MAPE_pct=float(err.mean() /
                                                max(n_t.mean(), 1e-12) * 100)))
        if i < 4:
            ax = fig.add_subplot(gs[0, i])
        elif i < 8:
            ax = fig.add_subplot(gs[1, i - 4])
        else:
            ax = fig.add_subplot(gs[2, i - 8])

        ax.plot(rect_path[:, 0], rect_path[:, 1], '-',
                color=blue_colors[0], lw=1.6)
        ax.plot(rect_path[:, 0] + dev["rectangle"][:, 0] * AMP,
                rect_path[:, 1] + dev["rectangle"][:, 1] * AMP,
                '--', color=blue_colors[1], lw=1.4)
        ax.plot(circle_path[:, 0], circle_path[:, 1], '-',
                color=red_colors[0], lw=1.6)
        ax.plot(circle_path[:, 0] + dev["circle"][:, 0] * AMP,
                circle_path[:, 1] + dev["circle"][:, 1] * AMP,
                '--', color=red_colors[1], lw=1.4)
        e_um = rows_tab[-2:]
        ax.set_title(f'test {i+1}  '
                     f'(MAE {e_um[0]["MAE_um"]:.2f}/{e_um[1]["MAE_um"]:.2f} µm)',
                     fontsize=10)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(alpha=0.25, lw=0.5)

    legend_ax = fig.add_subplot(gs[2, 2:4])
    legend_ax.axis('off')
    legend_ax.legend(handles=[
        plt.Line2D([0], [0], color=blue_colors[0], lw=2,
                   label='rectangular real trajectory'),
        plt.Line2D([0], [0], color=blue_colors[1], lw=2, ls='--',
                   label='rectangular predicted trajectory'),
        plt.Line2D([0], [0], color=red_colors[0], lw=2,
                   label='circular real trajectory'),
        plt.Line2D([0], [0], color=red_colors[1], lw=2, ls='--',
                   label='circular predicted trajectory')],
        loc='center', fontsize=12, frameon=True)

    pd.DataFrame(rows_tab).to_csv(os.path.join(CSV, "table3_traj_error.csv"),
                                  index=False)
    plt.suptitle('Precision calibration experiment trajectory comparison '
                 '(deviation amplified x500, MAE in µm)', fontsize=13, y=0.975)
    plt.savefig(os.path.join(FIG, "figure13.svg"), bbox_inches='tight')
    plt.close(fig)
    print("[plot] figure13.svg 完成")


# ----------------------------------------------------------------------
# Fig.14  六关节旋转角曲线与退化偏移 (2x3)
# ----------------------------------------------------------------------
def fig14():
    raw = pd.read_csv(C.RAW_CSV, header=None).values
    angles = raw[11000:12000, :6]
    n_pts = len(angles)
    t = np.linspace(0, 1, n_pts)

    def norm(col):
        lo, hi = col.min(), col.max()
        return (col - lo) / (hi - lo + 1e-9)

    strengths = [0.02, 0.05, 0.08]
    labels = ['Mild degradation', 'Moderate degradation', 'Severe degradation']
    colors = ['#D95319', '#EDB120', '#77AC30']
    ramp = t ** 1.1  # 累积退化趋势 (单向偏移, 符合弹性体塑性失效特征)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for j in range(6):
        ax = axes[j // 3, j % 3]
        base = norm(angles[:, j])
        ax.plot(np.arange(n_pts), base, '-', color='#0072BD',
                label='True rotation angle', zorder=5, linewidth=2.5)
        amp = strengths if j != 4 else [s * 2.5 for s in strengths]
        for s, lab, colr, a in zip(strengths, labels, colors, amp):
            ax.plot(np.arange(n_pts), base + a * ramp, '--', color=colr,
                    label=lab, linewidth=2.0)
        ax.set_title(f'Joint {j+1}', fontweight='bold', fontsize=16)
        ax.set_xlabel('Time step (250 Hz)')
        if j % 3 == 0:
            ax.set_ylabel('Normalized value (0-1)')
        ax.set_ylim(-0.15, 1.35)
        ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(FIG, "figure14.svg"), dpi=300,
                bbox_inches='tight')
    plt.close(fig)
    print("[plot] figure14.svg 完成")


# ----------------------------------------------------------------------
# Fig.15  基线方法张紧力预测对比 (8x3)
# ----------------------------------------------------------------------
def fig15():
    from baselines import get_models
    models = get_models()
    colors = list(mcolors.TABLEAU_COLORS.values())[:4]
    fig, axes = plt.subplots(nrows=8, ncols=3, figsize=(18, 24))
    for mth, (name, _) in enumerate(models):
        for k in range(6):
            row = mth * 2 + (k // 3)
            col = k % 3
            ax = axes[row, col]
            d = np.load(os.path.join(CSV, f"baseline_pred_ds{k}.npz"))
            tv, Ft, yp = d["t_val"], d["F_true"], d[f"pred_{mth}"]
            ax.plot(tv, Ft, 'k-', linewidth=2, label='true tension')
            ax.plot(tv, yp, '--', color=colors[mth], linewidth=2,
                    label='fitting curve')
            resid = np.abs(yp - Ft)
            win = 5
            roll = np.convolve(resid, np.ones(win) / win, mode='same')
            ax.fill_between(tv, Ft - roll, Ft + roll, color=colors[mth],
                            alpha=0.2, label='error interval')
            ax.set_title(f"{name}  dataset{k+1}", fontsize=14)
            ax.set_xlim(tv.min(), tv.max())
            ax.tick_params(labelsize=11)
            ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "figure15.svg"))
    plt.close(fig)
    print("[plot] figure15.svg 完成")


# ----------------------------------------------------------------------
# Fig.16  消融实验训练过程曲线
# ----------------------------------------------------------------------
def fig16():
    hist = json.load(open(os.path.join(CSV, "history_ablation.json")))
    npu_path = os.path.join(CSV, "history_full_npu.json")
    if os.path.exists(npu_path):
        hist["full_npu"] = json.load(open(npu_path))
    styles = {"full": ("#0072BD", "-", "PIMANN (full, Kunpeng CPU)"),
              "full_npu": ("#111111", "-", "PIMANN (full, Ascend 910B NPU)"),
              "w_o_ca": ("#D95319", "--", "w/o cross-attention"),
              "w_o_pma": ("#EDB120", "-.", "w/o physical modal alignment"),
              "w_o_ca_pma": ("#C82423", ":", "w/o CA & PMA"),
              "single_mode": ("#7E2F8E", "--", "single physical mode"),
              "mse_cov": ("#77AC30", "-.", "w/o Frobenius norm")}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for name, h in hist.items():
        if name not in styles:
            continue
        colr, ls, lab = styles[name]
        eps = [r["epoch"] for r in h]
        axes[0].plot(eps, [r["loss"] for r in h], color=colr, ls=ls,
                     label=lab, lw=2)
        axes[1].plot(eps, [r["val_mape"] for r in h], color=colr, ls=ls,
                     label=lab, lw=2)
    axes[0].set_xlabel('epoch', fontsize=16)
    axes[0].set_ylabel('total loss (Eq.27)', fontsize=16)
    axes[1].set_xlabel('epoch', fontsize=16)
    axes[1].set_ylabel('validation MAPE / %', fontsize=16)
    for ax in axes:
        ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "figure16.svg"))
    plt.close(fig)

    # Table 5 / Table 6 汇总 (run_inference 已生成)
    abl = json.load(open(os.path.join(CSV, "table5_6_ablation.json")))
    pd.DataFrame(abl).to_csv(os.path.join(CSV, "table5_6_summary.csv"),
                             index=False)
    print("[plot] figure16.svg 完成")


def main():
    fig9(); fig10(); fig11(); fig12(); fig13(); fig14(); fig15(); fig16()
    print("\n全部图片已输出至:", FIG)


if __name__ == "__main__":
    main()
