# -*- coding: utf-8 -*-
"""
PIMANN 复现 —— 训练与推理主流程 (论文 Table 1 伪代码对应)

训练目标 (Eq.27):
    l_total = ||Cs - Ct||_F^2 + λ·MSE(ŷ, y)_源域标签 + λ_few·MSE(目标域少样本标签)
附带: 多步时序张紧力预测模块 + 本征特征重建损失。

消融配置 (Fig.16 / Table 5 / Table 6):
    w/o CA, w/o PMA, w/o CA&PMA, 单模态, mse_cov(无F范数)

用法 (支持多进程并行训练不同变体, 充分利用鲲鹏多核):
    python train.py --variant full
    python train.py --variant w_o_ca
    python train.py --infer
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import config as C
from model import PIMANN, compute_alignment_loss

# 标签归一化: 张紧力以 F0 归一, 训练输出无量纲
NORM_F = C.F0

# 变体定义: (use_ca, use_pma, align_loss, use_multi_mode)
VARIANTS = {
    "full":        dict(use_ca=True,  use_pma=True,  align_loss="bures",
                        use_multi_mode=True),
    "w_o_ca":      dict(use_ca=False, use_pma=True,  align_loss="bures",
                        use_multi_mode=True),
    "w_o_pma":     dict(use_ca=True,  use_pma=False, align_loss="bures",
                        use_multi_mode=True),
    "w_o_ca_pma":  dict(use_ca=False, use_pma=False, align_loss="bures",
                        use_multi_mode=True),
    "single_mode": dict(use_ca=True,  use_pma=True,  align_loss="bures",
                        use_multi_mode=False),
    "mse_cov":     dict(use_ca=True,  use_pma=True,  align_loss="mse_cov",
                        use_multi_mode=True),
}
VARIANT_ZH = {
    "full": "PIMANN 完整模型", "w_o_ca": "w/o CA", "w_o_pma": "w/o PMA",
    "w_o_ca_pma": "w/o CA&PMA", "single_mode": "单物理模态",
    "mse_cov": "无Frobenius范数",
}


# ----------------------------------------------------------------------
# 设备: 昇腾 NPU 优先, 回退鲲鹏 CPU
# ----------------------------------------------------------------------
def get_device():
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            print("[device] 使用昇腾 NPU:", torch.npu.get_device_name(0))
            return torch.device("npu:0")
    except Exception:
        pass
    torch.set_num_threads(16)  # 鲲鹏 NUMA 结构下 16 线程实测最优
    return torch.device("cpu")


# ----------------------------------------------------------------------
# 数据装载
# ----------------------------------------------------------------------
def load_data():
    data = {"src": [], "tgt": None}
    for k in range(C.N_SRC_DATASETS):
        d = np.load(os.path.join(C.DATA_DIR, f"source_ds{k}.npz"))
        data["src"].append(dict(
            windows=d["windows"], F=d["F"], t=d["t"], ds_id=k))
    d = np.load(os.path.join(C.DATA_DIR, "target.npz"))
    data["tgt"] = dict(windows=d["windows"], F=d["F"], t=d["t"],
                       calib_idx=d["calib_idx"], calib_F=d["calib_F"])
    seq = np.load(os.path.join(C.DATA_DIR, "src_seq.npz"))
    data["seq_X"], data["seq_Y"] = seq["X"], seq["Y"]
    return data


def split_train_val(wins, F, t, ratio=0.8):
    n_cut = int(len(F) * ratio)
    return (wins[:n_cut], F[:n_cut], t[:n_cut]), (wins[n_cut:], F[n_cut:], t[n_cut:])


def to_tensor(x, dev):
    return torch.from_numpy(np.ascontiguousarray(x)).float().to(dev)


def batch_encode(expert, x_seq: torch.Tensor) -> torch.Tensor:
    """批量多步编码: (B, T, C, W) -> CLS 编码序列 (B, T, d)"""
    b, t, c, w = x_seq.shape
    toks, _ = expert(x_seq.reshape(b * t, c, w))
    return toks[:, 0].reshape(b, t, -1)


# ----------------------------------------------------------------------
# 单个变体训练
# ----------------------------------------------------------------------
def train_variant(tag, dev, data, epochs=None, verbose=True, tag_out=None):
    v = VARIANTS[tag]
    out_tag = tag_out or tag
    epochs = epochs or (C.EPOCHS if tag == "full" else C.ABL_EPOCHS)
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    model = PIMANN(C, use_ca=v["use_ca"],
                   use_multi_mode=v["use_multi_mode"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR,
                            weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    tr_w, tr_F, va_w, va_F = [], [], [], []
    for ds in data["src"]:
        # 实验室数据全程带标签: 训练使用全部窗口,
        # 验证集取末 20% (时间外推, 仅用于监控)
        (w1, f1, _), (w2, f2, _) = split_train_val(
            ds["windows"], ds["F"], ds["t"])
        tr_w.append(ds["windows"]), tr_F.append(ds["F"])
        va_w.append(w2), va_F.append(f2)
    tr_w, tr_F = np.concatenate(tr_w), np.concatenate(tr_F)
    va_w, va_F = np.concatenate(va_w), np.concatenate(va_F)
    tgt = data["tgt"]
    calib_idx, calib_F = tgt["calib_idx"], tgt["calib_F"]
    seq_X, seq_Y = data["seq_X"], data["seq_Y"]
    n_tr, n_tg, n_seq = len(tr_F), len(tgt["F"]), len(seq_Y)

    history = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm_s = np.random.permutation(n_tr)
        ep_loss, n_step = 0.0, max(1, n_tr // C.BATCH)
        for b in range(n_step):
            si = perm_s[b * C.BATCH:(b + 1) * C.BATCH]
            if len(si) < C.BATCH:
                si = np.random.choice(n_tr, C.BATCH, replace=False)
            ti = np.random.choice(n_tg, C.BATCH, replace=False)
            qi = np.random.choice(n_seq, C.BATCH, replace=False)

            x_s = to_tensor(tr_w[si], dev)
            y_s = to_tensor(tr_F[si] / NORM_F, dev)
            x_t = to_tensor(tgt["windows"][ti], dev)

            f_src, f_tgt, z, Cs, Ct, z_s, z_t = model(x_s, x_t)

            # ---- Eq.27 总损失 ----
            loss = C.LAMBDA_MSE * F.mse_loss(f_src, y_s)

            # 目标域少样本标签 (10 次标定) 单独前向, 保证两域批尺寸一致
            ci_ref = np.random.choice(n_tr, len(calib_idx), replace=False)
            f_cal = model(to_tensor(tr_w[ci_ref], dev),
                          to_tensor(tgt["windows"][calib_idx], dev))[1]
            loss = loss + C.LAMBDA_FEW * F.mse_loss(
                f_cal, to_tensor(calib_F / NORM_F, dev))
            if v["use_pma"]:
                loss = loss + C.LAMBDA_PMA * compute_alignment_loss(
                    Cs, Ct, v["align_loss"])

            # ---- 多步时序预测模块 (批量编码) ----
            qx = to_tensor(seq_X[qi], dev)
            qy = to_tensor(seq_Y[qi] / NORM_F, dev)
            zs_seq = batch_encode(model.expert_src, qx)
            loss = loss + 0.5 * F.mse_loss(model.msp(zs_seq), qy)

            # ---- 本征一致性特征重建 (Fig.10) ----
            loss = loss + 0.3 * (
                F.mse_loss(model.reconstruct(z_s), x_s[:, 16])
                + F.mse_loss(model.reconstruct(z_t), x_t[:, 16]))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        sched.step()

        # ---- 验证 ----
        model.eval()
        cal_w = tgt["windows"][calib_idx]
        with torch.no_grad():
            n_va = min(len(va_w), 64)
            va_xs = va_w[:n_va]
            va_xt = np.tile(cal_w, (int(np.ceil(n_va / len(cal_w))), 1, 1))[:n_va]
            pv = model(to_tensor(va_xs, dev), to_tensor(va_xt, dev))[0] * NORM_F
            va_Ft = to_tensor(va_F[:n_va], dev)
            mae = float(torch.mean(torch.abs(pv - va_Ft)))
            mape = float(torch.mean(torch.abs(
                (pv - va_Ft) / va_Ft.clamp(min=1.0))) * 100)
            pc = model(to_tensor(tr_w[:len(cal_w)], dev),
                       to_tensor(cal_w, dev))[1] * NORM_F
            mae_cal = float(torch.mean(torch.abs(
                pc - to_tensor(calib_F, dev))))
        history.append(dict(epoch=ep, loss=ep_loss / n_step,
                            val_mae=mae, val_mape=mape, calib_mae=mae_cal))
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  [{tag}] ep {ep:3d} loss={ep_loss/n_step:.4f} "
                  f"val_MAE={mae:.2f}N MAPE={mape:.2f}% "
                  f"calib_MAE={mae_cal:.2f}N")

    print(f"  [{tag}] {VARIANT_ZH[tag]} 训练完成, 用时 {time.time()-t0:.1f}s")
    with open(os.path.join(C.CSV_DIR, f"history_{out_tag}.json"), "w") as f:
        json.dump(history, f)
    if tag == "full":
        torch.save(model.state_dict(),
                   os.path.join(C.CKPT_DIR, f"pimann_{out_tag}.pt"))
    return model, history


# ----------------------------------------------------------------------
# 推理: 生成绘图所需的全部结果
# ----------------------------------------------------------------------
def inference(model, dev, data):
    model.eval()
    out = C.CSV_DIR
    res = {}
    # 交叉注意力要求两域批尺寸一致: 用拼接的全源域窗口作参考流
    src_ref = np.concatenate([ds["windows"] for ds in data["src"]], axis=0)
    tgt_ref = data["tgt"]["windows"]
    with torch.no_grad():
        # ---- 源域 6 组数据集张紧力预测 (Fig.9 / Fig.11) ----
        metrics = []
        for k, ds in enumerate(data["src"]):
            w, Ft, t = ds["windows"], ds["F"], ds["t"]
            chunks = []
            for i in range(0, len(w), 128):
                xs = w[i:i + 128]
                chunks.append(model(to_tensor(xs, dev),
                                    to_tensor(tgt_ref[:len(xs)], dev))[0]
                              .cpu().numpy())
            Fp = np.concatenate(chunks) * NORM_F
            err = Fp - Ft
            mse = float(np.mean(err ** 2))
            mae = float(np.mean(np.abs(err)))
            ss_tot = float(np.sum((Ft - Ft.mean()) ** 2)) + 1e-9
            metrics.append(dict(dataset=k + 1, traj=C.TEST_TRAJ[k],
                                days=C.TEST_DAYS[k], MSE=mse, MAE=mae,
                                RMSE=float(np.sqrt(mse)),
                                R2=1.0 - float(np.sum(err ** 2)) / ss_tot,
                                MAPE=float(np.mean(np.abs(err) /
                                                   np.maximum(Ft, 1.0)) * 100)))
            np.savez(os.path.join(out, f"src_pred_ds{k}.npz"),
                     t=t * C.TEST_DAYS[k], F_true=Ft, F_pred=Fp)
        with open(os.path.join(out, "fig11_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print("[infer] 源域拟合指标 (Fig.9/11):")
        for m in metrics:
            print(f"   dataset{m['dataset']} (轨迹{m['traj']}, {m['days']}天): "
                  f"MAE={m['MAE']:.2f}N RMSE={m['RMSE']:.2f}N "
                  f"R2={m['R2']:.4f}")
        res["src_metrics"] = metrics

        # ---- 目标域张紧力预测与 10 点标定匹配 (Fig.12) ----
        tg = data["tgt"]
        w, Ft, t = tg["windows"], tg["F"], tg["t"]
        chunks = []
        for i in range(0, len(w), 128):
            xt = w[i:i + 128]
            chunks.append(model(to_tensor(src_ref[:len(xt)], dev),
                                to_tensor(xt, dev))[1].cpu().numpy())
        Fp = np.concatenate(chunks) * NORM_F
        ci, cF = tg["calib_idx"], tg["calib_F"]
        match = np.stack([cF, Fp[ci],
                          np.abs(cF - Fp[ci]) / cF * 100], axis=1)
        np.savez(os.path.join(out, "target_pred.npz"),
                 t=t * C.PROD_DAYS, F_true=Ft, F_pred=Fp,
                 calib_idx=ci, calib_F_measured=cF, calib_F_matched=Fp[ci])
        np.savetxt(os.path.join(out, "fig12_matching.csv"), match,
                   delimiter=",", header="measured_N,matched_N,rel_err_%",
                   comments="")
        print(f"[infer] 目标域标定匹配 (Fig.12): 平均相对误差 "
              f"{match[:,2].mean():.2f}%, 最大 {match[:,2].max():.2f}%")

        # ---- 本征一致性特征重建 (Fig.10): 早/中/晚三个退化阶段 ----
        demo = {"pos": [], "src_obs": [], "src_recon": [],
                "tgt_obs": [], "tgt_recon": []}
        wset_s, wset_t = data["src"][0]["windows"], w
        for frac in (0.15, 0.5, 0.85):
            i0 = int(len(wset_t) * frac)
            i1 = min(int(len(wset_s) * frac), len(wset_s) - 1)
            _, _, z, _, _, _, _ = model(to_tensor(wset_s[i1:i1 + 1], dev),
                                        to_tensor(wset_t[i0:i0 + 1], dev))
            rec = model.reconstruct(z)[0].cpu().numpy()
            demo["pos"].append(frac)
            demo["src_obs"].append(wset_s[i1, 16])
            demo["tgt_obs"].append(wset_t[i0, 16])
            demo["src_recon"].append(rec)
            demo["tgt_recon"].append(rec)
        np.savez(os.path.join(out, "fig10_recon_demo.npz"),
                 pos=np.array(demo["pos"]),
                 src_obs=np.stack(demo["src_obs"]),
                 src_recon=np.stack(demo["src_recon"]),
                 tgt_obs=np.stack(demo["tgt_obs"]),
                 tgt_recon=np.stack(demo["tgt_recon"]))

        # ---- 多步退化曲线外推 + RUL (Eq.10) ----
        from kinematics import health_indicator, predict_rul
        msp_rows = []
        starts = np.linspace(0, len(w) - C.SEQ_T - 2, 8).astype(int)
        for start in starts:
            xs = to_tensor(w[start:start + C.SEQ_T].reshape(
                1, C.SEQ_T, C.N_CH, C.WIN), dev)
            z_seq = batch_encode(model.expert_tgt, xs)
            pred = model.msp(z_seq)[0].cpu().numpy() * NORM_F
            msp_rows.append(dict(start_day=float(t[start] * C.PROD_DAYS),
                                 pred_F=pred.tolist()))
        with open(os.path.join(out, "msp_rollout.json"), "w") as f:
            json.dump(msp_rows, f, indent=1)
        hi_now = health_indicator(float(Fp[ci[-1]]))
        rul = predict_rul(hi_now)
        rul_info = dict(HI_now=float(hi_now), RUL_days=float(rul),
                        F_now=float(Fp[ci[-1]]))
        with open(os.path.join(out, "rul_summary.json"), "w") as f:
            json.dump(rul_info, f, indent=2)
        print(f"[infer] 当前 HI={hi_now:.3f}, 预测 RUL≈{rul:.1f} 天 "
              f"(Eq.10, F_now={Fp[ci[-1]]:.1f}N)")
        res["rul"] = rul_info
    return res


def run_inference():
    dev = get_device()
    data = load_data()
    model = PIMANN(C).to(dev)
    model.load_state_dict(torch.load(os.path.join(C.CKPT_DIR, "pimann_full.pt"),
                                     map_location=dev, weights_only=True))
    inference(model, dev, data)

    # 汇总各变体历史 + 消融表
    all_hist = {}
    for tag in VARIANTS:
        p = os.path.join(C.CSV_DIR, f"history_{tag}.json")
        if os.path.exists(p):
            all_hist[tag] = json.load(open(p))
    with open(os.path.join(C.CSV_DIR, "history_ablation.json"), "w") as f:
        json.dump(all_hist, f)
    rows = []
    for tag, h in all_hist.items():
        rows.append(dict(variant=VARIANT_ZH[tag], tag=tag,
                         final_MAPE=h[-1]["val_mape"],
                         final_MAE_N=h[-1]["val_mae"],
                         calib_MAE_N=h[-1]["calib_mae"],
                         best_epoch=int(np.argmin(
                             [r["val_mape"] for r in h]))))
    with open(os.path.join(C.CSV_DIR, "table5_6_ablation.json"), "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print("[infer] 消融汇总已写入 table5_6_ablation.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full", choices=list(VARIANTS))
    ap.add_argument("--infer", action="store_true")
    args = ap.parse_args()
    if args.infer:
        run_inference()
        return
    dev = get_device()
    data = load_data()
    train_variant(args.variant, dev, data)


if __name__ == "__main__":
    main()
