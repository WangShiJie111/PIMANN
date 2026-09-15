# -*- coding: utf-8 -*-
"""昇腾 NPU 上训练完整 PIMANN 并推理 (与 CPU 实验对照)"""
import json
import os

import torch

import config as C
import train


def main():
    dev = train.get_device()
    data = train.load_data()
    model, hist = train.train_variant("full", dev, data, epochs=40,
                                      verbose=True, tag_out="full_npu")
    np = __import__("numpy")
    model.eval()
    src_ref = np.concatenate([ds["windows"] for ds in data["src"]], axis=0)
    tgt = data["tgt"]
    w = tgt["windows"]
    with torch.no_grad():
        chunks = []
        for i in range(0, len(w), 128):
            xt = w[i:i + 128]
            chunks.append(model(train.to_tensor(src_ref[:len(xt)], dev),
                                train.to_tensor(xt, dev))[1].cpu().numpy())
    Fp = np.concatenate(chunks) * train.NORM_F
    ci = tgt["calib_idx"]
    err = np.abs(tgt["calib_F"] - Fp[ci]) / tgt["calib_F"] * 100
    np.savez(os.path.join(C.CSV_DIR, "npu_target_pred.npz"),
             t=tgt["t"] * C.PROD_DAYS, F_true=tgt["F"], F_pred=Fp)
    print(f"[npu] 标定匹配平均相对误差 {err.mean():.2f}% (NPU)")
    print(f"[npu] 最终验证: MAE={hist[-1]['val_mae']:.2f}N "
          f"MAPE={hist[-1]['val_mape']:.2f}% calib={hist[-1]['calib_mae']:.2f}N")


if __name__ == "__main__":
    main()
