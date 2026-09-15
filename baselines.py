# -*- coding: utf-8 -*-
"""
PIMANN 复现 —— 基线对比实验 (论文 4.4 节, Fig.15 / Table 4)

基线方法 (小样本多元回归):
  baseline 1: RandomForest
  baseline 2: GradientBoosting
  baseline 3: SVR(rbf)
  baseline 4: KNN
流程: 基线仅用前 80% 窗口标签训练 (传统方法无跨域迁移能力,
不使用目标域标定标签与时间特征), 对全部窗口预测并统计指标;
PIMANN 则利用全部源域标签 + 目标域 10 点标定少样本监督。
"""
import json
import os

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import config as C


def window_features(wins: np.ndarray) -> np.ndarray:
    """窗口级手工特征: 均值/方差/RMS/峰峰值 + 各阶差分能量"""
    f = []
    f.append(wins.mean(axis=-1))
    f.append(wins.std(axis=-1))
    f.append(np.sqrt((wins ** 2).mean(axis=-1)))
    f.append(wins.max(axis=-1) - wins.min(axis=-1))
    d = np.diff(wins, axis=-1)
    f.append(np.sqrt((d ** 2).mean(axis=-1)))
    return np.concatenate(f, axis=1)


def get_models():
    return [
        ("baseline 1", RandomForestRegressor(n_estimators=80, random_state=C.SEED)),
        ("baseline 2", GradientBoostingRegressor(n_estimators=80, random_state=C.SEED)),
        ("baseline 3", SVR(kernel="rbf", C=10, gamma="scale")),
        ("baseline 4", KNeighborsRegressor(n_neighbors=5)),
    ]


def main():
    out = C.CSV_DIR
    method_err = {name: {"MAE": [], "MAPE": [], "RMSE": []}
                  for name, _ in get_models()}

    for k in range(C.N_SRC_DATASETS):
        d = np.load(os.path.join(C.DATA_DIR, f"source_ds{k}.npz"))
        wins, Ft, t = d["windows"], d["F"], d["t"]
        X = window_features(wins)
        n_cut = int(len(Ft) * 0.8)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[:n_cut])
        X_all = scaler.transform(X)
        y_tr = Ft[:n_cut]

        preds = {}
        for name, mdl in get_models():
            mdl.fit(X_tr, y_tr)
            yp = mdl.predict(X_all)
            preds[name] = yp
            err = np.abs(yp - Ft)
            method_err[name]["MAE"].append(float(err.mean()))
            method_err[name]["MAPE"].append(
                float((err / np.maximum(Ft, 1.0)).mean() * 100))
            method_err[name]["RMSE"].append(
                float(np.sqrt((err ** 2).mean())))

        np.savez(os.path.join(out, f"baseline_pred_ds{k}.npz"),
                 t_val=t * C.TEST_DAYS[k], F_true=Ft,
                 **{f"pred_{i}": preds[name]
                    for i, (name, _) in enumerate(get_models())})
        print(f"[baseline] dataset{k+1} 完成")

    # Table 4 对比 (百分比误差)
    table = []
    for name, _ in get_models():
        table.append(dict(method=name,
                          MAE_N=float(np.mean(method_err[name]["MAE"])),
                          MAPE_pct=float(np.mean(method_err[name]["MAPE"])),
                          RMSE_N=float(np.mean(method_err[name]["RMSE"]))))
    # 追加 PIMANN 自身指标 (由 train.py 生成)
    p_path = os.path.join(out, "fig11_metrics.json")
    if os.path.exists(p_path):
        pm = json.load(open(p_path))
        table.append(dict(method="PIMANN (ours)",
                          MAE_N=float(np.mean([m["MAE"] for m in pm])),
                          MAPE_pct=float(np.mean([m["MAPE"] for m in pm])),
                          RMSE_N=float(np.mean([m["RMSE"] for m in pm]))))
    with open(os.path.join(out, "table4_comparison.json"), "w") as f:
        json.dump(table, f, indent=2)
    print("[baseline] Table 4 对比结果:")
    for r in table:
        print(f"   {r['method']:<18} MAE={r['MAE_N']:.2f}N  "
              f"MAPE={r['MAPE_pct']:.2f}%  RMSE={r['RMSE_N']:.2f}N")


if __name__ == "__main__":
    main()
