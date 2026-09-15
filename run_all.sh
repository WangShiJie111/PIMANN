#!/bin/bash
# PIMANN 复现一键流程 (华为鲲鹏 920 / openEuler 22.03)
set -e
source ~/gggg_h/miniconda3/etc/profile.d/conda.sh
conda activate pimann
cd "$(dirname "$0")"
mkdir -p logs

echo "===== [1/5] 检查预处理域数据 ====="
python - <<'EOF'
import os, sys
need = ["results/data/target.npz"] + [f"results/data/source_ds{k}.npz" for k in range(6)]
miss = [p for p in need if not os.path.exists(p)]
if miss:
    sys.exit(f"缺少预处理数据: {miss}, 请先运行内部数据预处理脚本生成 results/data/*.npz")
print("预处理域数据齐备:", len(need), "份")
EOF

echo "===== [2/5] 并行训练 PIMANN 及消融变体 (鲲鹏多核) ====="
for v in full w_o_ca w_o_pma w_o_ca_pma single_mode mse_cov; do
    python train.py --variant $v > logs/train_$v.log 2>&1 &
done
wait
tail -2 logs/train_full.log

echo "===== [3/5] 推理: 张紧力预测/标定匹配/退化曲线/RUL ====="
python train.py --infer

echo "===== [4/5] 基线对比实验 ====="
python baselines.py

echo "===== [5/5] 重绘论文 Fig.9 ~ Fig.16 ====="
python plot_figures.py

echo "完成! 图片: results/figures  指标: results/csv"
