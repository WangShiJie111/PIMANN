#!/bin/bash
# 守候: 等待 6 个训练变体全部结束, 然后自动执行推理/基线/绘图
source ~/gggg_h/miniconda3/etc/profile.d/conda.sh
conda activate pimann
cd /home/sjwang/PIMANN/reproduce

echo "[wait] 开始守候 $(date '+%H:%M:%S')"
while true; do
    # 还有 train.py 进程在跑就继续等
    if ! pgrep -f "train.py --variant" > /dev/null; then
        break
    fi
    sleep 30
done
echo "[wait] 训练进程已全部结束 $(date '+%H:%M:%S')"

# 校验 6 份 history 是否齐全
miss=""
for v in full w_o_ca w_o_pma w_o_ca_pma single_mode mse_cov; do
    [ -f results/csv/history_$v.json ] || miss="$miss $v"
done
if [ -n "$miss" ]; then
    echo "[wait] 缺少 history:$miss —— 中止后续流程"
    exit 1
fi
[ -f results/ckpt/pimann_full.pt ] || { echo "[wait] 缺少 pimann_full.pt"; exit 1; }
echo "[wait] 6 份 history 与 ckpt 齐全, 开始推理"

echo "===== 推理 ====="
python train.py --infer || { echo "[wait] infer 失败"; exit 1; }
echo "===== 基线对比 ====="
python baselines.py || { echo "[wait] baselines 失败"; exit 1; }
echo "===== 绘图 ====="
python plot_figures.py || { echo "[wait] plot 失败"; exit 1; }
echo "[wait] 全部完成 $(date '+%H:%M:%S')"
