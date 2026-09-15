# PIMANN — Physics-Informed Modal Alignment Neural Network

> 论文复现：*Welding robot digital twin and precision degradation prediction
> based on physics-informed modal alignment neural network*
> （Journal of Manufacturing Systems, 2026）
>
> 运行平台：**华为鲲鹏 920（aarch64, 192 核）CPU** 与 **昇腾 910B3 NPU**
> （torch_npu + CANN 9.0）。纯 PyTorch 实现，无 CUDA 依赖，同一份代码双后端运行。

**亮点**

- 完整复现论文方法论：双域专家 Transformer、物理模态对齐（Bures-PMA，含可微
  矩阵开平方算子）、跨域交叉注意力融合、少样本标定监督、多步退化外推与 RUL；
- 源域 6 组加速退化测试拟合 R² 0.72–0.98；目标域 10 点标定匹配平均误差 2.13%；
  张紧力预测 RMSE 12.51 N，优于 RF/GBR/SVR/KNN 全部基线；
- 消融实验与 640 次小样本×超参数重复训练验证模块有效性；
- 重绘论文 Fig.8–Fig.16 全部图件（`results/figures/`，svg + png）；
- 昇腾 910B3 单卡训练 40 epoch 仅 72 s（鲲鹏 CPU 16 线程 4241 s，加速比 ≈59×）。

---

## 1. 方法概览与论文对应

| 论文内容 | 公式 | 代码位置 |
| --- | --- | --- |
| 物理模态正交基分解（傅里叶/贝塞尔/勒让德）| Eq.14 | `modal_ops.py`: `FourierModal` / `BesselModal` / `LegendreModal` |
| 模态协方差矩阵 | Eq.15/16 | `modal_ops.py`: `modal_covariance` |
| 模态对齐损失（Frobenius / Bures，可微矩阵开平方）| Eq.17 | `modal_ops.py`: `pma_loss`, `_sqrtm_spd` |
| ADAP 自适应前馈算子 | Eq.21 | `modal_ops.py`: `ADAP` |
| 多头自注意力专家（源/目标双域）| Eq.20–22 | `model.py`: `ExpertTransformer`, `MHSA` |
| 跨域交叉注意力交换与融合 | Eq.23/24 | `model.py`: `MHCA`, `_cross_fuse` |
| 融合编码 Z = ADAP(CLS(Xs)) + ADAP(GAP(Xt)) | Eq.26 | `model.py`: `PIMANN.forward` |
| 总损失（模态对齐 + 少样本标签监督）| Eq.27 | `train.py` 训练循环 |
| 张紧力附加转角 θ0=(F0−F)L/(Er) | Eq.5 | `kinematics.py`: `belt_additional_angle` |
| DH 正运动学与末端轨迹偏差 | Eq.6–8/11/12 | `kinematics.py`: `forward_kinematics`, `trajectory_deviation` |
| 精度退化度与 RUL 预测 | Eq.9/10 | `kinematics.py`: `weighted_precision_index`, `predict_rul` |
| 多步时序张紧力预测模块 | 3.4 节 | `model.py`: `MultiStepPredictor` |

网络数据流：源/目标域 18 通道滑窗信号 → 双域专家 Transformer → 三组正交物理
模态基投影 + 协方差 → Bures-PMA 对齐损失；两域特征经交叉注意力交换后融合为
Z，送入张紧力预测头与多步预测头；目标域仅 10 个精度标定标签参与监督。

---

## 2. 仓库结构

```
.
├── config.py          # 全局超参 / 物理参数 / DH 参数 / 路径
├── modal_ops.py       # 自定义算子: 正交模态基 / 模态协方差 / Bures-PMA / ADAP
├── model.py           # PIMANN 网络: 双专家 + 交叉注意力 + 融合编码 + 预测头
├── kinematics.py      # DH 正运动学 / 附加转角 / 轨迹偏差 / HI / RUL
├── train.py           # 训练与推理入口 (CPU/NPU 自动选择, 6 个消融变体)
├── baselines.py       # 基线对比 RF/GBR/SVR/KNN (论文 4.4 节, Table 4)
├── fig8_fewshot.py    # 论文 Fig.8: 小样本比例 × 超参数 反复训练实验
├── plot_figures.py    # 重绘论文 Fig.9 ~ Fig.16
├── export_samples.py  # 导出少量测试数据样本 (CSV) 便于查看
├── run_all.sh         # 一键全流程 (CPU)
└── results/
    ├── data/          # 预处理后的域数据 (.npz, 随仓库提供)
    ├── ckpt/          # 训练权重 (pimann_full.pt / pimann_full_npu.pt)
    ├── csv/           # 指标与预测结果 (Table 3–6 对应数据)
    └── figures/       # figure8~16 (.svg, png/ 子目录为位图)
```

---

## 3. 安装

### 3.1 CPU 环境（训练 / 推理 / 绘图，必需）
```bash
conda create -n pimann python=3.10 -y && conda activate pimann
pip install torch numpy pandas matplotlib scikit-learn
pip install cairosvg          # 可选: SVG 转 PNG
```

### 3.2 昇腾 NPU 环境（可选，训练加速 ≈59×）
```bash
conda create -n pimann_npu python=3.10 -y && conda activate pimann_npu
# 安装与 CANN 配套的 torch / torch_npu (本仓库验证组合: torch 2.12.0 + torch_npu 2.12.0)
pip install decorator attrs psutil tornado ml-dtypes pyyaml   # torch_npu 运行时依赖
source ~/Ascend/cann-9.0.0-beta.1/set_env.sh                   # 每次使用前加载 CANN
```

> 注意：`torch`、`torch_npu`、CANN 三者版本需按官方兼容表配套；
> 脚本启动时会打印实际后端（`[device] 使用昇腾 NPU: Ascend910B3`），
> 若环境缺 torch_npu 会回退 CPU 并明确提示。

---

## 4. 数据

- **原始测试数据**：机器人六轴记录 `data.csv`（6 关节旋转角 + 6 速度 + 6 电流，
  18 通道），放置于仓库上级目录（`../data.csv`）；
- **预处理域数据**：`results/data/*.npz` 已随仓库提供，可直接训练：
  - `source_ds0~5.npz`：6 组实验室加速退化测试工况（论文 Table 2），键为
    `windows` (N,18,256) / `F` (N,) 张紧力标签 / `t` (N,) 归一化时间；
  - `target.npz`：生产线数据，额外含 `calib_idx` / `calib_F`（10 次精度标定的
    少样本标签）；
- 预处理协议：滑窗 256 点 / 步长 64，逐通道零均值单位方差标准化；
- `python export_samples.py` 可导出少量样本 CSV 便于查看数据格式。

---

## 5. 快速开始

```bash
conda activate pimann
bash run_all.sh        # 校验数据 → 6 变体并行训练 → 推理 → 基线 → 绘图
```

分步执行：

```bash
python train.py --variant full     # 训练完整 PIMANN (NPU 环境内运行则自动用 NPU)
python train.py --infer            # 推理: 拟合指标/标定匹配/退化外推/RUL/消融汇总
python baselines.py                # 基线对比 (Table 4 / Fig.15)
python plot_figures.py             # 重绘 Fig.9 ~ Fig.16
python fig8_fewshot.py             # Fig.8 小样本反复训练实验 (建议 NPU, 约 40 分钟)
```

昇腾 NPU 训练：
```bash
conda activate pimann_npu && source ~/Ascend/cann-9.0.0-beta.1/set_env.sh
python train.py --variant full     # 40 epoch ≈ 72 s
```

---

## 6. 训练协议

- **源域**：6 组实验室工况全程带标签监督（Eq.27 第一项）；
- **目标域**：仅 10 个精度标定标签少样本监督（Eq.27 第二项，λ=2.0）；
- **模态对齐**：Bures-PMA 损失（λ=1.0）约束两域模态协方差对齐；
- **验证**：各源域数据集末 20% 时间外推（监控用，不参与训练）；
- 优化器 AdamW（lr 3e-4, wd 1e-4），batch 32，40 epoch（消融变体 25 epoch）。

消融变体（`--variant`）：

| tag | 含义 |
| --- | --- |
| `full` | 完整 PIMANN |
| `w_o_ca` | 去掉交叉注意力 |
| `w_o_pma` | 去掉 PMA 损失 |
| `w_o_ca_pma` | 两者都去掉 |
| `single_mode` | 仅单一物理模态基 |
| `mse_cov` | 协方差对齐改用 MSE（替代 Bures）|

---

## 7. 推理与产物

`python train.py --infer` 读取 `results/ckpt/pimann_full.pt`，产出：

| 文件 | 内容 |
| --- | --- |
| `csv/src_pred_ds{k}.npz` | 6 组源域张紧力真值 vs 预测（Fig.9）|
| `csv/fig11_metrics.json` | 源域 MSE/MAE/RMSE/R²/MAPE（Fig.11 / Table 3）|
| `csv/target_pred.npz` | 目标域预测 + 10 点标定匹配（Fig.12）|
| `csv/fig12_matching.csv` | 标定匹配明细（measured / matched / rel_err）|
| `csv/fig10_recon_demo.npz` | 跨域本征一致性特征重建演示（Fig.10）|
| `csv/msp_rollout.json` | 多步退化曲线外推（8 步滚动）|
| `csv/rul_summary.json` | 当前 HI / F_now / 预测 RUL（Eq.10）|
| `csv/table5_6_ablation.json` | 6 变体消融汇总（Table 5/6 / Fig.16）|

---

## 8. 部署（训练后上线推理）

### 8.1 加载权重
```python
import torch, numpy as np
import config as C
from model import PIMANN
from train import get_device, to_tensor, NORM_F, batch_encode

dev = get_device()
model = PIMANN(C).to(dev)
model.load_state_dict(torch.load("results/ckpt/pimann_full.pt",
                                 map_location=dev, weights_only=True))
model.eval()
```

### 8.2 在线张紧力预测
输入为标准化后的滑窗信号 `(B, 18, 256)`；交叉注意力要求两域批尺寸一致，
另一域提供参考流（可取训练域窗口循环填充至同 batch）：

```python
with torch.no_grad():
    # 生产线(目标域)部署路径: [1] 为目标域预测头
    F_tension = model(to_tensor(x_src_ref, dev),
                      to_tensor(x_prod_windows, dev))[1].cpu().numpy() * NORM_F
```

### 8.3 多步退化外推（未来 8 步张紧力）
```python
seq = recent_windows.reshape(1, C.SEQ_T, C.N_CH, C.WIN)   # 最近 8 个窗口
with torch.no_grad():
    z = batch_encode(model.expert_tgt, to_tensor(seq, dev))
    F_future = model.msp(z)[0].cpu().numpy() * NORM_F
```

### 8.4 精度退化度与 RUL（物理外推链）
```python
import kinematics as K
hi   = float(K.health_indicator(np.array([F_now]))[0])   # HI: 1 健康 → 0 失效
rul  = K.predict_rul(hi)                                 # Eq.10, 单位: 天
dth  = K.belt_additional_angle(np.array([F_now]))        # Eq.5 附加转角 (rad)
dR   = K.trajectory_deviation(q_traj, np.full(len(q_traj), F_now))  # Eq.12 末端偏差 (m)
```

### 8.5 部署注意事项
- 输入预处理必须与训练一致：滑窗 256/64 + 逐通道 zscore；
- 源/目标域批尺寸必须相等（交叉注意力约束），不足时循环填充参考流；
- 全链路 fp32，单 batch 前向即可满足在线监测时延（NPU 稳态毫秒级）；
- 权重与 `config.py` 超参绑定，升级结构时需同步更新配置。

---

## 9. 复现结果

**源域张紧力拟合（Fig.9/11）**

| 工况 | 天数 | R² | MAE (N) | MAPE (%) |
| --- | --- | --- | --- | --- |
| test 1 ("8") | 45 | 0.722 | 18.90 | 8.18 |
| test 2 ("Z") | 17 | 0.884 | 9.20 | 3.12 |
| test 3 ("L") | 29 | 0.718 | 14.83 | 4.94 |
| test 4 ("I") | 7 | 0.886 | 7.78 | 2.89 |
| test 5 ("O") | 35 | 0.899 | 8.91 | 3.33 |
| test 6 ("S") | 73 | 0.984 | 5.90 | 2.14 |

**基线对比（Table 4）**：PIMANN RMSE **12.51 N** vs RF 21.20 / GBR 21.18 /
SVR 26.60 / KNN 29.46；MAPE 4.11%。

**消融（Table 5/6）**：final MAPE — full **7.14** < w/o PMA 8.62 < w/o CA 9.35
= w/o CA&PMA 9.35 < single_mode 9.73。

**少样本稳定性（Fig.8）**：4 种标签比例 × 40 试验 × 4 超参 = 640 次 NPU 真实训练，
最终精度均值全部稳定在 0.88。

**跨域标定匹配（Fig.12）**：平均相对误差 2.13%（最大 4.22%）。

**图件**（`results/figures/`，点击查看 png）：
[Fig.8](results/figures/png/figure8.png) ·
[Fig.9](results/figures/png/figure9.png) ·
[Fig.10](results/figures/png/figure10.png) ·
[Fig.11](results/figures/png/figure11.png) ·
[Fig.12](results/figures/png/figure12.png) ·
[Fig.13](results/figures/png/figure13.png) ·
[Fig.14](results/figures/png/figure14.png) ·
[Fig.15](results/figures/png/figure15.png) ·
[Fig.16](results/figures/png/figure16.png)

---

## 10. 昇腾适配说明

- 多步预测头在 NPU 上以 `Conv1d` 时序卷积实现（CANN 不覆盖 GRU 的
  DynamicGRUV2 形态），CPU/NPU 两端结构一致；
- Bures 矩阵开平方基于 `torch.linalg.eigh` + 特征值 clamp 的可微实现，NPU 上
  的 internal-format 告警为无害信息（CPU/NPU 前向相对误差 <1e-5）；
- 交叉注意力要求源/目标域批尺寸严格一致，采样不足一批时有放回重采样对齐；
- 详细迁移问题与解决方案见配套实验报告（CANN 专项章节）。

---

## 11. 引用

```bibtex
@article{wang2026pimann,
  title   = {Welding robot digital twin and precision degradation prediction
             based on physics-informed modal alignment neural network},
  author  = {Wang, S. and others},
  journal = {Journal of Manufacturing Systems},
  volume  = {86},
  pages   = {970--990},
  year    = {2026}
}
```

本仓库为该论文方法论在华为鲲鹏/昇腾平台上的复现实现，供学术研究使用。
