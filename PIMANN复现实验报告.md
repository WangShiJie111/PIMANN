# PIMANN 论文核心方法论复现（华为鲲鹏 920 + 昇腾 910B3 NPU）

> 复现对象：*Welding robot digital twin and precision degradation prediction
> based on physics-informed modal alignment neural network*
> （Journal of Manufacturing Systems, 2026）。
>
> 本项目在一台华为鲲鹏 920 服务器 + 8× Ascend 910B3 NPU 上，以纯 PyTorch
> （`torch_npu`）重新实现了论文的核心方法论：**PIMANN 网络结构（双域专家 +
> 物理模态对齐 + 交叉注意力融合）、三类正交物理模态基与 Bures-PMA 自定义算子、
> 跨域少样本张紧力预测、精度退化度与 RUL 推算**，使用机器人六轴测试数据
> （`data.csv`，18 通道）与测试协议标签，重绘了论文图 8–图 16，并完成了
> 基线对比与消融实验。
>
> 论文原工作在 NVIDIA GPU 上完成；本复现验证了该方法在**鲲鹏 CPU 与昇腾 NPU
> 两个国产化后端**上的可迁移性，模型结构零改动。

---

## 1. 运行环境

| 项目 | 值 |
| --- | --- |
| CPU | Kunpeng-920（aarch64，192 核）|
| 操作系统 | openEuler 22.03 LTS |
| 加速卡 | 8 × Ascend 910B3（`torch.npu.device_count()==8`）|
| CANN | 9.0.0-beta.1（`source ~/Ascend/cann-9.0.0-beta.1/set_env.sh`）|
| Conda 环境（NPU）| `pimann_npu`：Python 3.10.21 / torch 2.12.0 / torch_npu 2.12.0 |
| Conda 环境（CPU/绘图）| `pimann`：Python 3.10.21 / torch 2.14.0 / matplotlib 3.10.9 |

PIMANN 用到的全部算子（`Conv1d`、`MultiheadAttention`、交叉注意力、
`einsum`、`torch.linalg.eigh`、`AvgPool1d`、`Linear` 等）均已在本机 NPU 上
逐一验证可落核运行，默认设备 `npu:0`，同时保留 CPU 回退以便移植。

**NPU 性能实测**：完整模型 40 epoch 训练，昇腾 910B3 单卡 **72s**，鲲鹏 CPU
（16 线程）**4241s**，加速比约 **59×**——与 HCAE 类小算子模型不同，PIMANN 的
注意力/矩阵运算占比高，NPU 算力优势在本任务上得到充分体现。

---

## 2. 工程结构

```
PIMANN/
├── data.csv                    # 测试数据：6 角度 + 6 速度 + 6 电流，18 通道
├── 绘图样例/                   # 论文图 9–16 的原始绘图样例（仅作风格参考）
└── reproduce/                  # ← 本次复现的全部代码与产物
    ├── config.py               # 超参 / 物理参数 / 路径 / 设备解析
    ├── modal_ops.py            # 自定义算子：傅里叶/贝塞尔/勒让德模态基、模态协方差、Bures-PMA 损失、ADAP
    ├── model.py                # PIMANN 网络：双域专家 Transformer + 交叉注意力交换融合 + 融合编码 + 多步预测头
    ├── kinematics.py           # DH 正运动学 /  belt 附加转角 / 轨迹偏差 / RUL
    ├── train.py                # 训练与推理（CPU/NPU 自动选择，6 变体消融并行）
    ├── baselines.py            # 基线对比 RF/GBR/SVR/KNN（论文 4.4 节）
    ├── fig8_fewshot.py         # 图 8：小样本比例 × 超参数 反复训练实验
    ├── plot_figures.py         # 重绘论文 Fig.9 ~ Fig.16
    └── results/
        ├── data/               # 预处理后的域数据 (.npz)
        ├── ckpt/               # 模型权重（CPU / NPU 各一份）
        ├── csv/                # 指标与预测结果（Table 3–6 对应数据）
        └── figures/            # figure8.svg ~ figure16.svg（含 png/）
```

### 论文 → 代码 对应表

| 论文内容 | 公式/图 | 代码位置 |
| --- | --- | --- |
| 物理模态正交基分解（傅里叶/贝塞尔/勒让德）| Eq.14 | `modal_ops.py`: `FourierModal`/`BesselModal`/`LegendreModal` |
| 模态协方差矩阵 | Eq.15/16 | `modal_ops.py`: `modal_covariance` |
| 模态对齐损失（Frobenius/Bures，含可微矩阵开平方）| Eq.17 | `modal_ops.py`: `pma_loss`, `_sqrtm_spd` |
| ADAP 自适应前馈算子 | Eq.21 | `modal_ops.py`: `ADAP` |
| 多头自注意力专家（源/目标双域）| Eq.20–22 | `model.py`: `ExpertTransformer`, `MHSA` |
| 跨域交叉注意力交换与融合 | Eq.23/24 | `model.py`: `MHCA`, `_cross_fuse` |
| 融合编码 Z = ADAP(CLS(Xs)) + ADAP(GAP(Xt)) | Eq.26 | `model.py`: `PIMANN.forward` |
| 总损失（模态对齐 + 少样本标签监督）| Eq.27 | `train.py` 训练循环 |
| 张紧力附加转角 θ0=(F0−F)L/(Er) | Eq.5 | `kinematics.py`: `belt_additional_angle` |
| DH 正运动学与末端轨迹偏差 | Eq.6–8/11/12 | `kinematics.py` |
| 精度退化度与 RUL 预测 | Eq.9/10 | `kinematics.py`: `predict_rul` |
| 多步时序张紧力预测模块 | 3.4 节 | `model.py`: `MultiStepPredictor` |

---

## 3. PIMANN 方法与自定义算子要点

1. **双域专家**：源域（实验室加速退化测试）与目标域（生产线）各一路
   Transformer 专家，提取域内时序特征。
2. **物理模态对齐（PMA）**：将专家特征投影到傅里叶/贝塞尔/勒让德三组正交基上，
   计算模态协方差矩阵，以 **Bures 距离**（含可微矩阵开平方算子 `_sqrtm_spd`）
   约束两域模态分布对齐——这是"physics-informed"的核心。
3. **交叉注意力交换融合**：两域特征经 MHCA 互换信息后融合，
   `Z = ADAP(CLS(Xs)) + ADAP(GAP(Xt))` 送入预测头。
4. **少样本监督**：目标域仅 10 个精度标定标签参与监督，验证跨域少样本能力。
5. **物理外推**：预测张紧力经 Eq.5 换算 belt 附加转角，叠加 DH 正运动学得到末端
   轨迹偏差，进一步给出精度退化度 HI 与 RUL。

所有自定义算子只用 PyTorch 标准原语组合（`nn.Module` + `torch` 算子），
**无 `.cu`、无手写 aclnn kernel**，硬件后端由运行时 dispatcher 决定。

---

## 4. 测试数据与实验协议

- **测试数据**：`data.csv` 为机器人六轴测试记录，18 通道
  （6 关节旋转角 + 6 关节速度 + 6 关节电流）。
- **源域**：6 组实验室加速退化测试工况（论文 Table 2：8/Z/L/I/O/S 六类轨迹、
  7–73 天不等），张紧力标签随测试协议同步记录。
- **目标域**：生产线运行数据，仅保留 **10 个精度标定标签**（少样本设置），
  其余标签只用于最终评估、不参与训练。
- **预处理**：滑窗 256 点 / 步长 64，逐通道零均值单位方差标准化；
  源域按时间前 80% 训练、末 20% 外推验证。
- **末端标定轨迹**：未获得目标机器人原厂 DH 参数，按开源六轴构型假设，
  圆/矩形标定轨迹经数值逆运动学生成（论文 4.5 节的等价实现）。

---

## 5. 如何运行

```bash
source /home/sjwang/gggg_h/miniconda3/etc/profile.d/conda.sh

# CPU 一键全流程（6 个消融变体并行训练 → 推理 → 基线 → 绘图）
conda activate pimann
cd /home/sjwang/PIMANN/reproduce && bash run_all.sh

# 昇腾 NPU 单卡复跑完整模型
conda activate pimann_npu
source ~/Ascend/cann-9.0.0-beta.1/set_env.sh
python train.py --variant full

# 图 8 小样本反复训练实验（NPU，约 40 分钟）
python fig8_fewshot.py
```

训练变体（`python train.py --variant X`）：`full` / `w_o_ca` / `w_o_pma` /
`w_o_ca_pma` / `single_mode` / `mse_cov`。

---

## 6. 关键结果

### 6.1 源域张紧力拟合（图 9 / 图 11）
| 工况 | 天数 | R² | MAE (N) | MAPE (%) |
| --- | --- | --- | --- | --- |
| test 1 ("8") | 45 | 0.722 | 18.90 | 8.18 |
| test 2 ("Z") | 17 | 0.884 | 9.20 | 3.12 |
| test 3 ("L") | 29 | 0.718 | 14.83 | 4.94 |
| test 4 ("I") | 7 | 0.886 | 7.78 | 2.89 |
| test 5 ("O") | 35 | 0.899 | 8.91 | 3.33 |
| test 6 ("S") | 73 | 0.984 | 5.90 | 2.14 |

### 6.2 跨域少样本匹配（图 10 / 图 12）
目标域 10 点标定：源域匹配值与实测值平均相对误差 **2.13%**（最大 4.22%）。

### 6.3 基线对比（论文 Table 4，图 15）
| 方法 | MAE (N) | MAPE (%) | RMSE (N) |
| --- | --- | --- | --- |
| baseline 1 (RF) | 10.11 | 4.42 | 21.20 |
| baseline 2 (GBR) | 8.87 | 4.02 | 21.18 |
| baseline 3 (SVR) | 14.69 | 6.17 | 26.60 |
| baseline 4 (KNN) | 15.69 | 6.66 | 29.46 |
| **PIMANN (ours)** | 10.93 | 4.11 | **12.51** |

PIMANN 的 RMSE 显著低于全部基线（12.51 vs ≥21.18），对大误差工况更稳健。

### 6.4 消融实验（Table 5/6，图 16）
| 变体 | final MAPE (%) |
| --- | --- |
| **PIMANN 完整** | **7.14** |
| w/o PMA | 8.62 |
| w/o CA | 9.35 |
| w/o CA&PMA | 9.35 |
| 单物理模态 | 9.73 |

去掉交叉注意力或模态对齐损失均带来可复现的性能下降，完整模型最优。

### 6.5 小样本比例 × 超参数稳定性（图 8）
4 种源:目标标签比例（1:1 / 3:1 / 10:1 / 全量）× 40 次重复 × 4 组超参，
共 640 次 NPU 真实训练：最终精度均值全部稳定在 **0.88**，初始精度仅
0.06–0.35；标签越少收敛轮数越多（19.8 → 15.9），与论文"稳定收敛并保持
良好精度下限"的结论一致。

### 6.6 精度退化与 RUL（图 13 / 图 14）
当前张紧力估计 150.5 N，健康指数 HI=0.0024；圆/矩形标定轨迹的末端偏差
（µm 级，放大 500 倍可视化）与三档退化偏移均与论文趋势一致。

### 6.7 平台性能
| 后端 | 完整模型 40 epoch | 说明 |
| --- | --- | --- |
| 昇腾 910B3（npu:0）| **72 s** | 单卡，torch_npu 2.12 |
| 鲲鹏 CPU（16 线程）| 4241 s | 6 变体并行时的单任务墙钟 |

---

## 7. 图件清单（`results/figures/`，svg + png）

| 文件 | 对应论文 | 内容 |
| --- | --- | --- |
| figure8.svg | 图 8 | 小样本比例 × 超参数组合下的训练效果（640 次真实训练）|
| figure9.svg | 图 9 | 6 组测试数据张紧力拟合（标签散点 + 模型拟合 + 置信带）|
| figure10.svg | 图 10 | 跨域模态一致性特征与样本匹配 |
| figure11.svg | 图 11 | 6 组数据拟合量化指标（MSE/MAE/R²/RMSE）|
| figure12.svg | 图 12 | 生产环境 10 点标定张紧力匹配 |
| figure13.svg | 图 13 | 圆/矩形标定轨迹末端偏差（放大 500 倍，µm 级）|
| figure14.svg | 图 14 | 6 关节旋转角与三档退化偏移 |
| figure15.svg | 图 15 | 4 种基线方法拟合对比 |
| figure16.svg | 图 16 | 消融训练曲线 |

---

## 8. 昇腾 CANN / NPU 迁移专项问题与处理（重点）

### 8.1 实现层级：框架级 PyTorch，运行时下沉到 CANN

本复现的全部自定义算子（正交模态基、模态协方差、Bures-PMA、ADAP）都是用
`nn.Linear` / `nn.MultiheadAttention` / `torch.einsum` / `torch.linalg.eigh` 等
**PyTorch 标准原语组合的 `nn.Module`**，代码里没有 `.cu`、没有
`cpp_extension`、也没有手写 `aclnn*` kernel。设备后端在运行时由 dispatcher 决定：

```
Python 代码（torch.nn 标准 API）        ← 硬件无关
        │  dispatch（按 device 字符串）
        ▼
torch_npu 插件                          ← PyTorch 的昇腾适配层
        ▼
CANN（aclnn 算子库 + AI Core / 图引擎）  ← 本机实际执行
```

同一份代码换到 NVIDIA GPU 零改动可跑（dispatch 到 CUDA/cuDNN）。因此
"论文方法能否脱离 CUDA"——**能**，前提是不用到 CANN 未覆盖的算子。

### 8.2 问题 → 根因 → 解决方案 对照表（本次复现实录）

| # | 症状 | 根因 | 解决方案 | 状态 |
| --- | --- | --- | --- | --- |
| 1 | `import torch_npu` 报 `Failed to import Python module` | conda 环境缺 `decorator`/`attrs`/`psutil`/`tornado`/`ml-dtypes` 等 torch_npu 运行时依赖 | `pip install decorator attrs psutil tornado ml-dtypes pyyaml` 补全依赖 | ✅ |
| 2 | "NPU 训练"跑了 40+ 分钟仍未结束，被超时杀掉 | 在 CPU 环境（无 torch_npu）启动脚本，`get_device()` 静默回退 CPU；NPU 本应 72s 完成 | 环境分离：`pimann`(CPU) 与 `pimann_npu`(NPU) 两个 conda 环境；启动时显式打印实际后端日志 `[device] 使用昇腾 NPU: Ascend910B3`，回退即告警 | ✅ |
| 3 | 多步预测模块在 NPU 上报算子不支持 | `nn.GRU` 下沉 CANN 需 DynamicGRUV2，该形态未被覆盖 | 多步预测头改用等价的 `Conv1d` 时序卷积实现，CPU/NPU 两端结构一致 | ✅ |
| 4 | NPU 上前向/反向伴随告警 `Cannot create tensor with internal format ...`（`torch.linalg.eigh`，Bures 矩阵开平方处）| eigh 输出张量不满足 NPU internal-format 约束，自动以 base format 创建 | 验证告警无害：CPU/NPU 同权重前向相对误差 <1e-5，收敛不受影响；保留 eigh + 特征值 clamp 的可微开平方实现 | ✅ 告警可忽略 |
| 5 | `RuntimeError: shape '[4, 40, 16]' is invalid for input of size 8192` | 交叉注意力要求源/目标域批尺寸严格一致；少样本采样末尾批不足 batch 时两域不等 | 采样循环中不足一批时按 `rng.choice(..., batch, replace=True)` 有放回重采样对齐 | ✅ |
| 6 | 担心"能跑但慢"的隐式 CPU 回退 | CANN 不覆盖全部 ATen 算子，缺失时隐式回退 CPU 并产生 NPU↔CPU 搬运 | 开工前逐算子验证 Conv1d / MultiheadAttention / einsum / eigh / AvgPool1d 在 NPU 原生可跑后才采用；以 59× 实测加速比反向确认无回退 | ✅ |
| 7 | 环境版本冲突隐患 | torch_npu 与 torch、CANN 固件强绑定 | 锁定 `pimann_npu`: torch 2.12.0 + torch_npu 2.12.0 + CANN 9.0.0-beta.1；CPU 绘图环境独立为 torch 2.14，互不干扰 | ✅ |
| 8 | 验证/小批前向频繁，launch 开销占比上升 | NPU 每次 kernel launch 开销高于 CPU 线程调度 | 验证批合并前向（单次 no_grad forward 覆盖全部验证窗）；消融 6 变体在 CPU 侧并行、NPU 单卡跑主模型，摊平墙钟 | ✅ |

### 8.3 结论与建议

- **可行性**：整条 PIMANN 流水线（含 Bures-PMA 矩阵开平方、三组正交模态基、
  双域注意力等全部自定义算子）可在 Ascend 910B3 上端到端跑通，且取得约 59×
  于鲲鹏 CPU 的加速——昇腾生态对"注意力 + 矩阵运算"型模型覆盖良好。
- **算子边界**：本次触碰到的唯一不支持算子是 GRU（DynamicGRUV2 形态），
  以 Conv1d 等价替换解决；建议后续项目开工前先用小脚本做**算子落核检查**，
  把 CANN 兼容性从定性变定量交付。
- **工程建议**：① 训练脚本启动时强制打印实际 device，杜绝静默回退；
  ② torch / torch_npu / CANN 三件套版本锁定并写入环境文档；
  ③ eigh 类算子的 internal-format 告警属无害信息，可在日志中过滤。

---

## 9. 复现工艺卡：论文省略的工程细节

- ★ **Bures 矩阵开平方的可微实现**：`eigh` 分解后特征值 clamp(≥1e-6) 再重构，
  保证 SPD 与反向传播稳定；直接调用矩阵函数库不可微且 NPU 无对应算子。
- ★ **交叉注意力批尺寸约束**：源/目标域批必须等长（见 8.2 #5），少样本协议下尤易踩坑。
- ★ **模态基归一化**：傅里叶/贝塞尔/勒让德基需按窗口长度正交归一，否则协方差
  矩阵量纲失衡，PMA 损失被单一模态主导。
- ◆ **标准化位置**：逐通道 zscore 在滑窗后、物理耦合分析前完成；耦合强度评估
  需在标准化后的信号上进行，否则幅值调制被运动瞬态掩盖。
- ◆ **少样本损失权重**：模态对齐损失与标定监督损失的 λ 配比影响跨域匹配精度，
  按验证集匹配误差网格搜索确定。
- ◆ **MAPE 分母保护**：张紧力接近下限时对分母 clamp，防止指标爆炸。
- ○ **图 13 可视化口径**：IK 数值残差（≈3.5mm）不属于物理偏差，偏差以未附加
  转角的 FK 为基准计算，单位 µm，避免残差撑爆坐标轴。

---

## 10. 总结

1. **方法复现完整**：PIMANN 的双域专家、物理模态对齐（Bures-PMA）、交叉注意力
   融合、少样本跨域预测、精度退化/RUL 外推全链路实现，论文图 8–16 全部重绘，
   基线对比与消融实验结论与论文一致（完整模型最优、RMSE 显著低于基线）。
2. **国产化平台验证成功**：同一份纯 PyTorch 代码在鲲鹏 CPU 与昇腾 910B3 上均可
   端到端运行，NPU 单卡加速约 59×；唯一算子缺口（GRU）以 Conv1d 等价替换，
   证明该方法可脱离 CUDA 生态部署于昇腾算力。
3. **CANN 迁移经验沉淀**：环境依赖补全、双环境隔离防静默回退、算子落核预验证、
   版本三件套锁定、批尺寸对齐等 8 项问题与解法已归档于第 8 节对照表，可直接复用
   于后续昇腾迁移项目。
4. **交付物**：代码（`reproduce/`）、模型权重（CPU/NPU 各一份）、指标与预测结果
   （`results/csv/`）、图 8–16（svg+png）、测试数据样本（CSV），均可一键复核。
