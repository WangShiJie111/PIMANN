# -*- coding: utf-8 -*-
"""
PIMANN 复现 —— 全局配置
论文: Welding robot digital twin and precision degradation prediction (JMS 2026)
运行平台: 华为鲲鹏 920 (aarch64 CPU) / 昇腾 910B NPU(可选)
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_CSV = os.path.join(BASE_DIR, "..", "data.csv")
RESULT_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(RESULT_DIR, "data")
CKPT_DIR = os.path.join(RESULT_DIR, "ckpt")
FIG_DIR = os.path.join(RESULT_DIR, "figures")
CSV_DIR = os.path.join(RESULT_DIR, "csv")

for _d in (DATA_DIR, CKPT_DIR, FIG_DIR, CSV_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------- 数据与测试协议 ----------------
SEED = 42
SAMPLING_RATE_LAB = 250          # 实验室采样频率 (Hz), 论文 4.2
SAMPLING_RATE_PROD = 100         # 生产现场采样频率 (Hz), 论文 4.2
N_SRC_DATASETS = 6               # 6 组实验室加速退化实验 (Table 2)
SRC_LEN = 8000                   # 每组源域样本点数
TGT_LEN = 16000                  # 目标域(生产)样本点数
WIN = 256                        # 滑动窗口长度
STRIDE = 64                      # 滑动窗口步长

F0 = 350.0                       # 同步带初始张紧力 (N)
F_FAIL = 150.0                   # 张紧力失效阈值 (N)
COUPLE_ALPHA = 0.85              # 电流-张紧力耦合系数 (Eq.4 T=Km*Ia)
TEST_DAYS = [45, 17, 29, 7, 35, 73]          # 论文 Table 2 实验时长
TEST_TRAJ = ["8", "Z", "L", "I", "O", "S"]   # 论文 Table 2 轨迹类型
PROD_DAYS = 180                              # 生产数据时长(天)
N_CALIB = 10                                 # 生产现场 10 次精度标定(张紧力测量)

# ---------------- 模型 ----------------
N_MODAL = 16                     # 每种正交基保留的模态数 (Eq.14)
D_MODEL = 64                     # transformer 隐层维度
N_HEAD = 4                       # 多头注意力头数
N_LAYER = 2                      # expert transformer 层数
D_FF = 128                       # FFN 中间维度
N_CH = 18                        # 输入通道数(6角速度...共18列)
SEQ_T = 8                        # 多步预测输入序列长度
PRED_H = 8                       # 多步预测步长

# ---------------- 训练 ----------------
EPOCHS = 40
ABL_EPOCHS = 25                  # 消融变体训练轮数
BATCH = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4
LAMBDA_MSE = 1.0                 # Eq.27 中 λ
LAMBDA_FEW = 2.0                 # 目标域少量标签损失权重
LAMBDA_PMA = 1.0                 # 模态对齐损失权重
N_FEWSHOT = 10                   # 目标域少样本标签数(对应10次标定)
DEVICE = "auto"                  # auto: npu > cpu
NUM_WORKERS = 0

# ---------------- 精度评估模型 (Eq.5/Eq.10) ----------------
BELT_L = 0.85                    # 同步带长度 (m), 假设值
BELT_E = 5e7                     # 同步带等效弹性模量 (N), 假设值
BELT_R = 0.045                   # 带轮节圆半径 (m), 假设值
HI_FAIL = 0.3                    # Eq.10 失效阈值
RUL_BETA = 0.010                 # HI 衰减率
RUL_GAMMA = 8e-4                 # 衰减加速度
SIGMA_EPS = 0.01                 # 随机扰动 3σ

# 假设的 6 自由度焊接机器人 DH 参数(标准DH), 参考开源六轴构型
# [alpha(rad), a(m), d(m)]
DH_PARAMS = [
    (-1.5707963, 0.05, 0.45),
    (0.0,        0.60, 0.0),
    (1.5707963,  0.10, 0.0),
    (-1.5707963, 0.00, 0.65),
    (1.5707963,  0.00, 0.0),
    (0.0,        0.00, 0.10),
]
