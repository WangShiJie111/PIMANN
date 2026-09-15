# -*- coding: utf-8 -*-
"""
PIMANN 复现 —— 自定义算子层 (论文 3.2 节)

包含三类物理模态正交基变换 (Eq.14)、模态协方差矩阵 (Eq.15/16)
以及基于 Frobenius/Bures 距离的物理模态对齐损失 (Eq.17)。
全部用 PyTorch 原生算子实现, 支持 CPU 与昇腾 NPU 自动后端。
"""
import math

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# 1) 正交模态基: Fourier / Bessel / Legendre  (论文 Eq.14)
# ----------------------------------------------------------------------
class FourierModal(nn.Module):
    """傅里叶模态: 取前 K 阶频谱幅值作为模态序列"""

    def __init__(self, win_len: int, n_modal: int):
        super().__init__()
        self.n_modal = n_modal
        self.register_buffer("window", torch.hann_window(win_len), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, W) -> (B, C, K)
        spec = torch.fft.rfft(x * self.window, dim=-1)
        amp = spec.abs()[..., : self.n_modal]
        return amp


class LegendreModal(nn.Module):
    """勒让德多项式模态: f(t) = Σ a_n P_n(t), 用最小二乘投影求系数"""

    def __init__(self, win_len: int, n_modal: int):
        super().__init__()
        from numpy.polynomial.legendre import legvander
        grid = np.linspace(-1, 1, win_len)
        V = torch.from_numpy(legvander(grid, n_modal - 1)).float()  # (W, K)
        # 投影矩阵: (V^T V)^-1 V^T, 固定不参与训练 (物理先验)
        VtV_inv = torch.linalg.inv(V.T @ V + 1e-6 * torch.eye(n_modal))
        self.register_buffer("proj", VtV_inv @ V.T, persistent=False)  # (K, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, W) -> (B, C, K)
        return torch.matmul(x, self.proj.T)


class BesselModal(nn.Module):
    """贝塞尔函数模态: f(t) = Σ a_n J_n(kr), 固定基投影"""

    def __init__(self, win_len: int, n_modal: int):
        super().__init__()
        from scipy.special import jv

        r = torch.linspace(0.1, 1.0, win_len).numpy()
        # 各阶贝塞尔函数的第 1 个正零点作为特征波数 (物理边界约束)
        basis = np.zeros((n_modal, win_len), dtype=np.float32)
        for n in range(n_modal):
            k_n = _bessel_first_zero(n)
            basis[n] = jv(n, k_n * r)
        B = torch.from_numpy(basis).T  # (W, K)
        BtB_inv = torch.linalg.inv(B.T @ B + 1e-6 * torch.eye(n_modal))
        self.register_buffer("proj", BtB_inv @ B.T, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.proj.T)


def _bessel_first_zero(n: int) -> float:
    """J_n(x) 的第一个正零点近似 (McMahon 展开)"""
    if n == 0:
        return 2.4048255577
    beta = (n + 0.5) * math.pi
    return beta - 4.0 / (3.0 * beta)


class PhysicalModalBank(nn.Module):
    """三模态并行分解: 输出 S1(傅里叶) S2(贝塞尔) S3(勒让德)"""

    def __init__(self, win_len: int, n_modal: int):
        super().__init__()
        self.fourier = FourierModal(win_len, n_modal)
        self.bessel = BesselModal(win_len, n_modal)
        self.legendre = LegendreModal(win_len, n_modal)

    def forward(self, x: torch.Tensor):
        # x: (B, C, W) -> 三个 (B, C, K)
        s1 = self.fourier(x)
        s2 = self.bessel(x)
        s3 = self.legendre(x)
        return s1, s2, s3


# ----------------------------------------------------------------------
# 2) 模态协方差矩阵 (论文 Eq.15 / Eq.16)
# ----------------------------------------------------------------------
def modal_covariance(s1: torch.Tensor, s2: torch.Tensor, s3: torch.Tensor) -> torch.Tensor:
    """
    每个样本的三组模态序列展平后计算 3x3 协方差矩阵, 再取批均值。
    s_i: (B, C, K) -> 返回 (3, 3) 批平均协方差矩阵 (对称半正定)
    """
    B = s1.shape[0]
    S = torch.stack([s1.reshape(B, -1), s2.reshape(B, -1), s3.reshape(B, -1)], dim=1)
    # S: (B, 3, C*K), 批内标准化后对特征维求协方差 (数值稳定, 使损失量纲为 O(1))
    S = (S - S.mean(dim=0, keepdim=True)) / (S.std(dim=0, keepdim=True) + 1e-6)
    mu = S.mean(dim=-1, keepdim=True)
    S0 = S - mu
    C = torch.bmm(S0, S0.transpose(1, 2)) / (S.shape[-1] - 1)
    C = C.mean(dim=0)  # (3,3)
    # 加小量保证正定, 便于开平方
    C = C + 1e-4 * torch.eye(C.shape[0], device=C.device, dtype=C.dtype)
    return C


# ----------------------------------------------------------------------
# 3) 物理模态对齐损失 (论文 Eq.17, Frobenius/Bures 距离)
# ----------------------------------------------------------------------
def _sqrtm_spd(C: torch.Tensor) -> torch.Tensor:
    """对称正定矩阵开平方 (可微), 特征分解实现"""
    vals, vecs = torch.linalg.eigh(C)
    vals = torch.clamp(vals, min=1e-8)
    return (vecs * vals.sqrt().unsqueeze(-2)) @ vecs.transpose(-1, -2)


def pma_loss(Cs: torch.Tensor, Ct: torch.Tensor) -> torch.Tensor:
    """
    Eq.17: l = ||Cs - Ct||_F^2
         = Tr(Cs + Ct - 2 (Cs^{1/2} Ct Cs^{1/2})^{1/2})
    即两个零均值高斯分布之间的平方 Bures 距离。
    """
    sqrt_Cs = _sqrtm_spd(Cs)
    inner = _sqrtm_spd(sqrt_Cs @ Ct @ sqrt_Cs)
    loss = torch.trace(Cs) + torch.trace(Ct) - 2.0 * torch.trace(inner)
    return torch.relu(loss)


# ----------------------------------------------------------------------
# 4) ADAP 自适应前馈算子 (论文 Eq.21)
# ----------------------------------------------------------------------
class ADAP(nn.Module):
    """ADAP(X) = σ(X·W_D)·W_U, 降维-非线性-升维的自适应前馈"""

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.W_D = nn.Linear(d_in, d_hidden)
        self.act = nn.GELU()
        self.W_U = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_U(self.act(self.W_D(x)))
