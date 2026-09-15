# -*- coding: utf-8 -*-
"""
PIMANN 复现 —— 精度评估模型 (论文 3.1 节)

标准 DH 正运动学 (Eq.6/11)、张紧力引起的附加转角 (Eq.5)、
末端轨迹偏差 (Eq.7/8/12)、精度退化度与 RUL (Eq.9/10)。
DH 参数为假设的开源六轴焊接机器人构型 (无真实运动学数据)。
"""
import numpy as np

import config as C


def dh_transform(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    """标准 DH 单关节变换矩阵"""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def forward_kinematics(theta: np.ndarray, dh=None) -> np.ndarray:
    """Eq.11: R(t) = Π DH_i -> 末端位置 (x, y, z)"""
    dh = C.DH_PARAMS if dh is None else dh
    T = np.eye(4)
    for (alpha, a, d), th in zip(dh, theta):
        T = T @ dh_transform(alpha, a, d, th)
    return T[:3, 3]


def fk_jacobian(theta: np.ndarray, dh=None, eps=1e-6) -> np.ndarray:
    """数值雅可比 (位置部分, 3x6)"""
    dh = C.DH_PARAMS if dh is None else dh
    J = np.zeros((3, len(theta)))
    for i in range(len(theta)):
        dth = np.zeros_like(theta)
        dth[i] = eps
        J[:, i] = (forward_kinematics(theta + dth, dh)
                   - forward_kinematics(theta - dth, dh)) / (2 * eps)
    return J


def inverse_kinematics_path(path_xyz: np.ndarray, q0: np.ndarray,
                            n_iter=60, lam=1e-3) -> np.ndarray:
    """
    阻尼最小二乘数值 IK: 生成跟踪末端轨迹 (N,3) 的关节轨迹 (N,6)。
    仅位置约束, 姿态自由度冗余由零空间阻尼处理。
    """
    qs = np.zeros((len(path_xyz), len(q0)))
    q = q0.copy()
    for i, p_des in enumerate(path_xyz):
        for _ in range(n_iter):
            e = p_des - forward_kinematics(q)
            if np.linalg.norm(e) < 1e-5:
                break
            J = fk_jacobian(q)
            JJt = J @ J.T + lam * np.eye(3)
            dq = J.T @ np.linalg.solve(JJt, e)
            q = q + np.clip(dq, -0.15, 0.15)
        qs[i] = q
    return qs


def belt_additional_angle(F: np.ndarray) -> np.ndarray:
    """Eq.5: θ0 = (F0 - F)L / (E·r), 张紧力下降引起的附加转角 (rad)"""
    return np.maximum(C.F0 - F, 0.0) * C.BELT_L / (C.BELT_E * C.BELT_R)


def trajectory_deviation(theta_traj: np.ndarray, F: np.ndarray,
                         belt_joint: int = 4) -> np.ndarray:
    """
    Eq.12: ΔR = ||DH(θ) - DH(θ_k + θ_add)||₂
    theta_traj: (N, 6), F: (N,) -> 每时刻末端位置偏差 (m)
    """
    dR = np.zeros(len(F))
    th_add = belt_additional_angle(F)
    for i in range(len(F)):
        p1 = forward_kinematics(theta_traj[i])
        th2 = theta_traj[i].copy()
        th2[belt_joint] += th_add[i]
        p2 = forward_kinematics(th2)
        dR[i] = np.linalg.norm(p1 - p2)
    return dR


def health_indicator(F: np.ndarray) -> np.ndarray:
    """归一化健康指标: HI=1 健康, HI=0 失效"""
    return np.clip((F - C.F_FAIL) / (C.F0 - C.F_FAIL), 0.0, 1.0)


def predict_rul(hi0: float) -> float:
    """
    Eq.10: RUL = (γ − β + √(β² + 2γ(HI(t0) − HI_fail − 3σ_ε))) / γ
    γ→0 时退化为线性模型。
    """
    beta, gamma = C.RUL_BETA, C.RUL_GAMMA
    disc = hi0 - C.HI_FAIL - 3 * C.SIGMA_EPS
    if disc <= 0:
        return 0.0
    if abs(gamma) < 1e-12:
        return disc / beta
    return (-beta + np.sqrt(beta ** 2 + 2 * gamma * disc)) / gamma


def weighted_precision_index(dx, dy, dz, k_dev: np.ndarray = None,
                             w=(0.4, 0.4, 0.2), alpha=0.1) -> float:
    """Eq.9: 方向加权精度退化度 (RUL2 中的加权偏差项)"""
    dev = w[0] * np.asarray(dx) ** 2 + w[1] * np.asarray(dy) ** 2 \
        + w[2] * np.asarray(dz) ** 2
    if k_dev is None:
        k_dev = np.ones_like(dev)
    return float(np.mean(dev * k_dev) + alpha * np.std(np.sqrt(dev)))


def make_circle_path(center: np.ndarray, radius: float, n: int) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n)
    return center + radius * np.stack(
        [np.cos(t), np.sin(t), np.zeros(n)], axis=1)


def make_rectangle_path(center: np.ndarray, w: float, h: float, n: int) -> np.ndarray:
    hw, hh = w / 2, h / 2
    corners = np.array([[hw, hh], [-hw, hh], [-hw, -hh], [hw, -hh], [hw, hh]])
    seg = n // 4
    pts = []
    for i in range(4):
        pts.append(np.stack([
            np.linspace(corners[i, 0], corners[i + 1, 0], seg),
            np.linspace(corners[i, 1], corners[i + 1, 1], seg)], axis=1))
    xy = np.concatenate(pts, axis=0)
    return np.stack([xy[:, 0] + center[0], xy[:, 1] + center[1],
                     np.full(len(xy), center[2])], axis=1)


def reference_pose() -> np.ndarray:
    """一个工作空间内部的参考位形"""
    return np.array([0.0, -0.7, 1.3, 0.0, -0.6, 0.0])
