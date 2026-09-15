# -*- coding: utf-8 -*-
"""
PIMANN 复现 —— 网络主体 (论文 3.2 / 3.3 节, Fig.4-Fig.5, Table 1)

结构:
  源域/目标域 双专家 -> 物理模态分解(傅里叶/贝塞尔/勒让德)
  -> 多头自注意力 Transformer (Eq.20-22)
  -> 跨域交叉注意力交换与融合 (Eq.23-25)
  -> Z = ADAP(CLS(Xs)) + ADAP(GAP(Xt)) (Eq.26)
  -> 张紧力回归头 + 多步时序预测头
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from modal_ops import ADAP, PhysicalModalBank, modal_covariance, pma_loss


# ----------------------------------------------------------------------
# 多头自注意力 (论文 Eq.20)
# ----------------------------------------------------------------------
class MHSA(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x)
        return out


# ----------------------------------------------------------------------
# 多头交叉注意力 (论文 Eq.23): Q 来自域1, K/V 来自域2
# ----------------------------------------------------------------------
class MHCA(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)

    def forward(self, q_src: torch.Tensor, kv_src: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(q_src, kv_src, kv_src)
        return out


# ----------------------------------------------------------------------
# 域专家: 模态分解 + Transformer 块 (Eq.22)
# ----------------------------------------------------------------------
class ExpertTransformer(nn.Module):
    def __init__(self, n_ch: int, n_modal: int, win_len: int,
                 d_model: int, n_head: int, n_layer: int, d_ff: int,
                 use_multi_mode: bool = True):
        super().__init__()
        self.use_multi_mode = use_multi_mode
        self.modal_bank = PhysicalModalBank(win_len, n_modal)
        n_tok = 3 if use_multi_mode else 1          # 单模态消融: 仅傅里叶
        self.n_tok = n_tok
        self.token_proj = nn.Linear(n_ch * n_modal, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "ln1": nn.LayerNorm(d_model),
                "mhsa": MHSA(d_model, n_head),
                "ln2": nn.LayerNorm(d_model),
                "adap": ADAP(d_model, d_ff, d_model),
            }) for _ in range(n_layer)
        ])

    def decompose(self, x: torch.Tensor):
        s1, s2, s3 = self.modal_bank(x)
        if self.use_multi_mode:
            return (s1, s2, s3)
        return (s1,)

    def forward(self, x: torch.Tensor):
        """x: (B, C, W) -> tokens (B, 1+n_tok, d), modals tuple"""
        modals = self.decompose(x)
        B = x.shape[0]
        toks = [self.token_proj(m.reshape(B, -1)).unsqueeze(1) for m in modals]
        tokens = torch.cat(toks, dim=1)                       # (B, n_tok, d)
        tokens = torch.cat([self.cls_token.expand(B, -1, -1), tokens], dim=1)
        for lyr in self.layers:
            tokens = tokens + lyr["mhsa"](lyr["ln1"](tokens))           # Eq.22 残差
            tokens = tokens + lyr["adap"](lyr["ln2"](tokens))
        return tokens, modals


# ----------------------------------------------------------------------
# 多步时序张紧力预测模块 (论文 3.4: multi-step time series prediction)
# ----------------------------------------------------------------------
class MultiStepPredictor(nn.Module):
    """多步张紧力预测: 时序卷积实现 (避免 GRU 算子, 兼容 NPU/CPU)"""

    def __init__(self, d_model: int, horizon: int, seq_len: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(d_model * seq_len, d_model), nn.GELU(),
            nn.Linear(d_model, horizon))

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        """z_seq: (B, T, d) -> 未来 H 步张紧力 (B, H)"""
        h = self.conv(z_seq.transpose(1, 2)).transpose(1, 2)
        return self.head(h.reshape(h.shape[0], -1))


# ----------------------------------------------------------------------
# PIMANN 主网络
# ----------------------------------------------------------------------
class PIMANN(nn.Module):
    def __init__(self, cfg, use_ca: bool = True, use_multi_mode: bool = True):
        super().__init__()
        self.cfg = cfg
        self.use_ca = use_ca
        d = cfg.D_MODEL

        self.expert_src = ExpertTransformer(
            cfg.N_CH, cfg.N_MODAL, cfg.WIN, d, cfg.N_HEAD, cfg.N_LAYER, cfg.D_FF,
            use_multi_mode)
        self.expert_tgt = ExpertTransformer(
            cfg.N_CH, cfg.N_MODAL, cfg.WIN, d, cfg.N_HEAD, cfg.N_LAYER, cfg.D_FF,
            use_multi_mode)

        # 交叉注意力与跨域融合 (Eq.23 / Eq.24)
        if use_ca:
            self.mhca_s2t = MHCA(d, cfg.N_HEAD)
            self.mhca_t2s = MHCA(d, cfg.N_HEAD)
            self.fuse_src = ADAP(d, cfg.D_FF, d)
            self.fuse_tgt = ADAP(d, cfg.D_FF, d)

        # Eq.26 聚合: CLS(源域) + GAP(目标域)
        self.agg_src = ADAP(d, cfg.D_FF, d)
        self.agg_tgt = ADAP(d, cfg.D_FF, d)

        # 张紧力回归头 (少样本标签监督, Eq.27 第二项)
        self.tension_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

        # 对齐编码重建头: 从融合编码 Z 重建关节电流的"本征一致性特征"
        # (Fig.10 可视化: 不同域的观测信号对齐到同一本征信号)
        self.recon_head = nn.Sequential(
            nn.Linear(d, cfg.D_FF * 2), nn.GELU(), nn.Linear(cfg.D_FF * 2, cfg.WIN))

        # 多步退化曲线预测头 (时序卷积, NPU 兼容)
        self.msp = MultiStepPredictor(d, cfg.PRED_H, cfg.SEQ_T)

    # ---------------- 单域前向 ----------------
    def _encode(self, expert: ExpertTransformer, x: torch.Tensor):
        tokens, modals = expert(x)
        return tokens, modals

    def _cross_fuse(self, y_s: torch.Tensor, y_t: torch.Tensor):
        """Eq.24: φ(Ye1,Ye2) = σ(MHCA(...))W_U 跨域特征融合"""
        if not self.use_ca:
            return y_s, y_t
        c_s2t = self.mhca_s2t(y_s, y_t)          # 源域Q, 目标域KV
        c_t2s = self.mhca_t2s(y_t, y_s)          # 目标域Q, 源域KV
        y_s = y_s + self.fuse_src(torch.tanh(c_t2s))
        y_t = y_t + self.fuse_tgt(torch.tanh(c_s2t))
        return y_s, y_t

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor):
        """
        x_s: (B, C, W) 源域窗口; x_t: (B, C, W) 目标域窗口
        返回: 预测张紧力(源/目标), 对齐编码, 模态协方差矩阵
        """
        tok_s, mod_s = self._encode(self.expert_src, x_s)
        tok_t, mod_t = self._encode(self.expert_tgt, x_t)

        tok_s, tok_t = self._cross_fuse(tok_s, tok_t)

        # Eq.26: Z = ADAP(CLS(Xs)) + ADAP(GAP(Xt))
        z_s = self.agg_src(tok_s[:, 0])
        z_t = self.agg_tgt(tok_t[:, 1:].mean(dim=1))
        z = z_s + z_t

        f_src = self.tension_head(z_s).squeeze(-1)
        f_tgt = self.tension_head(z_s + z_t).squeeze(-1)

        # 模态协方差 (Eq.16), 用于 PMA 损失; 单模态消融时以傅里叶模态复制补齐三元组
        if len(mod_s) == 1:
            Cs = modal_covariance(mod_s[0], mod_s[0], mod_s[0])
            Ct = modal_covariance(mod_t[0], mod_t[0], mod_t[0])
        else:
            Cs = modal_covariance(*mod_s)
            Ct = modal_covariance(*mod_t)
        return f_src, f_tgt, z, Cs, Ct, z_s, z_t

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        """从对齐编码重建本征一致性特征序列 (B, W)"""
        return self.recon_head(z)

    def predict_sequence(self, x_seq: torch.Tensor, domain: str = "tgt"):
        """
        多步时序预测: x_seq (B, T, C, W) -> 未来 H 步张紧力曲线
        """
        B, T = x_seq.shape[:2]
        zs = []
        expert = self.expert_tgt if domain == "tgt" else self.expert_src
        with torch.no_grad():
            for t in range(T):
                tokens, _ = expert(x_seq[:, t])
                zs.append(tokens[:, 0])
        z_seq = torch.stack(zs, dim=1)
        return self.msp(z_seq)


def compute_alignment_loss(Cs: torch.Tensor, Ct: torch.Tensor,
                           loss_type: str = "bures") -> torch.Tensor:
    """
    模态对齐损失的不同形式 (消融实验, Table 6):
      bures   : 完整 Eq.17 (Frobenius/Bures 范数, 含矩阵开平方)
      mse_cov : 不使用 F 范数 -> 协方差逐元素 MSE
      l1_seq  : 不使用协方差矩阵 -> 模态序列直接 L1 距离
    """
    if loss_type == "bures":
        return pma_loss(Cs, Ct)
    if loss_type == "mse_cov":
        return F.mse_loss(Cs, Ct)
    raise ValueError(loss_type)
