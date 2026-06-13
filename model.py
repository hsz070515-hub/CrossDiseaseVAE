"""
跨疾病单细胞数据整合模型 -- CrossDiseaseVAE
===========================================
基于变分自编码器（VAE）的多任务深度学习架构，实现：
  1. Z_inv（条件不变表征）：刻画跨疾病的共性生物学特征（去批次效应后的纯净细胞状态）
  2. Z_spec（条件特异表征）：捕捉不同疾病或条件下的特异性差异信号
  3. 对抗训练（Adversarial Training）：通过判别器迫使 Z_inv 无法区分批次来源

理论创新点（用于论文）：
  - 双分支潜空间解耦：将细胞状态分解为疾病共享和疾病特异两个正交子空间
  - 梯度反转层（Gradient Reversal Layer, GRL）：端到端对抗训练，消除 Z_inv 中的批次信息
  - 条件嵌入（Condition Embedding）：将生物学条件注入解码器，引导模型理解疾病差异

参考文献：
  - inVAE (Aliee et al., bioRxiv 2024) -- 条件不变 VAE 架构
  - Domain-Adversarial Neural Networks (Ganin et al., JMLR 2016) -- GRL 理论
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# 梯度反转层（Gradient Reversal Layer, GRL）
# ============================================================
# 理论依据：Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016
# 前向传播：恒等映射 f(x) = x
# 反向传播：梯度乘以 -λ（反转方向），迫使上游网络学习"欺骗"下游判别器
#
# 在本项目中的作用：
#   编码器生成 Z_inv -> GRL -> 判别器预测 batch
#   GRL 反转梯度 -> 编码器被迫生成 Z_inv 使得判别器无法识别 batch
#   -> Z_inv 中的批次信息被消除，只保留共享生物学信号

class GradientReversalLayer(torch.autograd.Function):
    """
    梯度反转层的 autograd 实现。
    前向：y = x（恒等）
    反向：∂L/∂x = -λ * ∂L/∂y（梯度反转并缩放）
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        """
        前向传播：直接传递输入，不做任何变换

        Parameters
        ----------
        x : Tensor
            输入张量（此处为 Z_inv）
        lambda_ : float
            梯度反转系数，随训练进程从 0 增长到 1（渐进式对抗训练）
        """
        ctx.lambda_ = lambda_  # 保存 λ 供 backward 使用
        return x.view_as(x)   # 恒等映射

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：将上游梯度取反并缩放
        ∂L/∂x = -λ * ∂L/∂y

        这是对抗训练的核心数学操作：
        - 判别器想最小化 batch 分类误差 -> 梯度指向"更好识别 batch"的方向
        - GRL 反转梯度 -> 编码器收到"反方向"的信号
        - 编码器因此学会"消除 batch 信息"，使得判别器无法区分
        """
        return grad_output.neg() * ctx.lambda_, None  # neg() = 取反


def grad_reverse(x, lambda_=1.0):
    """GRL 的函数式接口，方便在 forward 中调用"""
    return GradientReversalLayer.apply(x, lambda_)


# ============================================================
# Encoder（编码器）-- 双分支潜空间
# ============================================================
# 架构说明：
#   输入：细胞的基因表达向量 x ∈ R^{n_genes}
#   共享部分：两层 MLP -> 分叉为四个头（μ_inv, logvar_inv, μ_spec, logvar_spec）
#
#   【Z_inv 分支】-- 条件不变表征
#     目的：捕获所有疾病条件下共享的生物学模式（如基础细胞类型、通用代谢通路）
#     维度：z_inv_dim（默认 32）
#     正则化：KL 散度约束 + 对抗训练（消除批次信息）
#
#   【Z_spec 分支】-- 条件特异表征
#     目的：捕获特定疾病条件下的特异性信号（如疾病相关基因表达变化）
#     维度：z_spec_dim（默认 16）
#     正则化：KL 散度约束
#
#   Z_inv 和 Z_spec 通过正交惩罚（见 losses.py）确保解耦，不编码重复信息

class Encoder(nn.Module):
    """
    VAE 编码器：将高维基因表达数据压缩到双潜空间。

    网络结构：
        input_dim -> 256 -> BatchNorm -> ReLU -> Dropout(0.2)
                 -> 128 -> BatchNorm -> ReLU
                 -> μ_inv (32维), logvar_inv (32维)   ← Z_inv 分支
                 -> μ_spec (16维), logvar_spec (16维)  ← Z_spec 分支
    """

    def __init__(self, input_dim, z_inv_dim=32, z_spec_dim=16, dropout=0.2):
        """
        Parameters
        ----------
        input_dim : int
            输入基因数（交集基因数）
        z_inv_dim : int
            Z_inv 潜空间维度（条件不变表征）。设较大以捕获丰富的共享生物学信息
        z_spec_dim : int
            Z_spec 潜空间维度（条件特异表征）。设较小以避免编码过多噪声
        dropout : float
            Dropout 比例，防止过拟合
        """
        super(Encoder, self).__init__()

        # ---- 共享特征提取层 ----
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)

        # ---- Z_inv 分支（条件不变表征）----
        # μ_inv：Z_inv 的均值向量，代表细胞在共享空间中的位置
        self.fc_mu_inv = nn.Linear(128, z_inv_dim)
        # logvar_inv：Z_inv 的对数方差，用于重参数化技巧和 KL 散度计算
        self.fc_logvar_inv = nn.Linear(128, z_inv_dim)

        # ---- Z_spec 分支（条件特异表征）----
        # μ_spec：Z_spec 的均值向量，代表细胞在特异空间中的位置
        self.fc_mu_spec = nn.Linear(128, z_spec_dim)
        # logvar_spec：Z_spec 的对数方差
        self.fc_logvar_spec = nn.Linear(128, z_spec_dim)

        # 保存维度信息
        self.z_inv_dim = z_inv_dim
        self.z_spec_dim = z_spec_dim

    def forward(self, x):
        """
        前向传播：x -> 共享层 -> μ_inv, logvar_inv, μ_spec, logvar_spec

        Parameters
        ----------
        x : Tensor, shape [batch_size, n_genes]
            基因表达向量（log-normalized）

        Returns
        -------
        mu_inv : Tensor, shape [batch_size, z_inv_dim]
            Z_inv 的均值
        logvar_inv : Tensor, shape [batch_size, z_inv_dim]
            Z_inv 的对数方差
        mu_spec : Tensor, shape [batch_size, z_spec_dim]
            Z_spec 的均值
        logvar_spec : Tensor, shape [batch_size, z_spec_dim]
            Z_spec 的对数方差
        """
        # 共享特征提取
        h = self.fc1(x)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout1(h)

        h = self.fc2(h)
        h = self.bn2(h)
        h = F.relu(h)

        # 双分支分叉：
        # ┌─ Z_inv：捕获跨条件的不变生物学特征（去除批次效应）
        # └─ Z_spec：捕获特定条件的差异信号（疾病特异性）
        mu_inv = self.fc_mu_inv(h)
        logvar_inv = self.fc_logvar_inv(h)
        mu_spec = self.fc_mu_spec(h)
        logvar_spec = self.fc_logvar_spec(h)

        return mu_inv, logvar_inv, mu_spec, logvar_spec


# ============================================================
# Decoder（解码器）-- 条件重建
# ============================================================
# 输入：z_inv（共享信号）+ z_spec（特异信号）+ condition（条件嵌入）
# 输出：重建的基因表达向量
#
# 设计原理：
#   解码器需要同时知道：
#     1. 细胞的"身份"（Z_inv -- 它是什么类型的细胞）
#     2. 细胞的"状态"（Z_spec -- 它在特定疾病条件下有什么变化）
#     3. 细胞的"背景"（condition -- 它来自哪种疾病条件）
#   这三个信息共同决定了细胞的基因表达模式。

class Decoder(nn.Module):
    """
    VAE 解码器：从潜空间和条件标签重建基因表达。

    网络结构：
        [z_inv, z_spec, condition_onehot] -> 128 -> BatchNorm -> ReLU
                                         -> 256 -> BatchNorm -> ReLU
                                         -> n_genes -> Softplus
    """

    def __init__(self, z_inv_dim=32, z_spec_dim=16, n_conditions=5, output_dim=2000):
        """
        Parameters
        ----------
        z_inv_dim : int
            Z_inv 维度
        z_spec_dim : int
            Z_spec 维度
        n_conditions : int
            生物学条件类别总数（用于条件嵌入）
        output_dim : int
            输出基因数（= 输入基因数）
        """
        super(Decoder, self).__init__()

        # 解码器输入 = Z_inv + Z_spec + 条件 one-hot
        self.input_dim = z_inv_dim + z_spec_dim + n_conditions

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)

        # 输出层：Softplus 激活确保非负（基因表达值 ≥ 0）
        self.fc_out = nn.Linear(256, output_dim)
        self.softplus = nn.Softplus()

        self.n_conditions = n_conditions

    def forward(self, z_inv, z_spec, condition_labels):
        """
        前向传播：潜变量 + 条件 -> 重建基因表达

        Parameters
        ----------
        z_inv : Tensor, shape [batch_size, z_inv_dim]
            条件不变表征
        z_spec : Tensor, shape [batch_size, z_spec_dim]
            条件特异表征
        condition_labels : Tensor, shape [batch_size]
            条件整数标签（0 到 n_conditions-1）

        Returns
        -------
        x_recon : Tensor, shape [batch_size, n_genes]
            重建的基因表达向量（非负，通过 Softplus 保证）
        """
        # 将条件标签转为 one-hot 编码，注入解码器
        condition_onehot = F.one_hot(condition_labels, num_classes=self.n_conditions).float()

        # 拼接三个信息源：
        # Z_inv（不变表征）+ Z_spec（特异表征）+ condition（条件背景）
        combined = torch.cat([z_inv, z_spec, condition_onehot], dim=1)

        h = self.fc1(combined)
        h = self.bn1(h)
        h = F.relu(h)

        h = self.fc2(h)
        h = self.bn2(h)
        h = F.relu(h)

        # 输出层用 Softplus 保证正值（基因表达量 ≥ 0）
        x_recon = self.softplus(self.fc_out(h))

        return x_recon


# ============================================================
# Discriminator（判别器）-- 批次分类器
# ============================================================
# 输入：Z_inv（条件不变表征）
# 输出：batch 来源预测（多分类）
#
# 对抗训练机制：
#   - 判别器的目标：准确识别每个细胞来自哪个 batch
#   - 编码器的目标（通过 GRL）：让判别器无法识别，即 Z_inv 不含批次信息
#   - 训练结果：Z_inv 中只保留与 batch 无关的生物学信息

class Discriminator(nn.Module):
    """
    批次判别器：从 Z_inv 预测细胞的 batch 来源。

    网络结构：
        Z_inv -> 64 -> ReLU -> n_batches -> Softmax

    训练时通过 GRL 与编码器对抗：
    - 判别器拼命想从 Z_inv 中读出 batch 信息
    - 编码器拼命消除 Z_inv 中的 batch 信息
    - 最终 Z_inv 中的批次效应被消除
    """

    def __init__(self, z_inv_dim=32, n_batches=10):
        """
        Parameters
        ----------
        z_inv_dim : int
            Z_inv 维度
        n_batches : int
            batch 类别总数
        """
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(z_inv_dim, 64)
        self.fc_out = nn.Linear(64, n_batches)

        self.n_batches = n_batches

    def forward(self, z_inv):
        """
        前向传播：Z_inv -> batch 预测

        Parameters
        ----------
        z_inv : Tensor, shape [batch_size, z_inv_dim]
            条件不变表征（来自编码器，经过 GRL 后输入）

        Returns
        -------
        batch_logits : Tensor, shape [batch_size, n_batches]
            每个 batch 类别的预测 logits
        """
        h = F.relu(self.fc1(z_inv))
        batch_logits = self.fc_out(h)
        return batch_logits


# ============================================================
# CrossDiseaseVAE（主模型）-- 完整的跨疾病整合框架
# ============================================================
# 组合 Encoder + Decoder + Discriminator，实现：
#   1. 数据压缩到双潜空间（Z_inv + Z_spec）
#   2. 从潜空间重建基因表达
#   3. 对抗训练消除 Z_inv 中的批次信息
#
# 重参数化技巧（Reparameterization Trick）：
#   z = μ + ε * exp(0.5 * logvar),  ε ~ N(0, I)
#   这使得采样操作可微，允许梯度通过随机节点反向传播

class CrossDiseaseVAE(nn.Module):
    """
    跨疾病条件单细胞数据整合模型（CrossDiseaseVAE / CondiModule）

    理论框架（用于结题报告）：
    ┌─────────────────────────────────────────────────────┐
    │  x (基因表达)                                         │
    │    ↓                                                │
    │  Encoder (共享层 + 双分叉)                             │
    │    ↓                        ↓                       │
    │  Z_inv (条件不变)          Z_spec (条件特异)           │
    │    ↓ 对抗训练                ↓                        │
    │  Discriminator              │                       │
    │  (被迫无法识别batch)         │                       │
    │    ↓                        ↓                       │
    │  [Z_inv, Z_spec, condition] -> Decoder -> x_recon      │
    └─────────────────────────────────────────────────────┘

    损失函数（详见 losses.py）：
      L_total = L_recon + β·L_kl + λ·L_adv + γ·L_orth
    """

    def __init__(self, n_genes, n_batches, n_conditions,
                 z_inv_dim=32, z_spec_dim=16, dropout=0.2):
        """
        Parameters
        ----------
        n_genes : int
            输入基因数
        n_batches : int
            batch 类别数（判别器输出维度）
        n_conditions : int
            condition 类别数（解码器条件输入维度）
        z_inv_dim : int
            Z_inv 潜空间维度（默认 32，捕获共享生物学特征）
        z_spec_dim : int
            Z_spec 潜空间维度（默认 16，捕获疾病特异特征）
        dropout : float
            Dropout 比例
        """
        super(CrossDiseaseVAE, self).__init__()

        self.z_inv_dim = z_inv_dim
        self.z_spec_dim = z_spec_dim
        self.n_genes = n_genes
        self.n_batches = n_batches
        self.n_conditions = n_conditions

        # 子模块
        self.encoder = Encoder(n_genes, z_inv_dim, z_spec_dim, dropout)
        self.decoder = Decoder(z_inv_dim, z_spec_dim, n_conditions, n_genes)
        self.discriminator = Discriminator(z_inv_dim, n_batches)

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧（Reparameterization Trick）

        从 N(μ, σ²) 采样 z，但使采样过程可微：
            z = μ + ε * σ,  ε ~ N(0, I)
        其中 σ = exp(0.5 * logvar)

        这一步是 VAE 能端到端训练的关键：
        - 前向：生成随机潜变量 z
        - 反向：梯度绕开随机采样，直接通过 μ 和 σ 传播

        Parameters
        ----------
        mu : Tensor
            均值向量
        logvar : Tensor
            对数方差向量

        Returns
        -------
        z : Tensor
            采样后的潜变量
        """
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5 * logvar)
        eps = torch.randn_like(std)     # ε ~ N(0, I)
        return mu + eps * std           # z = μ + ε * σ

    def forward(self, x, condition_labels, grl_lambda=1.0):
        """
        完整前向传播。

        Parameters
        ----------
        x : Tensor, shape [batch_size, n_genes]
            输入基因表达向量
        condition_labels : Tensor, shape [batch_size]
            条件整数标签（0 到 n_conditions-1）
        grl_lambda : float
            梯度反转系数（0=无对抗, 1=全对抗）

        Returns
        -------
        x_recon : Tensor
            重建的基因表达
        z_inv : Tensor
            条件不变表征（去批次效应的纯净细胞状态）
        z_spec : Tensor
            条件特异表征（疾病特异性差异信号）
        mu_inv, logvar_inv : Tensor
            Z_inv 的分布参数（用于 KL 散度）
        mu_spec, logvar_spec : Tensor
            Z_spec 的分布参数（用于 KL 散度）
        batch_logits : Tensor
            判别器预测的 batch logits（用于对抗损失）
        """
        # Step 1: 编码 -- 将高维基因表达压缩到双潜空间
        mu_inv, logvar_inv, mu_spec, logvar_spec = self.encoder(x)

        # Step 2: 重参数化 -- 从分布中采样潜变量（可微操作）
        z_inv = self.reparameterize(mu_inv, logvar_inv)
        z_spec = self.reparameterize(mu_spec, logvar_spec)

        # Step 3: 解码 -- 从潜空间和条件标签重建基因表达
        x_recon = self.decoder(z_inv, z_spec, condition_labels)

        # Step 4: 对抗判别 -- 梯度反转后让判别器尝试预测 batch
        #          GRL 使得编码器收到"反方向"的梯度，被迫消除 Z_inv 中的批次信息
        z_inv_reversed = grad_reverse(z_inv, lambda_=grl_lambda)
        batch_logits = self.discriminator(z_inv_reversed)

        return x_recon, z_inv, z_spec, mu_inv, logvar_inv, mu_spec, logvar_spec, batch_logits

    @torch.no_grad()
    def get_z_inv(self, x, condition_labels):
        """
        提取条件不变表征 Z_inv（评估/可视化专用，不需要梯度）

        Z_inv 是本项目的核心输出：
        - 消除了批次效应的纯净细胞表征
        - 可用于下游分析：聚类、可视化、标签转移、跨疾病比较

        Parameters
        ----------
        x : Tensor
            基因表达向量
        condition_labels : Tensor
            条件标签

        Returns
        -------
        z_inv : ndarray
            Z_inv 潜变量（NumPy 数组，方便存入 AnnData）
        """
        self.eval()
        mu_inv, logvar_inv, mu_spec, logvar_spec = self.encoder(x)
        z_inv = self.reparameterize(mu_inv, logvar_inv)
        return z_inv.cpu().numpy()
