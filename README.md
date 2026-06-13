# 🧬 CrossDiseaseVAE

> **跨疾病条件单细胞测序数据整合算法**  
> Cross-condition Single-cell RNA-seq Integration via Disentangled Variational Autoencoder

[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://share.streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)

---

## 🎯 项目简介

单细胞测序数据整合面临一个核心矛盾：**消除批次效应的同时，必须保留真实的生物学差异**。

CrossDiseaseVAE 设计了一个解耦变分自编码器（Disentangled VAE），将基因表达谱分解为两个隐空间：

| 隐变量 | 维度 | 含义 |
|--------|------|------|
| **Z_inv** (condition-invariant) | 32 维 | 与实验条件无关的细胞固有特征 → 批次校正后的整合表示 |
| **Z_spec** (condition-specific) | 16 维 | 与疾病/处理条件相关的特异性变化 |

通过**对抗域自适应**（Gradient Reversal Layer + Discriminator），迫使 Z_inv 对 batch/condition 标签不可区分，同时保留细胞的生物学身份。

---

## 🏗️ 模型架构

```
┌──────────────────────────────────────────────────┐
│                  Encoder (MLP)                    │
│         gene_expr → shared → ┬→ Z_inv (32d)     │
│                              └→ Z_spec (16d)     │
└──────────────────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
        ┌─────▼─────┐           ┌─────▼─────┐
        │  Decoder   │           │Discriminator│
        │ Z_inv+Z_spec│          │  Z_inv →    │
        │ → recon_x  │           │condition/batch│
        └────────────┘           │  label?    │
              │                 └─────────────┘
              │                       │
         Reconstruction          Adversarial
            Loss                  Loss (GRL)
```

**关键设计**：Discriminator 试图从 Z_inv 预测条件标签——但 **Gradient Reversal Layer 反转梯度**，Encoder 被迫学习让 Z_inv 对条件标签"不可区分"的表示，从而实现条件不变性。

---

## 🚀 在线体验

### 一键启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动网页
streamlit run app.py
```

### 使用方式

1. 打开浏览器中的本地地址（默认 `http://localhost:8501`）
2. 上传两个 `.h5ad` 文件（Dataset 1: 伤口/species A，Dataset 2: 黑色素瘤/species B）
3. 等待 5 步处理流水线自动完成：
   - **Step 1** → 数据加载
   - **Step 2** → 跨数据集基因对齐（基因名统一大写 → 取交集）
   - **Step 3** → 标签编码
   - **Step 4** → 加载预训练模型 + 提取 Z_inv
   - **Step 5** → UMAP 降维 + 生成三张验证图
4. 查看结果并下载整合数据

> **注意**：上传的 h5ad 文件要求 `.obs` 中包含 `batch` 和 `condition` 列。

---

## 📊 三图验证框架

| 图 | 着色依据 | 期望结果 | 证明内容 |
|----|---------|---------|---------|
| **图 1** | Batch | 颜色如芝麻撒匀 ✨ | 批次效应已被消除 |
| **图 2** | Dataset | 两种疾病各聚一方 🧫 | 生物学差异被完整保留 |
| **图 3** | Condition | 同类型细胞不因标签割裂 🧩 | Z_inv 做到了条件不变性 |

> **结题答辩一句话结论**：图 1 "乱"（批次混合） + 图 2 "不乱"（疾病分群） + 图 3 "自然"（condition 不人为割裂）= 算法在 **"去噪音"** 与 **"留生物"** 之间取得了平衡。

---

## 📁 项目结构

```
CrossDiseaseVAE/
├── app.py                          ← Streamlit 网页主程序
├── model.py                        ← VAE 模型定义
├── requirements.txt                ← Python 依赖
├── README.md                       ← 本文件
└── output/
    └── cross_disease_vae_final.pt  ← 预训练模型权重 (~780KB)
```

完整项目（含训练脚本、对比实验、下游分析）详见本地工程目录。

---

## 🔧 技术栈

- **深度学习**: PyTorch, Gradient Reversal Layer, Adversarial Domain Adaptation
- **单细胞**: Scanpy, AnnData
- **可视化**: Plotly (interactive scatter), UMAP
- **部署**: Streamlit + Streamlit Cloud

---

## 👤 作者

**郝思喆** — 哈尔滨工业大学 基础学部  
大一年度项目 · 结题答辩  
2025–2026
