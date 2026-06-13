"""
CrossDiseaseVAE --- 交互式演示网页
================================
真实集成：上传两个 h5ad 文件 -> 基因对齐 -> 加载模型 -> 提取 Z_inv
-> UMAP 降维 -> 三张验证图 -> 下载整合结果

运行方式: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import torch
import plotly.express as px
import tempfile
import shutil
import os

from model import CrossDiseaseVAE
from sklearn.preprocessing import LabelEncoder

# ====================================================================
# 页面配置
# ====================================================================
st.set_page_config(
    page_title="CrossDiseaseVAE · 单细胞整合演示",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>""",
    unsafe_allow_html=True,
)

# ====================================================================
# 辅助函数
# ====================================================================

@st.cache_resource
def load_model_checkpoint(checkpoint_path, n_genes, n_batches, n_conditions):
    """加载预训练模型"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = CrossDiseaseVAE(
        n_genes=n_genes,
        n_batches=n_batches,
        n_conditions=n_conditions,
        z_inv_dim=config.get("z_inv_dim", 32),
        z_spec_dim=config.get("z_spec_dim", 16),
        dropout=config.get("dropout", 0.2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def align_and_merge(adata_a, adata_b):
    """
    跨物种/跨数据集基因对齐：
    - 两个数据集的基因名统一转大写
    - 取交集基因
    - 合并为一个 AnnData
    """
    genes_a = [str(g).upper() for g in adata_a.var_names]
    genes_b = [str(g).upper() for g in adata_b.var_names]

    common = sorted(set(genes_a) & set(genes_b))

    if len(common) < 50:
        raise ValueError(
            f"两个数据集的共同基因过少 (仅 {len(common)} 个)，"
            "请确认上传的是基因表达矩阵，且物种一致或具有足够直系同源基因。"
        )

    def subset_to_common(adata, genes_upper, common_set):
        mask = [g in common_set for g in genes_upper]
        adata_sub = adata[:, mask].copy()
        adata_sub.var_names = [g for g, m in zip(genes_upper, mask) if m]
        adata_sub = adata_sub[:, ~adata_sub.var_names.duplicated()].copy()
        final_mask = [g in common_set for g in adata_sub.var_names]
        return adata_sub[:, final_mask].copy()

    adata_a_sub = subset_to_common(adata_a, genes_a, set(common))
    adata_b_sub = subset_to_common(adata_b, genes_b, set(common))

    final_genes = sorted(set(adata_a_sub.var_names) & set(adata_b_sub.var_names))
    adata_a_sub = adata_a_sub[:, final_genes].copy()
    adata_b_sub = adata_b_sub[:, final_genes].copy()

    adata_a_sub.obs["dataset"] = "Dataset_1"
    adata_b_sub.obs["dataset"] = "Dataset_2"

    merged = anndata.concat(
        [adata_a_sub, adata_b_sub],
        join="outer",
        index_unique="-",
        fill_value=0,
    )
    return merged, len(final_genes)


def extract_z_inv_batched(model, merged_adata, batch_size=256, status_placeholder=None):
    """分批提取 Z_inv，支持进度更新"""
    if hasattr(merged_adata.X, "toarray"):
        X = merged_adata.X.toarray().astype(np.float32)
    else:
        X = np.array(merged_adata.X, dtype=np.float32)

    condition_idx = merged_adata.obs["condition_idx"].values.astype(np.int64)
    n_cells = X.shape[0]

    X_tensor = torch.from_numpy(X)
    cond_tensor = torch.from_numpy(condition_idx)

    z_inv_all = np.zeros((n_cells, model.z_inv_dim), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n_cells, batch_size):
            end = min(i + batch_size, n_cells)
            z_inv_all[i:end] = model.get_z_inv(
                X_tensor[i:end], cond_tensor[i:end]
            )
            if status_placeholder is not None and (i // batch_size) % 5 == 0:
                status_placeholder.text(f"  推理中... {end}/{n_cells} 细胞")

    return z_inv_all


def compute_umap_plotly(z_inv, obs_df, color_col, title, subtitle):
    """计算 UMAP 并用 Plotly 绘制交互式散点图"""
    adata_tmp = anndata.AnnData(X=z_inv)
    adata_tmp.obsm["X_tmp"] = z_inv
    sc.pp.neighbors(adata_tmp, use_rep="X_tmp", random_state=42)
    sc.tl.umap(adata_tmp, random_state=42)

    df = pd.DataFrame({
        "UMAP_1": adata_tmp.obsm["X_umap"][:, 0],
        "UMAP_2": adata_tmp.obsm["X_umap"][:, 1],
        color_col: obs_df[color_col].values,
    })

    # 不同图用不同配色
    if color_col == "batch":
        palette = px.colors.qualitative.Plotly
    elif color_col == "dataset":
        palette = px.colors.qualitative.Set1
    else:
        palette = px.colors.qualitative.Set2

    fig = px.scatter(
        df,
        x="UMAP_1",
        y="UMAP_2",
        color=color_col,
        color_discrete_sequence=palette,
        width=750,
        height=550,
        template="plotly_white",
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:13px;color:#888;'>{subtitle}</span>",
            x=0.5,
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, title_text="UMAP 1")
    fig.update_yaxes(showgrid=False, zeroline=False, title_text="UMAP 2")

    return fig


# ====================================================================
# 页面渲染 - 标题
# ====================================================================
_, center, _ = st.columns([1, 3, 1])
with center:
    st.markdown(
        "<h1 style='text-align:center; margin-bottom:0;'>🧬 CrossDiseaseVAE</h1>"
        "<p style='text-align:center; font-size:18px; color:#555;'>"
        "跨疾病条件单细胞测序数据整合算法</p>"
        "<p style='text-align:center; font-size:14px; color:#999;'>"
        "上传两个单细胞数据集 (.h5ad)，一键完成批次效应消除与生物学特征保留验证</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

# ====================================================================
# 文件上传区
# ====================================================================
_, up_center, _ = st.columns([1, 3, 1])
with up_center:
    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader(
            "细胞数据集 1",
            type=["h5ad"],
            key="f1",
        )
    with col2:
        file2 = st.file_uploader(
            "细胞数据集 2",
            type=["h5ad"],
            key="f2",
        )

    repo_root = os.path.dirname(__file__) or "."
    checkpoint_path = os.path.join(repo_root, "cross_disease_vae_final.pt")
    has_checkpoint = os.path.exists(checkpoint_path)

    if not has_checkpoint:
        # 调试信息：列出实际存在的文件，帮助定位问题
        import glob
        repo_files = sorted(glob.glob(os.path.join(repo_root, "**", "*"), recursive=True))
        file_list = "\n".join(repo_files[:50]) if repo_files else "(空目录)"
        st.warning(
            f"未找到预训练模型\n\n"
            f"期望路径: `{checkpoint_path}`\n\n"
            f"仓库根目录 `{repo_root}` 下实际文件 (前 50 个):\n```\n{file_list}\n```"
        )

# ====================================================================
# 处理流程 (两个文件都上传后自动触发)
# ====================================================================
if file1 is not None and file2 is not None:

    tmp_dir = tempfile.mkdtemp()
    tmp1 = os.path.join(tmp_dir, "dataset1.h5ad")
    tmp2 = os.path.join(tmp_dir, "dataset2.h5ad")
    with open(tmp1, "wb") as f:
        f.write(file1.getbuffer())
    with open(tmp2, "wb") as f:
        f.write(file2.getbuffer())

    status = st.empty()

    try:
        # Step 1: 加载数据
        with st.spinner("Step 1/5: 加载数据..."):
            status.info("正在读取两个 .h5ad 文件...")
            adata1 = sc.read_h5ad(tmp1)
            adata2 = sc.read_h5ad(tmp2)
            st.success(
                f"数据已加载: Dataset1 = {adata1.shape[0]:,} cells x {adata1.shape[1]:,} genes  |  "
                f"Dataset2 = {adata2.shape[0]:,} cells x {adata2.shape[1]:,} genes"
            )

        # Step 2: 基因对齐
        with st.spinner("Step 2/5: 跨数据集基因对齐..."):
            status.info("正在进行基因名统一与交集提取...")
            merged, n_genes = align_and_merge(adata1, adata2)
            st.success(
                f"基因对齐完成: {n_genes:,} 共同基因  |  "
                f"合并后 {merged.shape[0]:,} 细胞"
            )

        # Step 3: 标签编码
        with st.spinner("Step 3/5: 标签编码..."):
            for col in ["batch", "condition"]:
                if col not in merged.obs.columns:
                    st.error(
                        f"数据缺少 `{col}` 列! "
                        "请确保 h5ad 的 .obs 中包含 `batch` 和 `condition` 字段。"
                    )
                    st.stop()

            batch_enc = LabelEncoder()
            cond_enc = LabelEncoder()
            merged.obs["batch_idx"] = batch_enc.fit_transform(
                merged.obs["batch"].astype(str)
            )
            merged.obs["condition_idx"] = cond_enc.fit_transform(
                merged.obs["condition"].astype(str)
            )
            n_batches = len(batch_enc.classes_)
            n_conditions = len(cond_enc.classes_)
            st.success(
                f"编码完成: Batch={n_batches} 类  |  Condition={n_conditions} 类"
            )

        # Step 4: 加载模型 + 推理
        with st.spinner("Step 4/5: 加载模型并提取 Z_inv 表征..."):
            status.info("正在加载 CrossDiseaseVAE 模型...")

            # 维度兼容性检查
            if has_checkpoint:
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                ckpt_n_genes = ckpt["model_state_dict"]["encoder.fc1.weight"].shape[1]

                if ckpt_n_genes != n_genes:
                    st.error(
                        f"模型基因数 ({ckpt_n_genes}) 与上传数据的共同基因数 ({n_genes}) 不匹配!\n\n"
                        f"当前上传的数据集共同基因数为 {n_genes}，"
                        f"但预训练模型期望 {ckpt_n_genes} 个基因。"
                        f"请使用与训练时相同的数据集文件。"
                    )
                    st.stop()

            model = load_model_checkpoint(
                checkpoint_path, n_genes, n_batches, n_conditions
            )

            status_proxy = st.empty()
            z_inv = extract_z_inv_batched(
                model, merged, status_placeholder=status_proxy
            )
            status_proxy.empty()
            st.success(
                f"Z_inv 提取完成: {z_inv.shape[1]} 维 x {z_inv.shape[0]:,} 细胞"
            )

        # Step 5: UMAP + 三张验证图
        with st.spinner("Step 5/5: UMAP 降维并生成三张验证图..."):
            status.info("正在计算 UMAP (可能 1-3 分钟)...")

            # 先跑一次 UMAP 存入 merged，避免 compute_umap_plotly 里重复跑三次
            adata_tmp = anndata.AnnData(X=z_inv)
            adata_tmp.obsm["X_tmp"] = z_inv
            sc.pp.neighbors(adata_tmp, use_rep="X_tmp", random_state=42)
            sc.tl.umap(adata_tmp, random_state=42)
            umap_coords = adata_tmp.obsm["X_umap"]
            merged.obsm["X_umap"] = umap_coords

            # --- 图 1: 按 Batch 着色 ---
            df_batch = pd.DataFrame({
                "UMAP_1": umap_coords[:, 0],
                "UMAP_2": umap_coords[:, 1],
                "Batch": merged.obs["batch"].values,
            })
            fig_batch = px.scatter(
                df_batch, x="UMAP_1", y="UMAP_2", color="Batch",
                color_discrete_sequence=px.colors.qualitative.Plotly,
                width=750, height=550, template="plotly_white",
            )
            fig_batch.update_traces(marker=dict(size=4, opacity=0.7))
            fig_batch.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(l=20, r=20, t=40, b=20),
                title=dict(
                    text="<b>图 1: 批次效应去除验证</b><br>"
                    "<span style='font-size:13px;color:#888;'>"
                    "不同批次是否像芝麻一样均匀撒满全图？越均匀 = 批次效应去除越彻底</span>",
                    x=0.5,
                ),
            )
            fig_batch.update_xaxes(showgrid=False, zeroline=False, title_text="UMAP 1")
            fig_batch.update_yaxes(showgrid=False, zeroline=False, title_text="UMAP 2")

            # --- 图 2: 按 Dataset 着色 ---
            df_dataset = pd.DataFrame({
                "UMAP_1": umap_coords[:, 0],
                "UMAP_2": umap_coords[:, 1],
                "Dataset": merged.obs["dataset"].values,
            })
            fig_dataset = px.scatter(
                df_dataset, x="UMAP_1", y="UMAP_2", color="Dataset",
                color_discrete_sequence=px.colors.qualitative.Set1,
                width=750, height=550, template="plotly_white",
            )
            fig_dataset.update_traces(marker=dict(size=4, opacity=0.7))
            fig_dataset.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(l=20, r=20, t=40, b=20),
                title=dict(
                    text="<b>图 2: 跨疾病生物特征保留验证</b><br>"
                    "<span style='font-size:13px;color:#888;'>"
                    "两个数据集是否各自成团？分得开 = 两组数据的真实生物学差异被完整保留</span>",
                    x=0.5,
                ),
            )
            fig_dataset.update_xaxes(showgrid=False, zeroline=False, title_text="UMAP 1")
            fig_dataset.update_yaxes(showgrid=False, zeroline=False, title_text="UMAP 2")

            # --- 图 3: 按 Condition 着色 ---
            df_cond = pd.DataFrame({
                "UMAP_1": umap_coords[:, 0],
                "UMAP_2": umap_coords[:, 1],
                "Condition": merged.obs["condition"].values,
            })
            fig_cond = px.scatter(
                df_cond, x="UMAP_1", y="UMAP_2", color="Condition",
                color_discrete_sequence=px.colors.qualitative.Set2,
                width=750, height=550, template="plotly_white",
            )
            fig_cond.update_traces(marker=dict(size=4, opacity=0.7))
            fig_cond.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(l=20, r=20, t=40, b=20),
                title=dict(
                    text="<b>图 3: Z_inv 条件不变性验证</b><br>"
                    "<span style='font-size:13px;color:#888;'>"
                    "Z_inv 对 condition 无偏——同类型细胞不会因 control/disease 标签被人为割裂</span>",
                    x=0.5,
                ),
            )
            fig_cond.update_xaxes(showgrid=False, zeroline=False, title_text="UMAP 1")
            fig_cond.update_yaxes(showgrid=False, zeroline=False, title_text="UMAP 2")

            status.empty()
            st.success("UMAP 计算完成")

        # ================================================================
        # 结果展示
        # ================================================================
        st.markdown("<br>", unsafe_allow_html=True)
        _, viz_center, _ = st.columns([1, 4, 1])
        with viz_center:
            st.plotly_chart(fig_batch, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.plotly_chart(fig_dataset, use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.plotly_chart(fig_cond, use_container_width=True)

            # 解读指南
            st.markdown(
                "<div style='padding:20px; background:#fafafa; border-radius:8px; "
                "border:1px solid #e0e0e0; font-size:14px; line-height:1.8;'>"
                "<b style='font-size:16px;'>三张图如何一起读</b><br><br>"
                "<b>图 1</b> 看颜色是否像芝麻撒匀——这是 <b>\"去批次\"</b> 的证明。<br>"
                "<b>图 2</b> 看两个数据集是否各占一方——这是 <b>\"保留生物学差异\"</b> 的证明。"
                "两种完全不同的疾病本来就不该混在一起。<br>"
                "<b>图 3</b> 是为了证明模型 <u>没有</u> 按 condition 标签强行分离细胞——"
                "Z_inv 呈现的是细胞自身的身份（成纤维细胞、T 细胞...），"
                "同一类型细胞无论来自 control 还是 disease 都会聚在一起。"
                "<br><br>"
                "<b>结论：</b>图 1 \"乱\"（批次混了）+ 图 2 \"不乱\""
                "（两种病各自成团）+ 图 3 \"自然\"（condition 不割裂细胞类型）"
                "= 完美平衡 <b>\"去批次噪音\"</b> 与 <b>\"留生物价值\"</b>。"
                "</div>",
                unsafe_allow_html=True,
            )

            # 下载
            st.markdown("<br>", unsafe_allow_html=True)

            out_adata = merged.copy()
            out_adata.obsm["X_crossdisease_vae"] = z_inv

            # 修复: pandas ArrowStringArray 不被 h5ad 支持, 转为普通 object 类型
            for col in out_adata.obs.columns:
                if out_adata.obs[col].dtype == "string" or "ArrowStringArray" in str(type(out_adata.obs[col].values)):
                    out_adata.obs[col] = out_adata.obs[col].astype(object)

            out_tmp = os.path.join(tmp_dir, "integrated_output.h5ad")
            out_adata.write(out_tmp, compression="gzip")

            with open(out_tmp, "rb") as f_out:
                st.download_button(
                    label="下载整合后的数据 (integrated_results.h5ad)",
                    data=f_out,
                    file_name="integrated_results.h5ad",
                    mime="application/octet-stream",
                )

            st.caption(
                "下载的 .h5ad 包含: 原始基因表达 + "
                f".obsm['X_crossdisease_vae'] ({z_inv.shape[1]} 维整合特征) + "
                "UMAP 坐标，可直接用于下游分析。"
            )

    except Exception as e:
        status.empty()
        st.error(f"处理出错: {e}")
        import traceback
        with st.expander("详细错误信息"):
            st.code(traceback.format_exc())

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
