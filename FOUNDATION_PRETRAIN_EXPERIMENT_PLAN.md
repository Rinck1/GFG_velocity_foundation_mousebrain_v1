# GFG Velocity Foundation Pilot：12M 预训练与 MouseBrain 泛化实验

更新日期：2026-08-30（Asia/Shanghai）

## 1. 实验目标

本轮不是直接宣称得到最终 foundation model，而是验证以下三个关键问题：

1. 修正后的 GFG 在扩大容量后，能否稳定进行跨 accession 的 spliced/unspliced 自监督预训练。
2. 12M 人类语料上的预训练初始化，是否优于相同大模型在 MouseBrain 上从头训练。
3. 预训练阶段过早加入现有 GFG ODE/NTPL 约束，是否优于先做 masked denoising 再微调。

正式训练和测试使用独立目录，不修改稳定快照：

- 稳定快照：`/data/yuchang/GFG_graphbatch_directed_v1`
- 本轮目录：`/home/yuchang/GFG_velocity_foundation_mousebrain_v1`
- 预训练数据：`/data/dataset/Velocyto_pretrain_v1`
- 下游测试：`/data/yuchang/dataset/MouseBrain.h5ad`

## 2. 数据边界

- 预训练语料包含 2,018 个 H5AD、12,431,916 个细胞；本轮默认使用已转换的 3,000 基因轴。
- MouseBrain 包含 3,365 个细胞、936 个原始基因，现有 scVelo 流程过滤后约 759 个基因。
- 预训练为人类 Ensembl 轴，MouseBrain 为小鼠基因符号。3k 人类词表只能用符号近似映射到 331/936 个 MouseBrain 基因。因此本轮主结论采用 GFG 的逐基因共享参数进行跨基因迁移，并在完整 MouseBrain 上以相同流程微调；共同基因上的 zero-shot 结果仅作为附加诊断。
- 当前 12M split 仍是 accession numeric-block proxy，不是严格 study/donor split；本轮结论属于开发实验，不作为正式论文中的 leave-study-out 结论。

## 3. 模型与损失

保留修正版 GFG 的核心结构：

- manifold/state encoder；
- velocity/tangent encoder；
- decoder JVP/NTPL；
- soft VQ 接口保留，但首阶段默认旁路；
- MouseBrain 微调阶段的 graph-batch、无向平滑和有向方向对齐。

本轮扩大 `gene_dim`、逐基因 MLP 和 Transformer 深度，但参数仍在不同基因数之间共享，使 3,000 基因预训练权重可以加载到 MouseBrain 的过滤后基因轴。

预训练输入使用每个细胞、每个 layer 独立的 `log1p(CPM)`，避免跨 accession 的原始 library size 差异。遮罩同时覆盖零值与非零值，并包含整基因遮罩和单 layer 遮罩。

### Variant A：Denoising-first

预训练目标以 masked reconstruction 为主：

`L = L_masked + 0.1 * L_visible`

不在预训练初期施加强 ODE/graph 约束；MouseBrain 微调时再加入修正版 GFG 的完整 velocity 损失。

### Variant B：Denoising + weak kinetic

与 A 使用相同模型和数据流，额外逐步加入低权重 ODE/NTPL：

`L = L_masked + 0.1 * L_visible + ramp(t) * L_ODE`

该对照直接测试“先常规自监督”与“从预训练开始加入现有 GFG 动力学约束”的差别。

### Scratch control

使用与预训练模型完全相同的扩大架构，只在 MouseBrain 上从头训练，控制参数量、随机种子、微调轮数和图评测流程。

## 4. GPU 分配与执行顺序

| GPU | 任务 | 主要输出 |
| ---: | --- | --- |
| 6 | Variant A：masked denoising 预训练 → MouseBrain 微调/评测 | checkpoint、训练曲线、CBDir/ICCoh |
| 7 | Scratch control → Variant B：denoising + weak kinetic 预训练 → MouseBrain 微调/评测 | checkpoint、训练曲线、CBDir/ICCoh |

两个预训练任务使用相同 step budget。启动前自动做 batch-size 探测；在不触发 OOM 的前提下逐步放大 batch，并保留安全余量。吞吐量、峰值显存和实际读取细胞数写入结构化日志。

启动前 smoke test 发现原 Soft-VQ 在 3–4 步内从 128 个 code 塌缩到 perplexity 约 1，因此正式 pilot 旁路 VQ。scratch、Variant A、Variant B 和 MouseBrain 微调均使用同一旁路设置，保证架构公平；VQ 将在后续以 residual/EMA 形式单独消融。

## 5. 评测指标

### 预训练验证

- masked reconstruction MSE；
- visible reconstruction MSE；
- ODE residual（仅 B 作为优化项，A 只监控）；
- codebook perplexity/使用率；
- cells/s、steps/s、峰值显存。

### MouseBrain 泛化与迁移

- zero-shot reconstruction/ODE 诊断；
- 固定轮数全量微调后的 final loss；
- velocity confidence；
- ICCoh；
- CBDir without graph；
- CBDir with graph；
- graph improvement；
- 与历史修正版 1.59M GFG、相同架构 scratch control 比较。

CBDir 使用修正版真实 kNN 索引和独立 `velocity_graph_umap` 口径；旧版错误邻居索引的绝对值不参与比较。

## 6. 判定标准

预训练被认为有初步价值，需要同时满足：

1. 相同扩大架构下，预训练初始化的 MouseBrain 指标优于 scratch control，而不是只优于旧小模型的单个随机种子。
2. 提升不能只来自 velocity confidence；至少 CBDir with graph 或 ICCoh 中一个有稳定改善。
3. Variant B 若不优于 A，则下一轮采用 denoising-first，并重新设计层级 kinetic loss，而不是继续放大现有逐 cell×gene 闭式 ODE 权重。
4. 如果增大 batch 导致 CBDir 明显下降，则优先使用梯度累积扩大有效 batch，而不牺牲 graph batch 内的局部边覆盖。

## 7. 本轮不做的事情

- 不直接训练十亿参数模型；
- 不把 36,601 基因作为 dense Transformer 序列；
- 不把近似的人鼠符号映射包装成严格 ortholog zero-shot；
- 不修改或覆盖现有稳定 GFG 代码和历史 checkpoint；
- 不以单次 MouseBrain 随机种子结果作为最终泛化结论。

## 8. 后续扩展

若本轮证实预训练初始化有效，下一阶段再实现全 36,601 基因词表、稀疏 gene token、latent Transformer/Bi-Mamba 对照，以及 gene-level/数据集层级的 `beta/gamma` 动力学参数。
