# GFG Velocity Foundation Pilot 实验总结

## 1. 实验目标

本轮实验的目标是评估：在修正后的 GFG 代码基础上，将模型扩展到约 1,460 万可训练参数，并使用约 1,243 万细胞的 Velocyto 语料进行预训练，能否获得可迁移到未见数据集的 RNA velocity 表征。

下游泛化测试统一使用 MouseBrain 数据集。实验重点比较三条路线：

1. 大模型不预训练，直接在 MouseBrain 上训练；
2. 只做 masked denoising 预训练，再进行零样本测试或下游微调；
3. masked denoising 加弱 kinetic/ODE 约束预训练，再进行零样本测试或下游微调。

本文件总结实际完成的实验、验证结果、主要结论及下一步建议。详细方案和原始结果分别见：

- [FOUNDATION_PRETRAIN_EXPERIMENT_PLAN.md](./FOUNDATION_PRETRAIN_EXPERIMENT_PLAN.md)
- [FINAL_RESULTS.md](./experiments/foundation_pilot_20260830/FINAL_RESULTS.md)

## 2. 数据、代码与运行环境

### 2.1 代码版本

- 修正后的稳定 GFG 快照：`/data/yuchang/GFG_graphbatch_directed_v1`
- 本轮独立实验目录：`/home/yuchang/GFG_velocity_foundation_mousebrain_v1`
- 本轮实验没有覆盖稳定快照，新增的预训练代码、日志和权重均保存在独立目录中。

稳定版本中已经包含此前对邻接错位问题的修复，以及 graph-batch、无向平滑损失和有向 velocity alignment 损失。

### 2.2 预训练数据

预训练语料来自：`/data/dataset/Velocyto_pretrain_v1`

| 项目 | 数值 |
|---|---:|
| H5AD 文件数 | 2,018 |
| 总细胞数 | 12,431,916 |
| 物种 | Human |
| 默认词表大小 | 3,000 genes |
| 脑组织占比 | 约 79% |

需要注意：当前预训练划分仍主要是文件级代理划分，并不等同于严格的 study/donor 独立划分；同时语料中脑组织占比较高，因此本轮结果不能直接解释为跨所有组织的普适泛化能力。

### 2.3 下游测试数据

MouseBrain 数据路径：`/data/yuchang/dataset/MouseBrain.h5ad`

- 原始规模：3,365 cells × 936 genes
- 经当前 GFG 预处理后：3,365 cells × 759 genes
- Human 3k 词表按符号只能直接对应 MouseBrain 原始 936 个基因中的约 331 个。

本轮迁移主要依靠 GFG 中跨基因共享的网络权重，并非基于严格的人鼠直系同源基因映射。因此，这是一轮 human-to-mouse 可行性试验，而不是最终的跨物种 benchmark。

### 2.4 GPU 配置

- GPU 6：denoising-first 预训练
- GPU 7：先运行 large scratch 对照，随后运行 weak-kinetic 预训练
- 两张卡均为约 96 GB 显存的 NVIDIA H20

两条预训练路线并行执行了相同的 15,000 个 optimizer steps。由于 kinetic/JVP 路线显存与计算开销更高，二者 batch size 不同，因此是 step-matched，而不是 cell-exposure-matched 的严格对照。

## 3. 模型设计

### 3.1 模型规模

| 模型 | 总参数量 | 可训练参数量 | 相对原 GFG |
|---|---:|---:|---:|
| 修正后的原 GFG | 1,588,082 | 约 1.59 M | 1.0× |
| Foundation pilot | 14,638,082 | 14,621,698 | 约 9.2× |

大模型的主要配置为：

- gene embedding dimension：64
- hidden dimensions：768 → 1536 → 1536 → 768
- attention heads：8
- attention layers：2
- feed-forward multiplier：4
- codebook size：128，但正式实验中 `use_vq=False`

### 3.2 保留的 GFG 结构

大模型仍保留 GFG 的核心设计：

- manifold/state encoder；
- velocity/tangent encoder；
- decoder JVP/NTPL 映射；
- graph batching；
- undirected graph smoothness；
- directed velocity alignment。

因此，本轮主要是在不改变 GFG 核心计算图的前提下，扩大网络容量并增加预训练阶段。

### 3.3 Soft-VQ 处理

实验开始前对现有 Soft-VQ 做了短程 smoke test。其 codebook perplexity 在 3–4 个 step 后迅速降至约 1，说明几乎所有样本都坍缩到单一 code。

为避免 VQ 坍缩混淆预训练结论，正式实验中 scratch、denoising-first 和 weak-kinetic 三条路线全部一致地关闭了 VQ。后续如果重新引入离散表征，建议改用 residual/EMA VQ，并加入 code usage、perplexity 和 dead-code 监控。

## 4. 完成的实验

| 实验 | 预训练目标 | Batch size | Steps / Epochs | 学习率 | 说明 |
|---|---|---:|---:|---:|---|
| 历史修正 GFG | 无 foundation pretrain | 原配置 | 原配置 | 原配置 | 作为小模型参考基线 |
| Large scratch | 无 | 384 | MouseBrain 10 epochs | 3e-4 | 同规模大模型，从随机初始化训练 |
| Denoising-first | masked denoising | 512 | 15,000 steps | 2e-4 | GPU 6，BF16 |
| Weak-kinetic | denoising + ODE ramp | 192 | 15,000 steps | 2e-4 | GPU 7，BF16；ODE 权重在 500 steps 内升至 0.2 |

对于两个预训练 checkpoint，分别完成了：

1. 不使用 MouseBrain 标签或训练的 zero-shot 评估；
2. 使用当前 GFG 全损失在 MouseBrain 上微调 10 epochs；
3. 计算 confidence、ICCoh、raw CBDir、graph CBDir 和 graph improvement。

Large scratch 与两条预训练路线的下游实验目前均只使用 seed 0；历史修正 GFG 另有 seed 0–4 的五种子均值。因此，大模型之间的比较可作为方向性结论，但尚不足以替代多随机种子统计检验。

## 5. 预训练运行结果

| 指标 | Denoising-first | Weak-kinetic |
|---|---:|---:|
| Optimizer steps | 15,000 | 15,000 |
| Cell exposures | 7.68 M | 2.88 M |
| 运行时间 | 5.66 h | 5.28 h |
| 吞吐量 | 376.8 cells/s | 151.5 cells/s |
| PyTorch peak memory | 76.40 GiB | 81.46 GiB |
| 第一个 total loss | 6.7925 | 6.6879 |
| 最后一个 total loss | 3.9558 | 4.0291 |
| 最终 loss EMA | 3.5424 | 3.5254 |
| 最终 ODE loss | 不适用 | 1.4e-7 |

两个任务均稳定完成 15,000 steps，没有出现 NaN/Inf。Weak-kinetic 的 JVP/ODE 计算显著降低吞吐量并限制 batch size；其最终 ODE loss 极小，但这并未转化为更好的 zero-shot 方向指标。

## 6. MouseBrain 泛化结果

### 6.1 完整对比

| 模型 / 阶段 | Confidence | ICCoh | Raw CBDir | Graph CBDir | Graph improvement |
|---|---:|---:|---:|---:|---:|
| 修正 GFG，seed 0 | **0.9783** | 0.9208 | **0.6099** | **0.7498** | 0.1399 |
| 修正 GFG，seed 0–4 均值 | 0.9585 | 0.8430 | 0.3457 | 0.6204 | 0.2747 |
| Large scratch，fine-tuned | 0.5610 | **0.9960** | -0.4718 | 0.1496 | **0.6214** |
| Denoising-first，zero-shot | 0.6587 | 0.6267 | 0.3647 | **0.7080** | 0.3433 |
| Denoising-first，fine-tuned | **0.6780** | 0.9826 | -0.4364 | 0.1500 | 0.5864 |
| Weak-kinetic，zero-shot | 0.5202 | 0.5507 | -0.1160 | 0.3928 | 0.5088 |
| Weak-kinetic，fine-tuned | 0.5919 | 0.9928 | **-0.3925** | **0.1905** | 0.5830 |

表中粗体只表示相应局部分组中较高的结果。对 velocity 方向正确性而言，本轮最关键的指标是 Graph CBDir，而不能单独依据 ICCoh 或 graph improvement 判断。

### 6.2 相对 Large scratch 的微调收益

| 预训练路线 | Δ Confidence | Δ ICCoh | Δ Raw CBDir | Δ Graph CBDir | Δ Graph improvement |
|---|---:|---:|---:|---:|---:|
| Denoising-first | +0.1170 | -0.0135 | +0.0354 | +0.0004 | -0.0351 |
| Weak-kinetic | +0.0309 | -0.0033 | +0.0794 | +0.0409 | -0.0384 |

Weak-kinetic 是微调后表现最好的大模型，Graph CBDir 为 0.1905，比 large scratch 高 0.0409；但它仍远低于历史修正 GFG 的 seed 0 结果 0.7498 和五种子均值 0.6204。

### 6.3 Zero-shot 与微调后的变化

| 路线 | Zero-shot Graph CBDir | Fine-tuned Graph CBDir | 微调变化 |
|---|---:|---:|---:|
| Denoising-first | **0.7080** | 0.1500 | **-0.5580** |
| Weak-kinetic | 0.3928 | 0.1905 | -0.2023 |

这一结果是本轮最重要的发现：预训练模型在不经过 MouseBrain 微调时已经具有可迁移的方向信息，但当前全损失微调会明显破坏这种信息。

## 7. 主要结论

### 结论一：预训练确实学到了可迁移的 velocity 方向

Denoising-first 的 zero-shot Graph CBDir 达到 0.7080：

- 接近修正 GFG seed 0 的 0.7498；
- 高于修正 GFG 五种子均值 0.6204；
- 显著高于同规模 large scratch 微调后的 0.1496。

这说明即使没有在预训练阶段使用强 kinetic loss，masked denoising 加共享的 GFG 表征结构也能够学习到具有跨数据集、跨物种迁移能力的信号。它是本轮最有价值的正结果。

### 结论二：当前 MouseBrain 全损失微调会覆盖预训练能力

Denoising-first 的 Graph CBDir 在微调后由 0.7080 降至 0.1500；weak-kinetic 也由 0.3928 降至 0.1905。当前微调仍使用原 GFG 中较强的组合权重，例如 ODE、smoothness 和 alignment 项。对于已经形成表征的 1,460 万参数模型，这种全参数、高学习率、强图损失微调很可能造成 catastrophic forgetting，并把模型推向局部但方向错误的解。

因此，下一轮首先需要修复的是下游适配策略，而不是继续增加预训练规模。

### 结论三：扩大当前 GFG 网络本身并不会自动改善结果

Large scratch 的 ICCoh 达到 0.9960，但 Graph CBDir 只有 0.1496，Raw CBDir 为 -0.4718。它明显弱于小模型的修正 GFG，说明在当前目标函数与优化方式下，参数量增加约 9.2 倍并没有带来更好的 velocity 方向。

### 结论四：高 ICCoh 可能对应方向错误的平滑解

三个微调大模型的 ICCoh 都在 0.98–1.00 左右，但 CBDir 很低甚至为负。模型可以通过输出高度一致、幅度很小或近似常量的 velocity 获得很高的局部相干性，却不代表生物学方向正确。

因此后续选模必须联合使用：

- Graph/Raw CBDir；
- velocity magnitude 与方差；
- ICCoh；
- graph improvement；
- 可视化和已知 lineage 方向。

不应再用单一 ICCoh 作为主要 early-stopping 或 checkpoint 选择指标。

### 结论五：当前弱 kinetic 预训练目标不优于纯 denoising

Weak-kinetic 的 zero-shot Graph CBDir 为 0.3928，低于 denoising-first 的 0.7080，同时计算成本更高。当前每个 cell × gene 独立闭式求解的 ODE 约束很容易被优化到接近零，但它未必要求模型学到跨细胞共享、可迁移的动力学规律。

由于两条路线的 batch size 和 cell exposures 不相同，这个结论仍需严格的 compute/data-matched 对照；但现有证据不支持立即扩大当前 kinetic 目标的训练规模。

## 8. 下一轮实验建议

建议按以下顺序推进：

1. **保留 denoising-first checkpoint 作为主起点。** 当前最佳 foundation 结果来自它的 zero-shot 表现。
2. **冻结预训练 encoder，只训练小型 velocity head。** 先验证能否保留 0.7080 附近的 Graph CBDir，同时适配 MouseBrain。
3. **降低微调学习率并分阶段解冻。** 建议从 head-only 开始，再逐层解冻，而不是直接全参数训练。
4. **对损失进行归一化和渐进 ramp。** 避免固定的 ODE=20、smooth=300、align=300 等权重在大模型上主导梯度。
5. **严格比较 batch size 128 与 384。** 排除大 batch 改变图采样与优化动力学所造成的影响。
6. **使用 3–5 个随机种子。** 对最佳适配策略报告均值、标准差和失败率。
7. **重构动力学参数化。** 将当前 cell × gene 独立 beta/gamma 求解改为 gene-level 或 hierarchical shared kinetics，使约束真正携带可迁移结构。
8. **后续再扩展全基因组词表。** 优先使用稀疏 gene-token 输入、物种/组织条件和严格 ortholog mapping；在证明适配稳定前，不建议直接投入更大的 Transformer/Mamba 混合模型。

一个最小但信息量高的下一轮矩阵是：

| 因素 | 候选值 |
|---|---|
| 初始化 | denoising checkpoint / scratch |
| 可训练范围 | head-only / gradual unfreeze / full model |
| Batch size | 128 / 384 |
| 图损失 | off / normalized+ramped |
| Seeds | 0–4 |

## 9. 完整性与可复现性检查

本轮已完成以下检查：

- 两个预训练最终 checkpoint 均可 strict load，`missing=0`、`unexpected=0`，所有参数有限；
- 三个 MouseBrain 最终 checkpoint 均可 strict load，所有参数有限；
- 三个评估 H5AD 均为 3,365 × 759，`velocity`、`velocity_umap` 和 `velocity_graph_umap` 均无 NaN/Inf；
- 从 H5AD 重新计算的 confidence、ICCoh、raw/graph CBDir 与汇总文件完全一致；
- 邻接对齐单元测试 4 项全部通过；
- graph training 测试退出码为 0；
- artifact SHA256 manifest 校验通过；
- 实验产物总量约 2.6 GB，实验结束后 GPU 6/7 已释放。

主要产物：

- 汇总表：`experiments/foundation_pilot_20260830/mousebrain_comparison.csv`
- 详细结果：`experiments/foundation_pilot_20260830/FINAL_RESULTS.md`
- Denoising checkpoint：`experiments/foundation_pilot_20260830/gpu6_denoise_pretrain/pretrain_final.pt`
- Weak-kinetic checkpoint：`experiments/foundation_pilot_20260830/gpu7_kinetic_pretrain/pretrain_final.pt`
- 完整性清单：`experiments/foundation_pilot_20260830/ARTIFACT_MANIFEST.sha256`
- 实验入口：`foundation_pilot.py`
- 汇总程序：`summarize_foundation_pilot.py`

## 10. 一句话总结

本轮没有证明“把 GFG 直接做大并用原损失微调”有效，但明确证明了 denoising 预训练能够产生很强的 MouseBrain zero-shot velocity 方向；当前瓶颈已经从预训练转移到了如何在不破坏该表征的前提下进行下游适配。
