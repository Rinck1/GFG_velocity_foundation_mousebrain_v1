# GFG scale-up 审计与实施方案

更新日期：2026-09-29

这份文档把原论文的几何约束、当前项目的实验结果和 v3 代码审计放在同一个协议里。目标是先得到一个**可验证、可恢复、不会静默污染基因语义**的训练基线，再扩大模型和语料。

## 1. 论文中必须保留的东西

GFG 的核心不是“把一个 RNA velocity MLP 做宽”，而是把状态和运动拆开，并让运动通过状态生成器的切空间。论文的主链路是：

```text
x -> state encoder -> z_s -> state/topology codebook -> G
x (+ state evidence) -> velocity encoder -> z_v -> velocity primitive codebook
(z_s, z_v) -> JVP(G, z_s, z_v) = J_G(z_s) z_v -> v_tan
```

这里有五个不能被 scale-up 破坏的不变量：

1. 输入和输出仍在基因表达空间；PCA 只能作为图索引或诊断，不能替代模型输入/输出。
2. state stream 与 velocity stream 参数独立，两个 codebook 也独立。state codebook 表示流形支撑，velocity codebook 表示可复用的运动原子。
3. 推理速度必须和训练速度使用同一个 decoder JVP。不能训练时用 JVP、评估时换成另一个 velocity head。
4. 动力学约束是物理/领域残差，而不是把 scVelo 当作真值。scVelo 只能作为独立的 weak teacher 指标。
5. 投影残差和长程 rollout 必须单独报告；重构误差或局部速度相干性不能代替方向正确性。

论文解释了两个失败源：相邻细胞的冲突速度被平均后产生 smoothing drift，以及在弯曲流形外直接做离散积分产生 ambient rollout drift。作者用 JVP 生成切向速度，用离散 velocity primitives 拆分冲突运动；官方 ICML 页面和作者的公开方法说明还报告了 JVP/VQ 消融、圆周 ODE 和四个 scRNA 数据集结果。正式 OpenReview PDF 在当前环境被验证页拦截，因此本审计同时核对了官方海报、作者公开方法页和仓库源码；公式和结构以这些材料及源码为准。

## 2. 推荐的 scale-up 形态

### 2.1 数据协议先于模型容量

训练集按 donor/study 划分后，扫描**所有**训练文件的 `var/_index`，生成固定的全局基因 panel 和每个文件的 `source_column -> panel_column` 映射。均值、方差和 HVG 只在训练 donor 上用流式 Welford/两遍统计拟合，并把 panel、映射版本、统计量和数据 hash 写入 manifest。

每个读取块都遵循同一个协议：

```text
sparse H5AD rows
  -> source vocabulary mapping
  -> log1p(CP10K) for each layer
  -> panel selection + valid mask
  -> z-score for the model
```

缺失基因必须保留独立的 `valid_u/valid_s` mask；负哨兵只是传输格式，不能进入物理损失。输入读取不依赖 PCA，也不假设所有 accession 的基因顺序相同。

Moments 是可选的邻域平滑输入，不应成为默认的隐式预处理。若启用，`Mu`/`Ms` 的方向、基因映射和生成版本必须写入 manifest，并在小样本上做数值回归测试。

### 2.2 模型：先做层级 set encoder，再扩大宽度

当前 ISAB 的时间复杂度是线性于基因数、但激活仍然需要同时保存 `B x G x d_model`；在 3k 基因和 JVP 双流下，显存主要由激活而非参数决定。推荐的最终结构是 permutation-equivariant 的两级 set encoder：

1. 共享的 per-gene tokenizer/MLP 处理 `u,s,u-s,mask_u,mask_s`。
2. 将基因分成固定大小的无序 chunk（例如 256–512）；chunk 内使用共享的 local mixer。
3. 每个 chunk 通过 gated pooling 产生少量 summary；summary 流入全局 inducing tokens。
4. 由全局 summary 调制每个 chunk 的 local token，再进入共享 decoder。

summary 可以写成

```text
r_c = sum_i softmax(q(h_i))_i h_i,
g = ISAB({r_c}),
h'_i = FFN([h_i, g, r_c])
```

不加入绝对基因位置编码，因此基因重排仍保持等变。训练时按 chunk 流式计算，激活从 `O(B G d)` 降为 `O(B g_chunk d + B C d)`；JVP 只对当前 chunk 的 decoder 输出求导，最终在 panel 维度拼接。

建议的容量阶梯：

| 阶段 | panel | `d_model` | global inducing | `K_s/K_v` | 目的 |
|---|---:|---:|---:|---:|---|
| correctness pilot | 3,000 | 384–512 | 96–128 | 64/32 | 验证数据、JVP、图和 checkpoint |
| foundation pilot | 8,000 | 512–768 | 128–256 | 128/64 | 计算匹配的 denoising 预训练 |
| full panel | 36,601 | 512–768 | 256 | 128/64 | chunked sparse 输入，不整批 materialize |

`K` 不应跟着参数量无限增长。codebook 的健康度由全局 token 数、perplexity、dead-code 比例和跨 rank usage 决定；采用 soft-to-hard 或 EMA+dead-code restart，不能只依赖硬最近邻 STE。`K_s` 和 `K_v` 不必相同，velocity primitive 通常应更小。

### 2.3 训练阶段

* **Stage A — state denoising**：只计算 masked/visible reconstruction 和 state codebook 项；不计算 velocity tower、JVP 或 velocity VQ 更新。目标是学稳定的 manifold support。
* **Stage B — velocity adaptation**：冻结 state encoder/decoder（或只给极小 lr），训练 velocity tower、velocity codebook 和小型 adapter。先使用切向 JVP，再逐步加入 graph/kinetic 项。
* **Stage C — joint**：只在验证集的 raw/graph CBDir、projection residual 和 rollout 都稳定后，用 5–10 倍更小的 lr 解冻共享部分；所有动力学权重按 optimizer step ramp，而不是在第一个 epoch 全量打开。

下游 MouseBrain 适配默认采用 `head-only -> gradual-unfreeze -> optional joint`。已有实验显示 denoising-first 的 zero-shot Graph CBDir 很高，而全参数高权重微调会严重覆盖方向信息；所以 checkpoint 选择不能看 ICCoh 或 reconstruction 单项。

### 2.4 动力学和图约束

当前 `cell x gene` 独立求解三参数 `(alpha,beta,gamma)`，每个位置只有两条速度方程，岭回归后残差天然接近零，不能作为可迁移的主要约束。推荐：

* 把标准化 decoder 输出和 JVP 速度还原到同一个 log-expression 坐标后再计算残差；不对 z-score 做 `clamp_min(0)`。
* 用 batch 内跨细胞共享的 gene-level 参数，或带 organ/donor random effect 的 hierarchical 参数；所有有效细胞共同约束一个 gene 的速率。
* mask 后按有效元素归一化，记录有效率、参数条件数和 residual quantiles；在参数化重构完成前，kinetic 只做小权重监控项。
* graph smooth/align 必须在同一布局上计算。模型速度是 `(B,G,2)` 的 gene-interleaved 布局，表达位置也必须转成 `(u_0,s_0,u_1,s_1,...)`；图邻居索引必须对应真实 cell id。

图缓存不能把 300 个 donor 文件的完整 3k/8k 矩阵常驻内存。每个文件最多抽取固定数量的代表细胞，在低维 sketch 上建近似 kNN；训练时按 file/organ/donor 加权采样。`n<2` 的文件跳过，空图时自动关闭图损失；真正的大规模 global kNN 应作为单独可复用 artifact 构建，不应在每个 DDP rank 的训练启动阶段重复 PCA。

## 3. 当前 v3 的阻断问题

### P0：必须先修

| 文件/位置 | 问题 | 后果 |
|---|---|---|
| `gfg3_train.py:395` | `train_blocks` 未定义 | GraphCache 修好后仍会在第一个 epoch 直接 `NameError` |
| `gfg3_train.py:169–207` | GraphCache 读完整 donor；`n=1` 仍请求 2 个邻居 | 现有 full run 已在 `n_neighbors=2, n_samples_fit=1` 中止；大语料还会 OOM |
| `gfg3_data.py:56–59` | `mismatched` 列表最后固定 `[:0]` | 词表审计永远报告无 mismatch |
| `gfg3_data.py:91–105,124–137` | union 词表只取前 50 文件；统计读取把原始列号当 union 列号 | 异构/重排词表会静默污染 gene semantics，统计也不是真正的 train-only union |
| `gfg3_data.py:170`、`gfg3_moments.py:135` | `range(0,max(n-block,1),block)` 丢掉最后一个 block，`n<=block` 时只读一块 | 大量细胞永远不参与训练/统计 |
| `gfg3_moments.py:38` | `_read_layers` 返回 `(u,s)`，调用却写成 `s_log,u_log` | 生成的 `Ms/Mu` 方向交换 |
| `gfg3_train.py:144` | z-score 输入被 `clamp_min(0)` 后送入 RNA ODE，缺失 sentinel 也被当成有效零 | kinetic 项物理语义错误 |
| `gfg3_train.py:148–155` | `v_x` 展平为 gene-interleaved，`x_self`/`x_nb` 却按 `[u;s]` block 展平 | smooth/align 的方向和速度错位 |
| `gfg3_train.py:287–343` | BlockStream 每个 rank 都遍历完整流；只改 steps 不能形成 DDP shard | 重复样本、有效 batch/学习率解释错误 |
| `gfg3_train.py:382–385` | 每个 epoch 的 `model.train()` 覆盖 `set_stage()` 设定的冻结子模块模式 | dropout、EMA VQ 和冻结阶段行为不再符合阶段定义 |
| `gfg3_eval.py:27–29` | numpy `gene_names` 不能用于 `or`；`genes` 未定义 | 评测入口必然报错 |

### P1：训练可跑后必须修

* `--min-cells` 解析但从未作用在 train/val/test 文件列表。
* `--scvelo-teacher` 只解析，没有计算或记录 teacher agreement。
* `GraphCache._cells()` 以 `min(rows):max(rows)` 读取，稀疏随机行会放大 I/O 和内存。
* `restart_dead_codes()` 只广播 embedding，EMA 的 cluster/sum 在 rank 间不完全一致，optimizer state 也不清理。
* `save_ckpt()` 定义了完整 checkpoint，但训练实际保存的是精简 payload；resume 不保存 scheduler、gstep、stage-specific epoch、RNG 和 VQ EMA。
* resume 只对 state 使用 `start_epoch`，velocity/joint 从 0 开始；没有验证 manifest、code hash 和统计量兼容性。
* 没有 validation loop，不能按 held-out donor 的 raw/graph CBDir 选 checkpoint。
* `gfg3_eval.py` 固定 `cuda:0`、重开 HDF5 不关闭、`n<21` 时 kNN 失败，且只取每文件前 `max-cells` 个细胞。

### P2：结果解释风险

* 仅比较 ICCoh 会选择近零或常量速度；必须联合 CBDir、速度幅度/方差、projection residual、rollout drift。
* full run 的 launch 使用 `batch=32` 与 `K=128`，codebook token/step 偏少，且没有 EMA/restart；应先观察 usage/perplexity 再决定是否扩大 K。
* xlarge 的 117.9M 参数主要增加容量，没有解决激活、数据覆盖和适配灾难遗忘；在 correctness pilot 通过前不应继续扩大。

## 4. 实施顺序和验收

1. 修复读取方向、映射、block 范围、P0 训练/评估入口和单元测试。
2. 用 2–4 个合成异构词表文件测试：列重排、缺失基因、`n=0/1/block/block+1`，并验证 panel 输出逐元素相等。
3. 在 CPU 小模型上完成 `forward -> backward -> save -> resume -> eval`；再用单 GPU 真实 24 文件跑两 epoch。
4. 用固定 donor manifest 和 shard-aware stream 跑双 rank smoke，验证每个 block 只被一个 rank 消费，global loss 与单 rank 误差在容差内。
5. 先跑 3k panel、denoising-first pilot；只有 validation raw/graph CBDir 和 rollout 都通过，才启用 velocity stage。
6. 最后才实现 chunked hierarchical encoder，并做 3k/8k/全 panel 的显存、cells/s、JVP 开销曲线。

验收标准不是“日志出现 DONE”，而是：无静默 gene misalignment；每个 split 有可追溯 manifest；checkpoint 可以在相同或不同 world size 恢复；P0 测试全部通过；held-out donor 指标和方向可重复。

