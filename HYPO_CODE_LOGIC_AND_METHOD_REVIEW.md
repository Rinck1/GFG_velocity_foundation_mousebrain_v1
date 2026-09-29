# Hypoblast 代码逻辑、模块接口与方法合理性审查

> 审查日期：2026-09-03  
> 审查对象：解压后的 `hypoblast-main` 代码快照  
> 最终目录：`/data/yuchang/hypo`  
> 说明：本报告基于仓库源码、配置、设计文档、测试代码和已提交 benchmark 证据进行静态审查；没有把设计文档中的计划当作已完成实验，也没有把模型内部权重解释为已验证的生物学因果关系。

---

## 1. 执行摘要

### 1.1 这个项目实际在做什么

Hypoblast 是一个两阶段的单细胞扰动预测系统：

1. **Stage-1：单细胞离散表征学习**
   - 输入为单细胞表达，实际主流程通常直接使用预计算的 scFoundation pooled embedding，形状 `[N, 3072]`。
   - 一个 MLP 将 3072 维特征压缩为 128 维连续状态。
   - 128 维状态被拆成 4 个 32 维 token，并分别在 4 个、每个含 128 个向量的 SoftVQ codebook 中做软分配。
   - 拼接后的量化状态通过 MLP decoder 重建 2000 或 5000 个 HVG。
   - 部分 Stage-1 变体加入 cell-type 分类损失或 supervised contrastive loss。

2. **Stage-2：样本群体层面的扰动流预测**
   - 它不是逐细胞预测，也不是细胞之间的 OT；每条记录是一对“control sample → treated sample”。
   - Stage-1 的四码本组合被整理为 128 个全局 prototype。
   - 每个样本被表示为 128 个 prototype 的比例、presence 和 centered log-ratio（CLR）质量坐标。
   - 药物组合先映射为 pathway 条件，再由带 self-attention、cross-attention 和可选 AdaLN 的条件流预测 prototype mass 的变化。
   - 终点输出映射为 cell-type proportion，并可经冻结的 Stage-1 decoder 生成 pseudo-bulk 与 cell-type expression。

### 1.2 总体判断

**作为研究原型，方法链条是连贯且有价值的；作为可作强生物学结论或可独立复现的交付物，目前还不够。**

合理之处包括：

- 将扰动预测放在样本/群体层面，避免把没有真实配对关系的细胞强行一一对应。
- 用完整的 4-codebook product tuple 定义 prototype，避免把四个边缘码本槽位误当作同一个生物状态。
- Stage-2 显式按 biological sample 划分，严格测试集协议清楚。
- drug→pathway 的已知边保留正负号，unknown edge 用低秩分支，且新分支多采用零初始化，归纳偏置较克制。
- 终点比例损失从真实 control 状态 rollout，避免直接借用含 terminal 信息的插值状态来“作弊”。
- v8 对显式 pathway→cell-type 通路做了 zero、shuffle、condition exchange 和 gate-zero 干预，机制审计思路正确。

主要问题包括：

- 当前压缩包不包含原始数据、checkpoint、`pathway_base.pt`、pair records 或依赖环境，不能独立端到端复现。
- 当前机器只有 Python 3.6.8，而代码至少需要支持 `from __future__ import annotations` 的新版 Python，并依赖较新的 PyTorch API；因此现机无法运行测试。
- Stage-1 的划分并不是完整的 sample-level holdout：只强制指定样本进入测试，其余补足的 test cells 和 validation cells仍按细胞划分。对于带 cell-type 监督的 Stage-1，这意味着下游 strict-test 样本可能在 Stage-1 已被看见，属于**跨阶段的 transductive 设置**，不能称为端到端 sample-level 泛化。
- celltype-balanced prototype 选择脚本只排除了 strict-test 样本，未排除 validation 样本；其 prototype vocabulary 会使用 validation cell-type 标签，因而 validation 不完全独立。strict test 仍被保护，但 validation 架构选择可能偏乐观。
- Stage-2 validation 的 loss 每轮重新随机采样 flow time，checkpoint 排名带额外随机噪声。
- rollout 全程固定使用 source presence；预测出的 terminal presence 不反馈进动力学，prototype 的出现/消失只在终点门控。
- `state_delta` 不是随 ODE 积分的状态，只在最终一次网络调用中直接读取，却用于表达解码；其“状态变化”解释弱于 mass flow。
- 多个核心诊断存在定义或数值问题，例如 SoftVQ 的 `avg_probs` 恒为 `1/K`、MSE 方差归一化无下界保护、per-gene R² 大量为极端负值。
- README 描述的是较早设计，与当前 prototype-mass flow、presence BCE、proportion JS、expression loss 和 19-path 主线不一致。

### 1.3 当前实验支持的选择

按仓库中最新且证据最完整的结果：

- **composition 任务工作点**：`celltype_hard_sign_global`，strict-test proportion MAE `0.038722`、RMSE `0.082228`。
- **扰动响应主工作点 / 当前推荐 baseline**：`celltype_lowrank_balanced`，PCC sample `0.693102`、PCC gene `0.475293`、R² delta `0.350239`、DEG@20 `0.537500`、Spearman `0.519208`。
- **聚合表达误差工作点**：`baseline_lowrank_balanced`，pseudo-bulk MSE `0.009030`、raw-sample PCC `0.938850`。
- `celltype_balanced_lowrank_balanced` 虽获得最低 cell-type expression MSE `0.037015` 和最高 DEG@20 `0.545833`，但 per-gene R² 为 `-152.481140`，在该异常解释前不宜替换主模型。
- v8 的显式 Pathway→Cell type M2 通路确实被模型使用，参数也有跨 seed 稳定性；但 strict-test 三 seed 均值上，baseline 在 14/14 个聚合指标上更好，因此 v8 报告最终为 `winner = null`。

---

## 2. 归档与目录处理记录

- 用户描述的 `/hypo*.zip` 和 `/root/hypo*.zip` 未找到。
- 服务器上唯一匹配的上传归档为 `/home/yuchang/hypoblast-main.zip`，据此将其视为目标文件。
- 原始归档大小：`18,283,791` bytes。
- SHA-256：`17956bb451b0209c4baa114e4b8fa966c88e6b6c8e97346c0abec11f63cd523b`。
- `unzip -t` 校验通过；共 754 个 ZIP entries。
- 解压前检查未发现绝对路径、`../` 路径穿越、反斜杠路径或异常符号链接条目。
- 归档已移动到 `/data/yuchang/hypoblast-main.zip`。
- 内容已解压并规范化为 `/data/yuchang/hypo`，而不是保留外层 `hypoblast-main` 名称。
- 解压后约 639 个文件、49 MB。
- 原 ZIP 被保留，便于校验和恢复；没有删除。

---

## 3. 仓库结构与边界

| 目录/文件 | 角色 | 是否属于当前主线 |
|---|---|---|
| `README.md` | 项目总体说明，但部分内容已落后于代码 | 参考，不能单独作为实现真相 |
| `CLAUDE.md` | 项目开发约定 | 工程说明 |
| `configs/stage1/` | 当前 Stage-1 SoftVQ 配置 | 是 |
| `configs/stage2/` | Stage-2 条件流配置 | 是 |
| `configs/legacy/` | 早期 VQ/MNIST/scRNA 配置 | 否，主要用于历史复现 |
| `data/` | dataset loader、pathway YAML、drug-pathway sign 表 | 是，但大数据未随 ZIP 提供 |
| `model/` | VQ-VAE、量化器、条件编码器、prototype flow | 是 |
| `experiments/scfoundation_vqvae/cell_embedding/` | embedding、Stage-1 训练/推理/评估工具 | 是 |
| `experiments/perturbation/` | prototype 数据准备、Stage-2 训练评估与可视化 | 是 |
| `common/`、`evaluation/` | 指标、损失和旧验证工具 | 部分是当前主线，部分是 legacy |
| `scfoundation/` | vendored scFoundation 加载与 backbone | 是，主要被冻结使用 |
| `benchmarks/` | 已完成实验的 manifest、指标、证据和报告 | 是，实验事实的主要来源 |
| `.skill-build/` | benchmark 自动化/报告基础设施 | 辅助，不参与模型前向 |
| `tests/perturbation/` | 6 个测试文件、87 个 test cases | 主要覆盖 Stage-2 |
| `experiments/vqvae_mnist/` | 旧 MNIST 路线 | 已禁用/不可作为当前有效主线 |

归档文件类型以结果表格和源码为主：205 个 CSV、173 个 Python、135 个 JSON、52 个 PNG、38 个 Markdown、17 个 YAML、16 个 shell script、2 个 PDF。核心 Python 代码约 2.24 万行；benchmark 证据占文件数的大部分。

### 3.1 压缩包不是自包含运行包

下列主流程依赖在本 ZIP 中缺失：

- 约 84 GB 的原始 h5ad。
- 约 31 GB 的 scFoundation `[2,550,159, 3072]` embeddings。
- 约 20 GB 的 HVG2000 targets 和约 51 GB 的 HVG5000 targets。
- scFoundation checkpoint 与 gene index。
- Stage-1/Stage-2 checkpoints。
- `results/pathway_base.pt`。
- `data/hypo_condition_small_molecule_metadata.csv`。
- prototype assignments、population cache、pair records。
- `requirements.txt`、`environment.yml` 或 `pyproject.toml`。

大量配置还硬编码 `/liaozizhuo/hypoblast` 和 `/liaozizhuo/iPStem`。因此该快照适合代码审阅和查看实验证据，不适合直接在当前目录执行训练。

---

## 4. 端到端数据流

### 4.1 核心张量符号

| 符号 | 典型形状 | 含义 |
|---|---:|---|
| `N` | 2,550,159 | 全部细胞数，来自登记文档 |
| `B` | batch size | 训练批量 |
| `G` | 4 | Stage-1 codebook group 数 |
| `K_code` | 128 | 每个 Stage-1 group 的 code 数 |
| `Dq` | 32 | 每个 group code 维度 |
| `Dz` | 128 | 拼接后的 Stage-1 latent 维度 |
| `K` | 128 | Stage-2 prototype 数 |
| `P` | 19 | 当前完整 pathway 数 |
| `D_drug` | 24 | sign table 中的 drug 数 |
| `C` | 10 | v7 cell-type 数 |
| `H` | 2000/5000 | 重建的 HVG 数 |

### 4.2 主流程图

```text
h5ad expression [N, genes]
  │  对齐 scFoundation gene vocabulary，log1p/scale，附加 total-count tokens
  ▼
frozen scFoundation
  ├─ pooled export: [N,3072]
  └─ optional token export: [N,S,768] + mask/length
  │
  ▼
Stage-1 MLP encoder head
  [N,3072] -> [N,128]
  │
  ▼
pre-quant + reshape
  [N,128] -> [N,4,32]
  │
  ▼
SoftVQ: E[4,128,32]
  ├─ probs [N,4,128]
  ├─ hard code_ids [N,4,1]
  └─ z_q [N,4,32] -> [N,128]
  │
  ├─ MLP decoder -> reconstructed HVG [N,H]
  └─ prototype export
       │
       ├─ 选择 128 个完整 product tuples [code0,code1,code2,code3]
       ├─ 每细胞对所选 tuples 的概率 [N,128]
       └─ prototype codebook [128,128]
              │
              ▼
       按 sample_id 聚合
         proportion [sample,128]
         presence   [sample,128]
         CLR logits [sample,128]
         expression [sample,H]
         cell-type proportion [sample,10]
              │
              ▼
       control-treated pair records
         l0,l1,m0,m1,drug_ids,targets
              │
              ├───────────────────────────────┐
              ▼                               ▼
       drug/pathway conditioner         prototype flow
       drug ids [B,L]                   K=128 slots
       -> pathway activity [B,19]       + mass/presence/time
       -> tokens [B,19,256/768]         + self/cross attention
       -> optional context [B,128,256]  -> velocity [B,128]
                                       -> presence logits [B,128]
                                       -> state delta [B,128,128]
                                              │
                                              ▼
                                     20-step Euler rollout
                                              │
                     ┌────────────────────────┴───────────────────────┐
                     ▼                                                ▼
             prototype proportion [B,128]                    frozen Stage-1 decoder
                     │                                       -> expression [B,H]
                     ▼
             prototype_to_celltype [128,10]
                     │
                     ▼
             cell-type proportion [B,10]
```

### 4.3 数据预处理

`ScRNADataset` 有两条输入路线：

1. **原始 h5ad 路线**
   - 读取 h5ad 的 `X`，支持 dense 或 CSR。
   - 按 scFoundation gene index 对齐表达。
   - 移除总表达为 0 的细胞。
   - `x` 使用 log1p(CPM) 加两个总量相关 token；`target` 为 HVG/full 的 log1p(CPM)。

2. **预计算路线（当前主要训练方式）**
   - `x` 来自 `cell_embeddings_3072.npy`。
   - 通过 metadata CSV 中 `cell_id` 对齐 h5ad 或 target NPY。
   - `target` 来自 `targets_hvg2000.npy`/`targets_hvg5000.npy`。
   - 可把每个 DDP rank 的 train/val/test shard 预载入内存。

### 4.4 Stage-1 前向和损失

当前代表配置 `stage1_softvq_hypo_hvg2000_celltype_updated_v7.py`：

```text
x [B,3072]
 -> MLP 3072→1024→256→128
 -> Linear 128→128
 -> reshape [B,1,128]→[B,4,32]
 -> SoftVQ, 4×128×32
 -> reshape [B,4,32]→[B,1,128]→[B,128]
 -> MLP decoder 128→256→1024→2000
```

主要损失：

```text
L_stage1 = MSE(x_hat,target) / x_train_var
         + entropy_loss
         + optional 0.25 * celltype_CE
         + optional scheduled supervised_contrastive_loss
```

SoftVQ 当前将 VQ、commitment、codebook losses 明确设为 0，实际量化约束来自 soft assignment、重建梯度和 entropy balance。

### 4.5 Prototype 定义

当前主线不是 README 早期写的 `[G,K,Dq]` 边缘 token population，而是：

- 从每个细胞的四个 hard code 组成完整 tuple，例如 `(c0,c1,c2,c3)`。
- 选择出现频率高或按 cell type 平衡的 128 个 tuple。
- 每个 prototype 向量是该 tuple 对应四个 32 维 code 的拼接，形状 `[128]`。
- 每个细胞对 tuple 的 soft probability 是四组 assignment probability 的乘积。
- 只保留所选 128 个 tuple 后重新归一化，得到 `[N,128]` assignments。

这比将四个码本的相同编号强行对齐更合理，但会丢弃所选 tuple 之外的概率质量，并使稀有状态较难进入 vocabulary。

### 4.6 样本群体状态

对每个 sample：

```text
proportion[k] = mean_cell assignment[cell,k]
hard_count[k] = count(argmax assignment == k)
presence[k] = hard_proportion[k] >= threshold AND hard_count[k] >= min_count
mass_logit[k] = log(proportion[k] + eps) - mean_k(log(proportion[k] + eps))
```

代表性 pipeline 配置使用 `K=128`、assignment temperature `0.07`、presence threshold `0.001` 或 config 中的 `0.02`、`min_count=3`。实际 shell command 参数会覆盖部分 config，因此 artifact manifest 比 config 更接近最终事实。

### 4.7 药物到 pathway 条件

当前 `data/drug_pathway_sign.csv` 含 24 条记录、24 个 drug；每个 drug 只有一条明确的 known pathway sign。19 条 pathway 为：

`activin, bmp, camp_pka, epigenetic_rest, erk_mapk, fgf_fgfr, hdac, jak_stat, jnk_mapk, nodal, pdgf_pdgfr, pi3k_akt, pkc, prc2, ra_rar_rxr, rock, src_family, tgfb, wnt_canonical`。

legacy condition 的已知边为：

```text
A_known[d,p] = sign[d,p] * softplus(raw_magnitude[d,p])
```

未列边可固定为 0，或使用 rank-4 unknown factors：

```text
A_unknown = tanh((U_drug @ V_pathway.T)/sqrt(rank) * pathway_gate)
```

药物集合先将单药效应求和并经过 `tanh`；多药时再加入一个 permutation-invariant residual。动态 pathway token 为：

```text
token[p] = LayerNorm(pathway_base[p] + activity[p] * normalized_direction[p])
```

### 4.8 Prototype mass flow

每个 prototype slot 的 token 由四部分相加：

```text
prototype_code_projection
+ scalar_mass_embedding
+ scalar_source_presence_embedding
+ time_embedding
```

每个 flow block 顺序为：

```text
self-attention over prototypes
-> cross-attention to pathway tokens
-> time AdaLN-like modulation
-> optional pathway-specific AdaLN-Zero
-> feed-forward network
```

网络输出：

- `mass_velocity [B,K]`
- `presence_logits [B,K]`
- `state_delta [B,K,128]`
- `hidden [B,K,256]`

训练路径为线性 conditional flow matching：

```text
t ~ Uniform(0,1)
l_t = (1-t) * l0 + t * l1
target_velocity = l1 - l0
```

同时额外在真实 source state、`t=0` 计算 velocity loss。终点用默认 20 步显式 Euler rollout。

### 4.9 Stage-2 损失

```text
L_flow = 0.25 * MSE(v(l_t), l1-l0)
       + 1.00 * MSE(v(l0,t=0), l1-l0)

L_total_base = L_flow
             + w_presence * BCE(terminal_presence_logits, terminal_presence)
             + w_proportion * JS(predicted_celltype_proportion,
                                  true_celltype_proportion)
             + w_expression * MSE(decoded_population_expression,
                                  true_population_expression)
```

根据 conditioner 还可加入：

- unknown-effect L1
- unknown-gate L1
- pathway→cell-type proportion auxiliary loss
- pathway→cell-type expression auxiliary loss
- pathway-row shuffle contrastive loss

checkpoint 选择使用 `base_total`，即不让这些额外 auxiliary regularizers 直接改变模型选择指标；这一点有助于公平比较机制分支，但 `base_total` 自身仍包含随机 time 的 flow loss。

---

## 5. 核心模块输入与输出

## 5.1 `data/`

| 模块 | 输入 | 输出 | 评价 |
|---|---|---|---|
| `data/scrna_dataset.py` | h5ad、gene index、HVG list；或 embedding NPY、metadata CSV、target NPY | 单样本 dict：`x`、`target`、`cell_id`、可选 `celltype_label` | 支持 lazy dense/CSR 和 memmap，接口完整；dense 零细胞检查会整体读入内存，CSR 检查逐行循环；不支持 CSC；未显式检查 metadata `cell_id` 唯一性 |
| `data/perturbation_schema.py` | sign matrix 或 CSV，drug/pathway vocabulary | 严格为 `-1/0/+1` 的 `[D,P]` float tensor | 校验简单明确；CSV 中缺失的组合补 0 |
| `data/pathways_v73_19path.yaml` | 19 个 pathway 对应的 gene list | pathway→genes 字典 | 词表顺序是模型 ABI；平均池化忽略基因方向、权重和集合重叠 |
| `data/drug_pathway_sign.csv` | 24 个 drug 的已知方向 | 稀疏 drug→pathway sign prior | 可审计，但先验极稀疏且无 dose；未列边究竟是“无效”还是“未知”需由配置解释 |

## 5.2 `scfoundation/`

| 模块 | 输入 | 输出/职责 |
|---|---|---|
| `scfoundation/load.py` | checkpoint 路径、model config、device | 加载预训练 scFoundation、词表和 padding 约定 |
| `pretrainmodels/select_model.py` | architecture 配置 | 选择 Transformer/Performer backbone |
| `pretrainmodels/mae_autobin.py` | 表达值/token 信息 | 自动离散化 embedding 与 MAE 相关组件 |
| `pretrainmodels/transformer.py` | gene-expression token sequence | Transformer encoded tokens |
| `pretrainmodels/pytorchTransformer.py` | sequence + positional embedding | PyTorch Transformer 版本的 token features |
| `pretrainmodels/performer.py` | 长序列 token | Performer attention features |
| `pretrainmodels/reversible.py` | hidden sequence | reversible block 内存优化 |

这些文件是 vendored 第三方核心，项目主线通常冻结其参数。实际更新主要发生在 Stage-1 MLP head、量化器和 decoder；如果启用 token attention 或 light fine-tune，才会改变相关部分。

## 5.3 `model/components/` 与 `model/model.py`

| 模块/类 | 输入 | 输出 | 备注 |
|---|---|---|---|
| `BaseEncoder` | 任意 `x` | `z_e` | 抽象接口 |
| `BaseDecoder` | `z_q` | `x_hat` | 抽象接口 |
| `BaseQuantizer` | `z_e` | `(z_q, losses, indices)` | 抽象接口 |
| `BasePrior` | indices/condition | logits 或 sampled indices | 旧 image prior 接口 |
| `ConvEncoder` | `[B,C,28,28]` | `[B,H,7,7]` | MNIST legacy；当前 VQVAE 明确拒绝 `flat=False` |
| `ConvDecoder` | spatial latent | `[B,C,28,28]` | 同上，现主类无法走通 |
| `ScFoundationEncoder` | raw `[B,n_gene+2]`、pooled `[B,3072]` 或 token `[B,S,768]` | `[B,latent_dim]` | token 模式可在 pool 前做 attention；pooled 3072 模式不能重新获得 token attention |
| `MlpDecoder` | `[B,Dz]` | `[B,H]` | 当前 scRNA decoder |
| `VQQuantizer` | `[B,L,D]` | hard nearest `z_q`、loss、`[B,1,L]` ids | 欧氏距离、straight-through |
| `MultiCodebookVQQuantizer` | `[B,L,D]` | grouped cosine hard VQ、`[B,G,L]` ids | UniTok 风格，含 entropy/EMA usage |
| `SoftVQQuantizer` | `[B,L,Dq]` | soft weighted `z_q`、loss、`[B,G,S]` ids | 当前主线；codebook assignment logits 对 embedding detach |
| `ReshapeProjection` | `[B,1,128]` 或 `[B,4,32]` | group split/merge | 当前 SoftVQ 使用 |
| `AttnProjection` | sequence tensor | dimension-changing attention output | 可用但不是当前主配置 |
| `ContrastiveProjectionHead` | `z_e [B,D]` | normalized contrastive feature | 仅 SupCon 变体 |
| `VQVAE.forward` | cell feature | `(x_hat, losses)` | `losses` 含 latent/quantizer/classifier/contrastive 信息，但 forward 不返回 indices |
| `VQVAE.encode` | cell feature | hard code ids `[B,G,L]` | 训练循环为统计 code usage 会二次前向 |
| `VQVAE.decode_from_indices` | code ids | reconstruction | 可用于 code 解释 |
| `PixelCNN` | image code grid + class | code logits/sample | legacy；与 flat-only VQVAE 不兼容 |
| `model/registry.py` | CONFIG dict | 组装 encoder/decoder/quantizer/VQVAE/prior | registry 清楚；配置 schema 主要靠运行期检查，无统一静态 schema |

### `ScFoundationEncoder` 细节

- token 序列 pool：最后 token、倒数第二 token、gene token max、gene token mean，各 768 维拼接为 3072。
- `use_precomputed=True` 时不会加载 scFoundation backbone。
- 若输入 `[B,S,768]`，可用自定义 attention 后再 pool。
- CUDA 路径强制 `FLASH_ATTENTION` backend，没有显式 fallback；硬件、dtype 或 mask 不支持时可能失败。
- 若只保存 `[B,3072]` pooled embedding，配置即使写 `use_attention=True` 也不能恢复 gene-token attention 的语义。

### `SoftVQQuantizer` 细节

- codebook：`[G,K,Dq]`，当前 `[4,128,32]`。
- assignment：`softmax(dot(normalize(z), normalize(E.detach())) / tau)`。
- quantized token：assignment 对 codebook 的加权平均。
- entropy loss 的内部默认 temperature 为 `0.01`，而 assignment `tau=0.07`；这是两个不同锐度，需作为明确超参数而不是隐含默认。
- `avg_probs = mean(mean(probs, dim=-1))` 数学上恒等于 `1/K`，因此该字段不能诊断 assignment 是否均衡。
- `show_usage` 的历史 buffer 初始为 code 0，窗口未填满前 code 0 utilization 会被高估。
- 未显式提前校验 `seq_length % num_codebooks == 0` 和最终 reshape 维度；错误会在 `view` 才暴露。

## 5.4 `model/train.py`

| 输入 | 输出 |
|---|---|
| Stage-1 config、预计算 embedding/target、metadata、DDP 环境变量 | best checkpoint、TensorBoard、split NPZ、test metrics 日志 |

主要职责：

- DDP/NCCL 初始化，单进程普通 `python` 不支持。
- 加载 scRNA 数据，生成或复用 split。
- 可选在训练 target 上拟合 PCA。
- 训练 reconstruction、quantizer、cell-type CE、SupCon。
- 聚合各 rank code usage 和 validation metrics。
- early stopping、atomic checkpoint、最终 held-out test evaluation。

优点：

- saved split 会检查覆盖、重叠和越界。
- class weights 只用 train split 统计。
- PCA 只在 train targets 上拟合。
- validation 指标按 sample count 跨 rank 汇总。
- checkpoint 写入采用临时文件替换，降低中断损坏风险。

问题：

- `init_codebook_from_embeddings` 明确读取其输入文件的**全部行和全部 cell-type labels**，没有接收 `train_idx`。只要启用 cell-type codebook init，就可能将 validation/test label statistics 注入初始化。
- group holdout 只保证配置列出的 group 全部进入 test；为了补足 `test_fraction`，其余 test 行仍按 cell 抽样，validation 也按 cell 抽样。
- `x_train_var` 只取 `train_idx` 的前最多 2000 个 target 估算，且 `reconstruction_loss` 没有 epsilon；极小或 0 方差会导致不稳定。
- 每个训练 batch 在 `model(x)` 后又调用 `model.module.encode(x)` 统计 ids，重复 encoder/quantizer 计算。
- DDP rank shard 在 world size 过大时可能得到空 val/test shard。
- MNIST 训练/验证代码仍在，但 VQVAE 已关闭 spatial 路线，属于死接口。

## 5.5 `model/perturbation/`

| 模块/类 | 输入 | 输出 | 当前用途 |
|---|---|---|---|
| `condition_types.ConditionEncoding` | — | `condition_tokens`、可选 `pathway_context`、activities、diagnostics | 统一三类 conditioner 契约 |
| `condition_io` | pathway YAML、pooled base PT、sign CSV、config、drug vocabulary | conditioner + drug/pathway 名称映射 | vocabulary 的顺序必须与 checkpoint 一致 |
| `SignedInteractionCondition` | `drug_ids [B,L]`、`drug_mask [B,L]` | tokens `[B,P,D]`、activity `[B,P]` | 当前 canonical baseline |
| `PathwayPrototypeCondition` | drug ids/mask、gene pathway embedding `[P,768]` | fixed tokens `[B,P,256]`、context `[B,K,256]` | v7 18-path AdaLN 实验 |
| `PathwayCelltypeCondition` | legacy encoding、`R[K,C]`、expression targets | legacy tokens + context `[B,K,256]` + auxiliary losses | v8 显式 `W[P,C]` 实验 |
| `PrototypeVQ` | generic latent vectors `[N,D]` | independent soft prototypes | 早期/通用实现，非当前 product-tuple 主线 |
| product prototype helpers | hard code tuples、group probs、labels/sample ids | selected tuples、prototype codebook、assignments、`R[K,C]` | 当前主线 |
| `TimeEmbedder` | `t [B]` | sinusoidal `[B,time_dim]` | flow time condition |
| `PrototypeMassFlow` | mass `[B,K]`、source presence、condition tokens、t、optional context | velocity、presence logits、state delta、hidden | Stage-2 核心 |
| `PrototypeMassFlowAdapter` | control/treated population、targets、conditions | 分项 loss 或 rollout endpoint | 对 flow 加终点监督和 decoder 边界 |

### 三类 conditioner 的差别

1. **legacy dynamic token**
   - 药物改变 pathway activity，activity 改变 pathway token。
   - flow 每层通过 cross-attention 使用这些动态 token。
   - 当前 19-path baseline 属于此类。

2. **pathway→prototype AdaLN**
   - cross-attention token 的 pathway gene semantics 固定。
   - drug activity 经显式 `W[P,K]` 生成每个 prototype 的 context。
   - context 在 flow block 之间或内部用 AdaLN-Zero 注入。

3. **pathway→cell-type→prototype AdaLN**
   - 学习显式 `W[P,C]`。
   - `activity * W` 得到每个 pathway 对 cell type 的贡献。
   - 再经固定 `R[K,C]` 投影到 prototype context。
   - `W` 同时受 cell-type proportion CLR delta 和 cell-type expression delta 的辅助监督。

### Conditioner 风险

- legacy 的 `tanh(sum effects)` 会饱和，组合大小和剂量信息丢失。
- 当前没有 dose；不同浓度被视作相同药物条件。
- 组合 residual 用平均 effect，可保持 permutation invariance，但对集合基数的表达能力有限。
- `SignedInteractionCondition.forward` 没有像 `PathwayPrototypeCondition` 那样完整检查 ids/mask 的形状与 id 范围。
- unknown effect 的 factor、gate、downstream pathway mapping 存在尺度/符号不可识别性。
- 在 pathway→prototype 模式中，known sign 只约束 drug→pathway activity；后续 `W[P,K]` 可以翻转对 prototype 的最终方向，不能把最终热图直接称为先验保号的生物效应。
- cell-type CLR auxiliary 对 0 proportion 使用 `clamp(1e-8)`；某类从 absent 到 present 时目标可能非常大。
- auxiliary expression head 直接从 pathway context 预测 delta，不保证主 flow 输出必须依赖它；v8 的 route intervention 正是为检查这种 bypass。

## 5.6 `experiments/perturbation/`

| 脚本 | 主要输入 | 主要输出 |
|---|---|---|
| `export_scfoundation_gene_embeddings.py` | scFoundation checkpoint、gene list | gene→768 维 embedding |
| `precompute_pathway_embeddings.py` | pathway YAML、gene embeddings | `pathway_base.pt`，每条 pathway 的 mean embedding |
| `export_prototype_assignments.py` | Stage-1 zq/ze/code_ids、metadata、checkpoint、prototype config、excluded samples | `assignments.npy [N,K]`、`prototype_artifacts.pt`、`sample_ids.pt`、manifest |
| `prepare_prototype_population_cache.py` | assignments、sample ids、prototype artifacts、可选 expression/metadata | 每样本 population records、expression、cell-type stats、`R[K,C]` |
| `prepare_prototype_pair_records.py` | population cache、control-treatment metadata、drug sign CSV | 包含 l0/l1、presence、drug ids、比例/表达 target 的 pair-record PT |
| `stage1_boundary.py` | Stage-1 config + checkpoint | 冻结的 post-quant projection + decoder |
| `train_prototype_mass_flow.py` | Stage-2 config、pair records、pathway base、torchrun DDP | last/best flow checkpoint、split manifest、TensorBoard |
| `evaluate_prototype_mass_flow.py` | config、checkpoint、pair records、split manifest、可选 population cache | proportion CSV、prototype diagnostics、expression tensors、14 指标 JSON |
| `audit_pathway_celltype_route.py` | v8 checkpoint、validation records、population cache | `original/w_zero/w_shuffle/condition_exchange/adaln_gate_zero/legacy_static` 干预结果 |
| `evaluate_with_h163_prediction_only.py` | 既有模型产物 + H163 Stage-1 export | 将无标签/额外样本加入 prediction-only 汇总 |
| `visualize_pathway_celltype_adaln.py` | checkpoint、records | A、W、A@W、sample-level contribution 的 CSV/PNG |
| `visualize_pathway_celltype_effects.py` | checkpoint、records、split | 单 pathway intervention、counterfactual proportion/expression、heatmaps |

### 数据泄漏边界

- `export_prototype_assignments.py` 支持通过 `--exclude-samples` 在 prototype fitting 时排除样本，这是正确接口。
- 当前 pipeline shell 脚本排除了 12 个 strict-test samples，但没有排除 21 个 validation samples。
- 对 `celltype_balanced` prototype selection，validation labels 因而参与 tuple 选择；validation 不是完全独立的模型选择集合。
- `prepare_prototype_population_cache.py --metadata-csv` 会用全量 labels 重新计算 `prototype_to_celltype`。同时它会保存 per-sample count matrix；当前 Stage-2 trainer 在这些 counts 存在时会按 train records 重建 `R`，从而保护 strict test。
- 如果旧 artifact 没有 `prototype_celltype_counts_by_sample`，trainer 会 fallback 到 cache 内全量 `R`，此时会发生 label leakage。
- 建议未来 artifact 中强制写入 fit sample ids 和 split hash，trainer 不允许静默 fallback。

### `stage1_boundary.py` 注意点

- checkpoint 以 `strict=False` 构建完整 VQVAE，但对 decoder/pre-quant prefix 做了部分缺失检查。
- 返回对象实际使用 `post_quant_proj` 和 decoder；检查列表却主要检查 decoder 与 `pre_quantization`，契约不完全对称。
- `decode_product_code` 假定 `post_quant_proj.groups` 可转为整数；reshape projection 可行，attention/identity projection 未必可行。

## 5.7 `experiments/scfoundation_vqvae/cell_embedding/`

| 脚本 | 输入 | 输出/用途 |
|---|---|---|
| `prepare_h5ad_subset.py` | h5ad + 精确 obs filters | 不修改源文件的 subset h5ad |
| `run_embedding_heatmap.py` | h5ad、scFoundation checkpoint、gene index、DDP | chunked pooled `[N,3072]` embeddings、metadata、QC |
| `run_token_embedding.py` | 同上 | padded `[N,S,768]` token embeddings、mask、lengths、shards |
| `check_and_resume_embedding.py` | embedding memmap、chunk metadata | 找出未写完 chunk，可续跑 |
| `merge_embedding_batches.py` | 多批 embedding 输出 | 按 cell id/metadata 对齐的 merged arrays |
| `precompute_targets.py` | h5ad、embedding metadata、HVG list | 与 embedding 行对齐的 target NPY；缺失 gene 填 0 |
| `precompute_aligned_batch_targets.py` | config/批次 | aligned target batch |
| `run.py` | Stage-1 config module、stage | 调用 `model.train` 进行 VQ-VAE/prior 训练 |
| `analyze_hypoblast_stage1_codes.py` | pooled embeddings、Stage-1 checkpoint/config | `cell_ze.npy`、`cell_zq.npy`、`code_ids.npy`、code usage report |
| `summarize_stage1_codebook_celltypes.py` | code ids、metadata、decoder | code↔cell-type counts、purity、代表 gene、报告/heatmap |
| `evaluate_test_set.py` | Stage-1 checkpoint、dataset/split | reconstruction、code usage、latent plots、annotation benchmark |
| `annotation_comparison.py` | scFoundation/z_e/z_q、labels、固定 split | 每种 frozen representation 的 MLP probe、F1/accuracy/ARI/NMI、plots |
| `plot_full_ze_umap.py` | embeddings、metadata、可选 annotation split | Scanpy UMAP、可选 MLP annotation |
| `postprocess_heatmap.py` | pooled embedding + metadata | cell-type mean、Pearson/Euclidean heatmaps |
| `scfoundation_annotation_retrain.py` | h5ad、scFoundation、labels | 轻量 fine-tuned annotation model、history、prediction |
| `add_pretrain_lineage.py` | metadata `Final_cell_type` | 规则映射的 broad lineage column |
| `run_hypoblast_stage1_code_analysis.sh` | input h5ad、output dir、filters | 串联 subset→scFoundation→Stage-1 code analysis |

`run_hypoblast_stage1_code_analysis.sh` 中最后使用的配置路径是 `configs/stage1_softvq_hypo_hvg5000.py`，而当前仓库配置实际位于 `configs/stage1/stage1_softvq_hypo_hvg5000.py`；该脚本在当前快照中很可能需要修正路径才能运行。

## 5.8 `common/` 与 `evaluation/`

| 模块 | 输入 | 输出 | 风险 |
|---|---|---|---|
| `common/evaluate_all.py` | truth/pred/control `[samples,genes]` | 11 个表达指标 | energy distance 用完整 `cdist`，O(n²)；Spearman 用双 argsort，不处理 ties；calibrated R² 在同一评估集拟合校准，偏乐观，只能作诊断 |
| `evaluation/losses.py` | reconstruction、latent 或 prior logits | normalized MSE、VQ losses、prior NLL | `x_train_var` 无最小值保护 |
| `evaluation/metrics.py` | assignments/ids/images | perplexity、usage、active code、SSIM | 部分 image 指标是 legacy |
| `evaluation/validation.py` | tuple-style DataLoader、VQVAE/prior | old validation metrics | 假设 MNIST tuple batch 和图像 code grid，不适用当前 scRNA dict batch |
| `evaluation/visualization.py` | image tensors/code counts | PNG grids/histogram | MNIST legacy |

`evaluate_prototype_mass_flow.py` 另报告：

- cell-type proportion MAE/RMSE
- prototype JS、presence precision/recall
- pseudo-bulk MSE
- gene-wise/sample-wise PCC
- matched-control DEG overlap
- `evaluate_all` 的 R² delta、raw PCC、per-gene R²、calibrated R²、energy distance、Spearman 等

需要特别注意：

- raw-sample PCC 很容易被不随扰动变化的基础表达主导，不应单独作为响应效果。
- calibrated R² 在评估数据本身上拟合线性校准，不能代表部署时无需标定的泛化性能。
- test 只有 12 个样本，per-gene R² 分母很不稳定；极端负值应报告分布、低方差基因和 finite count，而不是只报均值。

## 5.9 配置、脚本、测试与 benchmark

### 配置族

- `configs/stage1/`：HVG5000/HVG2000、baseline、cell-type CE、class-balanced CE、SupCon、codebook init、PD/public union。
- `configs/stage2/v7_hvg2000/`：Stage-1 三变体 × hard-sign/low-rank × global/balanced prototypes，以及 18/19-path、pathway-prototype AdaLN 分支。
- `configs/stage2/conditional_*`：更早的 HVG5000/13-path/19-path实验线。
- `configs/legacy/`：MNIST、早期 scFoundation VQ 结构。

配置优点是实验差异可见；缺点是：

- 大量绝对路径绑定单一服务器目录。
- 没有统一 schema、版本约束和 artifact hash 强制校验。
- 有些配置通过 `deepcopy`/import 继承，审计单文件时容易漏掉真实值。
- 同一参数既可能在 config 中出现，又被 shell CLI 覆盖。

### 运行脚本

- `prepare_v7_hvg2000_*_pipelines.sh`：Stage-1 inference → prototype export → population cache → pair records。
- `run_v7_hvg2000_*_queue.sh`：等待资源/前一任务后，以 4-GPU torchrun 训练并单 GPU strict evaluation。
- 均硬编码 `/liaozizhuo/hypoblast`，依赖外部共享结果目录。
- `--allow-skipped` 会让缺失/未知记录被跳过；虽输出 skipped 清单，但 benchmark 前应强制核对数量和原因。

### 测试

仓库共有 6 个 `tests/perturbation/test_*.py` 文件、87 个测试函数，覆盖：

- 19-path vocabulary 与 baseline config 不变量。
- known/unknown pathway effect 和组合不变性。
- pathway-prototype conditioner 与 AdaLN zero-init。
- pathway-celltype auxiliary losses、route dropout、interventions。
- benchmark manifest、finalist selection、strict reporting 和 hash/cohort 校验。
- pathway/prototype 权重导出。

明显缺口：

- 没有 `ScRNADataset`、Stage-1 split、codebook init leakage 边界的测试。
- 没有 SoftVQ 熵损失/`avg_probs`/usage buffer 的测试。
- 没有 Stage-1 train loop、DDP empty-shard 或 `x_train_var=0` 测试。
- 没有验证 state delta 是否应随 ODE 积分的契约测试。
- 没有完整环境/小数据 end-to-end CI。

### Benchmark 基础设施

`.skill-build/factorial-model-benchmark/` 和 `benchmarks/**` 管理：

- 预注册 manifest。
- config materialization。
- 远端训练/评估命令。
- result/evidence 收集。
- 指标总表、Pareto/filter、报告和 figures。
- checkpoint/config SHA-256、cohort invariants。

这是仓库工程质量较强的一部分，但报告中保留的远端绝对路径不代表当前 ZIP 内仍有对应 artifact。

---

## 6. 方法合理性评估

## 6.1 Stage-1：scFoundation + SoftVQ

### 合理性

- 使用大模型 embedding 减少直接处理 2.55M × 全基因矩阵的训练成本，工程上合理。
- factorized codebook 将组合容量从单码本 128 扩展到理论上的 `128^4`，同时每组只有 32 维，表达能力强。
- soft assignment 可避免 hard VQ 初期不稳定，重建梯度也能直接更新 codebook value。
- cell-type CE/SupCon 作为辅助目标可以让 latent 更适合下游类别结构。

### 局限

- `128^4` 理论容量不等于有效容量；下游只选 128 个 tuple，瓶颈最终仍非常强。
- SoftVQ 没有 commitment/codebook loss，码本可解释性主要依赖重建与 entropy；code 可能是分布式连续基底，而非离散生物状态。
- cell-type 监督提高 annotation 不等于提高 perturbation response；12-model benchmark 已显示目标间明显冲突。
- pooled `[3072]` 固化了 scFoundation 的池化信息，后续无法恢复 gene-token attention。
- Stage-1 的 cell-level split 和下游 strict-test 之间并非完全隔离，需明确其 transductive 属性。

### 结论

Stage-1 适合作为压缩和结构化表征器，但不能仅凭 code-celltype purity 将 code 命名为细胞命运或功能状态。若要声称对新 biological samples 泛化，应重新做全 pipeline 的 sample/donor holdout。

## 6.2 Product-tuple prototype

### 合理性

- prototype 是完整四码本组合，语义比独立 group histogram 更完整。
- frequency selection 对常见状态稳定，celltype-balanced selection 可防止大类完全占据 vocabulary。
- `prototype_to_celltype` 使用软 count 而非硬标签映射，可表达混合 prototype。

### 局限

- celltype-balanced selection 使用标签，本身是监督式模型选择步骤，必须只在 train split 拟合。
- 限制到 128 个 tuples 后重新归一化，未报告每个样本被保留的原始 probability mass；若 coverage 低，比例会被系统性扭曲。
- rare/transition state 可能因频率低被删掉；恰好这些状态可能是扰动响应重点。
- 对没有训练 count 的 prototype，`R` fallback 为均匀 cell type，虽防 NaN，但生物含义任意。

### 结论

该设计比早期 factorized marginal population 更可靠，但应新增 coverage 指标，并把 prototype fitting 绑定 train split hash。

## 6.3 Drug→Pathway prior

### 合理性

- known sign + positive magnitude 保留已知方向，同时允许强度学习。
- unknown 用 rank-4 factorization，比全 `D×P` 自由参数更稳健。
- multi-drug residual 零初始化且 singleton gate 为 0，便于从 additive baseline 开始。
- drug set 的求和/均值使顺序不敏感。

### 局限

- 24 个 drug 仅各有一条 known pathway edge，真实多靶点机制远比此复杂。
- 无 dose、duration、batch、donor、baseline state 之外的协变量。
- unknown 与“确实没有作用”混在一起；低秩分支可在几乎所有未列边上产生作用。
- `tanh` 既提供稳定性也造成饱和，强组合之间难以区分。
- pathway gene mean pooling忽略 pathway 内激活/抑制方向和基因重要性。

### 结论

适合作为可审计的弱先验，不适合称为真实药物机制模型。推荐同时提供 drug-only/no-prior baseline，以判断 pathway prior 是否真的带来预测增益。

## 6.4 Conditional prototype mass flow

### 合理性

- 对 sample population 建模符合数据粒度。
- self-attention 可建模 prototype 竞争/共变，cross-attention 可读 pathway 条件。
- random-time CFM 加 source-time loss，兼顾全路径与实际推理起点。
- rollout endpoint 的 proportion/expression 监督使训练目标更接近最终任务。
- AdaLN-Zero 的 identity initialization 便于做增量机制实验。

### 局限

- 训练目标是确定的 control→treated pair，线性 CFM target 恒为 `l1-l0`；在样本数只有 181、模型 hidden/FFN 很大的情况下，模型容量相对数据量很高。
- mass 是 CLR 坐标，目标 velocity 为零均值，但网络输出没有强制去均值；共同平移在 softmax 后不可识别。
- rollout 固定 20 步 Euler，未见 solver/step sensitivity 作为主报告的一部分。
- source presence 在整个 ODE 中固定，terminal presence 不改变轨迹内部表征。
- state delta 不积分；它更像终点 decoder residual head，而不是 flow state。
- `terminal_proportion` 参数传入 loss，但没有直接参与 loss；prototype-level 终点比例只通过 `terminal_mass_logit` 的 flow target间接监督。主比例损失监督的是 cell-type proportion。
- decoder 对每个 `prototype + state_delta` 独立解码再按比例平均。由于 decoder 非线性，该结果不等价于对群体 latent 均值解码；同时 state delta 可能离开 Stage-1 latent manifold。

### 结论

作为 sample-level conditional dynamics 的工程实现是合理的，但“flow”主要严格适用于 mass logits；presence 和 expression state 更像 endpoint heads。论文或报告应按这个实际边界描述。

## 6.5 显式 Pathway→Prototype / Pathway→Cell type 路线

### 合理性

- 显式 `W[P,K]` 或 `W[P,C]` 提供可审计的结构瓶颈。
- v8 用 train-only `R[K,C]`、多 seed、route interventions 和 strict test，方法学比只看热图明显更强。
- W-zero、W-row-shuffle、AdaLN gate-zero 均改变输出并降低主指标，说明不是纯旁路参数。

### 局限

- 权重稳定和通路被使用不等于每个条目具有生物学因果含义。
- `activity × W × pathway_embedding × projection × AdaLN` 有多层尺度和符号自由度。
- 低维 cell-type bottleneck 可能牺牲 prototype 内部的连续状态差异。
- v8 strict test 的结果明确显示机制增强并未提升预测。

### 结论

保留为机制研究分支，不应替换 canonical baseline。下一轮更值得测试“baseline 主干 + 小型 residual correction”，而不是继续增加 AdaLN 深度。

---

## 7. 实验演进：按文档日期与证据顺序

归档解压后大多数文件的 mtime 被统一为 2026-08-23 18:39:39，不能依靠文件系统时间精确排序；以下顺序采用文档文件名日期、registry 中的 audit 日期和 benchmark 报告自身说明。

### 7.1 2026-07-24：population output supervision 设计

设计重点是把 Stage-1 code 表征聚合到群体层面，再做扰动预测。README 仍大量保留这一早期表述，例如 13-path、边缘 codebook-token population、无额外 presence/proportion/expression loss。当前代码已明显演进，README 不再是准确实现说明。

### 7.2 2026-07-30：Stage-1 iPStem 恢复

恢复 scFoundation + VQ-VAE 训练、预计算 embedding/target、DDP 和 checkpoint 兼容。Stage-1 registry 后续记录了原始、updated、v7、HVG5000/HVG2000 等多组状态。

### 7.3 2026-08-12：HVG2000、cell-type 与 19-path 方向

- 引入 matched HVG2000 baseline 与 cell-type CE 对照。
- 清理旧 DNMT active 路径。
- 建立 hard-sign + low-rank unknown pathway 设计。

HVG2000 两个 matched Stage-1 配置除 cell-type classifier/CE 外，数据、split、模型和随机 codebook init 相同，是相对干净的对照。

### 7.4 2026-08-13：Stage-1 registry

登记显示：

- 原始历史流程中，`z_q` annotation test accuracy `0.8510`，优于 scFoundation `0.7807` 和 `z_e 0.8406`。
- updated 11-class 实验中，cell-type CE11 的最佳 frozen representation 为 `z_e`，accuracy `0.8182`；SupCon + random init 的 macro F1 `0.7935`。
- v7 HVG5000 cell-type checkpoint 与 annotation 已完成；HVG2000 baseline/cell-type 的 summary 当时尚未完成。
- registry 是 2026-08-13 的服务器审计快照，不等于 2026-09-03 当前真实远端状态。

### 7.5 2026-08-14 至 2026-08-17：prior、初始化和 mid-flow bridge 计划

这些 docs 记录 low-rank unknown、projected prior 初始化、mid-flow edge bridge 等设计/计划。当前 ZIP 的已提交主 benchmark 证据集中在后续 v7/v8 报告，不能仅凭 plan 文件声称这些所有变体均已完成并成功。

### 7.6 v7 HVG2000：8 模型 factorial benchmark

轴为：

- Stage-1：baseline / cell-type CE
- pathway effect：hard-sign / low-rank unknown
- prototype selection：global / celltype-balanced

strict test 为 12 个 samples、20 ODE steps、单 seed。结论：

- 比例最佳：`celltype_hard_sign_global`。
- 响应与表达结构最佳：`celltype_lowrank_balanced`。
- balanced prototypes 对表达响应指标总体更有利。
- low-rank unknown 通常改善表达响应，却可能牺牲比例误差。
- 不存在对所有目标的单一赢家。

### 7.7 v7 最终 12 模型

在上述 8 个模型基础上加入 4 个 class-balanced Stage-1 变体。

| 工作点 | 代表模型 | 关键结果 |
|---|---|---|
| 组成 | `celltype_hard_sign_global` | MAE `0.038722`、RMSE `0.082228` |
| 扰动响应 | `celltype_lowrank_balanced` | PCC sample `0.693102`、PCC gene `0.475293`、R² delta `0.350239` |
| Cell-type expression / DEG | `celltype_balanced_lowrank_balanced` | CT MSE `0.037015`、DEG@20 `0.545833` |
| 聚合表达 | `baseline_lowrank_balanced` | PB MSE `0.009030`、raw PCC `0.938850` |

新 class-balanced Stage-1 四组的 per-gene R² 为 `-338.28`、`-342.07`、`-132.66`、`-152.48`，远差于历史模型约 `-5` 至 `-13`。因此它们只能作为局部指标候选，不能作为默认模型。

### 7.8 2026-08-20：18-path Pathway→Prototype AdaLN

公平比较 18-path legacy control 与 direct、known prior、known+unknown 三种 AdaLN：

- control 在 14 项中的 13 项最好。
- `adaln_known_prior` 是三种候选中相对最好，但仍未达到 control。
- unknown branch 的有效 drug-pathway effect 平均绝对值只有 `0.000489`。
- 当时只有参数终点审计，尚未证明通道被因果使用。

结论：不替换 baseline。

### 7.9 2026-08-23：19-path baseline

在 matched 条件下只增加独立 `nodal` pathway：

- 19-path 相对 18-path 在 14 项中 12 项更好。
- PCC gene 从 `0.387570` 到 `0.475293`。
- R² delta 从 `0.275835` 到 `0.350239`。
- E-distance 从 `1.785609` 降到 `1.639225`。
- proportion MAE 略差 `0.000026`，per-gene R² 也变差。

这只是单 seed 对照，不能证明提升由 NODAL 的生物因果作用导致；但它足以支持将 19-path 配置设为后续工程 baseline。

### 7.10 2026-08-23：v8 Pathway→Cell type AdaLN

Screen 比较 baseline 和 M1–M7：

- 7 个候选都通过 mechanism gate。
- 所有候选在 validation 三项预注册主指标 `PCC_gene`、`Spearman`、`DEG@20` 上都低于 baseline。
- M2 是候选内部 Pareto finalist，因此进入 seed 42/43/44 confirmation。

M2 的 W route 审计：

- W-zero、W-row-shuffle、AdaLN gate-zero 在三个 seed 中均使三项主指标下降。
- W 的 pairwise Spearman 中位数 `0.840358`。
- top-quartile strong entries 跨 seed 符号一致率 `0.8125`。

Strict test 三 seed 均值：

| 指标 | Baseline | M2 | 更好 |
|---|---:|---:|---|
| Proportion MAE | `0.045469 ± 0.002140` | `0.054701 ± 0.004907` | Baseline |
| PB MSE | `0.009450 ± 0.000359` | `0.011076 ± 0.001588` | Baseline |
| PCC gene | `0.415085 ± 0.052840` | `0.386517 ± 0.051495` | Baseline |
| R² delta | `0.298996 ± 0.044402` | `0.175843 ± 0.096630` | Baseline |
| DEG@20 | `0.534722 ± 0.016839` | `0.493055 ± 0.034944` | Baseline |
| Spearman | `0.506467 ± 0.012231` | `0.471154 ± 0.044786` | Baseline |

14 个 strict aggregate 指标全部由 baseline 更优，最终 `winner = null`。这说明“机制通路确实存在并被模型使用”与“预测能力更强”是两件不同的事。

---

## 8. 风险清单与优先级

## P0：阻断独立复现

### P0-1 无环境定义且当前 Python 不兼容

- 当前 `/usr/bin/python3` 为 Python `3.6.8`。
- 多个文件使用 `from __future__ import annotations`，Python 3.6 无法识别。
- 代码还使用 `str | Path`、built-in generic annotations、`torch.load(weights_only=True)`、SDPA/Flash attention 等较新特性。
- 仓库没有 requirements/conda/pyproject lock。

建议：提供至少 Python 3.10/3.11 的 lockfile、PyTorch/CUDA 版本、scipy/sklearn/pandas/h5py/scanpy/pyyaml/tensorboard/torchvision/pytest 版本，并增加 CPU 小数据 smoke CI。

### P0-2 ZIP 缺少所有大数据与模型 artifact

建议：至少提供 `ARTIFACT_MANIFEST.md/json`，列出每个外部 artifact 的 URI、shape、dtype、SHA-256、生成命令和授权方式；不必把几十 GB 数据提交到 Git。

## P1：可能影响实验有效性

### P1-1 Stage-1 不是完整 sample-level holdout

当前配置只强制 `H226/H227/H240` 进入 Stage-1 test，其余补足 test fraction 的行按 cell 抽样，validation 也按 cell 抽样。更重要的是 Stage-2 strict-test 的另外 9 个 sample 可能被带 cell-type CE 的 Stage-1 看见。

影响：现 strict test 证明的是“冻结表征已接触部分样本后，Stage-2 对新 perturbation pair 的泛化”，不是完全新 sample 的端到端泛化。

修复：按 sample/donor/condition group 一次性生成全 pipeline split，并让 Stage-1、prototype fitting、R fitting、Stage-2、评估共享同一 split manifest。

### P1-2 Validation label 参与 celltype-balanced prototype selection

当前 preparation shell 的 excluded list 恰好是 12 个 strict-test samples，未排除 21 个 validation samples。celltype-balanced tuple selection 会使用 validation labels。

影响：strict test 仍独立，但 validation screen/finalist selection 可能偏乐观。

修复：prototype vocabulary 只在 Stage-2 train samples 上拟合；validation/test 仅 transform。

### P1-3 `prototype_to_celltype` 的静默 fallback 泄漏

trainer 在 per-sample counts 存在时按 train sample 重建 R，这是正确实现；若旧 payload 没 counts，则直接使用 cache 中可能由全量 labels 构建的 R。

修复：删除 fallback，或要求 artifact 明确证明 R 的 fit split 与当前 train split hash 相同。

### P1-4 Stage-2 validation checkpoint score 随机

`_mean_terms` 在 `eval()` 下调用 `loss_terms`，后者仍 `torch.rand` 采样 time。因此相同 checkpoint 的 validation `base_total` 不完全确定。

修复：validation 固定一组 time grid 或固定 per-record t；checkpoint selection 更应使用确定的 rollout endpoint 指标组合。

### P1-5 数据量与模型容量不匹配

已提交 v8 split 为 train 148、validation 21、test 12 个 pair records；flow 为 3 层、model dim 256、FFN 4096，另带多种条件分支。强正则、简单 baseline、多 seed 和更小模型消融都很重要。

## P2：模型语义或数值稳定性

### P2-1 fixed source presence

ODE 每一步都使用 `initial_presence`，终点预测 presence 不反馈轨迹。新 prototype 的出现和旧 prototype 的消失不能动态改变 token condition。

建议：将 presence logit 作为联合连续状态积分，或至少在 rollout 中平滑更新 presence embedding，并做 ablation。

### P2-2 state delta 不是 flow state

表达预测所用 `state_delta` 仅来自终点 head，不沿时间积分。建议改名 `terminal_state_residual`，或将 prototype state 一并纳入 ODE。

### P2-3 CLR velocity 未投影到零和子空间

建议对预测 velocity 做 `v = v - mean(v)`，消除 softmax 不可识别的 common mode。

### P2-4 SoftVQ 诊断缺陷

- `avg_probs` 恒定。
- usage buffer 对 code 0 有早期偏置。
- entropy temperature 与 assignment tau 隐式分离。

建议修正并用测试覆盖。

### P2-5 方差归一化与 per-gene R²

- reconstruction denominator 没有 clamp。
- per-gene R² 对 12 samples 和低方差 genes 极不稳定。

建议输出每基因 variance、R² quantiles、finite count、低方差过滤前后结果、PCC/MSE/MAE 分布。

### P2-6 prototype coverage 未报告

建议每个样本记录：selected tuples 覆盖的原始概率质量、renormalization factor、未覆盖 hard-code 比例、按 cell type 的 coverage。

## P3：工程债务与文档

- README 的 13-path、population state、decoder 维度和 Stage-3 loss 描述已过时。
- MNIST/PixelCNN API 留存但主 VQVAE 禁用 spatial path。
- legacy/current 文件共存，入口层次不够清晰。
- 配置和 shell 大量绝对路径。
- 没有统一 artifact schema/version。
- `run_hypoblast_stage1_code_analysis.sh` 配置路径疑似过时。
- Stage-1 关键逻辑没有测试。
- train loop 为 code stats 重复前向，增加计算成本。

---

## 9. 推荐的下一步

### 第一优先：先把评估边界做干净

1. 建立唯一 `global_split_manifest.csv`，按 biological sample/donor 分 train/val/test。
2. Stage-1 supervised loss、codebook init、prototype selection、`R[K,C]` 拟合全部只用 train。
3. validation 只用于 model selection；strict test 在配置冻结前不可读。
4. 明确当前历史结果属于 transductive Stage-1 还是 inductive end-to-end 设置。

### 第二优先：完成环境和 artifact 复现

1. 新增 `environment.yml` 或 `pyproject.toml + lock`。
2. 将所有绝对路径改为项目根目录、环境变量或 CLI。
3. 为每个大文件记录 shape/dtype/hash/provenance。
4. 提供合成小数据，使 `prepare → train 1 step → evaluate` 能在 CI 跑通。

### 第三优先：修正确定的实现/诊断问题

1. 固定 validation time grid。
2. 修复 SoftVQ `avg_probs` 和 usage warm-up。
3. 给 `x_train_var` 加 epsilon 和 finite check。
4. 禁止旧 `R` fallback。
5. prototype selection 强制接收 train sample manifest。
6. 从 `VQVAE.forward` 返回 indices，避免训练后二次 encode。
7. 更新 README，并将 legacy MNIST 明确隔离。

### 第四优先：有针对性的模型实验

1. 为 20-step Euler 做 5/10/20/50 steps sensitivity。
2. 比较 fixed presence、joint presence dynamics、无 presence head。
3. 比较 endpoint state residual 与真正 joint state flow。
4. 加入 control-only、drug-id-only、pathway-sign-only 简单 baseline。
5. 对当前 canonical `celltype_lowrank_balanced` 至少补 3–5 seeds。
6. v8 后续只测试小型 residual W-AdaLN，不再优先堆叠更深同类结构。
7. 在完全未读的新 test cohort 上验证后，才讨论 pathway-cell-type 热图的外部生物一致性。

---

## 10. 静态验证记录

已完成：

- ZIP CRC 完整性校验通过。
- ZIP 路径安全检查通过。
- 盘点源码、配置、docs、benchmarks、tests 和 shell pipelines。
- 核对当前 19-path/24-drug vocabulary。
- 核对 v7 8-model、v7 final-12、18→19-path、18-path AdaLN、v8 multi-seed strict 报告。
- 核对关键训练/评估代码的数据 split、prototype fitting、R fitting 和 loss 实现。

未能完成：

- **Python compile/test**：当前系统 Python 3.6.8 在解析 `from __future__ import annotations` 时失败；服务器未发现 `pytest` 命令。
- **端到端运行**：ZIP 不含数据、checkpoints 和依赖环境。

语法检查失败属于运行环境版本不满足代码要求，不代表列出的源码文件自身一定存在语法错误。要得到可信测试结论，应在项目原始 PyTorch/CUDA 环境或新建的锁定环境中运行全部 87 个 tests。

---

## 11. 最终结论

1. **代码主线逻辑清晰**：scFoundation cell representation → 4-group SoftVQ → product-tuple prototypes → sample population → drug/pathway-conditioned prototype mass flow → cell-type/expression output。
2. **当前最可靠的 Stage-2 响应基线仍是 19-path `celltype_lowrank_balanced` legacy dynamic-token 模型**；不要因显式通路更“可解释”就替换预测更好的 baseline。
3. **显式 pathway→cell-type 通道已被证明被模型使用，但未带来预测提升**；权重只可解释为模型内部机制，不能直接解释为生物因果效应。
4. **strict Stage-2 test 的样本级隔离较好，但整个 pipeline 不是完全 sample-level inductive**，因为 Stage-1 和 prototype fitting 的边界还不够严格。
5. **最大短板不是继续增加模型结构，而是复现环境、跨阶段 split、prototype coverage、确定性 validation 和逐基因指标诊断。**

建议在对外报告中把当前结果定位为：**小样本、样本级 Stage-2 holdout、部分 transductive representation 的探索性扰动预测结果**。完成全流程 sample-level holdout、多 seed 和独立 cohort 验证后，才适合升级为强泛化或机制结论。

