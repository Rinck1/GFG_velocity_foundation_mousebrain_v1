# Hypoblast 项目目的、科学问题与验证路径

> 本文结合项目所有者对终极目标的说明，以及当前代码、配置和实验设计，重新界定整套系统要解决的问题。其核心不是为了预测而预测，而是学习可用于选择干预策略的 `Pathway activation → terminal cell type` 关系。

## 一句话概括

Hypoblast 的终极目标，是学习一条可信、可干预并具有上下文依赖性的关系：**一条或多条 pathway 的激活或抑制，会怎样改变扰动后的终末细胞状态和各细胞类型占比**；进而能够针对给定的目标细胞类型，寻找使其最终占比最大的 pathway 或 pathway 组合。

换一种更直观的表达：

```text
学习：初始细胞群 + pathway 激活组合
      → 扰动后 latent 细胞状态
      → 终末 cell-type composition

再反向询问：为了最大化目标 cell type c* 的最终占比，
应该激活或抑制哪一条、哪几条 pathway？
```

由于不同初始状态、不同扰动和不同 pathway 组合可能产生不同的终末细胞类型，这不是一张固定的“pathway 对应 cell type”查找表，而应是一条条件响应函数。模型至少需要正确表示：

1. 扰动后的目标细胞状态在 latent 空间中的位置和分布。
2. 各种 cell type 的终末比例及其相对 control 的变化。
3. 整体和各 cell type 的基因表达状态。
4. 单 pathway 与 pathway 组合的作用、协同、拮抗及上下文依赖性。

---

## 1. 项目真正要回答的科学问题

这套代码最终试图回答的是：

> 对于一个给定的初始细胞群，如果能够调节某一条或若干条 pathway，哪些调节方案最有可能把系统推向研究者希望得到的目标细胞类型，并使该细胞类型在终末群体中的占比最大？

要可信地回答这个逆向设计问题，必须先解决四个前置问题。

### 1.1 如何准确表示扰动前后的细胞状态

不同样本包含的细胞数量不一致，直接比较两堆细胞比较困难。项目首先需要建立一套所有样本共享的细胞状态“词表”。

Stage-1 使用冻结的 scFoundation 表征和 SoftVQ，将每个细胞编码成四个 codebook 上的软离散状态。四个 code 的联合组合进一步形成 population model 使用的 prototype。

这一步希望实现：

- 将高维基因表达压缩为低维状态。
- 在不同样本之间复用同一套状态单位。
- 保留比单纯 cell-type label 更细的细胞内状态差异。
- 让样本可以表示为有限个 prototype 的组成，而不是数量不等的原始细胞集合。

因此，cell-type annotation、codebook purity、UMAP 和 expression reconstruction 的作用，是验证 latent 空间是否保留了终末命运和转录状态所需的信息。它们不是项目的终点，却是后续学习 pathway→cell type 关系的表示基础。

尤其需要验证：给定 control latent 和真实扰动条件后，模型预测的 terminal latent 是否能接近真实 treated latent。如果 latent 终态本身恢复不准，那么由它导出的 cell-type proportions、pathway 权重和最优 pathway 组合都缺乏可信基础。

### 1.2 如何描述扰动后的完整细胞群

Stage-2 不直接处理每个细胞，而是将同一样本内的细胞聚合为 prototype population：

```text
一个样本
→ 128 个 prototype 的比例
→ 每个 prototype 是否有足够细胞支持
→ prototype mass 的 CLR 坐标
→ 可选的总体和分 cell-type 表达状态
```

这种表示的目的，是把变长的细胞集合转成固定维度、可比较、可进行动力学建模的样本级状态，同时保留“哪些终末状态增加或减少”的信息。

### 1.3 如何从观测到的药物扰动中识别 pathway 作用

项目不希望只把药物名称当作一个没有结构的类别 ID，而是加入了明确的生物知识路径：

```text
Drug
→ Pathway activity
→ Prototype 或 cell type
→ Population change
```

已知药物—通路关系提供激活或抑制方向，模型学习作用强度；未列出的潜在关系可以选择固定为 0，或者通过低秩 unknown-effect 分支学习。

对多药组合，模型还允许出现偏离简单相加的作用，从而尝试表达协同或拮抗。

这里包含三个目标：

- 通过 pathway 共享结构，提高有限训练样本下的学习效率和对组合条件的泛化能力。
- 从不同药物及组合的观测中，学习 pathway activation 与 terminal cell state/cell-type proportion 之间的条件关系。
- 产生可检查的中间变量，使研究者能够追踪某个药物条件通过哪些 pathway 影响哪些 prototype 或 cell type，并为之后的 pathway 组合搜索提供接口。

### 1.4 如何从前向预测走向目标细胞类型的逆向设计

条件流模型接收：

- control sample 的 prototype mass 与 presence。
- 药物组合产生的 pathway tokens。
- 可选的 pathway→prototype 或 pathway→cell-type context。
- 连续时间变量 `t`。

模型学习从 control population 到 treated population 的 mass velocity，并通过数值积分得到终点状态。终点再被转换为：

- prototype proportions。
- cell-type proportions。
- pseudo-bulk expression。
- cell-type-specific expression。

前向模型可以抽象为：

```text
z_hat_1 = F_theta(z_0, a)
p_hat   = G(z_hat_1)
```

其中 `z_0` 是初始 population latent，`a` 是 pathway 激活/抑制向量，`z_hat_1` 是预测的扰动后 latent，`p_hat` 是终末 cell-type proportion。

终极目标是针对目标细胞类型 `c*` 求解：

```text
a* = argmax_a p_hat[c*]
```

实际应用中还应加入可实施性、稀疏性、剂量、安全性和组合规模等约束，例如惩罚过多 pathway 同时改变。由于最优 pathway 可能依赖 `z_0`，合理的答案通常是“在某类初始细胞状态下，某组 pathway 最优”，而不是宣称某条 pathway 对所有样本都固定产生同一种细胞类型。

因此，**准确恢复扰动后的 terminal latent 和 phenotype 是识别 pathway→cell type 关系的必要验证环节；学习并利用这条关系进行目标导向的 pathway 选择，才是最终目的。**

---

## 2. 为什么模型工作在样本群体层面

这套代码有一个很重要的方法选择：它不是逐细胞的扰动流。

普通单细胞实验通常在处理前后破坏性取样，同一个细胞无法同时拥有 control 和 treated 两种观测。因而不存在真实的：

```text
control cell i → treated cell i
```

逐细胞强行配对会引入无法验证的假设。项目选择使用真实存在的样本关系：

```text
control sample population
→ treated sample population
```

这使问题的数据粒度与模型粒度保持一致。它关注的是群体组成和分布的变化，而不是声称恢复每一个细胞的真实发育轨迹。

所以，这套模型不应被描述为传统意义上的 RNA velocity，也不是逐细胞 optimal transport。更准确的说法是：

> pathway-conditioned population perturbation dynamics。

---

## 3. 整套技术路线

```text
单细胞表达矩阵
        │
        ▼
冻结的 scFoundation
        │
        ▼
每细胞 3072 维基础表征
        │
        ▼
Stage-1 MLP + 四组 SoftVQ
        │
        ├─ 连续状态 z_e / z_q
        ├─ 四个 codebook assignment
        └─ HVG expression reconstruction
        │
        ▼
完整四码本组合构成 prototype vocabulary
        │
        ▼
按 biological sample 聚合
        │
        ├─ prototype proportion
        ├─ prototype presence
        ├─ CLR mass logits
        └─ population expression
        │
        ├──────────────────────────────┐
        │                              │
        ▼                              ▼
control population               drug combination
                                       │
                                       ▼
                              drug→pathway condition
        │                              │
        └──────────────┬───────────────┘
                       ▼
          conditional prototype mass flow
                       │
                       ▼
              treated population state
                       │
          ┌────────────┴─────────────┐
          ▼                          ▼
cell-type composition          expression response
          │                          │
          └────────────┬─────────────┘
                       ▼
       验证 terminal latent/phenotype 是否准确
                       │
                       ▼
       学习和审计 Pathway → terminal cell type 关系
                       │
                       ▼
  对目标 cell type 进行 pathway 单项/组合搜索与排序
```

---

## 4. 项目的目标层级

这些目标不是并列关系，而是一条从表示验证到科学应用的依赖链。

### 4.1 终极科学目标：学习 Pathway→终末细胞类型关系

模型最终要学习的是一个条件响应面：

```text
(initial cell state, pathway activation pattern)
→ terminal latent state
→ terminal cell-type proportions
```

研究重点包括：

- 某条 pathway 的激活或抑制会提高哪些终末 cell type 的比例。
- 多条 pathway 联合改变时，结果是否可由单项效应相加解释。
- 哪些组合存在协同、拮抗或条件依赖。
- 相同 pathway 在不同 initial state 下是否产生不同终末命运。
- 不同 drug perturbations 是否通过共享 pathway 到达相似终态，或通过不同 pathway 到达不同终态。

### 4.2 必要前提：准确恢复真实扰动后的 latent 终态

在解释 pathway 与 cell type 的关系之前，前向模型必须足够准确：

- `F(z0,a)` 能否恢复真实 treated population 的 latent state。
- 预测与真实终态在 prototype mass、presence 和状态方向上是否一致。
- 终态最近邻、latent correlation、distance 和 distributional metrics 是否可靠。
- 从预测 latent 解码出的 expression delta 是否接近真实扰动。
- 由预测 latent 映射出的 cell-type proportion 是否准确。

这是一个**可信度门槛**：如果真实 perturbation 的 latent 终态都不能恢复，模型对未观测 pathway 组合的排名就没有充分依据。

### 4.3 表示基础：有效的离散细胞状态空间

SoftVQ code 和 product prototype 应当：

- 保留足够的表达和扰动响应信息。
- 覆盖主要及稀有终末状态。
- 在不同样本间具有一致含义。
- 区分 cell type 与同一 cell type 内部的状态差异。
- 支持 population composition、terminal latent 和 expression decoding。

code 不必与 cell type 一一对应；但如果目标是优化某个终末 cell type，占据该 cell type 的 prototypes 必须有足够覆盖率和可校准映射。

### 4.4 知识桥梁：从 drug perturbation 识别 pathway activation

训练数据直接观测的是 drug/small-molecule perturbation，而终极查询对象是 pathway。因而模型必须可靠地区分：

```text
drug identity
→ drug 引起的 pathway activation/inhibition
→ pathway 对 terminal latent 的作用
→ terminal cell-type composition
```

known sign、learned magnitude 和 unknown low-rank effects 是对这一中间识别问题的建模。drug-only 与 no-prior baseline 很重要，因为它们能判断预测到底来自 pathway 结构，还是来自对 drug ID 的记忆。

### 4.5 应用目标：寻找最大化目标细胞类型的 pathway 组合

当且仅当前向模型和 pathway 通路通过验证后，才进入逆向优化：

```text
给定 initial state z0 和目标 cell type c*
枚举或优化 pathway activation vector a
计算 F(z0,a) 与 G(F(z0,a))[c*]
在可实施约束下选择得分最高的单 pathway 或组合
```

最终输出不应只是一个全局榜单，而应至少包含：

- 目标 cell type。
- 适用的 initial state / sample context。
- 推荐激活和抑制的 pathway 集合。
- 预测的目标占比及相对 control 增益。
- 对其他 cell types 和 expression state 的副作用。
- 单项效应、组合增益和不确定性。
- 能够实现这些 pathway 操作的候选 perturbations。

### 4.6 机制审计目标

项目希望建立可以检查的链条：

```text
A[D,P]: Drug → Pathway
W[P,K] 或 W[P,C]: Pathway → Prototype/Cell type
F: Pathway-conditioned population transition
G: Terminal latent → cell-type composition/expression
```

代码通过 W-zero、W-row shuffle、condition exchange、AdaLN gate-zero 和静态 token 等干预判断模型是否真正使用这条路径。这些审计是 pathway 优化的必要条件，但还需要外部扰动实验才能把模型关系提升为生物学因果关系。

---

## 5. 这套代码不以什么为最终目的

### 5.1 不只是 cell-type annotation

cell-type classifier 和 annotation benchmark 是检查 Stage-1 表征的工具。最终目标不是给细胞贴标签，而是学习哪些 pathway 操作能够把群体推向指定的终末 cell type。

### 5.2 不只是表达重建

Stage-1 decoder 的重建损失是为了保证压缩状态仍保存转录组信息。低 reconstruction MSE 不自动代表 terminal latent 恢复准确，更不代表 pathway 排名正确。

### 5.3 不只是把已知药物扰动预测准确

已知扰动预测是模型校准和验证环节。即使 drug-conditioned prediction 很准，如果模型只是记忆 drug identity、没有学到可迁移的 pathway 作用，也没有实现终极目标。

### 5.4 不只是得到漂亮的 pathway 热图

热图显示的是模型内部学到的参数或派生量。只有当相应通路经过干预验证、跨 seed 稳定并且在独立数据上与外部证据一致时，才可能进一步讨论其生物学意义。

### 5.5 不是恢复每个细胞的真实时间轨迹

模型运输的是 sample-level prototype masses。它可以描述群体状态从 control 到 treated 的变化，但不能证明某个特定 control cell 最终变成了哪个 treated cell。

---

## 6. 预测能力与可解释性的关系

在修正后的目标层级中，预测能力不是终极目的，却是学习 pathway→cell type 关系不可绕过的前提。这里应区分三种证据：

1. **前向准确性**：模型能否从 control 和真实 perturbation 恢复真实 terminal latent、cell-type proportions 和 expression response。
2. **通路依赖性**：改变 pathway 输入或中间 W 后，预测是否发生方向合理且可重复的变化。
3. **目标导向有效性**：模型推荐的 pathway 或组合，能否在未用于训练的新实验中提高目标 cell type 的最终占比。

只有第一项，可能只是准确的 drug-response predictor；只有第二项，可能只是一个被网络使用但预测较差的机制通道。项目终极目标要求三者形成闭环。

当前实验已经显示：

- 19-path legacy dynamic-token baseline 在多个扰动响应指标上表现较好。
- 显式 Pathway→Prototype AdaLN 没有超过对应的 legacy control。
- v8 的 Pathway→Cell type W 通路确实被模型使用，而且跨 seed 有一定稳定性。
- 但是 v8 M2 在 strict test 三 seed 平均结果中，14 个聚合指标全部落后 baseline。

当前实验意味着：

> 一个更容易画成机制图的模型，并不自动是更准确的模型；一个参数稳定且被网络使用，也不自动意味着其生物学解释成立。

同时也意味着，现有 baseline 即使预测较准，如果没有证明它在 pathway activation 层面具有可干预、可迁移的响应结构，也还没有完成终极目标。

因此，当前最合理的项目策略是：

1. 先建立能准确恢复真实 terminal latent 的前向主干。
2. 再让 pathway activity 成为主干中不可被绕开的、可干预变量。
3. 通过 zeroing、shuffle、counterfactual、held-out pathway/drug 泛化验证关系。
4. 在模型内进行 pathway 单项和组合搜索，输出目标 cell-type enrichment 排名及不确定性。
5. 用新的 wet-lab perturbation 验证推荐组合，形成 active-learning 闭环。

不应为了得到显式 W 而接受明显失真的 terminal latent；也不应因为黑盒预测准确，就跳过 pathway 层面的机制识别。

---

## 7. 我对项目最终愿景的理解

如果将目前分散在代码和实验中的意图合并，项目的最终愿景应当是：

> 学习一个条件化的 Pathway→terminal cell state/cell-type composition 响应模型：它首先能够在 latent 空间准确恢复已观测扰动后的目标细胞状态，然后能够针对研究者指定的目标 cell type，预测并筛选使其终末占比最大的单 pathway 或 pathway 组合，最终把这些模型建议转化为可实验验证的扰动方案。

Stage-1 离散表征、sample-level population、drug→pathway prior 和 conditional flow 都服务于这个目标：

- Stage-1 提供可比较的初始与终末 latent 状态。
- population aggregation 定义“目标 cell type 占比最大”的样本级输出。
- drug→pathway 映射把已观测小分子实验转成 pathway 层面的训练信号。
- flow 学习在 pathway 条件下从初始状态到终末状态的映射。
- pathway interventions 和 W route audit 检查模型建议是否真正来自 pathway 通道。
- inverse search 将训练好的前向模型用于目标 cell type 优化。

一个成熟版本最终应能回答：

1. 给定 control state 和真实 perturbation，能否准确恢复 treated terminal latent？
2. 预测 latent 是否对应正确的终末 cell type、比例和表达状态？
3. 哪些 pathway 是这一终态转变的必要或促进因素？
4. 对指定目标 cell type，单独激活哪条 pathway 能使其比例最高？
5. 哪几条 pathway 联合激活/抑制能进一步提高目标占比？
6. 组合收益是可加的，还是存在协同或拮抗？
7. 推荐是否依赖初始细胞群，是否会把其他样本推向不同终末类型？
8. 推荐能否映射回可执行的 drug/small-molecule perturbation，并在新实验中得到验证？

前两个问题是当前代码最需要先建立的预测可信度。第三个问题已有条件通路和 route audit 原型。第四至第八个问题是终极目标对应的 pathway optimization 与实验闭环，目前尚未完整实现。

---

## 8. 当前阶段的准确定位

目前最准确的定位不是“已经找到最大化目标 cell type 的 pathway 组合”，而是：

> 一个正在构建 Pathway→terminal cell type 响应函数的研究框架；当前代码已经具备前向扰动建模和 pathway route 审计组件，但 terminal latent 恢复验证、可识别性、逆向组合优化与新实验验证仍需补齐。

它已经具备：

- 大规模单细胞预训练表征接口。
- SoftVQ 离散状态学习。
- sample-level population representation。
- drug/pathway condition encoding。
- prototype mass conditional flow。
- cell-type proportion 与 expression 输出。
- 多模型 benchmark、strict test 和机制 route audit。

但当前实现对终极目标还有几个关键缺口：

- 现有主要评估偏重 cell-type proportion 和 decoder expression，尚未形成完整的 terminal latent recovery 指标体系。
- 训练输入主要是 drug IDs 及先验映射，而不是独立测量的 pathway activity；drug effect 与 pathway effect 的可识别性有限。
- 当前显式 Pathway→Cell type 分支虽然被模型使用，但预测性能弱于 baseline。
- 尚无正式的目标 cell type pathway search/optimization 模块。
- 尚无“模型推荐 pathway 组合 → 新 perturbation 实验 → 结果回流训练”的闭环。

它下一阶段最需要补足的是：

- 从 Stage-1 到 Stage-2 完全一致的 sample/donor-level holdout。
- 只用 train data 拟合 codebook initialization、prototype vocabulary 和 prototype→cell-type mapping。
- 多 seed 和独立 cohort 验证。
- 简单 drug-only/no-prior baselines。
- terminal latent 的 retrieval、distance、correlation、distribution 和 cell-state calibration 指标。
- held-out drug、held-out pathway mechanism 和 held-out combination 泛化。
- 带稀疏性、可实施性和不确定性的 pathway 组合搜索器。
- prototype coverage、低方差基因和 per-gene R² 诊断。
- 可复现的软件环境与 artifact manifest。

---

## 9. 最终总结

这套代码的主线不是“先做一个 VQ-VAE，再附加一个 flow”这么简单。它真正试图建立的是三个层次之间的连接：

```text
细胞层：学习可复用、能区分真实扰动终态的 latent 状态
  ↓
群体层：描述一个样本由哪些终末状态和 cell types 组成
  ↓
前向层：学习 pathway activation 如何改变 terminal latent 与组成
  ↓
决策层：选择最大化指定目标 cell type 的 pathway 或组合
  ↓
验证层：用新扰动实验检验推荐并更新模型
```

因此，项目最终成功的标准，不只是“对已知扰动预测准确”，而是模型能否在准确恢复真实 terminal latent 的基础上，学到可迁移、可干预的 pathway→cell type 关系，并通过新的 perturbation 实验证明推荐的 pathway 组合确实提高了目标细胞类型占比。

简而言之：

> **最大化目标细胞类型的 pathway 选择是终极任务；terminal latent 的准确恢复是前提，离散表征是状态基础，drug perturbation 是观测入口，机制审计和新实验是关系可信度的验证。**
