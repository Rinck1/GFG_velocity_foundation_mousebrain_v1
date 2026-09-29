# Hypoblast 文档、代码与数据状态总索引

更新日期：2026-09-05

## 1. 当前状态的直接结论

### 1.1 关于代码和数据梳理的文档在哪里

核心梳理文档都位于项目根目录：

| 文档 | 位置 | 内容 | 状态 |
|---|---|---|---|
| 代码逻辑与方法审查 | [HYPO_CODE_LOGIC_AND_METHOD_REVIEW.md](HYPO_CODE_LOGIC_AND_METHOD_REVIEW.md) | 端到端数据流、各模块输入输出、实验演进、方法合理性、风险和下一步 | 已完成，1030 行 |
| 项目目标理解 | [HYPO_PROJECT_PURPOSE.md](HYPO_PROJECT_PURPOSE.md) | pathway 到终末细胞状态和细胞类型组成的科学目标，以及反向 pathway 设计目标 | 已完成，454 行 |
| 实验方案改进建议 | [HYPO_EXPERIMENT_PLAN_IMPROVEMENTS.md](HYPO_EXPERIMENT_PLAN_IMPROVEMENTS.md) | 湿实验、数据划分、latent 评估、模型改进、组合搜索和 go/no-go 标准 | 已完成，396 行 |
| 本索引 | [HYPO_DOCUMENTATION_INDEX.md](HYPO_DOCUMENTATION_INDEX.md) | 汇总文档、代码、数据和当前实施状态 | 已完成 |

这四份文档的绝对目录都是：

    /data/yuchang/hypo/

### 1.2 改进的代码在哪里

目前没有新增或修改任何模型代码。

本轮完成的是：

- 解压和整理原始项目；
- 静态阅读代码与实验记录；
- 梳理项目目的；
- 评估现有方法；
- 提出实验和代码改进方案；
- 新增 Markdown 文档。

本轮没有修改：

- Python 源代码；
- Stage 1 或 Stage 2 配置；
- shell 运行脚本；
- 测试代码；
- pathway CSV 或 YAML；
- benchmark 结果。

因此，HYPO_EXPERIMENT_PLAN_IMPROVEMENTS.md 中的内容是待实施方案，不是已经落地的代码变更。目录中已有的 v7、v8、pathway-conditioned 和 AdaLN 等实现来自原压缩包，也不是本轮新增。

---

## 2. 推荐阅读顺序

### 第一步：先明确最终目标

阅读 [HYPO_PROJECT_PURPOSE.md](HYPO_PROJECT_PURPOSE.md)。

重点回答：

- 项目最终要学习什么；
- 为什么预测扰动后的 latent 终态只是前提；
- pathway、终末细胞状态和细胞类型比例之间是什么关系；
- 如何从前向预测走向 pathway 单独或组合干预的逆向设计。

### 第二步：理解目前代码实际做了什么

阅读 [HYPO_CODE_LOGIC_AND_METHOD_REVIEW.md](HYPO_CODE_LOGIC_AND_METHOD_REVIEW.md)。

重点包括：

- Stage 1 的 scFoundation、MLP、SoftVQ 和表达重建；
- product-tuple prototype 的构造；
- 样本群体的 prototype proportion、presence 和 CLR mass；
- drug 到 pathway 条件；
- Stage 2 conditional prototype mass flow；
- prototype 到 cell type 的映射；
- 每个模块的输入和输出；
- v7、v8 实验演进和当前证据；
- 数据泄漏、验证随机性、模型语义和复现风险。

### 第三步：确定下一轮实验与开发任务

阅读 [HYPO_EXPERIMENT_PLAN_IMPROVEMENTS.md](HYPO_EXPERIMENT_PLAN_IMPROVEMENTS.md)。

重点包括：

- pathway 效应的可辨识性；
- 同一 pathway 的多种独立干预、激活、抑制和 rescue；
- 早期 pathway 活性读出；
- terminal latent 恢复指标；
- leave-one-drug-out、leave-one-donor-out 和 pathway transfer；
- 从单 pathway 到组合干预的分阶段搜索；
- 前瞻性盲法验证；
- 各阶段 go/no-go 标准。

---

## 3. 原始代码在哪里

代码根目录：

    /data/yuchang/hypo

### 3.1 Stage 1：细胞表示和离散状态

| 模块 | 位置 | 作用 |
|---|---|---|
| 总模型 | [model/model.py](model/model.py) | 组装 encoder、quantizer、decoder |
| 通用训练 | [model/train.py](model/train.py) | Stage 1 训练和验证 |
| scFoundation encoder | [model/components/encoders/scfoundation_encoder.py](model/components/encoders/scfoundation_encoder.py) | 将 scFoundation 表征映射到可量化 latent |
| SoftVQ | [model/components/quantizers/soft_vq_quantizer.py](model/components/quantizers/soft_vq_quantizer.py) | 软向量量化 |
| 多 codebook | [model/components/quantizers/multi_codebook_vq_quantizer.py](model/components/quantizers/multi_codebook_vq_quantizer.py) | 组合多个 codebook |
| MLP decoder | [model/components/decoders/mlp_decoder.py](model/components/decoders/mlp_decoder.py) | 从 latent 重建 HVG 表达 |
| Stage 1 主配置 | [configs/stage1](configs/stage1) | 不同 HVG、cell-type 和初始化配置 |
| Stage 1 运行入口 | [experiments/scfoundation_vqvae/cell_embedding/run.py](experiments/scfoundation_vqvae/cell_embedding/run.py) | 运行 Stage 1 |

### 3.2 Stage 1 结果转成群体 prototype

| 模块 | 位置 | 作用 |
|---|---|---|
| prototype 导出 | [experiments/perturbation/export_prototype_assignments.py](experiments/perturbation/export_prototype_assignments.py) | 导出细胞到 prototype 的分配 |
| population cache | [experiments/perturbation/prepare_prototype_population_cache.py](experiments/perturbation/prepare_prototype_population_cache.py) | 聚合样本级 prototype mass、presence 和 cell-type 映射 |
| pair records | [experiments/perturbation/prepare_prototype_pair_records.py](experiments/perturbation/prepare_prototype_pair_records.py) | 构造扰动前后样本对 |
| 数据边界 | [experiments/perturbation/stage1_boundary.py](experiments/perturbation/stage1_boundary.py) | 处理 Stage 1/Stage 2 边界与划分 |

### 3.3 Stage 2：扰动条件和群体动力学

| 模块 | 位置 | 作用 |
|---|---|---|
| prototype mass flow | [model/perturbation/prototype_mass_flow.py](model/perturbation/prototype_mass_flow.py) | 预测 prototype 状态与质量的扰动后变化 |
| 基础 pathway conditioner | [model/perturbation/pathway_condition.py](model/perturbation/pathway_condition.py) | 将药物和 pathway 先验编码为条件 |
| Pathway→Prototype | [model/perturbation/pathway_prototype_condition.py](model/perturbation/pathway_prototype_condition.py) | 显式建模 pathway 对 prototype 的作用 |
| Pathway→Cell type | [model/perturbation/pathway_celltype_condition.py](model/perturbation/pathway_celltype_condition.py) | 显式建模 pathway 对 cell type 的作用 |
| 条件结构与 I/O | [model/perturbation/condition_types.py](model/perturbation/condition_types.py)、[model/perturbation/condition_io.py](model/perturbation/condition_io.py) | 条件数据结构与读写 |
| Stage 2 训练 | [experiments/perturbation/train_prototype_mass_flow.py](experiments/perturbation/train_prototype_mass_flow.py) | 训练群体 flow |
| Stage 2 评估 | [experiments/perturbation/evaluate_prototype_mass_flow.py](experiments/perturbation/evaluate_prototype_mass_flow.py) | 评估扰动终点预测 |
| Stage 2 配置 | [configs/stage2](configs/stage2) | baseline、19-path、cell-type、AdaLN 等配置 |

### 3.4 评估、可视化和测试

| 模块 | 位置 | 作用 |
|---|---|---|
| 指标 | [evaluation/metrics.py](evaluation/metrics.py) | 通用评估指标 |
| 验证 | [evaluation/validation.py](evaluation/validation.py) | 验证流程 |
| 损失 | [evaluation/losses.py](evaluation/losses.py) | 训练和评估损失 |
| 可视化 | [evaluation/visualization.py](evaluation/visualization.py) | 结果可视化 |
| pathway route 审计 | [experiments/perturbation/audit_pathway_celltype_route.py](experiments/perturbation/audit_pathway_celltype_route.py) | 检查显式 pathway 路径是否实际被使用 |
| pathway 效应图 | [experiments/perturbation/visualize_pathway_celltype_effects.py](experiments/perturbation/visualize_pathway_celltype_effects.py) | 绘制 pathway–cell-type 效应 |
| 测试 | [tests/perturbation](tests/perturbation) | pathway conditioner 和 benchmark 相关测试 |

---

## 4. 数据在哪里，以及当前具有什么

数据定义目录：

    /data/yuchang/hypo/data

当前压缩包包含：

| 文件 | 用途 |
|---|---|
| [data/scrna_dataset.py](data/scrna_dataset.py) | 单细胞数据读取、预处理和样本构造代码 |
| [data/perturbation_schema.py](data/perturbation_schema.py) | 扰动数据字段和 schema |
| [data/drug_pathway_sign.csv](data/drug_pathway_sign.csv) | 药物到 pathway 的已知方向先验 |
| [data/pathways.yaml](data/pathways.yaml) | pathway 配置 |
| [data/pathways_v73_18path.yaml](data/pathways_v73_18path.yaml) | v7.3 的 18-path 配置 |
| [data/pathways_v73_19path.yaml](data/pathways_v73_19path.yaml) | v7.3 的 19-path 配置 |

当前目录中没有发现 h5ad 数据文件。压缩包也没有包含大型表达矩阵、完整预训练权重或训练 checkpoint。因此：

- 现有文档能够梳理数据接口、字段、配置和代码预期；
- 不能仅凭当前目录核对真实细胞数、各条件样本数和原始表达矩阵；
- 不能在缺少外部数据与模型 artifact 的情况下完整复现实验；
- 若要做进一步的数据质量审查，需要补充实际 h5ad、样本 metadata、split manifest 和必要 checkpoint。

---

## 5. 项目自带的 Markdown 文档

以下文档来自原压缩包，不是本轮新写。

### 5.1 项目说明

- [README.md](README.md)：项目总体说明。
- [CLAUDE.md](CLAUDE.md)：原项目开发与协作约定。
- [configs/README.md](configs/README.md)：配置说明。
- [docs/stage1_experiment_registry.md](docs/stage1_experiment_registry.md)：Stage 1 实验登记。
- [experiments/scfoundation_vqvae/cell_embedding/README_hypoblast_stage1_code_analysis.md](experiments/scfoundation_vqvae/cell_embedding/README_hypoblast_stage1_code_analysis.md)：Stage 1 code 分析说明。

### 5.2 原项目设计文档

目录：

    /data/yuchang/hypo/docs/superpowers/specs

主要覆盖：

- Stage 1 iPStem restoration；
- Stage 1 HVG2000 baseline 和 cell-type；
- Stage 1 experiment registry；
- Stage 3 population output supervision；
- hard-sign、low-rank 和 unknown pathway；
- drug–pathway prior；
- projected-prior initialization；
- mid-flow edge bridge；
- Pathway→Prototype；
- Pathway→Cell type AdaLN；
- 19-path baseline comparison。

### 5.3 原项目实验计划

目录：

    /data/yuchang/hypo/docs/superpowers/plans

这些文件记录了对应设计的实施计划和时间顺序。代码演进的中文时间线已经汇总到 HYPO_CODE_LOGIC_AND_METHOD_REVIEW.md 的第 7 节。

### 5.4 原项目 benchmark 报告

关键报告：

- [benchmarks/v7-hvg2000/report.md](benchmarks/v7-hvg2000/report.md)
- [benchmarks/v7-hvg2000/final-12-models/report.md](benchmarks/v7-hvg2000/final-12-models/report.md)
- [benchmarks/v7-hvg2000/adaln-pathway-prototype/report.md](benchmarks/v7-hvg2000/adaln-pathway-prototype/report.md)
- [benchmarks/v7-hvg2000/19path-baseline-comparison/comparison.md](benchmarks/v7-hvg2000/19path-baseline-comparison/comparison.md)
- [benchmarks/v8-pathway-celltype-adaln/report.md](benchmarks/v8-pathway-celltype-adaln/report.md)

其中 report_prompt.md 是报告生成提示或模板，不应当作最终实验结论；阅读结果时应优先看 report.md 和 comparison.md。

### 5.5 工具构建文档

.skill-build/factorial-model-benchmark 下的 Markdown 属于 benchmark 工具说明和 schema，不是 Hypoblast 的核心生物学结论。

---

## 6. 已识别但尚未实施的代码改进

下表把建议映射到可能需要修改的代码位置。所有项目均为“待实施”。

| 优先级 | 待实施改进 | 主要代码位置 | 当前状态 |
|---|---|---|---|
| P0 | 建立贯穿 Stage 1、prototype、R 映射和 Stage 2 的不可变 split manifest | experiments/perturbation/stage1_boundary.py、prepare_prototype_population_cache.py、data/scrna_dataset.py | 未实施 |
| P0 | 补全 terminal latent 分布、delta direction、retrieval 和目标细胞特异指标 | evaluation/metrics.py、evaluation/validation.py、evaluate_prototype_mass_flow.py | 未实施 |
| P0 | 增加 control persistence、linear/additive、drug-ID-only 等基线 | configs/stage2、experiments/perturbation、common/evaluate_all.py | 未实施 |
| P1 | 消除 prototype_to_celltype 的标签泄漏 fallback | prepare_prototype_population_cache.py、train_prototype_mass_flow.py | 未实施 |
| P1 | 固定 validation flow time 和随机过程 | train_prototype_mass_flow.py、prototype_mass_flow.py | 未实施 |
| P1 | 报告 prototype coverage，并修正 SoftVQ 使用率诊断 | soft_vq_quantizer.py、Stage 1 评估脚本 | 未实施 |
| P1 | 将 CLR velocity 投影到零和子空间 | prototype_mass_flow.py | 未实施 |
| P1 | 明确 dynamic presence 和 state_delta 是否进入 rollout | prototype_mass_flow.py | 未实施 |
| P1 | 建立“全局 pathway 效应 + context residual” | pathway_celltype_condition.py、pathway_prototype_condition.py | 未实施 |
| P2 | 加入 pathway-transfer、leave-one-drug-out 和组合外推评估 | data schema、split 工具、训练与评估脚本 | 未实施 |
| P2 | 建立 pathway 组合反向搜索和不确定性约束 | 需要新增 optimizer/search 模块 | 未实施 |
| P2 | 补充环境锁定、artifact manifest 和可复现入口 | 项目根目录、配置和脚本 | 未实施 |

其中优先顺序应是：

1. 先清理数据划分和验证边界；
2. 再补全 terminal latent 与目标细胞特异评估；
3. 确认模型稳定优于简单基线；
4. 再改 pathway 解释层和动力学细节；
5. 最后实现组合干预的反向搜索。

不建议在前三步尚未完成时，直接加入更复杂的 pathway 组合优化器。

---

## 7. 当前产物清单

### 原始产物

- 压缩包：/data/yuchang/hypoblast-main.zip
- 解压项目：/data/yuchang/hypo

### 本轮新增产物

- /data/yuchang/hypo/HYPO_CODE_LOGIC_AND_METHOD_REVIEW.md
- /data/yuchang/hypo/HYPO_PROJECT_PURPOSE.md
- /data/yuchang/hypo/HYPO_EXPERIMENT_PLAN_IMPROVEMENTS.md
- /data/yuchang/hypo/HYPO_DOCUMENTATION_INDEX.md

### 本轮代码变更

    无

---

## 8. 一句话交接

目前已经完成“理解项目、梳理代码、澄清科学目标、识别风险和设计改进路线”，但尚未进入“修改代码、补测试、运行实验和验证改进”的实施阶段。

