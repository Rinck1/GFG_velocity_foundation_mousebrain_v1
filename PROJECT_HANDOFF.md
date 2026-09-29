# PROJECT HANDOFF — GFG 单细胞 RNA Velocity 项目

> 本文件供任意 AI agent（Codex / Claude / opencode 等）接管项目用。纯 Markdown，可复制到任何聊天窗口。
> 最后更新：2026-09-09

## 0. 一句话背景
GFG（Geometric Flow Grounding，ICML 2026）原论文三 case：稀疏 ODE 发现 / scRNA velocity / AI 视频鉴伪。
本项目的任务：把 GFG 从原版小模型（~2M, 基因级 u/s → 双编码器/VQ → 共享 decoder/JVP → kinetic+graph loss）**扩容到脑规模单细胞数据（~100M+ 参数）**，同时**保留原版任务定义**，不得改成 PCA 空间的 scVelo 速度蒸馏。

## 1. 关键路径

| 用途 | 路径 |
|---|---|
| **主工作目录（GFG-v3）** | `/home/yuchang/GFG_velocity_foundation_mousebrain_v1/` |
| 数据处理与评测脚本 | `/home/yuchang/data_download/` |
| 原版 GFG 代码（圆环 ODE + 视频） | `/home/yuchang/GFG-public/` |
| 原始 Velocyto 数据（31,952 h5ad, ~3.3TB） | `/data1/yuchang/Velocyto/`（`/data/dataset/Velocyto` 是符号链接） |
| 脑 shards（12.7M 细胞） | `/data/dataset/processed_full/` |
| pairs（全局 PCA 坐标） | `/data/dataset/pairs_full/`（450万）、`pairs_full12/`（1200万） |
| GFG-v3 训练输出 | `/data/dataset/gfg3_full/`（当前 run）、`gfg3_pilot/`（已作废） |
| Zero-shot 四数据集 | `/data/yuchang/dataset/{MouseBrain,DentateGyrus,erythroid_lineage,retina}.h5ad` |
| 历史 checkpoints（PCA 路线） | `/home/yuchang/GFG-public/GFG-EquationDiscovery/checkpoints/` |
| 交接/评测文档 | `data_download/REPORT_positive_results.md`、`GFG-public/GFG-EquationDiscovery/EVALUATION_NOTES.md`、本目录 `GFG_CODE_LOGIC.md` |

## 2. 硬件/环境
- 8× NVIDIA H20（96GB/卡）；Python venv：`/home/yuchang/venv-gfg/bin/python`（3.10, torch 2.8.0+cu126, scvelo 0.3.4, scanpy, anndata, h5py）
- `/data` 3.5T（NVMe, 剩 ~2.4T）、`/data1` 11T RAID（6.3GB/s, 剩 ~7.1T）
- 注意：系统 `python3` 是老版本（3.6），**必须用 venv 的 python**；`rsync` 未安装

## 3. 数据现状（已完成）
- 人类脑样本：2,173 清单 → 2,103 下载 → 9 shards = **12,736,995 细胞 × 36,601 基因**
- 全人类：**31,952 / 35,182**（~3.3TB），含血液 9,886、六器官语料 15,748 donor
  - 六器官：脑 1,881 / 血 9,888 / 肺 1,940 / 肝 1,145 / 肠 536 / 心 358（`data_download/organ_corpus_6.json`）
  - held-out 器官：肾 351 / 皮肤 818 / 乳腺 844（不参与训练）
- **基因对齐审计**：所有人类文件共享同一 36,601 基因词表，`process_data.py` 的 `take_cols` 列错位 bug 未触发（潜伏，已在新管线用 union 词表 + mask 规避）
- 已知坏文件：少数截断 h5ad（如 `ERX10019092` 附近的文件），新管线用 `filter_readable()` 剔除

## 4. GFG-v3 代码结构（当前主线）

| 文件 | 作用 |
|---|---|
| `gfg3_model.py` | 模型：基因 token(u,s,u−s,mask) → StateTower/VelocityTower（共享 per-gene MLP + ISAB 置换等变）→ 双 SoftVQ（余弦归一化 + 可学习 log_tau + **硬最近码字 STE**）→ 共享逐基因 decoder→(û,ŝ)；速度 `v_x = J_decoder(z_s)·z_v`（训练=推理同一算子）。三档配置：**original 1.53M / medium 18.88M / xlarge 117.90M** |
| `gfg3_data.py` | donor(文件级)划分、union 词表、`clip(±10)` z-score、缺失哨兵 **-100**、`BlockStream` 块级流式（h5py 行块）、`filter_readable` |
| `gfg3_train.py` | Stage A(掩码重构)/B(velocity: kinetic+graph smooth/align, 状态流冻结)/C(渐进解冻)；`GradNormBalancer` 梯度尺度自动归一化(DDP 同步)；`GraphCache` 真 kNN 图（按需读 h5，邻接常驻）；torchrun DDP |
| `gfg3_eval.py` | held-out donor 评测：重构 MSE、kinetic、幅值统计、VeloCoh、kNN 共识、teacher agreement（scVelo 仅参考）、rollout 漂移 |
| `model/Decoder.py` | 复用原版 `ode_min_residual_loss`（逐基因 ridge α/β/γ） |
| `gfg3_queue.sh` / `data_download/gfg3_full_launch.sh` | 排队与全量启动 |

已验证不变量：置换等变 6e-08、JVP=有限差分 2.8e-05、训练/推理同算子、DDP EMA allreduce、
`--no-vq` 消融、监控（H(C)/E[H(C|x)]/MI/hard usage/embed var/effective rank）。

## 5. 训练过的模型与结果（已完成）

| 模型 | 数据 | 结果 | 判定 |
|---|---|---|---|
| m12m 7.7M MLP | 140k 对(PCA路线) | loss_re 0.057, cos 0.833 | 基线 |
| m12m Transformer | 140k | loss_re 2.30 | ❌ 弃用 |
| xlarge 126M | 140k | loss_re 0.033, **cos 0.895 双杀 kNN 共识(0.876)** | ⭐ 同分布最强 |
| xlarge_full 945k | 混合坐标系 | cos 0.31 | ❌ 暴露 per-shard PCA 缺陷 |
| xlarge_global v0/v1 | 450万对全局坐标 | cos 0.694/0.684；**zero-shot B1 0.18/0.28/0.32** | ⭐ 迁移最强 |
| m200m 197M | 1200万对 | cos 0.737(+7%) 但 B1 全降 | ❌ 容量≠迁移 |
| ~~GFG-v3 pilot 1.53M~~ | 24 文件 | 常数场塌缩 | ❌ 作废（根因已修） |

Zero-shot B1（谱系内成熟方向，置换对照≈0；raw scVelo 基线 0.062/0.004/0.025）：
DentateGyrus 0.18 / erythroid 0.28 / retina 0.32（126M 全局版）。

## 6. 关键教训（不要重蹈）
1. **PCA 路线偏离论文定义**：用 scVelo PCA 速度当监督 = 任务改写，且 per-shard PCA 坐标系互不兼容（同分布 cos 0.31 vs 全局基 0.69）。→ 必须基因空间 + kinetic 残差。
2. **JVP 不能当速度读出**（潜步长 O(1) 时线性化失真）→ 评测要用与训练一致的离散算子。
3. **EMA 码本 DDP 必须 allreduce 充分统计量**，且每 step 每码本只更新一次（邻居前向要 `eval()`）。
4. **SoftVQ 均匀后验塌缩**：LayerNorm 等半径球 + 高维欧氏距离无区分度 → z_q 常数 → 常数速度场。修复=余弦度量+可学习 τ+**硬最近码字 STE**。监控必须看 MI/hard usage，不能只看 ppl。
5. **u/s 语义与哨兵一致性**：缺失哨兵统一 -100；模型 mask 用 `<-50`；z-score 后 clip ±10（unspliced 稀疏层否则 MSE 爆炸到千万级）。
6. **评测诚实性**：训练集上 model-vs-kNN 不能称“泛化”；主结果必须来自 held-out donor（`splits_manifest` 的 val/test）。
7. 幅值系统性失调 ~1.5×（基因空间 kinetic 监督后才可能根治）。

## 7. 当前状态（2026-09-09）
- **正在运行**：GFG-v3 **xlarge 117.9M × 六器官全量**，6 卡 torchrun
  - 命令：`bash /home/yuchang/data_download/gfg3_full_launch.sh`
  - 日志：`/home/yuchang/data_download/gfg3_full.log`；输出 `/data/dataset/gfg3_full/`
  - 配置：donor 划定 14172/787/788；3000 基因；Stage A6+B5+C1；batch 32/rank；2000 步/epoch；
    统计拟合 500 文件、图缓存 300 文件
  - 阶段：CPU 统计预处理（~30-60min）→ 自动上 GPU
- 6 器官语料、held-out 3 器官、血液已全部下载在盘

## 8. 接手建议（下一步）
1. 监控 `gfg3_full.log`：Stage A 的 recon_s/u 应稳定（不爆量级）；Stage B 的 kinetic/smooth/align 与 GradNorm 权重；VQ 监控（H(C)/MI/hard_usage）不应退化为均匀
2. 训完跑 `gfg3_eval.py --ckpt /data/dataset/gfg3_full/final.pt`（held-out donor）
3. 需要 medium/xlarge 对照、`--no-vq` 消融、多 seed（规格书要求 ≥3 seed mean/std）
4. 跨物种评测需**人鼠 ortholog 映射**（不能各自拟合 PCA 只做缩放）；现有 4 个 mouse 数据集在 `/data/yuchang/dataset/`
5. 长训练启动前建议再过一遍独立审计（本项目曾一次审计找出 8 个 blocking bug）

## 9. 传给其他 agent 的方式
- 本文件即交接文档；发给 Codex/Claude 时说：“读 `/home/yuchang/GFG_velocity_foundation_mousebrain_v1/PROJECT_HANDOFF.md`，然后继续”
- Codex CLI 会自动读项目根目录的 `AGENTS.md`：可把本文件复制/软链为 `AGENTS.md`
- 纯文本可粘贴进任意聊天窗口；也可 `git commit` 进仓库随代码走
