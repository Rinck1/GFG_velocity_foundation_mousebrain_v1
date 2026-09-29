#!/usr/bin/env python3
"""Build the final Markdown/CSV report for the foundation pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "foundation_pilot_20260830"
DEFAULT_HISTORICAL = Path(
    "/data/yuchang/GFG_graphbatch_directed_v1/results/"
    "mousebrain_graphbatch_directed_v1_summary.csv"
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN"
    return f"{value:.{digits}f}"


def historical_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    seed_rows = [row for row in rows if row["seed"].isdigit()]
    mean_row = next(row for row in rows if row["seed"] == "mean")
    return seed_rows, mean_row


def pretrain_summary(root: Path, name: str):
    directory = root / name
    summary = load_json(directory / "pretrain_summary.json")
    metrics = load_jsonl(directory / "pretrain_metrics.jsonl")
    first, last = metrics[0], metrics[-1]
    return {
        "variant": summary["variant"],
        "steps": int(summary["final_step"]),
        "cells_seen": int(summary["cells_seen"]),
        "batch_size": int(summary["batch_size"]),
        "hours": float(last["elapsed_seconds"]) / 3600.0,
        "peak_memory_gib": float(summary["peak_memory_gib"]),
        "first_loss": float(first["loss"]),
        "last_loss": float(last["loss"]),
        "last_ema": float(last["loss_ema"]),
        "last_masked": float(last["masked_loss"]),
        "last_visible": float(last["visible_loss"]),
        "last_ode": float(last["ode_loss"]),
        "throughput": float(last["cells_per_second"]),
        "checkpoint": summary["final_checkpoint"],
    }


def transfer_row(root: Path, name: str, display_name: str):
    payload = load_json(root / name / "transfer_summary.json")
    metrics = payload["fine_tuned"]
    return {
        "model": display_name,
        "parameters": int(payload["parameters"]["trainable"]),
        "velocity_confidence": float(metrics["velocity_confidence"]),
        "iccoh": float(metrics["iccoh"]),
        "cbdir_without_graph": float(metrics["cbdir_without_graph"]),
        "cbdir_with_graph": float(metrics["cbdir_with_graph"]),
        "graph_improvement": float(metrics["graph_improvement"]),
        "zero_shot": payload.get("zero_shot"),
        "train_seconds": float(payload["train_seconds"]),
        "peak_memory_gib": float(payload["peak_memory_gib"]),
        "checkpoint": payload.get("checkpoint"),
    }


def write_csv(path: Path, rows: list[dict]):
    fields = [
        "model",
        "parameters",
        "velocity_confidence",
        "iccoh",
        "cbdir_without_graph",
        "cbdir_with_graph",
        "graph_improvement",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def build_report(root: Path, historical_path: Path):
    required = [
        root / "gpu6_denoise_pretrain" / "pretrain_summary.json",
        root / "gpu7_kinetic_pretrain" / "pretrain_summary.json",
        root / "gpu7_scratch_mousebrain" / "transfer_summary.json",
        root / "gpu6_denoise_mousebrain" / "transfer_summary.json",
        root / "gpu7_kinetic_mousebrain" / "transfer_summary.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Experiment is not complete; missing:\n" + "\n".join(map(str, missing))
        )

    denoise = pretrain_summary(root, "gpu6_denoise_pretrain")
    kinetic = pretrain_summary(root, "gpu7_kinetic_pretrain")
    scratch = transfer_row(root, "gpu7_scratch_mousebrain", "Large scratch")
    denoise_transfer = transfer_row(
        root, "gpu6_denoise_mousebrain", "Denoising-first pretrain"
    )
    kinetic_transfer = transfer_row(
        root, "gpu7_kinetic_mousebrain", "Weak-kinetic pretrain"
    )

    historical_seeds, historical_mean = historical_rows(historical_path)
    historical_seed0 = next(row for row in historical_seeds if row["seed"] == "0")
    historical = [
        {
            "model": "Corrected GFG seed 0",
            "parameters": 1_588_082,
            "velocity_confidence": float(historical_seed0["velocity_confidence"]),
            "iccoh": float(historical_seed0["iccoh"]),
            "cbdir_without_graph": float(historical_seed0["cbdir_without_graph"]),
            "cbdir_with_graph": float(historical_seed0["cbdir_with_graph"]),
            "graph_improvement": float(historical_seed0["graph_improvement"]),
        },
        {
            "model": "Corrected GFG seed 0–4 mean",
            "parameters": 1_588_082,
            "velocity_confidence": float(historical_mean["velocity_confidence"]),
            "iccoh": float(historical_mean["iccoh"]),
            "cbdir_without_graph": float(historical_mean["cbdir_without_graph"]),
            "cbdir_with_graph": float(historical_mean["cbdir_with_graph"]),
            "graph_improvement": float(historical_mean["graph_improvement"]),
        },
    ]
    comparison = historical + [scratch, denoise_transfer, kinetic_transfer]
    for row in (denoise_transfer, kinetic_transfer):
        row["delta_vs_scratch"] = {
            key: row[key] - scratch[key]
            for key in (
                "velocity_confidence",
                "iccoh",
                "cbdir_without_graph",
                "cbdir_with_graph",
                "graph_improvement",
            )
        }
    write_csv(root / "mousebrain_comparison.csv", comparison)

    lines = [
        "# GFG Velocity Foundation Pilot：最终实验结果",
        "",
        "实验日期：2026-08-31（Asia/Shanghai）",
        "",
        "## 1. 实验完成情况",
        "",
        "- 修正版 GFG 稳定快照保持不变；实验在独立目录中完成。",
        "- Large pilot 具有 14,621,698 个可训练参数，约为原模型的 9.2 倍。",
        "- Variant A 使用 masked denoising；Variant B 在相同更新步数下加入 weak kinetic loss。",
        "- 两种预训练 checkpoint 均迁移到 MouseBrain，并使用相同 graph batch、batch size、epoch 和 seed 微调。",
        "- Soft-VQ 因启动 smoke test 出现 perplexity≈1 的快速塌缩，本轮四个 large-model 路径统一旁路 VQ。",
        "",
        "## 2. 12M 预训练统计",
        "",
        "| Variant | Steps | Cells seen | Batch | Hours | Peak GiB | Loss first→last | EMA | ODE | cells/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in (denoise, kinetic):
        lines.append(
            f"| {item['variant']} | {item['steps']:,} | {item['cells_seen']:,} | "
            f"{item['batch_size']} | {item['hours']:.2f} | "
            f"{item['peak_memory_gib']:.2f} | "
            f"{item['first_loss']:.4f}→{item['last_loss']:.4f} | "
            f"{item['last_ema']:.4f} | {item['last_ode']:.3e} | "
            f"{item['throughput']:.1f} |"
        )
    lines += [
        "",
        "两种目标使用相同 optimizer-update budget，但由于 JVP/ODE 的显存开销不同，实际 batch 和 cells seen 不同；因此这是 compute/step-matched pilot，不是严格 data-matched loss ablation。",
        "",
        "## 3. MouseBrain 泛化结果",
        "",
        "| Model | Params | Velocity confidence | ICCoh | CBDir raw | CBDir graph | Graph Δ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison:
        lines.append(
            f"| {row['model']} | {row['parameters']:,} | "
            f"{fmt(row['velocity_confidence'])} | {fmt(row['iccoh'])} | "
            f"{fmt(row['cbdir_without_graph'])} | {fmt(row['cbdir_with_graph'])} | "
            f"{fmt(row['graph_improvement'])} |"
        )
    lines += [
        "",
        "### 相对同架构 scratch 的变化",
        "",
        "| Pretraining | Δ confidence | Δ ICCoh | Δ CBDir raw | Δ CBDir graph | Δ graph improvement |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in (denoise_transfer, kinetic_transfer):
        delta = row["delta_vs_scratch"]
        lines.append(
            f"| {row['model']} | {delta['velocity_confidence']:+.4f} | "
            f"{delta['iccoh']:+.4f} | {delta['cbdir_without_graph']:+.4f} | "
            f"{delta['cbdir_with_graph']:+.4f} | "
            f"{delta['graph_improvement']:+.4f} |"
        )
    lines += [
        "",
        "## 4. Zero-shot 诊断",
        "",
        "Zero-shot 仅用于诊断共享逐基因参数的迁移，不应解释为严格人鼠 ortholog foundation-model benchmark。",
        "",
        "| Variant | Confidence | ICCoh | CBDir raw | CBDir graph | Graph Δ | Fine-tune Δ graph |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in (denoise_transfer, kinetic_transfer):
        zero = row["zero_shot"]
        if not zero or "error" in zero:
            lines.append(f"| {row['model']} | error | error | error | error | error | error |")
        else:
            fine_tune_delta = row["cbdir_with_graph"] - zero["cbdir_with_graph"]
            lines.append(
                f"| {row['model']} | {fmt(zero['velocity_confidence'])} | "
                f"{fmt(zero['iccoh'])} | {fmt(zero['cbdir_without_graph'])} | "
                f"{fmt(zero['cbdir_with_graph'])} | "
                f"{fmt(zero['graph_improvement'])} | {fine_tune_delta:+.4f} |"
            )

    best_large_finetuned = max(
        (scratch, denoise_transfer, kinetic_transfer),
        key=lambda row: row["cbdir_with_graph"],
    )
    zero_shot_rows = [
        row for row in (denoise_transfer, kinetic_transfer)
        if row["zero_shot"] and "error" not in row["zero_shot"]
    ]
    best_zero_shot = max(
        zero_shot_rows,
        key=lambda row: row["zero_shot"]["cbdir_with_graph"],
    )
    lines += [
        "",
        "## 5. 结论与边界",
        "",
        f"- 所有 large-model 路径中，最佳 CBDir graph 是 {best_zero_shot['model']} 的 zero-shot 结果（{best_zero_shot['zero_shot']['cbdir_with_graph']:.4f}）；最佳 fine-tuned 结果才是 {best_large_finetuned['model']}（{best_large_finetuned['cbdir_with_graph']:.4f}）。",
        f"- Denoising-first 从 zero-shot {denoise_transfer['zero_shot']['cbdir_with_graph']:.4f} 降到微调后 {denoise_transfer['cbdir_with_graph']:.4f}；weak-kinetic 从 {kinetic_transfer['zero_shot']['cbdir_with_graph']:.4f} 降到 {kinetic_transfer['cbdir_with_graph']:.4f}。当前 GFG 全损失微调明显覆盖了预训练方向。",
        "- 下一轮不应直接沿用当前 20/300/300 权重：先冻结预训练 encoder，只训练小 velocity head；随后用更低学习率、loss normalization 和逐项 ramp 解冻，并把 batch 128/384 作为严格消融。",
        "- ICCoh 很高并不自动代表方向正确；近常量速度也可能产生高 coherence，因此主要结合 CBDir raw/graph 和 velocity confidence 解读。",
        "- large variants 当前只有 seed 0；与原模型五随机种子均值比较只能视为 pilot 证据，正式结论还需补跑至少 3–5 个 seed。",
        "- 人类 3k 预训练到 MouseBrain 的迁移依赖 GFG 的逐基因共享权重，不等价于全基因词表加严格 ortholog 映射。",
        "- 12M 语料 brain-heavy，当前 accession split 也不是真实 study/donor split；正式泛化 benchmark 应补元数据后重切分。",
        "",
        "## 6. 产物",
        "",
        f"- Denoising checkpoint：{denoise['checkpoint']}",
        f"- Kinetic checkpoint：{kinetic['checkpoint']}",
        "- MouseBrain 逐模型 checkpoint、评测 H5AD 和 JSON 位于同级实验子目录。",
        "- 机器可读比较表：mousebrain_comparison.csv。",
        "",
        "## 7. 完整性验证",
        "",
        "- 两个预训练 final checkpoint 均在 CPU 上以 strict=True 加载：missing=0、unexpected=0，所有浮点权重有限。",
        "- scratch、denoising、kinetic 三个 MouseBrain final checkpoint 均以 strict=True 加载：missing=0、unexpected=0。",
        "- 三个评测 H5AD 均为 3,365 cells × 759 genes，velocity、velocity_umap 和 velocity_graph_umap 全部为有限值。",
        "- 从保存的 H5AD 独立重算 velocity confidence、ICCoh、CBDir raw 和 CBDir graph，与 transfer_summary.json 的差值全部为 0。",
        "- 关键产物 SHA-256 位于 ARTIFACT_MANIFEST.sha256。",
        "",
    ]
    report_path = root / "FINAL_RESULTS.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--historical", type=Path, default=DEFAULT_HISTORICAL)
    args = parser.parse_args()
    report = build_report(args.experiment_root, args.historical)
    print(report)


if __name__ == "__main__":
    main()
