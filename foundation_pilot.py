#!/usr/bin/env python3
"""12M denoising pretraining and MouseBrain transfer pilot for corrected GFG."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

os.environ.setdefault("MPLBACKEND", "Agg")

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from const.cluster_edges import all_edges
from model.Config import Config
from model.Decoder import split_us
from model.graph import GraphBatchSampler, build_root_directed_adjacency
from model.model import VeloModel
from preprocessing import load_data, preprocess_data_scv
from tool.evaluate import (
    compute_velocity_from_graph,
    evaluate_CBDir2,
    evaluate_ICCoh,
)
from tool.utils import compute_pred, find_cluster_key


DEFAULT_PRETRAIN_ROOT = Path("/data/dataset/Velocyto_pretrain_v1")
DEFAULT_MOUSEBRAIN = Path("/data/yuchang/dataset/MouseBrain.h5ad")


@dataclass(frozen=True)
class PilotModelSpec:
    gene_dim: int = 64
    codebook_size: int = 128
    use_vq: bool = False
    hidden_state: tuple[int, ...] = (768, 1536, 1536, 768)
    attention_heads: int = 8
    attention_layers: int = 2
    attention_ff_mult: int = 4
    mlp_dropout: float = 0.1
    attention_dropout: float = 0.1


def json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_cpm(
    matrix: np.ndarray,
    scale: float = 1.0e4,
    library_size: np.ndarray | None = None,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    library = (
        matrix.sum(axis=1, keepdims=True)
        if library_size is None
        else np.asarray(library_size, dtype=np.float32).reshape(-1, 1)
    )
    if library.shape[0] != matrix.shape[0]:
        raise ValueError(
            f"library_size rows={library.shape[0]} do not match matrix rows={matrix.shape[0]}"
        )
    normalized = matrix * (scale / np.maximum(library, 1.0))
    np.log1p(normalized, out=normalized)
    return normalized


class H5ADBatchStream(IterableDataset):
    """Stream pre-batched dense log-CPM tensors from small sparse H5AD files."""

    def __init__(
        self,
        paths: list[Path],
        batch_size: int,
        seed: int,
        cells_per_dataset_cap: int = 20_000,
        genes_per_batch: int | None = None,
    ):
        super().__init__()
        if not paths:
            raise ValueError("At least one H5AD path is required")
        self.paths = tuple(paths)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.cells_per_dataset_cap = int(cells_per_dataset_cap)
        self.genes_per_batch = (
            None if genes_per_batch is None else int(genes_per_batch)
        )
        if self.genes_per_batch is not None and self.genes_per_batch <= 0:
            raise ValueError("genes_per_batch must be positive")

    def __iter__(self) -> Iterable[torch.Tensor]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        worker_paths = list(self.paths[worker_id::num_workers])
        rng = np.random.default_rng(self.seed + 10_007 * worker_id)

        while True:
            rng.shuffle(worker_paths)
            for path in worker_paths:
                try:
                    adata = ad.read_h5ad(path)
                    u_layer = adata.layers["unspliced"]
                    s_layer = adata.layers["spliced"]
                    n_cells = int(adata.n_obs)
                    indices = np.arange(n_cells, dtype=np.int64)
                    if n_cells > self.cells_per_dataset_cap:
                        indices = rng.choice(
                            n_cells,
                            size=self.cells_per_dataset_cap,
                            replace=False,
                        )
                    rng.shuffle(indices)

                    for start in range(0, len(indices) - self.batch_size + 1, self.batch_size):
                        selected = indices[start : start + self.batch_size]
                        u = u_layer[selected]
                        s = s_layer[selected]
                        u_library = None
                        s_library = None
                        if (
                            self.genes_per_batch is not None
                            and self.genes_per_batch < adata.n_vars
                        ):
                            # Normalize against the complete transcriptome library,
                            # then expose a new uniform, non-HVG gene subset each step.
                            u_library = np.asarray(u.sum(axis=1)).reshape(-1, 1)
                            s_library = np.asarray(s.sum(axis=1)).reshape(-1, 1)
                            gene_indices = np.sort(
                                rng.choice(
                                    adata.n_vars,
                                    size=self.genes_per_batch,
                                    replace=False,
                                )
                            )
                            u = u[:, gene_indices]
                            s = s[:, gene_indices]
                        if sp.issparse(u):
                            u = u.toarray()
                        if sp.issparse(s):
                            s = s.toarray()
                        u = log_cpm(u, library_size=u_library)
                        s = log_cpm(s, library_size=s_library)
                        x = np.concatenate((u, s), axis=1)
                        yield torch.from_numpy(np.ascontiguousarray(x))
                except Exception as exc:
                    print(f"STREAM_WARNING path={path} error={type(exc).__name__}: {exc}", flush=True)
                finally:
                    if "adata" in locals():
                        del adata


class NormalizedMouseBrainData(Dataset):
    """MouseBrain data using the same per-cell log-CPM transform as pretraining."""

    def __init__(self, adata):
        self.adata = adata
        u = adata.layers["unspliced"]
        s = adata.layers["spliced"]
        if sp.issparse(u):
            u = u.toarray()
        if sp.issparse(s):
            s = s.toarray()
        self.unspliced = log_cpm(u)
        self.spliced = log_cpm(s)
        self.data = np.concatenate((self.unspliced, self.spliced), axis=1)
        self.layer_states = None

        if "is_root" in adata.obs:
            self.root_mask = adata.obs["is_root"].to_numpy(dtype=bool)
        else:
            u_total = np.asarray(u).sum(axis=1)
            s_total = np.asarray(s).sum(axis=1)
            ratio = u_total / np.maximum(u_total + s_total, 1.0e-8)
            root_count = min(50, len(ratio))
            root_indices = np.argpartition(ratio, -root_count)[-root_count:]
            self.root_mask = np.zeros(len(ratio), dtype=bool)
            self.root_mask[root_indices] = True

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int):
        return torch.from_numpy(self.data[index]), self.root_mask[index], index


class DummyLoader:
    def __init__(self):
        self.dataset = SimpleNamespace(layer_states=None)


def configure_model(config: Config, spec: PilotModelSpec, *, epochs: int, lr: float) -> None:
    config.config["model"].update(asdict(spec))
    config.config["train"]["num_epochs"] = int(epochs)
    config.config["train"]["lr"] = float(lr)


def build_model(num_genes: int, spec: PilotModelSpec, seed: int, *, epochs: int = 10, lr: float = 3e-4):
    config = Config(seed=seed)
    configure_model(config, spec, epochs=epochs, lr=lr)
    config.config["data"]["num_gene"] = int(num_genes)
    config.config["data"]["num_cell"] = None
    model = VeloModel(DummyLoader(), config=config, device="cuda")
    return model, config


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def make_mask(target: torch.Tensor, gene_mask_rate: float, layer_mask_rate: float):
    batch, two_genes = target.shape
    genes = two_genes // 2
    gene_mask = torch.rand((batch, genes), device=target.device) < gene_mask_rate
    u_mask = gene_mask | (torch.rand((batch, genes), device=target.device) < layer_mask_rate)
    s_mask = gene_mask | (torch.rand((batch, genes), device=target.device) < layer_mask_rate)
    mask = torch.cat((u_mask, s_mask), dim=1)
    masked = target.clone()
    masked[mask] = -1.0
    return masked, mask


def balanced_masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    squared_error = (pred.float() - target.float()).square()
    nonzero = mask & (target > 0)
    zero = mask & ~nonzero
    pieces = []
    if nonzero.any():
        pieces.append(squared_error[nonzero].mean())
    if zero.any():
        pieces.append(squared_error[zero].mean())
    masked_loss = torch.stack(pieces).mean()
    visible_loss = squared_error[~mask].mean()
    return masked_loss, visible_loss


def state_only_forward(model: VeloModel, inputs: torch.Tensor):
    z = model.manifold_encoder(inputs, layer_states=None)
    if model.use_vq:
        z_q, vq_loss, codes = model.manifold_codebook(z)
    else:
        zero = z.new_zeros(())
        z_q = z
        vq_loss = {"vq": zero, "commit": zero, "perplexity": zero}
        codes = torch.zeros(z.shape[:2], dtype=torch.long, device=z.device)
    token_output = model.decoder.statedecoder(z_q.reshape(-1, z_q.shape[-1]))
    u, s = split_us(token_output, model.G)
    return torch.cat((u, s), dim=1), vq_loss, codes


def checkpoint_state(model: VeloModel, variant: str) -> dict[str, torch.Tensor]:
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if variant == "denoise":
        for key in list(state):
            if key.startswith("velocity_encoder."):
                source = "manifold_encoder." + key[len("velocity_encoder.") :]
                state[key] = state[source].clone()
            elif key.startswith("velocity_codebook."):
                source = "manifold_codebook." + key[len("velocity_codebook.") :]
                if source in state and state[source].shape == state[key].shape:
                    state[key] = state[source].clone()
    return state


def load_paths(root: Path, split: str) -> list[Path]:
    path_file = root / "manifests" / f"{split}_paths.txt"
    paths = [Path(line.strip()) for line in path_file.read_text().splitlines() if line.strip()]
    return paths


def save_pretrain_checkpoint(
    path: Path,
    model: VeloModel,
    optimizer: torch.optim.Optimizer,
    *,
    variant: str,
    step: int,
    cells_seen: int,
    batch_size: int,
    spec: PilotModelSpec,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": checkpoint_state(model, variant),
        "optimizer_state": optimizer.state_dict(),
        "variant": variant,
        "step": step,
        "cells_seen": cells_seen,
        "batch_size": batch_size,
        "model_spec": asdict(spec),
        "elapsed_seconds": elapsed_seconds,
    }
    torch.save(payload, path)


def run_pretrain(args) -> Path:
    set_all_seeds(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")
    spec = PilotModelSpec()
    train_paths = load_paths(args.pretrain_root, "train")
    stream = H5ADBatchStream(
        train_paths,
        batch_size=args.batch_size,
        seed=args.seed,
        cells_per_dataset_cap=args.cells_per_dataset_cap,
    )
    loader = DataLoader(
        stream,
        batch_size=None,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )

    model, _ = build_model(3000, spec, args.seed, lr=args.pretrain_lr)
    model.to(device)
    model.train()
    parameters = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.pretrain_lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "pretrain_metrics.jsonl"
    metadata = {
        "variant": args.variant,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "max_hours": args.max_hours,
        "workers": args.workers,
        "parameters": parameters,
        "model_spec": asdict(spec),
        "physical_gpu": args.physical_gpu,
        "torch_version": torch.__version__,
        "started_at_unix": time.time(),
    }
    json_dump(output_dir / "pretrain_config.json", metadata)
    print("PRETRAIN_CONFIG", json.dumps(metadata, ensure_ascii=False), flush=True)

    start_time = time.monotonic()
    deadline = start_time + args.max_hours * 3600.0
    cells_seen = 0
    ema_loss = None
    iterator = iter(loader)
    torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, args.max_steps + 1):
        if time.monotonic() >= deadline:
            print(f"TIME_BUDGET_REACHED step={step - 1}", flush=True)
            break
        target = next(iterator).to(device=device, non_blocking=True)
        masked, mask = make_mask(target, args.gene_mask_rate, args.layer_mask_rate)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if args.variant == "denoise":
                prediction, vq_info, codes = state_only_forward(model, masked)
                ode_loss = prediction.new_zeros(())
                vq_loss = vq_info["vq"]
                manifold_perplexity = vq_info["perplexity"]
                velocity_perplexity = prediction.new_zeros(())
            else:
                prediction, _, losses, _, _ = model(masked, x_obs=target)
                ode_loss = losses[0]["L_ode"]
                vq_loss = losses[1]["vq"] + losses[2]["vq"]
                manifold_perplexity = losses[1]["perplexity"]
                velocity_perplexity = losses[2]["perplexity"]

            masked_loss, visible_loss = balanced_masked_mse(prediction, target, mask)
            ramp = min(1.0, step / max(1, args.kinetic_warmup_steps))
            kinetic_weight = args.kinetic_weight * ramp if args.variant == "kinetic" else 0.0
            total_loss = (
                masked_loss
                + args.visible_weight * visible_loss
                + args.vq_weight * vq_loss
                + kinetic_weight * ode_loss
            )

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Non-finite loss at step={step}: total={total_loss.item()}"
            )
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        cells_seen += int(target.shape[0])
        loss_value = float(total_loss.detach())
        ema_loss = loss_value if ema_loss is None else 0.98 * ema_loss + 0.02 * loss_value

        if step == 1 or step % args.log_every == 0:
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - start_time
            payload = {
                "step": step,
                "cells_seen": cells_seen,
                "loss": loss_value,
                "loss_ema": ema_loss,
                "masked_loss": float(masked_loss.detach()),
                "visible_loss": float(visible_loss.detach()),
                "ode_loss": float(ode_loss.detach()),
                "vq_loss": float(vq_loss.detach()),
                "manifold_perplexity": float(manifold_perplexity.detach()),
                "velocity_perplexity": float(velocity_perplexity.detach()),
                "grad_norm": float(torch.as_tensor(grad_norm).detach()),
                "elapsed_seconds": elapsed,
                "cells_per_second": cells_seen / max(elapsed, 1.0e-9),
                "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "reserved_memory_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                "kinetic_weight": kinetic_weight,
            }
            append_jsonl(metrics_path, payload)
            print("PRETRAIN_METRIC", json.dumps(payload, ensure_ascii=False), flush=True)

        if step % args.save_every == 0:
            save_pretrain_checkpoint(
                output_dir / f"checkpoint_step_{step}.pt",
                model,
                optimizer,
                variant=args.variant,
                step=step,
                cells_seen=cells_seen,
                batch_size=args.batch_size,
                spec=spec,
                elapsed_seconds=time.monotonic() - start_time,
            )

    final_step = step if "step" in locals() else 0
    final_path = output_dir / "pretrain_final.pt"
    save_pretrain_checkpoint(
        final_path,
        model,
        optimizer,
        variant=args.variant,
        step=final_step,
        cells_seen=cells_seen,
        batch_size=args.batch_size,
        spec=spec,
        elapsed_seconds=time.monotonic() - start_time,
    )
    summary = {
        **metadata,
        "finished_at_unix": time.time(),
        "final_step": final_step,
        "cells_seen": cells_seen,
        "loss_ema": ema_loss,
        "elapsed_seconds": time.monotonic() - start_time,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "final_checkpoint": str(final_path),
    }
    json_dump(output_dir / "pretrain_summary.json", summary)
    print("PRETRAIN_DONE", json.dumps(summary, ensure_ascii=False), flush=True)
    del model, optimizer, loader, iterator
    gc.collect()
    torch.cuda.empty_cache()
    return final_path


def prepare_mousebrain(path: Path, seed: int):
    adata = load_data(str(path))
    adata = preprocess_data_scv(adata, n_top_genes=None, n_neighbors=20, seed=seed)
    cluster_key, _ = find_cluster_key(adata)
    adata.obs["cluster"] = adata.obs[cluster_key]
    dataset = NormalizedMouseBrainData(adata)
    full_adj = adata.obsp["connectivities"]
    directed_adj, root_distance = build_root_directed_adjacency(full_adj, dataset.root_mask)
    adata.obs["root_distance"] = root_distance
    return adata, dataset, full_adj, directed_adj, cluster_key


def evaluate_mousebrain(model, adata, dataset, cluster_key: str, *, batch_size: int, device: str):
    import scvelo as scv

    evaluation_data = adata.copy()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    evaluation_data = compute_pred(evaluation_data, loader, model, device=device)
    scv.tl.velocity_embedding(evaluation_data, basis="umap", vkey="velocity")
    scv.tl.velocity_confidence(evaluation_data, vkey="velocity")
    confidence = float(evaluation_data.obs["velocity_confidence"].mean())
    iccoh = float(evaluate_ICCoh(evaluation_data, cluster_key=cluster_key, velocity_key="velocity"))

    edges = all_edges.get("MouseBrain.h5ad")
    cbdir_without = float("nan")
    cbdir_with = float("nan")
    if edges is not None:
        _, cbdir_without = evaluate_CBDir2(
            evaluation_data,
            cluster_key=cluster_key,
            velocity_key="velocity",
            cluster_edges=edges,
        )
        evaluation_data = compute_velocity_from_graph(
            evaluation_data,
            new_key="velocity_graph_umap",
        )
        _, cbdir_with = evaluate_CBDir2(
            evaluation_data,
            cluster_key=cluster_key,
            velocity_key="velocity_graph",
            cluster_edges=edges,
        )

    metrics = {
        "velocity_confidence": confidence,
        "iccoh": iccoh,
        "cbdir_without_graph": float(cbdir_without),
        "cbdir_with_graph": float(cbdir_with),
        "graph_improvement": float(cbdir_with - cbdir_without),
    }
    return metrics, evaluation_data


def load_pretrain_state(model: VeloModel, checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(payload["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Incompatible pretrain checkpoint: {incompatible}")
    return {
        key: payload.get(key)
        for key in ("variant", "step", "cells_seen", "batch_size", "elapsed_seconds")
    }


def run_transfer(args) -> dict:
    set_all_seeds(args.seed)
    torch.set_float32_matmul_precision("high")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = PilotModelSpec()
    adata, dataset, full_adj, directed_adj, cluster_key = prepare_mousebrain(
        args.mousebrain,
        args.seed,
    )
    graph_sampler = GraphBatchSampler(full_adj, batch_size=args.mouse_batch_size, seed=args.seed)
    train_loader = DataLoader(dataset, batch_sampler=graph_sampler)
    config = Config(adata, seed=args.seed)
    configure_model(config, spec, epochs=args.mouse_epochs, lr=args.mouse_lr)
    config.config["train"]["batch_size"] = args.mouse_batch_size
    model = VeloModel(train_loader, config=config, device="cuda:0")
    pretrain_metadata = None
    if args.checkpoint is not None:
        pretrain_metadata = load_pretrain_state(model, args.checkpoint)

    metadata = {
        "tag": args.tag,
        "seed": args.seed,
        "physical_gpu": args.physical_gpu,
        "mousebrain_path": str(args.mousebrain),
        "mousebrain_shape": [int(adata.n_obs), int(adata.n_vars)],
        "mouse_batch_size": args.mouse_batch_size,
        "mouse_epochs": args.mouse_epochs,
        "mouse_lr": args.mouse_lr,
        "parameters": count_parameters(model),
        "model_spec": asdict(spec),
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "pretrain_metadata": pretrain_metadata,
        "started_at_unix": time.time(),
    }
    json_dump(output_dir / "transfer_config.json", metadata)
    print("TRANSFER_CONFIG", json.dumps(metadata, ensure_ascii=False), flush=True)

    device = "cuda:0"
    model.to(device)
    zero_shot = None
    if args.checkpoint is not None:
        try:
            zero_shot, _ = evaluate_mousebrain(
                model,
                adata,
                dataset,
                cluster_key,
                batch_size=args.eval_batch_size,
                device=device,
            )
            print("ZERO_SHOT_METRICS", json.dumps(zero_shot, ensure_ascii=False), flush=True)
        except Exception as exc:
            zero_shot = {"error": f"{type(exc).__name__}: {exc}"}
            print("ZERO_SHOT_ERROR", json.dumps(zero_shot, ensure_ascii=False), flush=True)

    torch.cuda.reset_peak_memory_stats()
    train_start = time.monotonic()
    model.fit(
        adjacency_matrix=full_adj,
        directed_adjacency_matrix=directed_adj,
        device=device,
        save_path=str(output_dir / "checkpoint"),
    )
    train_seconds = time.monotonic() - train_start
    final_metrics, evaluated = evaluate_mousebrain(
        model,
        adata,
        dataset,
        cluster_key,
        batch_size=args.eval_batch_size,
        device=device,
    )
    evaluated.write_h5ad(output_dir / "mousebrain_evaluated.h5ad", compression="gzip")

    result = {
        **metadata,
        "zero_shot": zero_shot,
        "fine_tuned": final_metrics,
        "train_seconds": train_seconds,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "finished_at_unix": time.time(),
    }
    json_dump(output_dir / "transfer_summary.json", result)
    print("TRANSFER_DONE", json.dumps(result, ensure_ascii=False), flush=True)
    del model, train_loader, evaluated, adata, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return result


def add_common_model_arguments(parser):
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    pretrain = subparsers.add_parser("pretrain")
    add_common_model_arguments(pretrain)
    pretrain.add_argument("--variant", choices=("denoise", "kinetic"), required=True)
    pretrain.add_argument("--pretrain-root", type=Path, default=DEFAULT_PRETRAIN_ROOT)
    pretrain.add_argument("--batch-size", type=int, required=True)
    pretrain.add_argument("--workers", type=int, default=2)
    pretrain.add_argument("--cells-per-dataset-cap", type=int, default=20_000)
    pretrain.add_argument("--max-steps", type=int, default=1_000_000)
    pretrain.add_argument("--max-hours", type=float, default=5.5)
    pretrain.add_argument("--pretrain-lr", type=float, default=2e-4)
    pretrain.add_argument("--gene-mask-rate", type=float, default=0.20)
    pretrain.add_argument("--layer-mask-rate", type=float, default=0.10)
    pretrain.add_argument("--visible-weight", type=float, default=0.10)
    pretrain.add_argument("--vq-weight", type=float, default=0.02)
    pretrain.add_argument("--kinetic-weight", type=float, default=1.0)
    pretrain.add_argument("--kinetic-warmup-steps", type=int, default=500)
    pretrain.add_argument("--grad-clip", type=float, default=1.0)
    pretrain.add_argument("--log-every", type=int, default=10)
    pretrain.add_argument("--save-every", type=int, default=250)

    transfer = subparsers.add_parser("transfer")
    add_common_model_arguments(transfer)
    transfer.add_argument("--tag", required=True)
    transfer.add_argument("--checkpoint", type=Path)
    transfer.add_argument("--mousebrain", type=Path, default=DEFAULT_MOUSEBRAIN)
    transfer.add_argument("--mouse-batch-size", type=int, default=128)
    transfer.add_argument("--eval-batch-size", type=int, default=32)
    transfer.add_argument("--mouse-epochs", type=int, default=10)
    transfer.add_argument("--mouse-lr", type=float, default=3e-4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this pilot")
    if args.command == "pretrain":
        run_pretrain(args)
    elif args.command == "transfer":
        run_transfer(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
