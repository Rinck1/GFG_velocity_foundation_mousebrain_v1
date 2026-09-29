#!/usr/bin/env python3
"""Synchronous DDP pretraining for GFG-v2.

One optimizer step consumes one batch sampled from one H5AD file. Rank 0 loads
the complete global batch, broadcasts it to every rank, and each rank trains on
one disjoint slice. The phase-portrait target is computed from the complete
global batch on every rank before slicing, so it is independent of world size.

This entry point intentionally has no gradient-accumulation option: every step
contains exactly one forward pass, one backward pass, and one optimizer update.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import anndata as ad
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from foundation_v2 import (
    DEFAULT_PRETRAIN_ROOT,
    H5ADBatchStream,
    V2ModelSpec,
    VelocityFoundationV2,
    add_model_arguments,
    append_jsonl,
    balanced_masked_mse,
    build_model,
    count_parameters,
    json_dump,
    load_paths,
    make_spliced_mask,
    make_velocity_mask,
    phase_direction_loss,
    save_checkpoint,
    set_all_seeds,
    shared_phase_velocity,
    spec_from_args,
    vars_for_json,
    velocity_anti_collapse_loss,
)


class StateStageModule(nn.Module):
    """DDP-visible state-stage forward graph."""

    def __init__(self, model: VelocityFoundationV2):
        super().__init__()
        self.state_encoder = model.state_encoder
        self.state_decoder = model.state_decoder

    def forward(self, spliced: torch.Tensor) -> torch.Tensor:
        state = self.state_encoder(spliced)
        return self.state_decoder(state)


class VelocityStageModule(nn.Module):
    """DDP-visible velocity-stage graph, including the decoder JVP."""

    def __init__(self, model: VelocityFoundationV2, secant_epsilon: float = 0.0):
        super().__init__()
        self.state_encoder = model.state_encoder
        self.velocity_encoder = model.velocity_encoder
        self.state_decoder = model.state_decoder
        self.secant_epsilon = float(secant_epsilon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        unspliced, spliced = inputs.chunk(2, dim=1)
        with torch.no_grad():
            state = self.state_encoder(spliced)
        tangent = self.velocity_encoder(unspliced, spliced, state)
        _, velocity = torch.autograd.functional.jvp(
            self.state_decoder,
            (state,),
            (tangent,),
            create_graph=torch.is_grad_enabled(),
        )
        if self.secant_epsilon <= 0:
            return velocity
        decoded = self.state_decoder(state)
        decoded_step = self.state_decoder(state + self.secant_epsilon * tangent)
        secant = (decoded_step - decoded) / self.secant_epsilon
        return velocity, secant


def augment_inputs(
    inputs: torch.Tensor,
    gene_dropout_rate: float,
    noise_std: float,
) -> torch.Tensor:
    """Create a hidden measurement perturbation without adding mask flags."""

    augmented = inputs
    if gene_dropout_rate > 0:
        augmented = augmented.clone()
        hidden_dropout = (torch.rand_like(augmented) < gene_dropout_rate) & (
            augmented > 0
        )
        augmented[hidden_dropout] = 0.0
    if noise_std > 0:
        if augmented is inputs:
            augmented = augmented.clone()
        positive = augmented > 0
        noise = torch.randn_like(augmented) * noise_std
        augmented = torch.where(positive, (augmented + noise).clamp_min(0.0), augmented)
    return augmented


def gene_phase_direction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align each gene's velocity ordering across cells.

    The original phase loss compares genes within each cell. This complementary
    term compares cells within each gene, constraining induction/repression
    ordering while remaining invariant to a gene-specific positive scale.
    """

    prediction = prediction.float()
    target = target.float()
    prediction = prediction - prediction.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    target_energy = target.square().sum(dim=0)
    valid = target_energy > eps
    if not valid.any():
        zero = prediction.sum() * 0.0
        return zero, zero.detach()
    numerator = (prediction[:, valid] * target[:, valid]).sum(dim=0)
    denominator = (
        prediction[:, valid].square().sum(dim=0).clamp_min(eps).sqrt()
        * target_energy[valid].sqrt()
    )
    cosine = (numerator / denominator).clamp(min=-1.0, max=1.0)
    return (1.0 - cosine).mean(), cosine.mean().detach()


def distributed_context() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            "Launch this entry point with torchrun; missing environment: "
            + ", ".join(missing)
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank, device


def discover_gene_dimensions(
    args, rank: int, device: torch.device
) -> tuple[int, int]:
    dimensions = torch.zeros(2, dtype=torch.int64, device=device)
    if rank == 0:
        paths = load_paths(args.pretrain_root, "train")
        example = ad.read_h5ad(paths[0], backed="r")
        try:
            source_genes = int(example.n_vars)
        finally:
            example.file.close()
        training_genes = args.genes_per_batch or source_genes
        if training_genes <= 0 or training_genes > source_genes:
            raise ValueError(
                f"genes_per_batch={training_genes} must be in [1, {source_genes}]"
            )
        dimensions[:] = torch.tensor(
            (source_genes, training_genes), dtype=torch.int64, device=device
        )
    dist.broadcast(dimensions, src=0)
    return int(dimensions[0]), int(dimensions[1])


def make_rank0_iterator(
    args,
    rank: int,
    source_num_genes: int,
    training_num_genes: int,
) -> tuple[DataLoader | None, Iterable | None]:
    if rank != 0:
        return None, None
    paths = load_paths(args.pretrain_root, "train")
    stream = H5ADBatchStream(
        paths,
        batch_size=args.global_batch_size,
        seed=args.seed,
        cells_per_dataset_cap=args.cells_per_dataset_cap,
        genes_per_batch=(
            training_num_genes if training_num_genes < source_num_genes else None
        ),
    )
    loader = DataLoader(
        stream,
        batch_size=None,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
        multiprocessing_context="spawn" if args.workers > 0 else None,
    )
    return loader, iter(loader)


def broadcast_global_batch(
    iterator: Iterable | None,
    rank: int,
    global_batch_size: int,
    num_features: int,
    device: torch.device,
) -> torch.Tensor:
    """Broadcast one intact dataset batch; report loader failures to all ranks."""

    status = torch.ones(1, dtype=torch.int32, device=device)
    error_message = ""
    full_batch = None
    if rank == 0:
        try:
            cpu_batch = next(iterator)
            if tuple(cpu_batch.shape) != (global_batch_size, num_features):
                raise ValueError(
                    "Unexpected stream batch shape: "
                    f"got={tuple(cpu_batch.shape)} "
                    f"expected={(global_batch_size, num_features)}"
                )
            full_batch = cpu_batch.to(
                device=device, dtype=torch.float32, non_blocking=True
            ).contiguous()
        except Exception as exc:  # synchronize the failure instead of hanging peers
            status.zero_()
            error_message = f"{type(exc).__name__}: {exc}"
    dist.broadcast(status, src=0)
    if not bool(status.item()):
        messages = [error_message]
        dist.broadcast_object_list(messages, src=0, device=device)
        raise RuntimeError(f"Rank-0 data loader failed: {messages[0]}")
    if rank != 0:
        full_batch = torch.empty(
            (global_batch_size, num_features), dtype=torch.float32, device=device
        )
    dist.broadcast(full_batch, src=0)
    return full_batch


def local_slice(full_batch: torch.Tensor, rank: int, local_batch_size: int) -> torch.Tensor:
    start = rank * local_batch_size
    return full_batch.narrow(0, start, local_batch_size)


def global_mean(values: dict[str, torch.Tensor], world_size: int) -> dict[str, float]:
    keys = tuple(values)
    packed = torch.stack([values[key].detach().float() for key in keys])
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    packed /= world_size
    return {key: float(value) for key, value in zip(keys, packed, strict=True)}


def global_peak_memory_gib(device: torch.device) -> float:
    peak = torch.tensor(
        torch.cuda.max_memory_allocated(device) / 2**30,
        dtype=torch.float32,
        device=device,
    )
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    return float(peak)


def require_finite(loss: torch.Tensor, stage: str, step: int) -> None:
    finite = torch.isfinite(loss).to(dtype=torch.int32)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not bool(finite.item()):
        raise FloatingPointError(f"Non-finite {stage} loss at optimizer step={step}")


def ddp_kwargs(local_rank: int) -> dict:
    return {
        "device_ids": [local_rank],
        "output_device": local_rank,
        "broadcast_buffers": False,
    }


def train_state_stage(
    model: VelocityFoundationV2,
    iterator: Iterable | None,
    args,
    metrics_path: Path,
    rank: int,
    world_size: int,
    local_rank: int,
    local_batch_size: int,
    device: torch.device,
    start_time: float,
) -> tuple[int, dict]:
    parameters = model.set_training_stage("state")
    distributed_model = DDP(StateStageModule(model), **ddp_kwargs(local_rank))
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.state_lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    ema = None
    final = {}
    for step in range(1, args.state_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        full_target = broadcast_global_batch(
            iterator, rank, args.global_batch_size, 2 * model.G, device
        )
        target = local_slice(full_target, rank, local_batch_size)
        _, spliced = model.split_us(target)
        augmented_spliced = augment_inputs(
            spliced, args.input_gene_dropout_rate, args.input_noise_std
        )
        masked_spliced, mask = make_spliced_mask(
            augmented_spliced, args.state_mask_rate
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = distributed_model(masked_spliced)
            masked_loss, visible_loss = balanced_masked_mse(prediction, spliced, mask)
            loss = masked_loss + args.visible_weight * visible_loss
        require_finite(loss, "state", step)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step()
        averaged = global_mean(
            {
                "loss": loss,
                "masked_loss": masked_loss,
                "visible_loss": visible_loss,
                "grad_norm": torch.as_tensor(grad_norm, device=device),
            },
            world_size,
        )
        value = averaged["loss"]
        ema = value if ema is None else 0.98 * ema + 0.02 * value
        if step == 1 or step % args.log_every == 0 or step == args.state_steps:
            final = {
                "stage": "state",
                "step": step,
                "optimizer_steps": step,
                "cells_seen": step * args.global_batch_size,
                "loss": value,
                "loss_ema": ema,
                "masked_loss": averaged["masked_loss"],
                "visible_loss": averaged["visible_loss"],
                "grad_norm": averaged["grad_norm"],
                "world_size": world_size,
                "global_batch_size": args.global_batch_size,
                "local_batch_size": local_batch_size,
                "grad_accum_steps": 1,
                "elapsed_seconds": time.monotonic() - start_time,
                "peak_memory_gib": global_peak_memory_gib(device),
            }
            if rank == 0:
                append_jsonl(metrics_path, final)
                print("V2_DDP_METRIC", json.dumps(final, ensure_ascii=False), flush=True)
    del distributed_model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return args.state_steps * args.global_batch_size, final


def train_velocity_stage(
    model: VelocityFoundationV2,
    iterator: Iterable | None,
    args,
    metrics_path: Path,
    rank: int,
    world_size: int,
    local_rank: int,
    local_batch_size: int,
    device: torch.device,
    start_time: float,
) -> tuple[int, dict]:
    parameters = model.set_training_stage("velocity")
    secant_epsilon = args.secant_epsilon if args.secant_weight > 0 else 0.0
    distributed_model = DDP(
        VelocityStageModule(model, secant_epsilon=secant_epsilon),
        **ddp_kwargs(local_rank),
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.velocity_lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    ema = None
    final = {}
    for step in range(1, args.velocity_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        full_target = broadcast_global_batch(
            iterator, rank, args.global_batch_size, 2 * model.G, device
        )
        full_u, full_s = model.split_us(full_target)
        full_phase_target, slope, reliability = shared_phase_velocity(full_u, full_s)
        target = local_slice(full_target, rank, local_batch_size)
        phase_target = local_slice(full_phase_target, rank, local_batch_size)
        view_count = 2 if args.view_consistency_weight > 0 else 1
        masked_views = []
        for _ in range(view_count):
            augmented = augment_inputs(
                target, args.input_gene_dropout_rate, args.input_noise_std
            )
            masked, _ = make_velocity_mask(augmented, args.velocity_mask_rate)
            masked_views.append(masked)
        model_input = torch.cat(masked_views, dim=0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model_output = distributed_model(model_input)
            if secant_epsilon > 0:
                all_velocity, all_secant = model_output
                secant_views = all_secant.split(local_batch_size, dim=0)
            else:
                all_velocity = model_output
                secant_views = ()
            velocity_views = all_velocity.split(local_batch_size, dim=0)
            direction_pieces = []
            phase_cosine_pieces = []
            gene_direction_pieces = []
            gene_cosine_pieces = []
            for velocity_view in velocity_views:
                piece, cosine = phase_direction_loss(velocity_view, phase_target)
                direction_pieces.append(piece)
                phase_cosine_pieces.append(cosine)
                gene_piece, gene_cosine = gene_phase_direction_loss(
                    velocity_view, phase_target
                )
                gene_direction_pieces.append(gene_piece)
                gene_cosine_pieces.append(gene_cosine)
            direction_loss = torch.stack(direction_pieces).mean()
            phase_cosine = torch.stack(phase_cosine_pieces).mean()
            gene_direction_loss = torch.stack(gene_direction_pieces).mean()
            gene_phase_cosine = torch.stack(gene_cosine_pieces).mean()
            view_consistency_loss = all_velocity.new_zeros(())
            view_consistency_cosine = all_velocity.new_zeros(())
            if view_count == 2:
                view_consistency_loss, view_consistency_cosine = phase_direction_loss(
                    velocity_views[0], velocity_views[1]
                )
            secant_loss = all_velocity.new_zeros(())
            secant_cosine = all_velocity.new_zeros(())
            if secant_epsilon > 0:
                secant_pieces = []
                secant_cosines = []
                for velocity_view, secant_view in zip(
                    velocity_views, secant_views, strict=True
                ):
                    piece, cosine = phase_direction_loss(velocity_view, secant_view)
                    secant_pieces.append(piece)
                    secant_cosines.append(cosine)
                secant_loss = torch.stack(secant_pieces).mean()
                secant_cosine = torch.stack(secant_cosines).mean()
            collapse_loss = velocity_anti_collapse_loss(
                all_velocity, minimum_rms=args.minimum_velocity_rms
            )
            loss = (
                args.phase_weight * direction_loss
                + args.gene_direction_weight * gene_direction_loss
                + args.view_consistency_weight * view_consistency_loss
                + args.secant_weight * secant_loss
                + args.collapse_weight * collapse_loss
            )
        require_finite(loss, "velocity", step)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step()
        averaged = global_mean(
            {
                "loss": loss,
                "direction_loss": direction_loss,
                "phase_cosine": phase_cosine,
                "gene_direction_loss": gene_direction_loss,
                "gene_phase_cosine": gene_phase_cosine,
                "view_consistency_loss": view_consistency_loss,
                "view_consistency_cosine": view_consistency_cosine,
                "secant_loss": secant_loss,
                "secant_cosine": secant_cosine,
                "collapse_loss": collapse_loss,
                "velocity_rms": all_velocity.float().square().mean().sqrt(),
                "mean_phase_slope": slope.mean(),
                "mean_phase_reliability": reliability.mean(),
                "grad_norm": torch.as_tensor(grad_norm, device=device),
            },
            world_size,
        )
        value = averaged["loss"]
        ema = value if ema is None else 0.98 * ema + 0.02 * value
        if step == 1 or step % args.log_every == 0 or step == args.velocity_steps:
            final = {
                "stage": "velocity",
                "step": step,
                "optimizer_steps": step,
                "cells_seen": step * args.global_batch_size,
                "loss": value,
                "loss_ema": ema,
                "direction_loss": averaged["direction_loss"],
                "phase_cosine": averaged["phase_cosine"],
                "gene_direction_loss": averaged["gene_direction_loss"],
                "gene_phase_cosine": averaged["gene_phase_cosine"],
                "view_consistency_loss": averaged["view_consistency_loss"],
                "view_consistency_cosine": averaged["view_consistency_cosine"],
                "secant_loss": averaged["secant_loss"],
                "secant_cosine": averaged["secant_cosine"],
                "collapse_loss": averaged["collapse_loss"],
                "velocity_rms": averaged["velocity_rms"],
                "mean_phase_slope": averaged["mean_phase_slope"],
                "mean_phase_reliability": averaged["mean_phase_reliability"],
                "grad_norm": averaged["grad_norm"],
                "world_size": world_size,
                "global_batch_size": args.global_batch_size,
                "local_batch_size": local_batch_size,
                "phase_target_batch_size": args.global_batch_size,
                "grad_accum_steps": 1,
                "elapsed_seconds": time.monotonic() - start_time,
                "peak_memory_gib": global_peak_memory_gib(device),
            }
            if rank == 0:
                append_jsonl(metrics_path, final)
                print("V2_DDP_METRIC", json.dumps(final, ensure_ascii=False), flush=True)
    del distributed_model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return args.velocity_steps * args.global_batch_size, final


def validate_args(args, world_size: int) -> int:
    if args.global_batch_size <= 0:
        raise ValueError("--global-batch-size must be positive")
    if args.global_batch_size % world_size:
        raise ValueError(
            f"global batch {args.global_batch_size} is not divisible by {world_size} ranks"
        )
    if args.state_steps <= 0 or args.velocity_steps <= 0:
        raise ValueError("Both training-stage step counts must be positive")
    if args.direction_target != "phase":
        raise ValueError("The DDP scaling entry point currently supports phase target only")
    rates = {
        "input_gene_dropout_rate": args.input_gene_dropout_rate,
        "input_noise_std": args.input_noise_std,
        "gene_direction_weight": args.gene_direction_weight,
        "view_consistency_weight": args.view_consistency_weight,
        "secant_weight": args.secant_weight,
        "secant_epsilon": args.secant_epsilon,
    }
    if any(value < 0 for value in rates.values()):
        raise ValueError(f"Augmentation/loss controls must be non-negative: {rates}")
    if args.input_gene_dropout_rate >= 1:
        raise ValueError("--input-gene-dropout-rate must be less than one")
    if args.secant_weight > 0 and args.secant_epsilon <= 0:
        raise ValueError("Positive --secant-weight requires --secant-epsilon > 0")
    return args.global_batch_size // world_size


def run_pretrain(args) -> Path:
    rank, world_size, local_rank, device = distributed_context()
    try:
        local_batch_size = validate_args(args, world_size)
        set_all_seeds(args.seed)
        torch.set_float32_matmul_precision("high")
        source_num_genes, training_num_genes = discover_gene_dimensions(
            args, rank, device
        )
        spec: V2ModelSpec = spec_from_args(args)
        model = build_model(training_num_genes, spec, device)
        parameter_counts = count_parameters(model)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = args.output_dir / "pretrain_metrics.jsonl"
        final_path = args.output_dir / "pretrain_final.pt"
        config_path = args.output_dir / "pretrain_config.json"
        if metrics_path.exists() or final_path.exists() or config_path.exists():
            raise FileExistsError(
                f"Refusing to append to an existing DDP run: {args.output_dir}"
            )
        dist.barrier()

        loader, iterator = make_rank0_iterator(
            args, rank, source_num_genes, training_num_genes
        )
        physical_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "all-visible")
        metadata = {
            "seed": args.seed,
            "physical_gpus": physical_gpus,
            "pretrain_root": str(args.pretrain_root),
            "source_num_genes": source_num_genes,
            "training_num_genes": training_num_genes,
            "gene_sampling": (
                "all_simultaneous"
                if training_num_genes == source_num_genes
                else "uniform_without_replacement_per_global_batch"
            ),
            "world_size": world_size,
            "global_batch_size": args.global_batch_size,
            "local_batch_size": local_batch_size,
            "gradient_accumulation": False,
            "grad_accum_steps": 1,
            "phase_target_batch_size": args.global_batch_size,
            "batch_sampling": "one intact H5AD batch broadcast then rank-sliced",
            "augmentation": {
                "input_gene_dropout_rate": args.input_gene_dropout_rate,
                "input_noise_std": args.input_noise_std,
                "velocity_views": 2 if args.view_consistency_weight > 0 else 1,
            },
            "velocity_objective": {
                "cell_phase_weight": args.phase_weight,
                "gene_direction_weight": args.gene_direction_weight,
                "view_consistency_weight": args.view_consistency_weight,
                "secant_weight": args.secant_weight,
                "secant_epsilon": args.secant_epsilon,
                "collapse_weight": args.collapse_weight,
            },
            "state_steps": args.state_steps,
            "velocity_steps": args.velocity_steps,
            "model_spec": asdict(spec),
            "parameters": parameter_counts,
            "started_at_unix": time.time(),
        }
        if rank == 0:
            json_dump(config_path, metadata | vars_for_json(args))
            print("V2_DDP_CONFIG", json.dumps(metadata, ensure_ascii=False), flush=True)

        # Model weights are identical first; training-time masks and dropout differ by rank.
        rank_seed = args.seed + 100_003 * rank
        torch.manual_seed(rank_seed)
        torch.cuda.manual_seed_all(rank_seed)
        start_time = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)

        state_cells, state_final = train_state_stage(
            model,
            iterator,
            args,
            metrics_path,
            rank,
            world_size,
            local_rank,
            local_batch_size,
            device,
            start_time,
        )
        if args.save_state_checkpoint and not args.skip_checkpoints and rank == 0:
            save_checkpoint(
                args.output_dir / "state_final.pt",
                model,
                spec,
                metadata | {"stage": "state", "state_cells_seen": state_cells},
            )
        dist.barrier()

        velocity_cells, velocity_final = train_velocity_stage(
            model,
            iterator,
            args,
            metrics_path,
            rank,
            world_size,
            local_rank,
            local_batch_size,
            device,
            start_time,
        )
        peak_memory = global_peak_memory_gib(device)
        summary = metadata | {
            "stage": "complete",
            "state_cells_seen": state_cells,
            "velocity_cells_seen": velocity_cells,
            "state_final": state_final,
            "velocity_final": velocity_final,
            "direction_target": "phase",
            "elapsed_seconds": time.monotonic() - start_time,
            "peak_memory_gib": peak_memory,
            "final_checkpoint": None if args.skip_checkpoints else str(final_path),
            "finished_at_unix": time.time(),
        }
        if rank == 0:
            if not args.skip_checkpoints:
                save_checkpoint(final_path, model, spec, summary)
            json_dump(args.output_dir / "pretrain_summary.json", summary)
            print("V2_DDP_PRETRAIN_DONE", json.dumps(summary, ensure_ascii=False), flush=True)
        dist.barrier()
        del model, iterator, loader
        gc.collect()
        torch.cuda.empty_cache()
        return final_path
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synchronous DDP pretraining without gradient accumulation"
    )
    add_model_arguments(parser)
    parser.add_argument("--pretrain-root", type=Path, default=DEFAULT_PRETRAIN_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--global-batch-size", type=int, default=384)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cells-per-dataset-cap", type=int, default=20_000)
    parser.add_argument(
        "--genes-per-batch",
        type=int,
        help="Uniformly sample this many genes per global batch; default uses all genes",
    )
    parser.add_argument("--state-steps", type=int, required=True)
    parser.add_argument("--velocity-steps", type=int, required=True)
    parser.add_argument("--state-lr", type=float, default=2.0e-4)
    parser.add_argument("--velocity-lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--state-mask-rate", type=float, default=0.30)
    parser.add_argument("--velocity-mask-rate", type=float, default=0.10)
    parser.add_argument("--visible-weight", type=float, default=0.10)
    parser.add_argument("--phase-weight", type=float, default=1.0)
    parser.add_argument("--gene-direction-weight", type=float, default=0.0)
    parser.add_argument("--view-consistency-weight", type=float, default=0.0)
    parser.add_argument("--secant-weight", type=float, default=0.0)
    parser.add_argument("--secant-epsilon", type=float, default=0.10)
    parser.add_argument("--input-gene-dropout-rate", type=float, default=0.0)
    parser.add_argument("--input-noise-std", type=float, default=0.0)
    parser.add_argument("--direction-target", choices=("phase",), default="phase")
    parser.add_argument("--collapse-weight", type=float, default=0.01)
    parser.add_argument("--minimum-velocity-rms", type=float, default=1.0e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-state-checkpoint", action="store_true")
    parser.add_argument("--skip-checkpoints", action="store_true")
    return parser


def main() -> None:
    run_pretrain(build_parser().parse_args())


if __name__ == "__main__":
    main()
