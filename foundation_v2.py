#!/usr/bin/env python3
"""GFG-v2 pretraining pilot.

The v2 pilot separates stable cell state from velocity evidence:

* state branch: spliced-only set encoder -> spliced decoder;
* velocity branch: unspliced + spliced + detached state -> latent tangent;
* observed spliced velocity: decoder JVP at the state along the tangent;
* pretraining direction: a batch-shared phase-portrait residual, rather than
  the original independently solved cell-by-gene ODE residual.

Both standard self-attention and inducing-point Set Transformer encoders are
available so that architecture and loss changes can be ablated independently.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from foundation_pilot import (
    DEFAULT_MOUSEBRAIN,
    DEFAULT_PRETRAIN_ROOT,
    H5ADBatchStream,
    PilotModelSpec as V1ModelSpec,
    all_edges,
    append_jsonl,
    compute_velocity_from_graph,
    evaluate_mousebrain,
    evaluate_CBDir2,
    evaluate_ICCoh,
    json_dump,
    load_paths,
    prepare_mousebrain,
    set_all_seeds,
    build_model as build_v1_model,
)


@dataclass(frozen=True)
class V2ModelSpec:
    encoder_type: str = "set"
    d_model: int = 384
    num_heads: int = 8
    num_blocks: int = 2
    num_inducing: int = 128
    ff_mult: int = 2
    dropout: float = 0.10
    decoder_hidden: tuple[int, ...] = (768, 384)
    detach_state_for_velocity: bool = True

    def validate(self) -> None:
        if self.encoder_type not in {"set", "standard"}:
            raise ValueError(f"Unknown encoder_type={self.encoder_type!r}")
        if self.d_model <= 0 or self.d_model % self.num_heads:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if self.encoder_type == "set" and self.num_inducing <= 0:
            raise ValueError("num_inducing must be positive for Set Transformer")


class ResidualFeedForward(nn.Module):
    def __init__(self, d_model: int, ff_mult: int, dropout: float):
        super().__init__()
        hidden = d_model * ff_mult
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiheadAttentionBlock(nn.Module):
    """Set Transformer MAB with pre-normalized queries and key/value tokens."""

    def __init__(self, d_model: int, num_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.key_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.post_attention_norm = nn.LayerNorm(d_model)
        self.feed_forward = ResidualFeedForward(d_model, ff_mult, dropout)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        q = self.query_norm(query)
        kv = self.key_norm(key_value)
        attended, _ = self.attention(q, kv, kv, need_weights=False)
        hidden = self.post_attention_norm(query + attended)
        return self.output_norm(hidden + self.feed_forward(hidden))


class InducedSetAttentionBlock(nn.Module):
    """ISAB: O(G*m) interaction through m learned inducing tokens."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_inducing: int,
        ff_mult: int,
        dropout: float,
    ):
        super().__init__()
        self.inducing = nn.Parameter(torch.empty(1, num_inducing, d_model))
        nn.init.trunc_normal_(self.inducing, std=0.02)
        self.inducing_from_set = MultiheadAttentionBlock(
            d_model, num_heads, ff_mult, dropout
        )
        self.set_from_inducing = MultiheadAttentionBlock(
            d_model, num_heads, ff_mult, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inducing = self.inducing.expand(x.shape[0], -1, -1)
        hidden = self.inducing_from_set(inducing, x)
        return self.set_from_inducing(x, hidden)


class SetEncoderCore(nn.Module):
    def __init__(self, spec: V2ModelSpec):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                InducedSetAttentionBlock(
                    spec.d_model,
                    spec.num_heads,
                    spec.num_inducing,
                    spec.ff_mult,
                    spec.dropout,
                )
                for _ in range(spec.num_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(spec.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.output_norm(x)


class StandardAttentionCore(nn.Module):
    """Parameter-comparable full attention baseline without position encoding."""

    def __init__(self, spec: V2ModelSpec):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=spec.d_model,
            nhead=spec.num_heads,
            dim_feedforward=spec.d_model * spec.ff_mult,
            dropout=spec.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # One ISAB contains two attention blocks. Match that depth here.
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=2 * spec.num_blocks,
            norm=nn.LayerNorm(spec.d_model),
            enable_nested_tensor=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def make_encoder_core(spec: V2ModelSpec) -> nn.Module:
    if spec.encoder_type == "set":
        return SetEncoderCore(spec)
    if spec.encoder_type == "standard":
        return StandardAttentionCore(spec)
    raise AssertionError(spec.encoder_type)


class StateEncoder(nn.Module):
    """Encode only spliced expression; gene order remains permutation equivariant."""

    def __init__(self, spec: V2ModelSpec):
        super().__init__()
        self.tokenizer = nn.Sequential(
            nn.Linear(2, spec.d_model),
            nn.GELU(),
            nn.LayerNorm(spec.d_model),
        )
        self.core = make_encoder_core(spec)

    def forward(self, spliced: torch.Tensor) -> torch.Tensor:
        mask_flag = (spliced < 0).to(spliced.dtype)
        tokens = torch.stack((spliced, mask_flag), dim=-1)
        return self.core(self.tokenizer(tokens))


class VelocityEncoder(nn.Module):
    """Encode U/S phase evidence conditioned on the stable state representation."""

    def __init__(self, spec: V2ModelSpec):
        super().__init__()
        self.detach_state = spec.detach_state_for_velocity
        self.tokenizer = nn.Sequential(
            nn.Linear(spec.d_model + 5, spec.d_model),
            nn.GELU(),
            nn.LayerNorm(spec.d_model),
        )
        self.core = make_encoder_core(spec)
        self.direction = nn.Linear(spec.d_model, spec.d_model)
        self.speed = nn.Linear(spec.d_model, 1)

    def forward(
        self,
        unspliced: torch.Tensor,
        spliced: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        if self.detach_state:
            state = state.detach()
        u_mask = (unspliced < 0).to(unspliced.dtype)
        s_mask = (spliced < 0).to(spliced.dtype)
        measurements = torch.stack(
            (unspliced, spliced, unspliced - spliced, u_mask, s_mask),
            dim=-1,
        )
        hidden = self.tokenizer(torch.cat((measurements, state), dim=-1))
        hidden = self.core(hidden)
        direction = F.normalize(self.direction(hidden).float(), dim=-1).to(hidden.dtype)
        speed = F.softplus(self.speed(hidden).float()).to(hidden.dtype) + 1.0e-3
        return direction * speed


class SplicedDecoder(nn.Module):
    """Shared per-gene decoder used both for reconstruction and its JVP."""

    def __init__(self, spec: V2ModelSpec):
        super().__init__()
        layers: list[nn.Module] = []
        width = spec.d_model
        for hidden in spec.decoder_hidden:
            layers.extend((nn.Linear(width, hidden), nn.GELU()))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class VelocityFoundationV2(nn.Module):
    def __init__(self, num_genes: int, spec: V2ModelSpec):
        super().__init__()
        spec.validate()
        self.G = int(num_genes)
        self.spec = spec
        self.state_encoder = StateEncoder(spec)
        self.velocity_encoder = VelocityEncoder(spec)
        self.state_decoder = SplicedDecoder(spec)

    @staticmethod
    def split_us(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 2 or inputs.shape[1] % 2:
            raise ValueError(f"Expected inputs shaped (B, 2G), got {tuple(inputs.shape)}")
        return inputs.chunk(2, dim=1)

    def encode_state(self, spliced: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.state_encoder(spliced)
        reconstruction = self.state_decoder(state)
        return state, reconstruction

    def decode_jvp(
        self,
        state: torch.Tensor,
        tangent: torch.Tensor,
        create_graph: bool | None = None,
    ) -> torch.Tensor:
        if create_graph is None:
            create_graph = torch.is_grad_enabled()
        _, velocity = torch.autograd.functional.jvp(
            self.state_decoder,
            (state,),
            (tangent,),
            create_graph=create_graph,
        )
        return velocity

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        unspliced, spliced = self.split_us(inputs)
        state, spliced_hat = self.encode_state(spliced)
        tangent = self.velocity_encoder(unspliced, spliced, state)
        spliced_velocity = self.decode_jvp(state, tangent)
        return spliced_hat, spliced_velocity, state, tangent

    def set_training_stage(self, stage: str) -> list[nn.Parameter]:
        if stage not in {"state", "velocity", "joint"}:
            raise ValueError(stage)
        state_trainable = stage in {"state", "joint"}
        velocity_trainable = stage in {"velocity", "joint"}
        self.state_encoder.requires_grad_(state_trainable)
        self.state_decoder.requires_grad_(state_trainable)
        self.velocity_encoder.requires_grad_(velocity_trainable)
        self.state_encoder.train(state_trainable)
        self.state_decoder.train(state_trainable)
        self.velocity_encoder.train(velocity_trainable)
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def predict(self, batch: torch.Tensor, align: bool = False, device: str = "cpu"):
        del align
        self.eval()
        batch = batch.to(device)
        unspliced, spliced = self.split_us(batch)
        # Encoder inference does not need a parameter graph. JVP internally enables
        # differentiation only with respect to its state primal, and create_graph=False
        # prevents retaining a higher-order graph across evaluation batches.
        with torch.no_grad():
            state, spliced_hat = self.encode_state(spliced)
            tangent = self.velocity_encoder(unspliced, spliced, state)
        with torch.enable_grad():
            spliced_velocity = self.decode_jvp(
                state, tangent, create_graph=False
            )
        x_hat = torch.cat((unspliced, spliced_hat), dim=1)
        velocity = torch.cat((torch.zeros_like(spliced_velocity), spliced_velocity), dim=1)
        return x_hat.detach().cpu(), velocity.detach().cpu()


def count_parameters(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def make_spliced_mask(
    spliced: torch.Tensor, mask_rate: float
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.rand_like(spliced) < mask_rate
    masked = spliced.clone()
    masked[mask] = -1.0
    return masked, mask


def make_velocity_mask(
    inputs: torch.Tensor, mask_rate: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask_rate <= 0:
        return inputs, torch.zeros_like(inputs, dtype=torch.bool)
    mask = torch.rand_like(inputs) < mask_rate
    masked = inputs.clone()
    masked[mask] = -1.0
    return masked, mask


def balanced_masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    error = (prediction.float() - target.float()).square()
    pieces = []
    masked_nonzero = mask & (target > 0)
    masked_zero = mask & ~(target > 0)
    if masked_nonzero.any():
        pieces.append(error[masked_nonzero].mean())
    if masked_zero.any():
        pieces.append(error[masked_zero].mean())
    if not pieces:
        raise ValueError("At least one masked element is required")
    masked_loss = torch.stack(pieces).mean()
    visible_loss = error[~mask].mean() if (~mask).any() else error.new_zeros(())
    return masked_loss, visible_loss


@torch.no_grad()
def shared_phase_velocity(
    unspliced: torch.Tensor,
    spliced: torch.Tensor,
    eps: float = 1.0e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a dataset-batch-shared U~S residual direction.

    A single non-negative phase slope is fitted per gene across the batch.
    Unlike the old loss, no kinetic parameter is independently solved for each
    cell-by-gene observation and the target does not depend on model velocity.
    """

    u = unspliced.float()
    s = spliced.float()
    u_centered = u - u.mean(dim=0, keepdim=True)
    s_centered = s - s.mean(dim=0, keepdim=True)
    s_variance = s_centered.square().mean(dim=0)
    u_variance = u_centered.square().mean(dim=0)
    covariance = (u_centered * s_centered).mean(dim=0)
    slope = (covariance / (s_variance + eps)).clamp(min=0.0, max=10.0)
    equilibrium_u = u.mean(dim=0, keepdim=True) + slope * s_centered
    residual = u - equilibrium_u

    reliability = torch.sqrt(
        (s_variance / (s_variance + 0.05))
        * (u_variance / (u_variance + 0.05))
    )
    residual = residual * reliability.unsqueeze(0)
    return residual, slope, reliability


def phase_direction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = prediction.float()
    target = target.float()
    target_norm = target.norm(dim=1)
    valid = target_norm > eps
    if not valid.any():
        zero = prediction.sum() * 0.0
        return zero, zero.detach()
    cosine = F.cosine_similarity(prediction[valid], target[valid], dim=1, eps=eps)
    loss = (1.0 - cosine).mean()
    return loss, cosine.mean().detach()


def velocity_anti_collapse_loss(
    velocity: torch.Tensor, minimum_rms: float = 1.0e-3
) -> torch.Tensor:
    rms = velocity.float().square().mean().sqrt()
    return F.relu(velocity.new_tensor(minimum_rms) - rms).square()


def build_model(num_genes: int, spec: V2ModelSpec, device: torch.device):
    model = VelocityFoundationV2(num_genes, spec)
    return model.to(device)


def load_v1_teacher(
    checkpoint: Path,
    num_genes: int,
    seed: int,
    device: torch.device,
):
    teacher, _ = build_v1_model(num_genes, V1ModelSpec(), seed, lr=2.0e-4)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    incompatible = teacher.load_state_dict(payload["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Incompatible v1 teacher checkpoint: {incompatible}")
    teacher.requires_grad_(False)
    teacher.eval().to(device)
    return teacher, {
        key: payload.get(key)
        for key in ("variant", "step", "cells_seen", "batch_size", "elapsed_seconds")
    }


def v1_teacher_spliced_velocity(
    teacher,
    inputs: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    outputs = []
    for chunk in inputs.split(chunk_size, dim=0):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, velocity, _, _, _ = teacher(chunk)
        outputs.append(velocity[:, teacher.G :].float().detach())
    return torch.cat(outputs, dim=0)


def save_checkpoint(
    path: Path,
    model: VelocityFoundationV2,
    spec: V2ModelSpec,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "model_spec": asdict(spec),
        **metadata,
    }
    torch.save(payload, path)


def load_micro_targets(
    iterator: Iterable[torch.Tensor],
    args,
    device: torch.device,
) -> list[torch.Tensor]:
    """Load one optimizer batch and split it into memory-safe micro batches."""

    if args.stream_batch_size is None:
        cpu_targets = [next(iterator) for _ in range(args.grad_accum_steps)]
    else:
        effective_target = next(iterator)
        expected = args.batch_size * args.grad_accum_steps
        if effective_target.shape[0] != expected:
            raise ValueError(
                "Stream batch size does not match micro batch accumulation: "
                f"got={effective_target.shape[0]} expected={expected}"
            )
        cpu_targets = list(effective_target.split(args.batch_size, dim=0))
        if len(cpu_targets) != args.grad_accum_steps:
            raise AssertionError(
                f"Expected {args.grad_accum_steps} micro batches, got {len(cpu_targets)}"
            )
    return [
        target.to(device=device, non_blocking=True) for target in cpu_targets
    ]


def train_state_stage(
    model: VelocityFoundationV2,
    iterator: Iterable[torch.Tensor],
    args,
    metrics_path: Path,
    device: torch.device,
    start_time: float,
) -> tuple[int, dict]:
    parameters = model.set_training_stage("state")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.state_lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    cells_seen = 0
    ema = None
    final = {}
    for step in range(1, args.state_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        micro_targets = load_micro_targets(iterator, args, device)
        loss_values = []
        masked_loss_values = []
        visible_loss_values = []
        for micro_step, target in enumerate(micro_targets):
            _, spliced = model.split_us(target)
            masked_spliced, mask = make_spliced_mask(spliced, args.state_mask_rate)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, prediction = model.encode_state(masked_spliced)
                masked_loss, visible_loss = balanced_masked_mse(
                    prediction, spliced, mask
                )
                loss = masked_loss + args.visible_weight * visible_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite state loss at step={step}, micro_step={micro_step + 1}"
                )
            (loss / args.grad_accum_steps).backward()
            cells_seen += int(target.shape[0])
            loss_values.append(float(loss.detach()))
            masked_loss_values.append(float(masked_loss.detach()))
            visible_loss_values.append(float(visible_loss.detach()))
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step()
        value = float(np.mean(loss_values))
        masked_value = float(np.mean(masked_loss_values))
        visible_value = float(np.mean(visible_loss_values))
        ema = value if ema is None else 0.98 * ema + 0.02 * value
        if step == 1 or step % args.log_every == 0 or step == args.state_steps:
            final = {
                "stage": "state",
                "step": step,
                "cells_seen": cells_seen,
                "loss": value,
                "loss_ema": ema,
                "masked_loss": masked_value,
                "visible_loss": visible_value,
                "grad_accum_steps": args.grad_accum_steps,
                "grad_norm": float(torch.as_tensor(grad_norm)),
                "elapsed_seconds": time.monotonic() - start_time,
                "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            }
            append_jsonl(metrics_path, final)
            print("V2_METRIC", json.dumps(final, ensure_ascii=False), flush=True)
    return cells_seen, final


def train_velocity_stage(
    model: VelocityFoundationV2,
    iterator: Iterable[torch.Tensor],
    args,
    metrics_path: Path,
    device: torch.device,
    start_time: float,
    teacher=None,
) -> tuple[int, dict]:
    parameters = model.set_training_stage("velocity")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.velocity_lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    cells_seen = 0
    ema = None
    final = {}
    for step in range(1, args.velocity_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        micro_targets = load_micro_targets(iterator, args, device)
        effective_target = torch.cat(micro_targets, dim=0)
        effective_u, effective_s = model.split_us(effective_target)
        effective_phase_target, slope, reliability = shared_phase_velocity(
            effective_u, effective_s
        )
        phase_target_chunks = effective_phase_target.split(
            [target.shape[0] for target in micro_targets], dim=0
        )
        metric_values = {
            key: []
            for key in (
                "loss",
                "direction_loss",
                "phase_cosine",
                "teacher_loss",
                "teacher_cosine",
                "teacher_phase_cosine",
                "collapse_loss",
                "velocity_rms",
                "teacher_velocity_rms",
                "mean_phase_slope",
                "mean_phase_reliability",
            )
        }
        for micro_step, (target, phase_target) in enumerate(
            zip(micro_targets, phase_target_chunks, strict=True)
        ):
            target_u, target_s = model.split_us(target)
            masked, _ = make_velocity_mask(target, args.velocity_mask_rate)
            masked_u, masked_s = model.split_us(masked)
            teacher_target = None
            if teacher is not None:
                teacher_target = v1_teacher_spliced_velocity(
                    teacher, target, chunk_size=args.teacher_chunk_size
                )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    state, _ = model.encode_state(masked_s)
                tangent = model.velocity_encoder(masked_u, masked_s, state)
                velocity = model.decode_jvp(state, tangent)
                direction_loss, phase_cosine = phase_direction_loss(
                    velocity, phase_target
                )
                teacher_loss = velocity.new_zeros(())
                teacher_cosine = velocity.new_zeros(())
                teacher_phase_cosine = velocity.new_zeros(())
                if teacher_target is not None:
                    teacher_loss, teacher_cosine = phase_direction_loss(
                        velocity, teacher_target
                    )
                    _, teacher_phase_cosine = phase_direction_loss(
                        teacher_target, phase_target
                    )
                collapse_loss = velocity_anti_collapse_loss(
                    velocity, minimum_rms=args.minimum_velocity_rms
                )
                if args.direction_target == "phase":
                    directional_objective = args.phase_weight * direction_loss
                elif args.direction_target == "teacher":
                    directional_objective = args.teacher_weight * teacher_loss
                elif args.direction_target == "hybrid":
                    directional_objective = (
                        args.phase_weight * direction_loss
                        + args.teacher_weight * teacher_loss
                    )
                else:
                    raise AssertionError(args.direction_target)
                loss = directional_objective + args.collapse_weight * collapse_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite velocity loss at step={step}, micro_step={micro_step + 1}"
                )
            (loss / args.grad_accum_steps).backward()
            cells_seen += int(target.shape[0])
            metric_values["loss"].append(float(loss.detach()))
            metric_values["direction_loss"].append(float(direction_loss.detach()))
            metric_values["phase_cosine"].append(float(phase_cosine))
            metric_values["teacher_loss"].append(float(teacher_loss.detach()))
            metric_values["teacher_cosine"].append(float(teacher_cosine))
            metric_values["teacher_phase_cosine"].append(
                float(teacher_phase_cosine)
            )
            metric_values["collapse_loss"].append(float(collapse_loss.detach()))
            metric_values["velocity_rms"].append(
                float(velocity.float().square().mean().sqrt().detach())
            )
            metric_values["teacher_velocity_rms"].append(
                0.0
                if teacher_target is None
                else float(teacher_target.square().mean().sqrt())
            )
            metric_values["mean_phase_slope"].append(float(slope.mean()))
            metric_values["mean_phase_reliability"].append(
                float(reliability.mean())
            )
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step()
        averaged_metrics = {
            key: float(np.mean(values)) for key, values in metric_values.items()
        }
        value = averaged_metrics["loss"]
        ema = value if ema is None else 0.98 * ema + 0.02 * value
        if step == 1 or step % args.log_every == 0 or step == args.velocity_steps:
            final = {
                "stage": "velocity",
                "step": step,
                "cells_seen": cells_seen,
                "loss": value,
                "loss_ema": ema,
                "direction_loss": averaged_metrics["direction_loss"],
                "phase_cosine": averaged_metrics["phase_cosine"],
                "teacher_loss": averaged_metrics["teacher_loss"],
                "teacher_cosine": averaged_metrics["teacher_cosine"],
                "teacher_phase_cosine": averaged_metrics["teacher_phase_cosine"],
                "collapse_loss": averaged_metrics["collapse_loss"],
                "velocity_rms": averaged_metrics["velocity_rms"],
                "teacher_velocity_rms": averaged_metrics["teacher_velocity_rms"],
                "mean_phase_slope": averaged_metrics["mean_phase_slope"],
                "mean_phase_reliability": averaged_metrics[
                    "mean_phase_reliability"
                ],
                "grad_accum_steps": args.grad_accum_steps,
                "phase_target_batch_size": int(effective_target.shape[0]),
                "grad_norm": float(torch.as_tensor(grad_norm)),
                "elapsed_seconds": time.monotonic() - start_time,
                "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            }
            append_jsonl(metrics_path, final)
            print("V2_METRIC", json.dumps(final, ensure_ascii=False), flush=True)
    return cells_seen, final


def spec_from_args(args) -> V2ModelSpec:
    return V2ModelSpec(
        encoder_type=args.encoder_type,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        num_inducing=args.num_inducing,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        decoder_hidden=tuple(args.decoder_hidden),
    )


def run_pretrain(args) -> Path:
    set_all_seeds(args.seed)
    torch.set_float32_matmul_precision("high")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be positive")
    effective_batch_size = args.batch_size * args.grad_accum_steps
    stream_batch_size = args.stream_batch_size or args.batch_size
    if args.stream_batch_size is not None and stream_batch_size != effective_batch_size:
        raise ValueError(
            "--stream-batch-size must equal --batch-size * --grad-accum-steps"
        )
    device = torch.device("cuda:0")
    spec = spec_from_args(args)
    model = build_model(3000, spec, device)
    all_parameters = count_parameters(model)

    paths = load_paths(args.pretrain_root, "train")
    stream = H5ADBatchStream(
        paths,
        batch_size=stream_batch_size,
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
    iterator = iter(loader)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "pretrain_metrics.jsonl"
    metadata = {
        "seed": args.seed,
        "physical_gpu": args.physical_gpu,
        "pretrain_root": str(args.pretrain_root),
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": effective_batch_size,
        "stream_batch_size": stream_batch_size,
        "phase_target_batch_size": effective_batch_size,
        "state_steps": args.state_steps,
        "velocity_steps": args.velocity_steps,
        "model_spec": asdict(spec),
        "parameters": all_parameters,
        "started_at_unix": time.time(),
    }
    json_dump(args.output_dir / "pretrain_config.json", metadata | vars_for_json(args))
    print("V2_CONFIG", json.dumps(metadata, ensure_ascii=False), flush=True)

    start_time = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    state_cells, state_final = train_state_stage(
        model, iterator, args, metrics_path, device, start_time
    )
    state_path = args.output_dir / "state_final.pt"
    save_checkpoint(
        state_path,
        model,
        spec,
        metadata | {"stage": "state", "state_cells_seen": state_cells},
    )

    teacher = None
    teacher_metadata = None
    if args.direction_target in {"teacher", "hybrid"}:
        if args.teacher_checkpoint is None:
            raise ValueError(
                f"--teacher-checkpoint is required for direction-target={args.direction_target}"
            )
        teacher, teacher_metadata = load_v1_teacher(
            args.teacher_checkpoint,
            num_genes=3000,
            seed=args.seed,
            device=device,
        )

    velocity_cells, velocity_final = train_velocity_stage(
        model, iterator, args, metrics_path, device, start_time, teacher=teacher
    )
    final_path = args.output_dir / "pretrain_final.pt"
    summary = metadata | {
        "stage": "complete",
        "state_cells_seen": state_cells,
        "velocity_cells_seen": velocity_cells,
        "state_final": state_final,
        "velocity_final": velocity_final,
        "direction_target": args.direction_target,
        "teacher_checkpoint": (
            None if args.teacher_checkpoint is None else str(args.teacher_checkpoint)
        ),
        "teacher_metadata": teacher_metadata,
        "elapsed_seconds": time.monotonic() - start_time,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "final_checkpoint": str(final_path),
        "finished_at_unix": time.time(),
    }
    save_checkpoint(final_path, model, spec, summary)
    json_dump(args.output_dir / "pretrain_summary.json", summary)
    print("V2_PRETRAIN_DONE", json.dumps(summary, ensure_ascii=False), flush=True)
    del model, iterator, loader, teacher
    gc.collect()
    torch.cuda.empty_cache()
    return final_path


def load_v2_checkpoint(
    checkpoint: Path, num_genes: int, device: torch.device
) -> tuple[VelocityFoundationV2, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    spec = V2ModelSpec(**payload["model_spec"])
    model = build_model(num_genes, spec, device)
    incompatible = model.load_state_dict(payload["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise FloatingPointError("Checkpoint contains non-finite tensors")
    return model, payload


def run_zero_shot(args) -> dict:
    set_all_seeds(args.seed)
    device = torch.device("cuda:0")
    adata, dataset, _, _, cluster_key = prepare_mousebrain(args.mousebrain, args.seed)
    model, checkpoint_payload = load_v2_checkpoint(args.checkpoint, adata.n_vars, device)
    metrics, evaluated = evaluate_mousebrain(
        model,
        adata,
        dataset,
        cluster_key,
        batch_size=args.eval_batch_size,
        device=str(device),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluated.write_h5ad(args.output_dir / "mousebrain_zero_shot.h5ad", compression="gzip")
    result = {
        "checkpoint": str(args.checkpoint),
        "mousebrain": str(args.mousebrain),
        "mousebrain_shape": [int(adata.n_obs), int(adata.n_vars)],
        "seed": args.seed,
        "physical_gpu": args.physical_gpu,
        "model_spec": checkpoint_payload["model_spec"],
        "checkpoint_stage": checkpoint_payload.get("stage"),
        "zero_shot": metrics,
    }
    json_dump(args.output_dir / "zero_shot_summary.json", result)
    print("V2_ZERO_SHOT_DONE", json.dumps(result, ensure_ascii=False), flush=True)
    return result


def run_phase_baseline(args) -> dict:
    """Evaluate the shared phase target itself before asking a network to learn it."""
    import scvelo as scv

    set_all_seeds(args.seed)
    adata, dataset, _, _, cluster_key = prepare_mousebrain(args.mousebrain, args.seed)
    unspliced = torch.from_numpy(dataset.unspliced)
    spliced = torch.from_numpy(dataset.spliced)
    velocity, slope, reliability = shared_phase_velocity(unspliced, spliced)
    if not torch.isfinite(velocity).all():
        raise FloatingPointError("Phase baseline contains non-finite velocity")
    adata.layers["velocity"] = velocity.numpy()
    scv.tl.velocity_graph(adata, vkey="velocity")
    scv.tl.velocity_embedding(adata, basis="umap", vkey="velocity")
    scv.tl.velocity_confidence(adata, vkey="velocity")
    confidence = float(adata.obs["velocity_confidence"].mean())
    iccoh = float(evaluate_ICCoh(adata, cluster_key=cluster_key, velocity_key="velocity"))
    edges = all_edges.get("MouseBrain.h5ad")
    raw_cbdir = float("nan")
    graph_cbdir = float("nan")
    if edges is not None:
        _, raw_cbdir = evaluate_CBDir2(
            adata,
            cluster_key=cluster_key,
            velocity_key="velocity",
            cluster_edges=edges,
        )
        adata = compute_velocity_from_graph(adata, new_key="velocity_graph_umap")
        _, graph_cbdir = evaluate_CBDir2(
            adata,
            cluster_key=cluster_key,
            velocity_key="velocity_graph",
            cluster_edges=edges,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output_dir / "mousebrain_phase_baseline.h5ad", compression="gzip")
    result = {
        "mousebrain": str(args.mousebrain),
        "mousebrain_shape": [int(adata.n_obs), int(adata.n_vars)],
        "seed": args.seed,
        "mean_phase_slope": float(slope.mean()),
        "mean_phase_reliability": float(reliability.mean()),
        "velocity_rms": float(velocity.square().mean().sqrt()),
        "metrics": {
            "velocity_confidence": confidence,
            "iccoh": iccoh,
            "cbdir_without_graph": float(raw_cbdir),
            "cbdir_with_graph": float(graph_cbdir),
            "graph_improvement": float(graph_cbdir - raw_cbdir),
        },
    }
    json_dump(args.output_dir / "phase_baseline_summary.json", result)
    print("V2_PHASE_BASELINE_DONE", json.dumps(result, ensure_ascii=False), flush=True)
    return result


def vars_for_json(args) -> dict:
    payload = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return {"arguments": payload}


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--encoder-type", choices=("set", "standard"), required=True)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-inducing", type=int, default=128)
    parser.add_argument("--ff-mult", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--decoder-hidden", type=int, nargs="+", default=(768, 384))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    pretrain = subparsers.add_parser("pretrain")
    add_model_arguments(pretrain)
    pretrain.add_argument("--pretrain-root", type=Path, default=DEFAULT_PRETRAIN_ROOT)
    pretrain.add_argument("--output-dir", type=Path, required=True)
    pretrain.add_argument("--physical-gpu", type=int, required=True)
    pretrain.add_argument("--seed", type=int, default=0)
    pretrain.add_argument("--batch-size", type=int, required=True)
    pretrain.add_argument("--grad-accum-steps", type=int, default=1)
    pretrain.add_argument("--stream-batch-size", type=int)
    pretrain.add_argument("--workers", type=int, default=2)
    pretrain.add_argument("--cells-per-dataset-cap", type=int, default=20_000)
    pretrain.add_argument("--state-steps", type=int, required=True)
    pretrain.add_argument("--velocity-steps", type=int, required=True)
    pretrain.add_argument("--state-lr", type=float, default=2.0e-4)
    pretrain.add_argument("--velocity-lr", type=float, default=2.0e-4)
    pretrain.add_argument("--weight-decay", type=float, default=0.01)
    pretrain.add_argument("--state-mask-rate", type=float, default=0.30)
    pretrain.add_argument("--velocity-mask-rate", type=float, default=0.10)
    pretrain.add_argument("--visible-weight", type=float, default=0.10)
    pretrain.add_argument("--phase-weight", type=float, default=1.0)
    pretrain.add_argument(
        "--direction-target",
        choices=("phase", "teacher", "hybrid"),
        default="phase",
    )
    pretrain.add_argument("--teacher-checkpoint", type=Path)
    pretrain.add_argument("--teacher-weight", type=float, default=1.0)
    pretrain.add_argument("--teacher-chunk-size", type=int, default=32)
    pretrain.add_argument("--collapse-weight", type=float, default=0.01)
    pretrain.add_argument("--minimum-velocity-rms", type=float, default=1.0e-3)
    pretrain.add_argument("--grad-clip", type=float, default=1.0)
    pretrain.add_argument("--log-every", type=int, default=25)

    zero_shot = subparsers.add_parser("zero-shot")
    zero_shot.add_argument("--checkpoint", type=Path, required=True)
    zero_shot.add_argument("--mousebrain", type=Path, default=DEFAULT_MOUSEBRAIN)
    zero_shot.add_argument("--output-dir", type=Path, required=True)
    zero_shot.add_argument("--physical-gpu", type=int, required=True)
    zero_shot.add_argument("--eval-batch-size", type=int, default=32)
    zero_shot.add_argument("--seed", type=int, default=0)

    phase_baseline = subparsers.add_parser("phase-baseline")
    phase_baseline.add_argument("--mousebrain", type=Path, default=DEFAULT_MOUSEBRAIN)
    phase_baseline.add_argument("--output-dir", type=Path, required=True)
    phase_baseline.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.command == "pretrain":
        run_pretrain(args)
    elif args.command == "zero-shot":
        run_zero_shot(args)
    elif args.command == "phase-baseline":
        run_phase_baseline(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
