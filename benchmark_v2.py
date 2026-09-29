#!/usr/bin/env python3
"""Uniform zero-shot evaluation of a GFG-v2 checkpoint on an external dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader

from const.cluster_edges import all_edges
from foundation_pilot import (
    NormalizedMouseBrainData,
    compute_velocity_from_graph,
    json_dump,
    set_all_seeds,
)
from foundation_v2 import load_v2_checkpoint
from preprocessing import build_neighbor_indices
from tool.evaluate import evaluate_CBDir2
from tool.utils import find_cluster_key


def fast_iccoh(adata, cluster_key: str, velocity_key: str = "velocity") -> float:
    """Exact mean pairwise cosine via the norm-of-sum identity."""

    velocity = np.asarray(adata.layers[velocity_key], dtype=np.float64)
    valid = np.isfinite(velocity).all(axis=1)
    velocity = velocity[valid]
    labels = np.asarray(adata.obs[cluster_key].astype(str))[valid]
    scores = []
    weights = []
    for label in np.unique(labels):
        values = velocity[labels == label]
        norms = np.linalg.norm(values, axis=1)
        values = values[norms > 1.0e-9]
        norms = norms[norms > 1.0e-9]
        count = len(values)
        if count < 2:
            continue
        unit = values / norms[:, None]
        summed = unit.sum(axis=0)
        pairwise_mean = (float(summed @ summed) - count) / (count * (count - 1))
        scores.append(pairwise_mean)
        weights.append(count)
    return 0.0 if not scores else float(np.average(scores, weights=weights))


def predict_velocity(model, dataset, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    for batch in loader:
        inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
        _, velocity = model.predict(inputs, device=str(device))
        predictions.append(velocity[:, model.G :].numpy())
    return np.concatenate(predictions, axis=0)


def predict_velocity_with_backoff(
    model, dataset, batch_size: int, device: torch.device
) -> tuple[np.ndarray, int, list[int], float]:
    """Run inference and halve the cell batch only after a CUDA OOM."""

    current = max(1, int(batch_size))
    attempted = []
    while True:
        attempted.append(current)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        try:
            velocity = predict_velocity(model, dataset, current, device)
            peak_gib = (
                float(torch.cuda.max_memory_allocated(device)) / (1024**3)
                if device.type == "cuda"
                else 0.0
            )
            return velocity, current, attempted, peak_gib
        except torch.OutOfMemoryError:
            if current == 1:
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            current = max(1, current // 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--dataset-key", choices=tuple(all_edges), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-h5ad", action="store_true")
    args = parser.parse_args()

    set_all_seeds(args.seed)
    device = torch.device("cuda:0")
    start = time.monotonic()
    adata = ad.read_h5ad(args.dataset_cache)
    cluster_key, _ = find_cluster_key(adata)
    if cluster_key is None:
        raise ValueError(f"No cluster label in {args.dataset_cache}")
    dataset = NormalizedMouseBrainData(adata)
    model, checkpoint = load_v2_checkpoint(args.checkpoint, adata.n_vars, device)
    velocity, effective_batch_size, batch_attempts, peak_inference_memory_gib = (
        predict_velocity_with_backoff(model, dataset, args.batch_size, device)
    )
    adata.layers["velocity"] = velocity

    import scvelo as scv

    scv.tl.velocity_graph(adata, vkey="velocity")
    scv.tl.velocity_embedding(adata, basis="umap", vkey="velocity")
    scv.tl.velocity_confidence(adata, vkey="velocity")
    # Match the corrected historical GFG metric: real graph neighbors, padded
    # with -1 rather than accidental dense row offsets.
    adata = build_neighbor_indices(adata, n_neighbors=30)
    confidence = float(adata.obs["velocity_confidence"].mean())
    iccoh = fast_iccoh(adata, cluster_key)
    edges = all_edges[args.dataset_key]
    raw_scores, raw_cbdir = evaluate_CBDir2(
        adata,
        cluster_key=cluster_key,
        velocity_key="velocity",
        cluster_edges=edges,
    )
    adata = compute_velocity_from_graph(adata, new_key="velocity_graph_umap")
    graph_scores, graph_cbdir = evaluate_CBDir2(
        adata,
        cluster_key=cluster_key,
        velocity_key="velocity_graph",
        cluster_edges=edges,
    )
    result = {
        "model_id": args.model_id,
        "family": args.family,
        "checkpoint": str(args.checkpoint),
        "dataset_key": args.dataset_key,
        "dataset_cache": str(args.dataset_cache),
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "cluster_key": cluster_key,
        "seed": args.seed,
        "physical_gpu": args.physical_gpu,
        "requested_batch_size": int(args.batch_size),
        "effective_batch_size": int(effective_batch_size),
        "batch_attempts": [int(value) for value in batch_attempts],
        "peak_inference_memory_gib": float(peak_inference_memory_gib),
        "model_spec": checkpoint["model_spec"],
        "metrics": {
            "velocity_confidence": confidence,
            "iccoh": iccoh,
            "cbdir_without_graph": float(raw_cbdir),
            "cbdir_with_graph": float(graph_cbdir),
            "graph_improvement": float(graph_cbdir - raw_cbdir),
        },
        "raw_edge_scores": {f"{u} -> {v}": float(raw_scores[(u, v)]) for u, v in edges},
        "graph_edge_scores": {
            f"{u} -> {v}": float(graph_scores[(u, v)]) for u, v in edges
        },
        "elapsed_seconds": time.monotonic() - start,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_dump(args.output_dir / "benchmark_summary.json", result)
    if args.save_h5ad:
        adata.write_h5ad(args.output_dir / "evaluated.h5ad", compression="lzf")
    print("V2_BENCHMARK_DONE", json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
