import math
from collections import deque

import numpy as np
import scipy.sparse
import torch
from scipy.sparse.csgraph import dijkstra
from torch.utils.data import Sampler


def as_csr_adjacency(adjacency_matrix):
    """Convert a square adjacency matrix to CSR without changing node order."""
    if torch.is_tensor(adjacency_matrix):
        adjacency_matrix = adjacency_matrix.detach().cpu().numpy()

    adjacency = scipy.sparse.csr_matrix(adjacency_matrix, dtype=np.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency_matrix must be square, got shape={adjacency.shape}")

    adjacency = adjacency.copy()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency


def build_root_directed_adjacency(adjacency_matrix, root_mask):
    """Orient undirected graph edges from smaller to larger root distance."""
    adjacency = as_csr_adjacency(adjacency_matrix)
    root_mask = np.asarray(root_mask, dtype=bool)

    if root_mask.shape != (adjacency.shape[0],):
        raise ValueError(
            f"root_mask must have shape {(adjacency.shape[0],)}, got {root_mask.shape}"
        )

    root_indices = np.flatnonzero(root_mask)
    if root_indices.size == 0:
        raise ValueError("root_mask must contain at least one root cell")

    costs = adjacency.astype(np.float64, copy=True)
    costs.data = 1.0 / np.maximum(costs.data, 1e-12)
    distances = dijkstra(costs, directed=False, indices=root_indices)
    if distances.ndim == 1:
        distances = distances[None, :]
    root_distance = np.min(distances, axis=0)

    coo = adjacency.tocoo()
    keep = (
        np.isfinite(root_distance[coo.row])
        & np.isfinite(root_distance[coo.col])
        & (root_distance[coo.col] > root_distance[coo.row] + 1e-12)
    )
    directed = scipy.sparse.csr_matrix(
        (coo.data[keep], (coo.row[keep], coo.col[keep])),
        shape=adjacency.shape,
        dtype=np.float32,
    )
    directed.eliminate_zeros()

    finite = np.isfinite(root_distance)
    normalized_distance = np.full_like(root_distance, np.nan, dtype=np.float64)
    if finite.any():
        finite_distance = root_distance[finite]
        distance_range = finite_distance.max() - finite_distance.min()
        if distance_range > 0:
            normalized_distance[finite] = (
                finite_distance - finite_distance.min()
            ) / distance_range
        else:
            normalized_distance[finite] = 0.0

    return directed, normalized_distance


class GraphBatchSampler(Sampler):
    """Partition cells into reproducible graph-local batches once per epoch."""

    def __init__(self, adjacency_matrix, batch_size, seed=0):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.adjacency = as_csr_adjacency(adjacency_matrix)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return math.ceil(self.adjacency.shape[0] / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1

        num_nodes = self.adjacency.shape[0]
        remaining = np.ones(num_nodes, dtype=bool)
        start_order = rng.permutation(num_nodes)
        start_cursor = 0
        num_remaining = num_nodes

        while num_remaining:
            batch = []
            queued = np.zeros(num_nodes, dtype=bool)
            frontier = deque()

            while len(batch) < self.batch_size and num_remaining:
                while not frontier:
                    while (
                        start_cursor < num_nodes
                        and not remaining[start_order[start_cursor]]
                    ):
                        start_cursor += 1

                    if start_cursor < num_nodes:
                        start = int(start_order[start_cursor])
                        start_cursor += 1
                    else:
                        start = int(rng.choice(np.flatnonzero(remaining)))

                    frontier.append(start)
                    queued[start] = True

                node = frontier.popleft()
                if not remaining[node]:
                    continue

                remaining[node] = False
                num_remaining -= 1
                batch.append(node)

                row_start = self.adjacency.indptr[node]
                row_end = self.adjacency.indptr[node + 1]
                neighbors = self.adjacency.indices[row_start:row_end].copy()
                if neighbors.size:
                    neighbors = neighbors[
                        remaining[neighbors] & ~queued[neighbors]
                    ]
                    rng.shuffle(neighbors)
                    for neighbor in neighbors:
                        neighbor = int(neighbor)
                        queued[neighbor] = True
                        frontier.append(neighbor)

            rng.shuffle(batch)
            yield batch
