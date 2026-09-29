import unittest

import numpy as np
import pandas as pd
import scipy.sparse

from model.graph import GraphBatchSampler, build_root_directed_adjacency
from preprocessing import build_neighbor_indices
from tool.evaluate import evaluate_CBDir2


class MinimalAnnData:
    def __init__(self, distances):
        self.n_obs = distances.shape[0]
        self.obsp = {"distances": scipy.sparse.csr_matrix(distances)}
        self.uns = {"neighbors": {}}


class GraphTrainingTest(unittest.TestCase):
    def test_root_distance_orients_path_away_from_root(self):
        adjacency = scipy.sparse.csr_matrix(
            np.array(
                [
                    [0, 1, 0, 0],
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                ],
                dtype=np.float32,
            )
        )
        directed, root_distance = build_root_directed_adjacency(
            adjacency, np.array([True, False, False, False])
        )

        expected = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(directed.toarray(), expected)
        np.testing.assert_allclose(root_distance, [0, 1 / 3, 2 / 3, 1])

    def test_graph_batches_cover_every_node_once_and_preserve_more_edges(self):
        clique = np.ones((4, 4), dtype=np.float32) - np.eye(4, dtype=np.float32)
        adjacency = scipy.sparse.block_diag((clique, clique), format="csr")
        sampler = GraphBatchSampler(adjacency, batch_size=4, seed=3)
        batches = list(iter(sampler))

        flattened = [node for batch in batches for node in batch]
        self.assertEqual(sorted(flattened), list(range(8)))
        self.assertEqual(len(flattened), len(set(flattened)))

        graph_edges = sum(adjacency[batch][:, batch].nnz for batch in batches)
        mixed_batches = [[0, 1, 4, 5], [2, 3, 6, 7]]
        mixed_edges = sum(adjacency[batch][:, batch].nnz for batch in mixed_batches)
        self.assertGreater(graph_edges, mixed_edges)

    def test_sparse_neighbor_indices_exclude_missing_zero_entries_and_pad(self):
        distances = np.array(
            [
                [0.0, 0.1, 0.2, 0.0],
                [0.1, 0.0, 0.3, 0.0],
                [0.2, 0.3, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        adata = MinimalAnnData(distances)
        build_neighbor_indices(adata, n_neighbors=3)

        np.testing.assert_array_equal(
            adata.uns["neighbors"]["indices"][0], np.array([1, 2, -1])
        )
        np.testing.assert_array_equal(
            adata.uns["neighbors"]["indices"][3], np.array([-1, -1, -1])
        )

    def test_cbdir_can_read_graph_specific_umap_velocity(self):
        class EvaluationAnnData:
            pass

        adata = EvaluationAnnData()
        adata.obs = pd.DataFrame({"celltype": ["source", "target"]})
        adata.obsm = {
            "X_umap": np.array([[0.0, 0.0], [1.0, 0.0]]),
            "velocity_umap": np.array([[-1.0, 0.0], [0.0, 0.0]]),
            "velocity_graph_umap": np.array([[1.0, 0.0], [0.0, 0.0]]),
        }
        adata.uns = {
            "neighbors": {"indices": np.array([[1, -1], [0, -1]])}
        }

        _, score = evaluate_CBDir2(
            adata,
            cluster_key="celltype",
            velocity_key="velocity_graph",
            cluster_edges=[("source", "target")],
        )
        self.assertAlmostEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
