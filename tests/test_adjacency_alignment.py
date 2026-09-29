import unittest

import numpy as np
import scipy.sparse
import torch
from torch.utils.data import DataLoader

from model.dataset import Data
from model.model import slice_batch_adjacency


class FakeAnnData:
    def __init__(self, num_cells=8, num_genes=3):
        cell_ids = np.arange(num_cells, dtype=np.float32)[:, None]
        offsets = np.arange(num_genes, dtype=np.float32)[None, :] / 10
        self.layers = {
            "Mu": cell_ids + offsets,
            "Ms": 10 + cell_ids + offsets,
        }
        self.obs = {}
        self._num_cells = num_cells

    def __len__(self):
        return self._num_cells


class BatchAdjacencyAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.num_cells = 8
        values = np.arange(self.num_cells ** 2, dtype=np.float32)
        self.adjacency = values.reshape(self.num_cells, self.num_cells)

    def test_shuffled_batch_indices_match_inputs_and_sparse_adjacency(self):
        dataset = Data(FakeAnnData(self.num_cells), spliced_key="Ms", unspliced_key="Mu")
        generator = torch.Generator().manual_seed(7)
        loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=generator)

        inputs, _, batch_indices = next(iter(loader))
        np.testing.assert_array_equal(inputs[:, 0].numpy(), batch_indices.numpy())

        actual = slice_batch_adjacency(
            scipy.sparse.csr_matrix(self.adjacency), batch_indices, device="cpu"
        )
        indices = batch_indices.numpy()
        expected = self.adjacency[np.ix_(indices, indices)]
        torch.testing.assert_close(actual, torch.from_numpy(expected))

    def test_torch_adjacency_uses_the_same_row_and_column_order(self):
        batch_indices = torch.tensor([6, 1, 7, 2])
        actual = slice_batch_adjacency(
            torch.from_numpy(self.adjacency), batch_indices, device="cpu"
        )
        indices = batch_indices.numpy()
        expected = self.adjacency[np.ix_(indices, indices)]
        torch.testing.assert_close(actual, torch.from_numpy(expected))

    def test_rejects_invalid_shapes_and_indices(self):
        with self.assertRaisesRegex(ValueError, "must be square"):
            slice_batch_adjacency(np.zeros((3, 4)), torch.tensor([0, 1]))

        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            slice_batch_adjacency(self.adjacency, torch.tensor([[0, 1]]))

        with self.assertRaisesRegex(IndexError, "outside"):
            slice_batch_adjacency(self.adjacency, torch.tensor([0, self.num_cells]))


