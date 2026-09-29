import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from gfg3_data import (
    BlockStream,
    Split,
    _read_layers,
    audit_vocab,
    file_vocabulary,
    fit_vocab_stats,
)
from gfg3_moments import build_one
from gfg3_train import GraphCache


def write_h5ad(path, vocab, spliced, unspliced):
    def csr(x):
        indices, data, indptr = [], [], [0]
        for row in np.asarray(x):
            nz = np.flatnonzero(row)
            indices.extend(nz.tolist())
            data.extend(np.asarray(row[nz], dtype=np.float32).tolist())
            indptr.append(len(indices))
        return (np.asarray(data, np.float32), np.asarray(indices, np.int64),
                np.asarray(indptr, np.int64))

    with h5py.File(path, "w") as f:
        var = f.create_group("var")
        var.create_dataset("_index", data=np.asarray(vocab, dtype="S"))
        layers = f.create_group("layers")
        for name, values in (("spliced", spliced), ("unspliced", unspliced)):
            group = layers.create_group(name)
            data, indices, indptr = csr(values)
            group.create_dataset("data", data=data)
            group.create_dataset("indices", data=indices)
            group.create_dataset("indptr", data=indptr)


class GFG3DataContractTests(unittest.TestCase):
    def test_explicit_column_map_and_vocab_audit(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "a.h5ad"
            p2 = Path(td) / "b.h5ad"
            # File b is reordered and has an extra gene.  Its source columns
            # are [c,a,d], while the target union is [a,b,c,d].
            write_h5ad(p1, ["a", "b", "c"], [[1, 2, 0]], [[3, 0, 4]])
            write_h5ad(p2, ["c", "a", "d"], [[10, 20, 30]], [[1, 2, 3]])
            report = audit_vocab([str(p1), str(p2)])
            self.assertEqual(report["n_variants"], 2)
            self.assertEqual(report["mismatched"], [str(p2)])
            with h5py.File(p2, "r") as f:
                source = file_vocabulary(str(p2))
                union = ["a", "b", "c", "d"]
                mapping = np.asarray([union.index(g) for g in source], dtype=np.int64)
                u, s = _read_layers(f, 0, 1, len(union), column_map=mapping)
            # _read_layers returns log1p(CP10K) in (u,s) order.
            expected_s = np.log1p(np.asarray([[20, 0, 10, 30]], np.float32) /
                                  60.0 * 1e4)
            expected_u = np.log1p(np.asarray([[2, 0, 1, 3]], np.float32) /
                                  6.0 * 1e4)
            np.testing.assert_allclose(s, expected_s, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(u, expected_u, rtol=1e-6, atol=1e-6)

    def test_stats_and_block_stream_cover_reordered_files_and_tail(self):
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i, vocab in enumerate((["a", "b", "c"], ["c", "a", "d"])):
                path = Path(td) / f"{i}.h5ad"
                sp = np.array([[1 + i, 2, 0], [0, 3 + i, 4], [5, 0, 1]], np.float32)
                us = np.array([[2, 0, 1], [1, 1, 0], [0, 2, 3]], np.float32)
                write_h5ad(path, vocab, sp, us)
                paths.append(str(path))
            names, _, u_mu, u_sd, s_mu, s_sd, fh = fit_vocab_stats(paths, 4, block=2)
            self.assertEqual(set(names.tolist()), {"a", "b", "c", "d"})
            split = Split("train", paths, names, u_mu, u_sd, s_mu, s_sd, fh)
            stream = BlockStream(split, block=2, shuffle=False)
            self.assertEqual(len(stream), 4)  # 2 blocks per 3-cell file
            batches = list(stream)
            self.assertEqual([int(x["u"].shape[0]) for x in batches], [2, 1, 2, 1])
            self.assertTrue(all(x["u"].shape[1] == 4 for x in batches))

    def test_rank_shards_are_disjoint_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.h5ad"
            write_h5ad(p, ["a", "b"], np.ones((7, 2)), np.ones((7, 2)))
            names, _, u_mu, u_sd, s_mu, s_sd, fh = fit_vocab_stats([str(p)], 2, block=2)
            split = Split("train", [str(p)], names, u_mu, u_sd, s_mu, s_sd, fh)
            r0 = BlockStream(split, block=2, shuffle=False, rank=0, world_size=2)
            r1 = BlockStream(split, block=2, shuffle=False, rank=1, world_size=2)
            self.assertEqual(len(r0), len(r1))
            self.assertEqual(sum(int(x["u"].shape[0]) for x in r0), 4)
            self.assertEqual(sum(int(x["u"].shape[0]) for x in r1), 3)


class GFG3GraphAndMomentsTests(unittest.TestCase):
    def test_graph_cache_skips_single_cell_and_handles_two_cells(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "one.h5ad"
            p2 = Path(td) / "two.h5ad"
            write_h5ad(p1, ["a", "b"], [[1, 0]], [[0, 1]])
            write_h5ad(p2, ["a", "b"], [[1, 0], [0, 2]], [[0, 1], [1, 0]])
            cache = GraphCache([str(p1), str(p2)], np.asarray(["a", "b"]),
                               np.zeros(2), np.ones(2), np.zeros(2), np.ones(2),
                               k=3, min_cells=1, max_cells_per_file=8)
            self.assertEqual(len(cache), 2)
            a, nb = cache.sample(4, np.random.default_rng(0))
            self.assertEqual(tuple(a.shape), (4, 4))
            self.assertEqual(tuple(nb.shape), (4, 3, 4))

    def test_moments_keep_u_and_s_orientation(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "mom.h5ad"
            rng = np.random.default_rng(4)
            sp = rng.poisson(3, size=(50, 5)).astype(np.float32)
            us = rng.poisson(2, size=(50, 5)).astype(np.float32)
            write_h5ad(p, ["a", "b", "c", "d", "e"], sp, us)
            sid, status = build_one((str(p), td))
            self.assertEqual(status, "ok")
            out_s = np.load(Path(td) / f"{sid}.s.npy")
            out_u = np.load(Path(td) / f"{sid}.u.npy")
            self.assertEqual(out_s.shape, out_u.shape)
            # The two layers have distinct means; an accidental swap is
            # visible without relying on a particular neighbour ordering.
            self.assertNotAlmostEqual(float(out_s.mean()), float(out_u.mean()), places=3)
            self.assertGreater(float(out_s.mean()), float(out_u.mean()))


if __name__ == "__main__":
    unittest.main()
