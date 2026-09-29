import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from foundation_v2 import (
    V2ModelSpec,
    VelocityFoundationV2,
    load_micro_targets,
    phase_direction_loss,
    shared_phase_velocity,
)
from foundation_v2_ddp import (
    VelocityStageModule,
    augment_inputs,
    gene_phase_direction_loss,
)
from foundation_pilot import H5ADBatchStream, log_cpm


def small_spec(encoder_type="set"):
    return V2ModelSpec(
        encoder_type=encoder_type,
        d_model=32,
        num_heads=4,
        num_blocks=1,
        num_inducing=4,
        ff_mult=2,
        dropout=0.0,
        decoder_hidden=(48, 32),
    )


class FoundationV2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_state_branch_uses_only_spliced(self):
        model = VelocityFoundationV2(6, small_spec()).eval()
        spliced = torch.rand(3, 6)
        _, first = model.encode_state(spliced)
        _, second = model.encode_state(spliced.clone())
        torch.testing.assert_close(first, second)

    def test_set_encoder_is_gene_permutation_equivariant(self):
        model = VelocityFoundationV2(7, small_spec()).eval()
        inputs = torch.rand(2, 14)
        permutation = torch.tensor([3, 0, 6, 2, 5, 1, 4])
        u, s = inputs.chunk(2, dim=1)
        permuted = torch.cat((u[:, permutation], s[:, permutation]), dim=1)
        with torch.no_grad():
            original_state, original_recon = model.encode_state(s)
            permuted_state, permuted_recon = model.encode_state(s[:, permutation])
        torch.testing.assert_close(
            permuted_state, original_state[:, permutation], rtol=2e-5, atol=2e-6
        )
        torch.testing.assert_close(
            permuted_recon, original_recon[:, permutation], rtol=2e-5, atol=2e-6
        )

    def test_decoder_jvp_matches_finite_difference(self):
        model = VelocityFoundationV2(5, small_spec()).eval()
        state = torch.randn(2, 5, 32)
        tangent = torch.randn_like(state) * 0.1
        jvp = model.decode_jvp(state, tangent)
        epsilon = 1.0e-3
        finite = (
            model.state_decoder(state + epsilon * tangent)
            - model.state_decoder(state - epsilon * tangent)
        ) / (2 * epsilon)
        torch.testing.assert_close(jvp, finite, rtol=2e-2, atol=2e-3)

    def test_predict_matches_forward_without_retained_graph(self):
        model = VelocityFoundationV2(5, small_spec()).eval()
        inputs = torch.rand(3, 10)
        with torch.enable_grad():
            expected_reconstruction, expected_velocity, _, _ = model(inputs)
        reconstruction, velocity = model.predict(inputs)
        torch.testing.assert_close(reconstruction[:, 5:], expected_reconstruction)
        torch.testing.assert_close(velocity[:, 5:], expected_velocity)
        self.assertFalse(reconstruction.requires_grad)
        self.assertFalse(velocity.requires_grad)

    def test_shared_phase_target_is_finite_and_centered(self):
        u = torch.rand(32, 11)
        s = torch.rand(32, 11)
        target, slope, reliability = shared_phase_velocity(u, s)
        self.assertTrue(torch.isfinite(target).all())
        self.assertTrue(torch.isfinite(slope).all())
        self.assertTrue(torch.isfinite(reliability).all())
        torch.testing.assert_close(target.mean(0), torch.zeros(11), atol=2e-6, rtol=0)

    def test_velocity_stage_backpropagates_only_to_velocity_encoder(self):
        model = VelocityFoundationV2(8, small_spec())
        parameters = model.set_training_stage("velocity")
        inputs = torch.rand(4, 16)
        u, s = inputs.chunk(2, dim=1)
        with torch.no_grad():
            state, _ = model.encode_state(s)
            target, _, _ = shared_phase_velocity(u, s)
        tangent = model.velocity_encoder(u, s, state)
        velocity = model.decode_jvp(state, tangent)
        loss, _ = phase_direction_loss(velocity, target)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in parameters))
        self.assertTrue(
            all(parameter.grad is None for parameter in model.state_encoder.parameters())
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.state_decoder.parameters())
        )

    def test_standard_attention_forward_shape(self):
        model = VelocityFoundationV2(9, small_spec("standard")).eval()
        inputs = torch.rand(2, 18)
        spliced_hat, velocity, state, tangent = model(inputs)
        self.assertEqual(spliced_hat.shape, (2, 9))
        self.assertEqual(velocity.shape, (2, 9))
        self.assertEqual(state.shape, (2, 9, 32))
        self.assertEqual(tangent.shape, (2, 9, 32))

    def test_stream_batch_is_split_without_mixing_iterator_items(self):
        full_batch = torch.arange(48, dtype=torch.float32).reshape(6, 8)
        args = SimpleNamespace(
            batch_size=2,
            grad_accum_steps=3,
            stream_batch_size=6,
        )
        chunks = load_micro_targets(iter([full_batch]), args, torch.device("cpu"))
        self.assertEqual([chunk.shape[0] for chunk in chunks], [2, 2, 2])
        torch.testing.assert_close(torch.cat(chunks), full_batch)

    def test_legacy_accumulation_consumes_one_iterator_item_per_micro_batch(self):
        batches = [torch.full((2, 8), float(index)) for index in range(3)]
        args = SimpleNamespace(
            batch_size=2,
            grad_accum_steps=3,
            stream_batch_size=None,
        )
        chunks = load_micro_targets(iter(batches), args, torch.device("cpu"))
        for observed, expected in zip(chunks, batches, strict=True):
            torch.testing.assert_close(observed, expected)

    def test_hidden_augmentation_preserves_shape_and_nonnegativity(self):
        inputs = torch.rand(16, 20)
        inputs[:, ::5] = 0.0
        torch.manual_seed(11)
        augmented = augment_inputs(inputs, gene_dropout_rate=0.25, noise_std=0.03)
        self.assertEqual(augmented.shape, inputs.shape)
        self.assertTrue(torch.isfinite(augmented).all())
        self.assertTrue((augmented >= 0).all())
        self.assertTrue((augmented[inputs == 0] == 0).all())
        self.assertFalse(torch.equal(augmented, inputs))

    def test_log_cpm_can_preserve_full_transcriptome_library(self):
        subset = np.array([[1.0]], dtype=np.float32)
        normalized = log_cpm(subset, library_size=np.array([[10.0]]))
        np.testing.assert_allclose(normalized, np.log1p([[1000.0]]), rtol=1e-6)

    def test_stream_uniformly_subsamples_same_gene_axis_for_u_and_s(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.h5ad"
            values = np.arange(1, 25, dtype=np.float32).reshape(4, 6)
            adata = ad.AnnData(
                X=sp.csr_matrix((4, 6)),
                obs={"cell": ["a", "b", "c", "d"]},
            )
            adata.layers["unspliced"] = sp.csr_matrix(values)
            adata.layers["spliced"] = sp.csr_matrix(values * 2)
            adata.var_names = [f"g{i}" for i in range(6)]
            adata.write_h5ad(path)
            stream = H5ADBatchStream(
                [path], batch_size=2, seed=3, genes_per_batch=3
            )
            batch = next(iter(stream))
            self.assertEqual(tuple(batch.shape), (2, 6))
            self.assertTrue(torch.isfinite(batch).all())

    def test_gene_direction_loss_has_expected_orientation(self):
        target = torch.randn(32, 13)
        aligned_loss, aligned_cosine = gene_phase_direction_loss(target, target)
        reversed_loss, reversed_cosine = gene_phase_direction_loss(-target, target)
        torch.testing.assert_close(aligned_loss, torch.zeros(()), atol=2e-6, rtol=0)
        torch.testing.assert_close(aligned_cosine, torch.ones(()), atol=2e-6, rtol=0)
        torch.testing.assert_close(reversed_loss, torch.full((), 2.0), atol=2e-6, rtol=0)
        torch.testing.assert_close(reversed_cosine, -torch.ones(()), atol=2e-6, rtol=0)

    def test_velocity_module_returns_jvp_and_finite_secant(self):
        model = VelocityFoundationV2(6, small_spec())
        model.set_training_stage("velocity")
        module = VelocityStageModule(model, secant_epsilon=0.1)
        jvp, secant = module(torch.rand(4, 12))
        self.assertEqual(jvp.shape, (4, 6))
        self.assertEqual(secant.shape, (4, 6))
        self.assertTrue(torch.isfinite(jvp).all())
        self.assertTrue(torch.isfinite(secant).all())
        (jvp.square().mean() + secant.square().mean()).backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.velocity_encoder.parameters())
        )


if __name__ == "__main__":
    unittest.main()
