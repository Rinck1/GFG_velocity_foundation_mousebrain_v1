import unittest

import numpy as np
import torch

from gfg3_model import build
from gfg3_train import stage_b_losses
from model.Decoder import ode_shared_gene_residual_loss


class GFG3ModelTests(unittest.TestCase):
    def test_state_only_stage_does_not_compute_velocity(self):
        model = build(7, "original", use_vq=True)
        model.train()
        model.set_stage("state")
        out = model(torch.randn(2, 7), torch.randn(2, 7), compute_velocity=False)
        self.assertIsNone(out["z_v_raw"])
        self.assertEqual(float(out["v_x"].abs().sum()), 0.0)
        self.assertFalse(model.code_v.training)
        self.assertFalse(any(p.requires_grad for p in model.vel_tower.parameters()))

    def test_jvp_matches_finite_difference(self):
        torch.manual_seed(5)
        model = build(5, "original", use_vq=False).eval()
        z = torch.randn(2, 5, model.spec.d_model)
        dz = torch.randn_like(z)
        jvp = model.jvp_velocity(z, dz)
        eps = 1e-3
        fd = (model.decode_state(z + eps * dz) - model.decode_state(z - eps * dz)) / (2 * eps)
        torch.testing.assert_close(jvp, fd, rtol=3e-2, atol=2e-4)

    def test_graph_alignment_uses_gene_interleaved_layout(self):
        u = torch.tensor([[1.0, 2.0]])
        s = torch.tensor([[10.0, 20.0]])
        # Delta in gene-interleaved layout: [1,3,2,4].
        velocity = torch.tensor([[[1.0, 3.0], [2.0, 4.0]]])
        out = {
            "rec": torch.stack((u, s), -1), "v_x": velocity,
            "d_s": {"vq": torch.tensor(0.0)}, "d_v": {"vq": torch.tensor(0.0)},
        }
        # Neighbour x is loaded in layer-block layout (B,k,2,G).
        nbr = {
            "x": torch.tensor([[[[2.0, 4.0], [13.0, 24.0]]]]),
            "v_x": velocity.unsqueeze(1),
        }
        terms = stage_b_losses(out, u, s, nbr=nbr)
        self.assertAlmostEqual(float(terms["align"]), 0.0, places=6)
        self.assertAlmostEqual(float(terms["smooth"]), 0.0, places=6)

    def test_shared_gene_kinetics_recover_rates_and_reject_noise(self):
        torch.manual_seed(2)
        u = torch.rand(64, 3) + 0.1
        s = torch.rand(64, 3) + 0.1
        alpha = torch.tensor([1.0, 0.5, 0.2])
        beta = torch.tensor([0.4, 0.2, 0.8])
        gamma = torch.tensor([0.3, 0.7, 0.5])
        vu, vs = alpha - beta * u, beta * u - gamma * s
        clean, aux = ode_shared_gene_residual_loss(u, s, vu, vs, lam=1e-6)
        self.assertLess(float(clean), 1e-10)
        torch.testing.assert_close(aux["alpha"], alpha, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(aux["beta"], beta, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(aux["gamma"], gamma, rtol=1e-4, atol=1e-5)
        noisy, _ = ode_shared_gene_residual_loss(u, s, vu + 0.3 * torch.randn_like(vu), vs)
        self.assertGreater(float(noisy), 1e-3)


if __name__ == "__main__":
    unittest.main()
