"""CPU tests for safe all-level counterfactual bilevel invariants."""

import unittest

import torch

from semantic_bilevel import BilevelSemanticController


class BilevelSemanticControllerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.batch = 4
        self.parts = 6
        self.dim = 32
        self.controller = BilevelSemanticController(
            dim=self.dim,
            text_dim=24,
            num_parts=self.parts,
            semantic_rank=8,
            adapter_rank=8,
            policy_hidden_dim=16,
            reference_mix=0.5,
            relation_hvp_samples=1,
        )

    def _state(self, seed, compute_hvp=True):
        return self._state_for(self.controller, seed, compute_hvp=compute_hvp)

    def _state_for(self, controller, seed, compute_hvp=True):
        generator = torch.Generator().manual_seed(seed)
        part_tokens = torch.randn(
            self.batch, self.parts, self.dim, generator=generator,
            requires_grad=True,
        )
        policy_semantics = torch.randn(
            self.batch, self.parts, self.dim, generator=generator,
            requires_grad=True,
        )
        target_semantics = torch.randn(
            self.batch, self.parts, self.dim, generator=generator,
            requires_grad=True,
        )
        part_curvature = torch.rand(
            self.batch, self.parts, generator=generator
        ) + 0.2
        # 4x4 patch grid; positive attention is normalized by the encoder.
        part_attn = torch.rand(
            self.batch, self.parts, 16, generator=generator,
            requires_grad=True,
        )
        state = controller.make_state(
            part_tokens, policy_semantics, target_semantics, part_curvature
        )
        state["relation_state"] = controller.make_relation_state(
            part_tokens,
            policy_semantics,
            target_semantics,
            part_attn,
            part_curvature,
            compute_hvp=compute_hvp,
        )
        return state

    def test_all_level_counterfactual_router_has_identifiable_gradients(self):
        controller = BilevelSemanticController(
            dim=self.dim,
            text_dim=24,
            num_parts=self.parts,
            semantic_rank=8,
            adapter_rank=8,
            policy_hidden_dim=16,
            reference_mix=0.5,
            relation_hvp_samples=1,
            routing_scope="counterfactual",
        )
        support = self._state_for(controller, 21, compute_hvp=False)
        query = self._state_for(controller, 22, compute_hvp=False)
        relation_state = support["relation_state"]
        route, _ = controller.router(
            torch.rand(self.batch, self.parts),
            torch.rand(self.batch, controller.num_relations),
            support["curvature"],
            relation_state["curvature"],
        )
        self.assertEqual(tuple(route.shape), (self.batch, 3, 3))
        expected_prior = controller.router.prior.view(1, 1, 3).expand_as(route)
        self.assertTrue(torch.allclose(route, expected_prior))
        generator = torch.Generator().manual_seed(23)
        targets = torch.randn(
            self.batch, 3, self.dim, generator=generator
        )

        def outer_task_fn(state, params, reference):
            adapted = controller.adapt_parts(
                state["part_tokens"].detach().float(), params
            )
            pooled = (reference.unsqueeze(-1) * adapted).sum(dim=1)
            losses = torch.stack(
                [
                    (pooled - targets[:, task_idx]).square().mean(dim=-1)
                    for task_idx in range(3)
                ],
                dim=1,
            )
            return losses, torch.ones_like(losses)

        meta_loss, stats, aux = controller.meta_objective(
            support,
            query,
            inner_lr=0.1,
            scope="counterfactual",
            outer_task_fn=outer_task_fn,
            task_weight=1.0,
            semantic_weight=0.1,
            kl_weight=0.0,
            router_kl_weight=0.0,
            router_advantage_scale=1.0,
            normalize_inner_grad=True,
            safe_improvement_margin=1.0e-5,
            normalize_router_regret=True,
            router_regret_floor=1.0e-4,
            return_aux=True,
        )
        router_params = tuple(controller.router.parameters())
        router_grads = torch.autograd.grad(
            meta_loss, router_params, retain_graph=True
        )
        self.assertGreater(
            sum(grad.abs().sum() for grad in router_grads).item(), 0.0
        )
        for task_name in ("species", "family", "order"):
            self.assertIn(f"route_{task_name}_skip", stats)
            self.assertIn(f"route_{task_name}_part", stats)
            self.assertIn(f"route_{task_name}_relation", stats)
            self.assertIn(f"meta_{task_name}_part_improvement", stats)
            self.assertIn(f"meta_{task_name}_relation_improvement", stats)
        self.assertIn("meta_part_task_improvement", stats)
        self.assertIn("meta_relation_task_improvement", stats)
        self.assertIn("meta_router_raw_objective", stats)
        self.assertIn("meta_router_regret_rms", stats)
        self.assertIn("candidate_species_skip_rate", stats)
        self.assertIn("candidate_family_part_rate", stats)
        self.assertIn("candidate_order_relation_rate", stats)
        self.assertAlmostEqual(
            stats["meta_part_inner_step_norm"].item(), 0.1, places=5
        )
        self.assertAlmostEqual(
            stats["meta_relation_inner_step_norm"].item(), 0.1, places=5
        )
        self.assertEqual(
            tuple(aux["branch_eligibility"].shape), (self.batch, 3, 2)
        )
        self.assertEqual(aux["branch_eligibility"].dtype, torch.bool)

        for parameter in controller.parameters():
            parameter.grad = None
        real_loss, safe_stats = controller.real_weighted_alignment(
            support,
            scope="counterfactual",
            branch_eligibility=torch.zeros(
                self.batch, 3, 2, dtype=torch.bool
            ),
        )
        self.assertEqual(safe_stats["safe_route_skip"].item(), 1.0)
        self.assertEqual(safe_stats["safe_route_part"].item(), 0.0)
        self.assertEqual(safe_stats["safe_route_relation"].item(), 0.0)
        real_loss.backward()
        self.assertFalse(any(
            parameter.grad is not None
            for parameter in controller.all_policy_parameters()
        ))
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in controller.adapter.parameters()
        ))

    def test_counterfactual_safe_gate_validates_shape(self):
        controller = BilevelSemanticController(
            dim=self.dim,
            text_dim=24,
            num_parts=self.parts,
            semantic_rank=8,
            adapter_rank=8,
            policy_hidden_dim=16,
            reference_mix=0.5,
            relation_hvp_samples=1,
            routing_scope="counterfactual",
        )
        state = self._state_for(controller, 24, compute_hvp=False)
        with self.assertRaisesRegex(ValueError, "eligibility must have shape"):
            controller.real_weighted_alignment(
                state,
                scope="counterfactual",
                branch_eligibility=torch.ones(self.batch, 2),
            )

    def test_counterfactual_regret_calibration_preserves_branch_order(self):
        regret = torch.tensor([
            [[0.0, 0.20, -0.10]],
            [[0.0, -2.0e-5, 3.0e-5]],
            [[0.0, -2.0e-6, -3.0e-6]],
        ])
        calibrated, rms = self.controller._calibrate_counterfactual_regret(
            regret,
            improvement_margin=1.0e-5,
            scale_floor=1.0e-4,
            normalize=True,
        )
        # Relation is best in row 0, Part clears the margin in row 1, and
        # neither semantic branch clears the margin in row 2.
        self.assertEqual(calibrated.argmin(dim=-1).flatten().tolist(), [2, 1, 0])
        self.assertEqual(tuple(rms.shape), (3, 1, 1))
        self.assertTrue(torch.isfinite(calibrated).all())
        self.assertTrue(torch.isfinite(rms).all())

        raw, _ = self.controller._calibrate_counterfactual_regret(
            regret,
            improvement_margin=1.0e-5,
            scale_floor=1.0e-4,
            normalize=False,
        )
        expected = regret + regret.new_tensor([0.0, 1.0e-5, 1.0e-5])
        self.assertTrue(torch.allclose(raw, expected))

    def test_part_meta_gradient_and_direct_gradient_isolation(self):
        support = self._state(1, compute_hvp=False)
        query = self._state(2, compute_hvp=False)
        meta_loss, _ = self.controller.meta_objective(
            support, query, inner_lr=0.1, scope="part"
        )
        policy_params = tuple(self.controller.policy.parameters())
        meta_grads = torch.autograd.grad(meta_loss, policy_params)
        self.assertGreater(sum(g.abs().sum() for g in meta_grads).item(), 0.0)

        for param in self.controller.parameters():
            param.grad = None
        real_loss, _ = self.controller.real_weighted_alignment(
            support, scope="part"
        )
        real_loss.backward()
        self.assertFalse(any(param.grad is not None for param in policy_params))
        self.assertTrue(any(p.grad is not None for p in self.controller.adapter.parameters()))

    def test_relation_hvp_and_strict_bilevel_isolation(self):
        support = self._state(3, compute_hvp=True)
        query = self._state(4, compute_hvp=False)
        curvature = support["relation_state"]["curvature"]
        self.assertEqual(
            tuple(curvature.shape),
            (self.batch, self.parts * (self.parts - 1) // 2),
        )
        self.assertFalse(curvature.requires_grad)
        self.assertTrue(torch.isfinite(curvature).all())
        self.assertTrue((curvature >= 0).all())

        meta_loss, _ = self.controller.meta_objective(
            support, query, inner_lr=0.1, scope="relation"
        )
        policy_params = tuple(self.controller.relation_policy.parameters())
        meta_grads = torch.autograd.grad(meta_loss, policy_params)
        self.assertGreater(sum(g.abs().sum() for g in meta_grads).item(), 0.0)

        for param in self.controller.parameters():
            param.grad = None
        real_loss, _ = self.controller.real_weighted_alignment(
            support, scope="relation"
        )
        real_loss.backward()
        self.assertFalse(any(param.grad is not None for param in policy_params))
        self.assertTrue(
            any(p.grad is not None for p in self.controller.adapter.parameters())
        )

    def test_relation_geometry_is_horizontal_flip_invariant(self):
        generator = torch.Generator().manual_seed(11)
        attention = torch.rand(
            self.batch, self.parts, 4, 4, generator=generator
        )
        original = self.controller.relation_encoder.spatial_geometry(
            attention.flatten(2)
        )
        flipped = self.controller.relation_encoder.spatial_geometry(
            attention.flip(-1).flatten(2)
        )
        self.assertTrue(torch.allclose(original, flipped, atol=1e-5, rtol=1e-5))

    def test_reference_mixture_floor_for_parts_and_relations(self):
        state = self._state(5, compute_hvp=False)
        part_p, _ = self.controller.policy_distribution(
            state["part_tokens"], state["policy_semantics"], state["curvature"]
        )
        relation_p, _ = self.controller.relation_policy_distribution(
            state["relation_state"]
        )
        self.assertTrue(torch.all(part_p >= 0.5 / self.parts - 1e-7))
        self.assertTrue(
            torch.all(relation_p >= 0.5 / self.controller.num_relations - 1e-7)
        )
        self.assertTrue(torch.allclose(part_p.sum(1), torch.ones(self.batch)))
        self.assertTrue(torch.allclose(relation_p.sum(1), torch.ones(self.batch)))

    def test_adaptive_router_receives_task_hypergradient(self):
        support = self._state(6, compute_hvp=False)
        query = self._state(7, compute_hvp=False)
        target = torch.randn(self.batch, self.dim)

        def outer_task_fn(state, params, reference):
            adapted = self.controller.adapt_parts(
                state["part_tokens"].detach().float(), params
            )
            pooled = (reference.unsqueeze(-1) * adapted).sum(dim=1)
            return torch.nn.functional.mse_loss(pooled, target)

        meta_loss, stats = self.controller.meta_objective(
            support,
            query,
            inner_lr=0.1,
            scope="adaptive",
            outer_task_fn=outer_task_fn,
            task_weight=1.0,
            semantic_weight=0.0,
            kl_weight=0.0,
            router_kl_weight=0.0,
        )
        router_params = tuple(self.controller.router.parameters())
        router_grads = torch.autograd.grad(meta_loss, router_params)
        self.assertGreater(sum(g.abs().sum() for g in router_grads).item(), 0.0)
        self.assertIn("route_skip", stats)
        self.assertIn("meta_outer_task", stats)

    def test_adaptive_real_loss_cannot_update_any_policy(self):
        state = self._state(8, compute_hvp=False)
        policy_params = tuple(self.controller.all_policy_parameters())
        for parameter in self.controller.parameters():
            parameter.grad = None
        real_loss, _ = self.controller.real_weighted_alignment(
            state, scope="adaptive"
        )
        real_loss.backward()
        self.assertFalse(any(parameter.grad is not None for parameter in policy_params))
        self.assertTrue(any(
            parameter.grad is not None for parameter in self.controller.adapter.parameters()
        ))

    def test_relation_encoder_is_low_rank_and_semantic_target_is_frozen(self):
        self.assertEqual(
            self.controller.relation_encoder.relation_dim,
            self.controller.relation_dim,
        )
        self.assertTrue(all(
            not parameter.requires_grad
            for parameter in self.controller.relation_encoder.semantic_encoder.parameters()
        ))


if __name__ == "__main__":
    unittest.main()
