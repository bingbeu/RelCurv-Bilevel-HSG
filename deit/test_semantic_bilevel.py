"""CPU tests for V8.7.2 frozen-V8.5 and bilevel invariants."""

from argparse import Namespace
import inspect
import unittest

import torch

from method_presets import (
    apply_method_preset,
    preset_values,
    validate_method_configuration,
)
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

    def test_v872_presets_are_explicit_and_dataset_checked(self):
        manual = Namespace(
            method_preset="manual",
            data_set="AIR-HIER",
            meta_scope="adaptive",
        )
        self.assertEqual(apply_method_preset(manual), {})
        self.assertEqual(manual.meta_scope, "adaptive")

        cub = Namespace(
            method_preset="cub-v85",
            data_set="BIRD-HIER",
            enable_bilevel=False,
            meta_scope="adaptive",
            counterfactual_compose="residual",
            checkpoint_metric="auto",
        )
        cub_changes = apply_method_preset(cub)
        self.assertTrue(cub.enable_bilevel)
        self.assertEqual(cub.meta_scope, "counterfactual")
        self.assertEqual(cub.counterfactual_compose, "competitive")
        self.assertEqual(cub.counterfactual_solver, "v85-frozen")
        self.assertEqual(cub.checkpoint_metric, "fpa")
        self.assertTrue(cub.meta_inner_grad_normalization)
        self.assertAlmostEqual(cub.meta_safe_route_budget, 0.05)
        self.assertIn("counterfactual_compose", cub_changes)

        air = Namespace(
            method_preset="air-curvpart-v7",
            data_set="AIR-HIER",
            enable_bilevel=False,
            meta_scope="counterfactual",
            checkpoint_metric="fpa",
        )
        apply_method_preset(air)
        self.assertTrue(air.enable_bilevel)
        self.assertEqual(air.meta_scope, "part")
        self.assertEqual(air.counterfactual_solver, "unified")
        self.assertEqual(air.checkpoint_metric, "acc1")
        self.assertFalse(air.meta_inner_grad_normalization)
        self.assertAlmostEqual(air.meta_real_weight, 0.1)
        self.assertEqual(preset_values("air-curvpart-v7")["meta_scope"], "part")

        wrong_dataset = Namespace(
            method_preset="cub-v85",
            data_set="AIR-HIER",
        )
        with self.assertRaisesRegex(ValueError, "requires --data-set BIRD-HIER"):
            apply_method_preset(wrong_dataset)

        invalid_solver = Namespace(
            counterfactual_solver="v85-frozen",
            meta_scope="counterfactual",
            counterfactual_compose="residual",
        )
        with self.assertRaisesRegex(ValueError, "requires"):
            validate_method_configuration(invalid_solver)

        validate_method_configuration(cub)
        validate_method_configuration(air)

    def test_frozen_v85_source_preserves_original_operation_order(self):
        source = inspect.getsource(
            BilevelSemanticController._counterfactual_meta_objective_v85
        )
        relation_policy = source.index(
            "relation_policy, relation_policy_stats"
        )
        part_gradient = source.index("part_grads = torch.autograd.grad")
        self.assertLess(relation_policy, part_gradient)
        self.assertNotIn("species_anchor", source)

    def test_frozen_v85_solver_keeps_differentiable_feedback(self):
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
        support = self._state_for(controller, 41, compute_hvp=False)
        query = self._state_for(controller, 42, compute_hvp=False)
        targets = torch.randn(self.batch, 3, self.dim)

        def outer_task_fn(state, params, reference):
            adapted = controller.adapt_parts(
                state["part_tokens"].detach().float(), params
            )
            pooled = (reference.unsqueeze(-1) * adapted).sum(dim=1)
            losses = torch.stack(
                [
                    (pooled - targets[:, level]).square().mean(dim=-1)
                    for level in range(3)
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
            safe_route_budget=0.05,
            consistency_credit_weight=0.25,
            counterfactual_compose="competitive",
            counterfactual_solver="v85-frozen",
            safe_gate=False,
            return_aux=True,
        )
        policy_grads = torch.autograd.grad(
            meta_loss,
            tuple(controller.policy_parameters("counterfactual")),
            retain_graph=True,
        )
        self.assertGreater(
            sum(grad.abs().sum() for grad in policy_grads).item(), 0.0
        )
        self.assertEqual(stats["counterfactual_solver_v85_frozen"], 1.0)
        self.assertIn("branch_eligibility", aux)
        self.assertNotIn("species_no_regret_guard", aux)

        real_loss, real_stats = controller.real_weighted_alignment_v85(
            support,
            scope="counterfactual",
            branch_eligibility=aux["branch_eligibility"],
            branch_confidence=aux["branch_confidence"],
        )
        self.assertTrue(torch.isfinite(real_loss))
        self.assertNotIn("meta_real_residual_mode", real_stats)

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
            classification_losses = torch.stack(
                [
                    (pooled - targets[:, task_idx]).square().mean(dim=-1)
                    for task_idx in range(3)
                ],
                dim=1,
            )
            consistency_losses = torch.stack(
                [
                    (pooled - 0.5 * targets[:, task_idx]).square().mean(dim=-1)
                    for task_idx in range(3)
                ],
                dim=1,
            )
            return {
                "losses": classification_losses + 0.1 * consistency_losses,
                "mask": torch.ones_like(classification_losses),
                "classification_losses": classification_losses,
                "consistency_losses": consistency_losses,
                "species_anchor_losses": (
                    pooled - 0.75 * targets[:, 0]
                ).square().mean(dim=-1),
            }

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
            safe_route_budget=0.05,
            consistency_credit_weight=0.25,
            safe_gate=False,
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
        self.assertIn("meta_router_classification_regret_rms", stats)
        self.assertIn("meta_router_consistency_regret_rms", stats)
        self.assertIn("meta_part_classification_improvement", stats)
        self.assertIn("meta_relation_consistency_improvement", stats)
        self.assertIn("candidate_species_skip_rate", stats)
        self.assertIn("candidate_family_part_rate", stats)
        self.assertIn("candidate_order_relation_rate", stats)
        self.assertIn("two_stage_route_skip", stats)
        self.assertIn("two_stage_effective_budget", stats)
        self.assertIn("conditional_species_part", stats)
        self.assertAlmostEqual(
            stats["two_stage_route_skip"].item(), 0.95, places=6
        )
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
        self.assertEqual(
            tuple(aux["branch_confidence"].shape), (self.batch, 3, 2)
        )
        self.assertTrue((aux["branch_confidence"] >= 0).all())

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

    def test_residual_route_preserves_part_anchor(self):
        route = torch.tensor([[0.95, 0.04, 0.01]])
        competitive_part, competitive_relation = (
            self.controller._real_alignment_route_weights(
                route, "competitive"
            )
        )
        residual_part, residual_relation = (
            self.controller._real_alignment_route_weights(route, "residual")
        )
        self.assertTrue(torch.allclose(
            competitive_part, torch.tensor([0.04])
        ))
        self.assertTrue(torch.allclose(
            competitive_relation, torch.tensor([0.01])
        ))
        self.assertTrue(torch.allclose(
            residual_part, torch.tensor([0.05])
        ))
        self.assertTrue(torch.allclose(
            residual_relation, torch.tensor([0.01])
        ))

    def test_residual_unroll_uses_incremental_relation_and_species_guard(self):
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
        support = self._state_for(controller, 31, compute_hvp=False)
        query = self._state_for(controller, 32, compute_hvp=False)
        branch_calls = []
        branch_classification = (
            (1.0, 1.0, 1.0),
            (0.8, 0.8, 0.8),
            (0.9, 0.6, 0.6),
        )
        branch_anchor = (0.0, 0.001, 0.02)

        def outer_task_fn(state, params, reference):
            branch_idx = len(branch_calls)
            branch_calls.append(branch_idx)
            classification = reference.new_tensor(
                branch_classification[branch_idx]
            ).view(1, 3).expand(self.batch, -1)
            return {
                "losses": classification,
                "mask": torch.ones_like(classification),
                "classification_losses": classification,
                "consistency_losses": torch.zeros_like(classification),
                "species_anchor_losses": reference.new_full(
                    (self.batch,), branch_anchor[branch_idx]
                ),
            }

        _, stats, aux = controller.meta_objective(
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
            safe_improvement_margin=0.0,
            normalize_router_regret=True,
            router_regret_floor=1.0e-4,
            safe_route_budget=0.05,
            consistency_credit_weight=0.25,
            counterfactual_compose="residual",
            relation_residual_inner_scale=0.5,
            species_no_regret_margin=0.01,
            species_anchor_kl_margin=0.01,
            safe_gate=True,
            return_aux=True,
        )
        self.assertEqual(branch_calls, [0, 1, 2])
        self.assertEqual(stats["counterfactual_residual_mode"].item(), 1.0)
        self.assertAlmostEqual(
            stats["meta_part_inner_step_norm"].item(), 0.1, places=5
        )
        self.assertAlmostEqual(
            stats["meta_relation_inner_step_norm"].item(), 0.05, places=5
        )
        self.assertEqual(
            stats["meta_species_no_regret_accept_rate"].item(), 0.0
        )
        self.assertFalse(aux["species_no_regret_guard"].any().item())
        self.assertFalse(aux["branch_eligibility"][:, :, 1].any().item())
        self.assertGreater(
            stats[
                "meta_family_relation_incremental_classification_improvement"
            ].item(),
            0.0,
        )

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

    def test_two_stage_route_ignores_skip_and_respects_budget(self):
        learned = torch.tensor([[[0.99, 0.009, 0.001]]])
        both = torch.tensor([[[True, True]]])
        route, conditional, active, effective_budget = (
            self.controller._two_stage_counterfactual_route(
                learned, both, non_skip_budget=0.05
            )
        )
        self.assertTrue(torch.allclose(
            conditional, torch.tensor([[[0.9, 0.1]]])
        ))
        self.assertTrue(torch.allclose(
            route, torch.tensor([[[0.95, 0.045, 0.005]]])
        ))
        self.assertTrue(active.item())
        self.assertAlmostEqual(effective_budget.item(), 0.05, places=7)

        relation_only = torch.tensor([[[False, True]]])
        route, _, _, _ = self.controller._two_stage_counterfactual_route(
            learned, relation_only, non_skip_budget=0.05
        )
        self.assertTrue(torch.allclose(
            route, torch.tensor([[[0.95, 0.0, 0.05]]])
        ))

        none = torch.tensor([[[False, False]]])
        route, _, active, effective_budget = (
            self.controller._two_stage_counterfactual_route(
                learned, none, non_skip_budget=0.05
            )
        )
        self.assertTrue(torch.allclose(
            route, torch.tensor([[[1.0, 0.0, 0.0]]])
        ))
        self.assertFalse(active.item())
        self.assertEqual(effective_budget.item(), 0.0)

    def test_confidence_reallocates_fixed_safe_budget(self):
        learned = torch.tensor([[[0.50, 0.45, 0.05]]])
        eligible = torch.tensor([[[True, True]]])
        confidence = torch.tensor([[[0.25, 0.75]]])
        route, _, active, effective_budget = (
            self.controller._two_stage_counterfactual_route(
                learned,
                eligible,
                non_skip_budget=0.05,
                branch_confidence=confidence,
                confidence_scale=1.0,
            )
        )
        self.assertTrue(active.item())
        self.assertAlmostEqual(effective_budget.item(), 0.05, places=7)
        self.assertAlmostEqual(route[..., 0].item(), 0.95, places=7)
        self.assertAlmostEqual(route[..., 1].item(), 0.0375, places=7)
        self.assertAlmostEqual(route[..., 2].item(), 0.0125, places=7)
        self.assertAlmostEqual(route[..., 1:].sum().item(), 0.05, places=7)

        part_only = torch.tensor([[[True, False]]])
        route, _, _, effective_budget = (
            self.controller._two_stage_counterfactual_route(
                learned,
                part_only,
                non_skip_budget=0.05,
                branch_confidence=confidence,
                confidence_scale=1.0,
            )
        )
        self.assertAlmostEqual(effective_budget.item(), 0.05, places=7)
        self.assertAlmostEqual(route[..., 1].item(), 0.05, places=7)
        self.assertEqual(route[..., 2].item(), 0.0)

        saturated = torch.tensor([[[4.0, 2.0]]])
        route, _, _, effective_budget = (
            self.controller._two_stage_counterfactual_route(
                learned,
                eligible,
                non_skip_budget=0.05,
                branch_confidence=saturated,
                confidence_scale=1.0,
            )
        )
        self.assertAlmostEqual(effective_budget.item(), 0.05, places=7)
        self.assertAlmostEqual(route[..., 1:].sum().item(), 0.05, places=7)

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
        real_loss, real_stats = self.controller.real_weighted_alignment(
            support, scope="part"
        )
        self.assertEqual(real_stats["meta_real_part_weight"].item(), 1.0)
        self.assertEqual(real_stats["meta_real_relation_weight"].item(), 0.0)
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
