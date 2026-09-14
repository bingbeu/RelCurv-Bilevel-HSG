"""Curvature-aware semantic weighting with a differentiable bilevel loop.

V5 learns a policy over local semantic part tokens. V6 additionally builds
pairwise part relations (appearance + spatial layout), measures semantic
curvature on those relation tokens, and learns which relations should guide a
one-step support update by evaluating its effect on a separate query view.

Policy parameters ``phi`` are isolated from ordinary task losses: policy inputs
are stopped, and weights used by real losses and classification are detached.
Only the post-update outer objective updates ``phi``.

V8 adds a branch-counterfactual, all-level route.  Skip, part and relation
updates are unrolled independently and evaluated on Species, Family and Order
losses.  V8.2 calibrates branch regret inside every example and hierarchy
level. V8.3 separates safety from semantic selection. V8.4 scales the complete
budget by confidence. V8.5 restores a fixed safe budget, uses confidence only
for conditional Part/Relation allocation, and credits independently calibrated
taxonomy-consistency gains in every hierarchy-level route. V8.6 optionally
anchors every semantic update on Part and treats Relation as a guarded residual,
so relation consistency credit cannot replace a classification-helpful Part
update. V8.7 keeps that residual solver as an ablation and exposes explicit,
dataset-validated experiment presets: the best observed V8.5 competitive path
for CUB and the verified Part-only CurvPart V7 path for Aircraft.
"""

import math
from collections import OrderedDict
from contextlib import nullcontext
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticTokenBridge(nn.Module):
    """Produce P distinct, view-consistent semantic tokens."""

    def __init__(self, dim: int, text_dim: int, num_parts: int, rank: int = 64):
        super().__init__()
        self.dim = dim
        self.num_parts = num_parts
        self.anchors = nn.Parameter(torch.empty(1, num_parts, dim))
        self.part_basis = nn.Parameter(torch.empty(num_parts, rank, dim))
        self.visual_context = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, rank), nn.Tanh()
        )
        self.text_context = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, rank), nn.Tanh()
        )
        self.visual_scale = nn.Parameter(torch.tensor(0.1))
        self.text_scale = nn.Parameter(torch.tensor(0.1))
        nn.init.trunc_normal_(self.anchors, std=0.02)
        nn.init.trunc_normal_(self.part_basis, std=0.02)

    def _tokens(self, context: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        delta = torch.einsum("br,prc->bpc", context, self.part_basis)
        return self.anchors + torch.tanh(scale) * delta

    def from_visual(self, cls_feature: torch.Tensor) -> torch.Tensor:
        return self._tokens(self.visual_context(cls_feature), self.visual_scale)

    def from_text(self, text_feature: torch.Tensor) -> torch.Tensor:
        return self._tokens(self.text_context(text_feature.float()), self.text_scale)

    @staticmethod
    def _diversity_loss(tokens: torch.Tensor) -> torch.Tensor:
        proto = F.normalize(tokens.mean(dim=0), dim=-1)
        gram = proto @ proto.transpose(0, 1)
        eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        return ((gram - eye) ** 2).sum() / max(gram.numel() - gram.size(0), 1)

    def consistency_loss(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: Optional[torch.Tensor],
        diversity_weight: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        diversity = self._diversity_loss(visual_tokens)
        if text_tokens is None:
            distill = visual_tokens.new_zeros(())
        else:
            distill = (
                1.0 - F.cosine_similarity(visual_tokens, text_tokens, dim=-1)
            ).mean()
        total = distill + float(diversity_weight) * diversity
        return total, {
            "semantic_distill_loss": distill.detach(),
            "semantic_diversity_loss": diversity.detach(),
        }


class LowRankSemanticAdapter(nn.Module):
    """Small shared lower-level variable used by a virtual update."""

    def __init__(self, dim: int, rank: int = 64):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)
        nn.init.trunc_normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.trunc_normal_(self.up.weight, std=0.02)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(F.gelu(self.down(self.norm(x))))

    def functional_forward(
        self, x: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        h = F.layer_norm(
            x, (self.dim,), params["norm.weight"], params["norm.bias"],
            self.norm.eps,
        )
        h = F.gelu(F.linear(h, params["down.weight"], params["down.bias"]))
        h = F.linear(h, params["up.weight"], params["up.bias"])
        return x + h


class CurvatureSemanticWeightPolicy(nn.Module):
    """Predict an update policy from tokens, semantics and stopped HVP values."""

    def __init__(
        self, dim: int, hidden_dim: int = 128, tau: float = 1.0,
        reference_mix: float = 0.5,
    ):
        super().__init__()
        if tau <= 0:
            raise ValueError("meta policy temperature must be positive")
        if not 0.0 <= reference_mix <= 1.0:
            raise ValueError("reference_mix must be in [0, 1]")
        self.tau = float(tau)
        self.reference_mix = float(reference_mix)
        self.net = nn.Sequential(
            nn.LayerNorm(4 * dim + 2),
            nn.Linear(4 * dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Uniform at initialization; no random part/edge preference.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, visual_tokens: torch.Tensor, semantic_tokens: torch.Tensor,
        curvature: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        visual = F.normalize(visual_tokens, dim=-1)
        semantic = F.normalize(semantic_tokens, dim=-1)
        curvature = curvature.squeeze(-1) if curvature.dim() == 3 else curvature
        curvature = curvature / curvature.mean(dim=1, keepdim=True).clamp_min(1e-6)
        cosine = (visual * semantic).sum(dim=-1, keepdim=True)
        features = torch.cat(
            (visual, semantic, torch.abs(visual - semantic), visual * semantic,
             torch.log1p(curvature.clamp_min(0)).unsqueeze(-1), cosine),
            dim=-1,
        )
        logits = self.net(features).squeeze(-1)
        learned = torch.softmax(logits / self.tau, dim=1)
        reference = torch.full_like(learned, 1.0 / learned.size(1))
        p = (1.0 - self.reference_mix) * reference + self.reference_mix * learned
        entropy = -(p * p.clamp_min(1e-8).log()).sum(dim=1)
        return p, {
            "policy_entropy": entropy.mean().detach(),
            "policy_effective_items": entropy.exp().mean().detach(),
            "policy_max": p.max(dim=1).values.mean().detach(),
            "policy_logit_std": logits.std(dim=1, unbiased=False).mean().detach(),
        }


class AdaptiveGranularityRouter(nn.Module):
    """Meta-policy over skip, local-part and relation semantic updates.

    ``num_tasks=1`` preserves the V7 global router exactly.  ``num_tasks=3``
    produces a task-by-branch matrix for fine/species, family and basic/order.
    V8.3 retains the three logits for checkpoint compatibility, but its
    counterfactual solver removes the Skip logit and normalizes only the Part
    and Relation logits after the positive-gain safety decision.
    """

    TASK_NAMES = ("species", "family", "order")

    def __init__(
        self,
        hidden_dim: int = 32,
        prior=(0.50, 0.45, 0.05),
        num_tasks: int = 1,
    ):
        super().__init__()
        if num_tasks not in (1, 3):
            raise ValueError("granularity router num_tasks must be 1 or 3")
        self.num_tasks = int(num_tasks)
        prior_tensor = torch.as_tensor(prior, dtype=torch.float32)
        if prior_tensor.numel() != 3 or (prior_tensor <= 0).any():
            raise ValueError("router prior must contain three positive values")
        prior_tensor = prior_tensor / prior_tensor.sum()
        self.register_buffer("prior", prior_tensor)
        self.net = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3 * self.num_tasks),
        )
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.copy_(prior_tensor.log().repeat(self.num_tasks))

    def forward(
        self,
        part_error: torch.Tensor,
        relation_error: torch.Tensor,
        part_curvature: torch.Tensor,
        relation_curvature: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        def moments(value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            value = value.detach().float()
            return value.mean(dim=1), value.std(dim=1, unbiased=False)

        pe_mean, pe_std = moments(part_error)
        re_mean, re_std = moments(relation_error)
        pc_mean, pc_std = moments(part_curvature)
        rc_mean, rc_std = moments(relation_curvature)
        features = torch.stack(
            (pe_mean, pe_std, re_mean, re_std,
             pc_mean, pc_std, rc_mean, rc_std),
            dim=-1,
        )
        logits = self.net(features)
        if self.num_tasks == 1:
            route = torch.softmax(logits, dim=-1)
            entropy = -(route * route.clamp_min(1e-8).log()).sum(dim=-1)
            return route, {
                "route_skip": route[:, 0].mean().detach(),
                "route_part": route[:, 1].mean().detach(),
                "route_relation": route[:, 2].mean().detach(),
                "route_entropy": entropy.mean().detach(),
            }

        route = torch.softmax(
            logits.reshape(features.size(0), self.num_tasks, 3), dim=-1
        )
        entropy = -(route * route.clamp_min(1e-8).log()).sum(dim=-1)
        stats = {
            "route_skip": route[:, :, 0].mean().detach(),
            "route_part": route[:, :, 1].mean().detach(),
            "route_relation": route[:, :, 2].mean().detach(),
            "route_entropy": entropy.mean().detach(),
        }
        for task_idx, task_name in enumerate(self.TASK_NAMES):
            stats.update({
                f"route_{task_name}_skip": route[:, task_idx, 0].mean().detach(),
                f"route_{task_name}_part": route[:, task_idx, 1].mean().detach(),
                f"route_{task_name}_relation": route[:, task_idx, 2].mean().detach(),
                f"route_{task_name}_entropy": entropy[:, task_idx].mean().detach(),
            })
        return route, stats


class PairwiseRelationEncoder(nn.Module):
    """Encode all undirected part pairs as appearance-layout relations.

    With P=8 this produces 28 edges. Spatial descriptors are translation and
    horizontal-flip robust: absolute centroid displacement, distance, absolute
    spread difference, and soft-attention overlap.
    """

    def __init__(self, dim: int, num_parts: int, relation_dim: int = 64):
        super().__init__()
        if num_parts < 2:
            raise ValueError("relation modeling needs at least two parts")
        self.dim = dim
        self.relation_dim = relation_dim
        self.num_parts = num_parts
        self.register_buffer(
            "pair_index", torch.triu_indices(num_parts, num_parts, offset=1),
            persistent=False,
        )
        self.visual_encoder = nn.Sequential(
            nn.LayerNorm(2 * dim + 6),
            nn.Linear(2 * dim + 6, relation_dim),
            nn.GELU(),
            nn.LayerNorm(relation_dim),
        )
        self.semantic_encoder = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, relation_dim),
            nn.GELU(),
            nn.LayerNorm(relation_dim),
        )
        # A moving target encoder lets the relation loss minimize itself by
        # changing both sides.  Keep this mapping fixed: relation supervision
        # may update visual relations, but never its semantic target space.
        for parameter in self.semantic_encoder.parameters():
            parameter.requires_grad_(False)

    @property
    def num_relations(self) -> int:
        return int(self.pair_index.size(1))

    def _split_pairs(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            tokens.index_select(1, self.pair_index[0]),
            tokens.index_select(1, self.pair_index[1]),
        )

    @staticmethod
    def _patch_coordinates(
        num_patches: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        height = int(math.sqrt(num_patches))
        if height * height != num_patches:
            raise ValueError(
                "relation geometry expects a square patch grid; "
                f"received {num_patches} patches"
            )
        axis = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(num_patches, 2)

    def spatial_geometry(self, part_attn: torch.Tensor) -> torch.Tensor:
        if part_attn.dim() != 3 or part_attn.size(1) != self.num_parts:
            raise ValueError("part_attn must have shape (B, P, N)")
        prob = part_attn.clamp_min(0)
        prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        coords = self._patch_coordinates(prob.size(-1), prob.device, prob.dtype)
        centers = torch.einsum("bpn,nc->bpc", prob, coords)
        centered = coords.view(1, 1, -1, 2) - centers.unsqueeze(2)
        spreads = (prob.unsqueeze(-1) * centered.square()).sum(dim=2)

        center_l, center_r = self._split_pairs(centers)
        spread_l, spread_r = self._split_pairs(spreads)
        prob_l, prob_r = self._split_pairs(prob)
        center_delta = torch.abs(center_l - center_r)
        center_distance = torch.linalg.vector_norm(center_delta, dim=-1, keepdim=True)
        spread_delta = torch.abs(spread_l - spread_r)
        overlap = torch.sqrt((prob_l * prob_r).clamp_min(1e-8)).sum(
            dim=-1, keepdim=True
        )
        return torch.cat(
            (center_delta, center_distance, spread_delta, overlap), dim=-1
        )

    def visual_relations(
        self, part_tokens: torch.Tensor, part_attn: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        left, right = self._split_pairs(part_tokens)
        appearance = torch.cat((left * right, torch.abs(left - right)), dim=-1)
        geometry = self.spatial_geometry(part_attn)
        return self.visual_encoder(torch.cat((appearance, geometry), dim=-1)), geometry

    def semantic_relations(
        self, semantics: torch.Tensor, stop_gradient: bool = True
    ) -> torch.Tensor:
        left, right = self._split_pairs(semantics)
        relation = torch.cat((left * right, torch.abs(left - right)), dim=-1)
        encoded = self.semantic_encoder(relation)
        return encoded.detach() if stop_gradient else encoded


class BilevelSemanticController(nn.Module):
    """Part- and relation-level one-step differentiable bilevel controller."""

    VALID_SCOPES = (
        "part", "relation", "hybrid", "adaptive", "counterfactual"
    )

    def __init__(
        self, dim: int, text_dim: int, num_parts: int, semantic_rank: int = 64,
        adapter_rank: int = 64, policy_hidden_dim: int = 128,
        policy_tau: float = 1.0, reference_mix: float = 0.5,
        diversity_weight: float = 0.01, enable_relations: bool = True,
        relation_hvp_samples: int = 1,
        relation_contrastive_weight: float = 0.1,
        relation_temperature: float = 0.1,
        relation_dim: int = 64,
        router_hidden_dim: int = 32,
        router_prior=(0.50, 0.45, 0.05),
        routing_scope: str = "adaptive",
    ):
        super().__init__()
        if relation_hvp_samples < 1:
            raise ValueError("relation_hvp_samples must be >= 1")
        self.num_parts = num_parts
        self.diversity_weight = float(diversity_weight)
        self.enable_relations = bool(enable_relations)
        self.relation_hvp_samples = int(relation_hvp_samples)
        if relation_contrastive_weight < 0:
            raise ValueError("relation_contrastive_weight must be nonnegative")
        if relation_temperature <= 0:
            raise ValueError("relation_temperature must be positive")
        self.relation_contrastive_weight = float(relation_contrastive_weight)
        self.relation_temperature = float(relation_temperature)
        self.relation_dim = int(relation_dim)
        self._check_scope(routing_scope)
        self.routing_scope = routing_scope
        self.bridge = SemanticTokenBridge(dim, text_dim, num_parts, semantic_rank)
        self.adapter = LowRankSemanticAdapter(dim, adapter_rank)
        self.policy = CurvatureSemanticWeightPolicy(
            dim, policy_hidden_dim, policy_tau, reference_mix
        )
        if self.enable_relations:
            self.relation_encoder = PairwiseRelationEncoder(
                dim, num_parts, relation_dim=self.relation_dim
            )
            self.relation_policy = CurvatureSemanticWeightPolicy(
                self.relation_dim, policy_hidden_dim, policy_tau, reference_mix
            )
            self.router = AdaptiveGranularityRouter(
                hidden_dim=router_hidden_dim,
                prior=router_prior,
                num_tasks=(3 if routing_scope == "counterfactual" else 1),
            )

    @property
    def num_relations(self) -> int:
        return self.relation_encoder.num_relations if self.enable_relations else 0

    def visual_semantics(self, cls_feature: torch.Tensor) -> torch.Tensor:
        return self.bridge.from_visual(cls_feature)

    def text_semantics(self, text_feature: torch.Tensor) -> torch.Tensor:
        return self.bridge.from_text(text_feature)

    def bridge_loss(
        self, visual_semantics: torch.Tensor,
        text_semantics: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.bridge.consistency_loss(
            visual_semantics, text_semantics, self.diversity_weight
        )

    @staticmethod
    def make_state(
        part_tokens: torch.Tensor, policy_semantics: torch.Tensor,
        target_semantics: torch.Tensor, curvature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "part_tokens": part_tokens,
            "policy_semantics": policy_semantics,
            "target_semantics": target_semantics,
            "curvature": curvature,
        }

    def adapt_parts(
        self, part_tokens: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Shared lower-level variable used by all semantic granularities."""
        adapter_dtype = next(self.adapter.parameters()).dtype
        tokens = part_tokens.to(dtype=adapter_dtype)
        if params is None:
            return self.adapter(tokens)
        return self.adapter.functional_forward(tokens, params)

    @staticmethod
    def _direct_alignment_error(
        visual_tokens: torch.Tensor,
        target_semantics: torch.Tensor,
        contrastive_weight: float = 0.0,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        target_semantics = target_semantics.to(dtype=visual_tokens.dtype)
        cosine_error = 1.0 - F.cosine_similarity(
            visual_tokens, target_semantics, dim=-1
        )
        if contrastive_weight <= 0:
            return cosine_error

        # Every relation must match its own semantic edge rather than a shared
        # constant vector.  reduction='none' retains one selectable error per
        # edge, which is required by the bilevel policy.
        adapted_norm = F.normalize(visual_tokens, dim=-1)
        target_norm = F.normalize(target_semantics, dim=-1)
        logits = torch.matmul(
            adapted_norm, target_norm.transpose(1, 2)
        ) / float(temperature)
        batch_size, num_items, _ = logits.shape
        labels = torch.arange(num_items, device=logits.device)
        labels = labels.unsqueeze(0).expand(batch_size, -1).reshape(-1)
        contrastive_error = F.cross_entropy(
            logits.reshape(batch_size * num_items, num_items),
            labels,
            reduction="none",
        ).reshape(batch_size, num_items)
        return cosine_error + float(contrastive_weight) * contrastive_error

    def alignment_error(
        self, visual_tokens: torch.Tensor, target_semantics: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        adapted = self.adapt_parts(visual_tokens, params)
        return self._direct_alignment_error(adapted, target_semantics)

    def relation_alignment_error(
        self, visual_relations: torch.Tensor, target_relations: torch.Tensor,
    ) -> torch.Tensor:
        return self._direct_alignment_error(
            visual_relations,
            target_relations,
            contrastive_weight=self.relation_contrastive_weight,
            temperature=self.relation_temperature,
        )

    @staticmethod
    def reference_distribution(
        error: torch.Tensor, curvature: torch.Tensor, q_mode: str,
    ) -> torch.Tensor:
        if q_mode == "uniform":
            q = torch.full_like(error, 1.0 / error.size(1))
        elif q_mode == "hvp":
            q = curvature.squeeze(-1) if curvature.dim() == 3 else curvature
            q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-6)
        else:
            raise ValueError("meta q_mode must be 'uniform' or 'hvp'")
        return q.detach()

    @staticmethod
    def _stopped_policy_distribution(
        policy: CurvatureSemanticWeightPolicy,
        visual_tokens: torch.Tensor,
        policy_semantics: torch.Tensor,
        curvature: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        policy_dtype = next(policy.parameters()).dtype

        return policy(
            visual_tokens.detach().to(dtype=policy_dtype),
            policy_semantics.detach().to(dtype=policy_dtype),
            curvature.detach().to(dtype=policy_dtype),
        )

    def policy_distribution(
        self, part_tokens: torch.Tensor, policy_semantics: torch.Tensor,
        curvature: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self._stopped_policy_distribution(
            self.policy, part_tokens, policy_semantics, curvature
        )

    def relation_policy_distribution(
        self, relation_state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.enable_relations:
            raise RuntimeError("relation controller is disabled")
        return self._stopped_policy_distribution(
            self.relation_policy, relation_state["tokens"],
            relation_state["policy_semantics"], relation_state["curvature"],
        )

    def pool_parts(
        self, part_tokens: torch.Tensor, policy_semantics: torch.Tensor,
        curvature: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        p, stats = self.policy_distribution(part_tokens, policy_semantics, curvature)
        adapted = self.adapt_parts(part_tokens)
        return (p.detach().unsqueeze(-1) * adapted).sum(dim=1), stats

    def pool_relations(
        self, relation_state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        p, stats = self.relation_policy_distribution(relation_state)
        pooled = (p.detach().unsqueeze(-1) * relation_state["tokens"]).sum(dim=1)
        return pooled, {"rel_" + key: value for key, value in stats.items()}

    def _endpoint_prior(self, part_curvature: torch.Tensor) -> torch.Tensor:
        left, right = self.relation_encoder._split_pairs(part_curvature.unsqueeze(-1))
        prior = torch.sqrt((left.squeeze(-1) * right.squeeze(-1)).clamp_min(1e-8))
        return prior / prior.mean(dim=1, keepdim=True).clamp_min(1e-6)

    def relation_hvp_curvature(
        self, relation_tokens: torch.Tensor, target_relations: torch.Tensor,
    ) -> torch.Tensor:
        """Hutchinson HVP norm of stopped semantic relation alignment."""
        if not self.enable_relations:
            raise RuntimeError("relation controller is disabled")
        fp32_context = (
            torch.cuda.amp.autocast(enabled=False)
            if relation_tokens.is_cuda else nullcontext()
        )
        with torch.enable_grad():
            with fp32_context:
                z = relation_tokens.detach().float().requires_grad_(True)
                target = target_relations.detach().float()
                error = self.relation_alignment_error(z, target)
                gradient = torch.autograd.grad(
                    error.mean(), z, create_graph=True, retain_graph=True
                )[0]
                estimate = torch.zeros_like(error)
                for sample_idx in range(self.relation_hvp_samples):
                    vector = torch.empty_like(z).bernoulli_(0.5).mul_(2).sub_(1)
                    vector = vector / math.sqrt(z.size(-1))
                    hvp = torch.autograd.grad(
                        (gradient * vector).sum(), z,
                        retain_graph=(sample_idx + 1 < self.relation_hvp_samples),
                        create_graph=False,
                    )[0]
                    estimate = estimate + torch.linalg.vector_norm(hvp, dim=-1)
                estimate = estimate / float(self.relation_hvp_samples)
        estimate = estimate / estimate.mean(dim=1, keepdim=True).clamp_min(1e-6)
        return estimate.detach().to(relation_tokens.dtype)

    def make_relation_state(
        self, part_tokens: torch.Tensor, policy_semantics: torch.Tensor,
        target_semantics: torch.Tensor, part_attn: torch.Tensor,
        part_curvature: torch.Tensor, compute_hvp: bool,
    ) -> Dict[str, torch.Tensor]:
        if not self.enable_relations:
            raise RuntimeError("relation controller is disabled")
        adapted_parts = self.adapt_parts(part_tokens)
        visual_relations, geometry = self.relation_encoder.visual_relations(
            adapted_parts, part_attn
        )
        # Both policy context and target are semantic evidence, not trainable
        # shortcuts for the real alignment loss.  The visual relation tokens
        # remain live and receive gradients through the adapter/loss.
        policy_relations = self.relation_encoder.semantic_relations(
            policy_semantics.detach(), stop_gradient=True
        )
        target_relations = self.relation_encoder.semantic_relations(
            target_semantics.detach(), stop_gradient=True
        )
        curvature = (
            self.relation_hvp_curvature(visual_relations, target_relations)
            if compute_hvp else self._endpoint_prior(part_curvature.detach())
        )
        return {
            "part_tokens": part_tokens,
            "part_attn": part_attn,
            "tokens": visual_relations,
            "policy_semantics": policy_relations,
            "target_semantics": target_relations,
            "curvature": curvature,
            "geometry": geometry,
        }

    @classmethod
    def _check_scope(cls, scope: str) -> None:
        if scope not in cls.VALID_SCOPES:
            raise ValueError(f"meta scope must be one of {cls.VALID_SCOPES}")

    @staticmethod
    def _policy_kl(policy: torch.Tensor) -> torch.Tensor:
        uniform = torch.full_like(policy, 1.0 / policy.size(1))
        return (
            policy * (policy.clamp_min(1e-8).log() - uniform.log())
        ).sum(dim=1).mean()

    def _relation_tokens(
        self, relation_state: Dict[str, torch.Tensor],
        adapter_params: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        adapted = self.adapt_parts(
            relation_state["part_tokens"].detach().float(), adapter_params
        )
        attention = relation_state["part_attn"].detach().to(dtype=adapted.dtype)
        tokens, _ = self.relation_encoder.visual_relations(adapted, attention)
        return tokens

    def _route(
        self, scope: str, part_error: torch.Tensor,
        relation_error: Optional[torch.Tensor], part_curvature: torch.Tensor,
        relation_curvature: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch = part_error.size(0)
        if scope == "adaptive":
            if relation_error is None or relation_curvature is None:
                raise RuntimeError("adaptive routing requires relation evidence")
            return self.router(
                part_error, relation_error, part_curvature, relation_curvature
            )
        route = part_error.new_zeros(batch, 3)
        if scope == "part":
            route[:, 1] = 1.0
        elif scope == "relation":
            route[:, 2] = 1.0
        else:  # hybrid
            route[:, 1:] = 0.5
        return route, {
            "route_skip": route[:, 0].mean().detach(),
            "route_part": route[:, 1].mean().detach(),
            "route_relation": route[:, 2].mean().detach(),
            "route_entropy": route.new_zeros(()),
        }

    @staticmethod
    def _fast_params(
        base_params: "OrderedDict[str, torch.Tensor]",
        grads: Tuple[torch.Tensor, ...],
        inner_lr: float,
    ) -> "OrderedDict[str, torch.Tensor]":
        return OrderedDict(
            (name, param - float(inner_lr) * grad)
            for (name, param), grad in zip(base_params.items(), grads)
        )

    @staticmethod
    def _counterfactual_fast_params(
        base_params: "OrderedDict[str, torch.Tensor]",
        grads: Tuple[torch.Tensor, ...],
        inner_lr: float,
        normalize: bool,
    ) -> Tuple["OrderedDict[str, torch.Tensor]", torch.Tensor, torch.Tensor]:
        """Apply one branch-specific virtual step and report its scale.

        With normalization enabled, ``inner_lr`` is the L2 norm of the whole
        virtual adapter step.  This preserves each branch's gradient direction
        while preventing a tiny raw gradient from making all counterfactual
        query losses numerically indistinguishable.
        """
        if not grads:
            raise ValueError("counterfactual virtual update requires gradients")
        squared_norm = sum(
            (grad.float().square().sum() for grad in grads),
            grads[0].new_zeros((), dtype=torch.float32),
        )
        raw_norm = squared_norm.clamp_min(0.0).sqrt()
        denominator = raw_norm.clamp_min(1.0e-12) if normalize else raw_norm.new_ones(())
        update_grads = tuple(grad / denominator.to(grad) for grad in grads)
        fast_params = BilevelSemanticController._fast_params(
            base_params, update_grads, inner_lr
        )
        step_squared_norm = sum(
            ((float(inner_lr) * grad).float().square().sum()
             for grad in update_grads),
            raw_norm.new_zeros(()),
        )
        return fast_params, raw_norm, step_squared_norm.clamp_min(0.0).sqrt()

    @staticmethod
    def _calibrate_counterfactual_regret(
        relative_regret: torch.Tensor,
        improvement_margin: float,
        scale_floor: float,
        normalize: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calibrate branch evidence within each example and task level.

        Skip keeps zero regret. The positive margin is added only to Part and
        Relation, so a semantic branch must beat Skip by more than the safety
        threshold. RMS scaling is stopped: it changes the strength, not the
        direction, of the differentiable counterfactual signal.
        """
        if relative_regret.size(-1) != 3:
            raise ValueError("counterfactual regret must contain three branches")
        if improvement_margin < 0:
            raise ValueError("counterfactual improvement margin must be nonnegative")
        if scale_floor <= 0:
            raise ValueError("counterfactual regret scale floor must be positive")
        branch_margin = relative_regret.new_tensor(
            (0.0, float(improvement_margin), float(improvement_margin))
        )
        adjusted_regret = relative_regret + branch_margin
        regret_rms = adjusted_regret.detach().square().mean(
            dim=-1, keepdim=True
        ).sqrt()
        if not normalize:
            return adjusted_regret, regret_rms
        calibrated = adjusted_regret / regret_rms.clamp_min(
            float(scale_floor)
        )
        return calibrated, regret_rms

    @staticmethod
    def _two_stage_counterfactual_route(
        learned_route: torch.Tensor,
        branch_eligibility: Optional[torch.Tensor],
        non_skip_budget: float,
        branch_confidence: Optional[torch.Tensor] = None,
        confidence_scale: float = 1.0,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Let safety decide Skip and learn only Part-versus-Relation.

        ``learned_route`` keeps the legacy Skip/Part/Relation shape, but its
        Skip logit is intentionally removed by conditioning on the two
        semantic branches.  Eligibility is stopped counterfactual evidence.
        V8.5 keeps the safe non-Skip budget fixed whenever at least one branch
        has positive stopped evidence.  Confidence only reweights the learned
        conditional Part-versus-Relation allocation.  This preserves the
        bounded V8.3 update strength while allowing stronger counterfactual
        evidence to choose which semantic update receives that budget.
        """
        if learned_route.size(-1) != 3:
            raise ValueError("two-stage route requires Skip/Part/Relation logits")
        if not 0.0 <= float(non_skip_budget) <= 1.0:
            raise ValueError("counterfactual non-Skip budget must be in [0, 1]")
        if confidence_scale <= 0:
            raise ValueError("counterfactual confidence scale must be positive")

        semantic_route = learned_route[..., 1:]
        conditional_route = semantic_route / semantic_route.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        if branch_eligibility is None:
            eligible = torch.ones_like(conditional_route)
        else:
            if tuple(branch_eligibility.shape) != tuple(conditional_route.shape):
                raise ValueError(
                    "counterfactual branch eligibility must have shape "
                    f"{tuple(conditional_route.shape)}, got "
                    f"{tuple(branch_eligibility.shape)}"
                )
            eligible = branch_eligibility.detach().to(
                device=conditional_route.device,
                dtype=conditional_route.dtype,
            )

        if branch_confidence is None:
            confidence_weight = torch.ones_like(conditional_route)
        else:
            if tuple(branch_confidence.shape) != tuple(conditional_route.shape):
                raise ValueError(
                    "counterfactual branch confidence must have shape "
                    f"{tuple(conditional_route.shape)}, got "
                    f"{tuple(branch_confidence.shape)}"
                )
            confidence = branch_confidence.detach().to(
                device=conditional_route.device,
                dtype=conditional_route.dtype,
            ).clamp_min(0.0)
            confidence_weight = (
                confidence / float(confidence_scale)
            ).clamp(min=1.0e-8, max=1.0)

        eligible_route = conditional_route * eligible * confidence_weight
        eligible_mass = eligible_route.sum(dim=-1, keepdim=True)
        has_eligible = eligible_mass.gt(0.0)
        eligible_route = eligible_route / eligible_mass.clamp_min(1.0e-8)
        semantic_budget = (
            has_eligible.to(conditional_route.dtype)
            * float(non_skip_budget)
        )
        safe_non_skip = semantic_budget * eligible_route
        safe_skip = 1.0 - safe_non_skip.sum(dim=-1, keepdim=True)
        safe_route = torch.cat((safe_skip, safe_non_skip), dim=-1)
        return (
            safe_route,
            conditional_route,
            has_eligible.squeeze(-1),
            semantic_budget.squeeze(-1),
        )

    @staticmethod
    def _real_alignment_route_weights(
        route: torch.Tensor,
        counterfactual_compose: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map counterfactual branches to executable alignment weights.

        In competitive mode Part and Relation are independent alternatives.
        In residual mode the Relation branch is ``Part -> Relation``, so its
        probability contributes to the Part anchor as well as the Relation
        residual.  This is the key invariant that prevents relation credit
        from stealing useful Part supervision on Aircraft.
        """
        if route.size(-1) != 3:
            raise ValueError("counterfactual route must contain three branches")
        if counterfactual_compose not in ("competitive", "residual"):
            raise ValueError(
                "counterfactual compose must be 'competitive' or 'residual'"
            )
        part_weight = route[..., 1]
        relation_weight = route[..., 2]
        if counterfactual_compose == "residual":
            part_weight = part_weight + relation_weight
        return part_weight, relation_weight

    @staticmethod
    def _task_output(
        value,
        batch_size: int,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Validate an all-level outer evaluator and expose route credits.

        Counterfactual routing needs unreduced losses so the support/query pair
        keeps its own route.  A rich dictionary may additionally separate
        supervised classification and hierarchy-consistency losses.  Legacy
        ``(losses, mask)`` evaluators remain valid and receive zero independent
        consistency credit.
        """
        if isinstance(value, dict):
            losses = value.get("losses")
            mask = value.get("mask")
            classification_losses = value.get("classification_losses", losses)
            consistency_losses = value.get("consistency_losses")
            species_anchor_losses = value.get("species_anchor_losses")
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            losses, mask = value
            classification_losses = losses
            consistency_losses = None
            species_anchor_losses = None
        else:
            raise TypeError(
                "counterfactual outer_task_fn must return (losses, mask) or "
                "a component dictionary with shape (B, 3)"
            )
        if not isinstance(losses, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise TypeError("counterfactual task losses and mask must be tensors")
        if tuple(losses.shape) != (batch_size, 3):
            raise ValueError(
                "counterfactual task losses must have shape "
                f"({batch_size}, 3), got {tuple(losses.shape)}"
            )
        if tuple(mask.shape) != tuple(losses.shape):
            raise ValueError("counterfactual task mask must match task losses")
        if not isinstance(classification_losses, torch.Tensor):
            raise TypeError("counterfactual classification losses must be a tensor")
        if tuple(classification_losses.shape) != tuple(losses.shape):
            raise ValueError(
                "counterfactual classification losses must match task losses"
            )
        if consistency_losses is None:
            consistency_losses = torch.zeros_like(losses)
        if not isinstance(consistency_losses, torch.Tensor):
            raise TypeError("counterfactual consistency losses must be a tensor")
        if tuple(consistency_losses.shape) != tuple(losses.shape):
            raise ValueError(
                "counterfactual consistency losses must match task losses"
            )
        if species_anchor_losses is None:
            species_anchor_losses = losses.new_zeros(batch_size)
        if not isinstance(species_anchor_losses, torch.Tensor):
            raise TypeError("counterfactual species anchor losses must be a tensor")
        if tuple(species_anchor_losses.shape) != (batch_size,):
            raise ValueError(
                "counterfactual species anchor losses must have shape (B,)"
            )
        return (
            losses.float(),
            mask.to(device=losses.device, dtype=losses.dtype),
            classification_losses.float(),
            consistency_losses.float(),
            species_anchor_losses.float(),
        )

    def _counterfactual_alignment(
        self,
        query: Dict[str, torch.Tensor],
        params: Dict[str, torch.Tensor],
        q_part: torch.Tensor,
        q_relation: torch.Tensor,
        relation_weight: float,
    ) -> torch.Tensor:
        """Per-example fixed-reference semantic loss for one virtual branch."""
        query_tokens = query["part_tokens"].detach().float()
        query_target = query["target_semantics"].detach().float()
        part_error = self.alignment_error(query_tokens, query_target, params)
        part_loss = (q_part * part_error).sum(dim=1)

        query_relation = query["relation_state"]
        relation_tokens = self._relation_tokens(query_relation, params)
        relation_error = self.relation_alignment_error(
            relation_tokens,
            query_relation["target_semantics"].detach().float(),
        )
        relation_loss = (q_relation * relation_error).sum(dim=1)
        normalizer = 1.0 + float(relation_weight)
        return (
            part_loss + float(relation_weight) * relation_loss
        ) / max(normalizer, 1.0)

    def _counterfactual_meta_objective(
        self,
        support: Dict[str, torch.Tensor],
        query: Dict[str, torch.Tensor],
        inner_lr: float,
        q_mode: str,
        kl_weight: float,
        relation_weight: float,
        outer_task_fn: Callable,
        task_weight: float,
        semantic_weight: float,
        router_kl_weight: float,
        task_level_weights: Tuple[float, float, float],
        router_advantage_scale: float,
        normalize_inner_grad: bool,
        safe_improvement_margin: float,
        normalize_router_regret: bool,
        router_regret_floor: float,
        safe_route_budget: float,
        safe_confidence_scale: float,
        safe_confidence_budget: bool,
        consistency_credit_weight: float,
        counterfactual_compose: str,
        relation_residual_inner_scale: float,
        species_no_regret_margin: float,
        species_anchor_kl_margin: float,
        safe_gate: bool,
    ):
        """Three virtual branches evaluated independently at all three levels."""
        if not self.enable_relations:
            raise RuntimeError("counterfactual routing requires relation evidence")
        if self.router.num_tasks != 3:
            raise RuntimeError(
                "controller was not constructed with routing_scope='counterfactual'"
            )
        if outer_task_fn is None:
            raise RuntimeError("counterfactual routing requires an outer task evaluator")
        if router_advantage_scale < 0:
            raise ValueError("router_advantage_scale must be nonnegative")
        if safe_improvement_margin < 0:
            raise ValueError("safe improvement margin must be nonnegative")
        if router_regret_floor <= 0:
            raise ValueError("router regret floor must be positive")
        if not 0.0 <= float(safe_route_budget) <= 1.0:
            raise ValueError("safe route budget must be in [0, 1]")
        if safe_confidence_scale <= 0:
            raise ValueError("safe confidence scale must be positive")
        if consistency_credit_weight < 0:
            raise ValueError("consistency credit weight must be nonnegative")
        if counterfactual_compose not in ("competitive", "residual"):
            raise ValueError(
                "counterfactual compose must be 'competitive' or 'residual'"
            )
        if relation_residual_inner_scale <= 0:
            raise ValueError("relation residual inner scale must be positive")
        if species_no_regret_margin < 0:
            raise ValueError("species no-regret margin must be nonnegative")
        if species_anchor_kl_margin < 0:
            raise ValueError("species anchor KL margin must be nonnegative")

        level_weights = torch.as_tensor(
            task_level_weights, dtype=torch.float32,
            device=support["part_tokens"].device,
        )
        if level_weights.numel() != 3 or (level_weights < 0).any():
            raise ValueError("task_level_weights must contain three nonnegative values")
        if float(level_weights.sum()) <= 0:
            raise ValueError("at least one task-level weight must be positive")

        support_tokens = support["part_tokens"].detach().float()
        support_target = support["target_semantics"].detach().float()
        support_curvature = support["curvature"].detach().float()
        base_params = OrderedDict(self.adapter.named_parameters())
        support_adapted = self.adapt_parts(support_tokens, base_params)

        part_error = self._direct_alignment_error(support_adapted, support_target)
        part_policy, part_policy_stats = self.policy(
            support_tokens,
            support["policy_semantics"].detach().float(),
            support_curvature,
        )
        part_inner_per_example = (
            self.num_parts * part_policy * part_error
        ).mean(dim=1)
        part_kl = self._policy_kl(part_policy)

        support_relation = support["relation_state"]
        base_relation_tokens = self.relation_encoder.visual_relations(
            support_adapted,
            support_relation["part_attn"].detach().float(),
        )[0]
        base_relation_error = self.relation_alignment_error(
            base_relation_tokens,
            support_relation["target_semantics"].detach().float(),
        )
        relation_curvature = support_relation["curvature"].detach().float()

        # Part is always unrolled from the base adapter.  In competitive mode
        # Relation is also unrolled from the base (exact V8.5 behavior).  In
        # residual mode it is unrolled from the Part state and therefore
        # measures only the incremental value of adding Relation after Part.
        base_values = tuple(base_params.values())
        part_grads = torch.autograd.grad(
            part_inner_per_example.mean(), base_values,
            create_graph=True, retain_graph=True, allow_unused=False,
        )
        part_fast_params, part_grad_norm, part_step_norm = (
            self._counterfactual_fast_params(
                base_params, part_grads, inner_lr, normalize_inner_grad
            )
        )
        if counterfactual_compose == "residual":
            relation_source_params = part_fast_params
            relation_source_adapted = self.adapt_parts(
                support_tokens, relation_source_params
            )
            relation_tokens = self.relation_encoder.visual_relations(
                relation_source_adapted,
                support_relation["part_attn"].detach().float(),
            )[0]
            relation_step_lr = (
                float(inner_lr) * float(relation_residual_inner_scale)
            )
        else:
            relation_source_params = base_params
            relation_tokens = base_relation_tokens
            relation_step_lr = float(inner_lr)

        relation_error = self.relation_alignment_error(
            relation_tokens,
            support_relation["target_semantics"].detach().float(),
        )
        relation_policy, relation_policy_stats = self.relation_policy(
            relation_tokens.detach(),
            support_relation["policy_semantics"].detach().float(),
            relation_curvature,
        )
        relation_inner_per_example = (
            self.num_relations * relation_policy * relation_error
        ).mean(dim=1)
        relation_kl = self._policy_kl(relation_policy)
        relation_grads = torch.autograd.grad(
            float(relation_weight) * relation_inner_per_example.mean(),
            tuple(relation_source_params.values()),
            create_graph=True, allow_unused=False,
        )
        relation_fast_params, relation_grad_norm, relation_step_norm = (
            self._counterfactual_fast_params(
                relation_source_params,
                relation_grads,
                relation_step_lr,
                normalize_inner_grad,
            )
        )
        branch_params = (
            base_params,
            part_fast_params,
            relation_fast_params,
        )

        route, route_stats = self.router(
            part_error,
            base_relation_error,
            support_curvature,
            relation_curvature,
        )

        # A single stopped reference is shared by all branches; otherwise a
        # branch could look better merely by changing how it is evaluated.
        query_tokens = query["part_tokens"].detach().float()
        query_target = query["target_semantics"].detach().float()
        query_part_base = self.alignment_error(
            query_tokens, query_target, base_params
        )
        q_part = self.reference_distribution(
            query_part_base, query["curvature"].detach().float(), q_mode
        )
        query_relation = query["relation_state"]
        query_relation_base_tokens = self._relation_tokens(
            query_relation, base_params
        )
        query_relation_base = self.relation_alignment_error(
            query_relation_base_tokens,
            query_relation["target_semantics"].detach().float(),
        )
        q_relation = self.reference_distribution(
            query_relation_base,
            query_relation["curvature"].detach().float(),
            q_mode,
        )

        task_branches = []
        classification_branches = []
        consistency_branches = []
        species_anchor_branches = []
        align_branches = []
        task_mask = None
        batch_size = support_tokens.size(0)
        for params in branch_params:
            (
                task_losses,
                branch_mask,
                classification_losses,
                consistency_losses,
                species_anchor_losses,
            ) = self._task_output(
                outer_task_fn(query, params, q_part), batch_size
            )
            if task_mask is None:
                task_mask = branch_mask
            elif not torch.equal(task_mask.bool(), branch_mask.bool()):
                raise ValueError("all counterfactual branches must use the same task mask")
            task_branches.append(task_losses)
            classification_branches.append(classification_losses)
            consistency_branches.append(consistency_losses)
            species_anchor_branches.append(species_anchor_losses)
            align_branches.append(self._counterfactual_alignment(
                query, params, q_part, q_relation, relation_weight
            ))

        # (B, task, branch), with branch order skip/part/relation.
        branch_task_loss = torch.stack(task_branches, dim=-1)
        branch_classification_loss = torch.stack(
            classification_branches, dim=-1
        )
        branch_consistency_loss = torch.stack(
            consistency_branches, dim=-1
        )
        branch_species_anchor_loss = torch.stack(
            species_anchor_branches, dim=-1
        )
        branch_align_loss = torch.stack(align_branches, dim=-1)
        weighted_mask = task_mask * level_weights.view(1, 3)

        def weighted_task_reduce(value: torch.Tensor) -> torch.Tensor:
            task_means = (
                (value * task_mask).sum(dim=0)
                / task_mask.sum(dim=0).clamp_min(1.0)
            )
            return (level_weights * task_means).sum()

        # Classification and hierarchy consistency have very different raw
        # scales.  Calibrate their branch regrets independently before adding
        # consistency credit; otherwise CE can hide a relation update that
        # improves Species/Family/Order agreement.
        skip_classification = branch_classification_loss[:, :, 0].detach()
        skip_consistency = branch_consistency_loss[:, :, 0].detach()
        if counterfactual_compose == "residual":
            classification_baseline = torch.stack(
                (
                    skip_classification,
                    skip_classification,
                    branch_classification_loss[:, :, 1].detach(),
                ),
                dim=-1,
            )
            consistency_baseline = torch.stack(
                (
                    skip_consistency,
                    skip_consistency,
                    branch_consistency_loss[:, :, 1].detach(),
                ),
                dim=-1,
            )
        else:
            classification_baseline = skip_classification.unsqueeze(-1)
            consistency_baseline = skip_consistency.unsqueeze(-1)
        classification_relative_regret = (
            branch_classification_loss - classification_baseline
        ) / classification_baseline.abs().clamp_min(1.0e-3)
        classification_relative_regret = classification_relative_regret.clamp(
            min=-10.0, max=10.0
        )
        consistency_relative_regret = (
            branch_consistency_loss - consistency_baseline
        ) / consistency_baseline.abs().clamp_min(1.0e-3)
        consistency_relative_regret = consistency_relative_regret.clamp(
            min=-10.0, max=10.0
        )
        (
            calibrated_classification_regret,
            classification_regret_rms,
        ) = self._calibrate_counterfactual_regret(
            classification_relative_regret,
            improvement_margin=safe_improvement_margin,
            scale_floor=router_regret_floor,
            normalize=normalize_router_regret,
        )
        (
            calibrated_consistency_regret,
            consistency_regret_rms,
        ) = self._calibrate_counterfactual_regret(
            consistency_relative_regret,
            improvement_margin=0.0,
            scale_floor=router_regret_floor,
            normalize=normalize_router_regret,
        )
        calibrated_regret = (
            calibrated_classification_regret
            + float(consistency_credit_weight)
            * calibrated_consistency_regret
        )
        relative_regret = (
            classification_relative_regret
            + float(consistency_credit_weight) * consistency_relative_regret
        )
        regret_rms = torch.sqrt(
            classification_regret_rms.square()
            + (
                float(consistency_credit_weight) * consistency_regret_rms
            ).square()
        )
        if normalize_router_regret:
            confidence_regret = calibrated_regret
        else:
            normalized_classification_regret, _ = (
                self._calibrate_counterfactual_regret(
                    classification_relative_regret,
                    improvement_margin=safe_improvement_margin,
                    scale_floor=router_regret_floor,
                    normalize=True,
                )
            )
            normalized_consistency_regret, _ = (
                self._calibrate_counterfactual_regret(
                    consistency_relative_regret,
                    improvement_margin=0.0,
                    scale_floor=router_regret_floor,
                    normalize=True,
                )
            )
            confidence_regret = (
                normalized_classification_regret
                + float(consistency_credit_weight)
                * normalized_consistency_regret
            )
        # The stopped gate owns the Skip decision.  The learned router is
        # conditioned on semantic updates and can only choose Part versus
        # Relation.  This prevents expected-regret training from escaping into
        # the global all-Skip solution observed in V8.1/V8.2.
        combined_gain = -calibrated_regret.detach()[:, :, 1:]
        branch_confidence = (
            -confidence_regret.detach()[:, :, 1:]
        ).clamp_min(0.0)
        branch_eligibility = (
            (combined_gain > 0.0)
            & task_mask.bool().unsqueeze(-1)
        )
        species_no_regret_guard = torch.ones(
            batch_size, dtype=torch.bool, device=task_mask.device
        )
        species_supervised_guard = species_no_regret_guard
        species_anchor_guard = species_no_regret_guard
        if counterfactual_compose == "residual":
            # The residual branch is admissible only on top of a safe Part
            # branch and may not materially degrade Species relative to Part.
            # Where a Species label is unavailable, stopped KL to the base
            # prediction supplies a label-free trust-region guard.
            species_available = task_mask[:, 0].bool()
            part_species = branch_classification_loss[:, 0, 1].detach()
            residual_species = branch_classification_loss[:, 0, 2].detach()
            species_relative_delta = (
                residual_species - part_species
            ) / part_species.abs().clamp_min(1.0e-3)
            species_supervised_guard = (
                (~species_available)
                | (species_relative_delta <= float(species_no_regret_margin))
            )
            species_anchor_delta = (
                branch_species_anchor_loss[:, 2].detach()
                - branch_species_anchor_loss[:, 1].detach()
            )
            species_anchor_guard = (
                species_anchor_delta <= float(species_anchor_kl_margin)
            )
            species_no_regret_guard = (
                species_supervised_guard & species_anchor_guard
            )
            branch_eligibility[:, :, 1] = (
                branch_eligibility[:, :, 1]
                & branch_eligibility[:, :, 0]
                & species_no_regret_guard.unsqueeze(1)
            )
        route_eligibility = branch_eligibility if safe_gate else None
        route_confidence = (
            branch_confidence
            if safe_gate and safe_confidence_budget
            else None
        )
        (
            two_stage_route,
            conditional_route,
            has_eligible,
            effective_budget,
        ) = self._two_stage_counterfactual_route(
            route,
            route_eligibility,
            non_skip_budget=safe_route_budget,
            branch_confidence=route_confidence,
            confidence_scale=safe_confidence_scale,
        )
        best_branch = calibrated_regret.detach().argmin(dim=-1)

        expected_task_per_level = (
            two_stage_route * branch_task_loss
        ).sum(dim=-1)
        outer_task = weighted_task_reduce(expected_task_per_level)
        task_before = weighted_task_reduce(branch_task_loss[:, :, 0])
        raw_expected_regret = (
            two_stage_route * relative_regret
        ).sum(dim=-1)
        raw_router_objective = weighted_task_reduce(raw_expected_regret)
        expected_regret = (
            two_stage_route * calibrated_regret
        ).sum(dim=-1)
        router_objective = weighted_task_reduce(expected_regret)

        # Convert task-specific routes into one per-example semantic route only
        # for the auxiliary alignment evaluator.  Classification routing remains
        # fully task-specific above.
        route_weight = weighted_mask / weighted_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        semantic_route = (
            route_weight.unsqueeze(-1) * two_stage_route
        ).sum(dim=1)
        outer_align = (semantic_route * branch_align_loss).sum(dim=-1).mean()
        align_before = branch_align_loss[:, 0].mean()

        prior = self.router.prior.to(route)
        conditional_prior = prior[1:] / prior[1:].sum().clamp_min(1.0e-8)
        conditional_kl = (
            conditional_route * (
                conditional_route.clamp_min(1.0e-8).log()
                - conditional_prior.clamp_min(1.0e-8).log()
            )
        ).sum(dim=-1)
        router_kl = weighted_task_reduce(conditional_kl)
        meta_loss = (
            float(task_weight) * (
                outer_task + float(router_advantage_scale) * router_objective
            )
            + float(semantic_weight) * outer_align
            + float(kl_weight) * (part_kl + relation_kl)
            + float(router_kl_weight) * router_kl
        )

        def masked_level_mean(value: torch.Tensor, task_idx: int) -> torch.Tensor:
            mask = task_mask[:, task_idx]
            return (value[:, task_idx] * mask).sum() / mask.sum().clamp_min(1.0)

        route_denominator = task_mask.sum().clamp_min(1.0)

        stats: Dict[str, torch.Tensor] = {
            "meta_loss": meta_loss.detach(),
            "meta_inner_align": (
                0.5 * (
                    part_inner_per_example.mean()
                    + float(relation_weight) * relation_inner_per_example.mean()
                )
            ).detach(),
            "meta_outer_task": outer_task.detach(),
            "meta_task_improvement": (task_before - outer_task).detach(),
            "meta_task_improvement_x1e4": (
                task_before - outer_task
            ).detach() * 1.0e4,
            "meta_router_objective": router_objective.detach(),
            "meta_router_raw_objective": raw_router_objective.detach(),
            "meta_router_regret_rms": (
                regret_rms.squeeze(-1) * task_mask
            ).sum().div(task_mask.sum().clamp_min(1.0)).detach(),
            "meta_router_classification_regret_rms": (
                classification_regret_rms.squeeze(-1) * task_mask
            ).sum().div(task_mask.sum().clamp_min(1.0)).detach(),
            "meta_router_consistency_regret_rms": (
                consistency_regret_rms.squeeze(-1) * task_mask
            ).sum().div(task_mask.sum().clamp_min(1.0)).detach(),
            "meta_router_consistency_credit_weight": calibrated_regret.new_tensor(
                float(consistency_credit_weight)
            ).detach(),
            "counterfactual_residual_mode": calibrated_regret.new_tensor(
                float(counterfactual_compose == "residual")
            ).detach(),
            "meta_relation_residual_inner_scale": calibrated_regret.new_tensor(
                float(relation_residual_inner_scale)
            ).detach(),
            "meta_species_no_regret_accept_rate": (
                species_no_regret_guard.float().mean().detach()
            ),
            "meta_species_supervised_guard_rate": (
                species_supervised_guard.float().mean().detach()
            ),
            "meta_species_anchor_guard_rate": (
                species_anchor_guard.float().mean().detach()
            ),
            "meta_router_calibrated_abs": (
                calibrated_regret.detach().abs().mean(dim=-1) * task_mask
            ).sum().div(task_mask.sum().clamp_min(1.0)).detach(),
            "meta_part_inner_grad_norm": part_grad_norm.detach(),
            "meta_relation_inner_grad_norm": relation_grad_norm.detach(),
            "meta_part_inner_step_norm": part_step_norm.detach(),
            "meta_relation_inner_step_norm": relation_step_norm.detach(),
            "meta_outer_align": outer_align.detach(),
            "meta_improvement": (align_before - outer_align).detach(),
            "meta_policy_kl": (part_kl + relation_kl).detach(),
            "meta_router_kl": router_kl.detach(),
            "two_stage_route_skip": (
                two_stage_route[:, :, 0] * task_mask
            ).sum().div(route_denominator).detach(),
            "two_stage_route_part": (
                two_stage_route[:, :, 1] * task_mask
            ).sum().div(route_denominator).detach(),
            "two_stage_route_relation": (
                two_stage_route[:, :, 2] * task_mask
            ).sum().div(route_denominator).detach(),
            "two_stage_active_rate": (
                has_eligible.float() * task_mask
            ).sum().div(route_denominator).detach(),
            "two_stage_effective_budget": (
                effective_budget * task_mask
            ).sum().div(route_denominator).detach(),
            "two_stage_gain_confidence": (
                branch_confidence.max(dim=-1).values * task_mask
            ).sum().div(route_denominator).detach(),
        }
        task_names = AdaptiveGranularityRouter.TASK_NAMES
        for task_idx, task_name in enumerate(task_names):
            skip_value = masked_level_mean(branch_task_loss[:, :, 0], task_idx)
            part_value = masked_level_mean(branch_task_loss[:, :, 1], task_idx)
            relation_value = masked_level_mean(branch_task_loss[:, :, 2], task_idx)
            skip_classification_value = masked_level_mean(
                branch_classification_loss[:, :, 0], task_idx
            )
            part_classification_value = masked_level_mean(
                branch_classification_loss[:, :, 1], task_idx
            )
            relation_classification_value = masked_level_mean(
                branch_classification_loss[:, :, 2], task_idx
            )
            skip_consistency_value = masked_level_mean(
                branch_consistency_loss[:, :, 0], task_idx
            )
            part_consistency_value = masked_level_mean(
                branch_consistency_loss[:, :, 1], task_idx
            )
            relation_consistency_value = masked_level_mean(
                branch_consistency_loss[:, :, 2], task_idx
            )
            stats.update({
                f"meta_{task_name}_skip_task": skip_value.detach(),
                f"meta_{task_name}_part_task": part_value.detach(),
                f"meta_{task_name}_relation_task": relation_value.detach(),
                f"meta_{task_name}_part_improvement": (
                    skip_value - part_value
                ).detach(),
                f"meta_{task_name}_relation_improvement": (
                    skip_value - relation_value
                ).detach(),
                f"meta_{task_name}_relation_incremental_improvement": (
                    part_value - relation_value
                ).detach(),
                f"meta_{task_name}_part_classification_improvement": (
                    skip_classification_value - part_classification_value
                ).detach(),
                f"meta_{task_name}_relation_classification_improvement": (
                    skip_classification_value - relation_classification_value
                ).detach(),
                f"meta_{task_name}_relation_incremental_classification_improvement": (
                    part_classification_value - relation_classification_value
                ).detach(),
                f"meta_{task_name}_part_consistency_improvement": (
                    skip_consistency_value - part_consistency_value
                ).detach(),
                f"meta_{task_name}_relation_consistency_improvement": (
                    skip_consistency_value - relation_consistency_value
                ).detach(),
                f"meta_{task_name}_relation_incremental_consistency_improvement": (
                    part_consistency_value - relation_consistency_value
                ).detach(),
                f"safe_{task_name}_part_accept_rate": (
                    branch_eligibility[:, task_idx, 0].float()
                    * task_mask[:, task_idx]
                ).sum().div(task_mask[:, task_idx].sum().clamp_min(1.0)).detach(),
                f"safe_{task_name}_relation_accept_rate": (
                    branch_eligibility[:, task_idx, 1].float()
                    * task_mask[:, task_idx]
                ).sum().div(task_mask[:, task_idx].sum().clamp_min(1.0)).detach(),
                f"conditional_{task_name}_part": masked_level_mean(
                    conditional_route[:, :, 0], task_idx
                ).detach(),
                f"conditional_{task_name}_relation": masked_level_mean(
                    conditional_route[:, :, 1], task_idx
                ).detach(),
                f"two_stage_{task_name}_skip": masked_level_mean(
                    two_stage_route[:, :, 0], task_idx
                ).detach(),
                f"two_stage_{task_name}_part": masked_level_mean(
                    two_stage_route[:, :, 1], task_idx
                ).detach(),
                f"two_stage_{task_name}_relation": masked_level_mean(
                    two_stage_route[:, :, 2], task_idx
                ).detach(),
                f"two_stage_{task_name}_active_rate": masked_level_mean(
                    has_eligible.float(), task_idx
                ).detach(),
                f"two_stage_{task_name}_effective_budget": masked_level_mean(
                    effective_budget, task_idx
                ).detach(),
                f"two_stage_{task_name}_gain_confidence": masked_level_mean(
                    branch_confidence.max(dim=-1).values, task_idx
                ).detach(),
            })
            for branch_idx, branch_name in enumerate(
                ("skip", "part", "relation")
            ):
                stats[f"candidate_{task_name}_{branch_name}_rate"] = (
                    best_branch[:, task_idx].eq(branch_idx).float()
                    * task_mask[:, task_idx]
                ).sum().div(
                    task_mask[:, task_idx].sum().clamp_min(1.0)
                ).detach()
        part_task = weighted_task_reduce(branch_task_loss[:, :, 1])
        relation_task = weighted_task_reduce(branch_task_loss[:, :, 2])
        classification_before = weighted_task_reduce(
            branch_classification_loss[:, :, 0]
        )
        part_classification = weighted_task_reduce(
            branch_classification_loss[:, :, 1]
        )
        relation_classification = weighted_task_reduce(
            branch_classification_loss[:, :, 2]
        )
        consistency_before = weighted_task_reduce(
            branch_consistency_loss[:, :, 0]
        )
        part_consistency = weighted_task_reduce(
            branch_consistency_loss[:, :, 1]
        )
        relation_consistency = weighted_task_reduce(
            branch_consistency_loss[:, :, 2]
        )
        stats.update({
            "meta_part_task_improvement": (task_before - part_task).detach(),
            "meta_relation_task_improvement": (
                task_before - relation_task
            ).detach(),
            "meta_relation_incremental_task_improvement": (
                part_task - relation_task
            ).detach(),
            "meta_part_classification_improvement": (
                classification_before - part_classification
            ).detach(),
            "meta_relation_classification_improvement": (
                classification_before - relation_classification
            ).detach(),
            "meta_relation_incremental_classification_improvement": (
                part_classification - relation_classification
            ).detach(),
            "meta_part_consistency_improvement": (
                consistency_before - part_consistency
            ).detach(),
            "meta_relation_consistency_improvement": (
                consistency_before - relation_consistency
            ).detach(),
            "meta_relation_incremental_consistency_improvement": (
                part_consistency - relation_consistency
            ).detach(),
        })
        stats.update(route_stats)
        stats.update({"part_" + key: value for key, value in part_policy_stats.items()})
        stats.update({"rel_" + key: value for key, value in relation_policy_stats.items()})
        return meta_loss, stats, {
            "branch_eligibility": branch_eligibility,
            "branch_confidence": branch_confidence,
            "species_no_regret_guard": species_no_regret_guard,
        }

    def meta_objective(
        self, support: Dict[str, torch.Tensor], query: Dict[str, torch.Tensor],
        inner_lr: float, q_mode: str = "uniform", kl_weight: float = 0.01,
        scope: str = "adaptive", relation_weight: float = 1.0,
        outer_task_fn: Optional[
            Callable[[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor], torch.Tensor]
        ] = None,
        task_weight: float = 1.0, semantic_weight: float = 0.1,
        router_kl_weight: float = 0.001,
        task_level_weights: Tuple[float, float, float] = (1.0, 0.5, 0.5),
        router_advantage_scale: float = 100.0,
        normalize_inner_grad: bool = False,
        safe_improvement_margin: float = 0.0,
        normalize_router_regret: bool = False,
        router_regret_floor: float = 1.0e-4,
        safe_route_budget: float = 0.05,
        safe_confidence_scale: float = 1.0,
        safe_confidence_budget: bool = True,
        consistency_credit_weight: float = 0.25,
        counterfactual_compose: str = "competitive",
        relation_residual_inner_scale: float = 0.5,
        species_no_regret_margin: float = 0.01,
        species_anchor_kl_margin: float = 0.01,
        safe_gate: bool = True,
        return_aux: bool = False,
    ):
        """Joint one-step inner update with a task-feedback query evaluator.

        Part/relation item policies and the granularity route influence only the
        support update of the shared adapter.  The query task uses a stopped
        reference distribution, so no learned weight multiplies its raw error.
        """
        self._check_scope(scope)
        if scope == "counterfactual":
            result = self._counterfactual_meta_objective(
                support=support,
                query=query,
                inner_lr=inner_lr,
                q_mode=q_mode,
                kl_weight=kl_weight,
                relation_weight=relation_weight,
                outer_task_fn=outer_task_fn,
                task_weight=task_weight,
                semantic_weight=semantic_weight,
                router_kl_weight=router_kl_weight,
                task_level_weights=task_level_weights,
                router_advantage_scale=router_advantage_scale,
                normalize_inner_grad=normalize_inner_grad,
                safe_improvement_margin=safe_improvement_margin,
                normalize_router_regret=normalize_router_regret,
                router_regret_floor=router_regret_floor,
                safe_route_budget=safe_route_budget,
                safe_confidence_scale=safe_confidence_scale,
                safe_confidence_budget=safe_confidence_budget,
                consistency_credit_weight=consistency_credit_weight,
                counterfactual_compose=counterfactual_compose,
                relation_residual_inner_scale=relation_residual_inner_scale,
                species_no_regret_margin=species_no_regret_margin,
                species_anchor_kl_margin=species_anchor_kl_margin,
                safe_gate=safe_gate,
            )
            return result if return_aux else result[:2]
        use_part = scope in ("part", "hybrid", "adaptive")
        use_relation = scope in ("relation", "hybrid", "adaptive")
        if use_relation and not self.enable_relations:
            raise RuntimeError("relation meta scope requested but disabled")

        support_tokens = support["part_tokens"].detach().float()
        support_target = support["target_semantics"].detach().float()
        support_curvature = support["curvature"].detach().float()
        base_params = OrderedDict(self.adapter.named_parameters())
        support_adapted = self.adapt_parts(support_tokens, base_params)
        part_error = self._direct_alignment_error(support_adapted, support_target)

        part_policy_stats: Dict[str, torch.Tensor] = {}
        if use_part:
            part_policy, part_policy_stats = self.policy(
                support_tokens,
                support["policy_semantics"].detach().float(),
                support_curvature,
            )
            part_inner = (self.num_parts * part_policy * part_error).mean(dim=1)
            part_kl = self._policy_kl(part_policy)
        else:
            part_inner = part_error.new_zeros(part_error.size(0))
            part_kl = part_error.new_zeros(())

        relation_error = None
        relation_curvature = None
        relation_policy_stats: Dict[str, torch.Tensor] = {}
        if use_relation:
            support_relation = support["relation_state"]
            relation_tokens = self.relation_encoder.visual_relations(
                support_adapted,
                support_relation["part_attn"].detach().float(),
            )[0]
            relation_error = self.relation_alignment_error(
                relation_tokens,
                support_relation["target_semantics"].detach().float(),
            )
            relation_curvature = support_relation["curvature"].detach().float()
            relation_policy, relation_policy_stats = self.relation_policy(
                relation_tokens.detach(),
                support_relation["policy_semantics"].detach().float(),
                relation_curvature,
            )
            relation_inner = (
                self.num_relations * relation_policy * relation_error
            ).mean(dim=1)
            relation_kl = self._policy_kl(relation_policy)
        else:
            relation_inner = part_inner.new_zeros(part_inner.shape)
            relation_kl = part_error.new_zeros(())

        route, route_stats = self._route(
            scope, part_error, relation_error, support_curvature,
            relation_curvature,
        )
        inner_loss = (
            route[:, 1] * part_inner
            + route[:, 2] * float(relation_weight) * relation_inner
        ).mean()
        inner_grads = torch.autograd.grad(
            inner_loss, tuple(base_params.values()), create_graph=True,
            allow_unused=False,
        )
        fast_params = OrderedDict(
            (name, param - float(inner_lr) * grad)
            for (name, param), grad in zip(base_params.items(), inner_grads)
        )

        query_tokens = query["part_tokens"].detach().float()
        query_target = query["target_semantics"].detach().float()
        query_curvature = query["curvature"].detach().float()
        part_before = self.alignment_error(query_tokens, query_target, base_params)
        part_after = self.alignment_error(query_tokens, query_target, fast_params)
        q_part = self.reference_distribution(part_after, query_curvature, q_mode)
        part_outer = (q_part * part_after).sum(dim=1).mean()
        part_before_value = (q_part * part_before.detach()).sum(dim=1).mean()

        outer_align = part_outer.new_zeros(())
        align_before = part_outer.new_zeros(())
        normalizer = 0.0
        if use_part:
            outer_align = outer_align + part_outer
            align_before = align_before + part_before_value
            normalizer += 1.0
        if use_relation:
            query_relation = query["relation_state"]
            relation_before_tokens = self._relation_tokens(query_relation, base_params)
            relation_after_tokens = self._relation_tokens(query_relation, fast_params)
            relation_before = self.relation_alignment_error(
                relation_before_tokens,
                query_relation["target_semantics"].detach().float(),
            )
            relation_after = self.relation_alignment_error(
                relation_after_tokens,
                query_relation["target_semantics"].detach().float(),
            )
            q_relation = self.reference_distribution(
                relation_after,
                query_relation["curvature"].detach().float(),
                q_mode,
            )
            outer_align = outer_align + float(relation_weight) * (
                q_relation * relation_after
            ).sum(dim=1).mean()
            align_before = align_before + float(relation_weight) * (
                q_relation * relation_before.detach()
            ).sum(dim=1).mean()
            normalizer += float(relation_weight)
        outer_align = outer_align / max(normalizer, 1.0)
        align_before = align_before / max(normalizer, 1.0)

        outer_task = outer_align.new_zeros(())
        task_before = outer_align.new_zeros(())
        if outer_task_fn is not None:
            outer_task = outer_task_fn(query, fast_params, q_part)
            with torch.no_grad():
                task_before = outer_task_fn(query, base_params, q_part).detach()

        router_kl = outer_align.new_zeros(())
        if scope == "adaptive":
            prior = self.router.prior.to(route)
            router_kl = (
                route * (route.clamp_min(1e-8).log() - prior.log())
            ).sum(dim=1).mean()
        meta_loss = (
            float(task_weight) * outer_task
            + float(semantic_weight) * outer_align
            + float(kl_weight) * (part_kl + relation_kl)
            + float(router_kl_weight) * router_kl
        )
        stats: Dict[str, torch.Tensor] = {
            "meta_loss": meta_loss.detach(),
            "meta_inner_align": inner_loss.detach(),
            "meta_outer_task": outer_task.detach(),
            "meta_task_improvement": task_before - outer_task.detach(),
            "meta_task_improvement_x1e4": (
                task_before - outer_task.detach()
            ) * 1.0e4,
            "meta_outer_align": outer_align.detach(),
            "meta_improvement": align_before - outer_align.detach(),
            "meta_policy_kl": (part_kl + relation_kl).detach(),
            "meta_router_kl": router_kl.detach(),
        }
        stats.update(route_stats)
        stats.update({"part_" + key: value for key, value in part_policy_stats.items()})
        stats.update({"rel_" + key: value for key, value in relation_policy_stats.items()})
        if return_aux:
            return meta_loss, stats, {}
        return meta_loss, stats

    def real_weighted_alignment(
        self, state: Dict[str, torch.Tensor], scope: str = "adaptive",
        relation_weight: float = 1.0,
        task_level_weights: Tuple[float, float, float] = (1.0, 0.5, 0.5),
        branch_eligibility: Optional[torch.Tensor] = None,
        branch_confidence: Optional[torch.Tensor] = None,
        safe_route_budget: float = 0.05,
        safe_confidence_scale: float = 1.0,
        counterfactual_compose: str = "competitive",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Real shared-adapter loss with every learned policy detached."""
        self._check_scope(scope)
        use_part = scope in ("part", "hybrid", "adaptive", "counterfactual")
        use_relation = scope in (
            "relation", "hybrid", "adaptive", "counterfactual"
        )
        adapted = self.adapt_parts(state["part_tokens"])
        part_error = self._direct_alignment_error(
            adapted, state["target_semantics"]
        )
        if use_part:
            part_policy, part_stats = self.policy_distribution(
                state["part_tokens"], state["policy_semantics"],
                state["curvature"],
            )
            part_loss = (
                self.num_parts * part_policy.detach() * part_error
            ).mean(dim=1)
        else:
            part_loss = part_error.new_zeros(part_error.size(0))
            part_stats = {}

        relation_error = None
        relation_curvature = None
        if use_relation:
            relation_state = state["relation_state"]
            relation_tokens = self.relation_encoder.visual_relations(
                adapted,
                relation_state["part_attn"].to(dtype=adapted.dtype),
            )[0]
            relation_error = self.relation_alignment_error(
                relation_tokens, relation_state["target_semantics"]
            )
            relation_curvature = relation_state["curvature"]
            relation_policy, relation_stats = self._stopped_policy_distribution(
                self.relation_policy,
                relation_tokens,
                relation_state["policy_semantics"],
                relation_curvature,
            )
            relation_loss = (
                self.num_relations * relation_policy.detach() * relation_error
            ).mean(dim=1)
        else:
            relation_loss = part_loss.new_zeros(part_loss.shape)
            relation_stats = {}

        if scope == "counterfactual":
            if self.router.num_tasks != 3:
                raise RuntimeError(
                    "controller was not constructed with "
                    "routing_scope='counterfactual'"
                )
            task_route, route_stats = self.router(
                part_error.detach(), relation_error.detach(),
                state["curvature"].detach(), relation_curvature.detach(),
            )
            (
                task_route,
                conditional_route,
                has_eligible,
                effective_budget,
            ) = self._two_stage_counterfactual_route(
                task_route,
                branch_eligibility,
                non_skip_budget=safe_route_budget,
                branch_confidence=branch_confidence,
                confidence_scale=safe_confidence_scale,
            )
            safe_entropy = -(
                task_route * task_route.clamp_min(1.0e-8).log()
            ).sum(dim=-1)
            route_stats.update({
                "safe_route_skip": task_route[:, :, 0].mean().detach(),
                "safe_route_part": task_route[:, :, 1].mean().detach(),
                "safe_route_relation": task_route[:, :, 2].mean().detach(),
                "safe_route_entropy": safe_entropy.mean().detach(),
                "safe_route_active_rate": has_eligible.float().mean().detach(),
                "safe_route_effective_budget": effective_budget.mean().detach(),
                "conditional_route_part": (
                    conditional_route[:, :, 0].mean().detach()
                ),
                "conditional_route_relation": (
                    conditional_route[:, :, 1].mean().detach()
                ),
            })
            for task_idx, task_name in enumerate(
                AdaptiveGranularityRouter.TASK_NAMES
            ):
                route_stats.update({
                    f"safe_route_{task_name}_skip": (
                        task_route[:, task_idx, 0].mean().detach()
                    ),
                    f"safe_route_{task_name}_part": (
                        task_route[:, task_idx, 1].mean().detach()
                    ),
                    f"safe_route_{task_name}_relation": (
                        task_route[:, task_idx, 2].mean().detach()
                    ),
                    f"safe_route_{task_name}_active_rate": (
                        has_eligible[:, task_idx].float().mean().detach()
                    ),
                    f"safe_route_{task_name}_effective_budget": (
                        effective_budget[:, task_idx].mean().detach()
                    ),
                    f"conditional_route_{task_name}_part": (
                        conditional_route[:, task_idx, 0].mean().detach()
                    ),
                    f"conditional_route_{task_name}_relation": (
                        conditional_route[:, task_idx, 1].mean().detach()
                    ),
                })
            level_weights = torch.as_tensor(
                task_level_weights, device=task_route.device,
                dtype=task_route.dtype,
            )
            if level_weights.numel() != 3 or (level_weights < 0).any():
                raise ValueError(
                    "task_level_weights must contain three nonnegative values"
                )
            route = (
                task_route * level_weights.view(1, 3, 1)
            ).sum(dim=1) / level_weights.sum().clamp_min(1.0e-8)
            part_route_weight, relation_route_weight = (
                self._real_alignment_route_weights(
                    route, counterfactual_compose
                )
            )
        else:
            route, route_stats = self._route(
                scope, part_error.detach(),
                relation_error.detach() if relation_error is not None else None,
                state["curvature"].detach(),
                relation_curvature.detach() if relation_curvature is not None else None,
            )
            part_route_weight = route[:, 1]
            relation_route_weight = route[:, 2]
        loss = (
            part_route_weight.detach() * part_loss
            + relation_route_weight.detach()
            * float(relation_weight) * relation_loss
        ).mean()
        stats: Dict[str, torch.Tensor] = {
            "meta_real_align": loss.detach(),
            "meta_real_part_weight": part_route_weight.mean().detach(),
            "meta_real_relation_weight": relation_route_weight.mean().detach(),
            "meta_real_residual_mode": route.new_tensor(
                float(
                    scope == "counterfactual"
                    and counterfactual_compose == "residual"
                )
            ).detach(),
            **route_stats,
        }
        stats.update({"part_" + key: value for key, value in part_stats.items()})
        stats.update({"rel_" + key: value for key, value in relation_stats.items()})
        return loss, stats

    def policy_parameters(self, scope: str) -> Iterable[nn.Parameter]:
        """Return exactly the parameters owned by the requested meta policy."""
        self._check_scope(scope)
        if scope == "part":
            return self.policy.parameters()
        if scope == "relation":
            if not self.enable_relations:
                raise RuntimeError("relation meta scope requested but disabled")
            return self.relation_policy.parameters()
        parameters = tuple(self.policy.parameters()) + tuple(
            self.relation_policy.parameters()
        )
        if scope in ("adaptive", "counterfactual"):
            parameters = parameters + tuple(self.router.parameters())
        return parameters

    def all_policy_parameters(self) -> Iterable[nn.Parameter]:
        """All phi parameters, used to exclude policies from the main optimizer."""
        parameters = tuple(self.policy.parameters())
        if self.enable_relations:
            parameters = (
                parameters
                + tuple(self.relation_policy.parameters())
                + tuple(self.router.parameters())
            )
        return parameters
