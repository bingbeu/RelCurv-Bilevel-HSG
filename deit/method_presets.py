"""Explicit, reproducible method presets for the V8.7 experiments.

The preset is selected by the command line.  Nothing in the model silently
branches on a dataset name.  Dataset validation exists only to catch an
accidental command/configuration mismatch.
"""

from typing import Dict, Mapping, Tuple


MANUAL_PRESET = "manual"
CUB_V85_PRESET = "cub-v85"
AIR_CURVPART_V7_PRESET = "air-curvpart-v7"
METHOD_PRESET_CHOICES = (
    MANUAL_PRESET,
    CUB_V85_PRESET,
    AIR_CURVPART_V7_PRESET,
)


_PRESETS: Mapping[str, Tuple[str, Mapping[str, object]]] = {
    CUB_V85_PRESET: (
        "BIRD-HIER",
        {
            # Freeze the best CUB path exactly at V8.5 semantics.
            "enable_bilevel": True,
            "meta_scope": "counterfactual",
            "counterfactual_compose": "competitive",
            "checkpoint_metric": "fpa",
            "num_parts": 8,
            "semantic_rank": 64,
            "meta_adapter_rank": 64,
            "meta_policy_hidden": 128,
            "meta_policy_tau": 1.0,
            "meta_start_epoch": 5,
            "meta_inner_lr": 0.1,
            "meta_inner_grad_normalization": True,
            "meta_lr": 1.0e-4,
            "meta_weight_decay": 1.0e-4,
            "meta_clip_grad": 5.0,
            "meta_reference_mix": 0.5,
            "meta_q": "uniform",
            "meta_real_weight": 0.1,
            "meta_kl_weight": 0.01,
            "meta_task_weight": 1.0,
            "meta_semantic_weight": 0.1,
            "meta_router_kl_weight": 0.001,
            "meta_router_advantage_scale": 0.1,
            "meta_router_regret_normalization": True,
            "meta_router_regret_floor": 1.0e-4,
            "meta_safe_improvement_margin": 1.0e-5,
            "meta_safe_route_budget": 0.05,
            "meta_safe_confidence_scale": 1.0,
            "meta_safe_confidence_budget": True,
            "no_meta_safe_gate": False,
            "meta_consistency_weight": 0.1,
            "meta_consistency_credit_weight": 0.25,
            "meta_fine_weight": 1.0,
            "meta_family_weight": 0.5,
            "meta_basic_weight": 0.5,
            "meta_relation_weight": 1.0,
            "relation_dim": 64,
            "relation_hvp_samples": 1,
            "relation_contrastive_weight": 0.1,
            "relation_temperature": 0.1,
            "router_hidden_dim": 32,
            "router_prior": (0.50, 0.45, 0.05),
            "lam_cls": 0.0,
            "lam_attr": 1.0,
            "proto_align_weight": 0.0,
        },
    ),
    AIR_CURVPART_V7_PRESET: (
        "AIR-HIER",
        {
            # Exact Part-only task-feedback path used by CurvPart-HSG V7.
            # Non-counterfactual Part scope uses the raw one-step gradient and
            # a full real alignment route; relation modules are not built.
            "enable_bilevel": True,
            "meta_scope": "part",
            "counterfactual_compose": "competitive",
            "checkpoint_metric": "acc1",
            "num_parts": 8,
            "semantic_rank": 64,
            "meta_adapter_rank": 64,
            "meta_policy_hidden": 128,
            "meta_policy_tau": 1.0,
            "meta_start_epoch": 5,
            "meta_inner_lr": 0.1,
            "meta_inner_grad_normalization": False,
            "meta_lr": 1.0e-4,
            "meta_weight_decay": 1.0e-4,
            "meta_clip_grad": 5.0,
            "meta_reference_mix": 0.5,
            "meta_q": "uniform",
            "meta_real_weight": 0.1,
            "meta_kl_weight": 0.01,
            "meta_task_weight": 1.0,
            "meta_semantic_weight": 0.1,
            "meta_fine_weight": 1.0,
            "meta_family_weight": 0.5,
            "meta_basic_weight": 0.5,
            "lam_cls": 0.0,
            "lam_attr": 1.0,
            "proto_align_weight": 0.0,
        },
    ),
}


def apply_method_preset(args) -> Dict[str, Tuple[object, object]]:
    """Validate and apply a named preset to an argparse-like namespace.

    Returns the changed values as ``name -> (old, new)`` for a concise startup
    audit.  Manual mode leaves every user-provided argument untouched.
    """
    preset = getattr(args, "method_preset", MANUAL_PRESET)
    if preset not in METHOD_PRESET_CHOICES:
        raise ValueError(
            f"method preset must be one of {METHOD_PRESET_CHOICES}, got {preset!r}"
        )
    if preset == MANUAL_PRESET:
        return {}

    expected_dataset, values = _PRESETS[preset]
    actual_dataset = getattr(args, "data_set", None)
    if actual_dataset != expected_dataset:
        raise ValueError(
            f"method preset {preset!r} requires --data-set "
            f"{expected_dataset}, got {actual_dataset!r}"
        )

    changes: Dict[str, Tuple[object, object]] = {}
    for name, value in values.items():
        old_value = getattr(args, name, None)
        setattr(args, name, value)
        if old_value != value:
            changes[name] = (old_value, value)
    return changes


def preset_values(name: str) -> Dict[str, object]:
    """Return a copy for tests and configuration introspection."""
    if name == MANUAL_PRESET:
        return {}
    if name not in _PRESETS:
        raise ValueError(f"unknown method preset: {name!r}")
    return dict(_PRESETS[name][1])
