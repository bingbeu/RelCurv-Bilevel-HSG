# Calibrated Safe Counterfactual Bilevel Optimization (V8.2)

Let `theta` denote the backbone and classifier, `psi` the shared semantic
adapter, and `phi` the token policies plus granularity router.

For a support view, local and relation policies produce `p_part` and `p_rel`.
V8.2 does not mix their losses before the virtual update. It constructs three
counterfactual adapter states:

```text
psi_skip = psi
psi_part = psi - inner_lr * normalize(grad_psi L_part)
psi_rel  = psi - inner_lr * normalize(grad_psi L_relation)
```

Each state is evaluated on the paired query view for Species, Family and Order:

```text
J[t,b] = hierarchy_task_t(query; psi_b)
t in {species, family, order}
b in {skip, part, relation}
```

The router produces a per-example `3 x 3` matrix rather than one global vector:

```text
R_phi[x,t,b] = P(branch=b | example=x, hierarchy-level=t)
```

This keeps all branches available to all hierarchy levels.  Relation curvature
is not hard-coded as coarse-only, and part curvature is not hard-coded as
fine-only.

Let the raw relative branch regret be

```text
D[x,t,b] = (J[x,t,b] - J[x,t,skip])
           / max(abs(J[x,t,skip]), 1e-3)
```

V8.2 adds the positive-gain margin to non-skip branches and calibrates the
signal independently inside every example and hierarchy level:

```text
D_margin[x,t,:] = D[x,t,:] + [0, safe_margin, safe_margin]
scale[x,t] = max(RMS_b(stopgrad(D_margin[x,t,b])), regret_floor)
D_cal[x,t,b] = D_margin[x,t,b] / scale[x,t]
```

The upper-level task objective is the expected counterfactual query loss plus
the calibrated regret term:

```text
L_outer = task_weight * sum_x,t,b w_t M[x,t] R[x,t,b] J[x,t,b]
        + advantage_scale * sum_x,t,b w_t M[x,t] R[x,t,b] D_cal[x,t,b]
        + semantic_weight * L_sem_ref(query; psi_b, q_part, q_rel)
        + kl_weight * item_policy_regularization
        + router_kl_weight * router_prior_regularization
```

`q_part` and `q_rel` are uniform or stopped HVP distributions. Neither learned
item weights nor learned route weights multiply the raw query loss. Thus the
policy is rewarded for selecting an update that improves post-update query
performance, not for selecting items with an already small error. Calibration
preserves branch ordering while preventing small dataset-level averages from
forcing a global Skip solution. The mask `M` follows the official free-grained
label protocol. `--no-meta-router-regret-normalization` restores V8.1 behavior.

## Safe real update

For each example, hierarchy level and non-skip branch, V8.2 computes the stopped
relative query improvement

```text
A[t,b] = (J[t,skip] - J[t,b]) / max(abs(J[t,skip]), 1e-3)
```

Only branches with `A[t,b] > --meta-safe-improvement-margin` may contribute to
the real adapter loss. Probability assigned to a rejected branch is transferred
to `skip`; it is not renormalized onto another semantic branch. The eligibility
mask is detached, while the router and item policies are still optimized by the
differentiable counterfactual outer objective. `--no-meta-safe-gate` is the
required unsafe ablation.

Part and Relation raw gradients are normalized separately over the complete
shared-adapter parameter vector. Consequently, `--meta-inner-lr` is the L2 norm
of each virtual step. `--no-meta-inner-grad-normalization` restores the V8 raw
gradient update for ablation.

## FPA and TICE surrogates

The weighted Species/Family/Order query losses form a differentiable surrogate
for full-path accuracy: all three levels must improve for their joint loss to
fall.  For CUB and Aircraft, the fixed taxonomy additionally projects the
Species distribution to implied Family and Order distributions.  Jensen-Shannon
consistency with the two classifier distributions is added with
`--meta-consistency-weight`.  This uses predictions and the public class tree,
never an unavailable fine label.

## Strict separation

- `create_graph=True` retains the hypergradient through both independent
  non-skip virtual updates.
- `phi` is excluded from the main optimizer.
- The meta optimizer contains only the policies/router selected by
  `--meta-scope`.
- Real alignment uses detached `p_part`, `p_rel` and route probabilities.
- The outer query backbone is stopped; only functional counterfactual adapters
  are differentiated.
- The semantic relation target encoder is frozen.

This is a differentiable one-step truncated solver of a bilevel objective, not
an assertion that the inner argmin is solved to convergence.

## Relation representation

For part attention centroid `c_k` and part feature `h_k`, an undirected edge
uses appearance and normalized layout evidence:

```text
[h_a * h_b, |h_a - h_b|, |c_a - c_b|,
 distance(c_a,c_b), |spread_a-spread_b|, overlap(A_a,A_b)]
```

The descriptor is projected to `--relation-dim` (64 by default). Relation HVP
is computed in FP32 and stopped before entering the policy.

## Required ablations

1. E2 without bilevel optimization.
2. Part-only V7 (`--meta-scope part`).
3. Fixed hybrid (`--meta-scope hybrid`).
4. V7 pre-mixed routing (`--meta-scope adaptive`).
5. V8.2 calibrated safe counterfactual routing (`--meta-scope counterfactual`).
6. V8.2 without the consistency surrogate (`--meta-consistency-weight 0`).
7. V8.2 without relation HVP (`--no-relation-hvp`).
8. Uniform item policies (`--meta-reference-mix 0`).
9. Task-free outer objective (`--meta-task-weight 0`) as a diagnostic only.
10. Raw virtual gradients (`--no-meta-inner-grad-normalization`).
11. Unsafe real update (`--no-meta-safe-gate`).
12. Raw V8.1 router regret (`--no-meta-router-regret-normalization`).

Use identical initialization, schedules and paired seeds for every comparison.
