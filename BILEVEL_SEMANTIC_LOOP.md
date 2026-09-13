# Consistency-Credited Counterfactual Bilevel Optimization (V8.5)

Let `theta` denote the backbone and classifier, `psi` the shared semantic
adapter, and `phi` the token policies plus granularity router.

For a support view, local and relation policies produce `p_part` and `p_rel`.
V8.5 does not mix their losses before the virtual update. It constructs three
counterfactual adapter states:

```text
psi_skip = psi
psi_part = psi - inner_lr * normalize(grad_psi L_part)
psi_rel  = psi - inner_lr * normalize(grad_psi L_relation)
```

Each state is evaluated on the paired query view for Species, Family and Order:

```text
J_cls[t,b]  = supervised_classification_t(query; psi_b)
J_cons[t,b] = taxonomy_consistency_t(query; psi_b)
J[t,b]      = J_cls[t,b] + consistency_weight * J_cons[t,b]
t in {species, family, order}
b in {skip, part, relation}
```

The legacy-compatible router first produces a per-example `3 x 3` matrix:

```text
R_phi[x,t,b] = P(branch=b | example=x, hierarchy-level=t)
```

V8.5 does not allow the learned Skip logit to compete with semantic learning.
It removes Skip and forms the conditional Part/Relation route

```text
C_phi[x,t,:] = normalize(R_phi[x,t,{part,relation}])
```

This keeps Part and Relation available to all hierarchy levels. Relation
curvature is not hard-coded as coarse-only, and part curvature is not
hard-coded as fine-only.

Classification and consistency regrets are formed separately:

```text
D_cls[x,t,b] = (J_cls[x,t,b] - J_cls[x,t,skip])
               / max(abs(J_cls[x,t,skip]), 1e-3)
D_cons[x,t,b] = (J_cons[x,t,b] - J_cons[x,t,skip])
                / max(abs(J_cons[x,t,skip]), 1e-3)
```

Each signal is RMS-calibrated independently inside every example and hierarchy
level.  The safe margin applies to classification; zero consistency change does
not create an artificial non-Skip penalty:

```text
D_cls_margin[x,t,:] = D_cls[x,t,:] + [0, safe_margin, safe_margin]
D_cls_cal  = D_cls_margin / max(RMS_b(D_cls_margin), regret_floor)
D_cons_cal = D_cons / max(RMS_b(D_cons), regret_floor)
D_joint = D_cls_cal + consistency_credit_weight * D_cons_cal
```

For each example and level, stopped positive-gain evidence defines eligibility

```text
E[x,t,b] = 1[D_joint[x,t,b] < 0], b in {part, relation}
```

The conditional router is masked and reweighted by stopped branch confidence:

```text
confidence[x,t,b] = relu(-D_joint[x,t,b])
W[x,t,b] = C_phi[x,t,b] * E[x,t,b]
           * clip(confidence[x,t,b] / confidence_scale, 1e-8, 1)
```

Unlike V8.4, confidence does not shrink the full update.  Whenever at least one
branch is safe, `beta = --meta-safe-route-budget` is fixed.  The route used by
both the outer solver and real semantic loss is

```text
if no semantic branch is eligible:
    R_safe[x,t,:] = [1, 0, 0]
else:
    R_safe[x,t,skip] = 1 - beta
    R_safe[x,t,{part,relation}] = beta * normalize(W[x,t,:])
```

The upper-level task objective is the expected counterfactual query loss plus
the calibrated regret term:

```text
L_outer = task_weight * (
            sum_x,t,b w_t M[x,t] R_safe[x,t,b] J[x,t,b]
          + advantage_scale
            * sum_x,t,b w_t M[x,t] R_safe[x,t,b] D_joint[x,t,b]
          )
        + semantic_weight * L_sem_ref(query; psi_b, q_part, q_rel)
        + kl_weight * item_policy_regularization
        + router_kl_weight * router_prior_regularization
```

`q_part` and `q_rel` are uniform or stopped HVP distributions. Learned item
weights never multiply the raw query error; the conditional route combines
only task losses measured after the separate virtual updates. Thus the policy
is rewarded for producing and selecting an update that improves post-update
query performance, not for selecting items with an already small error.
Independent calibration prevents CE scale from hiding consistency gains, while
two-stage routing makes Skip a safety outcome instead of a learnable escape.
Confidence is detached and changes only the conditional Part/Relation split,
so it cannot suppress the full real update or shortcut the hypergradient.
The mask `M` follows the official free-grained label protocol.
`--no-meta-router-regret-normalization` restores raw router regret without
changing the confidence normalization.

## Safe real update

For each example, hierarchy level and non-skip branch, V8.5 computes stopped,
independently calibrated classification and consistency improvements. Their
weighted sum determines eligibility:

```text
A_joint[t,b] = -D_cls_cal[t,b]
               - consistency_credit_weight * D_cons_cal[t,b]
```

Only branches with positive `A_joint` may contribute to the real adapter loss.
If at least one branch is accepted, the full fixed
`--meta-safe-route-budget` is distributed over eligible Part/Relation branches.
Confidence reweights this split but never changes the total budget. Eligibility
and confidence are detached, while the conditional router and item policies are
still optimized through the post-update query objective.
`--no-meta-safe-confidence-routing` disables confidence reweighting (the legacy
alias `--no-meta-safe-confidence-budget` is retained). `--no-meta-safe-gate`
treats both semantic branches as eligible and uses the full bounded budget.

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
`--meta-consistency-weight`. The same consistency losses are also exposed
unscaled to the router, calibrated separately from classification regret, and
credited with `--meta-consistency-credit-weight`. This uses predictions and the
public class tree, never an unavailable fine label.

## Strict separation

- `create_graph=True` retains the hypergradient through both independent
  non-skip virtual updates.
- `phi` is excluded from the main optimizer.
- The meta optimizer contains only the policies/router selected by
  `--meta-scope`.
- Real alignment uses detached `p_part`, `p_rel` and route probabilities.
- The discrete positive-gain gate is stopped and cannot provide a shortcut
  around the virtual-update hypergradient.
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
5. V8.5 consistency-credited routing (`--meta-scope counterfactual`).
6. V8.5 without router consistency credit
   (`--meta-consistency-credit-weight 0`).
7. V8.5 without the consistency surrogate (`--meta-consistency-weight 0`).
8. V8.5 without confidence allocation
   (`--no-meta-safe-confidence-routing`).
9. V8.5 without relation HVP (`--no-relation-hvp`).
10. Uniform item policies (`--meta-reference-mix 0`).
11. Task-free outer objective (`--meta-task-weight 0`) as a diagnostic only.
12. Raw virtual gradients (`--no-meta-inner-grad-normalization`).
13. Unsafe real update (`--no-meta-safe-gate`).
14. Raw V8.1 router regret (`--no-meta-router-regret-normalization`).
15. Unbounded semantic route (`--meta-safe-route-budget 1.0`).

Use identical initialization, schedules and paired seeds for every comparison.
