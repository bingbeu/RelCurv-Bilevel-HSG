# Adaptive-Granularity Task-Feedback Bilevel Optimization

Let `theta` denote the backbone and classifier, `psi` the shared semantic
adapter, and `phi` the token policies plus granularity router.

For a support view, local and relation policies produce `p_part` and `p_rel`.
The router produces:

```text
[g_skip, g_part, g_rel] = Router_phi(stopped error/curvature statistics)
```

The lower-level update is

```text
L_inner = g_part * sum_i P * p_part[i] * e_part[i]
        + g_rel  * sum_m R * p_rel[m] * e_rel[m]

psi+ = psi - inner_lr * grad_psi L_inner
```

Both branches update the same `psi`. Relation tokens are recomputed from
adapted part tokens, so relation-guided updates are evaluated through the
ordinary part-based classifier rather than an extra relation classifier.

The upper-level objective is

```text
L_outer = task_weight * L_hier(query; psi+, q_part)
        + semantic_weight * L_sem_ref(query; psi+, q_part, q_rel)
        + kl_weight * item_policy_regularization
        + router_kl_weight * router_prior_regularization
```

`q_part` and `q_rel` are uniform or stopped HVP distributions. Neither learned
item weights nor learned route weights multiply the raw query loss. Thus the
policy is rewarded for selecting an update that improves post-update query
performance, not for selecting items with an already small error.

## Strict separation

- `create_graph=True` retains the hypergradient through the virtual update.
- `phi` is excluded from the main optimizer.
- The meta optimizer contains only the policies/router selected by
  `--meta-scope`.
- Real alignment uses detached `p_part`, `p_rel` and route probabilities.
- The outer query backbone is stopped; only functional `psi+` is differentiated.
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
4. Adaptive routing (`--meta-scope adaptive`).
5. Adaptive routing without relation HVP (`--no-relation-hvp`).
6. Uniform item policies (`--meta-reference-mix 0`).
7. Task-free outer objective (`--meta-task-weight 0`) as a diagnostic only.

Use identical initialization, schedules and paired seeds for every comparison.
