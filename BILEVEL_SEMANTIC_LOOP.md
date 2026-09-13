# Part-Anchored Residual Counterfactual Bilevel Optimization (V8.6)

Let `theta` denote the backbone/classifiers, `psi` the shared semantic adapter,
and `phi` the item policies plus the granularity router. V8.6 retains the V8.5
competitive solver and adds a part-anchored residual solver selected by
`--counterfactual-compose`.

## Virtual branches

The V8.5-compatible `competitive` mode constructs three alternatives from the
same base adapter:

```text
psi_skip = psi
psi_part = psi - eta * normalize(grad_psi L_part)
psi_rel  = psi - eta * normalize(grad_psi L_relation)
```

The V8.6 `residual` mode instead composes the semantic updates:

```text
psi_skip = psi
psi_part = psi - eta * normalize(grad_psi L_part)
psi_res  = psi_part
           - eta * relation_residual_inner_scale
             * normalize(grad_psi_part L_relation(psi_part))
```

The three branch labels in tensors remain `skip / part / relation` for
checkpoint and logging compatibility; in residual mode the third label means
`part + relation residual`. Both inner gradients retain `create_graph=True`,
so the outer loss differentiates through the sequential update.

Part and Relation are available at Species, Family and Order. The method is not
hard-coded as “Part for fine classes, Relation for coarse classes.” Each level
receives its own route and its own counterfactual evidence.

## Query evaluation and incremental credit

Every branch is evaluated on the paired query view:

```text
J_cls[t,b]  = supervised_classification_t(query; psi_b)
J_cons[t,b] = taxonomy_consistency_t(query; psi_b)
J[t,b]      = J_cls[t,b] + consistency_weight * J_cons[t,b]
```

For competitive mode, both semantic branches are compared with Skip. For
residual mode, Part is compared with Skip and the Relation residual is compared
with Part:

```text
D_part = relative_regret(J_part, J_skip)
D_res  = relative_regret(J_res,  J_part)
```

Classification and hierarchy-consistency regrets are normalized separately
inside every example and hierarchy level:

```text
D_cls_cal  = calibrate(D_cls + safe_margin)
D_cons_cal = calibrate(D_cons)
D_joint    = D_cls_cal
             + consistency_credit_weight * D_cons_cal
```

This distinction is essential. In V8.5, a Relation branch could receive
consistency credit while replacing a Part branch with a larger classification
gain. In V8.6, Relation receives credit only for what it adds after Part.

## Species no-regret guard

The residual branch must first pass the ordinary positive-gain test and the
Part branch must be safe for the same example/level. Two additional stopped
conditions protect the fine-grained task:

```text
(CE_species_res - CE_species_part) / max(|CE_species_part|, 1e-3)
    <= species_no_regret_margin

KL(p_base_species || p_res_species)
    - KL(p_base_species || p_part_species)
    <= species_anchor_kl_margin
```

The first check uses the Species label when it is available. The second is a
label-free trust region against the stopped base query prediction and applies
to every example. The final residual eligibility is

```text
E_res[t] = positive_incremental_gain[t]
           AND E_part[t]
           AND species_supervised_guard
           AND species_anchor_guard
```

The guards are detached: they control safety but cannot become a shortcut for
the hypergradient.

## Two-stage route

The legacy-compatible router emits a per-example `3 x 3` matrix:

```text
R_phi[x,t,b] = P(branch=b | example=x, hierarchy-level=t)
```

Skip is a stopped safety outcome. The learned router controls only the
conditional Part/Relation split:

```text
C_phi = normalize(R_phi[..., {part, relation}])
confidence = relu(-D_joint)
W = C_phi * eligibility
    * clip(confidence / confidence_scale, 1e-8, 1)
```

If neither semantic branch is eligible, the route is all Skip. Otherwise the
fixed safe budget `beta = --meta-safe-route-budget` is distributed over the
eligible semantic branches:

```text
R_safe[skip] = 1 - beta
R_safe[{part, relation}] = beta * normalize(W)
```

Confidence changes the Part/Relation allocation but does not shrink the total
safe budget.

## Real update composition

Competitive mode keeps disjoint Part and Relation weights:

```text
w_part = R_safe[part]
w_rel  = R_safe[relation]
```

Residual mode executes the same composition represented by its virtual branch:

```text
w_part = R_safe[part] + R_safe[relation]
w_rel  = R_safe[relation]

L_real = mean(w_part * L_part + w_rel * relation_weight * L_relation)
```

Thus selecting Relation never removes the Part anchor. All route and item
policy weights are detached in `L_real`; only the outer objective updates
`phi`.

## Outer objective

The task objective remains an expected post-update query loss plus calibrated
regret, fixed-reference semantic alignment, and policy regularization:

```text
L_outer = task_weight * (
            sum R_safe * J
            + advantage_scale * sum R_safe * D_joint
          )
        + semantic_weight * L_sem_ref
        + kl_weight * item_policy_regularization
        + router_kl_weight * conditional_router_regularization
```

Learned item weights never multiply raw query error. They affect the query
objective only through the virtual adapter update. The official free-grained
label mask is retained for Species and Family; the taxonomy-consistency term
uses predicted distributions and the public hierarchy, not unavailable labels.

## Invariants

- `competitive` preserves V8.5 branch construction and real weight mapping.
- `residual` always unrolls Relation from `psi_part`, never from the base.
- A residual route contributes its probability to both Part and Relation in
  the real update.
- Relation residual eligibility implies Part eligibility.
- The base Species distribution is detached before the virtual evaluations.
- `phi` is excluded from the main optimizer.
- The real loss detaches item policies, route probabilities, confidence and
  discrete guards.
- Query backbone features and semantic targets are stopped.
- Relation descriptors are training-time update evidence and are not
  concatenated into the inference classifier.
- The method is a differentiable one-step truncated bilevel solver; it does not
  claim to solve the inner argmin to convergence.

## Required comparisons

1. E2 without bilevel optimization.
2. CurvPart V7 / part-only (`--meta-scope part`).
3. AdaCurv V7 (`--meta-scope adaptive`).
4. V8.5 competitive (`--meta-scope counterfactual
   --counterfactual-compose competitive`).
5. V8.6 residual (`--meta-scope counterfactual
   --counterfactual-compose residual`).
6. Residual without relation consistency credit
   (`--meta-consistency-credit-weight 0`).
7. Residual without Species guards (large margins; diagnostic only).
8. Residual relation step scales such as `0.25`, `0.5`, and `1.0`, selected on
   validation data.

Use identical initialization, schedules and paired seeds. Run V8.6 on both CUB
and Aircraft before adopting any dataset-specific composition policy.
