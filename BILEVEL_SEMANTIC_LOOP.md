# Execution-Isolated Bilevel Presets (V8.7.1)

Let `theta` denote the backbone and hierarchy classifiers, `psi` the shared
semantic adapter, and `phi` the semantic item policies and optional router.
V8.7.1 exposes two explicit dataset-validated presets. It does not branch on a
dataset name inside the model.

| Preset | Hierarchical solver | Validation metric |
|---|---|---|
| `cub-v85` | frozen V8.5 Skip/Part/Relation competitive solver | FPA |
| `air-curvpart-v7` | unchanged full-strength Part task-feedback path | Species Acc@1 |

Both paths use Species, Family and Order feedback. Part-only therefore means
the semantic update granularity, not a restriction to fine labels.

## Why execution isolation is necessary

V8.5 and V8.7 CUB runs were identical through epoch 4, then diverged exactly
when bilevel optimization began at epoch 5. V8.6 had preserved the competitive
equations but reordered Part-gradient and Relation construction operations and
added a Species-anchor graph. Mathematically equivalent execution was not a
reproducible freeze of the original optimizer trajectory.

The `v85-frozen` solver therefore contains independent copies of the original
V8.5 methods:

- `_task_output_v85`;
- `_counterfactual_meta_objective_v85`;
- `real_weighted_alignment_v85`.

Their method bodies match the V8.5 source apart from method names. They retain
the original order: construct Part and Relation evidence, compute both policies,
then compute the two independent virtual gradients. No residual intermediate or
Species-anchor graph is constructed on this path.

`--counterfactual-solver v85-frozen` is valid only with:

```text
meta_scope = counterfactual
counterfactual_compose = competitive
```

Invalid combinations fail before model construction. The V8.6 residual solver
remains available through `--counterfactual-solver unified`.

## Frozen CUB V8.5 solver

The support view constructs independent virtual adapters from the same base:

```text
psi_skip = psi
psi_part = psi - eta * normalize(grad_psi L_part)
psi_rel  = psi - eta * normalize(grad_psi L_relation)
```

Each branch is evaluated on the paired query view at every hierarchy level:

```text
J_cls[t,b]  = supervised_classification_t(query; psi_b)
J_cons[t,b] = taxonomy_consistency_t(query; psi_b)
J[t,b]      = J_cls[t,b] + consistency_weight * J_cons[t,b]
```

Classification and consistency regrets are calibrated independently per
example and level. Positive-gain gating owns Skip. Whenever a semantic branch
is eligible, a fixed 5% budget is allocated between Part and Relation by the
conditional router. Learned route and policy weights are detached in the real
model loss.

## Frozen Aircraft V8.7 path

Aircraft constructs no Relation modules and uses the raw Part inner gradient:

```text
psi_part = psi - eta * grad_psi L_part
```

The query task remains hierarchical:

```text
J_part = 1.0 * J_species
       + 0.5 * J_family
       + 0.5 * J_order
```

The real Part route stays at one:

```text
L_real = mean(stop_gradient(p_part) * L_part)
```

V8.7.1 intentionally does not attempt to repair the observed uniform Aircraft
item policy. The `70.387` FPA / `9.541` TICE path is held fixed as the performance
anchor; policy-learning changes belong in a later isolated experiment.

## Strict bilevel invariants

- `phi` is excluded from the main optimizer.
- Policy inputs, query backbone features, and query semantic targets are
  stopped.
- Only post-update query objectives update `phi`.
- Virtual inner gradients retain `create_graph=True`.
- Real-loss policy, route, confidence and eligibility weights are detached.
- The official free-grained Species/Family masks are preserved.
- Consistency uses predictions and the public taxonomy, not hidden labels.
- Relation evidence is training-only and is never concatenated into the
  inference classifier.
- The method is a differentiable one-step truncated bilevel solver, not an
  exact inner argmin.

## Reproducibility contract

Every training row records the preset, resolved scope, composition, solver and
checkpoint metric. Checkpoints store the complete resolved argument namespace.
The frozen solver also records `counterfactual_solver_v85_frozen = 1.0` in meta
statistics.

Use identical initialization, schedule, data split and paired seeds. The full
two-GPU training and evaluation commands are in `README.md`.
