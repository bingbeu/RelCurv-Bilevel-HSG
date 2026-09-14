# Explicit Dataset-Validated Bilevel Presets (V8.7)

Let `theta` denote the backbone and hierarchy classifiers, `psi` the shared
semantic adapter, and `phi` the item policy (plus the granularity router when
that router is enabled). V8.7 does not introduce a hidden dataset condition in
the model. It exposes two named command-line presets whose compatibility with
the requested dataset is validated before the model and optimizers are built.

| Preset | Dataset | Solver | Validation metric |
|---|---|---|---|
| `cub-v85` | `BIRD-HIER` | V8.5 competitive Skip/Part/Relation | FPA |
| `air-curvpart-v7` | `AIR-HIER` | CurvPart V7 Part-only feedback | Species Acc@1 |

Both solvers optimize the complete Species/Family/Order hierarchy. “Part-only”
describes the semantic update type; it does not mean fine-level-only learning.

## CUB: V8.5 competitive counterfactual solver

The CUB preset constructs three virtual adapters independently from the same
base adapter:

```text
psi_skip = psi
psi_part = psi - eta * normalize(grad_psi L_part)
psi_rel  = psi - eta * normalize(grad_psi L_relation)
```

Every branch is evaluated on a stopped paired query view at Species, Family and
Order:

```text
J_cls[t,b]  = supervised_classification_t(query; psi_b)
J_cons[t,b] = taxonomy_consistency_t(query; psi_b)
J[t,b]      = J_cls[t,b] + consistency_weight * J_cons[t,b]
```

Classification and taxonomy-consistency regrets are calibrated independently
within each example and hierarchy level. Positive-gain gating supplies a fixed
5% non-Skip budget; confidence only allocates that budget between eligible Part
and Relation branches. The real semantic update uses detached route and item
weights, while the outer loss alone updates `phi`.

This is the exact competitive V8.5 path retained because it produced the best
observed CUB result. V8.6 residual composition remains available only in
`--method-preset manual` ablations.

## Aircraft: CurvPart V7 Part-only solver

The Aircraft preset constructs no Relation encoder, Relation policy, or
granularity router. Its one-step virtual adapter uses the raw Part gradient,
matching CurvPart-HSG V7:

```text
psi_part = psi - eta * grad_psi L_part
```

The paired query objective is still hierarchical:

```text
J_part = 1.0 * J_species
       + 0.5 * J_family
       + 0.5 * J_order
```

The Part policy receives task feedback only through `J_part(psi_part)`. For the
real model update, Part alignment is full strength rather than constrained by a
1--5% counterfactual routing budget:

```text
L_real = mean(stop_gradient(p_part) * L_part)
```

The top-level multiplier remains `--meta-real-weight 0.1`, as in the verified
V7 configuration. Best checkpoints are selected by Species Acc@1, matching the
historical Aircraft protocol.

## Bilevel separation invariants

- `phi` is excluded from the main optimizer.
- Policy inputs, query backbone features, and query semantic targets are
  stopped.
- Real-loss policy and routing weights are detached.
- Only the post-update outer objective updates `phi`.
- Virtual inner gradients retain `create_graph=True` so task feedback can pass
  through the one-step adapter update.
- Free-grained Species and Family label masks follow the official protocol.
- Taxonomy consistency uses predictions and the public hierarchy, not hidden
  labels.
- Relation descriptors are training-time evidence and are never concatenated
  into the inference classifier.
- This is a differentiable one-step truncated bilevel method, not an exact
  solution of an inner argmin.

## Reproducibility and comparison

The preset name, resolved scope, resolved composition, and checkpoint metric
are stored in every `log.txt` row; the full resolved `args` namespace is stored
in checkpoints. A preset rejects the wrong dataset instead of silently changing
behavior.

For a controlled comparison, keep initialization, schedule, data split, and
paired seeds identical. Report at least three seeds for the two V8.7 presets
and retain these ablations:

1. E2 without bilevel optimization.
2. CurvPart V7 / Part-only.
3. AdaCurv V7 / adaptive.
4. V8.5 competitive.
5. V8.6 Part-anchored residual.

The complete two-GPU training and evaluation commands are in `README.md`.
