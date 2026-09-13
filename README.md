# RelCurv-Bilevel-HSG V8.1

**Task-Feedback Adaptive-Granularity Curvature-Aware Semantic Grounding for
Free-Grained Hierarchical Recognition**

V8.1 unrolls `skip`, local-part alignment and relation alignment as three separate
virtual updates.  A paired query view evaluates every branch independently on
Species, Family and Order.  The router learns a per-example `3 x 3` decision
matrix, so every semantic granularity may help every hierarchy level when its
counterfactual task feedback is positive.

## Why V8.1

- The virtual adapter is part of the real inference path, so meta improvement
  can affect downstream predictions.
- Part and relation alignment update the **same** adapter.
- Relation descriptors are not concatenated into the classifier.
- The router no longer sees only one pre-mixed post-update scalar; it receives
  identifiable skip/part/relation outcomes for each hierarchy level.
- Each Part/Relation virtual gradient is L2-normalized independently, making
  `--meta-inner-lr` a bounded virtual step norm instead of an uncontrolled raw
  gradient multiplier.
- A stopped positive-gain mask transfers rejected Part/Relation probability to
  `skip` before the real adapter update, preventing measured negative transfer.
- Counterfactual training checkpoints are selected by all-level FPA by default;
  legacy scopes retain Species Acc@1 selection.
- A fixed-taxonomy Jensen-Shannon term is a differentiable TICE surrogate and
  uses no unavailable training label.
- The relation embedding is low-rank (`384 -> 64` by default), avoiding the
  previous approximately one-million-parameter relation branch.
- Policy and router parameters are updated only by the outer hypergradient;
  every weight in the real model loss is detached.
- All classification residual gates start at zero, preserving the safe E2
  initialization.

See [`BILEVEL_SEMANTIC_LOOP.md`](BILEVEL_SEMANTIC_LOOP.md) for equations and
implementation invariants.

## Installation

Recommended: Python 3.10, PyTorch 2.1.2, torchvision 0.16.2 and CUDA 12.1.

```bash
conda create -n relcurv python=3.10
conda activate relcurv
pip install -r requirements.txt
pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121
export PYTHONPATH=deit/:deit/dataset/:$PYTHONPATH
```

Download `deit_small_patch16_224-cd65a155.pth`. Bilevel runs refuse silent
random initialization unless `--allow-random-init` is passed explicitly.

## Aircraft: adaptive local/relation routing

```bash
CUDA_VISIBLE_DEVICES=3 python deit/main_hier_partial.py \
  --model deit_small_patch16_224 \
  --batch-size 32 --epochs 100 --lr 5e-4 \
  --weight-decay 0.05 --warmup-epochs 5 \
  --seed 0 --random_seed 0 --num_workers 8 \
  --data-set AIR-HIER \
  --data-path /raid/datasets/fgvc-aircraft \
  --output_dir ./output/air_safe_counterfactual_v81_seed0 \
  --texts captions/air_caps.txt \
  --sim_loss_weight 1 --sp_proportion 0.3 --fm_proportion 0.6 \
  --finetune /path/deit_small_patch16_224-cd65a155.pth \
  --enable-bilevel --meta-scope counterfactual --num-parts 8 \
  --lam-cls 0.0 --lam-attr 1.0 --proto-align-weight 0.0 \
  --meta-start-epoch 5 --meta-inner-lr 0.1 \
  --meta-lr 1e-4 --meta-weight-decay 1e-4 \
  --meta-real-weight 0.1 --meta-kl-weight 0.01 \
  --meta-reference-mix 0.5 --meta-q uniform \
  --meta-task-weight 1.0 --meta-semantic-weight 0.1 \
  --meta-router-kl-weight 0.001 \
  --meta-router-advantage-scale 100 \
  --meta-safe-improvement-margin 1e-5 \
  --meta-consistency-weight 0.1 \
  --relation-dim 64 --relation-hvp-samples 1 \
  --relation-contrastive-weight 0.1 --relation-temperature 0.1 \
  --router-prior 0.50 0.45 0.05
```

Start with batch size 32. Increase it only after measuring memory on the target
GPU. Use a new output directory.

## CUB

Use the same counterfactual method with the official CUB proportions and paths:

```bash
CUDA_VISIBLE_DEVICES=3 python deit/main_hier_partial.py \
  --model deit_small_patch16_224 \
  --batch-size 32 --epochs 100 --lr 5e-4 \
  --weight-decay 0.05 --warmup-epochs 5 \
  --seed 0 --random_seed 0 --num_workers 8 \
  --data-set BIRD-HIER --data-path /data/CUB_200_2011/images_split \
  --output_dir ./output/bird_safe_counterfactual_v81_seed0 \
  --texts captions/cub_caps.txt --sim_loss_weight 1 \
  --sp_proportion 0.1 --fm_proportion 0.5 \
  --finetune /path/deit_small_patch16_224-cd65a155.pth \
  --enable-bilevel --meta-scope counterfactual --num-parts 8 \
  --meta-start-epoch 5 --meta-inner-lr 0.1 --meta-lr 1e-4 \
  --meta-real-weight 0.1 --meta-q uniform \
  --meta-task-weight 1.0 --meta-semantic-weight 0.1 \
  --meta-router-advantage-scale 100 \
  --meta-safe-improvement-margin 1e-5 \
  --meta-consistency-weight 0.1 \
  --relation-dim 64 --relation-hvp-samples 1 \
  --router-prior 0.50 0.45 0.05
```

The defaults enable normalized inner gradients, the safe gate, and automatic
FPA checkpoint selection for `counterfactual`. Their diagnostic ablations are:

```bash
--no-meta-inner-grad-normalization
--no-meta-safe-gate
--checkpoint-metric acc1
```

## Checkpoint migration

The V8/V8.1 counterfactual router has nine logits instead of the V7 router's three.
Do **not** resume a V7 optimizer/model state into `--meta-scope counterfactual`.
Start from the same E2 or ImageNet checkpoint with `--finetune`; use `--resume`
only for a V8 checkpoint from the same configuration.  The old `adaptive` scope
is retained so existing V7 checkpoints remain loadable.

## Verification and logging

```bash
python -m compileall -q deit
PYTHONPATH=deit python -m unittest -v deit/test_semantic_bilevel.py
```

Training logs expose aggregate routes plus `route_species_*`, `route_family_*`,
`route_order_*`, every branch's per-level improvement,
`meta_part_task_improvement`, `meta_relation_task_improvement`, policy entropy
and `meta_policy_grad_norm`. V8.1 additionally reports raw branch-gradient norms,
bounded virtual-step norms, per-level safe acceptance rates, safe routes, and
per-epoch FPA/TICE. Report at least three paired seeds and use a held-out
validation split for formal model selection rather than the test set.

This repository derives from the official implementation of *Free-Grained
Hierarchical Visual Recognition*. Retain the upstream license and attribution.
