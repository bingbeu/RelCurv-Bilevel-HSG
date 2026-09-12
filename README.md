# RelCurv-Bilevel-HSG V7

**Task-Feedback Adaptive-Granularity Curvature-Aware Semantic Grounding for
Free-Grained Hierarchical Recognition**

V7 replaces fixed relation fusion with a meta-learned update route. The policy
chooses `skip`, local-part alignment or relation alignment on a support view.
It performs one differentiable virtual update of a shared low-rank semantic
adapter; an independent query view then judges the updated adapter with the
hierarchical recognition loss and a fixed-reference semantic evaluator.

## Why V7

- The virtual adapter is part of the real inference path, so meta improvement
  can affect downstream predictions.
- Part and relation alignment update the **same** adapter.
- Relation descriptors are not concatenated into the classifier.
- A learned granularity router can reject noisy relation updates on CUB and
  retain them only when query task feedback supports them.
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
  --batch-size 256 --epochs 100 --lr 5e-4 \
  --weight-decay 0.05 --warmup-epochs 5 \
  --seed 0 --random_seed 0 --num_workers 8 \
  --data-set AIR-HIER \
  --data-path /raid/datasets/fgvc-aircraft \
  --output_dir ./output/air_adacurv_v7_seed0 \
  --texts captions/air_caps.txt \
  --sim_loss_weight 1 --sp_proportion 0.3 --fm_proportion 0.6 \
  --finetune /path/deit_small_patch16_224-cd65a155.pth \
  --enable-bilevel --meta-scope adaptive --num-parts 8 \
  --lam-cls 0.0 --lam-attr 1.0 --proto-align-weight 0.0 \
  --meta-start-epoch 5 --meta-inner-lr 0.1 \
  --meta-lr 1e-4 --meta-weight-decay 1e-4 \
  --meta-real-weight 0.1 --meta-kl-weight 0.01 \
  --meta-reference-mix 0.5 --meta-q uniform \
  --meta-task-weight 1.0 --meta-semantic-weight 0.1 \
  --meta-router-kl-weight 0.001 \
  --relation-dim 64 --relation-hvp-samples 1 \
  --relation-contrastive-weight 0.1 --relation-temperature 0.1 \
  --router-prior 0.50 0.45 0.05
```

Start with batch size 32. Increase it only after measuring memory on the target
GPU. Use a new output directory.

## CUB

Use the same adaptive method with the official CUB proportions and paths:

```bash
CUDA_VISIBLE_DEVICES=3 python deit/main_hier_partial.py \
  --model deit_small_patch16_224 \
  --batch-size 256 --epochs 100 --lr 5e-4 \
  --weight-decay 0.05 --warmup-epochs 5 \
  --seed 0 --random_seed 0 --num_workers 8 \
  --data-set BIRD-HIER --data-path /data/CUB_200_2011/images_split \
  --output_dir ./output/cub_adacurv_v7_seed0 \
  --texts captions/cub_caps.txt --sim_loss_weight 1 \
  --sp_proportion 0.1 --fm_proportion 0.5 \
  --finetune /path/deit_small_patch16_224-cd65a155.pth \
  --enable-bilevel --meta-scope adaptive --num-parts 8 \
  --meta-start-epoch 5 --meta-inner-lr 0.1 --meta-lr 1e-4 \
  --meta-real-weight 0.1 --meta-q uniform \
  --meta-task-weight 1.0 --meta-semantic-weight 0.1 \
  --relation-dim 64 --relation-hvp-samples 1 \
  --router-prior 0.50 0.45 0.05
```

## Checkpoint migration

V7 changes the relation encoder shape, removes the standalone relation adapter
and adds a router. Do **not** resume a V6 optimizer state. Start from the same
E2 or ImageNet checkpoint with `--finetune`; use `--resume` only for a V7
checkpoint from the same configuration.

## Verification and logging

```bash
python -m compileall -q deit
PYTHONPATH=deit python -m unittest -v deit/test_semantic_bilevel.py
```

Training logs expose `route_skip`, `route_part`, `route_relation`,
`meta_outer_task`, `meta_task_improvement`, policy entropy and
`meta_policy_grad_norm`. Report at least three paired seeds and select epochs on
a held-out validation set rather than the test set.

This repository derives from the official implementation of *Free-Grained
Hierarchical Visual Recognition*. Retain the upstream license and attribution.

