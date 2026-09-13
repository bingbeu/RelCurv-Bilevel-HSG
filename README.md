# RelCurv-Bilevel-HSG V8.6

**Part-Anchored Residual Relation Curvature for Free-Grained Hierarchical
Recognition**

V8.6 targets the remaining Aircraft failure without giving up the CUB gain.
V8.5 made Part and Relation compete for one fixed semantic-update budget. The
logs showed that Part had the stronger Aircraft classification gain, while
Relation mainly supplied hierarchy-consistency credit. Giving Relation budget
therefore reduced the useful Part update. V8.6 adds a second composition mode:

- `--counterfactual-compose competitive` exactly preserves the V8.5
  `skip / part / relation` comparison.
- `--counterfactual-compose residual` evaluates
  `skip / part / (part -> relation)`. Relation can only be added after Part;
  it cannot replace the Part anchor.

The residual branch is credited only for its incremental improvement over
Part. It is accepted only when the Part branch is safe and the additional
Relation step passes both a supervised Species no-regret check and a
label-free KL trust region against the stopped base Species prediction.

This is one codebase with an explicit runtime switch, not a dataset-name
shortcut. First evaluate V8.6 residual mode on both datasets. Only after those
results should the final dataset policy be chosen; if needed, CUB can retain
`competitive` while Aircraft uses `residual` without maintaining two forks.

See [`BILEVEL_SEMANTIC_LOOP.md`](BILEVEL_SEMANTIC_LOOP.md) for equations and
implementation invariants.

## What changed from V8.5

- Part is always unrolled from the base adapter.
- In residual mode, Relation is unrolled from the virtual Part state with a
  separately bounded step (`0.5 * meta-inner-lr` by default).
- Relation regret is measured relative to Part, not relative to Skip.
- A residual Relation route contributes to both the Part anchor and Relation
  loss in the real update: `w_part = p_part + p_relation` and
  `w_relation = p_relation`.
- Relation is ineligible unless Part is eligible for the same hierarchy level.
- The extra Relation step must not exceed the configured Species CE or stopped
  prediction-KL margins.
- V8.5 remains reproducible by selecting `competitive`; tensor shapes and
  checkpoint architecture are unchanged.

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

Place `deit_small_patch16_224-cd65a155.pth` in the repository root. Bilevel
training refuses silent random initialization unless `--allow-random-init` is
passed explicitly.

## Training on two separate GPUs

Run the Aircraft block in one terminal on GPU 7 and the CUB block in another
terminal on GPU 6. Both commands start cleanly from the same ImageNet DeiT
checkpoint; do not resume a V8.5 optimizer state for the controlled comparison.

### Aircraft training on GPU 7

```bash
mkdir -p ./output/air_part_anchored_v86_seed0

CUDA_VISIBLE_DEVICES=7 python deit/main_hier_partial.py \
  --model deit_small_patch16_224 \
  --batch-size 256 --epochs 100 --lr 5e-4 \
  --weight-decay 0.05 --warmup-epochs 5 --num_workers 8 \
  --seed 0 --random_seed 0 \
  --data-set AIR-HIER \
  --data-path /raid/datasets/fgvc-aircraft \
  --output_dir ./output/air_part_anchored_v86_seed0 \
  --filename final_epoch_eval.csv \
  --texts captions/air_caps.txt --sim_loss_weight 1 \
  --sp_proportion 0.3 --fm_proportion 0.6 \
  --finetune deit_small_patch16_224-cd65a155.pth \
  --enable-bilevel --meta-scope counterfactual \
  --counterfactual-compose residual \
  --meta-relation-residual-inner-scale 0.5 \
  --meta-species-no-regret-margin 0.01 \
  --meta-species-anchor-kl-margin 0.01 \
  --checkpoint-metric fpa --num-parts 8 \
  --semantic-rank 64 --meta-adapter-rank 64 \
  --meta-policy-hidden 128 --meta-policy-tau 1.0 \
  --lam-cls 0.0 --lam-attr 1.0 --proto-align-weight 0.0 \
  --meta-start-epoch 5 --meta-inner-lr 0.1 \
  --meta-lr 1e-4 --meta-weight-decay 1e-4 \
  --meta-real-weight 0.1 --meta-kl-weight 0.01 \
  --meta-reference-mix 0.5 --meta-q uniform \
  --meta-task-weight 1.0 --meta-semantic-weight 0.1 \
  --meta-router-kl-weight 0.001 \
  --meta-router-advantage-scale 0.1 \
  --meta-router-regret-floor 1e-4 \
  --meta-safe-improvement-margin 1e-5 \
  --meta-safe-route-budget 0.05 \
  --meta-safe-confidence-scale 1.0 \
  --meta-consistency-weight 0.1 \
  --meta-consistency-credit-weight 0.25 \
  --meta-fine-weight 1.0 --meta-family-weight 0.5 \
  --meta-basic-weight 0.5 --meta-relation-weight 1.0 \
  --relation-dim 64 --relation-hvp-samples 1 \
  --relation-contrastive-weight 0.1 --relation-temperature 0.1 \
  --router-hidden-dim 32 --router-prior 0.50 0.45 0.05 \
  2>&1 | tee ./output/air_part_anchored_v86_seed0/train.log
```

### CUB training on GPU 6

```bash
mkdir -p ./output/bird_part_anchored_v86_seed0

CUDA_VISIBLE_DEVICES=6 python deit/main_hier_partial.py \
  --model deit_small_patch16_224 \
  --batch-size 256 --epochs 100 --lr 5e-4 \
  --weight-decay 0.05 --warmup-epochs 5 --num_workers 8 \
  --seed 0 --random_seed 0 \
  --data-set BIRD-HIER \
  --data-path /raid/datasets/cub-200/CUB_200_2011/images_split \
  --output_dir ./output/bird_part_anchored_v86_seed0 \
  --filename final_epoch_eval.csv \
  --texts captions/cub_caps.txt --sim_loss_weight 1 \
  --sp_proportion 0.1 --fm_proportion 0.5 \
  --finetune deit_small_patch16_224-cd65a155.pth \
  --enable-bilevel --meta-scope counterfactual \
  --counterfactual-compose residual \
  --meta-relation-residual-inner-scale 0.5 \
  --meta-species-no-regret-margin 0.01 \
  --meta-species-anchor-kl-margin 0.01 \
  --checkpoint-metric fpa --num-parts 8 \
  --semantic-rank 64 --meta-adapter-rank 64 \
  --meta-policy-hidden 128 --meta-policy-tau 1.0 \
  --lam-cls 0.0 --lam-attr 1.0 --proto-align-weight 0.0 \
  --meta-start-epoch 5 --meta-inner-lr 0.1 \
  --meta-lr 1e-4 --meta-weight-decay 1e-4 \
  --meta-real-weight 0.1 --meta-kl-weight 0.01 \
  --meta-reference-mix 0.5 --meta-q uniform \
  --meta-task-weight 1.0 --meta-semantic-weight 0.1 \
  --meta-router-kl-weight 0.001 \
  --meta-router-advantage-scale 0.1 \
  --meta-router-regret-floor 1e-4 \
  --meta-safe-improvement-margin 1e-5 \
  --meta-safe-route-budget 0.05 \
  --meta-safe-confidence-scale 1.0 \
  --meta-consistency-weight 0.1 \
  --meta-consistency-credit-weight 0.25 \
  --meta-fine-weight 1.0 --meta-family-weight 0.5 \
  --meta-basic-weight 0.5 --meta-relation-weight 1.0 \
  --relation-dim 64 --relation-hvp-samples 1 \
  --relation-contrastive-weight 0.1 --relation-temperature 0.1 \
  --router-hidden-dim 32 --router-prior 0.50 0.45 0.05 \
  2>&1 | tee ./output/bird_part_anchored_v86_seed0/train.log
```

## Evaluation commands

Evaluation uses `--resume` to load the complete V8.6 checkpoint. Do not add
`--finetune` to these commands.

### Aircraft evaluation on GPU 7

```bash
mkdir -p ./output/air_part_anchored_v86_seed0

CUDA_VISIBLE_DEVICES=7 python deit/main_hier_partial.py \
  --model deit_small_patch16_224 --batch-size 256 --num_workers 8 \
  --data-set AIR-HIER --data-path /raid/datasets/fgvc-aircraft \
  --output_dir ./output/air_part_anchored_v86_seed0 \
  --texts captions/air_caps.txt \
  --sp_proportion 0.3 --fm_proportion 0.6 \
  --seed 0 --random_seed 0 \
  --enable-bilevel --meta-scope counterfactual \
  --counterfactual-compose residual --num-parts 8 \
  --semantic-rank 64 --meta-adapter-rank 64 \
  --meta-policy-hidden 128 --relation-dim 64 \
  --router-hidden-dim 32 --router-prior 0.50 0.45 0.05 \
  --lam-cls 0.0 --lam-attr 1.0 --proto-align-weight 0.0 \
  --resume ./output/air_part_anchored_v86_seed0/best_checkpoint.pth \
  --filename ./output/air_part_anchored_v86_seed0/eval_detail.csv \
  --eval 2>&1 | tee ./output/air_part_anchored_v86_seed0/test_eval.log
```

### CUB evaluation on GPU 6

```bash
mkdir -p ./output/bird_part_anchored_v86_seed0

CUDA_VISIBLE_DEVICES=6 python deit/main_hier_partial.py \
  --model deit_small_patch16_224 --batch-size 256 --num_workers 8 \
  --data-set BIRD-HIER \
  --data-path /raid/datasets/cub-200/CUB_200_2011/images_split \
  --output_dir ./output/bird_part_anchored_v86_seed0 \
  --texts captions/cub_caps.txt \
  --sp_proportion 0.1 --fm_proportion 0.5 \
  --seed 0 --random_seed 0 \
  --enable-bilevel --meta-scope counterfactual \
  --counterfactual-compose residual --num-parts 8 \
  --semantic-rank 64 --meta-adapter-rank 64 \
  --meta-policy-hidden 128 --relation-dim 64 \
  --router-hidden-dim 32 --router-prior 0.50 0.45 0.05 \
  --lam-cls 0.0 --lam-attr 1.0 --proto-align-weight 0.0 \
  --resume ./output/bird_part_anchored_v86_seed0/best_checkpoint.pth \
  --filename ./output/bird_part_anchored_v86_seed0/eval_detail.csv \
  --eval 2>&1 | tee ./output/bird_part_anchored_v86_seed0/test_eval.log
```

## V8.5 compatibility and later dataset split

To reproduce V8.5 inside this branch, replace only:

```bash
--counterfactual-compose residual
```

with:

```bash
--counterfactual-compose competitive
```

Use a different output directory when comparing the two modes. Do not decide
the permanent CUB/Aircraft split before V8.6 finishes. After paired results,
the same V8.6 branch can run CUB with `competitive` and Aircraft with
`residual` if that is the Pareto-optimal policy.

## Checkpoint migration

V8.6 does not change parameter tensor shapes, but it changes the meaning of the
third virtual branch and its real-update weights in `residual` mode. For a
controlled V8.5-versus-V8.6 result, train from the same ImageNet/E2 source via
`--finetune`; do not resume a V8.5 optimizer state. Use `--resume` only to
continue or evaluate a checkpoint trained with the same composition mode. V7
`adaptive` checkpoints remain loadable through the retained legacy scope.

## Verification and diagnostics

```bash
python -m compileall -q deit
PYTHONPATH=deit python -m unittest -v deit/test_semantic_bilevel.py
```

In addition to V8.5 metrics, V8.6 logs:

```text
train_counterfactual_residual_mode
train_meta_relation_residual_inner_scale
train_meta_species_no_regret_accept_rate
train_meta_species_supervised_guard_rate
train_meta_species_anchor_guard_rate
train_meta_relation_incremental_task_improvement
train_meta_relation_incremental_classification_improvement
train_meta_relation_incremental_consistency_improvement
train_meta_real_part_weight
train_meta_real_relation_weight
```

For formal claims, run at least three paired seeds and choose hyperparameters
on a validation split rather than the official test set.

This repository derives from the official implementation of *Free-Grained
Hierarchical Visual Recognition*. Retain the upstream license and attribution.
