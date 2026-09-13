# RelCurv-Bilevel-HSG V8.5

**Task-Feedback Adaptive-Granularity Curvature-Aware Semantic Grounding for
Free-Grained Hierarchical Recognition**

V8.5 unrolls `skip`, local-part alignment and relation alignment as three
separate virtual updates. A paired query view evaluates every branch
independently on Species, Family and Order. Positive-gain evidence decides
whether a semantic update is safe; the router learns the conditional
Part-versus-Relation choice independently for all three hierarchy levels.
Classification and taxonomy-consistency regrets are calibrated separately;
confidence chooses Part versus Relation without shrinking the fixed safe
non-Skip budget.

## Why V8.5

- The virtual adapter is part of the real inference path, so meta improvement
  can affect downstream predictions.
- Part and relation alignment update the **same** adapter.
- Relation descriptors are not concatenated into the classifier.
- The router no longer sees only one pre-mixed post-update scalar; it receives
  identifiable skip/part/relation outcomes for each hierarchy level.
- Each Part/Relation virtual gradient is L2-normalized independently, making
  `--meta-inner-lr` a bounded virtual step norm instead of an uncontrolled raw
  gradient multiplier.
- A stopped positive-gain mask owns the Skip decision. The learned Skip logit
  cannot collapse the real update to an all-Skip solution.
- Whenever a semantic branch is safe, a fixed 5% probability is distributed
  across eligible Part/Relation branches. Calibrated confidence reweights only
  this conditional choice and can no longer suppress the entire real update.
- Supervised classification regret and fixed-taxonomy consistency regret are
  RMS-calibrated independently. `--meta-consistency-credit-weight` gives
  relation updates explicit credit when they improve Species/Family/Order
  agreement even if their immediate CE gain is smaller.
- Branch regret is RMS-calibrated inside every example and hierarchy level with
  a noise floor. Positive subpopulations are no longer drowned out by a small
  or negative dataset-level average improvement.
- Counterfactual training checkpoints are selected by all-level FPA by default;
  legacy scopes retain Species Acc@1 selection.
- A fixed-taxonomy Jensen-Shannon term is a differentiable TICE surrogate and
  uses no unavailable training label.
- The relation embedding is low-rank (`384 -> 64` by default), avoiding the
  previous approximately one-million-parameter relation branch.
- Policy and conditional Part/Relation router parameters are updated only by
  the outer hypergradient; every weight in the real model loss is detached.
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

## Training: Aircraft on GPU 7

```bash
mkdir -p ./output/air_consistency_credit_v85_seed0

CUDA_VISIBLE_DEVICES=7 python deit/main_hier_partial.py \
  --model deit_small_patch16_224 \
  --batch-size 256 --epochs 100 --lr 5e-4 \
  --weight-decay 0.05 --warmup-epochs 5 --num_workers 8 \
  --seed 0 --random_seed 0 \
  --data-set AIR-HIER \
  --data-path /raid/datasets/fgvc-aircraft \
  --output_dir ./output/air_consistency_credit_v85_seed0 \
  --filename final_epoch_eval.csv \
  --texts captions/air_caps.txt --sim_loss_weight 1 \
  --sp_proportion 0.3 --fm_proportion 0.6 \
  --finetune deit_small_patch16_224-cd65a155.pth \
  --enable-bilevel --meta-scope counterfactual \
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
  2>&1 | tee ./output/air_consistency_credit_v85_seed0/train.log
```

## Training: CUB on GPU 6

```bash
mkdir -p ./output/bird_consistency_credit_v85_seed0

CUDA_VISIBLE_DEVICES=6 python deit/main_hier_partial.py \
  --model deit_small_patch16_224 \
  --batch-size 256 --epochs 100 --lr 5e-4 \
  --weight-decay 0.05 --warmup-epochs 5 --num_workers 8 \
  --seed 0 --random_seed 0 \
  --data-set BIRD-HIER \
  --data-path /raid/datasets/cub-200/CUB_200_2011/images_split \
  --output_dir ./output/bird_consistency_credit_v85_seed0 \
  --filename final_epoch_eval.csv \
  --texts captions/cub_caps.txt --sim_loss_weight 1 \
  --sp_proportion 0.1 --fm_proportion 0.5 \
  --finetune deit_small_patch16_224-cd65a155.pth \
  --enable-bilevel --meta-scope counterfactual \
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
  2>&1 | tee ./output/bird_consistency_credit_v85_seed0/train.log
```

## Evaluation: Aircraft best-FPA checkpoint on GPU 7

```bash
CUDA_VISIBLE_DEVICES=7 python deit/main_hier_partial.py \
  --model deit_small_patch16_224 --batch-size 256 --num_workers 8 \
  --data-set AIR-HIER --data-path /raid/datasets/fgvc-aircraft \
  --output_dir ./output/air_consistency_credit_v85_seed0 \
  --texts captions/air_caps.txt \
  --sp_proportion 0.3 --fm_proportion 0.6 \
  --seed 0 --random_seed 0 \
  --enable-bilevel --meta-scope counterfactual --num-parts 8 \
  --semantic-rank 64 --meta-adapter-rank 64 \
  --meta-policy-hidden 128 --relation-dim 64 \
  --router-hidden-dim 32 --router-prior 0.50 0.45 0.05 \
  --lam-cls 0.0 --lam-attr 1.0 --proto-align-weight 0.0 \
  --resume ./output/air_consistency_credit_v85_seed0/best_checkpoint.pth \
  --filename ./output/air_consistency_credit_v85_seed0/eval_detail.csv \
  --eval 2>&1 | tee ./output/air_consistency_credit_v85_seed0/test_eval.log
```

## Evaluation: CUB best-FPA checkpoint on GPU 6

```bash
CUDA_VISIBLE_DEVICES=6 python deit/main_hier_partial.py \
  --model deit_small_patch16_224 --batch-size 256 --num_workers 8 \
  --data-set BIRD-HIER \
  --data-path /raid/datasets/cub-200/CUB_200_2011/images_split \
  --output_dir ./output/bird_consistency_credit_v85_seed0 \
  --texts captions/cub_caps.txt \
  --sp_proportion 0.1 --fm_proportion 0.5 \
  --seed 0 --random_seed 0 \
  --enable-bilevel --meta-scope counterfactual --num-parts 8 \
  --semantic-rank 64 --meta-adapter-rank 64 \
  --meta-policy-hidden 128 --relation-dim 64 \
  --router-hidden-dim 32 --router-prior 0.50 0.45 0.05 \
  --lam-cls 0.0 --lam-attr 1.0 --proto-align-weight 0.0 \
  --resume ./output/bird_consistency_credit_v85_seed0/best_checkpoint.pth \
  --filename ./output/bird_consistency_credit_v85_seed0/eval_detail.csv \
  --eval 2>&1 | tee ./output/bird_consistency_credit_v85_seed0/test_eval.log
```

The defaults enable normalized inner gradients, per-example regret calibration,
the two-stage positive-gain gate, a fixed 5% safe non-Skip budget,
confidence-weighted Part/Relation allocation, independently calibrated
consistency credit, and automatic FPA checkpoint selection for
`counterfactual`. Their diagnostic ablations are:

```bash
--no-meta-inner-grad-normalization
--no-meta-router-regret-normalization
--no-meta-safe-gate
--no-meta-safe-confidence-routing
--meta-consistency-credit-weight 0
--meta-safe-route-budget 1.0
--checkpoint-metric acc1
```

## Checkpoint migration

The V8/V8.1/V8.2/V8.3/V8.4/V8.5 counterfactual router has nine logits instead
of V7's three. V8.3--V8.5 retain that tensor shape for compatibility but use
only the Part-to-Relation conditional ratio after the safety decision.
Do **not** resume a V7 optimizer/model state into `--meta-scope counterfactual`.
For a clean V8.5 comparison, start from the same E2 or ImageNet checkpoint with
`--finetune`; do not resume a V8.3/V8.4 optimizer state. Use `--resume` only for
a V8.5 checkpoint from the same configuration. The old `adaptive` scope is
retained so existing V7 checkpoints remain loadable.

## Verification and logging

```bash
python -m compileall -q deit
PYTHONPATH=deit python -m unittest -v deit/test_semantic_bilevel.py
```

Training logs expose aggregate routes plus `route_species_*`, `route_family_*`,
`route_order_*`, every branch's per-level improvement,
`meta_part_task_improvement`, `meta_relation_task_improvement`, policy entropy
and `meta_policy_grad_norm`. V8.5 additionally reports raw branch-gradient
norms, bounded virtual-step norms, calibrated/raw router regret, per-level
candidate rates, conditional Part/Relation probabilities, two-stage active
rates, fixed effective per-level budgets, budgeted safe routes, separate
classification/consistency improvements and regret RMS values, and per-epoch
FPA/TICE. Report at least three paired seeds and use a held-out validation split
for formal model selection rather than the test set.

This repository derives from the official implementation of *Free-Grained
Hierarchical Visual Recognition*. Retain the upstream license and attribution.
