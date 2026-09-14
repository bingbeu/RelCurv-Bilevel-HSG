# RelCurv-Bilevel-HSG V8.7.1

**Execution-Isolated Dataset Presets for Free-Grained Hierarchical
Recognition**

V8.7 produced the strongest observed Aircraft trade-off (`70.387` FPA and
`9.541` TICE), but the CUB `cub-v85` preset did not reproduce the original
V8.5 trajectory. The two runs were identical through epoch 4 and diverged
exactly when bilevel optimization started at epoch 5. Code inspection showed
that V8.6 had reordered shared competitive/residual operations even though the
competitive equations remained mathematically equivalent.

V8.7.1 therefore separates the solvers at execution level:

| Dataset | Preset | Solver | Checkpoint metric |
|---|---|---|---|
| CUB (`BIRD-HIER`) | `cub-v85` | byte-matched frozen V8.5 competitive solver | FPA |
| Aircraft (`AIR-HIER`) | `air-curvpart-v7` | unchanged V8.7 full-strength Part path | Species Acc@1 |

The CUB preset now resolves `--counterfactual-solver v85-frozen`. Its task
output validator, competitive meta solver and real semantic update are copied
from the original V8.5 source with the original tensor-operation order. It does
not construct the later Species-anchor graph or residual intermediates.

The Aircraft preset remains unchanged: it constructs no Relation encoder,
Relation policy or router; its real Part weight is one and its query objective
still covers Species, Family and Order. “Part-only” describes the semantic
update type, not a fine-level-only task.

The selection is explicit on the command line. Dataset validation catches an
accidental preset mismatch; the model never silently selects a method from a
dataset name. See [`BILEVEL_SEMANTIC_LOOP.md`](BILEVEL_SEMANTIC_LOOP.md) for
the optimization invariants.

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

Place `deit_small_patch16_224-cd65a155.pth` in the repository root. Both fresh
runs must start from that checkpoint; do not resume a V8.5--V8.7 optimizer
state.

## Training on two separate GPUs

Run Aircraft in one terminal on GPU 7 and CUB in another terminal on GPU 6.

### Aircraft / unchanged V8.7 Part path on GPU 7

The existing `air_v87_curvpart_v7_seed0/best_checkpoint.pth` remains valid. A
fresh controlled run uses:

```bash
mkdir -p ./output/air_v871_curvpart_v7_seed0

CUDA_VISIBLE_DEVICES=7 python deit/main_hier_partial.py \
  --method-preset air-curvpart-v7 \
  --model deit_small_patch16_224 \
  --batch-size 256 --epochs 100 --lr 5e-4 \
  --weight-decay 0.05 --warmup-epochs 5 --num_workers 8 \
  --seed 0 --random_seed 0 \
  --data-set AIR-HIER \
  --data-path /raid/datasets/fgvc-aircraft \
  --output_dir ./output/air_v871_curvpart_v7_seed0 \
  --filename final_epoch_eval.csv \
  --texts captions/air_caps.txt --sim_loss_weight 1 \
  --sp_proportion 0.3 --fm_proportion 0.6 \
  --finetune deit_small_patch16_224-cd65a155.pth \
  2>&1 | tee ./output/air_v871_curvpart_v7_seed0/train.log
```

### CUB / execution-frozen V8.5 solver on GPU 6

```bash
mkdir -p ./output/bird_v871_frozen_v85_seed0

CUDA_VISIBLE_DEVICES=6 python deit/main_hier_partial.py \
  --method-preset cub-v85 \
  --model deit_small_patch16_224 \
  --batch-size 256 --epochs 100 --lr 5e-4 \
  --weight-decay 0.05 --warmup-epochs 5 --num_workers 8 \
  --seed 0 --random_seed 0 \
  --data-set BIRD-HIER \
  --data-path /raid/datasets/cub-200/CUB_200_2011/images_split \
  --output_dir ./output/bird_v871_frozen_v85_seed0 \
  --filename final_epoch_eval.csv \
  --texts captions/cub_caps.txt --sim_loss_weight 1 \
  --sp_proportion 0.1 --fm_proportion 0.5 \
  --finetune deit_small_patch16_224-cd65a155.pth \
  2>&1 | tee ./output/bird_v871_frozen_v85_seed0/train.log
```

At startup, CUB must print values equivalent to:

```text
method_preset='cub-v85'
meta_scope='counterfactual'
counterfactual_compose='competitive'
counterfactual_solver='v85-frozen'
checkpoint_metric='fpa'
```

## Evaluation commands

Evaluation loads the complete checkpoint using `--resume`; do not add
`--finetune`.

### Aircraft evaluation on GPU 7

```bash
mkdir -p ./output/air_v871_curvpart_v7_seed0

CUDA_VISIBLE_DEVICES=7 python deit/main_hier_partial.py \
  --method-preset air-curvpart-v7 \
  --model deit_small_patch16_224 \
  --batch-size 256 --num_workers 8 \
  --data-set AIR-HIER \
  --data-path /raid/datasets/fgvc-aircraft \
  --output_dir ./output/air_v871_curvpart_v7_seed0 \
  --texts captions/air_caps.txt \
  --sp_proportion 0.3 --fm_proportion 0.6 \
  --seed 0 --random_seed 0 \
  --resume ./output/air_v871_curvpart_v7_seed0/best_checkpoint.pth \
  --filename ./output/air_v871_curvpart_v7_seed0/eval_detail.csv \
  --eval 2>&1 | tee ./output/air_v871_curvpart_v7_seed0/test_eval.log
```

To evaluate the already completed V8.7 Aircraft run instead, replace every
`air_v871_curvpart_v7_seed0` above with `air_v87_curvpart_v7_seed0`.

### CUB evaluation on GPU 6

```bash
mkdir -p ./output/bird_v871_frozen_v85_seed0

CUDA_VISIBLE_DEVICES=6 python deit/main_hier_partial.py \
  --method-preset cub-v85 \
  --model deit_small_patch16_224 \
  --batch-size 256 --num_workers 8 \
  --data-set BIRD-HIER \
  --data-path /raid/datasets/cub-200/CUB_200_2011/images_split \
  --output_dir ./output/bird_v871_frozen_v85_seed0 \
  --texts captions/cub_caps.txt \
  --sp_proportion 0.1 --fm_proportion 0.5 \
  --seed 0 --random_seed 0 \
  --resume ./output/bird_v871_frozen_v85_seed0/best_checkpoint.pth \
  --filename ./output/bird_v871_frozen_v85_seed0/eval_detail.csv \
  --eval 2>&1 | tee ./output/bird_v871_frozen_v85_seed0/test_eval.log
```

## Verify solver selection

```bash
tail -n 1 ./output/bird_v871_frozen_v85_seed0/log.txt | \
  grep -o '"method_preset"[^,]*\|"resolved_meta_scope"[^,]*\|"resolved_counterfactual_compose"[^,]*\|"resolved_counterfactual_solver"[^,]*\|"checkpoint_metric"[^,]*'

tail -n 1 ./output/air_v871_curvpart_v7_seed0/log.txt | \
  grep -o '"method_preset"[^,]*\|"resolved_meta_scope"[^,]*\|"resolved_counterfactual_solver"[^,]*\|"checkpoint_metric"[^,]*'
```

`cub-v85` should also log:

```text
train_counterfactual_solver_v85_frozen: 1.0
```

## Manual and ablation modes

`--method-preset manual` preserves prior commands. Solver combinations are
explicitly validated:

```bash
# Original V8.5 execution path
--counterfactual-solver v85-frozen \
--meta-scope counterfactual \
--counterfactual-compose competitive

# V8.6 residual-capable path
--counterfactual-solver unified \
--meta-scope counterfactual \
--counterfactual-compose residual
```

`v85-frozen` rejects residual composition. For formal claims, select presets
using validation data and report at least three paired seeds.

## Verification

```bash
python -m compileall -q deit
PYTHONPATH=deit python -m unittest -v deit/test_semantic_bilevel.py
```

This repository derives from the official implementation of *Free-Grained
Hierarchical Visual Recognition*. Retain the upstream license and attribution.
