# RelCurv-Bilevel-HSG V8.7

**Explicit Dataset-Validated Semantic Presets for Free-Grained Hierarchical
Recognition**

V8.7 follows the result of the V8.6 experiment instead of hiding it. The
part-anchored Relation residual reduced FPA on both datasets, suppressed the
Relation route by 89--96%, and reduced the meta-policy gradient by about 69%.
The residual path is therefore retained only as an ablation.

The recommended experiment now uses two explicit, reproducible presets:

| Dataset | Preset | Resolved method | Checkpoint metric |
|---|---|---|---|
| CUB (`BIRD-HIER`) | `cub-v85` | V8.5 all-level competitive counterfactual routing | FPA |
| Aircraft (`AIR-HIER`) | `air-curvpart-v7` | CurvPart V7 full-strength Part task-feedback | Species Acc@1 |

The preset must be selected on the command line. The model never silently
branches on a dataset name. A dataset compatibility check only prevents an
accidental command mismatch. The selected preset and resolved scope are saved
in checkpoints and written into every `log.txt` row.

## Resolved preset invariants

`cub-v85` freezes the best observed CUB path:

- `--meta-scope counterfactual`
- `--counterfactual-compose competitive`
- independent Skip/Part/Relation virtual branches at Species, Family and Order
- normalized virtual steps, fixed 5% safe non-Skip budget and consistency credit
- FPA checkpoint selection

`air-curvpart-v7` restores the verified Aircraft path:

- `--meta-scope part`
- raw one-step Part inner gradient, matching CurvPart-HSG V7
- full real Part route (`meta_real_part_weight = 1`), rather than a gated
  1--2% counterfactual route
- no Relation encoder, Relation policy or granularity router is constructed
- Species/Family/Order task-feedback remains in the query objective
- Species Acc@1 checkpoint selection, matching the historical V7 protocol

Both methods remain hierarchical: neither preset restricts learning to only a
fine or only a coarse level.

See [`BILEVEL_SEMANTIC_LOOP.md`](BILEVEL_SEMANTIC_LOOP.md) for the separation
and optimization invariants.

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

Place `deit_small_patch16_224-cd65a155.pth` in the repository root. Start both
controlled runs from this same pretrained checkpoint; do not resume a V8.5 or
V8.6 optimizer state.

## Training on two separate GPUs

Run Aircraft in one terminal on GPU 7 and CUB in another terminal on GPU 6.

### Aircraft / CurvPart V7 preset on GPU 7

```bash
mkdir -p ./output/air_v87_curvpart_v7_seed0

CUDA_VISIBLE_DEVICES=7 python deit/main_hier_partial.py \
  --method-preset air-curvpart-v7 \
  --model deit_small_patch16_224 \
  --batch-size 256 --epochs 100 --lr 5e-4 \
  --weight-decay 0.05 --warmup-epochs 5 --num_workers 8 \
  --seed 0 --random_seed 0 \
  --data-set AIR-HIER \
  --data-path /raid/datasets/fgvc-aircraft \
  --output_dir ./output/air_v87_curvpart_v7_seed0 \
  --filename final_epoch_eval.csv \
  --texts captions/air_caps.txt --sim_loss_weight 1 \
  --sp_proportion 0.3 --fm_proportion 0.6 \
  --finetune deit_small_patch16_224-cd65a155.pth \
  2>&1 | tee ./output/air_v87_curvpart_v7_seed0/train.log
```

### CUB / V8.5 competitive preset on GPU 6

```bash
mkdir -p ./output/bird_v87_cub_v85_seed0

CUDA_VISIBLE_DEVICES=6 python deit/main_hier_partial.py \
  --method-preset cub-v85 \
  --model deit_small_patch16_224 \
  --batch-size 256 --epochs 100 --lr 5e-4 \
  --weight-decay 0.05 --warmup-epochs 5 --num_workers 8 \
  --seed 0 --random_seed 0 \
  --data-set BIRD-HIER \
  --data-path /raid/datasets/cub-200/CUB_200_2011/images_split \
  --output_dir ./output/bird_v87_cub_v85_seed0 \
  --filename final_epoch_eval.csv \
  --texts captions/cub_caps.txt --sim_loss_weight 1 \
  --sp_proportion 0.1 --fm_proportion 0.5 \
  --finetune deit_small_patch16_224-cd65a155.pth \
  2>&1 | tee ./output/bird_v87_cub_v85_seed0/train.log
```

The preset resolves the complete method configuration before the model and
optimizers are created. Any conflicting method flags on the same command are
overridden and printed once at startup.

## Evaluation commands

Evaluation loads the complete checkpoint with `--resume`. Do not add
`--finetune`.

### Aircraft evaluation on GPU 7

```bash
mkdir -p ./output/air_v87_curvpart_v7_seed0

CUDA_VISIBLE_DEVICES=7 python deit/main_hier_partial.py \
  --method-preset air-curvpart-v7 \
  --model deit_small_patch16_224 \
  --batch-size 256 --num_workers 8 \
  --data-set AIR-HIER \
  --data-path /raid/datasets/fgvc-aircraft \
  --output_dir ./output/air_v87_curvpart_v7_seed0 \
  --texts captions/air_caps.txt \
  --sp_proportion 0.3 --fm_proportion 0.6 \
  --seed 0 --random_seed 0 \
  --resume ./output/air_v87_curvpart_v7_seed0/best_checkpoint.pth \
  --filename ./output/air_v87_curvpart_v7_seed0/eval_detail.csv \
  --eval 2>&1 | tee ./output/air_v87_curvpart_v7_seed0/test_eval.log
```

### CUB evaluation on GPU 6

```bash
mkdir -p ./output/bird_v87_cub_v85_seed0

CUDA_VISIBLE_DEVICES=6 python deit/main_hier_partial.py \
  --method-preset cub-v85 \
  --model deit_small_patch16_224 \
  --batch-size 256 --num_workers 8 \
  --data-set BIRD-HIER \
  --data-path /raid/datasets/cub-200/CUB_200_2011/images_split \
  --output_dir ./output/bird_v87_cub_v85_seed0 \
  --texts captions/cub_caps.txt \
  --sp_proportion 0.1 --fm_proportion 0.5 \
  --seed 0 --random_seed 0 \
  --resume ./output/bird_v87_cub_v85_seed0/best_checkpoint.pth \
  --filename ./output/bird_v87_cub_v85_seed0/eval_detail.csv \
  --eval 2>&1 | tee ./output/bird_v87_cub_v85_seed0/test_eval.log
```

## Verify the resolved method

At startup, Aircraft must print values equivalent to:

```text
method_preset='air-curvpart-v7'
meta_scope='part'
checkpoint_metric='acc1'
meta_inner_grad_normalization=False
```

CUB must print:

```text
method_preset='cub-v85'
meta_scope='counterfactual'
counterfactual_compose='competitive'
checkpoint_metric='fpa'
```

After training, verify the recorded configuration:

```bash
tail -n 1 ./output/air_v87_curvpart_v7_seed0/log.txt | \
  grep -o '"method_preset"[^,]*\|"resolved_meta_scope"[^,]*\|"checkpoint_metric"[^,]*'

tail -n 1 ./output/bird_v87_cub_v85_seed0/log.txt | \
  grep -o '"method_preset"[^,]*\|"resolved_meta_scope"[^,]*\|"resolved_counterfactual_compose"[^,]*\|"checkpoint_metric"[^,]*'
```

## Manual and ablation modes

`--method-preset manual` is the default and preserves every existing V7--V8.6
command. In particular, V8.6 residual remains available as an ablation:

```bash
--method-preset manual \
--enable-bilevel \
--meta-scope counterfactual \
--counterfactual-compose residual
```

Do not present the two recommended presets as per-sample automatic routing.
They are dataset-wise configurations and must be selected using validation
data. For formal claims, run at least three paired seeds.

## Verification

```bash
python -m compileall -q deit
PYTHONPATH=deit python -m unittest -v deit/test_semantic_bilevel.py
```

This repository derives from the official implementation of *Free-Grained
Hierarchical Visual Recognition*. Retain the upstream license and attribution.
