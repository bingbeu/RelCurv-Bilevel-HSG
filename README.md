# RelCurv-Bilevel-HSG V8.7.2

**Deterministic training and exact checkpoint resume for free-grained
hierarchical recognition**

V8.7.2 fixes the execution protocol without changing the Part, Relation,
router, task-feedback, or bilevel formulas.

The fix follows two controlled observations:

- CUB V8.5 and V8.7.1 were identical through epoch 4, then diverged when the
  second-order counterfactual update started at epoch 5. The frozen V8.5 tensor
  path was already source-identical, so strict CUDA determinism is now
  available and auditable.
- Aircraft V8.7 and a fresh V8.7.1 run were identical through epoch 82. The old
  V8.7 checkpoint records a training `resume` from its own
  `best_checkpoint.pth`; the old loader restored weights and optimizers but not
  RNG state, and advanced the restored scheduler one extra time. This explains
  the first divergence at epoch 83.

V8.7.2 therefore provides:

- Python, NumPy, Torch CPU, and Torch CUDA seeding;
- deterministic PyTorch algorithms, deterministic cuDNN, deterministic cuBLAS,
  and disabled TF32 under `--strict-reproducibility`;
- explicitly seeded DataLoader workers;
- RNG snapshots in both latest and best checkpoints;
- exact RNG restoration on resume;
- scheduler restoration without an extra `step(start_epoch)`;
- protection against training from `best_checkpoint.pth` by mistake;
- protection against mixing a fresh run with an existing `log.txt` or
  checkpoint.

The model presets remain unchanged:

| Dataset | Preset | Solver | Best-checkpoint metric |
|---|---|---|---|
| CUB (`BIRD-HIER`) | `cub-v85` | frozen V8.5 competitive solver | FPA |
| Aircraft (`AIR-HIER`) | `air-curvpart-v7` | full-strength Part path | Species Acc@1 |

The original V8.5 CUB and V8.7 Aircraft checkpoints are not deleted or
overwritten. The reported Aircraft `70.387` FPA checkpoint remains usable, but
it must be labelled as a resumed trajectory rather than a clean uninterrupted
100-epoch run.

See [`BILEVEL_SEMANTIC_LOOP.md`](BILEVEL_SEMANTIC_LOOP.md) for the optimization
and reproducibility invariants.

## Installation

Recommended: Python 3.10, PyTorch 2.1.2, torchvision 0.16.2, and CUDA 12.1.

```bash
conda create -n relcurv python=3.10
conda activate relcurv
pip install -r requirements.txt
pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121
export PYTHONPATH=deit/:deit/dataset/:$PYTHONPATH
```

Place `deit_small_patch16_224-cd65a155.pth` in the repository root. Fresh runs
must use `--finetune` from this ImageNet checkpoint and a new output directory.

## Fresh training on two GPUs

Run Aircraft in one terminal on GPU 7 and CUB in another terminal on GPU 6.
The two environment variables must be present when Python starts. Exact RNG
resume currently supports these single-process runs; strict distributed resume
is rejected rather than silently restoring rank 0 state on every rank.

### Aircraft on GPU 7

```bash
mkdir -p ./output/air_v872_strict_curvpart_v7_seed0

PYTHONHASHSEED=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES=7 \
python deit/main_hier_partial.py \
  --strict-reproducibility \
  --method-preset air-curvpart-v7 \
  --model deit_small_patch16_224 \
  --batch-size 256 --epochs 100 --lr 5e-4 \
  --weight-decay 0.05 --warmup-epochs 5 --num_workers 8 \
  --seed 0 --random_seed 0 \
  --data-set AIR-HIER \
  --data-path /raid/datasets/fgvc-aircraft \
  --output_dir ./output/air_v872_strict_curvpart_v7_seed0 \
  --filename final_epoch_eval.csv \
  --texts captions/air_caps.txt --sim_loss_weight 1 \
  --sp_proportion 0.3 --fm_proportion 0.6 \
  --finetune deit_small_patch16_224-cd65a155.pth \
  2>&1 | tee ./output/air_v872_strict_curvpart_v7_seed0/train.log
```

### CUB on GPU 6

```bash
mkdir -p ./output/bird_v872_strict_frozen_v85_seed0

PYTHONHASHSEED=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES=6 \
python deit/main_hier_partial.py \
  --strict-reproducibility \
  --method-preset cub-v85 \
  --model deit_small_patch16_224 \
  --batch-size 256 --epochs 100 --lr 5e-4 \
  --weight-decay 0.05 --warmup-epochs 5 --num_workers 8 \
  --seed 0 --random_seed 0 \
  --data-set BIRD-HIER \
  --data-path /raid/datasets/cub-200/CUB_200_2011/images_split \
  --output_dir ./output/bird_v872_strict_frozen_v85_seed0 \
  --filename final_epoch_eval.csv \
  --texts captions/cub_caps.txt --sim_loss_weight 1 \
  --sp_proportion 0.1 --fm_proportion 0.5 \
  --finetune deit_small_patch16_224-cd65a155.pth \
  2>&1 | tee ./output/bird_v872_strict_frozen_v85_seed0/train.log
```

Strict startup must print values equivalent to:

```text
"strict": true
"deterministic_algorithms": true
"cudnn_deterministic": true
"cudnn_benchmark": false
"cuda_matmul_tf32": false
"cudnn_tf32": false
```

CUB must additionally resolve:

```text
method_preset='cub-v85'
meta_scope='counterfactual'
counterfactual_compose='competitive'
counterfactual_solver='v85-frozen'
checkpoint_metric='fpa'
```

## Exact interrupted-run resume

Resume only the latest `checkpoint.pth`. Do not use `best_checkpoint.pth` for
continuation: that forks an earlier best epoch and is reserved for evaluation.

Strict resume is available only for checkpoints created by V8.7.2 because
older checkpoints do not contain RNG snapshots.

### Resume Aircraft on GPU 7

```bash
PYTHONHASHSEED=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES=7 \
python deit/main_hier_partial.py \
  --strict-reproducibility \
  --method-preset air-curvpart-v7 \
  --model deit_small_patch16_224 \
  --batch-size 256 --epochs 100 --lr 5e-4 \
  --weight-decay 0.05 --warmup-epochs 5 --num_workers 8 \
  --seed 0 --random_seed 0 \
  --data-set AIR-HIER \
  --data-path /raid/datasets/fgvc-aircraft \
  --output_dir ./output/air_v872_strict_curvpart_v7_seed0 \
  --filename final_epoch_eval.csv \
  --texts captions/air_caps.txt --sim_loss_weight 1 \
  --sp_proportion 0.3 --fm_proportion 0.6 \
  --resume ./output/air_v872_strict_curvpart_v7_seed0/checkpoint.pth \
  2>&1 | tee -a ./output/air_v872_strict_curvpart_v7_seed0/train.log
```

### Resume CUB on GPU 6

```bash
PYTHONHASHSEED=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES=6 \
python deit/main_hier_partial.py \
  --strict-reproducibility \
  --method-preset cub-v85 \
  --model deit_small_patch16_224 \
  --batch-size 256 --epochs 100 --lr 5e-4 \
  --weight-decay 0.05 --warmup-epochs 5 --num_workers 8 \
  --seed 0 --random_seed 0 \
  --data-set BIRD-HIER \
  --data-path /raid/datasets/cub-200/CUB_200_2011/images_split \
  --output_dir ./output/bird_v872_strict_frozen_v85_seed0 \
  --filename final_epoch_eval.csv \
  --texts captions/cub_caps.txt --sim_loss_weight 1 \
  --sp_proportion 0.1 --fm_proportion 0.5 \
  --resume ./output/bird_v872_strict_frozen_v85_seed0/checkpoint.pth \
  2>&1 | tee -a ./output/bird_v872_strict_frozen_v85_seed0/train.log
```

Training from `best_checkpoint.pth` is rejected by default. The override
`--allow-resume-best` exists only for an explicitly named trajectory-fork
ablation and must not be used for the main result.

## Evaluation commands

Evaluation intentionally loads `best_checkpoint.pth` with `--resume --eval`.
It does not restore optimizer/RNG state and is not blocked by the training
resume guard.

### Evaluate Aircraft on GPU 7

```bash
PYTHONHASHSEED=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES=7 \
python deit/main_hier_partial.py \
  --strict-reproducibility \
  --method-preset air-curvpart-v7 \
  --model deit_small_patch16_224 \
  --batch-size 256 --num_workers 8 \
  --data-set AIR-HIER \
  --data-path /raid/datasets/fgvc-aircraft \
  --output_dir ./output/air_v872_strict_curvpart_v7_seed0 \
  --texts captions/air_caps.txt \
  --sp_proportion 0.3 --fm_proportion 0.6 \
  --seed 0 --random_seed 0 \
  --resume ./output/air_v872_strict_curvpart_v7_seed0/best_checkpoint.pth \
  --filename ./output/air_v872_strict_curvpart_v7_seed0/eval_detail.csv \
  --eval 2>&1 | tee ./output/air_v872_strict_curvpart_v7_seed0/test_eval.log
```

### Evaluate CUB on GPU 6

```bash
PYTHONHASHSEED=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES=6 \
python deit/main_hier_partial.py \
  --strict-reproducibility \
  --method-preset cub-v85 \
  --model deit_small_patch16_224 \
  --batch-size 256 --num_workers 8 \
  --data-set BIRD-HIER \
  --data-path /raid/datasets/cub-200/CUB_200_2011/images_split \
  --output_dir ./output/bird_v872_strict_frozen_v85_seed0 \
  --texts captions/cub_caps.txt \
  --sp_proportion 0.1 --fm_proportion 0.5 \
  --seed 0 --random_seed 0 \
  --resume ./output/bird_v872_strict_frozen_v85_seed0/best_checkpoint.pth \
  --filename ./output/bird_v872_strict_frozen_v85_seed0/eval_detail.csv \
  --eval 2>&1 | tee ./output/bird_v872_strict_frozen_v85_seed0/test_eval.log
```

## Verification

Check the resolved solver and reproducibility fields:

```bash
tail -n 1 ./output/bird_v872_strict_frozen_v85_seed0/log.txt | \
  grep -o '"resolved_counterfactual_solver"[^,]*\|"strict_reproducibility"[^,]*\|"resume_rng_restored"[^,]*\|"checkpoint_metric"[^,]*'
```

Inspect checkpoint resume metadata:

```bash
python - <<'PY'
import torch

for path in [
    "output/air_v872_strict_curvpart_v7_seed0/checkpoint.pth",
    "output/bird_v872_strict_frozen_v85_seed0/checkpoint.pth",
]:
    checkpoint = torch.load(path, map_location="cpu")
    print(path)
    print("  epoch:", checkpoint.get("epoch"))
    print("  reproducibility_version:",
          checkpoint.get("reproducibility_version"))
    print("  rng_keys:", sorted(checkpoint.get("rng_state", {})))
PY
```

Run CPU verification:

```bash
python -m compileall -q deit
PYTHONPATH=deit python -m unittest -v \
  deit/test_reproducibility.py \
  deit/test_semantic_bilevel.py
```

For formal claims, use fresh strict runs, select methods using validation data,
and report at least three paired seeds. A resumed/forked checkpoint must not be
silently mixed with uninterrupted-run results.

This repository derives from the official implementation of *Free-Grained
Hierarchical Visual Recognition*. Retain the upstream license and attribution.
