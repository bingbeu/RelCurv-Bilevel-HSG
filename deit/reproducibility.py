"""Reproducibility and checkpoint-resume helpers for hierarchical training.

The training entry point intentionally keeps these helpers independent from
the model and optimizer implementation.  That makes the resume invariants
testable on CPU and prevents a reproducibility fix from changing the bilevel
objective itself.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping, MutableMapping
from urllib.parse import urlparse

import numpy as np
import torch


_CUBLAS_WORKSPACE_CONFIGS = (":4096:8", ":16:8")


def configure_reproducibility(seed: int, strict: bool = False) -> Mapping[str, Any]:
    """Seed every process-level RNG and optionally require deterministic CUDA.

    ``PYTHONHASHSEED`` only takes effect when the Python interpreter starts, so
    strict mode validates it instead of pretending that setting it here would
    be sufficient. Strict mode validates that both startup environment
    variables were exported before the entry point creates a CUDA tensor.
    """
    if strict:
        python_hash_seed = os.environ.get("PYTHONHASHSEED")
        if python_hash_seed != str(seed):
            raise RuntimeError(
                "strict reproducibility requires PYTHONHASHSEED to match "
                f"--seed before Python starts; expected {seed!r}, got "
                f"{python_hash_seed!r}"
            )

        workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace_config is None:
            raise RuntimeError(
                "strict reproducibility requires CUBLAS_WORKSPACE_CONFIG "
                "before Python starts (for example :4096:8)"
            )
        if workspace_config not in _CUBLAS_WORKSPACE_CONFIGS:
            raise RuntimeError(
                "strict reproducibility requires CUBLAS_WORKSPACE_CONFIG to "
                f"be one of {_CUBLAS_WORKSPACE_CONFIGS}, got "
                f"{workspace_config!r}"
            )

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if strict:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    return {
        "seed": seed,
        "strict": bool(strict),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }


def seed_data_worker(_worker_id: int) -> None:
    """Seed Python and NumPy from the worker seed assigned by DataLoader."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_rng_state() -> MutableMapping[str, Any]:
    """Capture all RNG state needed to continue at the next epoch."""
    state: MutableMapping[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(
    state: Mapping[str, Any] | None,
    *,
    strict: bool = False,
) -> bool:
    """Restore an epoch-boundary RNG snapshot.

    Legacy checkpoints remain loadable in normal mode.  Strict resume refuses
    them because silently reseeding cannot reproduce the next data order,
    augmentation sequence, dropout masks, or stochastic meta operations.
    """
    if not state:
        if strict:
            raise RuntimeError(
                "strict resume requires a V8.7.2 checkpoint containing "
                "rng_state; start a fresh strict run or resume without "
                "--strict-reproducibility"
            )
        return False

    required = ("python", "numpy", "torch_cpu")
    missing = [name for name in required if name not in state]
    if missing:
        raise RuntimeError(
            "checkpoint RNG state is incomplete; missing " + ", ".join(missing)
        )

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])

    cuda_state = state.get("torch_cuda")
    if torch.cuda.is_available():
        if cuda_state is None:
            if strict:
                raise RuntimeError(
                    "strict CUDA resume requires torch_cuda RNG state"
                )
        else:
            expected = torch.cuda.device_count()
            if len(cuda_state) != expected:
                raise RuntimeError(
                    "checkpoint CUDA RNG state count does not match visible "
                    f"devices: checkpoint={len(cuda_state)}, visible={expected}"
                )
            torch.cuda.set_rng_state_all(cuda_state)
    return True


def restore_scheduler_state(scheduler: Any, checkpoint: Mapping[str, Any]) -> bool:
    """Load the saved scheduler without advancing it a second time."""
    state = checkpoint.get("lr_scheduler")
    if state is None:
        return False
    scheduler.load_state_dict(state)
    return True


def validate_run_paths(
    output_dir: str,
    resume: str,
    *,
    eval_only: bool,
    allow_resume_best: bool = False,
    allow_existing_output: bool = False,
) -> None:
    """Prevent accidental trajectory forks and mixed fresh-run logs."""
    if resume and not eval_only:
        parsed = urlparse(resume)
        resume_name = Path(parsed.path).name
        if resume_name == "best_checkpoint.pth" and not allow_resume_best:
            raise ValueError(
                "training resume must use checkpoint.pth (the latest epoch), "
                "not best_checkpoint.pth; pass --allow-resume-best only for "
                "an explicitly named trajectory-fork experiment"
            )

    if not output_dir or eval_only or resume or allow_existing_output:
        return

    directory = Path(output_dir)
    collisions = [
        path.name
        for path in (
            directory / "log.txt",
            directory / "checkpoint.pth",
            directory / "best_checkpoint.pth",
        )
        if path.exists()
    ]
    if collisions:
        raise ValueError(
            "fresh training would mix with existing run artifacts in "
            f"{directory}: {', '.join(collisions)}; choose a new output_dir, "
            "resume checkpoint.pth, or explicitly pass "
            "--allow-existing-output"
        )
