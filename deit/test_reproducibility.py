"""CPU regression tests for deterministic runs and exact resume semantics."""

import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from reproducibility import (
    capture_rng_state,
    configure_reproducibility,
    restore_rng_state,
    restore_scheduler_state,
    validate_run_paths,
)


class _SchedulerProbe:
    def __init__(self):
        self.loaded = None
        self.step_calls = []

    def load_state_dict(self, state):
        self.loaded = state

    def step(self, epoch):
        self.step_calls.append(epoch)


class ReproducibilityTest(unittest.TestCase):
    def test_rng_snapshot_round_trip(self):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        state = capture_rng_state()

        expected = (
            random.random(),
            np.random.rand(4),
            torch.rand(4),
        )
        random.random()
        np.random.rand(2)
        torch.rand(2)

        self.assertTrue(restore_rng_state(state, strict=True))
        actual = (
            random.random(),
            np.random.rand(4),
            torch.rand(4),
        )
        self.assertEqual(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_strict_resume_rejects_legacy_checkpoint(self):
        self.assertFalse(restore_rng_state(None, strict=False))
        with self.assertRaisesRegex(RuntimeError, "requires a V8.7.2 checkpoint"):
            restore_rng_state(None, strict=True)

    def test_scheduler_restore_does_not_advance_epoch(self):
        scheduler = _SchedulerProbe()
        state = {"last_epoch": 82, "value": 0.25}
        restored = restore_scheduler_state(
            scheduler,
            {"lr_scheduler": state},
        )
        self.assertTrue(restored)
        self.assertEqual(scheduler.loaded, state)
        self.assertEqual(scheduler.step_calls, [])

    def test_strict_configuration_is_auditable(self):
        previous_deterministic = (
            torch.are_deterministic_algorithms_enabled()
        )
        previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        try:
            with mock.patch.dict(
                os.environ,
                {
                    "PYTHONHASHSEED": "23",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                },
            ):
                settings = configure_reproducibility(23, strict=True)
            self.assertTrue(settings["strict"])
            self.assertTrue(settings["deterministic_algorithms"])
            self.assertFalse(settings["cuda_matmul_tf32"])
            self.assertFalse(settings["cudnn_tf32"])
        finally:
            torch.use_deterministic_algorithms(previous_deterministic)
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
            torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32

    def test_strict_configuration_requires_startup_hash_seed(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "PYTHONHASHSEED"):
                configure_reproducibility(0, strict=True)

    def test_run_path_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "log.txt").write_text("existing\n")

            with self.assertRaisesRegex(ValueError, "mix with existing"):
                validate_run_paths(
                    str(output), "", eval_only=False,
                )
            validate_run_paths(
                str(output), "", eval_only=False,
                allow_existing_output=True,
            )

        with self.assertRaisesRegex(ValueError, "must use checkpoint.pth"):
            validate_run_paths(
                "output/run",
                "output/run/best_checkpoint.pth",
                eval_only=False,
            )
        validate_run_paths(
            "output/run",
            "output/run/checkpoint.pth",
            eval_only=False,
        )
        validate_run_paths(
            "output/run",
            "output/run/best_checkpoint.pth",
            eval_only=True,
        )


if __name__ == "__main__":
    unittest.main()
