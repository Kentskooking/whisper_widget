"""Device selection must respect user configuration without GPU identity checks."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from app.bark_backend import configure_environment, resolve_device, resolve_dtype


class DeviceConfigTests(unittest.TestCase):
    def runtime(self, available=True, count=3, bf16=False):
        return SimpleNamespace(
            device=torch.device, bfloat16=torch.bfloat16,
            float16=torch.float16, float32=torch.float32,
            cuda=SimpleNamespace(is_available=Mock(return_value=available),
                                 device_count=Mock(return_value=count),
                                 is_bf16_supported=Mock(return_value=bf16)),
        )

    def test_auto_uses_visible_cuda_or_cpu(self):
        self.assertEqual(resolve_device(self.runtime()), "cuda:0")
        self.assertEqual(resolve_device(self.runtime(available=False)), "cpu")

    def test_explicit_device_is_honored_and_unavailable_device_fails(self):
        runtime = self.runtime()
        self.assertEqual(resolve_device(runtime, "cuda:2"), "cuda:2")
        self.assertEqual(resolve_device(runtime, "cpu"), "cpu")
        with self.assertRaises(RuntimeError):
            resolve_device(runtime, "cuda:3")
        with self.assertRaises(RuntimeError):
            resolve_device(self.runtime(available=False), "cuda")

    def test_dtype_does_not_require_bf16_hardware(self):
        self.assertIs(resolve_dtype(self.runtime(), "cuda:0"), torch.float32)
        self.assertIs(resolve_dtype(self.runtime(bf16=True), "cuda:0"), torch.bfloat16)
        runtime = self.runtime(bf16=True)
        self.assertIs(resolve_dtype(runtime, "cpu"), torch.float32)
        runtime.cuda.is_bf16_supported.assert_not_called()
        with self.assertRaises(ValueError):
            resolve_dtype(self.runtime(), "cuda:0", "bfloat16")
        with self.assertRaises(ValueError):
            resolve_dtype(self.runtime(), "cpu", "float16")

    def test_environment_configuration_preserves_user_devices_and_cache(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,0", "HF_HOME": "custom-cache"}):
            configure_environment()
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2,0")
            self.assertEqual(os.environ["HF_HOME"], "custom-cache")
