# test_cuda.py

import torch
import pytest

def test_cuda_available():
    assert torch.cuda.is_available(), "CUDA is not available."

def test_cuda_device_count():
    count = torch.cuda.device_count()
    assert count > 0, f"No CUDA devices found. Device count: {count}"

def test_tensor_cuda():
    try:
        x = torch.tensor([1.0, 2.0]).to("cuda")
        y = torch.tensor([3.0, 4.0]).to("cuda")
        z = x + y
        assert z.is_cuda, "Resulting tensor is not on CUDA."
    except Exception as e:
        pytest.fail(f"Tensor CUDA operation failed: {e}")
