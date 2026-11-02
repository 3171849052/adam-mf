import os
import torch
import pytest
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
@pytest.fixture(autouse=True)
def bounded_threads():torch.set_num_threads(2)
