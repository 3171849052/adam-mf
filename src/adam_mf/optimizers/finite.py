"""Finite checks for tensors on the worker's single device."""
import torch


def all_finite(tensors):
    return torch.stack([torch.isfinite(t).all() for t in tensors]).all().item()
