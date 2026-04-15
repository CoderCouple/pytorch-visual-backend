"""Converts PyTorch tensors to JSON-serializable format."""
import torch
import numpy as np
from typing import Any


def tensor_to_dict(t: torch.Tensor) -> dict[str, Any]:
    """Convert a tensor to a JSON-serializable dictionary."""
    data = t.detach().cpu()
    return {
        "data": data.tolist(),
        "shape": list(data.shape),
        "dtype": str(data.dtype).replace("torch.", ""),
        "requires_grad": data.requires_grad,
    }


def serialize_value(val: Any) -> Any:
    """Recursively serialize a value that may contain tensors."""
    if isinstance(val, torch.Tensor):
        return tensor_to_dict(val)
    if isinstance(val, np.ndarray):
        return {"data": val.tolist(), "shape": list(val.shape), "dtype": str(val.dtype)}
    if isinstance(val, (list, tuple)):
        return [serialize_value(v) for v in val]
    if isinstance(val, dict):
        return {k: serialize_value(v) for k, v in val.items()}
    if isinstance(val, (int, float, bool, str, type(None))):
        return val
    return str(val)
