"""Executes user-written Python code and captures all tensors for visualization."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import io
import contextlib
from typing import Any
from .tensor_serializer import tensor_to_dict


# Allowed builtins for safety
_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "float": float, "int": int, "len": len,
    "list": list, "map": map, "max": max, "min": min, "print": print,
    "range": range, "round": round, "sorted": sorted, "str": str,
    "sum": sum, "tuple": tuple, "type": type, "zip": zip,
    "True": True, "False": False, "None": None,
    "__import__": None,  # block imports
}


def run_user_code(code: str) -> dict[str, Any]:
    """
    Execute Python code, capture all tensor variables and print output.

    Returns:
      - tensors: dict of variable_name -> serialized tensor
      - stdout: captured print output
      - error: error message if execution failed
    """
    # Prepare the execution namespace
    namespace: dict[str, Any] = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "np": np,
        "numpy": np,
    }

    stdout_capture = io.StringIO()
    error_msg = None

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, namespace)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"

    # Collect all tensor/array variables from namespace
    tensors: dict[str, Any] = {}
    for name, val in namespace.items():
        if name.startswith("_") or name in ("torch", "nn", "F", "np", "numpy"):
            continue
        if isinstance(val, torch.Tensor):
            t_dict = tensor_to_dict(val)
            t_dict["name"] = name
            tensors[name] = t_dict
        elif isinstance(val, np.ndarray):
            t_dict = {
                "data": val.tolist(),
                "shape": list(val.shape),
                "dtype": str(val.dtype),
                "name": name,
            }
            tensors[name] = t_dict

    # Build step-by-step from the code lines
    steps = []
    for name, t_data in tensors.items():
        steps.append({
            "title": name,
            "description": f"{t_data['dtype']} tensor with shape [{', '.join(str(s) for s in t_data['shape'])}]",
            "output": t_data,
        })

    return {
        "tensors": tensors,
        "steps": steps,
        "stdout": stdout_capture.getvalue(),
        "error": error_msg,
    }
