"""Executes PyTorch operations and returns step-by-step results."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from .tensor_serializer import tensor_to_dict, serialize_value


def execute_operation(op_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Execute a PyTorch operation and return step-by-step results."""
    handler = OPERATION_HANDLERS.get(op_name)
    if not handler:
        raise ValueError(f"Unknown operation: {op_name}")
    return handler(params)


# --- Tensor Basics ---

def _tensor_create(params: dict) -> dict:
    data = params.get("data", [[1, 2], [3, 4]])
    t = torch.tensor(data, dtype=torch.float32)
    ndim = t.dim()
    dim_desc = f"{ndim}D tensor" if ndim <= 3 else f"{ndim}D tensor"
    return {
        "steps": [
            {
                "title": "Create Tensor",
                "description": f"Creating {dim_desc} from input data with shape {list(t.shape)}",
                "output": tensor_to_dict(t),
            },
            {
                "title": "Tensor Properties",
                "description": f"dtype: {t.dtype}, ndim: {ndim}, numel: {t.numel()}, contiguous: {t.is_contiguous()}",
                "output": {
                    "dtype": str(t.dtype).replace("torch.", ""),
                    "ndim": ndim,
                    "numel": t.numel(),
                    "contiguous": t.is_contiguous(),
                },
            },
        ],
        "result": tensor_to_dict(t),
    }


def _tensor_zeros(params: dict) -> dict:
    shape = params.get("shape", [3, 3])
    t = torch.zeros(shape)
    return {
        "steps": [
            {"title": "Create Zero Tensor", "description": f"torch.zeros({shape}) — all elements initialized to 0", "output": tensor_to_dict(t)}
        ],
        "result": tensor_to_dict(t),
    }


def _tensor_rand(params: dict) -> dict:
    shape = params.get("shape", [3, 3])
    t = torch.rand(shape)
    ndim = t.dim()
    return {
        "steps": [
            {"title": "Create Random Tensor", "description": f"torch.rand({shape}) — {ndim}D tensor with uniform random values in [0, 1)", "output": tensor_to_dict(t)}
        ],
        "result": tensor_to_dict(t),
    }


def _tensor_ones(params: dict) -> dict:
    shape = params.get("shape", [3, 3])
    t = torch.ones(shape)
    ndim = t.dim()
    return {
        "steps": [
            {"title": "Create Ones Tensor", "description": f"torch.ones({shape}) — {ndim}D tensor with all elements set to 1", "output": tensor_to_dict(t)}
        ],
        "result": tensor_to_dict(t),
    }


def _tensor_arange(params: dict) -> dict:
    start = params.get("start", 0)
    end = params.get("end", 12)
    step = params.get("step", 1)
    t = torch.arange(start, end, step, dtype=torch.float32)

    steps = [
        {"title": "Create Range", "description": f"torch.arange({start}, {end}, {step}) — 1D tensor with values from {start} to {end} (exclusive), step {step}", "output": tensor_to_dict(t)},
    ]

    # If a reshape shape is provided, show the reshape step too
    reshape_to = params.get("reshape")
    if reshape_to:
        reshaped = t.reshape(reshape_to)
        steps.append({
            "title": "Reshape",
            "description": f"Reshape from {list(t.shape)} to {reshape_to} — now a {reshaped.dim()}D tensor",
            "output": tensor_to_dict(reshaped),
        })
        return {"steps": steps, "result": tensor_to_dict(reshaped)}

    return {"steps": steps, "result": tensor_to_dict(t)}


# --- Math Operations ---

def _add(params: dict) -> dict:
    a_data = params.get("a", [[1, 2], [3, 4]])
    b_data = params.get("b", [[10, 20], [30, 40]])
    a = torch.tensor(a_data, dtype=torch.float32)
    b = torch.tensor(b_data, dtype=torch.float32)

    steps = [
        {"title": "Input A", "description": f"First tensor with shape {list(a.shape)}", "output": tensor_to_dict(a)},
        {"title": "Input B", "description": f"Second tensor with shape {list(b.shape)}", "output": tensor_to_dict(b)},
    ]

    # Check broadcasting
    if a.shape != b.shape:
        a_broad, b_broad = torch.broadcast_tensors(a, b)
        steps.append({
            "title": "Broadcasting",
            "description": f"Shapes {list(a.shape)} and {list(b.shape)} broadcast to {list(a_broad.shape)}",
            "output": {"a_broadcast": tensor_to_dict(a_broad), "b_broadcast": tensor_to_dict(b_broad)},
        })

    result = torch.add(a, b)
    steps.append({
        "title": "Element-wise Addition",
        "description": "Each element: result[i,j] = a[i,j] + b[i,j]",
        "output": tensor_to_dict(result),
    })

    return {"steps": steps, "result": tensor_to_dict(result)}


# --- Linear Algebra ---

def _matmul(params: dict) -> dict:
    a_data = params.get("a", [[1, 2], [3, 4]])
    b_data = params.get("b", [[5, 6], [7, 8]])
    a = torch.tensor(a_data, dtype=torch.float32)
    b = torch.tensor(b_data, dtype=torch.float32)

    steps = [
        {"title": "Input A", "description": f"Matrix A with shape {list(a.shape)}", "output": tensor_to_dict(a)},
        {"title": "Input B", "description": f"Matrix B with shape {list(b.shape)}", "output": tensor_to_dict(b)},
    ]

    # Show intermediate dot products
    result = torch.matmul(a, b)
    intermediates = []
    if a.dim() == 2 and b.dim() == 2:
        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
                row = a[i].tolist()
                col = b[:, j].tolist()
                products = [a[i][k].item() * b[k][j].item() for k in range(a.shape[1])]
                intermediates.append({
                    "row": i, "col": j,
                    "row_values": row, "col_values": col,
                    "products": products,
                    "sum": result[i][j].item(),
                })

    steps.append({
        "title": "Dot Products",
        "description": f"For each (i,j): sum of row_i(A) × col_j(B)",
        "output": {"intermediates": intermediates},
    })
    steps.append({
        "title": "Result",
        "description": f"Result matrix with shape {list(result.shape)}",
        "output": tensor_to_dict(result),
    })

    return {"steps": steps, "result": tensor_to_dict(result)}


# --- Reshape ---

def _reshape(params: dict) -> dict:
    data = params.get("data", [[1, 2, 3], [4, 5, 6]])
    new_shape = params.get("new_shape", [3, 2])
    t = torch.tensor(data, dtype=torch.float32)
    result = t.reshape(new_shape)

    steps = [
        {"title": "Input", "description": f"Original tensor with shape {list(t.shape)}", "output": tensor_to_dict(t)},
        {"title": "Flatten Order", "description": f"Elements in row-major order: {t.flatten().tolist()}", "output": {"flat": t.flatten().tolist()}},
        {"title": "Reshape", "description": f"Rearrange into shape {new_shape}", "output": tensor_to_dict(result)},
    ]

    return {"steps": steps, "result": tensor_to_dict(result)}


# --- Indexing ---

def _indexing(params: dict) -> dict:
    data = params.get("data", [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    index_expr = params.get("index", "0:2, 1:")
    t = torch.tensor(data, dtype=torch.float32)

    # Safe evaluation of indexing
    try:
        idx = eval(f"t[{index_expr}]")
    except Exception as e:
        return {"error": str(e), "steps": [], "result": None}

    # Build mask of selected indices
    mask = torch.zeros_like(t, dtype=torch.bool)
    try:
        mask_idx = eval(f"mask[{index_expr}]")
        exec(f"mask[{index_expr}] = True")
    except Exception:
        pass

    steps = [
        {"title": "Input", "description": f"Tensor with shape {list(t.shape)}", "output": tensor_to_dict(t)},
        {"title": "Selection", "description": f"Indexing with [{index_expr}]", "output": {"mask": mask.tolist()}},
        {"title": "Result", "description": f"Selected elements with shape {list(idx.shape)}", "output": tensor_to_dict(idx)},
    ]

    return {"steps": steps, "result": tensor_to_dict(idx)}


# --- Activations ---

def _relu(params: dict) -> dict:
    data = params.get("data", [[-2, -1, 0, 1, 2]])
    t = torch.tensor(data, dtype=torch.float32)
    result = F.relu(t)

    # Generate curve points for visualization
    x_vals = torch.linspace(-5, 5, 100)
    y_vals = F.relu(x_vals)

    steps = [
        {"title": "Input", "description": f"Input tensor", "output": tensor_to_dict(t)},
        {
            "title": "Apply ReLU",
            "description": "ReLU(x) = max(0, x) — zeroes out negative values",
            "output": {
                "result": tensor_to_dict(result),
                "curve": {"x": x_vals.tolist(), "y": y_vals.tolist()},
                "points": {"x": t.flatten().tolist(), "y": result.flatten().tolist()},
            },
        },
    ]

    return {"steps": steps, "result": tensor_to_dict(result)}


OPERATION_HANDLERS = {
    "tensor_create": _tensor_create,
    "tensor_zeros": _tensor_zeros,
    "tensor_ones": _tensor_ones,
    "tensor_rand": _tensor_rand,
    "tensor_arange": _tensor_arange,
    "add": _add,
    "matmul": _matmul,
    "reshape": _reshape,
    "indexing": _indexing,
    "relu": _relu,
}
