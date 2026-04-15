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


# --- Shape Operations ---

def _shape_size(params: dict) -> dict:
    data = params.get("data", [[1, 2, 3], [4, 5, 6]])
    t = torch.tensor(data, dtype=torch.float32)
    dim_sizes = {f"dim {i}": s for i, s in enumerate(t.shape)}
    return {
        "steps": [
            {"title": "Tensor", "description": f"Input tensor with shape {list(t.shape)}", "output": tensor_to_dict(t)},
            {
                "title": "Shape Info",
                "description": f"shape={list(t.shape)}, size={t.numel()}, ndim={t.dim()}",
                "output": {
                    "shape": list(t.shape),
                    "size": t.numel(),
                    "ndim": t.dim(),
                    "numel": t.numel(),
                    "dim_sizes": dim_sizes,
                },
            },
        ],
        "result": tensor_to_dict(t),
    }


def _view(params: dict) -> dict:
    data = params.get("data", [[1, 2, 3], [4, 5, 6]])
    new_shape = params.get("new_shape", [3, 2])
    t = torch.tensor(data, dtype=torch.float32)
    result = t.view(new_shape)
    return {
        "steps": [
            {"title": "Input", "description": f"Original tensor with shape {list(t.shape)}", "output": tensor_to_dict(t)},
            {"title": "Flatten Order", "description": f"Elements in row-major order: {t.flatten().tolist()}", "output": {"flat": t.flatten().tolist()}},
            {"title": "Reshape", "description": f"Rearrange into shape {new_shape}", "output": tensor_to_dict(result)},
        ],
        "result": tensor_to_dict(result),
    }


def _permute(params: dict) -> dict:
    data = params.get("data", [[[1, 2], [3, 4], [5, 6]]])
    dims = params.get("dims", [2, 0, 1])
    t = torch.tensor(data, dtype=torch.float32)
    result = t.permute(dims)
    axis_map = {f"new dim {i}": f"old dim {dims[i]}" for i in range(len(dims))}
    return {
        "steps": [
            {"title": "Input", "description": f"Tensor with shape {list(t.shape)}", "output": tensor_to_dict(t)},
            {
                "title": "Axis Reorder",
                "description": f"Permute dims {list(range(len(dims)))} → {dims}",
                "output": {
                    "dims": dims,
                    "axis_map": axis_map,
                    "original_shape": list(t.shape),
                    "new_shape": list(result.shape),
                },
            },
            {"title": "Result", "description": f"Permuted tensor with shape {list(result.shape)}", "output": tensor_to_dict(result)},
        ],
        "result": tensor_to_dict(result),
    }


def _unsqueeze_squeeze(params: dict) -> dict:
    data = params.get("data", [[1, 2, 3], [4, 5, 6]])
    t = torch.tensor(data, dtype=torch.float32)
    unsqueezed = t.unsqueeze(0)
    squeezed = unsqueezed.squeeze(0)
    return {
        "steps": [
            {"title": "Input", "description": f"Tensor with shape {list(t.shape)}", "output": tensor_to_dict(t)},
            {"title": "Unsqueeze", "description": f"unsqueeze(0): {list(t.shape)} → {list(unsqueezed.shape)} — adds dim at position 0", "output": {**tensor_to_dict(unsqueezed), "new_shape": list(unsqueezed.shape)}},
            {"title": "Squeeze", "description": f"squeeze(0): {list(unsqueezed.shape)} → {list(squeezed.shape)} — removes dim of size 1", "output": {**tensor_to_dict(squeezed), "new_shape": list(squeezed.shape)}},
        ],
        "result": tensor_to_dict(squeezed),
    }


# --- More Math Operations ---

def _mul(params: dict) -> dict:
    a_data = params.get("a", [[1, 2], [3, 4]])
    b_data = params.get("b", [[10, 20], [30, 40]])
    a = torch.tensor(a_data, dtype=torch.float32)
    b = torch.tensor(b_data, dtype=torch.float32)

    steps = [
        {"title": "Input A", "description": f"First tensor with shape {list(a.shape)}", "output": tensor_to_dict(a)},
        {"title": "Input B", "description": f"Second tensor with shape {list(b.shape)}", "output": tensor_to_dict(b)},
    ]

    if a.shape != b.shape:
        a_broad, b_broad = torch.broadcast_tensors(a, b)
        steps.append({
            "title": "Broadcasting",
            "description": f"Shapes {list(a.shape)} and {list(b.shape)} broadcast to {list(a_broad.shape)}",
            "output": {"a_broadcast": tensor_to_dict(a_broad), "b_broadcast": tensor_to_dict(b_broad)},
        })

    result = torch.mul(a, b)
    steps.append({
        "title": "Element-wise Multiplication",
        "description": "Each element: result[i,j] = a[i,j] * b[i,j]",
        "output": tensor_to_dict(result),
    })

    return {"steps": steps, "result": tensor_to_dict(result)}


def _sum(params: dict) -> dict:
    data = params.get("data", [[1, 2, 3], [4, 5, 6]])
    dim = params.get("dim", 1)
    t = torch.tensor(data, dtype=torch.float32)
    result = t.sum(dim=dim)

    # Build intermediates showing per-row/col computation
    intermediates = []
    if t.dim() == 2:
        if dim == 1:
            for i in range(t.shape[0]):
                vals = t[i].tolist()
                intermediates.append({"index": i, "values": vals, "sum": sum(vals)})
        else:
            for j in range(t.shape[1]):
                vals = t[:, j].tolist()
                intermediates.append({"index": j, "values": vals, "sum": sum(vals)})

    return {
        "steps": [
            {"title": "Input", "description": f"Tensor with shape {list(t.shape)}", "output": tensor_to_dict(t)},
            {
                "title": "Reduction",
                "description": f"Sum along dim={dim}: {list(t.shape)} → {list(result.shape)}",
                "output": {
                    "dim": dim,
                    "intermediates": intermediates,
                    "operation": "sum",
                },
            },
            {"title": "Result", "description": f"Result with shape {list(result.shape)}", "output": tensor_to_dict(result)},
        ],
        "result": tensor_to_dict(result),
    }


def _mean(params: dict) -> dict:
    data = params.get("data", [[1, 2, 3], [4, 5, 6]])
    dim = params.get("dim", 1)
    t = torch.tensor(data, dtype=torch.float32)
    result = t.mean(dim=dim)

    intermediates = []
    if t.dim() == 2:
        if dim == 1:
            for i in range(t.shape[0]):
                vals = t[i].tolist()
                intermediates.append({"index": i, "values": vals, "mean": sum(vals) / len(vals)})
        else:
            for j in range(t.shape[1]):
                vals = t[:, j].tolist()
                intermediates.append({"index": j, "values": vals, "mean": sum(vals) / len(vals)})

    return {
        "steps": [
            {"title": "Input", "description": f"Tensor with shape {list(t.shape)}", "output": tensor_to_dict(t)},
            {
                "title": "Reduction",
                "description": f"Mean along dim={dim}: {list(t.shape)} → {list(result.shape)}",
                "output": {
                    "dim": dim,
                    "intermediates": intermediates,
                    "operation": "mean",
                },
            },
            {"title": "Result", "description": f"Result with shape {list(result.shape)}", "output": tensor_to_dict(result)},
        ],
        "result": tensor_to_dict(result),
    }


# --- Autograd ---

def _autograd(params: dict) -> dict:
    x_data = params.get("x", [1.0, 2.0, 3.0])
    x = torch.tensor(x_data, dtype=torch.float32, requires_grad=True)
    y = x ** 2
    z = y.sum()
    z.backward()

    # Build computation graph
    nodes = [
        {"id": "x", "label": "x", "values": x.data.tolist()},
        {"id": "y", "label": "y = x²", "values": y.data.tolist()},
        {"id": "z", "label": "z = sum(y)", "values": [z.item()]},
    ]
    edges = [
        {"from": "x", "to": "y", "label": "x²"},
        {"from": "y", "to": "z", "label": "sum"},
    ]

    return {
        "steps": [
            {
                "title": "Input",
                "description": f"x = {x.data.tolist()} with requires_grad=True",
                "output": {**tensor_to_dict(x.data), "requires_grad": True},
            },
            {
                "title": "Forward Pass",
                "description": "y = x² → z = sum(y)",
                "output": {
                    "nodes": nodes,
                    "edges": edges,
                    "y": y.data.tolist(),
                    "z": z.item(),
                },
            },
            {
                "title": "Backward",
                "description": f"dz/dx = 2x = {x.grad.tolist()}",
                "output": {
                    "gradients": x.grad.tolist(),
                    "nodes": nodes,
                    "edges": edges,
                },
            },
        ],
        "result": {"data": x.grad.tolist(), "shape": list(x.grad.shape), "dtype": "float32"},
    }


# --- NN Layers ---

def _linear(params: dict) -> dict:
    in_features = params.get("in_features", 3)
    out_features = params.get("out_features", 2)
    input_data = params.get("input", [1.0, 2.0, 3.0])

    torch.manual_seed(42)
    layer = nn.Linear(in_features, out_features)
    x = torch.tensor(input_data, dtype=torch.float32)
    if x.dim() == 1:
        x = x.unsqueeze(0)

    weight = layer.weight.data
    bias = layer.bias.data
    result = layer(x)

    # Compute dot products for visualization
    intermediates = []
    for j in range(out_features):
        w_row = weight[j].tolist()
        x_flat = x.squeeze().tolist()
        products = [round(w_row[k] * x_flat[k], 4) for k in range(in_features)]
        dot = round(sum(products), 4)
        b = round(bias[j].item(), 4)
        final = round(dot + b, 4)
        intermediates.append({
            "output_idx": j,
            "weights": w_row,
            "products": products,
            "dot": dot,
            "bias": b,
            "final": final,
        })

    return {
        "steps": [
            {"title": "Input", "description": f"Input vector with {in_features} features", "output": tensor_to_dict(x.squeeze())},
            {"title": "Weight Matrix", "description": f"Weight shape: [{out_features}, {in_features}]", "output": tensor_to_dict(weight)},
            {"title": "Bias", "description": f"Bias vector with {out_features} elements", "output": tensor_to_dict(bias)},
            {
                "title": "Dot Products",
                "description": f"output[j] = dot(weight[j], input) + bias[j]",
                "output": {"intermediates": intermediates},
            },
            {"title": "Result", "description": f"Output with {out_features} features", "output": tensor_to_dict(result.squeeze())},
        ],
        "result": tensor_to_dict(result.squeeze()),
    }


def _conv2d(params: dict) -> dict:
    input_data = params.get("input", [[1, 2, 3, 0], [4, 5, 6, 1], [7, 8, 9, 2], [0, 1, 2, 3]])
    kernel_data = params.get("kernel", [[1, 0], [0, -1]])

    inp = torch.tensor(input_data, dtype=torch.float32)
    kernel = torch.tensor(kernel_data, dtype=torch.float32)

    # Reshape for F.conv2d: (batch, channels, H, W)
    inp_4d = inp.unsqueeze(0).unsqueeze(0)
    ker_4d = kernel.unsqueeze(0).unsqueeze(0)
    result = F.conv2d(inp_4d, ker_4d).squeeze()

    # Build sliding window visualization
    kh, kw = kernel.shape
    oh, ow = result.shape
    windows = []
    for i in range(oh):
        for j in range(ow):
            patch = inp[i:i+kh, j:j+kw].tolist()
            products = (inp[i:i+kh, j:j+kw] * kernel).tolist()
            s = result[i][j].item()
            windows.append({
                "row": i, "col": j,
                "patch": patch,
                "products": products,
                "sum": round(s, 4),
            })

    return {
        "steps": [
            {"title": "Input", "description": f"Input matrix {list(inp.shape)}", "output": tensor_to_dict(inp)},
            {"title": "Kernel", "description": f"Convolution kernel {list(kernel.shape)}", "output": tensor_to_dict(kernel)},
            {
                "title": "Sliding Window",
                "description": f"Slide kernel across input, element-wise multiply & sum",
                "output": {"windows": windows, "output_shape": list(result.shape)},
            },
            {"title": "Result", "description": f"Output feature map {list(result.shape)}", "output": tensor_to_dict(result)},
        ],
        "result": tensor_to_dict(result),
    }


# --- Softmax ---

def _softmax(params: dict) -> dict:
    data = params.get("data", [2.0, 1.0, 0.1])
    t = torch.tensor(data, dtype=torch.float32)
    exp_vals = torch.exp(t)
    exp_sum = exp_vals.sum().item()
    probs = F.softmax(t, dim=0)

    labels = [f"class_{i}" for i in range(len(data))]
    curve_x = torch.linspace(-3, 3, 100).tolist()
    curve_y = [float(torch.exp(torch.tensor(v))) for v in curve_x]

    return {
        "steps": [
            {"title": "Input Logits", "description": f"Raw logits: {t.tolist()}", "output": tensor_to_dict(t)},
            {
                "title": "Exponentiate",
                "description": f"e^x for each logit",
                "output": {
                    "exp_values": exp_vals.tolist(),
                    "exp_sum": round(exp_sum, 4),
                },
            },
            {
                "title": "Normalize",
                "description": "Divide each e^x by the sum to get probabilities",
                "output": {
                    "probabilities": probs.tolist(),
                    "labels": labels,
                    "logits": t.tolist(),
                    "exp_values": exp_vals.tolist(),
                },
            },
        ],
        "result": {"data": probs.tolist(), "shape": list(probs.shape), "dtype": "float32"},
    }


# --- Loss Functions ---

def _cross_entropy(params: dict) -> dict:
    logits_data = params.get("logits", [2.0, 1.0, 0.1])
    target = params.get("target", 0)

    logits = torch.tensor(logits_data, dtype=torch.float32)
    target_t = torch.tensor(target, dtype=torch.long)
    probs = F.softmax(logits, dim=0)
    loss = F.cross_entropy(logits.unsqueeze(0), target_t.unsqueeze(0))

    class_details = []
    for i in range(len(logits_data)):
        class_details.append({
            "index": i,
            "logit": logits_data[i],
            "probability": round(probs[i].item(), 4),
            "is_target": i == target,
        })

    return {
        "steps": [
            {"title": "Logits", "description": f"Raw model output: {logits_data}", "output": tensor_to_dict(logits)},
            {"title": "Target", "description": f"True class index: {target}", "output": {"target": target, "num_classes": len(logits_data)}},
            {
                "title": "Softmax Probabilities",
                "description": f"softmax(logits) → probabilities",
                "output": {"probabilities": probs.tolist(), "class_details": class_details},
            },
            {
                "title": "Cross-Entropy Loss",
                "description": f"loss = -log(p[target]) = -log({probs[target].item():.4f}) = {loss.item():.4f}",
                "output": {"loss": round(loss.item(), 4), "target_prob": round(probs[target].item(), 4)},
            },
        ],
        "result": {"data": [round(loss.item(), 4)], "shape": [1], "dtype": "float32"},
    }


def _mse_loss(params: dict) -> dict:
    predictions_data = params.get("predictions", [2.5, 0.5, 2.1, 7.8])
    targets_data = params.get("targets", [3.0, -0.5, 2.0, 7.5])

    predictions = torch.tensor(predictions_data, dtype=torch.float32)
    targets = torch.tensor(targets_data, dtype=torch.float32)
    loss = F.mse_loss(predictions, targets)

    per_element = []
    for i in range(len(predictions_data)):
        diff = predictions_data[i] - targets_data[i]
        sq_diff = diff ** 2
        per_element.append({
            "index": i,
            "prediction": predictions_data[i],
            "target": targets_data[i],
            "diff": round(diff, 4),
            "sq_diff": round(sq_diff, 4),
        })

    return {
        "steps": [
            {"title": "Predictions", "description": f"Model predictions: {predictions_data}", "output": tensor_to_dict(predictions)},
            {"title": "Targets", "description": f"Ground truth: {targets_data}", "output": tensor_to_dict(targets)},
            {
                "title": "Squared Differences",
                "description": "(prediction - target)² for each element",
                "output": {"per_element": per_element},
            },
            {
                "title": "MSE Loss",
                "description": f"Mean of squared differences = {loss.item():.4f}",
                "output": {"loss": round(loss.item(), 4), "per_element": per_element},
            },
        ],
        "result": {"data": [round(loss.item(), 4)], "shape": [1], "dtype": "float32"},
    }


# --- Optimization ---

def _optimizer(params: dict) -> dict:
    lr = params.get("lr", 0.1)
    num_steps = params.get("num_steps", 5)
    opt_type = params.get("optimizer", "SGD")

    # Minimize f(w) = (w - 3)^2
    w = torch.tensor([0.0], requires_grad=True)
    if opt_type == "Adam":
        opt = torch.optim.Adam([w], lr=lr)
    else:
        opt = torch.optim.SGD([w], lr=lr)

    trajectory = []
    for step in range(num_steps):
        opt.zero_grad()
        loss = (w - 3) ** 2
        loss.backward()
        trajectory.append({
            "step": step,
            "w": round(w.item(), 4),
            "loss": round(loss.item(), 4),
            "grad": round(w.grad.item(), 4),
        })
        opt.step()

    # Final point
    final_loss = ((w - 3) ** 2).item()
    trajectory.append({
        "step": num_steps,
        "w": round(w.item(), 4),
        "loss": round(final_loss, 4),
        "grad": 0.0,
    })

    # Loss landscape curve
    w_vals = torch.linspace(-1, 7, 100)
    loss_vals = ((w_vals - 3) ** 2).tolist()

    return {
        "steps": [
            {
                "title": "Setup",
                "description": f"Minimize f(w) = (w-3)², optimizer={opt_type}, lr={lr}",
                "output": {"optimizer": opt_type, "lr": lr, "target": 3.0},
            },
            {
                "title": "Loss Landscape",
                "description": "f(w) = (w - 3)² — a simple quadratic",
                "output": {"curve": {"w": w_vals.tolist(), "loss": loss_vals}},
            },
            {
                "title": "Optimization Steps",
                "description": f"Running {num_steps} steps of {opt_type}",
                "output": {"trajectory": trajectory, "curve": {"w": w_vals.tolist(), "loss": loss_vals}},
            },
        ],
        "result": {"data": [round(w.item(), 4)], "shape": [1], "dtype": "float32"},
    }


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
    # New operations
    "shape_size": _shape_size,
    "view": _view,
    "permute": _permute,
    "unsqueeze_squeeze": _unsqueeze_squeeze,
    "mul": _mul,
    "sum": _sum,
    "mean": _mean,
    "autograd": _autograd,
    "linear": _linear,
    "conv2d": _conv2d,
    "softmax": _softmax,
    "cross_entropy": _cross_entropy,
    "mse_loss": _mse_loss,
    "optimizer": _optimizer,
}
