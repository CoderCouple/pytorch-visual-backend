"""Unified operations router."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any
from services.executor import execute_operation
from services.code_runner import run_user_code

router = APIRouter(prefix="/api", tags=["operations"])


class OperationRequest(BaseModel):
    operation: str
    params: dict[str, Any] = {}


class CodeRequest(BaseModel):
    code: str


@router.post("/operations/execute")
async def execute(req: OperationRequest):
    try:
        result = execute_operation(req.operation, req.params)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run-code")
async def run_code(req: CodeRequest):
    """Execute user-written Python/PyTorch code and return all tensors."""
    try:
        result = run_user_code(req.code)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/operations/list")
async def list_operations():
    from services.executor import OPERATION_HANDLERS
    return {"operations": list(OPERATION_HANDLERS.keys())}
