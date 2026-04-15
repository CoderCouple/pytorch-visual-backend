"""FastAPI entry point for PyTorch Visual backend."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.operations import router as operations_router

app = FastAPI(title="PyTorch Visual API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(operations_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
