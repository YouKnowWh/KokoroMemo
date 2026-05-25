"""Agent Task API - Multi-device worker orchestration.

Endpoints:
  POST /agent/register   - Register worker capabilities
  GET  /agent/tasks      - Poll pending tasks for a worker
  POST /agent/results    - Submit task execution result
  POST /agent/heartbeat  - Worker keepalive ping
"""

from __future__ import annotations

import json
import os
import time
import secrets
from pathlib import Path

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

router = APIRouter()

# ── Storage paths ──────────────────────────────────────────────────────────
_DATA_DIR = Path(os.getenv("KOKOROMEMO_DATA_DIR", Path(__file__).resolve().parent.parent.parent / "data"))
_AGENT_DIR = _DATA_DIR / "agent"
_WORKERS_FILE = _AGENT_DIR / "workers.json"
_TASKS_DIR = _AGENT_DIR / "tasks"
_RESULTS_DIR = _AGENT_DIR / "results"

# Agent API token - simple shared secret
_AGENT_TOKEN = os.getenv("AGENT_API_TOKEN", "agent-token-kokoromemo-2026")


def _ensure_dirs():
    _AGENT_DIR.mkdir(parents=True, exist_ok=True)
    _TASKS_DIR.mkdir(parents=True, exist_ok=True)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _verify_token(request: Request) -> str:
    """Verify agent API token from Authorization header. Returns worker_id if valid."""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
    else:
        token = auth
    if token != _AGENT_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid agent token")
    return token


def _load_workers() -> dict:
    _ensure_dirs()
    if _WORKERS_FILE.exists():
        try:
            return json.loads(_WORKERS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_workers(workers: dict):
    _ensure_dirs()
    _WORKERS_FILE.write_text(json.dumps(workers, indent=2, ensure_ascii=False))


# ── Models ─────────────────────────────────────────────────────────────────

class WorkerRegisterRequest(BaseModel):
    worker_id: str
    hostname: str = ""
    capabilities: list[str] = []
    limits: dict = {}


class TaskResultRequest(BaseModel):
    task_id: str
    worker_id: str
    status: str  # done, error, timeout, blocked
    exit_code: int = 0
    summary: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""
    changed_files: list[str] = []
    evidence: list[str] = []


# ── Routes ─────────────────────────────────────────────────────────────────

@router.post("/agent/register")
async def agent_register(payload: WorkerRegisterRequest, request: Request):
    """Register or update a worker's capabilities."""
    _verify_token(request)
    workers = _load_workers()

    workers[payload.worker_id] = {
        "worker_id": payload.worker_id,
        "hostname": payload.hostname,
        "capabilities": payload.capabilities,
        "limits": payload.limits,
        "registered_at": time.time(),
        "last_heartbeat": time.time(),
        "status": "online",
    }

    _save_workers(workers)
    return {"status": "ok", "worker_id": payload.worker_id}


@router.post("/agent/heartbeat")
async def agent_heartbeat(payload: dict, request: Request):
    """Worker keepalive ping."""
    _verify_token(request)
    worker_id = payload.get("worker_id", "")
    if not worker_id:
        raise HTTPException(status_code=400, detail="Missing worker_id")

    workers = _load_workers()
    if worker_id in workers:
        workers[worker_id]["last_heartbeat"] = time.time()
        workers[worker_id]["status"] = "online"
        _save_workers(workers)

    return {"status": "ok", "server_time": time.time()}


@router.get("/agent/tasks")
async def agent_get_tasks(request: Request, worker_id: str = ""):
    """Get pending tasks for a worker."""
    _verify_token(request)
    _ensure_dirs()

    if not worker_id:
        raise HTTPException(status_code=400, detail="Missing worker_id")

    # Update worker heartbeat
    workers = _load_workers()
    if worker_id in workers:
        workers[worker_id]["last_heartbeat"] = time.time()
        workers[worker_id]["status"] = "online"
        _save_workers(workers)

    # Find tasks assigned to this worker
    tasks = []
    for tf in sorted(_TASKS_DIR.glob(f"{worker_id}__*.json")):
        try:
            task = json.loads(tf.read_text())
            tasks.append(task)
        except (json.JSONDecodeError, OSError):
            pass

    # Also check for wildcard tasks (any worker can pick up)
    for tf in sorted(_TASKS_DIR.glob("any__*.json")):
        try:
            task = json.loads(tf.read_text())
            tasks.append(task)
        except (json.JSONDecodeError, OSError):
            pass

    return {"tasks": tasks, "server_time": time.time()}


@router.post("/agent/results")
async def agent_post_result(payload: TaskResultRequest, request: Request):
    """Submit a task execution result."""
    return await _save_result_and_cleanup(payload, request=request)

@router.post("/agent/tasks/complete")
async def agent_tasks_complete(payload: TaskResultRequest, request: Request):
    """Submit a task execution result (alias used by desktop WSL2 daemon)."""
    return await _save_result_and_cleanup(payload, request=request)


async def _save_result_and_cleanup(payload: TaskResultRequest, request: Request):
    """Core logic: save result and remove task file."""
    _verify_token(request)
    _ensure_dirs()
    result = payload.model_dump()
    result["received_at"] = time.time()

    # Write result
    result_file = _RESULTS_DIR / f"{payload.task_id}.json"
    result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # Remove the task file
    task_file = _TASKS_DIR / f"{payload.worker_id}__{payload.task_id}.json"
    if not task_file.exists():
        task_file = _TASKS_DIR / f"any__{payload.task_id}.json"
    if task_file.exists():
        task_file.unlink()

    return {"status": "ok", "task_id": payload.task_id}


@router.get("/agent/workers")
async def agent_list_workers(request: Request):
    """List all registered workers and their status."""
    _verify_token(request)
    workers = _load_workers()

    # Mark stale workers
    now = time.time()
    for w in workers.values():
        if now - w.get("last_heartbeat", 0) > 60:
            w["status"] = "offline"

    return {"workers": list(workers.values()), "server_time": now}


# ── Admin endpoints (for Hermes / VPS internal use) ────────────────────────

@router.post("/agent/dispatch")
async def agent_dispatch(payload: dict, request: Request):
    """Dispatch a task to a specific worker. Internal use by Hermes/VPS."""
    _verify_token(request)
    _ensure_dirs()

    worker_id = payload.get("worker_id", "any")
    task_id = payload.get("task_id", "")
    if not task_id:
        task_id = f"task_{int(time.time())}_{secrets.token_hex(4)}"

    task = {
        "task_id": task_id,
        "worker_id": worker_id,
        "type": payload.get("type", "shell_readonly"),
        "goal": payload.get("goal", ""),
        "command": payload.get("command", ""),
        "workdir": payload.get("workdir", "/tmp"),
        "constraints": payload.get("constraints", {
            "timeout_sec": 300,
            "allow_write": False,
            "allow_restart": False,
        }),
        "dispatched_at": time.time(),
    }

    task_file = _TASKS_DIR / f"{worker_id}__{task_id}.json"
    task_file.write_text(json.dumps(task, indent=2, ensure_ascii=False))

    return {
        "status": "ok",
        "task_id": task_id,
        "worker_id": worker_id,
    }


@router.get("/agent/results/{task_id}")
async def agent_get_result(task_id: str, request: Request):
    """Get a specific task result. Internal use."""
    _verify_token(request)
    _ensure_dirs()

    result_file = _RESULTS_DIR / f"{task_id}.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Result not found")

    return json.loads(result_file.read_text())
