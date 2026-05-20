"""OpenAI Responses API proxy routes (/v1/responses).

Converts Responses API format ↔ OpenAI chat completions.
Supports streaming, tools, and memory injection via ChatPipeline.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.pipeline.chat import ChatPipeline

router = APIRouter()


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"input_text", "output_text", "text"}:
                parts.append(item.get("text", ""))
        return "\n".join(p for p in parts if p)
    return ""


def _role(item: dict) -> str:
    r = item.get("role", "user")
    return r if r in {"system", "user", "assistant", "developer"} else "user"


# --- Responses → OpenAI ---

def _input_to_openai(input_data: Any) -> list[dict[str, Any]]:
    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]
    if not isinstance(input_data, list):
        return []
    msgs = []
    for item in input_data:
        if not isinstance(item, dict):
            continue
        text = _extract_text(item.get("content"))
        if text:
            msgs.append({"role": _role(item), "content": text})
    return msgs


def _tools_to_openai(tools: list[dict]) -> list[dict]:
    result = []
    for t in tools:
        if t.get("type") == "function":
            result.append({"type": "function", "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("parameters") or {"type": "object", "properties": {}},
            }})
    return result


def request_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": body.get("model"),
        "messages": _input_to_openai(body.get("input")),
        "stream": bool(body.get("stream")),
    }
    if body.get("temperature") is not None:
        payload["temperature"] = body["temperature"]
    if body.get("max_output_tokens") is not None:
        payload["max_tokens"] = body["max_output_tokens"]
    if body.get("top_p") is not None:
        payload["top_p"] = body["top_p"]
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        payload["tools"] = _tools_to_openai(tools)
        if body.get("tool_choice") is not None:
            payload["tool_choice"] = body["tool_choice"]
    return payload


# --- OpenAI → Responses ---

def response_from_openai(data: dict[str, Any], request_body: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices", [])
    message = choices[0].get("message", {}) if choices else {}
    content = message.get("content") or ""
    usage = data.get("usage") or {}
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"

    output_content = []
    if content:
        output_content.append({"type": "output_text", "text": content, "annotations": []})
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            args = {}
        output_content.append({
            "type": "function_call",
            "id": tc.get("id", ""),
            "call_id": tc.get("id", ""),
            "name": fn.get("name", ""),
            "arguments": json.dumps(args, ensure_ascii=False),
        })

    return {
        "id": resp_id,
        "object": "response",
        "created_at": data.get("created"),
        "model": data.get("model") or request_body.get("model"),
        "output": [{
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": output_content,
        }],
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


# --- Streaming ---

async def _stream_to_responses(openai_response: StreamingResponse, body: dict[str, Any]):
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    model = body.get("model", "")

    created = {
        "type": "response.created",
        "response": {"id": resp_id, "object": "response", "model": model, "output": [], "usage": None},
    }
    yield f"data: {json.dumps(created, ensure_ascii=False)}\n\n"

    async for chunk in openai_response.body_iterator:
        text = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                payload = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if payload.get("error"):
                yield f"data: {json.dumps({'type': 'error', 'error': payload['error']}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            choice = (payload.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                delta_ev = {
                    "type": "response.output_text.delta",
                    "item_id": msg_id, "output_index": 0, "content_index": 0, "delta": content,
                }
                yield f"data: {json.dumps(delta_ev, ensure_ascii=False)}\n\n"

    completed = {
        "type": "response.completed",
        "response": {
            "id": resp_id, "object": "response", "model": model,
            "output": [{"id": msg_id, "type": "message", "role": "assistant", "status": "completed",
                         "content": [{"type": "output_text", "text": "", "annotations": []}]}],
            "usage": {},
        },
    }
    yield f"data: {json.dumps(completed, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


# --- Routes ---

@router.post("/v1/responses")
@router.post("/responses")
async def responses_handler(request: Request):
    raw_body = await request.json()
    openai_body = request_to_openai(raw_body)
    pipeline_response = await ChatPipeline().handle(request, raw_body=openai_body)

    if isinstance(pipeline_response, StreamingResponse):
        return StreamingResponse(
            _stream_to_responses(pipeline_response, raw_body),
            media_type="text/event-stream",
        )

    if isinstance(pipeline_response, JSONResponse):
        payload = json.loads(pipeline_response.body.decode("utf-8"))
        if payload.get("error"):
            return JSONResponse(status_code=pipeline_response.status_code, content=payload)
        return JSONResponse(status_code=pipeline_response.status_code,
                            content=response_from_openai(payload, raw_body))

    return JSONResponse(status_code=500, content={"error": {"message": "Unexpected proxy response", "type": "proxy_error"}})
