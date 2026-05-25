#!/usr/bin/env python3
"""Test KokoroMemo Anthropic endpoint: tool use, streaming, system prompts."""
import json, httpx, asyncio, sys

BASE = "http://127.0.0.1:14514"
HEADERS = {"Content-Type": "application/json", "x-api-key": "test", "anthropic-version": "2023-06-01"}

TOOLS = [
    {"name": "read_file", "description": "Read a file", "input_schema": {
        "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]
    }},
    {"name": "bash", "description": "Execute a bash command", "input_schema": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]
    }},
    {"name": "write_file", "description": "Write content to a file", "input_schema": {
        "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]
    }},
]

async def test_basic():
    print("=== 1. Basic request ===")
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{BASE}/anthropic/v1/messages", headers=HEADERS, json={
            "model": "Coder", "max_tokens": 100,
            "messages": [{"role": "user", "content": "What is 3+4?"}],
        })
        d = r.json()
        ok = d.get("type") == "message" and d["content"][0]["type"] == "text"
        print(f"  {'PASS' if ok else 'FAIL'}: {d.get('content', [{}])[0].get('text', d)[:100]}")

async def test_tool_use():
    print("=== 2. Tool use (single) ===")
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{BASE}/anthropic/v1/messages", headers=HEADERS, json={
            "model": "Coder", "max_tokens": 500,
            "system": "You are a coding assistant. Use tools when needed.",
            "tools": TOOLS,
            "messages": [{"role": "user", "content": "Read /etc/hostname and tell me what's in it"}],
        })
        d = r.json()
        has_tool = any(b.get("type") == "tool_use" for b in d.get("content", []))
        tool_name = next((b["name"] for b in d.get("content", []) if b.get("type") == "tool_use"), "none")
        print(f"  {'PASS' if has_tool else 'WARN'}: stop_reason={d.get('stop_reason')} tool={tool_name}")

async def test_tool_roundtrip():
    print("=== 3. Tool use round-trip ===")
    async with httpx.AsyncClient(timeout=60) as c:
        # Step 1: get tool call
        r1 = await c.post(f"{BASE}/anthropic/v1/messages", headers=HEADERS, json={
            "model": "Coder", "max_tokens": 500,
            "tools": TOOLS,
            "messages": [{"role": "user", "content": "Read /etc/hostname"}],
        })
        d1 = r1.json()
        tool_blocks = [b for b in d1.get("content", []) if b.get("type") == "tool_use"]
        if not tool_blocks:
            print(f"  FAIL: no tool_use in response. stop_reason={d1.get('stop_reason')} content_types={[b['type'] for b in d1.get('content',[])]}")
            return
        tool = tool_blocks[0]
        print(f"  Step1: tool={tool['name']} input={tool['input']}")

        # Step 2: submit tool result
        r2 = await c.post(f"{BASE}/anthropic/v1/messages", headers=HEADERS, json={
            "model": "Coder", "max_tokens": 200,
            "tools": TOOLS,
            "messages": [
                {"role": "user", "content": "Read /etc/hostname"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": tool["id"], "name": tool["name"], "input": tool["input"]}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool["id"], "content": "kokoromemo-server"}]},
            ],
        })
        d2 = r2.json()
        text = next((b["text"] for b in d2.get("content", []) if b.get("type") == "text"), "")
        print(f"  {'PASS' if text else 'FAIL'}: response={text[:150]}")

async def test_streaming():
    print("=== 4. Streaming ===")
    async with httpx.AsyncClient(timeout=60) as c:
        events = []
        async with c.stream("POST", f"{BASE}/anthropic/v1/messages", headers=HEADERS, json={
            "model": "Coder", "max_tokens": 500,
            "messages": [{"role": "user", "content": "Count from 1 to 5"}],
            "stream": True,
        }) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
        types = [e["type"] for e in events]
        has_start = "message_start" in types
        has_content = "content_block_delta" in types
        has_stop = "message_stop" in types
        print(f"  {'PASS' if all([has_start, has_stop]) else 'FAIL'}: events={types}")

async def test_memory_injection_anthropic():
    print("=== 5. Memory injection via Anthropic ===")
    async with httpx.AsyncClient(timeout=60) as c:
        # Write a memory
        await c.post(f"{BASE}/anthropic/v1/messages", headers=HEADERS, json={
            "model": "Coder", "max_tokens": 200,
            "messages": [{"role": "user", "content": "请记住：服务器的hostname是kokoromemo-server"}],
        })
        await asyncio.sleep(2)  # wait for post-processing

        # Recall
        r = await c.post(f"{BASE}/anthropic/v1/messages", headers=HEADERS, json={
            "model": "Coder", "max_tokens": 200,
            "messages": [{"role": "user", "content": "这台服务器的hostname是什么？"}],
        })
        text = next((b["text"] for b in r.json().get("content", []) if b.get("type") == "text"), "")
        has_memory = "kokoromemo" in text.lower()
        print(f"  {'PASS' if has_memory else 'INFO'}: response={text[:200]}")

async def main():
    print("KokoroMemo Anthropic Endpoint Test\n")
    await test_basic()
    await test_tool_use()
    await test_tool_roundtrip()
    await test_streaming()
    await test_memory_injection_anthropic()
    print("\nDone.")

asyncio.run(main())
