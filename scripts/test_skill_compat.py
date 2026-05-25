#!/usr/bin/env python3
"""Test Claude Code skill compatibility through KokoroMemo Anthropic endpoint.

Simulates: system prompts, tool definitions, multi-turn tool calling,
content arrays with mixed text/tool_use/tool_result blocks.
"""

import httpx, json, asyncio

BASE = "http://127.0.0.1:14514"
HEADERS = {"Content-Type": "application/json", "x-api-key": "test", "anthropic-version": "2023-06-01"}

# Realistic Claude Code tools
TOOLS = [
    {"name": "Read", "description": "Read a file from the local filesystem", "input_schema": {
        "type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]
    }},
    {"name": "Write", "description": "Write a file to the local filesystem", "input_schema": {
        "type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]
    }},
    {"name": "Edit", "description": "Make exact string replacements in a file", "input_schema": {
        "type": "object", "properties": {"file_path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}, "required": ["file_path", "old_string", "new_string"]
    }},
    {"name": "Bash", "description": "Execute a bash command", "input_schema": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]
    }},
    {"name": "Task", "description": "Launch a new agent for complex tasks", "input_schema": {
        "type": "object", "properties": {"description": {"type": "string"}, "prompt": {"type": "string"}, "subagent_type": {"type": "string"}}, "required": ["description", "prompt"]
    }},
]

SKILL_SYSTEM = (
    "You are Claude Code, a coding assistant. You have access to skills.\n\n"
    "## Skills\n"
    "- /using-superpowers: Core workflow skill\n"
    "- /test-driven-development: TDD with red-green-refactor\n\n"
    "When user invokes a skill with /skill-name, follow the instructions.\n"
    "Always use tools when needed. Respond in Chinese."
)

SKILL_TASK = (
    "Write a Python script /tmp/hello.py that prints 'Hello from skill test'.\n"
    "Then run it with bash to verify.\n"
    "Report the output."
)


async def test_skill_flow():
    """Multi-turn: system prompt + tools + tool_use + tool_result"""
    print("=" * 50)
    print("Test 1: Skill-like multi-turn tool flow")
    print("=" * 50)

    async with httpx.AsyncClient(timeout=120) as c:
        # Turn 1: send task
        r1 = await c.post(f"{BASE}/anthropic/v1/messages", headers=HEADERS, json={
            "model": "Coder", "max_tokens": 1000,
            "system": SKILL_SYSTEM,
            "tools": TOOLS,
            "messages": [{"role": "user", "content": SKILL_TASK}],
        })

        if r1.status_code != 200:
            print(f"  FAIL: status={r1.status_code} body={r1.text[:300]}")
            return False

        d1 = r1.json()
        tool_blocks = [b for b in d1.get("content", []) if b.get("type") == "tool_use"]
        text_blocks = [b for b in d1.get("content", []) if b.get("type") == "text"]

        print(f"  Turn1: stop={d1.get('stop_reason')} tool_calls={len(tool_blocks)} text_blocks={len(text_blocks)}")

        if not tool_blocks:
            print(f"  FAIL: expected tool_use")
            if text_blocks:
                print(f"  Text: {text_blocks[0].get('text','')[:300]}")
            return False

        for tb in tool_blocks:
            print(f"  -> {tb['name']}({json.dumps(tb['input'], ensure_ascii=False)[:100]})")

        # Build tool results
        messages = [{"role": "user", "content": SKILL_TASK}]
        assistant_content = list(d1["content"])
        messages.append({"role": "assistant", "content": assistant_content})

        tool_results = []
        for tb in tool_blocks:
            if tb["name"] == "Write":
                tool_results.append({"type": "tool_result", "tool_use_id": tb["id"],
                                     "content": f"File written: {tb['input'].get('file_path','')}"})
            elif tb["name"] == "Bash":
                tool_results.append({"type": "tool_result", "tool_use_id": tb["id"],
                                     "content": "Hello from skill test"})
            else:
                tool_results.append({"type": "tool_result", "tool_use_id": tb["id"],
                                     "content": f"OK: {tb['name']}"})

        messages.append({"role": "user", "content": tool_results})

        # Turn 2: send results
        r2 = await c.post(f"{BASE}/anthropic/v1/messages", headers=HEADERS, json={
            "model": "Coder", "max_tokens": 500,
            "system": SKILL_SYSTEM,
            "tools": TOOLS,
            "messages": messages,
        })

        if r2.status_code != 200:
            print(f"  FAIL: turn2 status={r2.status_code} body={r2.text[:300]}")
            return False

        d2 = r2.json()
        text2 = next((b["text"] for b in d2.get("content", []) if b.get("type") == "text"), "")
        tools2 = [b for b in d2.get("content", []) if b.get("type") == "tool_use"]

        print(f"  Turn2: stop={d2.get('stop_reason')} tools={len(tools2)} text_len={len(text2)}")
        if tools2:
            for tb in tools2:
                print(f"  -> {tb['name']}({json.dumps(tb['input'], ensure_ascii=False)[:100]})")
        if text2:
            print(f"  Text: {text2[:300]}")

        # If model wants more tools, do turn 3
        if tools2 and not text2:
            messages.append({"role": "assistant", "content": list(d2["content"])})
            tr3 = []
            for tb in tools2:
                if tb["name"] == "Bash":
                    tr3.append({"type": "tool_result", "tool_use_id": tb["id"],
                                "content": "Hello from skill test"})
                else:
                    tr3.append({"type": "tool_result", "tool_use_id": tb["id"],
                                "content": f"OK: {tb['name']}"})
            messages.append({"role": "user", "content": tr3})

            r3 = await c.post(f"{BASE}/anthropic/v1/messages", headers=HEADERS, json={
                "model": "Coder", "max_tokens": 500,
                "system": SKILL_SYSTEM, "tools": TOOLS, "messages": messages,
            })
            d3 = r3.json()
            text3 = next((b["text"] for b in d3.get("content", []) if b.get("type") == "text"), "")
            tools3 = [b for b in d3.get("content", []) if b.get("type") == "tool_use"]
            print(f"  Turn3: stop={d3.get('stop_reason')} tools={len(tools3)} text_len={len(text3)}")
            print(f"  Response: {text3[:300]}")
            text2 = text3  # use turn3 text for validation

        if text2 and ("Hello from skill test" in text2 or "hello" in text2.lower() or "skill" in text2.lower()):
            print(f"  PASS: skill flow completed\n")
            return True
        elif text2:
            print(f"  WARN: got response but no expected content\n")
            return True
        else:
            print(f"  FAIL: stuck in tool loop\n")
            return False


async def test_long_system_prompt():
    """System prompt with markdown, code blocks, special chars"""
    print("=" * 50)
    print("Test 2: Long system prompt with markdown")
    print("=" * 50)

    long_system = (
        "# Skill: Code Review\n\n"
        "## Instructions\n"
        "1. Check for security issues\n"
        "2. Verify error handling\n"
        "3. Suggest improvements\n\n"
        "```python\n"
        "def example():\n"
        "    return 'ok'\n"
        "```\n\n"
        "Special: !@#$%^&*() 中文日本語 🎮\n"
        + "x" * 300
    )

    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{BASE}/anthropic/v1/messages", headers=HEADERS, json={
            "model": "Coder", "max_tokens": 150,
            "system": long_system,
            "messages": [{"role": "user", "content": "What is your role? Reply in 1 sentence."}],
        })
        d = r.json()
        text = next((b["text"] for b in d.get("content", []) if b.get("type") == "text"), "")
        has_content = len(text) > 0
        print(f"  Status: {r.status_code} text_len={len(text)}")
        print(f"  Response: {text[:200]}")
        print(f"  {'PASS' if has_content else 'FAIL'}: system prompt preserved\n")


async def test_multi_block_content():
    """Message with multiple content blocks survives conversion"""
    print("=" * 50)
    print("Test 3: Multi-block content fidelity")
    print("=" * 50)

    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{BASE}/anthropic/v1/messages", headers=HEADERS, json={
            "model": "Coder", "max_tokens": 300,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "First question: what is 2+2?"},
                    {"type": "text", "text": "Second question: what is the capital of France?"},
                ]},
            ],
        })
        d = r.json()
        text_blocks = [b["text"] for b in d.get("content", []) if b.get("type") == "text"]
        combined = " ".join(text_blocks)
        answers_both = ("4" in combined or "four" in combined.lower()) and ("paris" in combined.lower())
        print(f"  Blocks: {len(d.get('content',[]))} text_blocks: {len(text_blocks)}")
        print(f"  Combined: {combined[:300]}")
        print(f"  {'PASS' if answers_both else 'WARN'}: multi-block handled\n")


async def test_tool_schema_edge_cases():
    """Tool schemas with complex nested properties"""
    print("=" * 50)
    print("Test 4: Complex tool schemas")
    print("=" * 50)

    complex_tool = {
        "name": "complex_tool",
        "description": "A tool with nested object and array parameters",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": ["option_a", "option_b"]},
                "config": {
                    "type": "object",
                    "properties": {
                        "timeout": {"type": "integer", "minimum": 1, "maximum": 300},
                        "retries": {"type": "integer", "default": 3},
                    }
                },
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name"],
        }
    }

    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{BASE}/anthropic/v1/messages", headers=HEADERS, json={
            "model": "Coder", "max_tokens": 300,
            "tools": [complex_tool],
            "messages": [{"role": "user", "content": "Call complex_tool with name=option_a and tags=['test','demo']"}],
        })
        d = r.json()
        tool_blocks = [b for b in d.get("content", []) if b.get("type") == "tool_use"]
        if tool_blocks:
            tb = tool_blocks[0]
            inp = tb["input"]
            match_name = inp.get("name") == "option_a"
            has_tags = isinstance(inp.get("tags"), list)
            print(f"  Tool: {tb['name']} name={inp.get('name')} tags={inp.get('tags')}")
            print(f"  {'PASS' if match_name and has_tags else 'WARN'}: complex schema handled")
        else:
            text = next((b["text"] for b in d.get("content", []) if b.get("type") == "text"), "")
            print(f"  No tool call. Text: {text[:200]}")
            print(f"  WARN: model chose not to use tool")


async def main():
    print("KokoroMemo Skill Compatibility Test\n")
    results = []
    results.append(await test_skill_flow())
    await asyncio.sleep(1)
    await test_long_system_prompt()
    await asyncio.sleep(1)
    await test_multi_block_content()
    await asyncio.sleep(1)
    await test_tool_schema_edge_cases()

    passed = sum(1 for r in results if r)
    print(f"\n{'='*50}")
    print(f"Skill compatibility: {passed}/{len(results)} core tests passed")
    print(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(main())
