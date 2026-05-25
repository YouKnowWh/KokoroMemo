#!/usr/bin/env python3
"""Compare native DeepSeek Anthropic vs KokoroMemo Anthropic.
Measures latency overhead and response quality."""
import httpx, json, asyncio, time

NATIVE_URL = "https://api.deepseek.com/anthropic/v1/messages"
NATIVE_KEY = "sk-0a1e8108b9d14835916dc654b63b6d6f"
NATIVE_MODEL = "deepseek-chat"

KOKORO_URL = "http://127.0.0.1:14514/anthropic/v1/messages"
KOKORO_KEY = "test"
KOKORO_MODEL = "Coder"

TOOLS = [
    {"name": "Read", "description": "Read a file", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}},
    {"name": "Bash", "description": "Run a bash command", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "Write", "description": "Write a file", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}},
]


async def call(url, headers, model, messages, max_tokens=300, tools=None, system=None, timeout=120):
    """Single Anthropic API call."""
    body = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if tools: body["tools"] = tools
    if system: body["system"] = system

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(url, headers=headers, json=body)
        elapsed = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            return {"ok": False, "elapsed_ms": elapsed, "error": r.text[:300]}
        d = r.json()
        blocks = d.get("content", [])
        text = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        tools_found = [b for b in blocks if b.get("type") == "tool_use"]
        return {"ok": True, "elapsed_ms": elapsed, "stop_reason": d.get("stop_reason"),
                "text": text, "text_len": len(text), "tools": tools_found,
                "input_tokens": d.get("usage", {}).get("input_tokens", 0),
                "output_tokens": d.get("usage", {}).get("output_tokens", 0)}
    except Exception as e:
        return {"ok": False, "elapsed_ms": (time.perf_counter() - t0) * 1000, "error": str(e)}


def cmp(name, native, kokoro):
    """Compare and print results."""
    print(f"--- {name} ---")
    n_ok = native.get("ok")
    k_ok = kokoro.get("ok")

    if n_ok and k_ok:
        oh = kokoro["elapsed_ms"] - native["elapsed_ms"]
        n_info = f"{native['elapsed_ms']:.0f}ms stop={native['stop_reason']} tok={native.get('input_tokens',0)}+{native.get('output_tokens',0)}"
        k_info = f"{kokoro['elapsed_ms']:.0f}ms stop={kokoro['stop_reason']} tok={kokoro.get('input_tokens',0)}+{kokoro.get('output_tokens',0)}"
        quality = "✓" if native["stop_reason"] == kokoro["stop_reason"] and abs(native["text_len"] - kokoro["text_len"]) < 50 else "⚠ diff"
        print(f"  Native:     {n_info}")
        print(f"  KokoroMemo: {k_info}")
        print(f"  Overhead: {oh:+.0f}ms  quality={quality}")
        if native["text"]:
            print(f"  Native text:     {native['text'][:120]}")
        if kokoro["text"]:
            print(f"  KokoroMemo text: {kokoro['text'][:120]}")
    elif n_ok:
        print(f"  Native: {native['elapsed_ms']:.0f}ms OK")
        print(f"  KokoroMemo: ERROR {kokoro.get('error','')[:150]}")
    else:
        print(f"  Native: ERROR {native.get('error','')[:150]}")
        print(f"  KokoroMemo: {kokoro.get('error','')[:150]}")
    print()


async def test_basic():
    msgs = [{"role": "user", "content": "1+1=? One word answer."}]
    native = await call(NATIVE_URL, {"Content-Type": "application/json", "x-api-key": NATIVE_KEY, "anthropic-version": "2023-06-01"}, NATIVE_MODEL, msgs, max_tokens=50)
    kokoro = await call(KOKORO_URL, {"Content-Type": "application/json", "x-api-key": KOKORO_KEY, "anthropic-version": "2023-06-01"}, KOKORO_MODEL, msgs, max_tokens=50)
    cmp("Basic chat", native, kokoro)


async def test_tool_use():
    msgs = [{"role": "user", "content": "Read the file /etc/hostname"}]
    native = await call(NATIVE_URL, {"Content-Type": "application/json", "x-api-key": NATIVE_KEY, "anthropic-version": "2023-06-01"}, NATIVE_MODEL, msgs, tools=TOOLS, max_tokens=300)
    kokoro = await call(KOKORO_URL, {"Content-Type": "application/json", "x-api-key": KOKORO_KEY, "anthropic-version": "2023-06-01"}, KOKORO_MODEL, msgs, tools=TOOLS, max_tokens=300)
    cmp("Single tool_use", native, kokoro)


async def test_roundtrip():
    """Multi-turn tool calling."""
    H_N = {"Content-Type": "application/json", "x-api-key": NATIVE_KEY, "anthropic-version": "2023-06-01"}
    H_K = {"Content-Type": "application/json", "x-api-key": KOKORO_KEY, "anthropic-version": "2023-06-01"}

    async def do_roundtrip(url, headers, model, label):
        # Turn 1
        t1 = await call(url, headers, model, [{"role": "user", "content": "Write /tmp/x.txt with 'hello', then read it"}], tools=TOOLS, max_tokens=300)
        if not t1["ok"] or not t1["tools"]:
            return [t1]

        msgs = [{"role": "user", "content": "Write /tmp/x.txt with 'hello', then read it"}]
        msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": tb["id"], "name": tb["name"], "input": tb["input"]} for tb in t1["tools"]]})
        tr = []
        for tb in t1["tools"]:
            if tb["name"] == "Write": tr.append({"type": "tool_result", "tool_use_id": tb["id"], "content": "Written: /tmp/x.txt"})
            elif tb["name"] == "Read": tr.append({"type": "tool_result", "tool_use_id": tb["id"], "content": "hello"})
            else: tr.append({"type": "tool_result", "tool_use_id": tb["id"], "content": "OK"})
        msgs.append({"role": "user", "content": tr})

        # Turn 2
        t2 = await call(url, headers, model, msgs, tools=TOOLS, max_tokens=300)
        if not t2["ok"]: return [t1, t2]

        if t2["tools"]:
            msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": tb["id"], "name": tb["name"], "input": tb["input"]} for tb in t2["tools"]]})
            tr2 = [{"type": "tool_result", "tool_use_id": tb["id"], "content": "hello"} for tb in t2["tools"]]
            msgs.append({"role": "user", "content": tr2})
            t3 = await call(url, headers, model, msgs, tools=TOOLS, max_tokens=300)
            return [t1, t2, t3]
        return [t1, t2]

    native_turns = await do_roundtrip(NATIVE_URL, H_N, NATIVE_MODEL, "N")
    kokoro_turns = await do_roundtrip(KOKORO_URL, H_K, KOKORO_MODEL, "K")

    for i, (nt, kt) in enumerate(zip(native_turns, kokoro_turns)):
        cmp(f"Round-trip T{i+1}", nt, kt)


async def test_system_prompt():
    system = "# Rules\n1. Reply in Chinese\n2. Use ``` for code blocks\n" + "padding_" * 100
    msgs = [{"role": "user", "content": "What is 2+2?"}]
    native = await call(NATIVE_URL, {"Content-Type": "application/json", "x-api-key": NATIVE_KEY, "anthropic-version": "2023-06-01"}, NATIVE_MODEL, msgs, system=system, max_tokens=100)
    kokoro = await call(KOKORO_URL, {"Content-Type": "application/json", "x-api-key": KOKORO_KEY, "anthropic-version": "2023-06-01"}, KOKORO_MODEL, msgs, system=system, max_tokens=100)
    cmp("System prompt", native, kokoro)


async def main():
    print("=" * 60)
    print("Native DeepSeek vs KokoroMemo Anthropic Comparison")
    print("=" * 60)
    await test_basic(); await asyncio.sleep(1)
    await test_tool_use(); await asyncio.sleep(1)
    await test_roundtrip(); await asyncio.sleep(1)
    await test_system_prompt()
    print("Done.")

asyncio.run(main())
