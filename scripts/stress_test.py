#!/usr/bin/env python3
"""Stress test for KokoroMemo with rapid topic switching and concurrency.

Tests:
1. Single-user rapid topic switching (10 turns, 5 topic changes)
2. Concurrent multi-user requests
3. Empty / very long messages
4. Memory injection + extraction under load
"""

from __future__ import annotations

import asyncio, json, sys, time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_URL = "http://127.0.0.1:14514"
MODEL = "Chatter"


@dataclass
class TurnResult:
    turn: int
    user: str
    topic: str
    status: int
    elapsed_ms: float
    content_len: int
    error: str | None = None


@dataclass
class TestReport:
    name: str
    results: list[TurnResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 1.0
        return sum(1 for r in self.results if r.status == 200) / len(self.results)

    @property
    def avg_latency(self) -> float:
        ok = [r.elapsed_ms for r in self.results if r.status == 200]
        return sum(ok) / len(ok) if ok else 0

    @property
    def p95_latency(self) -> float:
        ok = sorted([r.elapsed_ms for r in self.results if r.status == 200])
        if not ok:
            return 0
        return ok[int(len(ok) * 0.95)]


TOPICS = [
    ("饮食", "推荐一家川菜馆"),
    ("饮食", "我不吃香菜和芹菜"),
    ("工作", "我在字节跳动做后端开发"),
    ("工作", "我用Go和Rust写代码"),
    ("生活", "我养了一只叫豆包的橘猫"),
    ("生活", "我周末喜欢去攀岩"),
    ("娱乐", "最近有什么好玩的游戏推荐吗"),
    ("娱乐", "进击的巨人最终季好看吗"),
    ("技术", "Kubernetes集群迁移有什么注意事项"),
    ("技术", "Rust的async和tokio怎么用"),
]


async def single_user_rapid_topics(client: httpx.AsyncClient, user_id: str) -> TestReport:
    """10 turns with topic changing every 2 turns."""
    report = TestReport(name=f"rapid_topics_{user_id}")
    messages: list[dict] = []

    for i, (topic, msg) in enumerate(TOPICS):
        messages.append({"role": "user", "content": msg})
        t0 = time.perf_counter()
        try:
            resp = await client.post(f"{BASE_URL}/v1/chat/completions", json={
                "model": MODEL,
                "messages": list(messages),
                "stream": False,
            }, headers={"Authorization": f"Bearer {user_id}"}, timeout=60.0)
            elapsed = (time.perf_counter() - t0) * 1000
            if resp.status_code == 200:
                body = resp.json()
                assistant = body["choices"][0]["message"]["content"]
                messages.append({"role": "assistant", "content": assistant[:200]})
                content_len = len(assistant)
            else:
                content_len = 0
                report.errors.append(f"turn={i} status={resp.status_code} body={resp.text[:200]}")
            report.results.append(TurnResult(i, user_id, topic, resp.status_code, elapsed, content_len))
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            report.results.append(TurnResult(i, user_id, topic, 0, elapsed, 0, str(e)))
            report.errors.append(f"turn={i} exception={e}")

        await asyncio.sleep(0.5)  # let post-processing complete

    return report


async def concurrent_users(client: httpx.AsyncClient, n_users: int = 3, n_turns: int = 3) -> TestReport:
    """Multiple users sending requests concurrently."""
    report = TestReport(name="concurrent")

    async def user_session(uid: int):
        messages = []
        for t in range(n_turns):
            topic, msg = TOPICS[(uid + t) % len(TOPICS)]
            messages.append({"role": "user", "content": msg})
            t0 = time.perf_counter()
            try:
                resp = await client.post(f"{BASE_URL}/v1/chat/completions", json={
                    "model": MODEL,
                    "messages": list(messages),
                    "stream": False,
                }, headers={"Authorization": f"Bearer stress_u{uid}"}, timeout=60.0)
                elapsed = (time.perf_counter() - t0) * 1000
                if resp.status_code == 200:
                    body = resp.json()
                    assistant = body["choices"][0]["message"]["content"]
                    messages.append({"role": "assistant", "content": assistant[:200]})
                    clen = len(assistant)
                else:
                    clen = 0
                    report.errors.append(f"user={uid} turn={t} status={resp.status_code}")
                report.results.append(TurnResult(t, f"u{uid}", topic, resp.status_code, elapsed, clen))
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                report.results.append(TurnResult(t, f"u{uid}", topic, 0, elapsed, 0, str(e)))
                report.errors.append(f"user={uid} turn={t} exception={e}")
            await asyncio.sleep(0.3)

    tasks = [user_session(i) for i in range(n_users)]
    await asyncio.gather(*tasks)
    return report


async def edge_cases(client: httpx.AsyncClient) -> TestReport:
    """Edge cases: empty message, very long message, special characters."""
    report = TestReport(name="edge_cases")
    cases = [
        ("empty", ""),
        ("short", "hi"),
        ("long", "请记住：" + "这是一个测试。" * 50),
        ("special", "emoji 🎮🎯🎲 日文 こんにちは 特殊符号 !@#$%^&*()"),
        ("newlines", "第一行\n第二行\n\n第三行\n"),
    ]
    for name, msg in cases:
        t0 = time.perf_counter()
        try:
            resp = await client.post(f"{BASE_URL}/v1/chat/completions", json={
                "model": MODEL,
                "messages": [{"role": "user", "content": msg}],
                "stream": False,
            }, headers={"Authorization": f"Bearer stress_edge"}, timeout=60.0)
            elapsed = (time.perf_counter() - t0) * 1000
            report.results.append(TurnResult(0, "edge", name, resp.status_code, elapsed,
                                              len(resp.text) if resp.status_code == 200 else 0))
            if resp.status_code != 200:
                report.errors.append(f"edge_case={name} status={resp.status_code} body={resp.text[:200]}")
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            report.results.append(TurnResult(0, "edge", name, 0, elapsed, 0, str(e)))
            report.errors.append(f"edge_case={name} exception={e}")
        await asyncio.sleep(1)
    return report


def print_report(report: TestReport):
    print(f"\n{'='*60}")
    print(f"  {report.name}")
    print(f"{'='*60}")
    print(f"  Requests: {len(report.results)}")
    print(f"  Success:  {report.success_rate:.0%}")
    print(f"  Avg latency: {report.avg_latency:.0f} ms")
    print(f"  P95 latency: {report.p95_latency:.0f} ms")
    if report.errors:
        print(f"  ERRORS ({len(report.errors)}):")
        for e in report.errors[:5]:
            print(f"    - {e[:120]}")
    else:
        print(f"  ERRORS: 0")
    # Per-turn detail
    for r in report.results:
        status = "✓" if r.status == 200 else f"✗ {r.status}"
        print(f"  [{r.user}] turn={r.turn:2d} [{r.topic:4s}] {r.elapsed_ms:6.0f}ms {status}")


async def main():
    print("KokoroMemo Stress Test")
    print(f"Target: {BASE_URL}")

    # Health check
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(f"{BASE_URL}/v1/models")
            print(f"Health: {r.status_code}")
        except Exception as e:
            print(f"FATAL: Cannot reach server: {e}")
            return 1

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Test 1: Single user rapid topic switching
        r1 = await single_user_rapid_topics(client, "stress_main")
        print_report(r1)

        # Test 2: Concurrent users
        r2 = await concurrent_users(client, n_users=3, n_turns=3)
        print_report(r2)

        # Test 3: Edge cases
        r3 = await edge_cases(client)
        print_report(r3)

    # Summary
    all_reports = [r1, r2, r3]
    total = sum(len(r.results) for r in all_reports)
    ok = sum(sum(1 for x in r.results if x.status == 200) for r in all_reports)
    errors = sum(len(r.errors) for r in all_reports)
    print(f"\n{'='*60}")
    print(f"  OVERALL: {ok}/{total} passed, {errors} errors")
    print(f"{'='*60}")

    # Check server health
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(f"{BASE_URL}/v1/models")
            print(f"Server still alive: {r.status_code == 200}")
        except Exception:
            print("Server DEAD after test!")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
