"""Server status dashboard + admin command endpoints."""
from __future__ import annotations
import time, os
import httpx
from fastapi import APIRouter, Header, HTTPException
from app.core.state import get_config

router = APIRouter()
CONSOLE_TOKEN = "kokoromemo-console-2026"



@router.get("/status")
async def status():
    cfg = get_config()
    result = {"server": {}, "memory": {}, "cache": {}, "gpt": {}}

    try:
        import psutil
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage(cfg.storage.root_dir)
        result["server"] = {
            "uptime_seconds": round(time.time() - psutil.Process(os.getpid()).create_time()),
            "memory_used_pct": round(mem.percent, 1),
            "disk_used_pct": round(disk.percent, 1),
        }
    except:
        result["server"] = {"uptime_seconds": 0}

    try:
        import aiosqlite
        async with aiosqlite.connect(cfg.storage.sqlite.memory_db, timeout=10.0) as db:
            cur = await db.execute("SELECT count(*) FROM memory_cards WHERE status = 'approved'")
            row = await cur.fetchone()
            result["memory"]["approved_cards"] = row[0] if row else 0
            cur = await db.execute("SELECT count(*) FROM retrieval_decisions WHERE created_at > datetime('now', '-1 day')")
            row = await cur.fetchone()
            result["memory"]["retrievals_24h"] = row[0] if row else 0
    except Exception as e:
        result["memory"]["error"] = str(e)

    try:
        from app.memory.retrieval_embedding_cache import get_retrieval_cache
        rc = get_retrieval_cache()
        result["cache"]["retrieval"] = {"entries": len(rc), "hits": rc.stats.hits, "misses": rc.stats.misses}
        from app.memory.context_embedding_cache import get_context_cache
        cc = get_context_cache()
        result["cache"]["context"] = {"entries": len(cc)}
    except Exception as e:
        result["cache"]["error"] = str(e)

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get("http://127.0.0.1:10531/v1/models")
            if r.status_code == 200:
                models = [m["id"] for m in r.json().get("data", [])]
                result["gpt"] = {"status": "ok", "models": models}
                # Decode subscription info from OAuth token
                import base64
                try:
                    with open("/home/ubuntu/.codex/auth.json") as af:
                        auth = __import__("json").load(af)
                    id_token = auth.get("tokens", {}).get("id_token", "")
                    if id_token and "." in id_token:
                        payload = id_token.split(".")[1] + "=" * (4 - len(id_token.split(".")[1]) % 4)
                        claims = __import__("json").loads(base64.urlsafe_b64decode(payload))
                        oai = claims.get("https://api.openai.com/auth", {})
                        result["gpt"]["subscription"] = {
                            "plan": oai.get("chatgpt_plan_type", "unknown"),
                            "active_until": oai.get("chatgpt_subscription_active_until"),
                            "last_refresh": auth.get("last_refresh"),
                        }
                except:
                    pass
            else:
                result["gpt"] = {"status": "error", "code": r.status_code}
    except Exception as e:
        result["gpt"] = {"status": "down", "error": str(e)}

    return result


@router.get("/status/usage")
async def status_usage():
    """GPT token usage: 5h / 7d windows"""
    from app.core.state import get_config
    import aiosqlite, time
    cfg = get_config()
    now = time.time()
    windows = {"5h": (18000, 17700000), "7d": (604800, 70800000)}
    result = {}
    try:
        async with aiosqlite.connect(cfg.storage.sqlite.app_db, timeout=10.0) as db:
            # Also count from request_counter for backward compat
            for label, (secs, limit) in windows.items():
                cur = await db.execute(
                    "SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(cached_tokens),0), COALESCE(SUM(output_tokens),0), COALESCE(MIN(created_at),?) FROM token_usage WHERE created_at > ?",
                    (now, now - secs)
                )
                row = await cur.fetchone()
                in_tok, c_tok, out_tok, oldest = (row[0], row[1], row[2], row[3]) if row else (0, 0, 0, now)
                total_tok = in_tok + out_tok
                pct = round(total_tok / limit * 100, 3) if limit else 0
                reset_at = (oldest or now) + secs
                reset_secs = max(0, reset_at - now)
                result[label] = {"total_tokens": total_tok, "input_tokens": in_tok, "cached_tokens": c_tok, "output_tokens": out_tok, "limit": limit, "pct": pct, "reset_secs": round(reset_secs)}
    except Exception as e:
        result["error"] = str(e)
    return result


@router.post("/console/restart")
async def admin_restart(x_console_token: str = Header(None, alias="X-Console-Token")):
    if x_console_token != CONSOLE_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid console token")
    import signal
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "restarting"}

@router.post("/console/clear-cache")
async def admin_clear_cache(x_console_token: str = Header(None, alias="X-Console-Token")):
    if x_console_token != CONSOLE_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid console token")
    from app.memory.retrieval_embedding_cache import get_retrieval_cache
    from app.memory.context_embedding_cache import get_context_cache
    get_retrieval_cache().clear()
    get_context_cache().clear()
    return {"status": "ok", "message": "Caches cleared"}

@router.post("/console/rebuild-index")
async def admin_rebuild_index(x_console_token: str = Header(None, alias="X-Console-Token")):
    if x_console_token != CONSOLE_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid console token")
    from app.memory.retrieval_embedding_cache import get_retrieval_cache
    from app.memory.context_embedding_cache import get_context_cache
    get_retrieval_cache().clear()
    get_context_cache().clear()
    return {"status": "ok", "message": "Caches cleared for rebuild"}

@router.get('/status/balances')
async def status_balances():
    import httpx
    result = {}
    # DeepSeek
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get('https://api.deepseek.com/user/balance',
                headers={'Authorization': 'Bearer sk-0a1e8108b9d14835916dc654b63b6d6f'})
            if r.status_code == 200:
                d = r.json()
                bi = d.get('balance_infos', [{}])
                bal = bi[0].get('total_balance', '?') if bi else '?'
                result['deepseek'] = {'balance': bal, 'status': 'ok'}
            else:
                result['deepseek'] = {'status': 'error', 'code': r.status_code}
    except Exception as e:
        result['deepseek'] = {'status': 'down', 'error': str(e)[:100]}
    # SiliconFlow
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get('https://api.siliconflow.cn/v1/user/info',
                headers={'Authorization': 'Bearer sk-olueugsredhhxmivvkwklrowwqhajockffwljwcezfmsbysf'})
            if r.status_code == 200:
                d = r.json().get('data', {})
                result['siliconflow'] = {'balance': d.get('totalBalance', '?'), 'status': 'ok'}
            else:
                result['siliconflow'] = {'status': 'error', 'code': r.status_code}
    except Exception as e:
        result['siliconflow'] = {'status': 'down', 'error': str(e)[:100]}
    return result


# Cache for Codex quota API (TTL 30s)
_codex_cache = {'data': None, 'ts': 0}

@router.get('/status/codex')
async def status_codex():
    global _codex_cache
    import time, json, httpx
    now = time.monotonic()
    if _codex_cache['data'] and (now - _codex_cache['ts']) < 30:
        return _codex_cache['data']

    try:
        with open('/home/ubuntu/.codex/auth.json') as f:
            auth = json.load(f)
        token = auth.get('tokens', {}).get('access_token', '')
        if not token:
            return {'error': 'no token'}

        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get('https://chatgpt.com/backend-api/wham/usage',
                headers={'Authorization': f'Bearer {token}'})
            if r.status_code != 200:
                return {'error': f'status {r.status_code}'}
            d = r.json()

        rl = d.get('rate_limit', {})
        pw = rl.get('primary_window', {})
        sw = rl.get('secondary_window', {})
        result = {
            'plan': d.get('plan_type', 'unknown'),
            'email': d.get('email', ''),
            '5h': {
                'used_pct': pw.get('used_percent', 0),
                'reset_secs': pw.get('reset_after_seconds', 0),
                'limit_reached': rl.get('limit_reached', False),
            },
            '7d': {
                'used_pct': sw.get('used_percent', 0),
                'reset_secs': sw.get('reset_after_seconds', 0),
            },
            'limit_reached': rl.get('limit_reached', False),
        }
        _codex_cache = {'data': result, 'ts': now}
        return result
    except Exception as e:
        return {'error': str(e)[:200]}

@router.get("/memory/cards")
async def memory_cards(character_id: str = None):
    """Return approved memory cards, optionally filtered by character_id."""
    from app.core.state import get_config
    import aiosqlite
    cfg = get_config()
    try:
        async with aiosqlite.connect(cfg.storage.sqlite.memory_db, timeout=10.0) as db:
            if character_id:
                cur = await db.execute(
                    "SELECT card_id, content, scope, card_type, importance FROM memory_cards WHERE status='approved' AND character_id=? ORDER BY importance DESC LIMIT 20",
                    (character_id,))
            else:
                cur = await db.execute(
                    "SELECT card_id, content, scope, card_type, importance FROM memory_cards WHERE status='approved' ORDER BY importance DESC LIMIT 20")
            rows = await cur.fetchall()
            cards = [{"id": r[0], "content": r[1], "scope": r[2], "type": r[3], "importance": r[4]} for r in rows]
        return {"cards": cards, "count": len(cards)}
    except Exception as e:
        return {"error": str(e)}
