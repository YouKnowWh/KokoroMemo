"""Chat completion request pipeline for the OpenAI-compatible proxy."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.background import spawn_background
from app.core.conversation_locks import conversation_lock
from app.core.ids import generate_id
from app.core.services import ServiceRegistry, get_service_registry
from app.core.state import get_config
from app.memory.card_injector import inject_cards
from app.memory.context_embedding_cache import get_context_cache
from app.memory.query_builder import build_retrieval_query
from app.memory.retrieval_gate import RetrievalGateInput, decide_retrieval
from app.memory.state_injector import inject_state_board
from app.memory.state_schema import StateRenderOptions
from app.memory.state_filler import StateFillerConfigView
from app.memory.state_table_filler import fill_conversation_state_tables
from app.memory.state_table_renderer import render_state_tables
from app.proxy.request_parser import RequestContext, resolve_context
from app.storage.migrations import apply_startup_migrations
from app.storage.sqlite_app import upsert_character, upsert_conversation
from app.storage.sqlite_conversation import (
    get_turn_count,
    init_chat_db,
    save_injected_memory_log,
    save_raw_request,
    save_raw_response,
    save_turn_and_messages,
)
from app.storage.sqlite_state import SQLiteStateStore

logger = logging.getLogger("kokoromemo.pipeline.chat")


def _extra_trigger_keywords(cfg) -> list[str]:
    """Return additional trigger keywords for languages other than the config default."""
    from app.core.prompts import TRIGGER_KEYWORDS

    lang = cfg.language
    extra = []
    for key, words in TRIGGER_KEYWORDS.items():
        if key != lang:
            extra.extend(words)
    return extra


@dataclass
class ChatPipelineContext:
    request: Request
    cfg: Any
    ctx: RequestContext
    raw_body: dict[str, Any]
    messages: list[dict]
    injected_messages: list[dict]
    state_store: SQLiteStateStore | None
    conversation_config: Any | None
    should_inject_state: bool
    should_inject_memory: bool
    state_row_count: int = 0
    avg_state_confidence: float | None = None


class ChatPipeline:
    """Coordinates request persistence, memory injection, forwarding and post-processing."""

    def __init__(self, services: ServiceRegistry | None = None) -> None:
        self.services = services or get_service_registry()

    async def handle(self, request: Request, raw_body: dict[str, Any] | None = None):
        prepared = await self.prepare(request, raw_body=raw_body)
        await self.inject_state(prepared)
        await self.inject_memory(prepared)
        return await self.forward(prepared)

    async def prepare(self, request: Request, raw_body: dict[str, Any] | None = None) -> ChatPipelineContext:
        cfg = get_config()
        if raw_body is None:
            raw_body = await request.json()
        ctx = await resolve_context(request, raw_body, cfg.storage.root_dir, cfg)
        await _persist_request(cfg, ctx, deepcopy(raw_body))

        messages = deepcopy(raw_body.get("messages", []))
        self._resolve_system_variables(cfg, ctx, messages)

        state_store = None
        conversation_config = None
        if cfg.memory.enabled:
            try:
                state_store = SQLiteStateStore(cfg.storage.sqlite.memory_db)
                conversation_config = await state_store.ensure_conversation_config(ctx.conversation_id)
            except Exception as exc:
                logger.warning("Conversation policy loading failed (degraded): %s", exc)

        injection_policy = conversation_config.injection_policy if conversation_config else "mixed"
        return ChatPipelineContext(
            request=request,
            cfg=cfg,
            ctx=ctx,
            raw_body=raw_body,
            messages=messages,
            injected_messages=messages,
            state_store=state_store,
            conversation_config=conversation_config,
            should_inject_state=injection_policy in {"state_only", "state_first", "mixed"},
            should_inject_memory=injection_policy in {"memory_only", "state_first", "mixed"},
        )

    async def inject_state(self, prepared: ChatPipelineContext) -> None:
        cfg = prepared.cfg
        if not (cfg.memory.enabled and cfg.memory.hot_context.enabled and prepared.should_inject_state):
            return
        try:
            state_store = prepared.state_store or SQLiteStateStore(cfg.storage.sqlite.memory_db)
            prepared.state_store = state_store
            ctx = prepared.ctx
            table_template = await state_store.get_conversation_table_template(ctx.conversation_id)
            table_rows = await state_store.list_table_rows(
                ctx.conversation_id,
                table_template.template_id if table_template else None,
            )
            active_rows = [row for row in table_rows if row.status == "active"]
            prepared.state_row_count = len(active_rows)
            if active_rows:
                prepared.avg_state_confidence = sum(row.confidence for row in active_rows) / len(active_rows)

            render_options = StateRenderOptions(max_chars=cfg.memory.hot_context.max_chars)
            state_text = render_state_tables(table_template, table_rows, render_options, lang=cfg.language)
            if state_text:
                prepared.injected_messages = inject_state_board(prepared.injected_messages, state_text)
        except Exception as exc:
            logger.warning("Hot context injection failed (degraded): %s", exc)

    async def inject_memory(self, prepared: ChatPipelineContext) -> None:
        cfg = prepared.cfg
        if not (
            cfg.memory.enabled
            and cfg.memory.inject_enabled
            and cfg.embedding.enabled
            and prepared.should_inject_memory
        ):
            return
        try:
            ctx = prepared.ctx
            query = build_retrieval_query(
                prepared.messages,
                ctx.user_id,
                ctx.character_id,
                ctx.conversation_id,
                max_recent_turns=cfg.memory.max_recent_turns_for_query,
            )
            should_retrieve = True
            if cfg.memory.retrieval_gate.enabled:
                turn_index = await get_turn_count(ctx.chat_db_path, ctx.conversation_id)
                decision = decide_retrieval(
                    RetrievalGateInput(
                        query=query,
                        state_row_count=prepared.state_row_count,
                        avg_state_confidence=prepared.avg_state_confidence,
                        turn_index=turn_index,
                        mode=cfg.memory.retrieval_gate.mode,
                        vector_search_on_new_session=cfg.memory.retrieval_gate.vector_search_on_new_session,
                        vector_search_every_n_turns=cfg.memory.retrieval_gate.vector_search_every_n_turns,
                        vector_search_when_state_confidence_below=cfg.memory.retrieval_gate.vector_search_when_state_confidence_below,
                        trigger_keywords=cfg.memory.retrieval_gate.trigger_keywords + _extra_trigger_keywords(cfg),
                        skip_when_latest_user_text_chars_below=cfg.memory.retrieval_gate.skip_when_latest_user_text_chars_below,
                        skip_when_state_is_sufficient=cfg.memory.retrieval_gate.skip_when_state_is_sufficient,
                    )
                )
                should_retrieve = decision.should_retrieve
                await self._record_retrieval_decision(prepared, query, decision, turn_index)

            if not should_retrieve:
                return

            ep = self.services.get_embedding_provider(cfg)
            store = self.services.get_lancedb_store(cfg)
            if not ep or not store:
                return

            from app.memory.card_retriever import retrieve_cards

            allowed_scopes = {
                scope for scope, enabled in (
                    ("global", cfg.memory.scopes.include_global),
                    ("character", cfg.memory.scopes.include_character),
                    ("conversation", cfg.memory.scopes.include_conversation),
                ) if enabled
            }
            candidates = await retrieve_cards(
                query,
                ep,
                store,
                cards_db_path=cfg.storage.sqlite.memory_db,
                vector_top_k=cfg.memory.vector_top_k,
                final_top_k=cfg.memory.final_top_k,
                allowed_scopes=allowed_scopes,
            )
            if candidates:
                prepared.injected_messages = inject_cards(
                    prepared.injected_messages,
                    candidates,
                    max_chars=cfg.memory.max_injected_chars,
                    max_count=cfg.memory.final_top_k,
                    username=ctx.user_id,
                    character_name=ctx.character_id,
                    model_name=cfg.llm.model,
                    conversation_id=ctx.conversation_id,
                )
                await _persist_injection(ctx, prepared.injected_messages, candidates)
                logger.info("Injected %d memories for conv=%s", len(candidates), ctx.conversation_id)
        except Exception as exc:
            logger.warning("Memory retrieval failed (degraded): %s", exc)

    async def forward(self, prepared: ChatPipelineContext):
        cfg = prepared.cfg
        raw_body = prepared.raw_body
        request = prepared.request
        forward_body = deepcopy(raw_body)
        forward_body["messages"] = prepared.injected_messages

        from app.proxy.llm_providers import create_llm_provider

        client_auth = request.headers.get("authorization", "")
        client_api_key = client_auth.replace("Bearer ", "").strip() if client_auth.startswith("Bearer ") else ""
        client_model = raw_body.get("model", "")
        if cfg.llm.forward_mode == "passthrough":
            final_api_key = cfg.llm.get_api_key() or client_api_key
            final_model = client_model or cfg.llm.model
        else:
            final_api_key = cfg.llm.get_api_key()
            final_model = cfg.llm.model or client_model

        if not cfg.llm.base_url:
            return JSONResponse(status_code=500, content={
                "error": {"message": "未配置 LLM Base URL，请在设置中配置对话大模型", "type": "config_error", "param": None, "code": "no_base_url"}
            })
        if final_model:
            forward_body["model"] = final_model

        provider = create_llm_provider(
            provider=cfg.llm.provider,
            base_url=cfg.llm.base_url,
            api_key=final_api_key,
            model=final_model,
        )
        if raw_body.get("stream", False):
            return StreamingResponse(
                _stream_proxy(provider, forward_body, cfg.llm.timeout_seconds, prepared.ctx, cfg, prepared.messages, self.services),
                media_type="text/event-stream",
            )
        return await _non_stream_proxy(provider, forward_body, cfg.llm.timeout_seconds, prepared.ctx, cfg, prepared.messages, self.services)

    async def _record_retrieval_decision(self, prepared: ChatPipelineContext, query, decision, turn_index: int) -> None:
        try:
            cfg = prepared.cfg
            ctx = prepared.ctx
            state_store = prepared.state_store or SQLiteStateStore(cfg.storage.sqlite.memory_db)
            prepared.state_store = state_store
            await state_store.record_retrieval_decision(
                request_id=ctx.request_id,
                conversation_id=ctx.conversation_id,
                user_id=ctx.user_id,
                character_id=ctx.character_id,
                mode=decision.mode,
                should_retrieve=decision.should_retrieve,
                reason=decision.reason,
                reasons=decision.reasons,
                latest_user_text=query.latest_user_text,
                state_item_count=decision.state_item_count,
                avg_state_confidence=decision.avg_state_confidence,
                turn_index=turn_index,
            )
        except Exception as exc:
            logger.warning("Failed to persist retrieval gate decision: %s", exc)

    def _resolve_system_variables(self, cfg, ctx: RequestContext, messages: list[dict]) -> None:
        from app.core.variables import resolve_variables

        var_kwargs = dict(
            username=ctx.user_id,
            character_name=ctx.character_id,
            model_name=cfg.llm.model,
            conversation_id=ctx.conversation_id,
        )
        for index, message in enumerate(messages):
            if message.get("role") == "system" and "{{" in message.get("content", ""):
                messages[index] = dict(message)
                messages[index]["content"] = resolve_variables(message["content"], **var_kwargs)


async def _persist_injection(ctx: RequestContext, injected_messages: list[dict], candidates: list[Any]) -> None:
    injected_text = ""
    for msg in injected_messages:
        content = msg.get("content", "")
        if msg.get("role") == "system" and content.startswith("【KokoroMemo 长期记忆】"):
            injected_text = content
            break

    if not injected_text:
        return

    try:
        card_ids = [getattr(candidate, "card_id", "") for candidate in candidates]
        await save_injected_memory_log(
            ctx.chat_db_path,
            generate_id("inj_"),
            ctx.request_id,
            ctx.conversation_id,
            injected_text,
            json.dumps([card_id for card_id in card_ids if card_id], ensure_ascii=False),
        )
    except Exception as e:
        logger.warning("Failed to persist injection log: %s", e)


async def _persist_request(cfg, ctx: RequestContext, raw_body: dict) -> None:
    try:
        await apply_startup_migrations(cfg)
        await init_chat_db(ctx.chat_db_path)
        await upsert_conversation(
            cfg.storage.sqlite.app_db, ctx.conversation_id,
            ctx.user_id, ctx.character_id, ctx.client_name, ctx.conv_dir,
        )
        if ctx.character_id:
            await upsert_character(
                cfg.storage.sqlite.app_db, ctx.character_id, ctx.user_id,
            )
        await _apply_character_defaults_if_new(cfg, ctx)
        await _apply_default_mount_preset_if_new(cfg, ctx)
        await save_raw_request(
            ctx.chat_db_path, ctx.request_id, ctx.conversation_id,
            json.dumps(raw_body, ensure_ascii=False),
        )
    except Exception as e:
        logger.warning("Failed to persist request: %s", e)


async def _apply_character_defaults_if_new(cfg, ctx: RequestContext) -> None:
    """Auto-apply character defaults to a new conversation (no existing mounts)."""
    if not ctx.character_id:
        return
    try:
        from app.storage.sqlite_app import get_character_defaults
        from app.storage.sqlite_cards import get_conversation_mounts, get_mount_preset, set_conversation_mounts
        from app.storage.sqlite_state import SQLiteStateStore
        import json as json_mod

        mounts = await get_conversation_mounts(cfg.storage.sqlite.memory_db, ctx.conversation_id)
        if mounts and any(m.get("library_id") != "lib_default" for m in mounts):
            return

        defaults = await get_character_defaults(cfg.storage.sqlite.app_db, ctx.character_id)
        if not defaults or not defaults.get("auto_apply"):
            return

        preset = await get_mount_preset(cfg.storage.sqlite.memory_db, defaults.get("mount_preset_id")) if defaults.get("mount_preset_id") else None
        if preset:
            library_ids = json_mod.loads(preset.get("library_ids_json") or "[]") or ["lib_default"]
            write_library_id = preset.get("write_library_id") or library_ids[0]
        else:
            library_ids = defaults.get("library_ids") or ["lib_default"]
            write_library_id = defaults.get("write_library_id") or library_ids[0]
        await set_conversation_mounts(cfg.storage.sqlite.memory_db, ctx.conversation_id, library_ids, write_library_id)

        store = SQLiteStateStore(cfg.storage.sqlite.memory_db)
        await store.set_conversation_config({
            "conversation_id": ctx.conversation_id,
            "profile_id": defaults.get("profile_id"),
            "table_template_id": defaults.get("table_template_id"),
            "mount_preset_id": defaults.get("mount_preset_id"),
            "memory_write_policy": defaults.get("memory_write_policy"),
            "state_update_policy": defaults.get("state_update_policy"),
            "injection_policy": defaults.get("injection_policy"),
            "created_from_default": True,
        })
    except Exception as e:
        logger.debug("Character defaults auto-apply skipped: %s", e)


async def _apply_default_mount_preset_if_new(cfg, ctx: RequestContext) -> None:
    """Auto-apply the global default mount preset to conversations without custom mounts."""
    try:
        from app.storage.sqlite_cards import get_conversation_mounts, get_mount_preset, set_conversation_mounts
        from app.storage.sqlite_state import SQLiteStateStore
        import json as json_mod

        mounts = await get_conversation_mounts(cfg.storage.sqlite.memory_db, ctx.conversation_id)
        if mounts and any(mount.get("library_id") != "lib_default" for mount in mounts):
            return

        default_config = await SQLiteStateStore(cfg.storage.sqlite.memory_db).get_default_conversation_config()
        if not default_config.mount_preset_id:
            return

        preset = await get_mount_preset(cfg.storage.sqlite.memory_db, default_config.mount_preset_id)
        if not preset:
            return
        library_ids = json_mod.loads(preset.get("library_ids_json") or "[]") or ["lib_default"]
        await set_conversation_mounts(
            cfg.storage.sqlite.memory_db,
            ctx.conversation_id,
            library_ids,
            preset.get("write_library_id") or library_ids[0],
        )
    except Exception as e:
        logger.debug("Default mount preset auto-apply skipped: %s", e)


async def _persist_response_turn(
    ctx: RequestContext,
    original_messages: list[dict],
    assistant_text: str,
    response_json: str | None,
    stream_text: str | None,
) -> tuple[str | None, int | None]:
    try:
        resp_id = generate_id("resp_")
        await save_raw_response(
            ctx.chat_db_path, resp_id, ctx.request_id, ctx.conversation_id,
            body_json=response_json, stream_text=stream_text,
        )
        all_msgs = list(original_messages)
        if assistant_text:
            all_msgs.append({"role": "assistant", "content": assistant_text})
        turn_id = generate_id("turn_")
        turn_index = await get_turn_count(ctx.chat_db_path, ctx.conversation_id)
        await save_turn_and_messages(
            ctx.chat_db_path, turn_id, ctx.conversation_id,
            ctx.user_id, ctx.character_id, ctx.request_id, turn_index, all_msgs,
        )
        return turn_id, turn_index
    except Exception as exc:
        logger.warning("Failed to persist response: %s", exc)
        return None, None


async def _schedule_post_process_turn(
    ctx: RequestContext,
    cfg,
    services: ServiceRegistry,
    original_messages: list[dict],
    assistant_text: str,
    turn_id: str | None,
    turn_index: int | None,
    *,
    name: str,
) -> None:
    task = spawn_background(
        _post_process_turn(
            ctx,
            cfg,
            services,
            original_messages,
            assistant_text,
            turn_id,
            turn_index,
        ),
        name=name,
    )


async def _post_process_turn(
    ctx: RequestContext,
    cfg,
    services: ServiceRegistry,
    original_messages: list[dict],
    assistant_text: str,
    turn_id: str | None,
    turn_index: int | None,
) -> None:
    async with conversation_lock(ctx.conversation_id):
        await _update_state_and_extract_memories(
            ctx,
            cfg,
            services,
            original_messages,
            assistant_text,
            turn_id,
            turn_index,
        )


async def _update_state_and_extract_memories(
    ctx: RequestContext,
    cfg,
    services: ServiceRegistry,
    original_messages: list[dict],
    assistant_text: str,
    turn_id: str | None,
    turn_index: int | None,
) -> None:
    conversation_config = None
    if cfg.memory.enabled:
        try:
            conversation_config = await SQLiteStateStore(cfg.storage.sqlite.memory_db).ensure_conversation_config(ctx.conversation_id)
        except Exception as exc:
            logger.warning("Conversation policy loading failed during extraction (degraded): %s", exc)

    state_update_policy = conversation_config.state_update_policy if conversation_config else "auto"
    memory_write_policy = conversation_config.memory_write_policy if conversation_config else "candidate"

    if (
        cfg.memory.enabled
        and cfg.memory.state_updater.enabled
        and cfg.memory.state_updater.update_after_each_turn
        and state_update_policy == "auto"
    ):
        if assistant_text and _should_run_state_updater(cfg, turn_index):
            user_msg = _latest_user_message(original_messages)
            if user_msg:
                try:
                    await fill_conversation_state_tables(
                        db_path=cfg.storage.sqlite.memory_db,
                        conversation_id=ctx.conversation_id,
                        user_message=user_msg,
                        assistant_message=assistant_text,
                        turn_id=turn_id,
                        config=StateFillerConfigView(
                            provider=cfg.memory.state_updater.provider,
                            base_url=cfg.memory.state_updater.base_url or cfg.memory.judge.base_url or cfg.llm.base_url,
                            api_key=cfg.memory.state_updater.get_api_key() or cfg.memory.judge.get_api_key() or cfg.llm.get_api_key(),
                            model=cfg.memory.state_updater.model or cfg.memory.judge.model or cfg.llm.model,
                            timeout_seconds=cfg.memory.state_updater.timeout_seconds,
                            temperature=cfg.memory.state_updater.temperature,
                            min_confidence=cfg.memory.state_updater.min_confidence,
                            prompt=cfg.memory.state_updater.prompt,
                        ),
                        lang=cfg.language,
                    )
                except Exception as exc:
                    logger.warning("State updater failed: %s", exc)

    if not cfg.memory.enabled or not cfg.memory.extraction_enabled:
        extraction_enabled = False
    elif memory_write_policy == "disabled":
        logger.info("Memory extraction skipped for conv=%s by policy=disabled", ctx.conversation_id)
        extraction_enabled = False
    elif not assistant_text:
        extraction_enabled = False
    elif not _latest_user_message(original_messages):
        extraction_enabled = False
    else:
        extraction_enabled = True

    if extraction_enabled:
        user_msg = _latest_user_message(original_messages)
        try:
            from app.memory.card_extractor import extract_and_route
            from app.memory.judge import MemoryJudgeConfigView

            ep = services.get_embedding_provider(cfg)
            store = services.get_lancedb_store(cfg)
            judge_config = None
            if cfg.memory.judge.enabled:
                user_rules = cfg.memory.judge.user_rules
                if memory_write_policy == "stable_only":
                    user_rules = (
                        f"{user_rules}\n\n"
                        "当前会话策略为 stable_only：只允许用户偏好、角色稳定设定、世界观常识、稳定关系等长期事实进入记忆候选；"
                        "临时事件、机械状态、剧情进度、资源变化、任务进度、小人即时状态等必须判为不写入长期记忆。"
                    ).strip()
                judge_config = MemoryJudgeConfigView(
                    provider=cfg.memory.judge.provider,
                    base_url=cfg.memory.judge.base_url or cfg.llm.base_url,
                    api_key=cfg.memory.judge.get_api_key() or cfg.llm.get_api_key(),
                    model=cfg.memory.judge.model or cfg.llm.model,
                    timeout_seconds=cfg.memory.judge.timeout_seconds,
                    temperature=cfg.memory.judge.temperature,
                    mode=cfg.memory.judge.mode,
                    user_rules=user_rules,
                    prompt=cfg.memory.judge.prompt,
                )

            await extract_and_route(
                db_path=cfg.storage.sqlite.memory_db,
                user_message=user_msg,
                assistant_message=assistant_text,
                user_id=ctx.user_id,
                character_id=ctx.character_id,
                conversation_id=ctx.conversation_id,
                embedding_provider=ep,
                lancedb_store=store,
                min_importance=cfg.memory.extraction.min_importance,
                min_confidence=cfg.memory.extraction.min_confidence,
                judge_config=judge_config,
                lang=cfg.language,
                discarded_keep_limit=cfg.memory.extraction.discarded_keep_limit,
            )
        except Exception as exc:
            logger.warning("Memory extraction failed: %s", exc)

    # Precompute context embedding for the next retrieval turn
    if cfg.memory.enabled and cfg.embedding.enabled and assistant_text and original_messages:
        try:
            await _precompute_context_embedding(cfg, services, ctx, original_messages, assistant_text, turn_index)
        except Exception as exc:
            logger.warning("Context embedding precomputation failed (degraded): %s", exc)


def _latest_user_message(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def _should_run_state_updater(cfg, turn_index: int | None) -> bool:
    every_n = cfg.memory.state_updater.update_every_n_turns
    if every_n <= 1 or turn_index is None:
        return True
    return turn_index % every_n == 0


def _build_recent_context_text(messages: list[dict], assistant_text: str, max_turns: int) -> str:
    """Build a recent context string (last N turns including the latest assistant response)."""
    non_system = [m for m in messages if m.get("role") != "system"]
    recent = non_system[-(max_turns * 2):]
    recent.append({"role": "assistant", "content": assistant_text})

    lines = []
    for m in recent:
        role = m.get("role", "")
        content = m.get("content", "")[:120]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


_CONTEXT_REFRESH_INTERVAL = 3  # force refresh every N turns regardless of topic


async def _check_topic_changed(prev_context: str, curr_context: str, cfg) -> bool:
    """Ask LLM whether the latest messages introduce a new topic."""
    if not prev_context.strip():
        return True

    # Only compare the new portion: last 300 chars of current vs last 300 of previous
    prev_tail = prev_context[-300:].strip()
    curr_tail = curr_context[-300:].strip()
    if not curr_tail or curr_tail == prev_tail:
        return False

    prompt = (
        "Does the SECOND text introduce a clearly different topic from the FIRST?\n\n"
        f"FIRST:\n{prev_tail}\n\n"
        f"SECOND:\n{curr_tail}\n\n"
        "Answer ONLY the word YES or NO. Do not explain."
    )

    try:
        base_url = cfg.memory.judge.base_url or cfg.llm.base_url
        api_key = cfg.memory.judge.get_api_key() or cfg.llm.get_api_key()
        model = cfg.memory.judge.model or cfg.llm.model
        if not base_url or not api_key:
            return True

        url = f"{base_url.rstrip('/')}/chat/completions"
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(url, json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a classifier. Output only YES or NO. No reasoning."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 100,
                "temperature": 0,
            }, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            })
            if resp.status_code != 200:
                logger.warning("Topic check LLM returned %d, defaulting to refresh", resp.status_code)
                return True
            body = resp.json()
            msg = body["choices"][0]["message"]
            answer = (msg.get("content") or msg.get("reasoning_content") or "").strip().upper()
            if not answer:
                logger.info("Topic check: empty answer, defaulting to refresh")
                return True
            # Try prefix first, then search anywhere
            if answer.startswith("YES"):
                changed = True
            elif answer.startswith("NO"):
                changed = False
            elif "YES" in answer and "NO" not in answer:
                changed = True
            elif "NO" in answer and "YES" not in answer:
                changed = False
            else:
                logger.info("Topic check: ambiguous answer, defaulting to refresh. answer=%s", answer[:60])
                return True
            logger.info("Topic check: prev_tail=%d curr_tail=%d changed=%s answer=%s",
                        len(prev_tail), len(curr_tail), changed, answer[:40])
            return changed
    except Exception as exc:
        logger.warning("Topic check failed (degraded, will refresh): %s", exc)
        return True


async def _precompute_context_embedding(cfg, services, ctx, original_messages: list[dict],
                                         assistant_text: str, turn_index: int | None = None) -> None:
    """Precompute and cache the embedding of the recent conversation context.

    Skips refresh unless topic has changed or N turns have passed since last refresh.
    """
    ep = services.get_embedding_provider(cfg)
    if not ep:
        return

    context_text = _build_recent_context_text(
        original_messages, assistant_text, cfg.memory.max_recent_turns_for_query,
    )
    if not context_text.strip():
        return

    cache = get_context_cache()
    turn = turn_index or 0
    prev_text, last_turn = cache.get_meta(ctx.conversation_id)

    should_refresh = True
    if prev_text and last_turn is not None:
        turns_since = turn - last_turn
        if turns_since >= _CONTEXT_REFRESH_INTERVAL:
            logger.info("Context refresh forced (N=%d): conv=%s turns_since=%d", _CONTEXT_REFRESH_INTERVAL, ctx.conversation_id[:12], turns_since)
        else:
            topic_changed = await _check_topic_changed(prev_text, context_text, cfg)
            if not topic_changed:
                should_refresh = False
                logger.info("Context refresh skipped (topic unchanged): conv=%s turns_since=%d", ctx.conversation_id[:12], turns_since)
            else:
                logger.info("Context refresh triggered (topic changed): conv=%s turns_since=%d", ctx.conversation_id[:12], turns_since)

    if should_refresh:
        vector = await ep.embed_text(context_text)
        cache.put(ctx.conversation_id, ep.model, vector, context_text=context_text, turn_index=turn)
        logger.info("Context embedding refreshed: conv=%s chars=%d turn=%d", ctx.conversation_id[:12], len(context_text), turn)


async def _non_stream_proxy(provider, body: dict, timeout: int, ctx: RequestContext, cfg, original_messages: list[dict], services: ServiceRegistry) -> JSONResponse:
    try:
        resp_data = await provider.chat(body, timeout)

        assistant_text = ""
        choices = resp_data.get("choices", [])
        if choices:
            assistant_text = choices[0].get("message", {}).get("content", "")

        turn_id, turn_index = await _persist_response_turn(
            ctx, original_messages, assistant_text, json.dumps(resp_data, ensure_ascii=False), None
        )
        await _schedule_post_process_turn(
            ctx, cfg, services, original_messages, assistant_text, turn_id, turn_index,
            name=f"post_process_turn:{ctx.request_id}",
        )

        return JSONResponse(content=resp_data, status_code=200)
    except httpx.TimeoutException:
        return JSONResponse(status_code=504, content={"error": {"message": "Upstream LLM request timed out", "type": "proxy_error", "param": None, "code": "upstream_timeout"}})
    except Exception as e:
        logger.error("Upstream request failed: %s", e)
        return JSONResponse(status_code=502, content={"error": {"message": f"Upstream LLM request failed: {e}", "type": "proxy_error", "param": None, "code": "upstream_error"}})


async def _stream_proxy(provider, body: dict, timeout: int, ctx: RequestContext, cfg, original_messages: list[dict], services: ServiceRegistry):
    collected_text: list[str] = []
    try:
        async for line in provider.stream_chat(body, timeout):
            yield f"{line}\n\n"
            if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                try:
                    chunk = json.loads(line[6:])
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        collected_text.append(content)
                except (json.JSONDecodeError, IndexError, KeyError):
                    pass
    except httpx.TimeoutException:
        yield 'data: {"error":{"message":"Upstream LLM stream timed out","type":"proxy_error","code":"upstream_timeout"}}\n\n'
        return
    except Exception as e:
        logger.error("Stream proxy error: %s", e)
        yield f'data: {{"error":{{"message":"Stream error: {e}","type":"proxy_error","code":"upstream_error"}}}}\n\n'
        return

    full_text = "".join(collected_text)
    turn_id, turn_index = await _persist_response_turn(
        ctx, original_messages, full_text, None, full_text
    )
    await _schedule_post_process_turn(
        ctx, cfg, services, original_messages, full_text, turn_id, turn_index,
        name=f"post_process_turn_stream:{ctx.request_id}",
    )

