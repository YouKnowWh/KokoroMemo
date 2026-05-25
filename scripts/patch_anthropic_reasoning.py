#!/usr/bin/env python3
"""Patch routes_anthropic.py to preserve DeepSeek reasoning_content across turns."""

import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/apps/kokoromemo/app/api/routes_anthropic.py"

with open(path) as f:
    content = f.read()

# 1. Add reasoning cache functions after the imports section
cache_code = """

# === DeepSeek reasoning_content cache (must pass back in multi-turn tool calls) ===
_reasoning_cache: dict[str, str] = {}


def _save_reasoning(conversation_id: str, reasoning: str) -> None:
    if reasoning:
        _reasoning_cache[conversation_id] = reasoning


def _get_reasoning(conversation_id: str) -> str:
    return _reasoning_cache.pop(conversation_id, "")

"""

# Find insertion point: after last import/from statement
lines = content.split("\n")
insert_idx = 0
for i, line in enumerate(lines):
    if line.startswith("import ") or line.startswith("from "):
        insert_idx = i + 1
    # Stop when we hit non-import, non-blank, non-comment lines
    elif line.strip() and not line.startswith("#") and not line.startswith('"""'):
        # Check if this is still in the docstring
        if i > 20:  # imports should be in first ~20 lines
            break

# Insert after the last import
lines.insert(insert_idx, cache_code)
content = "\n".join(lines)

# 2. In _openai_response_to_anthropic, save reasoning_content from the OpenAI message
# The message is extracted at: message = choices[0].get("message", {}) if choices else {}
old = 'message = choices[0].get("message", {}) if choices else {}\n    content_blocks = _openai_message_to_anthropic_content(message)'
new = '''message = choices[0].get("message", {}) if choices else {}
    reasoning = message.get("reasoning_content") or ""
    if reasoning:
        _reasoning_cache["_last"] = reasoning
    content_blocks = _openai_message_to_anthropic_content(message)'''
content = content.replace(old, new)

# 3. In _anthropic_to_openai_messages, add reasoning_content to assistant messages with tool_calls
old = 'if tool_calls:\n                assistant_message["tool_calls"] = tool_calls'
new = '''if tool_calls:
                assistant_message["tool_calls"] = tool_calls
                reasoning = _reasoning_cache.pop("_last", "")
                if reasoning:
                    assistant_message["reasoning_content"] = reasoning'''
content = content.replace(old, new)

with open(path, "w") as f:
    f.write(content)

print(f"Patched {path}")
print("Changes:")
print("  1. Added _reasoning_cache dict + helpers")
print("  2. _openai_response_to_anthropic: saves reasoning_content")
print("  3. _anthropic_to_openai_messages: adds reasoning_content to assistant with tool_calls")
