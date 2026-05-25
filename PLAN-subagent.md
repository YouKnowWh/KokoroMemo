# 执行计划: Plana 子 Agent + 视觉模型收尾

## 工作流
Codex plan → Claude execute → 直接改代码

## 改动清单

### 1. Hermes config — 添加 Plana 角色人设
`~/.hermes/config.yaml` → `personalities` 段

### 2. Hermes config — delegation 配置调整
- 子 Agent 走 KokoroMemo（同主 Agent）
- personality 自动切换为 plana
- 主 Agent 继续用 arona

### 3. KokoroMemo exposed_models — 暴露 Qwen3.5-27B
已在 LiteLLM 配好了，KokoroMemo 的 exposed_models 已更新 ✅

## Plana 人设定稿

普拉娜（Plana / プラナ）：
- Blue Archive 中 ARONA 的"妹妹"型 AI
- 性格认真、专注、注重效率
- 语言简洁直接，不带表情符号
- 专注于任务执行不闲聊
- 称呼用户为"老师"但比 ARONA 更简洁
- 用色是白/浅蓝调性

## 不做的改动
- ❌ 不动 LiteLLM config（视觉模型已配好）
- ❌ 不动 KokoroMemo 核心逻辑
- ❌ 不加 Anthropic 协议（OpenAI 协议足够）
