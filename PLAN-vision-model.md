# 计划: KokoroMemo 视觉模型路由

## 问题
RikkaHub 通过 ModelRegistry 匹配模型名来判断是否支持 vision。`deepseek-v4-flash` 没有 `visionInput()`，所以发图时 RikkaHub 自动触发 OCR。

## 方案
在 KokoroMemo 中增加一个**模型名转换层**：请求模型 `Qwen3.5-27B` → 实际转发到 `gpt-5.5`。
- `Qwen3.5-27B` 匹配 RikkaHub 的 `QWEN_3_5` 注册项 → `visionInput()` ✅
- `gpt-5.5` 是 SiliconFlow 上已有的模型，也支持视觉

## 改动点

### 1. `config.yaml` — 新增 `compatibility.exposed_models`
```yaml
compatibility:
  exposed_models:
    - deepseek-v4-flash
    - deepseek-v4-pro
    - gpt-5.5
    - gpt-5.4
    - gpt-5.4-mini
    - gpt-5.3-codex
    - Qwen3.5-27B       # 新增: 别名模型，路由到 gpt-5.5
```

### 2. `app/api/protocol_common.py` — 如果 exposed_models 为空才动态获取
当前逻辑是: `if exposed_models: return exposed_models` 已有此功能 ✅

### 3. `app/pipeline/chat.py` — 在 `forward()` 中加模型别名映射
```python
# 模型别名映射表
_MODEL_ALIASES = {
    "Qwen3.5-27B": "gpt-5.5",
    "Qwen2.5-VL-7B-Instruct": "gpt-5.5",
}
# 在 forward() 中:
client_model = raw_body.get("model", "")
client_model = _MODEL_ALIASES.get(client_model, client_model)
```

## 测试
1. 重启 KokoroMemo 后，curl /v1/models 应返回 Qwen3.5-27B
2. RikkaHub 连接到 KokoroMemo，应看到 Qwen3.5-27B 在模型列表中
3. 选择 Qwen3.5-27B 发送图片 → OCR 应被跳过（RikkaHub 认为它支持 vision）
4. 实际请求路由到 gpt-5.5 正常完成

## 回滚
配置 exposed_models 改回空列表即可恢复动态模式
