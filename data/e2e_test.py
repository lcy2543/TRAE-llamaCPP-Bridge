"""端到端测试脚本：测试代理的模型列表、非流式、流式、多模态接口"""
import base64
import json
import sys

import requests

BASE = "http://127.0.0.1:8800"
ok = True


def check(name, cond, detail=""):
    global ok
    print(f"[{'通过' if cond else '失败'}] {name} {detail}")
    if not cond:
        ok = False


# 1. /v1/models
r = requests.get(f"{BASE}/v1/models", timeout=10)
check("GET /v1/models", r.status_code == 200 and any(
    m["id"] == "qwen3.6-35b" for m in r.json()["data"]))

# 2. 非流式
r = requests.post(f"{BASE}/v1/chat/completions", timeout=600, json={
    "model": "qwen3.6-35b",
    "messages": [{"role": "user", "content": "1+1等于几？只回答数字"}],
    "max_tokens": 100,
})
data = r.json()
content = data["choices"][0]["message"]["content"]
check("非流式聊天", r.status_code == 200 and content.strip() and
      "reasoning_content" not in data["choices"][0]["message"], f"→ {content!r}")

# 3. 流式
r = requests.post(f"{BASE}/v1/chat/completions", timeout=600, stream=True, json={
    "model": "qwen3.6-35b",
    "messages": [{"role": "user", "content": "用一句话说明什么是 llama.cpp"}],
    "max_tokens": 300,
    "stream": True,
})
text, has_reasoning, done = "", False, False
buf = ""
for chunk in r.iter_content(chunk_size=None):
    if not chunk:
        continue
    buf += chunk.decode("utf-8", "replace")
    while "\n" in buf:
        line, buf = buf.split("\n", 1)
        line = line.rstrip("\r")
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload.strip() == "[DONE]":
            done = True
            break
        obj = json.loads(payload)
        d = obj["choices"][0]["delta"]
        if d.get("content"):
            text += d["content"]
        if "reasoning_content" in d:
            has_reasoning = True
    if done:
        break
check("流式聊天(SSE)", done and text.strip() and not has_reasoning,
      f"→ {text[:80]!r}...")

# 4. 多模态（图片理解，验证 mmproj 生效）
with open(r"data\test_image.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
r = requests.post(f"{BASE}/v1/chat/completions", timeout=600, json={
    "model": "qwen3.6-35b",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "图片里有什么文字和颜色？简短回答"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ],
    }],
    "max_tokens": 200,
})
content = r.json()["choices"][0]["message"]["content"]
check("多模态图片理解(mmproj)", r.status_code == 200 and content.strip(),
      f"→ {content!r}")

# 5. 未知模型 ID 兜底
r = requests.post(f"{BASE}/v1/chat/completions", timeout=600, json={
    "model": "whatever-trae-sends",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 30,
})
check("未知模型ID兜底", r.status_code == 200)

print("\n全部通过 ✔" if ok else "\n存在失败项 ✘")
sys.exit(0 if ok else 1)
