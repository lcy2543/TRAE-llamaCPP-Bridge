"""模型家族预设与显存自动优化

- 官方推荐参数来自各模型官方文档（Google Gemma 模型卡片、Qwen 官方文档等）
- detect_preset(): 根据模型文件名识别家族，返回官方推荐参数
- auto_optimize(): 根据显存大小自动推荐 ctx/ngl/KV 缓存量化
"""

import os
import subprocess
from pathlib import Path

# 各模型家族官方推荐参数（来源见各条 source 字段）
MODEL_PRESETS = [
    {
        "family": "Gemma 4 (Google)",
        "match": ["gemma4", "gemma-4"],
        # Google Gemma 4 模型卡片：所有用例统一 temp=1.0 / top_p=0.95 / top_k=64
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        # llama.cpp 官方 Gemma 4 指南：--jinja 必开（chat 模板与工具调用）
        "jinja": True,
        "reasoning": "auto",
        "source": "Google Gemma 4 模型卡片 (ai.google.dev/gemma/docs/core/model_card_4)",
    },
    {
        "family": "Qwen3.x (Alibaba)",
        "match": ["qwen3", "qwen2.5", "qwq"],
        # Qwen 官方文档：思考模式 temp=0.6 / top_p=0.95 / top_k=20（禁止贪心解码）
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "jinja": True,
        "reasoning": "budget",
        "reasoning_budget": 4096,   # 官方建议 ≥1024 才有实质收益
        "source": "Qwen 官方文档 (qwen.readthedocs.io Quickstart)",
    },
    {
        "family": "DeepSeek R1/V 系列",
        "match": ["deepseek-r1", "deepseek-v", "deepseek"],
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "jinja": True,
        "reasoning": "auto",
        "source": "DeepSeek 模型卡片推荐参数",
    },
    {
        "family": "Llama 3/4 (Meta)",
        "match": ["llama-3", "llama-4", "llama3", "llama4", "metallama"],
        "temperature": 0.6,
        "top_p": 0.9,
        "top_k": 20,
        "jinja": True,
        "reasoning": "auto",
        "source": "Meta Llama 文档推荐参数",
    },
    {
        "family": "GLM (Zhipu)",
        "match": ["glm-4", "glm4", "glm-5", "glm5"],
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "jinja": True,
        "reasoning": "auto",
        "source": "GLM 模型卡片",
    },
    {
        "family": "Mistral",
        "match": ["mistral", "mixtral"],
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "jinja": True,
        "reasoning": "auto",
        "source": "Mistral 文档",
    },
]

# 与 config.DEFAULT_MODEL 对齐的键（预设只覆盖这些）
PRESET_KEYS = ("temperature", "top_p", "top_k", "jinja",
               "reasoning", "reasoning_budget")


def detect_preset(filename_or_id: str):
    """根据文件名/模型 ID 识别家族，返回 (预设dict, None) 或 (None, 家族名)"""
    name = (filename_or_id or "").lower()
    for p in MODEL_PRESETS:
        for kw in p["match"]:
            if kw in name:
                return p, None
    return None, None


# ---------------- 显存检测 ----------------

def get_vram_gb() -> float | None:
    """获取 GPU 显存（GB）。NVIDIA 用 nvidia-smi；无独显返回 None"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=5, stderr=subprocess.DEVNULL)
        vals = [float(x) for x in out.strip().splitlines() if x.strip()]
        return max(vals) / 1024.0 if vals else None
    except Exception:
        return None


# ---------------- 显存自动优化 ----------------

# 粗略估算：KV 缓存(GB) ≈ ctx × 系数。系数按量化档位（对 20~35B 级模型近似），
# 宁可保守（多估），不足时用户可手动微调。
KV_GB_PER_16K = {"f16": 2.0, "q8_0": 1.1, "q4_0": 0.6}
VRAM_RESERVE_GB = 1.6      # 系统/驱动/mmproj 预留
MMPROJ_GB = 0.9            # mmproj 视觉投影粗略占用

CTX_TIERS = [49152, 32768, 24576, 16384, 8192, 4096]


def _file_size_gb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 ** 3)
    except OSError:
        return 0.0


def auto_optimize(model: dict, vram_gb: float | None = None) -> tuple[dict, str]:
    """根据显存与模型大小推荐 ctx_size / ngl / KV 缓存量化 / flash_attn。

    返回 (推荐的参数增量 dict, 说明文字)。
    """
    notes = []
    if vram_gb is None:
        vram_gb = get_vram_gb()
    weights_gb = _file_size_gb(model.get("model_path", "")) or 16.0
    has_mmproj = bool(model.get("mmproj_path"))

    rec = {"flash_attn": "on", "extra_args": "-np 1"}

    # ---- 无独显：纯 CPU/核显方案 ----
    if vram_gb is None:
        rec.update(ngl=0, ctx_size=8192)
        rec["extra_args"] = "--cache-type-k q8_0 --cache-type-v q8_0 -np 1"
        notes.append("未检测到独立显卡：已按纯 CPU 方案配置（ngl=0，上下文 8192）")
        return rec, "；".join(notes)

    notes.append(f"检测到显存 {vram_gb:.0f}GB，模型权重约 {weights_gb:.1f}GB")

    # ---- 有独显：按剩余空间选 KV 量化与上下文 ----
    fixed = VRAM_RESERVE_GB + (MMPROJ_GB if has_mmproj else 0)
    budget = vram_gb - fixed
    if weights_gb <= budget * 0.92:
        # 权重能全进显存 → 优化目标是塞下尽量大的上下文
        for kv_type in ("q8_0", "q4_0"):
            coef = KV_GB_PER_16K[kv_type]
            for ctx in CTX_TIERS:
                kv_gb = ctx / 16384 * coef
                if weights_gb + kv_gb <= budget:
                    rec.update(ngl=999, ctx_size=ctx)
                    rec["extra_args"] = (f"--cache-type-k {kv_type} "
                                         f"--cache-type-v {kv_type} -np 1")
                    notes.append(f"权重全部进显存：上下文 {ctx}，KV 缓存 {kv_type}")
                    return rec, "；".join(notes)
        rec.update(ngl=999, ctx_size=4096)
        rec["extra_args"] = "--cache-type-k q4_0 --cache-type-v q4_0 -np 1"
        notes.append("显存非常紧张：已用最小配置（上下文 4096 + q4_0 缓存）")
        return rec, "；".join(notes)

    # ---- 权重超预算 → 按比例分配 GPU 层数，其余走内存 ----
    kv_type = "q8_0"
    usable = max(budget - 0.8, budget * 0.6)   # 留 KV 的空间
    ratio = usable / weights_gb
    ngl = max(1, int(ratio * 999))
    rec.update(ngl=ngl, ctx_size=8192)
    rec["extra_args"] = f"--cache-type-k {kv_type} --cache-type-v {kv_type} -np 1"
    notes.append(f"权重超过显存：约 {int(ratio*100)}% 层进显存（ngl={ngl}），"
                 f"其余走内存（速度会下降），上下文 8192")
    return rec, "；".join(notes)


def apply_preset_and_optimize(model: dict, optimize_vram: bool = True) -> tuple[dict, str]:
    """便捷入口：预设 + 显存优化一起套用，返回 (新模型dict, 说明)"""
    result = dict(model)
    preset, _ = detect_preset(model.get("model_path", "") or model.get("id", ""))
    if preset:
        for k in PRESET_KEYS:
            if preset.get(k) is not None:
                result[k] = preset[k]
    note = ""
    if optimize_vram:
        rec, note = auto_optimize(result)
        result.update(rec)
    return result, note
