"""配置管理：全局设置 + 模型列表，持久化为 data/config.json"""

import json
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Optional

# PyInstaller 打包后数据目录跟随 exe 所在目录；源码运行时为项目根目录
if getattr(sys, "frozen", False):
    ROOT_DIR = Path(sys.executable).resolve().parent
else:
    ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CONFIG_FILE = DATA_DIR / "config.json"
BACKEND_LOG = DATA_DIR / "llama-server.log"

DEFAULT_GLOBAL = {
    "proxy_port": 8800,          # 对外 OpenAI 兼容端口（TRAE 填这个）
    "backend_port": 8801,        # llama-server 内部端口
    "llama_server_path": "",     # 留空则自动探测
    "auto_load": True,           # 收到请求时按需自动加载模型
    "strip_think": True,         # 剥离模型回复中的思考标签，界面更干净
    "default_model": "",         # 默认模型 ID（TRAE 填的 ID 不匹配时的兜底）
}

DEFAULT_MODEL = {
    "id": "",                    # 模型 ID（TRAE 中填写）
    "model_path": "",            # LLM GGUF 路径
    "mmproj_path": "",           # 多模态 mmproj GGUF 路径（可空）
    "ctx_size": 8192,            # 上下文长度（越小越省显存/内存）
    "ngl": 999,                  # GPU 卸载层数（999=全部卸载到显卡）
    "threads": -1,               # CPU 线程数，-1=自动
    "flash_attn": "auto",        # on / off / auto，auto 或 on 可省显存
    "reasoning": "auto",         # auto=模型默认思考；off=禁用思考；budget=用思考预算
    "reasoning_budget": 4096,    # 思考 token 上限（reasoning=budget 时生效；官方建议>=1024）
    "jinja": False,              # --jinja 启用工具调用模板（Agent 模式需要）
    # 采样参数（Qwen 官方推荐；请求未指定时注入）
    "temperature": 0.6,          # 思考模式官方推荐 0.6（非思考模式推荐 0.7）
    "top_p": 0.95,               # 思考模式官方推荐 0.95（非思考模式推荐 0.8）
    "top_k": 20,                 # 官方推荐 20
    "extra_args": "",            # 额外命令行参数
}


def _deep_copy(d):
    return json.loads(json.dumps(d))


class Config:
    def __init__(self):
        self._lock = threading.RLock()
        self.global_ = _deep_copy(DEFAULT_GLOBAL)
        self.models = []  # list[dict]
        self.load()

    # ---------- 持久化 ----------
    def load(self):
        with self._lock:
            if CONFIG_FILE.exists():
                try:
                    data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                    g = dict(DEFAULT_GLOBAL)
                    g.update({k: v for k, v in data.get("global", {}).items() if k in g})
                    self.global_ = g
                    models = []
                    for m in data.get("models", []):
                        mm = _deep_copy(DEFAULT_MODEL)
                        mm.update({k: v for k, v in m.items() if k in mm})
                        if mm.get("id") and mm.get("model_path"):
                            models.append(mm)
                    self.models = models
                except Exception as e:
                    print(f"[配置] 读取失败，使用默认配置: {e}")
            else:
                self.save()

    def save(self):
        with self._lock:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            CONFIG_FILE.write_text(
                json.dumps({"global": self.global_, "models": self.models},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    # ---------- 模型 ----------
    def get_model(self, model_id: str) -> Optional[dict]:
        with self._lock:
            for m in self.models:
                if m["id"] == model_id:
                    return m
            return None

    def upsert_model(self, model: dict):
        with self._lock:
            model = {**_deep_copy(DEFAULT_MODEL), **model}
            for i, m in enumerate(self.models):
                if m["id"] == model["id"]:
                    self.models[i] = model
                    break
            else:
                self.models.append(model)
            if not self.global_.get("default_model"):
                self.global_["default_model"] = model["id"]
            self.save()

    def remove_model(self, model_id: str):
        with self._lock:
            self.models = [m for m in self.models if m["id"] != model_id]
            if self.global_.get("default_model") == model_id:
                self.global_["default_model"] = self.models[0]["id"] if self.models else ""
            self.save()

    # ---------- llama-server 可执行文件探测 ----------
    def find_llama_server(self) -> Optional[str]:
        with self._lock:
            # 1. 配置中显式指定
            p = self.global_.get("llama_server_path", "")
            if p and Path(p).is_file():
                return str(Path(p))
        # 2. 环境变量
        p = os.environ.get("LLAMA_SERVER_PATH", "")
        if p and Path(p).is_file():
            return p
        # 3. 项目根 / 相邻目录下的发行包
        exe_name = "llama-server.exe" if os.name == "nt" else "llama-server"
        candidates = [ROOT_DIR / exe_name]
        # 相邻的 llama.cpp 发行目录，如 llama-bxxxx-bin-win-cuda-x-x64
        for parent in [ROOT_DIR.parent, ROOT_DIR]:
            for d in sorted(parent.glob("llama-b*")):
                if d.is_dir():
                    candidates.append(d / exe_name)
        for c in candidates:
            if c.is_file():
                return str(c)
        # 4. PATH
        return shutil.which("llama-server")


def resolve_model_id(cfg: Config, requested: str) -> Optional[str]:
    """把请求中的 model 字段解析为已配置的模型 ID。"""
    if requested and cfg.get_model(requested):
        return requested
    default = cfg.global_.get("default_model", "")
    if default and cfg.get_model(default):
        return default
    if cfg.models:
        return cfg.models[0]["id"]
    return None
