"""llama-server 进程管理：启动 / 停止 / 健康检查 / 按需切换模型

资源节约策略：
- 同一时刻只运行一个 llama-server 实例
- 切换模型 = 终止旧进程再启动新进程（显存/内存不会叠加）
- 使用 --no-webui 关闭内置 Web 界面
"""

import os
import shlex
import subprocess
import threading
import time
from typing import Callable, Optional

import requests

from .config import BACKEND_LOG, Config

# 大模型从磁盘加载可能需要几分钟
STARTUP_TIMEOUT = 600
HEALTH_INTERVAL = 1.0


class ModelManager:
    def __init__(self, cfg: Config, log_cb: Optional[Callable[[str], None]] = None):
        self.cfg = cfg
        self.log_cb = log_cb or (lambda msg: print(msg, flush=True))
        self._lock = threading.RLock()
        self.proc: Optional[subprocess.Popen] = None
        self.active_model: Optional[str] = None
        self._log_file = None
        self._reader_thread: Optional[threading.Thread] = None

    # ---------- 日志 ----------
    def log(self, msg: str):
        self.log_cb(msg)

    def _pump_logs(self):
        """后台线程：把 llama-server 的 stdout/stderr 写入日志文件并回调"""
        try:
            BACKEND_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(BACKEND_LOG, "a", encoding="utf-8", errors="replace") as f:
                f.write(f"\n===== llama-server 启动 {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
                stream = self.proc.stdout
                if stream is None:
                    return
                for raw in iter(stream.readline, b""):
                    if self.proc.poll() is not None:
                        break
                    try:
                        line = raw.decode("utf-8", errors="replace").rstrip()
                    except Exception:
                        continue
                    f.write(line + "\n")
                    f.flush()
        except Exception:
            pass

    # ---------- 命令行 ----------
    def _build_cmd(self, model: dict) -> list:
        exe = self.cfg.find_llama_server()
        if not exe:
            raise FileNotFoundError(
                "未找到 llama-server 可执行文件。请在配置中设置 llama_server_path，"
                "或将 llama.cpp 发行包目录放在本项目同级/项目内。")
        port = self.cfg.global_["backend_port"]
        cmd = [
            exe,
            "-m", model["model_path"],
            "--host", "127.0.0.1",
            "--port", str(port),
            "-c", str(int(model.get("ctx_size", 8192))),
            "-ngl", str(int(model.get("ngl", 999))),
            "--no-webui",
        ]
        threads = int(model.get("threads", -1) or -1)
        if threads > 0:
            cmd += ["-t", str(threads)]
        fa = model.get("flash_attn", "auto")
        if fa in ("on", "off", "auto"):
            cmd += ["-fa", fa]
        if model.get("mmproj_path"):
            cmd += ["--mmproj", model["mmproj_path"]]
        # 思考模式控制（Qwen/llama.cpp 官方语义）
        # off: --reasoning-budget 0 硬关闭；budget: --reasoning-budget N 限制思考 token
        reasoning = model.get("reasoning", "auto")
        if reasoning == "off":
            cmd += ["--reasoning-budget", "0"]
        elif reasoning == "budget":
            budget = int(model.get("reasoning_budget", 4096) or 4096)
            if budget > 0:
                cmd += ["--reasoning-budget", str(budget)]
        if model.get("jinja"):
            cmd += ["--jinja"]
        extra = (model.get("extra_args") or "").strip()
        if extra:
            cmd += shlex.split(extra, posix=(os.name != "nt"))
        return cmd

    # ---------- 健康检查 ----------
    def backend_healthy(self, timeout: float = 2.0) -> bool:
        try:
            r = requests.get(
                f"http://127.0.0.1:{self.cfg.global_['backend_port']}/health",
                timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    def backend_props(self) -> dict | None:
        """读取 llama-server /props（含 model_path 等），用于识别已运行实例加载的模型"""
        try:
            r = requests.get(
                f"http://127.0.0.1:{self.cfg.global_['backend_port']}/props",
                timeout=3)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    def backend_model_matches(self, model_path: str) -> bool:
        """判断端口上已运行的 llama-server 是否加载了指定模型"""
        props = self.backend_props()
        if not props:
            return False
        import json as _json
        blob = _json.dumps(props).lower()
        return model_path.lower().replace("\\", "/") in blob.replace("\\", "/")

    def _kill_orphan_backend(self):
        """尽力终止占用内部端口的孤儿 llama-server（非本管理器启动的实例）"""
        port = self.cfg.global_["backend_port"]
        try:
            if os.name == "nt":
                out = subprocess.check_output(["netstat", "-ano"], text=True)
                pids = {ln.split()[-1] for ln in out.splitlines()
                        if f"127.0.0.1:{port}" in ln and "LISTENING" in ln}
                for pid in pids:
                    subprocess.run(["taskkill", "/F", "/PID", pid],
                                   capture_output=True)
                    self.log(f"[后端] 已终止占用端口 {port} 的进程 PID {pid}")
            else:
                out = subprocess.check_output(["ss", "-tlnp"], text=True,
                                              stderr=subprocess.DEVNULL)
                for ln in out.splitlines():
                    if f":{port}" in ln:
                        for tok in ln.split():
                            if tok.startswith("pid="):
                                subprocess.run(["kill", tok[4:]], capture_output=True)
                                self.log(f"[后端] 已终止占用端口 {port} 的进程")
            time.sleep(1)
        except Exception as e:
            self.log(f"[后端] 清理端口占用失败: {e}")

    def _wait_healthy(self) -> bool:
        deadline = time.time() + STARTUP_TIMEOUT
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                self.log(f"[后端] llama-server 进程异常退出 (code={self.proc.returncode})，详见 {BACKEND_LOG}")
                return False
            if self.backend_healthy():
                return True
            time.sleep(HEALTH_INTERVAL)
        self.log("[后端] 等待模型加载超时")
        return False

    # ---------- 生命周期 ----------
    def start(self, model_id: str) -> bool:
        """启动指定模型（若已有实例在跑其他模型，先停掉）"""
        with self._lock:
            model = self.cfg.get_model(model_id)
            if not model:
                self.log(f"[后端] 未找到模型配置: {model_id}")
                return False
            if (self.active_model == model_id and self.proc is not None
                    and self.proc.poll() is None and self.backend_healthy()):
                return True
            self.stop()
            # 处理非本管理器启动的实例（上次异常退出残留的孤儿 llama-server）
            if self.backend_healthy() and (self.proc is None
                                           or self.proc.poll() is not None):
                if self.backend_model_matches(model["model_path"]):
                    self.log(f"[后端] 检测到已运行的 llama-server 已加载 {model_id}，直接复用")
                    self.active_model = model_id
                    return True
                self.log("[后端] 内部端口被残留的 llama-server 实例占用（模型不同），正在自动清理...")
                self._kill_orphan_backend()
            try:
                cmd = self._build_cmd(model)
            except FileNotFoundError as e:
                self.log(f"[后端] {e}")
                return False
            self.log(f"[后端] 启动模型 {model_id} ...")
            self.log("[后端] " + " ".join(str(c) for c in cmd))
            creation = 0
            if os.name == "nt":
                creation = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, creationflags=creation)
            self.active_model = model_id
            self._reader_thread = threading.Thread(target=self._pump_logs, daemon=True)
            self._reader_thread.start()
            ok = self._wait_healthy()
            if ok:
                self.log(f"[后端] 模型 {model_id} 已就绪 (127.0.0.1:{self.cfg.global_['backend_port']})")
            else:
                self.stop()
            return ok

    def stop(self):
        with self._lock:
            proc, self.proc = self.proc, None
            self.active_model = None
            if proc is not None and proc.poll() is None:
                self.log("[后端] 停止 llama-server ...")
                try:
                    proc.terminate()
                    proc.wait(timeout=8)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                self.log("[后端] 已停止")

    def ensure(self, model_id: str) -> bool:
        """确保指定模型在运行；auto_load 关闭时不做任何事"""
        if not self.cfg.global_.get("auto_load", True):
            return self.active_model == model_id and self.backend_healthy()
        with self._lock:
            if (self.active_model == model_id and self.proc is not None
                    and self.proc.poll() is None and self.backend_healthy()):
                return True
        return self.start(model_id)

    def status(self) -> dict:
        running = self.proc is not None and self.proc.poll() is None
        return {
            "running": running,
            "healthy": self.backend_healthy() if running else False,
            "active_model": self.active_model,
            "backend_port": self.cfg.global_["backend_port"],
        }
