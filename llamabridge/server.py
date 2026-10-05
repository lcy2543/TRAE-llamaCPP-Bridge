"""OpenAI 兼容代理服务器（固定端口），把请求转发给 llama-server。

- GET  /v1/models           列出已配置模型
- POST /v1/chat/completions 聊天补全（支持 SSE 流式透传、多模态图片透传）
- GET  /health              桥接自身健康检查
- 其余路径原样转发给 llama-server（如 /v1/embeddings、/completion）

收到请求时若目标模型未加载，会自动按需加载（懒加载）。
"""

import json
import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import requests

from .config import Config, resolve_model_id
from .manager import ModelManager


def get_lan_ip() -> str:
    """获取本机局域网 IP（不发真实流量，仅用于选路）"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# 思考标签（用拼接方式构造，避免源码中出现特殊标签字面量）
THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "<" + "/think" + ">"
THINK_RE = re.compile(
    re.escape(THINK_OPEN) + r".*?" + re.escape(THINK_CLOSE)
    + r"|" + re.escape(THINK_OPEN) + r".*$",
    re.DOTALL,
)

UPSTREAM_TIMEOUT = (10, 600)  # (连接, 读取) 秒；读取超时给长生成留足时间


class BridgeServer:
    def __init__(self, cfg: Config, manager: ModelManager, log_cb=None):
        self.cfg = cfg
        self.manager = manager
        self.log = log_cb or (lambda msg: print(msg, flush=True))
        self.httpd: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    # ---------- 生命周期 ----------
    def start(self) -> bool:
        if self.httpd is not None:
            return True
        port = self.cfg.global_["proxy_port"]

        class Handler(BridgeHandler):
            bridge = self

        # 监听 0.0.0.0：新版 TRAE 拒绝 127.0.0.1 回环地址，
        # 需用本机局域网 IP（如 192.168.x.x）访问
        try:
            self.httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        except OSError as e:
            self.log(f"[代理] 端口 {port} 启动失败: {e}")
            self.httpd = None
            return False
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        lan = get_lan_ip()
        self.log(f"[代理] OpenAI 兼容接口已就绪: http://{lan}:{port}/v1")
        if lan != "127.0.0.1":
            self.log(f"[代理] TRAE 请填写局域网地址（新版 TRAE 不允许 127.0.0.1）: "
                     f"http://{lan}:{port}/v1")
        return True

    def stop(self):
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
            self.log("[代理] 已停止")

    @property
    def running(self) -> bool:
        return self.httpd is not None


class BridgeHandler(BaseHTTPRequestHandler):
    bridge: "BridgeServer"  # 由子类注入
    protocol_version = "HTTP/1.1"

    # ---------- 工具 ----------
    def log_message(self, fmt, *args):  # 静默默认访问日志
        pass

    def _send_json(self, code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _backend_url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.bridge.cfg.global_['backend_port']}{path}"

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length > 0 else b""

    def _strip_think(self, text: str) -> str:
        return THINK_RE.sub("", text)

    # ---------- 路由 ----------
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/health":
            self._send_json(200, {"ok": True})
        elif path == "/v1/models":
            data = [{"id": m["id"], "object": "model", "owned_by": "llama.cpp"}
                    for m in self.bridge.cfg.models]
            self._send_json(200, {"object": "list", "data": data})
        elif path == "/bridge/status":
            st = self.bridge.manager.status()
            st["proxy_port"] = self.bridge.cfg.global_["proxy_port"]
            st["llama_server"] = self.bridge.cfg.find_llama_server()
            self._send_json(200, st)
        else:
            self._forward("GET")

    def do_POST(self):
        path = self.path.split("?")[0]
        if path in ("/v1/chat/completions", "/v1/responses"):
            self._handle_chat()
        else:
            self._forward("POST")

    def do_DELETE(self):
        self._forward("DELETE")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    # ---------- 核心逻辑 ----------
    def _ensure_model(self, body: dict) -> Optional[str]:
        """根据请求中的 model 字段确保后端已加载对应模型，返回模型 ID"""
        cfg = self.bridge.cfg
        requested = str((body or {}).get("model") or "") if isinstance(body, dict) else ""
        model_id = resolve_model_id(cfg, requested)
        if not model_id:
            self._send_json(503, {"error": {
                "message": "没有已配置的模型，请先在管理界面添加模型",
                "type": "no_model"}})
            return None
        if requested and requested != model_id:
            self.bridge.log(f"[代理] 未知模型 ID '{requested}'，使用 '{model_id}'")
        if not self.bridge.manager.ensure(model_id):
            self._send_json(502, {"error": {
                "message": f"模型 {model_id} 加载失败，请查看日志",
                "type": "backend_error"}})
            return None
        if isinstance(body, dict):
            body["model"] = model_id
        return model_id

    def _inject_sampling(self, body: dict):
        """客户端（TRAE 等）未指定采样参数时，注入模型配置的官方推荐值。

        Qwen 官方建议：思考模式 temperature=0.6 / top_p=0.95 / top_k=20；
        非思考模式 temperature=0.7 / top_p=0.8。禁止贪心解码，否则可能无限重复。
        """
        try:
            model = self.bridge.cfg.get_model(str(body.get("model") or ""))
            if not model:
                return
            for key in ("temperature", "top_p", "top_k"):
                val = model.get(key)
                if val is not None and key not in body:
                    body[key] = val
        except Exception:
            pass

    def _handle_chat(self):
        raw = self._read_body()
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": {"message": "请求体不是合法 JSON",
                                            "type": "invalid_request_error"}})
            return
        if not self._ensure_model(body):
            return
        self._inject_sampling(body)
        strip = bool(self.bridge.cfg.global_.get("strip_think", True))
        stream = bool(body.get("stream"))
        try:
            upstream = requests.post(
                self._backend_url(self.path.split("?")[0]),
                json=body, stream=True, timeout=UPSTREAM_TIMEOUT)
        except requests.RequestException as e:
            self._send_json(502, {"error": {"message": f"连接 llama-server 失败: {e}",
                                            "type": "backend_error"}})
            return
        if upstream.status_code != 200:
            self._relay_error(upstream)
            return
        if stream:
            self._relay_sse(upstream, strip)
        else:
            self._relay_json(upstream, strip)

    def _relay_error(self, upstream):
        try:
            payload = upstream.content
            self.send_response(upstream.status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception:
            pass
        finally:
            upstream.close()

    def _relay_json(self, upstream, strip: bool):
        """非流式：直接转发，可选剥离思考标签"""
        try:
            data = upstream.json()
            if strip:
                try:
                    for ch in data.get("choices", []):
                        msg = ch.get("message") or {}
                        if msg.get("content"):
                            msg["content"] = self._strip_think(msg["content"])
                        msg.pop("reasoning_content", None)
                except Exception:
                    pass
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception:
            self._relay_error(upstream)
        finally:
            upstream.close()

    def _relay_sse(self, upstream, strip: bool):
        """流式：SSE 透传；开启 strip 时逐块清洗思考标签"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            if not strip:
                # 纯字节透传，开销最小
                for chunk in upstream.iter_content(chunk_size=4096):
                    if chunk:
                        self._write_chunked(chunk)
            else:
                # 按 SSE 行清洗（手动缓冲，避免 chunk 边界切断行）
                in_think = False
                buf = b""
                for chunk in upstream.iter_content(chunk_size=None):
                    if not chunk:
                        continue
                    buf += chunk
                    while b"\n" in buf:
                        raw, buf = buf.split(b"\n", 1)
                        out = self._clean_sse_line(
                            raw.decode("utf-8", "replace"), in_think)
                        in_think = out[1]
                        self._write_chunked((out[0] + "\n").encode("utf-8"))
                if buf:
                    out = self._clean_sse_line(
                        buf.decode("utf-8", "replace"), in_think)
                    self._write_chunked((out[0] + "\n").encode("utf-8"))
                self._write_chunked(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")  # 结束 chunked 编码
        except (BrokenPipeError, ConnectionResetError):
            pass  # 客户端中断（如 TRAE 取消生成）
        except Exception:
            pass
        finally:
            upstream.close()

    def _clean_sse_line(self, line: str, in_think: bool):
        """清洗一行 SSE 数据，返回 (新行, 是否处于未闭合思考块)"""
        if line.startswith("data:") and line.strip() != "data: [DONE]":
            try:
                obj = json.loads(line[5:].strip())
                in_think = self._clean_chunk(obj, in_think)
                return "data: " + json.dumps(obj, ensure_ascii=False), in_think
            except Exception:
                return line, in_think
        return line, in_think

    def _clean_chunk(self, obj: dict, in_think: bool) -> bool:
        """清洗 SSE chunk 中的思考内容，返回是否处于未闭合的思考块内"""
        try:
            for ch in obj.get("choices", []):
                delta = ch.get("delta") or {}
                if "reasoning_content" in delta:
                    delta.pop("reasoning_content", None)
                text = delta.get("content")
                if not text:
                    continue
                if in_think:
                    idx = text.find(THINK_CLOSE)
                    if idx >= 0:
                        in_think = False
                        delta["content"] = text[idx + len(THINK_CLOSE):]
                    else:
                        delta["content"] = ""
                else:
                    idx = text.find(THINK_OPEN)
                    if idx >= 0:
                        end = text.find(THINK_CLOSE)
                        if end >= 0:
                            delta["content"] = text[:idx] + text[end + len(THINK_CLOSE):]
                        else:
                            in_think = True
                            delta["content"] = text[:idx]
                delta.pop("reasoning_content", None)
        except Exception:
            pass
        return in_think

    def _write_chunked(self, data: bytes):
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    # ---------- 通用转发 ----------
    def _forward(self, method: str):
        """其余请求原样转发给 llama-server"""
        body = self._read_body() if method in ("POST", "DELETE") else None
        if not self.bridge.manager.backend_healthy():
            self._send_json(503, {"error": {"message": "后端未启动",
                                            "type": "backend_unavailable"}})
            return
        try:
            upstream = requests.request(
                method, self._backend_url(self.path), data=body,
                stream=True, timeout=UPSTREAM_TIMEOUT,
                headers={"Content-Type": self.headers.get("Content-Type",
                                                          "application/json")})
        except requests.RequestException as e:
            self._send_json(502, {"error": {"message": str(e), "type": "backend_error"}})
            return
        try:
            self.send_response(upstream.status_code)
            self.send_header("Content-Type",
                             upstream.headers.get("Content-Type", "application/json"))
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for chunk in upstream.iter_content(chunk_size=8192):
                if chunk:
                    self._write_chunked(chunk)
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            pass
        finally:
            upstream.close()
