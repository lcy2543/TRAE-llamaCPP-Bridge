"""TRAE-llamaCPP-Bridge 命令行入口（Linux / Windows 通用）

用法示例：
  python cli_app.py add qwen35 -m /models/qwen.gguf --mmproj /models/mmproj.gguf -c 8192
  python cli_app.py list
  python cli_app.py serve                # 前台运行代理（可先 --model 预加载）
  python cli_app.py start qwen35         # 单独加载模型（不启动代理）
  python cli_app.py stop / status / remove / set
"""

import argparse
import signal
import sys
import time

from llamabridge.config import Config
from llamabridge.manager import ModelManager
from llamabridge.server import BridgeServer


def build_parser():
    p = argparse.ArgumentParser(
        prog="cli_app.py",
        description="TRAE-llamaCPP-Bridge：让 TRAE 通过 llama.cpp 使用本地 GGUF 模型")
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("add", help="添加或更新模型配置")
    sp.add_argument("id", help="模型 ID（TRAE 中填写）")
    sp.add_argument("-m", "--model", required=True, help="LLM GGUF 文件路径")
    sp.add_argument("--mmproj", default="", help="多模态 mmproj GGUF 路径（可选）")
    sp.add_argument("-c", "--ctx", type=int, default=8192, help="上下文长度，默认 8192")
    sp.add_argument("-ngl", "--ngl", type=int, default=999, help="GPU 卸载层数，默认 999（全部）")
    sp.add_argument("-t", "--threads", type=int, default=-1, help="CPU 线程数，-1 自动")
    sp.add_argument("-fa", "--flash-attn", default="auto", choices=["on", "off", "auto"])
    sp.add_argument("--no-think", action="store_true",
                    help="禁用思考模式（--reasoning-budget 0，节省 token）")
    sp.add_argument("--think-budget", type=int, default=0,
                    help="思考 token 上限（官方建议≥1024；0=不限制）")
    sp.add_argument("--temp", type=float, default=0.6,
                    help="temperature，Qwen 官方推荐思考模式 0.6 / 非思考 0.7")
    sp.add_argument("--top-p", type=float, default=0.95,
                    help="top_p，Qwen 官方推荐思考模式 0.95 / 非思考 0.8")
    sp.add_argument("--top-k", type=int, default=20,
                    help="top_k，Qwen 官方推荐 20")
    sp.add_argument("--jinja", action="store_true", help="启用 --jinja（工具调用模板）")
    sp.add_argument("--extra", default="", help="额外 llama-server 参数")
    sp.add_argument("--default", action="store_true", help="设为默认模型")

    sub.add_parser("list", help="列出已配置模型")

    sp = sub.add_parser("remove", help="删除模型配置")
    sp.add_argument("id")

    sp = sub.add_parser("serve", help="前台运行 OpenAI 兼容代理")
    sp.add_argument("--model", default="", help="启动时预加载的模型 ID")

    sp = sub.add_parser("start", help="加载指定模型（不启动代理）")
    sp.add_argument("id")

    sub.add_parser("stop", help="停止 llama-server")
    sub.add_parser("status", help="查看运行状态")
    sub.add_parser("trae", help="打印 TRAE 需要填写的参数（--copy 同时写入剪贴板）").add_argument(
        "--copy", action="store_true", help="把 API 地址复制到剪贴板")

    sp = sub.add_parser("set", help="修改全局设置")
    sp.add_argument("--proxy-port", type=int)
    sp.add_argument("--backend-port", type=int)
    sp.add_argument("--llama-server", help="llama-server 可执行文件路径")
    sp.add_argument("--auto-load", choices=["on", "off"])
    sp.add_argument("--strip-think", choices=["on", "off"])
    sp.add_argument("--default-model")

    return p


def cmd_add(cfg: Config, a):
    cfg.upsert_model({
        "id": a.id,
        "model_path": a.model,
        "mmproj_path": a.mmproj,
        "ctx_size": a.ctx,
        "ngl": a.ngl,
        "threads": a.threads,
        "flash_attn": a.flash_attn,
        "reasoning": ("off" if a.no_think
                      else ("budget" if a.think_budget > 0 else "auto")),
        "reasoning_budget": a.think_budget or 4096,
        "temperature": a.temp,
        "top_p": a.top_p,
        "top_k": a.top_k,
        "jinja": a.jinja,
        "extra_args": a.extra,
    })
    if a.default:
        cfg.global_["default_model"] = a.id
        cfg.save()
    print(f"已保存模型: {a.id}")


def cmd_list(cfg: Config):
    if not cfg.models:
        print("（还没有配置任何模型，用 add 命令添加）")
        return
    default = cfg.global_.get("default_model", "")
    print(f"{'*默认' if default else ''} 已配置 {len(cfg.models)} 个模型：")
    for m in cfg.models:
        tag = " [默认]" if m["id"] == default else ""
        mm = " +mmproj" if m.get("mmproj_path") else ""
        print(f"  - {m['id']}{tag}{mm}  ctx={m['ctx_size']} ngl={m['ngl']}  {m['model_path']}")


def cmd_set(cfg: Config, a):
    if a.proxy_port:
        cfg.global_["proxy_port"] = a.proxy_port
    if a.backend_port:
        cfg.global_["backend_port"] = a.backend_port
    if a.llama_server:
        cfg.global_["llama_server_path"] = a.llama_server
    if a.auto_load:
        cfg.global_["auto_load"] = a.auto_load == "on"
    if a.strip_think:
        cfg.global_["strip_think"] = a.strip_think == "on"
    if a.default_model:
        if cfg.get_model(a.default_model):
            cfg.global_["default_model"] = a.default_model
        else:
            print(f"模型不存在: {a.default_model}")
            return
    cfg.save()
    print("已保存全局设置")


def cmd_serve(cfg: Config, a):
    from llamabridge.server import get_lan_ip
    mgr = ModelManager(cfg)
    bridge = BridgeServer(cfg, mgr)
    if not bridge.start():
        sys.exit(1)
    exe = cfg.find_llama_server()
    host = get_lan_ip()
    print(f"llama-server: {exe or '未找到（请用 set --llama-server 指定）'}")
    print(f"TRAE 设置 → API 地址: http://{host}:{cfg.global_['proxy_port']}/v1")
    print(f"（新版 TRAE 不允许 127.0.0.1，请使用上面的局域网 IP）")
    for m in cfg.models:
        print(f"  模型 ID 可填: {m['id']}")

    if a.model:
        if not mgr.start(a.model):
            print("模型加载失败，退出")
            sys.exit(1)
    elif cfg.models and cfg.global_.get("auto_load", True):
        print("（auto_load=on：收到首个请求时将自动加载默认模型）")

    def _shutdown(sig, frame):
        print("\n正在退出 ...")
        mgr.stop()
        bridge.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    while True:
        time.sleep(3600)


def main():
    a = build_parser().parse_args()
    if not a.cmd:
        build_parser().print_help()
        return
    cfg = Config()

    if a.cmd == "add":
        cmd_add(cfg, a)
    elif a.cmd == "list":
        cmd_list(cfg)
    elif a.cmd == "remove":
        cfg.remove_model(a.id)
        print(f"已删除: {a.id}")
    elif a.cmd == "set":
        cmd_set(cfg, a)
    elif a.cmd == "serve":
        cmd_serve(cfg, a)
    elif a.cmd == "start":
        mgr = ModelManager(cfg)
        ok = mgr.start(a.id)
        print("加载成功" if ok else "加载失败")
        if ok:
            print(f"后端: http://127.0.0.1:{cfg.global_['backend_port']} （Ctrl+C 退出并停止模型）")
            try:
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                mgr.stop()
    elif a.cmd == "stop":
        # stop 只对同进程有效；跨进程通过杀端口占用来停止
        import subprocess
        port = cfg.global_["backend_port"]
        try:
            if sys.platform == "win32":
                out = subprocess.check_output(
                    ["netstat", "-ano"], text=True).splitlines()
                pids = {ln.split()[-1] for ln in out
                        if f"127.0.0.1:{port}" in ln and "LISTENING" in ln}
                for pid in pids:
                    subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
            else:
                out = subprocess.check_output(
                    ["ss", "-tlnp"], text=True, stderr=subprocess.DEVNULL)
                for ln in out.splitlines():
                    if f":{port}" in ln:
                        for tok in ln.split():
                            if tok.startswith("pid="):
                                subprocess.run(["kill", tok[4:]], capture_output=True)
            print("已停止 llama-server")
        except Exception as e:
            print(f"停止失败: {e}")
    elif a.cmd == "trae":
        from llamabridge.server import get_lan_ip
        port = cfg.global_["proxy_port"]
        default = cfg.global_.get("default_model") or "(未设置默认模型)"
        host = get_lan_ip()
        url = f"http://{host}:{port}/v1"
        print("在 TRAE 中：设置 → 模型 → 添加模型，填写以下参数：\n")
        print(f"  API 格式        OpenAI Chat Completions 格式")
        print(f"  自定义请求地址  {url}")
        print(f"  模型 ID         {default}")
        print(f"  API 密钥        sk-local (任意)")
        print(f"\n  注意：新版 TRAE 不允许 127.0.0.1/localhost，请使用上面的局域网 IP")
        print(f"  若开启「完整 URL」开关，请求地址改为: {url}/chat/completions")
        if getattr(a, "copy", False):
            try:
                import subprocess
                if sys.platform == "win32":
                    subprocess.run(["clip"], input=url.encode(), check=True)
                elif sys.platform == "darwin":
                    subprocess.run(["pbcopy"], input=url.encode(), check=True)
                else:
                    subprocess.run(["xclip", "-selection", "clipboard"],
                                   input=url.encode(), check=True)
                print(f"\n已复制到剪贴板: {url}")
            except Exception as e:
                print(f"\n复制失败（{e}），请手动复制上面的地址")
    elif a.cmd == "status":
        import requests
        port = cfg.global_["proxy_port"]
        try:
            r = requests.get(f"http://127.0.0.1:{port}/bridge/status", timeout=3)
            st = r.json()
            print(f"代理: 运行中 (端口 {port})")
            print(f"后端: {'运行中' if st['healthy'] else '未就绪'}  "
                  f"活动模型: {st['active_model'] or '无'}")
            print(f"llama-server: {st['llama_server'] or '未找到'}")
        except Exception:
            print(f"代理: 未运行 (端口 {port})")


if __name__ == "__main__":
    main()
