"""TRAE-llamaCPP-Bridge Windows 图形界面（tkinter，无额外依赖）

双击 Start-Win.bat 或运行 `python gui_app.py` 启动。
"""

import queue
import subprocess
import threading
import webbrowser
import os
import sys

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llamabridge.config import Config
from llamabridge.manager import ModelManager
from llamabridge.server import BridgeServer, get_lan_ip


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("TRAE-llamaCPP-Bridge 本地模型桥接")
        root.geometry("880x640")
        root.minsize(760, 520)

        self.cfg = Config()
        self.logq: "queue.Queue[str]" = queue.Queue()
        self.mgr = ModelManager(self.cfg, log_cb=self.logq.put)
        self.bridge = BridgeServer(self.cfg, self.mgr, log_cb=self.logq.put)

        self._build_menu()
        self._build_toolbar()
        self._build_main()
        self._build_statusbar()
        self._poll_log()

        # 关键：启动时把已保存的模型配置加载进列表，否则界面显示为空
        self.reload_models()
        n = len(self.cfg.models)
        self.log(f"[配置] 已加载 {n} 个已保存的模型"
                 + (f"，默认模型: {self.cfg.global_.get('default_model')}"
                    if self.cfg.global_.get("default_model") else ""))

        # 启动时自动开启代理
        if self.bridge.start():
            self.refresh_status()

    # ---------------- UI 构建 ----------------
    def _build_menu(self):
        bar = tk.Menu(self.root)
        m = tk.Menu(bar, tearoff=0)
        m.add_command(label="添加模型…", command=self.add_model)
        m.add_command(label="编辑模型…", command=self.edit_model)
        m.add_command(label="删除模型", command=self.remove_model)
        m.add_separator()
        m.add_command(label="设为默认模型", command=self.set_default_model)
        m.add_separator()
        m.add_command(label="全局设置…", command=self.open_settings)
        m.add_separator()
        m.add_command(label="退出", command=self.on_close)
        bar.add_cascade(label="模型", menu=m)
        h = tk.Menu(bar, tearoff=0)
        h.add_command(label="TRAE 配置说明", command=self.show_help)
        h.add_command(label="打开项目 README", command=lambda: webbrowser.open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")))
        bar.add_cascade(label="帮助", menu=h)
        self.root.config(menu=bar)

    def _build_toolbar(self):
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(fill="x")
        self.btn_start = ttk.Button(bar, text="启动 / 切换模型", command=self.start_model)
        self.btn_start.pack(side="left")
        ttk.Button(bar, text="停止模型", command=self.stop_model).pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="TRAE 参数(一键复制)", command=self.show_help).pack(
            side="left", padx=(8, 0))
        ttk.Button(bar, text="添加模型", command=self.add_model).pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="编辑", command=self.edit_model).pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="删除", command=self.remove_model).pack(side="left", padx=(8, 0))
        self.btn_start.state(["disabled"])

    def _build_main(self):
        paned = ttk.Panedwindow(self.root, orient="vertical")
        paned.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        # 上：模型列表
        top = ttk.Frame(paned)
        paned.add(top, weight=1)
        cols = ("id", "mmproj", "ctx", "ngl", "default")
        self.tree = ttk.Treeview(top, columns=cols, show="headings", height=7)
        self.tree.heading("id", text="模型 ID（TRAE 中填写）")
        self.tree.heading("mmproj", text="多模态")
        self.tree.heading("ctx", text="上下文")
        self.tree.heading("ngl", text="GPU层")
        self.tree.heading("default", text="默认")
        self.tree.column("id", width=280)
        self.tree.column("mmproj", width=70, anchor="center")
        self.tree.column("ctx", width=80, anchor="center")
        self.tree.column("ngl", width=70, anchor="center")
        self.tree.column("default", width=60, anchor="center")
        sb = ttk.Scrollbar(top, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tree.bind("<Double-1>", lambda e: self.edit_model())
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # 下：日志
        bottom = ttk.LabelFrame(paned, text="运行日志（含 llama-server 输出，完整日志在 data/llama-server.log）")
        paned.add(bottom, weight=1)
        self.txt = tk.Text(bottom, height=10, wrap="none", state="disabled",
                           font=("Consolas", 9))
        sb2 = ttk.Scrollbar(bottom, orient="vertical", command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb2.set)
        self.txt.pack(side="left", fill="both", expand=True)
        sb2.pack(side="right", fill="y")

    def _build_statusbar(self):
        bar = ttk.Frame(self.root, padding=(8, 4))
        bar.pack(fill="x")
        self.var_status = tk.StringVar(value="就绪")
        ttk.Label(bar, textvariable=self.var_status).pack(side="left")
        ttk.Label(bar, foreground="#666",
                  text="TRAE → 自定义模型 → API 格式: OpenAI  ← 详细说明见「帮助」菜单"
                  ).pack(side="right")

    # ---------------- 逻辑 ----------------
    def log(self, msg: str):
        self.logq.put(msg)

    def _poll_log(self):
        try:
            while True:
                msg = self.logq.get_nowait()
                self.txt.configure(state="normal")
                self.txt.insert("end", msg + "\n")
                self.txt.see("end")
                self.txt.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(300, self._poll_log)

    def reload_models(self):
        self.tree.delete(*self.tree.get_children())
        default = self.cfg.global_.get("default_model", "")
        for m in self.cfg.models:
            self.tree.insert("", "end", iid=m["id"], values=(
                m["id"],
                "mmproj" if m.get("mmproj_path") else "-",
                m.get("ctx_size", 8192),
                m.get("ngl", 999),
                "是" if m["id"] == default else "",
            ))
        self.refresh_status()

    def _on_select(self, _e=None):
        if self.tree.selection():
            self.btn_start.state(["!disabled"])

    def selected_id(self):
        sel = self.tree.selection()
        return sel[0] if sel else None

    def refresh_status(self):
        st = self.mgr.status()
        port = self.cfg.global_["proxy_port"]
        proxy = "运行中" if self.bridge.running else "已停止"
        if st["running"] and st["healthy"]:
            backend = f"模型已加载: {st['active_model']}"
        elif st["running"]:
            backend = "模型加载中…"
        else:
            backend = "后端未运行"
        self.var_status.set(
            f"代理端口 {port}: {proxy}   |   {backend}   |   "
            f"TRAE API 地址: http://127.0.0.1:{port}/v1")
        # 只启动一个周期刷新循环，避免 reload_models/启动流程重复调度
        if not getattr(self, "_status_loop", False):
            self._status_loop = True
            self.root.after(2000, self._status_tick)

    def _status_tick(self):
        if not self.root.winfo_exists():
            return
        st = self.mgr.status()
        port = self.cfg.global_["proxy_port"]
        proxy = "运行中" if self.bridge.running else "已停止"
        if st["running"] and st["healthy"]:
            backend = f"模型已加载: {st['active_model']}"
        elif st["running"]:
            backend = "模型加载中…"
        else:
            backend = "后端未运行"
        self.var_status.set(
            f"代理端口 {port}: {proxy}   |   {backend}   |   "
            f"TRAE API 地址: http://{get_lan_ip()}:{port}/v1")
        self.root.after(2000, self._status_tick)

    def start_model(self):
        mid = self.selected_id()
        if not mid:
            messagebox.showinfo("提示", "请先在列表中选择一个模型")
            return
        threading.Thread(target=self._start_bg, args=(mid,), daemon=True).start()

    def _start_bg(self, mid):
        self.mgr.start(mid)

    def stop_model(self):
        threading.Thread(target=self.mgr.stop, daemon=True).start()

    def add_model(self):
        dlg = ModelDialog(self.root, self.cfg, None)
        if dlg.result:
            self.cfg.upsert_model(dlg.result)
            self.reload_models()

    def edit_model(self):
        mid = self.selected_id()
        if not mid:
            return
        m = self.cfg.get_model(mid)
        dlg = ModelDialog(self.root, self.cfg, m)
        if dlg.result:
            if dlg.result["id"] != mid:
                self.cfg.remove_model(mid)
            self.cfg.upsert_model(dlg.result)
            self.reload_models()

    def remove_model(self):
        mid = self.selected_id()
        if not mid:
            return
        if messagebox.askyesno("确认", f"删除模型配置 {mid}？"):
            self.cfg.remove_model(mid)
            self.reload_models()

    def set_default_model(self):
        mid = self.selected_id()
        if not mid:
            return
        self.cfg.global_["default_model"] = mid
        self.cfg.save()
        self.reload_models()

    def open_settings(self):
        dlg = SettingsDialog(self.root, self.cfg)
        if dlg.changed:
            self.log("[配置] 全局设置已更新（代理端口重启程序后生效）")

    def show_help(self):
        TraeConfigDialog(self.root, self.cfg)

    def on_close(self):
        if messagebox.askyesno("退出", "退出程序并停止本地模型？"):
            self.mgr.stop()
            self.bridge.stop()
            self.root.destroy()


class TraeConfigDialog(tk.Toplevel):
    """TRAE 配置速查对话框：每项参数一键复制到剪贴板"""

    def __init__(self, parent, cfg: Config):
        super().__init__(parent)
        self.title("TRAE 配置参数（点击「复制」即可粘贴到 TRAE）")
        self.resizable(False, False)
        self.grab_set()
        port = cfg.global_["proxy_port"]
        default = cfg.global_.get("default_model") or ""
        host = get_lan_ip()

        frm = ttk.Frame(self, padding=14)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="TRAE 中：设置 → 模型 → 添加模型，然后逐项复制粘贴：",
                  font=("", 10, "bold")).grid(row=0, column=0, columnspan=3,
                                              sticky="w", pady=(0, 10))

        rows = [
            ("API 格式", "OpenAI Chat Completions 格式", True),
            ("自定义请求地址", f"http://{host}:{port}/v1", False),
            ("（完整URL开启时用）", f"http://{host}:{port}/v1/chat/completions", False),
            ("模型 ID", default or "（请先在本程序添加模型）", False),
            ("API 密钥", "sk-local", False),
        ]
        self._values = []
        for i, (label, value, selectable) in enumerate(rows, start=1):
            ttk.Label(frm, text=label + "：").grid(row=i, column=0, sticky="w", pady=4)
            var = tk.StringVar(value=value)
            self._values.append(var)
            if selectable:
                ttk.Label(frm, textvariable=var, foreground="#0a6").grid(
                    row=i, column=1, sticky="w", padx=(0, 12))
            else:
                ent = ttk.Entry(frm, textvariable=var, width=48)
                ent.grid(row=i, column=1, sticky="we", padx=(0, 12))
                if not value:
                    ent.state(["disabled"])
            ttk.Button(frm, text="复制", width=6,
                       command=lambda v=var: self._copy(v)).grid(
                row=i, column=2, pady=2)

        ttk.Separator(frm).grid(row=6, column=0, columnspan=3, sticky="we", pady=10)
        ttk.Button(frm, text="一键复制全部（含换行，逐行粘贴）",
                   command=self._copy_all).grid(row=7, column=0, columnspan=3,
                                                sticky="we", pady=(0, 6))
        ttk.Label(frm, foreground="#c33", justify="left", text=(
            "注意：新版 TRAE 不允许 127.0.0.1/localhost，请使用上面自动填写的\n"
            "本机局域网 IP 地址（代理已监听全部网卡）。\n"
            "若 IP 变化（如切换 Wi-Fi），请重新打开本对话框获取新地址。")).grid(
            row=8, column=0, columnspan=3, sticky="w")
        ttk.Label(frm, foreground="#666", justify="left", text=(
            "说明：「完整 URL」开关保持关闭即可；模型展示名称随意；\n"
            "点击 TRAE「添加模型」会做一次连通性测试（消耗少量 token）。\n"
            "切换本地模型只需在本程序操作，TRAE 无需改动。")).grid(
            row=9, column=0, columnspan=3, sticky="w", pady=(6, 0))

        self.bind("<Escape>", lambda e: self.destroy())

    def _copy(self, var: tk.StringVar):
        value = var.get()
        if not value or value.startswith("（"):
            return
        self.clipboard_clear()
        self.clipboard_append(value)
        self.update()  # 确保剪贴板在窗口关闭后仍可用
        self.title("TRAE 配置参数 —— 已复制到剪贴板 ✓")
        self.after(1500, lambda: self.title(
            "TRAE 配置参数（点击「复制」即可粘贴到 TRAE）"))

    def _copy_all(self):
        lines = [v.get() for v in self._values if v.get()]
        self.clipboard_clear()
        self.clipboard_append("\n".join(lines))
        self.update()
        self.title("TRAE 配置参数 —— 全部参数已复制 ✓")
        self.after(1500, lambda: self.title(
            "TRAE 配置参数（点击「复制」即可粘贴到 TRAE）"))


class ModelDialog(tk.Toplevel):
    """模型配置对话框（新增 / 编辑）"""

    def __init__(self, parent, cfg: Config, model: dict | None):
        super().__init__(parent)
        self.result = None
        self.cfg = cfg
        self.title("编辑模型" if model else "添加模型")
        self.resizable(False, False)
        self.grab_set()
        m = model or {}

        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        def row(label, r):
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", pady=3, padx=(0, 8))
            return r

        r = 0
        ttk.Label(frm, text="模型 ID *").grid(row=r, column=0, sticky="w", pady=3)
        self.ent_id = ttk.Entry(frm, width=52)
        self.ent_id.insert(0, m.get("id", ""))
        self.ent_id.grid(row=r, column=1, columnspan=2, sticky="we", pady=3)

        r += 1
        ttk.Label(frm, text="LLM 模型 (GGUF) *").grid(row=r, column=0, sticky="w", pady=3)
        self.ent_model = ttk.Entry(frm, width=46)
        self.ent_model.insert(0, m.get("model_path", ""))
        self.ent_model.grid(row=r, column=1, sticky="we", pady=3)
        ttk.Button(frm, text="浏览…", width=8,
                   command=lambda: self._browse(self.ent_model, [("GGUF", "*.gguf")])
                   ).grid(row=r, column=2, padx=(4, 0))

        r += 1
        ttk.Label(frm, text="mmproj (多模态)").grid(row=r, column=0, sticky="w", pady=3)
        self.ent_mmproj = ttk.Entry(frm, width=46)
        self.ent_mmproj.insert(0, m.get("mmproj_path", ""))
        self.ent_mmproj.grid(row=r, column=1, sticky="we", pady=3)
        ttk.Button(frm, text="浏览…", width=8,
                   command=lambda: self._browse(self.ent_mmproj, [("GGUF mmproj", "*.gguf")])
                   ).grid(row=r, column=2, padx=(4, 0))

        r += 1
        params = ttk.Frame(frm)
        params.grid(row=r, column=0, columnspan=3, sticky="we", pady=6)
        ttk.Label(params, text="上下文长度").pack(side="left")
        self.sp_ctx = tk.IntVar(value=m.get("ctx_size", 8192))
        ttk.Spinbox(params, from_=512, to=1048576, increment=512, width=8,
                    textvariable=self.sp_ctx).pack(side="left", padx=(4, 16))
        ttk.Label(params, text="GPU 卸载层数").pack(side="left")
        self.sp_ngl = tk.IntVar(value=m.get("ngl", 999))
        ttk.Spinbox(params, from_=0, to=999, width=6,
                    textvariable=self.sp_ngl).pack(side="left", padx=(4, 16))
        ttk.Label(params, text="CPU 线程(-1自动)").pack(side="left")
        self.sp_threads = tk.IntVar(value=m.get("threads", -1))
        ttk.Spinbox(params, from_=-1, to=256, width=6,
                    textvariable=self.sp_threads).pack(side="left", padx=4)

        r += 1
        opts = ttk.Frame(frm)
        opts.grid(row=r, column=0, columnspan=3, sticky="w", pady=3)
        self.var_fa = tk.StringVar(value=m.get("flash_attn", "auto"))
        ttk.Label(opts, text="Flash Attention:").pack(side="left")
        for v in ("auto", "on", "off"):
            ttk.Radiobutton(opts, text=v, value=v, variable=self.var_fa).pack(side="left")
        self.var_reasoning = tk.StringVar(value=m.get("reasoning", "auto"))
        ttk.Label(opts, text="  思考模式:").pack(side="left")
        for v, txt in (("auto", "模型默认"), ("off", "禁用(省token)")):
            ttk.Radiobutton(opts, text=txt, value=v,
                            variable=self.var_reasoning).pack(side="left")
        self.var_jinja = tk.BooleanVar(value=bool(m.get("jinja")))
        ttk.Checkbutton(opts, text="Jinja 工具调用 (--jinja)",
                        variable=self.var_jinja).pack(side="left", padx=(16, 0))

        r += 1
        ttk.Label(frm, text="额外参数").grid(row=r, column=0, sticky="w", pady=3)
        self.ent_extra = ttk.Entry(frm, width=52)
        self.ent_extra.insert(0, m.get("extra_args", ""))
        self.ent_extra.grid(row=r, column=1, columnspan=2, sticky="we", pady=3)

        btns = ttk.Frame(frm)
        btns.grid(row=r + 1, column=0, columnspan=3, pady=(10, 0))
        ttk.Button(btns, text="保存", command=self._save).pack(side="left", padx=4)
        ttk.Button(btns, text="取消", command=self.destroy).pack(side="left", padx=4)

        self.wait_window()

    def _browse(self, entry, ftypes):
        p = filedialog.askopenfilename(filetypes=ftypes + [("所有文件", "*.*")])
        if p:
            entry.delete(0, "end")
            entry.insert(0, p)
            if entry is self.ent_model and not self.ent_id.get():
                base = os.path.splitext(os.path.basename(p))[0]
                self.ent_id.insert(0, base)

    def _save(self):
        mid = self.ent_id.get().strip()
        path = self.ent_model.get().strip()
        if not mid or not path:
            messagebox.showwarning("提示", "模型 ID 和 GGUF 路径为必填项", parent=self)
            return
        if not os.path.isfile(path):
            if not messagebox.askyesno("提示", f"文件不存在：\n{path}\n仍要保存吗？",
                                       parent=self):
                return
        mm = self.ent_mmproj.get().strip()
        if mm and not os.path.isfile(mm):
            if not messagebox.askyesno("提示", f"mmproj 文件不存在：\n{mm}\n仍要保存吗？",
                                       parent=self):
                return
        self.result = {
            "id": mid,
            "model_path": path,
            "mmproj_path": mm,
            "ctx_size": int(self.sp_ctx.get()),
            "ngl": int(self.sp_ngl.get()),
            "threads": int(self.sp_threads.get()),
            "flash_attn": self.var_fa.get(),
            "reasoning": self.var_reasoning.get(),
            "jinja": bool(self.var_jinja.get()),
            "extra_args": self.ent_extra.get().strip(),
        }
        self.destroy()


class SettingsDialog(tk.Toplevel):
    def __init__(self, parent, cfg: Config):
        super().__init__(parent)
        self.cfg = cfg
        self.changed = False
        self.title("全局设置")
        self.resizable(False, False)
        self.grab_set()
        g = cfg.global_

        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="代理端口 (TRAE 用)").grid(row=0, column=0, sticky="w", pady=4)
        self.sp_proxy = tk.IntVar(value=g["proxy_port"])
        ttk.Spinbox(frm, from_=1024, to=65535, width=8,
                    textvariable=self.sp_proxy).grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(frm, text="llama-server 内部端口").grid(row=1, column=0, sticky="w", pady=4)
        self.sp_backend = tk.IntVar(value=g["backend_port"])
        ttk.Spinbox(frm, from_=1024, to=65535, width=8,
                    textvariable=self.sp_backend).grid(row=1, column=1, sticky="w", padx=8)

        ttk.Label(frm, text="llama-server 路径").grid(row=2, column=0, sticky="w", pady=4)
        self.ent_exe = ttk.Entry(frm, width=48)
        self.ent_exe.insert(0, g.get("llama_server_path", ""))
        self.ent_exe.grid(row=2, column=1, columnspan=2, sticky="we", padx=8)
        ttk.Button(frm, text="浏览…", width=8,
                   command=self._browse_exe).grid(row=2, column=3)

        self.var_auto = tk.BooleanVar(value=bool(g.get("auto_load", True)))
        ttk.Checkbutton(frm, text="收到请求时自动加载模型（懒加载，推荐）",
                        variable=self.var_auto).grid(row=3, column=0, columnspan=3,
                                                     sticky="w", pady=4)
        self.var_strip = tk.BooleanVar(value=bool(g.get("strip_think", True)))
        ttk.Checkbutton(frm, text="剥离回复中的思考标签（界面更干净）",
                        variable=self.var_strip).grid(row=4, column=0, columnspan=3,
                                                      sticky="w", pady=4)

        ttk.Label(frm, foreground="#666",
                  text="自动探测顺序：此处指定 > 环境变量 LLAMA_SERVER_PATH > 项目内/同级 llama-b* 目录 > PATH"
                  ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 0))

        btns = ttk.Frame(frm)
        btns.grid(row=6, column=0, columnspan=3, pady=(10, 0))
        ttk.Button(btns, text="保存", command=self._save).pack(side="left", padx=4)
        ttk.Button(btns, text="取消", command=self.destroy).pack(side="left", padx=4)
        self.wait_window()

    def _browse_exe(self):
        if sys.platform == "win32":
            p = filedialog.askopenfilename(filetypes=[("llama-server", "llama-server*.exe")])
        else:
            p = filedialog.askopenfilename(filetypes=[("llama-server", "llama-server")])
        if p:
            self.ent_exe.delete(0, "end")
            self.ent_exe.insert(0, p)

    def _save(self):
        g = self.cfg.global_
        g["proxy_port"] = int(self.sp_proxy.get())
        g["backend_port"] = int(self.sp_backend.get())
        g["llama_server_path"] = self.ent_exe.get().strip()
        g["auto_load"] = bool(self.var_auto.get())
        g["strip_think"] = bool(self.var_strip.get())
        self.cfg.save()
        self.changed = True
        self.destroy()


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except Exception:
        pass
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
