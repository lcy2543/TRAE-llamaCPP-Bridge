# TRAE-llamaCPP-Bridge

[简体中文](#简体中文) | [English](#english)

---

# 简体中文

让 TRAE（及其他 IDE）直接使用 **llama.cpp 本地部署的 GGUF 模型**（含多模态模型），绕开 IDE 不能自定义模型的限制。

> 本项目参考并引用了 [Nolan0722/TRAE-Ollama-Bridge](https://github.com/Nolan0722/TRAE-Ollama-Bridge)（MIT License）的思路 —— 将本地模型包装为 OpenAI 兼容接口供 TRAE 调用。
> 区别在于：本项目基于 **llama.cpp（llama-server）** 而非 Ollama，无需安装 Ollama，直接加载 GGUF 文件，且原生支持 **LLM + mmproj（视觉）同时加载**。

---

## 目录

- [工作原理](#工作原理)
- [资源节约设计](#资源节约设计)
- [准备工作](#一准备工作)
- [Windows 使用（图形界面）](#二windows-使用图形界面)
- [Linux 使用（命令行）](#三linux-使用命令行)
- [TRAE 设置方法](#四trae-设置方法重要)
- [目录结构](#五目录结构)
- [常见问题](#六常见问题)

## 工作原理

```
TRAE ──OpenAI 格式──> Python 代理 (127.0.0.1:8800) ──转发──> llama-server (127.0.0.1:8801)
                          │                                    │
                          │ 管理/按需加载/切换模型                └─ -m model.gguf --mmproj mmproj.gguf
                          └─ 固定端口+模型ID映射，TRAE 设置一次即可
```

- `llama-server` 本身提供 OpenAI 兼容 API 和多模态支持，Python 程序只负责**进程管理 + 轻量代理**（实测代理自身仅占约 30MB 内存）
- **懒加载**：TRAE 发来请求时才加载模型，平时不占显存
- **一键切换**：TRAE 里的设置永远不变（端口固定），在管理界面切换模型即可

## 资源节约设计

| 措施 | 说明 |
|---|---|
| 单实例运行 | 同一时刻只跑一个 llama-server，切换模型 = 先卸载再加载，显存/内存不叠加 |
| 懒加载 | `auto_load=on` 时收到首个请求才加载模型，无人使用不占资源 |
| 关闭内置 WebUI | 启动参数带 `--no-webui` |
| 上下文长度可控 | 默认 8192，可按需调小（上下文是显存大户） |
| GPU 卸载层数可控 | `-ngl` 默认 999（全部进显存），显存不足可调低 |
| Flash Attention | 默认 `auto`，可进一步省显存 |
| 禁用思考模式 | 思考型模型（Qwen3 等）可选 `禁用思考`，避免 token 浪费 |

---

## 一、准备工作

1. **Windows 图形界面（exe 版）**：无需安装任何依赖，开箱即用
   **Windows 源码运行 / Linux**：安装 **Python 3.10+**（Windows 安装时勾选 *Add to PATH*）
2. 下载 **llama.cpp** 发行包（如 `llama-bXXXX-bin-win-cuda-x.x-x64.zip`），
   解压到本项目**同级目录**或项目内即可被自动探测（也可在设置里手动指定 `llama-server` 路径）
3. 准备好 GGUF 模型文件；多模态模型请同时准备配套的 `mmproj-*.gguf`

## 二、Windows 使用（图形界面）

### 方式 A：exe 一键版（推荐）

从本仓库 **`dist/TRAE-llamaCPP-Bridge.exe`**（约 13MB）直接下载，**无需安装 Python**，已内置全部依赖，下载后双击即可运行。

- exe 与其配置（`data\` 文件夹）始终在一起：把 exe 放到哪里，`data\config.json` 与日志就在哪里生成
- llama-server 的自动探测同样有效：把 exe 与 llama.cpp 发行包（如 `llama-bXXXX-bin-win-cuda-x.x-x64`）放在同一目录即可被自动找到
- 首次运行如遇 Windows SmartScreen 提示，选择「更多信息 → 仍要运行」（PyInstaller 单文件 exe 的常见误报）
- 若你自行修改了源码，可用以下命令重新打包：

```bash
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --onefile --noconsole --name TRAE-llamaCPP-Bridge gui_app.py
```

### 方式 B：源码运行

1. 双击 **`Start-Win.bat`**（首次运行自动安装依赖）
2. 菜单栏 `模型 → 添加模型…`：
   - **模型 ID**：自定义名称，如 `qwen3.6-35b`（TRAE 中要填的就是它）
   - **LLM 模型 (GGUF)**：浏览选择主模型文件
   - **mmproj (多模态)**：浏览选择视觉投影文件（非多模态模型留空）
   - 按需调整上下文长度 / GPU 层数 / 思考模式等参数
3. 在列表中选中模型，点击 **「启动 / 切换模型」**
4. 切换模型：选中另一个模型 → 再点「启动 / 切换模型」即可，**TRAE 无需任何改动**

### 图形界面详细操作说明

| 界面元素 | 作用 |
|---|---|
| **启动 / 切换模型** 按钮 | 加载列表中选中的模型；若其他模型正在运行会先卸载再加载新模型 |
| **停止模型** 按钮 | 立即卸载当前模型，释放显存/内存 |
| **添加模型 / 编辑 / 删除** | 管理模型配置，保存在 `data\config.json` |
| **设为默认模型** | 收到的请求未匹配任何模型 ID 时，自动使用默认模型加载 |
| **全局设置** | 代理端口、内部端口、llama-server 路径、懒加载开关、思考标签剥离开关 |
| **帮助 → TRAE 配置说明** | 弹窗显示当前环境应填写的 TRAE 配置（可直接照抄） |
| 状态栏 | 实时显示代理端口、当前加载的模型、给 TRAE 填的 API 地址 |
| 日志窗口 | 显示运行日志；llama-server 的完整输出在 `data\llama-server.log` |

**模型参数详解（添加/编辑模型对话框）**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| 模型 ID | （必填） | 在 TRAE「模型 ID」一栏填写的名称，建议英文数字 |
| LLM 模型 (GGUF) | （必填） | 主模型权重文件 |
| mmproj (多模态) | 空 | 视觉编码器投影文件；填写后即支持图片输入；纯文本模型留空 |
| 上下文长度 | 8192 | 单次对话可用的 token 上限。显存不足时优先调小此值 |
| GPU 卸载层数 | 999 | 999 = 全部层进显存；显存不足时逐层下调（如 40、20），剩余层走内存 |
| CPU 线程 | -1 | -1 = 自动；纯 CPU 推理时可设为物理核心数 |
| Flash Attention | auto | auto/on 可显著降低长上下文的显存占用 |
| 思考模式 | 模型默认 | 三档：模型默认 / 限制思考预算 / 禁用。Qwen 官方建议预算 ≥1024 才有实质收益 |
| 思考预算 | 4096 | 思考 token 上限（`--reasoning-budget`），防止思考失控空耗 token |
| temperature | 0.6 | **Qwen 官方推荐**：思考模式 0.6，非思考模式 0.7。官方明确禁止贪心解码（会导致无限重复） |
| top_p | 0.95 | **Qwen 官方推荐**：思考模式 0.95，非思考模式 0.8 |
| top_k | 20 | Qwen 官方推荐值 |
| Jinja 工具调用 | 关 | 勾选后传 `--jinja`，支持 OpenAI 风格 function calling（TRAE Agent 模式需要） |
| 额外参数 | 空 | 直接追加到 llama-server 命令行，如 `--cache-type-k q8_0 --cache-type-v q8_0`（KV 缓存量化，长上下文省一半显存） |

> 采样参数（temperature/top_p/top_k）在 TRAE 等客户端**未主动指定**时自动注入请求；客户端自带参数时优先使用客户端的。
> 以上推荐值来自 [Qwen 官方文档](https://qwen.readthedocs.io/en/stable/getting_started/quickstart.html)（Quickstart → Best Practices）。

## 三、Linux 使用（命令行）

```bash
chmod +x start-linux.sh
./start-linux.sh                 # 前台运行代理（Ctrl+C 退出并停止模型）

# 或使用完整命令：
python3 cli_app.py add qwen3.6-35b -m /models/qwen.gguf --mmproj /models/mmproj.gguf -c 8192 --no-think --default
python3 cli_app.py list          # 查看已配置模型
python3 cli_app.py serve         # 前台运行代理（首次请求自动加载默认模型）
python3 cli_app.py serve --model qwen3.6-35b   # 启动时预加载指定模型
python3 cli_app.py start <ID>    # 只加载模型不启动代理
python3 cli_app.py stop          # 停止 llama-server
python3 cli_app.py status        # 查看运行状态
python3 cli_app.py remove <ID>   # 删除模型配置
python3 cli_app.py set --llama-server /path/to/llama-server --proxy-port 8800 ...
```

### 命令行详细说明

**`add` 添加/更新模型**（同名 ID 会覆盖旧配置）：

```bash
python3 cli_app.py add <模型ID> -m <主模型.gguf> [选项]
```

| 选项 | 说明 |
|---|---|
| `-m / --model` | 主模型 GGUF 路径（必填） |
| `--mmproj <路径>` | 多模态 mmproj 文件，支持图片输入 |
| `-c / --ctx <N>` | 上下文长度，默认 8192 |
| `-ngl / --ngl <N>` | GPU 卸载层数，默认 999（全部进显存） |
| `-t / --threads <N>` | CPU 线程数，-1 自动 |
| `-fa / --flash-attn <on\|off\|auto>` | Flash Attention，默认 auto |
| `--no-think` | 禁用思考模式（省 token，回复直接输出） |
| `--jinja` | 启用工具调用模板（function calling） |
| `--extra "..."` | 追加的 llama-server 参数 |
| `--default` | 同时设为默认模型 |

**`serve` 运行代理**：启动后终端会打印 TRAE 应填的 API 地址和可用模型 ID；`--model` 可选预加载。建议配合 systemd / supervisor / tmux 常驻运行。

**`set` 修改全局设置**：

```bash
python3 cli_app.py set --proxy-port 8800        # 对外 API 端口
python3 cli_app.py set --backend-port 8801      # llama-server 内部端口
python3 cli_app.py set --llama-server /opt/llama.cpp/build/bin/llama-server
python3 cli_app.py set --auto-load off          # 关闭懒加载（只手动 start）
python3 cli_app.py set --strip-think off        # 保留思考内容
python3 cli_app.py set --default-model qwen3.6-35b
```

llama-server 查找顺序：`set --llama-server` 指定 > 环境变量 `LLAMA_SERVER_PATH` > 项目内/同级 `llama-b*` 目录 > 系统 PATH。

## 四、TRAE 设置方法（重要）

在 TRAE 中：**设置 → 模型 → 添加模型**（即「自定义模型」对话框），按下面填写：

| 设置项 | 填写内容 |
|---|---|
| **API 格式** | `OpenAI Chat Completions 格式` |
| **完整 URL** 开关 | 建议关闭（默认） |
| **自定义请求地址** | `http://127.0.0.1:8800/v1` |
| **模型 ID** | 你在桥接程序中配置的模型 ID，如 `qwen3.6-35b` |
| **模型展示名称** | 随意，如 `本地Qwen`（不填默认显示模型 ID） |
| **API 密钥** | 任意填写，如 `sk-local` |

说明：
- 若你打开了「完整 URL」开关，则地址需填完整端点：`http://127.0.0.1:8800/v1/chat/completions`
- 端口 `8800` 是默认代理端口，如修改过请对应更改
- 点击「添加模型」时 TRAE 会做一次连通性测试（会消耗少量 token），通过即配置成功
- 之后在聊天框选择该自定义模型即可使用；**切换本地模型只需在桥接程序操作，TRAE 不用动**

## 五、目录结构

```
TRAE-llamaCPP-Bridge/
├── dist/
│   └── TRAE-llamaCPP-Bridge.exe  # Windows 图形界面（可直接下载使用，免 Python）
├── Start-Win.bat          # Windows 源码一键启动（GUI，需 Python）
├── start-linux.sh         # Linux 一键启动（CLI）
├── gui_app.py             # Windows 图形界面（源码）
├── cli_app.py             # 命令行界面（Linux/Win 通用）
├── requirements.txt       # 仅依赖 requests
├── llamabridge/
│   ├── config.py          # 配置管理（data/config.json）
│   ├── manager.py         # llama-server 进程管理
│   └── server.py          # OpenAI 兼容代理（SSE 流式透传）
└── data/                  # 运行时生成：配置、日志、测试脚本
    └── e2e_test.py        # 端到端自测脚本（需先启动服务）
```

## 六、常见问题

**Q: 连通性测试失败 / 聊天无响应？**
- 查看桥接程序日志窗口（或 `data/llama-server.log`）是否有报错
- 确认模型已加载：状态栏应显示「模型已加载: xxx」
- 首次请求会触发模型加载，大模型（如 15GB+）从磁盘加载可能需要 1~3 分钟，请耐心等待；建议先在管理界面点「启动 / 切换模型」再点 TRAE 的连通性测试
- 流式请求在模型加载期间会自动发送 SSE 心跳保活，不会超时；非流式请求只能阻塞等待

**Q: TRAE 添加模型时报 "body cannot be replayed safely"？**
- 多为连通性测试在模型加载完成前超时。解决：先在桥接程序启动模型（状态栏显示「模型已加载」），再回 TRAE 点「添加模型」
- 若仍失败：打开任务管理器结束残留的 `llama-server` 进程（程序下次启动会自动清理或复用残留实例），然后重试

**Q: 显存不够（CUDA OOM）？**
- 调低模型配置中的「GPU 卸载层数」（ngl），让部分层走内存
- 调小「上下文长度」
- 使用量化程度更高的 GGUF 文件

**Q: 回复是空的 / 很久才开始回复？**
- 思考型模型（Qwen3 系列等）会先输出长篇思考再回答；在模型配置里选「思考模式: 禁用(省token)」即可

**Q: 想看模型回复的思考过程？**
- 全局设置里关闭「剥离回复中的思考标签」

**Q: 端口冲突？**
- 全局设置里修改「代理端口」，重启程序；TRAE 中对应修改请求地址

## 许可证

MIT

## 致谢

- 本项目思路与 TRAE 接入方案参考自 [Nolan0722/TRAE-Ollama-Bridge](https://github.com/Nolan0722/TRAE-Ollama-Bridge)（MIT License）
- 模型推理由 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) 提供

---
---

# English

Use **locally deployed GGUF models via llama.cpp** (including multimodal models) directly in TRAE and other IDEs, bypassing the restriction that custom models cannot be used.

> This project is inspired by and references [Nolan0722/TRAE-Ollama-Bridge](https://github.com/Nolan0722/TRAE-Ollama-Bridge) (MIT License) — wrapping local models behind an OpenAI-compatible API for TRAE.
> The difference: this project is built on **llama.cpp (llama-server)** instead of Ollama. No Ollama installation is required; GGUF files are loaded directly, with native support for **loading the LLM and mmproj (vision) simultaneously**.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Resource-Saving Design](#resource-saving-design)
- [Preparation](#1-preparation)
- [Windows Usage (GUI)](#2-windows-usage-gui)
- [Linux Usage (CLI)](#3-linux-usage-cli)
- [TRAE Configuration](#4-trae-configuration)
- [Project Layout](#5-project-layout)
- [FAQ](#6-faq)

## How It Works

```
TRAE ──OpenAI format──> Python proxy (127.0.0.1:8800) ──forward──> llama-server (127.0.0.1:8801)
                              │                                       │
                              │ manage / lazy-load / switch models     └─ -m model.gguf --mmproj mmproj.gguf
                              └─ fixed port + model ID mapping → configure TRAE once
```

- `llama-server` already provides an OpenAI-compatible API and multimodal support. The Python program only handles **process management + a lightweight proxy** (the proxy itself uses ~30MB of RAM in practice)
- **Lazy loading**: the model is loaded only when TRAE sends a request, so no VRAM is used while idle
- **One-click switching**: TRAE settings never change (fixed port); switch models in the management UI instead

## Resource-Saving Design

| Measure | Description |
|---|---|
| Single instance | Only one llama-server runs at a time; switching models = unload first, then load (no stacked VRAM/memory) |
| Lazy loading | With `auto_load=on`, the model loads on the first request; no resource usage when idle |
| Built-in WebUI off | `--no-webui` is always passed |
| Configurable context | Default 8192; lower it as needed (context is a major VRAM consumer) |
| Configurable GPU offload | `-ngl` defaults to 999 (all layers in VRAM); reduce it if VRAM is insufficient |
| Flash Attention | Defaults to `auto`; saves more VRAM |
| Disable thinking | Optional for reasoning models (Qwen3 etc.) to avoid wasting tokens |

---

## 1. Preparation

1. **Windows GUI (exe)**: no dependencies required — ready to use out of the box.
   **Windows source / Linux**: install **Python 3.10+** (check *Add to PATH* on Windows)
2. Download a **llama.cpp** release (e.g. `llama-bXXXX-bin-win-cuda-x.x-x64.zip`) and extract it **next to this project** or inside it — it will be auto-detected (you can also set the `llama-server` path manually in settings)
3. Prepare your GGUF model files; for multimodal models, also grab the matching `mmproj-*.gguf`

## 2. Windows Usage (GUI)

### Option A: Standalone exe (recommended)

Download **`dist/TRAE-llamaCPP-Bridge.exe`** (~13MB) directly from this repository — **no Python required**, all dependencies bundled. Just download and double-click to run.

- The exe and its configuration (`data\` folder) stay together: wherever you put the exe, `data\config.json` and logs are created there
- Auto-detection of llama-server works the same: put the exe in the same folder as your llama.cpp release (e.g. `llama-bXXXX-bin-win-cuda-x.x-x64`)
- If Windows SmartScreen warns on first run, choose "More info → Run anyway" (a common false positive for PyInstaller onefile executables)
- If you modify the source, rebuild the exe with:

```bash
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --onefile --noconsole --name TRAE-llamaCPP-Bridge gui_app.py
```

### Option B: Run from source

1. Double-click **`Start-Win.bat`** (dependencies are installed automatically on first run)
2. Menu `Model → Add Model…`:
   - **Model ID**: a custom name such as `qwen3.6-35b` (this is what you fill in TRAE)
   - **LLM model (GGUF)**: browse and select the main model file
   - **mmproj (multimodal)**: browse and select the vision projector file (leave empty for text-only models)
   - Adjust context size / GPU layers / thinking mode as needed
3. Select the model in the list and click **"Start / Switch Model"**
4. To switch models: select another model → click "Start / Switch Model" again — **TRAE needs no changes**

### GUI Reference

| UI element | Purpose |
|---|---|
| **Start / Switch Model** | Loads the selected model; if another model is running it is unloaded first |
| **Stop Model** | Immediately unloads the current model, freeing VRAM/memory |
| **Add / Edit / Delete** | Manage model configurations, persisted in `data\config.json` |
| **Set as Default** | Requests whose model ID matches nothing are served by the default model |
| **Global Settings** | Proxy port, internal port, llama-server path, lazy-load toggle, think-tag stripping toggle |
| **Help → TRAE config guide** | Popup showing exactly what to fill in TRAE for the current environment |
| Status bar | Live proxy port, currently loaded model, and the API URL for TRAE |
| Log panel | Runtime logs; full llama-server output goes to `data\llama-server.log` |

**Model parameter reference (Add/Edit dialog)**:

| Parameter | Default | Description |
|---|---|---|
| Model ID | (required) | The name entered as "Model ID" in TRAE; alphanumeric recommended |
| LLM model (GGUF) | (required) | Main model weights |
| mmproj | empty | Vision projector; enables image input; leave empty for text-only models |
| Context size | 8192 | Token budget per conversation; lower this first when VRAM is tight |
| GPU offload layers | 999 | 999 = all layers in VRAM; decrease gradually (e.g. 40, 20) if OOM — remaining layers stay in RAM |
| CPU threads | -1 | -1 = auto; set to physical core count for CPU-only inference |
| Flash Attention | auto | auto/on significantly reduces VRAM for long contexts |
| Thinking mode | model default | Three levels: model default / thinking budget / disabled. Qwen officially recommends budget ≥1024 for meaningful gains |
| Thinking budget | 4096 | Max thinking tokens (`--reasoning-budget`); prevents runaway thinking |
| temperature | 0.6 | **Official Qwen recommendation**: 0.6 for thinking mode, 0.7 for non-thinking. Greedy decoding is explicitly forbidden (causes endless repetition) |
| top_p | 0.95 | **Official Qwen recommendation**: 0.95 for thinking mode, 0.8 for non-thinking |
| top_k | 20 | Official Qwen recommendation |
| Jinja tool calls | off | Passes `--jinja` for OpenAI-style function calling (needed by TRAE Agent mode) |
| Extra args | empty | Appended to the llama-server command line, e.g. `--cache-type-k q8_0 --cache-type-v q8_0` (quantized KV cache, halves cache VRAM for long contexts) |

> Sampling params (temperature/top_p/top_k) are injected automatically only when the client (TRAE etc.) does not specify them; client-provided values always take precedence.
> Recommendations are from the [official Qwen docs](https://qwen.readthedocs.io/en/stable/getting_started/quickstart.html) (Quickstart → Best Practices).

## 3. Linux Usage (CLI)

```bash
chmod +x start-linux.sh
./start-linux.sh                 # Run the proxy in the foreground (Ctrl+C exits and stops the model)

# Or with full commands:
python3 cli_app.py add qwen3.6-35b -m /models/qwen.gguf --mmproj /models/mmproj.gguf -c 8192 --no-think --default
python3 cli_app.py list          # List configured models
python3 cli_app.py serve         # Run the proxy in the foreground (default model auto-loads on first request)
python3 cli_app.py serve --model qwen3.6-35b   # Preload a specific model at startup
python3 cli_app.py start <ID>    # Load a model without starting the proxy
python3 cli_app.py stop          # Stop llama-server
python3 cli_app.py status        # Show running status
python3 cli_app.py remove <ID>   # Remove a model configuration
python3 cli_app.py set --llama-server /path/to/llama-server --proxy-port 8800 ...
```

### CLI Reference

**`add` — add/update a model** (an existing ID is overwritten):

```bash
python3 cli_app.py add <model-id> -m <main.gguf> [options]
```

| Option | Description |
|---|---|
| `-m / --model` | Path to the main GGUF (required) |
| `--mmproj <path>` | Multimodal mmproj file, enables image input |
| `-c / --ctx <N>` | Context size, default 8192 |
| `-ngl / --ngl <N>` | GPU offload layers, default 999 (all in VRAM) |
| `-t / --threads <N>` | CPU threads, -1 = auto |
| `-fa / --flash-attn <on\|off\|auto>` | Flash Attention, default auto |
| `--no-think` | Disable thinking mode (saves tokens, answers directly) |
| `--jinja` | Enable tool-calling template (function calling) |
| `--extra "..."` | Extra llama-server arguments |
| `--default` | Also set as the default model |

**`serve` — run the proxy**: prints the API URL and available model IDs for TRAE; `--model` optionally preloads. Use systemd / supervisor / tmux to keep it running.

**`set` — global settings**:

```bash
python3 cli_app.py set --proxy-port 8800        # Public API port
python3 cli_app.py set --backend-port 8801      # Internal llama-server port
python3 cli_app.py set --llama-server /opt/llama.cpp/build/bin/llama-server
python3 cli_app.py set --auto-load off          # Disable lazy loading (manual start only)
python3 cli_app.py set --strip-think off        # Keep thinking content in replies
python3 cli_app.py set --default-model qwen3.6-35b
```

llama-server lookup order: `set --llama-server` > `LLAMA_SERVER_PATH` env var > `llama-b*` directories in/next to the project > system PATH.

## 4. TRAE Configuration

In TRAE: **Settings → Models → Add Model** (the "Custom Model" dialog), fill in:

| Field | Value |
|---|---|
| **API format** | `OpenAI Chat Completions format` |
| **Full URL** toggle | Keep it off (default) |
| **Custom request URL** | `http://127.0.0.1:8800/v1` |
| **Model ID** | The model ID configured in the bridge, e.g. `qwen3.6-35b` |
| **Display name** | Anything, e.g. `Local Qwen` (defaults to the model ID) |
| **API key** | Any value, e.g. `sk-local` |

Notes:
- If the **Full URL** toggle is ON, enter the complete endpoint instead: `http://127.0.0.1:8800/v1/chat/completions`
- `8800` is the default proxy port; change it accordingly if you customized it
- TRAE performs a connectivity test when adding the model (consumes a few tokens); once it passes, you're done
- Select the custom model in the chat box to start using it; **switching local models only happens in the bridge app — TRAE never needs changes**

## 5. Project Layout

```
TRAE-llamaCPP-Bridge/
├── dist/
│   └── TRAE-llamaCPP-Bridge.exe  # Windows GUI (download & run, no Python needed)
├── Start-Win.bat          # Windows one-click launcher for source (GUI, needs Python)
├── start-linux.sh         # Linux one-click launcher (CLI)
├── gui_app.py             # Windows GUI (source)
├── cli_app.py             # CLI (Linux/Windows)
├── requirements.txt       # Only depends on requests
├── llamabridge/
│   ├── config.py          # Configuration (data/config.json)
│   ├── manager.py         # llama-server process management
│   └── server.py          # OpenAI-compatible proxy (SSE streaming passthrough)
└── data/                  # Generated at runtime: config, logs, tests
    └── e2e_test.py        # End-to-end test script (requires the service running)
```

## 6. FAQ

**Q: Connectivity test fails / no response in chat?**
- Check the log panel (or `data/llama-server.log`) for errors
- Confirm the model is loaded: the status bar should show "Model loaded: xxx"
- The first request triggers model loading; large models (15GB+) can take 1–3 minutes to load from disk — be patient. Recommended: click "Start / Switch Model" in the bridge app BEFORE running TRAE's connectivity test
- Streaming requests are kept alive with SSE heartbeats during model loading, so they never time out; non-streaming requests simply block until ready

**Q: TRAE reports "body cannot be replayed safely" when adding a model?**
- Usually the connectivity test timed out before the model finished loading. Fix: start the model in the bridge app first (status bar shows "Model loaded"), then click "Add Model" in TRAE
- If it still fails: open Task Manager and kill leftover `llama-server` processes (the app now auto-cleans or reuses leftover instances on next start), then retry

**Q: Out of VRAM (CUDA OOM)?**
- Lower "GPU offload layers" (ngl) so some layers stay in system RAM
- Reduce the context size
- Use a more heavily quantized GGUF

**Q: Empty reply / very long wait before output?**
- Reasoning models (Qwen3 family) produce long thinking traces first; set "Thinking mode: off" in the model configuration

**Q: Want to see the model's thinking process?**
- Turn off "Strip thinking tags" in Global Settings

**Q: Port conflict?**
- Change the proxy port in Global Settings and restart; update the URL in TRAE accordingly

## License

MIT

## Acknowledgments

- The TRAE integration approach is inspired by and references [Nolan0722/TRAE-Ollama-Bridge](https://github.com/Nolan0722/TRAE-Ollama-Bridge) (MIT License)
- Model inference powered by [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
