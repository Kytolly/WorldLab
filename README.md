# WorldLab

## Linux 运行指南

下面的流程用于在 Linux + NVIDIA GPU 机器上验证完整的 OpenPI 到 WorldLab
再到 Panel 的可运行流。OpenPI 官方 serving 目前以 Ubuntu 22.04 为主要支持
环境；WorldLab 客户端和 UI 可以在另一个 Python 环境中运行。

> Windows note: OpenPI 当前固定依赖 `jax[cuda12]`，该 CUDA 插件没有 Windows
> wheel。因此 Windows 上建议让原生 `env_worldlab` 运行 WorldLab/Panel，把真实
> OpenPI 模型服务放到 WSL2 Ubuntu（或独立 Linux 主机）。两者仍通过
> `ws://127.0.0.1:8000` 通信。

### 0. 前置条件

准备 Ubuntu 22.04、可用的 NVIDIA 驱动和至少约 8 GB 可用显存，并安装
Conda、Git、Git LFS 和 uv：

```bash
sudo apt update
sudo apt install -y git git-lfs curl
git lfs install
curl -LsSf https://astral.sh/uv/install.sh | sh
```

重新打开终端后确认：

```bash
conda --version
uv --version
nvidia-smi
```

Windows + WSL2 用户需要安装 Ubuntu 22.04 WSL2，并在 Ubuntu 终端中确认
`nvidia-smi` 能看到 GPU。Windows 中已有的 `openpi` Conda 环境不能替代这个
Linux serving 环境；在 Windows 上解析官方依赖会因 CUDA JAX 插件无 wheel 而失败。

### 1. 获取代码

```bash
git clone <your-worldlab-repository> WorldLab
cd WorldLab
git checkout dev
git submodule update --init --recursive
```

确认当前分支包含 v0.2.5 的 OpenPI/UI 实现：

```bash
git log -1 --oneline
test -f scripts/serve_policy_pi05.sh
test -f packages/worldlab-integrations-openpi/src/worldlab_openpi/openpi_serving/serve_pi05.py
```

### 2. 创建 WorldLab 环境

`env_worldlab` 包含 WorldLab、Panel、OpenPI client 和测试工具，不包含完整
的 GPU 模型 serving 依赖：

```bash
conda env create -f environment.yml
conda activate env_worldlab
```

如果环境已存在：

```bash
conda env update -f environment.yml --prune
conda activate env_worldlab
```

将仓库内 OpenPI 源码注册到该环境，使 IDE 和 Python 能解析 `import openpi`：

```bash
python -m pip install --no-deps -e third-party/openpi
```

检查基础导入、自动化验收和测试：

```bash
python -c "import worldlab, openpi, worldlab_openpi; print('imports ok')"
python -m worldlab --acceptance --goal 3 --seed 0
pytest -q
python -m mypy --config-file pyproject.toml
```

`python -m worldlab --acceptance` 是不依赖 GPU、权重或网络的确定性发布验收；
`pytest -q` 会同时收集核心、Panel 和 OpenPI fake-server 协议测试。测试中的
fake OpenPI 服务只属于测试模块，不会被安装为运行时服务。

### 3. 准备 OpenPI serving 环境

不要在 `env_worldlab` 中安装 OpenPI 的完整 GPU 依赖。使用 OpenPI submodule
自己的 uv 环境：

```bash
cd third-party/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
source .venv/bin/activate
cd ../..
```

pi05 PyTorch 模型需要 OpenPI 的 Transformers overlay：

```bash
cp -r third-party/openpi/src/openpi/models_pytorch/transformers_replace/* \
  third-party/openpi/.venv/lib/python3.11/site-packages/transformers/
```

确认当前终端使用的是 OpenPI serving 环境：

```bash
which python
python -c "import openpi, flax, torch; print(openpi.__file__); print(torch.cuda.is_available())"
```

如果 `python` 指向 `env_worldlab`，或出现 `No module named flax`，说明尚未激活
`third-party/openpi/.venv`。

Windows + WSL2 用户在 Ubuntu 终端执行本节。仓库可以通过 `/mnt/d/...` 访问，
但建议将 OpenPI checkout 和 checkpoint 放在 WSL2 文件系统中，以减少模型加载
 时的跨文件系统开销；Windows 原生终端只运行第 6 节的 Panel 命令。

### Windows + Docker Desktop

如果希望不切换到 Linux 系统，可以使用 Docker Desktop 的 WSL2 backend。仓库提供
`docker-compose.openpi.yml`：容器使用 OpenPI 官方 CUDA image 和依赖，Windows
的 `env_worldlab` 只运行 Panel 客户端。

先在 Docker Desktop 中启用 WSL2 backend、GPU support，并确认 Docker 能访问 GPU：

```powershell
docker run --rm --gpus all nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 nvidia-smi
```

然后在 WorldLab 根目录构建并启动 OpenPI 容器：

```powershell
docker compose -f docker-compose.openpi.yml up --build
```

容器首次构建会下载 OpenPI 的 CUDA/JAX/PyTorch 依赖，耗时和磁盘占用都较大。
服务启动后，Windows 终端运行第 6 节的 Panel 命令；Docker Compose 默认将主机
端口映射为 `9000 -> 8000`，所以 `--policy-url` 使用
`ws://127.0.0.1:9000`。健康检查为：

```powershell
curl.exe http://127.0.0.1:9000/healthz
```

停止服务：

```powershell
docker compose -f docker-compose.openpi.yml down
```

如果宿主机的 9000 端口被占用，可使用 `OPENPI_PORT=9001`，同时将 Panel 的
`--policy-url` 改为 `ws://127.0.0.1:9001`。Docker Desktop 的 GPU 验证失败时，
应改用 WSL2 Ubuntu 或独立 Linux 主机，不要尝试把 `jax[cuda12]` 强行安装进
Windows `openpi` Conda 环境。

### 4. 放置 pi05 权重

脚本默认读取：

```text
model_zoo/agibot-world/Genie-Envisioner-Sim-v2.0/
└── checkpoints/pi05_gesim_g01op_test/
    ├── model.safetensors
    └── assets/gesim/norm_stats.json
```

权重文件不进入 Git。也可以通过 `OPENPI_CKPT` 指定其他目录：

```bash
export OPENPI_CKPT=/absolute/path/to/pi05_gesim_g01op_test
```

### 5. 启动 OpenPI 服务

在终端 A，保持 OpenPI `.venv` 激活：

```bash
PORT=8000 ASSET_ID=gesim bash scripts/serve_policy_pi05.sh
```

脚本会启动包内的：

```text
worldlab_openpi.openpi_serving.serve_pi05
```

服务启动后，可以先检查 HTTP health endpoint：

```bash
curl http://127.0.0.1:8000/healthz
```

预期返回 `OK`。如果端口被占用：

```bash
PORT=8001 bash scripts/serve_policy_pi05.sh
```

### 6. 启动 WorldLab + Panel 流

在终端 B，切换回 WorldLab 环境：

```bash
conda activate env_worldlab
python examples/openpi_panel_smoke.py --policy-url ws://127.0.0.1:8000 --host 127.0.0.1 --port 5018 --goal 6 --chunk-size 4 --step-delay 0.5
```

浏览器打开：

```text
http://127.0.0.1:5018
```

Panel 会先立即打开，OpenPI 连接在后台等待；服务握手完成后才开始 rollout。
可运行的验收现象包括：

- 页面状态先显示 `idle`/`running`，闭环结束后变为 `completed`；
- synthetic frame 正常显示；
- action 表和 predicted state 表出现 `4 x 16` 数据；
- 显示 reward、chunk index 和 model latency；
- 事件时间线出现 `policy_acted`、`environment_stepped`、
  `transition_committed` 和 `episode_ended`。

这个 smoke 流使用的是随机 synthetic frames，主要验证 OpenPI WebSocket、动作
shape/layout、WorldLab 闭环和 Panel 实时展示，不代表随机画面下的策略动作具有
任务语义。

### 7. 常见问题

`ModuleNotFoundError: openpi`：确认执行过
`python -m pip install --no-deps -e third-party/openpi`，并确认 IDE 解释器是
`env_worldlab`。

`ModuleNotFoundError: flax` 或 `jax`：说明服务是在 `env_worldlab` 中启动的，
应先激活 `third-party/openpi/.venv`。

页面一直停留在 `idle`：这是客户端仍在等待 OpenPI WebSocket 端口，先检查终端 A
是否已经输出 serving 日志。页面会在 `--connect-timeout`（默认 60 秒）后把连接
错误写入事件时间线。

OpenPI 端口只接受 WebSocket，不是浏览器页面。`curl /healthz` 返回 HTTP `200`
是正常的；如果访问 `http://127.0.0.1:8000/` 或误把 `http://` 写进
`--policy-url`，终端出现 HTTP `426 Upgrade Required` 也是正常的协议拒绝。浏览器
应打开 `http://127.0.0.1:5018`，客户端地址必须使用
`ws://127.0.0.1:8000`（或 `wss://`）。

显存不足：pi05 推理通常需要超过 8 GB 显存；优先关闭其他 GPU 进程，或使用
OpenPI 服务端的 `--device` 参数选择正确的 GPU。

## Observe the closed loop

Run the dependency-free demo with a phase-by-phase runtime trace:

```text
python -m worldlab --goal 3 --trace
```

The trace records episode reset, policy actions, environment steps, committed
transitions, episode completion, per-phase durations, and runtime failures. It
ends with an ordering and completeness diagnosis.

Programmatic use keeps the raw typed events available for inspection:

```python
from worldlab import EnvironmentLoop, TraceRecorder, build_demo

env, agent = build_demo(goal=3)
trace = TraceRecorder()

with EnvironmentLoop(env, agent, trace=trace) as loop:
    result = loop.run_episode(seed=0)

print(trace.format_timeline())
assert trace.diagnose().healthy
```

If reset, policy inference, environment stepping, agent observation, rendering,
or a callback fails, WorldLab records a `RuntimeErrorEvent` with the exact
runtime phase and traceback before re-raising the original exception.

## v0.3.1 TODO

This release uses fixed-length action chunks at the Environment boundary. A
chunk whose leading dimension is not exactly `chunk_size` is rejected.

- Decide where padding belongs: Policy, ActionAdapter, Simulator, or model.
- Decide whether truncation means rejecting an oversized chunk or explicitly
  slicing it before simulator execution.
- Define variable-length action spaces without weakening fixed-shape batch
  validation.
- Add the optional Gymnasium Dict/Tuple adapter without making Gymnasium a core
  dependency.
- Define configuration-to-space construction and range normalization rules.

## v0.3.4 TODO

`ComposableTask` now connects the three managers to the legacy `Task` interface
with a fixed signal order and namespaced diagnostics. The following work remains
intentionally deferred:

- Define configuration/factory construction for managers and terms.
- Add actor/critic observation selection.
- Add optional Gymnasium conversion and integration tests.
- Keep observation history, noise/corruption, vectorized execution, and async
  reward batching out of this release.

