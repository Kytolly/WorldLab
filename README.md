# WorldLab

## Linux 运行指南

下面的流程用于在 Linux + NVIDIA GPU 机器上验证完整的 OpenPI 到 WorldLab
再到 Panel 的可运行流。OpenPI 官方 serving 目前以 Ubuntu 22.04 为主要支持
环境；WorldLab 客户端和 UI 可以在另一个 Python 环境中运行。

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

### 1. 获取代码

```bash
git clone <your-worldlab-repository> WorldLab
cd WorldLab
git checkout dev
git submodule update --init --recursive
```

确认当前分支包含 v0.2.5 或之后的 OpenPI/UI 实现：

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

检查基础导入和测试：

```bash
python -c "import worldlab, openpi, worldlab_openpi; print('imports ok')"
pytest -q tests packages/worldlab-integrations-openpi/tests
python -m mypy --config-file pyproject.toml
```

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
python examples/openpi_panel_smoke.py \
  --policy-url ws://127.0.0.1:8000 \
  --host 127.0.0.1 \
  --port 5018 \
  --goal 6 \
  --chunk-size 4 \
  --step-delay 0.5
```

浏览器打开：

```text
http://127.0.0.1:5018
```

可运行的验收现象包括：

- 页面状态从连接中变为 `completed`；
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

页面一直 `connecting`：先检查终端 A 是否已经输出 serving 日志，再执行
`curl http://127.0.0.1:8000/healthz`，并确认客户端的 `--policy-url` 与服务端
端口一致。

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


