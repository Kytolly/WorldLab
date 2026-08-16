# WorldLab OpenPI serving image.
# The official OpenPI lockfile is retained, with a cu128 PyTorch wheel added
# for RTX 5080 (sm_120) support.

# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y git git-lfs linux-headers-generic build-essential clang

ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV PATH="/.venv/bin:${PATH}"

COPY third-party/openpi/uv.lock /app/uv.lock
COPY third-party/openpi/pyproject.toml /app/pyproject.toml
COPY third-party/openpi/packages/openpi-client/pyproject.toml /app/packages/openpi-client/pyproject.toml
COPY third-party/openpi/packages/openpi-client/src /app/packages/openpi-client/src

RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT
RUN --mount=type=cache,target=/root/.cache/uv \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev

# OpenPI's PyTorch overlay is required by pi05.
COPY third-party/openpi/src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" | xargs dirname | xargs -I{} cp -r /tmp/transformers_replace/* {} && rm -rf /tmp/transformers_replace

# The official lock currently resolves a cu126 wheel. RTX 5080 requires the
# cu128 build; keep the rest of the locked dependency set unchanged.
RUN /bin/uv pip install --python /.venv/bin/python --reinstall --no-deps \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.7.1 torchvision==0.22.1
