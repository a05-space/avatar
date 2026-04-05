# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04

ARG APT_MIRROR=mirrors.aliyun.com
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_TRUSTED_HOST=
ARG FLASH_ATTN_WHEEL_URL=

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    MODELSCOPE_CACHE=/app/.cache/modelscope

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    set -eux; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i "s|http://archive.ubuntu.com/ubuntu/|https://${APT_MIRROR}/ubuntu/|g; s|http://security.ubuntu.com/ubuntu/|https://${APT_MIRROR}/ubuntu/|g" /etc/apt/sources.list; \
    fi; \
    if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
        sed -i "s|http://archive.ubuntu.com/ubuntu/|https://${APT_MIRROR}/ubuntu/|g; s|http://security.ubuntu.com/ubuntu/|https://${APT_MIRROR}/ubuntu/|g" /etc/apt/sources.list.d/ubuntu.sources; \
    fi; \
    rm -f /etc/apt/apt.conf.d/docker-clean; \
    printf 'Binary::apt::APT::Keep-Downloaded-Packages "true";\n' > /etc/apt/apt.conf.d/keep-cache; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libgomp1 \
        libsndfile1 \
        python3 \
        python3-pip \
        python3-venv; \
    python3 -m venv "${VIRTUAL_ENV}"

RUN mkdir -p /root/.pip && \
    printf "[global]\nindex-url = %s\ndefault-timeout = 100\nretries = 10\n" "${PIP_INDEX_URL}" > /root/.pip/pip.conf && \
    if [ -n "${PIP_TRUSTED_HOST}" ]; then \
        printf "trusted-host = %s\n" "${PIP_TRUSTED_HOST}" >> /root/.pip/pip.conf; \
    fi

COPY requirements.txt ./requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    set -eux; \
    retry() { \
        n=0; \
        until "$@"; do \
            n=$((n + 1)); \
            if [ "$n" -ge 3 ]; then \
                return 1; \
            fi; \
            echo "Command failed. Retry $n/2 after backoff..."; \
            sleep $((n * 5)); \
        done; \
    }; \
    tr -d '\r' < requirements.txt | grep -E '^(--extra-index-url|torch==|torchaudio==)' > /tmp/requirements.torch.txt; \
    tr -d '\r' < requirements.txt | grep -vE '^(--extra-index-url|torch==|torchaudio==|flash[_-]attn-.*\.whl|# flash-attention-2)$' > /tmp/requirements.docker.txt; \
    echo "Installing packaging toolchain..."; \
    retry "${VIRTUAL_ENV}/bin/python" -m pip install --upgrade pip setuptools wheel; \
    echo "Installing torch packages from PyTorch index..."; \
    retry "${VIRTUAL_ENV}/bin/python" -m pip install -r /tmp/requirements.torch.txt; \
    echo "Installing remaining Python dependencies..."; \
    retry "${VIRTUAL_ENV}/bin/python" -m pip install -r /tmp/requirements.docker.txt; \
    if [ -n "${FLASH_ATTN_WHEEL_URL}" ]; then \
        "${VIRTUAL_ENV}/bin/python" -m pip install "${FLASH_ATTN_WHEEL_URL}"; \
    else \
        echo "FLASH_ATTN_WHEEL_URL is empty, skip flash-attn wheel install."; \
    fi

COPY . .

RUN chmod +x /app/docker-entrypoint.sh && \
    mkdir -p /app/audio /app/.cache/huggingface /app/.cache/modelscope

EXPOSE 8109

ENTRYPOINT ["/app/docker-entrypoint.sh"]
