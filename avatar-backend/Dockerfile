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

RUN set -eux; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i "s|http://archive.ubuntu.com/ubuntu/|https://${APT_MIRROR}/ubuntu/|g; s|http://security.ubuntu.com/ubuntu/|https://${APT_MIRROR}/ubuntu/|g" /etc/apt/sources.list; \
    fi; \
    if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
        sed -i "s|http://archive.ubuntu.com/ubuntu/|https://${APT_MIRROR}/ubuntu/|g; s|http://security.ubuntu.com/ubuntu/|https://${APT_MIRROR}/ubuntu/|g" /etc/apt/sources.list.d/ubuntu.sources; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libgomp1 \
        libsndfile1 \
        python3 \
        python3-pip \
        python3-venv; \
    python3 -m venv "${VIRTUAL_ENV}"; \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.pip && \
    printf "[global]\nindex-url = %s\ndefault-timeout = 100\nretries = 10\n" "${PIP_INDEX_URL}" > /root/.pip/pip.conf && \
    if [ -n "${PIP_TRUSTED_HOST}" ]; then \
        printf "trusted-host = %s\n" "${PIP_TRUSTED_HOST}" >> /root/.pip/pip.conf; \
    fi

COPY requirements.txt ./requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    set -eux; \
    tr -d '\r' < requirements.txt | grep -vE '^(flash[_-]attn-.*\.whl|# flash-attention-2)$' > /tmp/requirements.docker.txt; \
    "${VIRTUAL_ENV}/bin/python" -m pip install --upgrade pip setuptools wheel; \
    "${VIRTUAL_ENV}/bin/python" -m pip install -r /tmp/requirements.docker.txt; \
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
