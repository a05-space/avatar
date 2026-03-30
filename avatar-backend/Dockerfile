FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

ARG APT_MIRROR=mirrors.aliyun.com
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ARG FLASH_ATTN_WHEEL_URL=

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
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
    ln -sf /usr/bin/python3 /usr/local/bin/python; \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip; \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.pip && \
    printf "[global]\nindex-url = %s\ntrusted-host = %s\n" "${PIP_INDEX_URL}" "${PIP_TRUSTED_HOST}" > /root/.pip/pip.conf

COPY requirements.txt ./requirements.txt

RUN set -eux; \
    grep -vE '^flash_attn-.*\.whl$' requirements.txt > /tmp/requirements.docker.txt; \
    python -m pip install --upgrade pip setuptools wheel; \
    python -m pip install -r /tmp/requirements.docker.txt; \
    if [ -n "${FLASH_ATTN_WHEEL_URL}" ]; then \
        python -m pip install "${FLASH_ATTN_WHEEL_URL}"; \
    else \
        echo "FLASH_ATTN_WHEEL_URL is empty, skip flash-attn wheel install."; \
    fi

COPY . .

RUN chmod +x /app/docker-entrypoint.sh && \
    mkdir -p /app/audio /app/.cache/huggingface /app/.cache/modelscope

EXPOSE 8109

ENTRYPOINT ["/app/docker-entrypoint.sh"]
