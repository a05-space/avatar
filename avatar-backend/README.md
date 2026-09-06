# avatar-backend

数字人语音、口型和表情推理服务，默认监听容器端口 `8109`，开发 Compose 将宿主机端口映射为 `8210`。

## Docker 挂载式开发

开发环境使用 `docker-compose.dev.yml`，生产环境仍使用 `docker-compose.yml`。开发镜像只提供 CUDA、系统库和 Python 虚拟环境，业务源码通过 `.:/app` 挂载；修改 Python 文件后 Uvicorn 会自动 reload。

### 前置条件

- Linux 主机安装 NVIDIA 驱动和 NVIDIA Container Toolkit。
- Docker 能够使用 `runtime: nvidia`，并且 GPU 显存满足当前推理/TTS 模型需求。
- 准备模型目录：`pretrained_models/`、`Qwen3-TTS-12Hz-1.7B-Base/`。目录不存在时 Docker 可能创建空目录，服务会因缺少模型启动失败。
- 准备 `.env.docker`：

```bash
cp .env.docker.example .env.docker
```

至少检查 `BASE_URL`、`API_KEY`、`TTS_PROVIDER` 以及阿里云 TTS 配置。

### 启动

```bash
docker compose --env-file .env.docker -f docker-compose.dev.yml up --build
```

后台运行和查看日志：

```bash
docker compose --env-file .env.docker -f docker-compose.dev.yml up -d --build
docker compose --env-file .env.docker -f docker-compose.dev.yml logs -f avatar-backend
```

健康检查：

```bash
curl http://127.0.0.1:8210/healthz
```

返回 `{"status":"ok"}` 即表示服务已启动。

### 依赖和重载

- 容器启动时执行 `pip install -r requirements.txt`，pip 缓存和 `/opt/venv` 使用 Docker 命名卷。
- 修改 `requirements.txt` 后执行 `docker compose --env-file .env.docker -f docker-compose.dev.yml restart avatar-backend`。
- 修改代码会自动 reload，但每次 reload 都会重新初始化推理和 TTS 模型，可能需要较长时间并再次占用大量显存。
- 清理开发虚拟环境、缓存和容器（不会删除源码和模型目录）：

```bash
docker compose --env-file .env.docker -f docker-compose.dev.yml down -v
```

## 与面试后端联调

三个项目使用独立 Compose 时，面试后端容器通过宿主机端口访问本服务：

```env
AVATAR_BACKEND_BASE_URL=http://host.docker.internal:8210
```

面试后端 Compose 已配置 `host.docker.internal:host-gateway`。Linux 环境需确认 Docker 支持该映射；也可以将地址改成宿主机实际 IP。

推荐启动顺序：先启动 RAG，再启动 avatar，最后启动面试后端。

## 生产部署

生产部署继续使用 `Dockerfile` 和 `docker-compose.yml`。生产镜像复制代码，不开启 `--reload`；开发和生产配置不要混用。
