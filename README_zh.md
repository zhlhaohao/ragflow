docker pull searxng/searxng:2025.5.7-f52cd3f00
docker tag searxng/searxng:2025.5.7-f52cd3f00 10.119.101.20:9869/searxng:2025.5.7-f52cd3f00
docker push 10.119.101.20:9869/searxng:2025.5.7-f52cd3f00


docker pull ghcr.io/intergalacticalvariable/reader:latest
docker tag ghcr.io/intergalacticalvariable/reader:latest 10.119.101.20:9869/jina-reader:2025.5.7
docker push 10.119.101.20:9869/jina-reader:2025.5.7


## 此项目的配置文件详细说明

### docker/.env
管服务的，docker-compose-base.yaml\docker-compose.yaml需要读此文件，它指定了es、infinity、mysql、minio、redis等数据库服务的端口和密码等信息,容器启动后对外的端口和密码是由.env指定的

### conf/service_conf.yaml
管应用的，这个是后台应用读取的配置文件，指定了es、infinity、mysql、minio、redis等数据库服务的ip地址、端口和密码等信息，所以必须和.env的内容一致

### conf/local.service_conf.yaml
这个优先级比conf/service_conf.yaml，如果存在这个文件就读这个配置文件

### docker/service_conf.yaml.template
这个文件在通过Dockerfile创建镜像的过程中有用，因为会将该文件在docker build的时候，拷贝到conf/service_conf.yaml，为应用提供配置，所以也必须跟conf/service_conf.yaml一致

### docker/service_conf.yaml
无用，只是符号链接到conf/service_conf.yaml

### 项目根目录/.env
```
PYTHONPATH=/home/lianghao/github/ragflow
HF_ENDPOINT=https://hf-mirror.com
```

### docker-compose-base.yaml
docker compose编排文件，配置es\mysql\redis\minio 等的docker镜像在本地启动，其配置来源于同目录的 .env




### 设置默认模型 
1. local.service_conf.yaml 或者 service_conf.yaml

```
user_default_llm:
  factory: 'Tongyi-Qianwen'   # 新建了用户和tenant的时候，所设置的默认的模型
  api_key: 'sk-db2ad92210a348fd884c3b94655095c5'   # 默认的api_key
```

2. api/settings.py

这里是设置了默认模型后，该模型的一些其它参数，例如 factory: 'Tongyi-Qianwen'，那么默认的 chat_model、embedding_model 就从这儿取
```python
default_llm = {
   "Tongyi-Qianwen": {
         "chat_model": "qwen-plus",
         "embedding_model": "text-embedding-v3",  # 这个会影响EMBEDDING_MDL的值，继而新建tenant记录的时候的embd_id字段默认值，如果某个用户新建知识库，那么嵌入模型就取自该用户在tenant表的embd_id字段
         # "embedding_model": "text-embedding-v2",
         "image2text_model": "qwen-vl-max",
         "asr_model": "paraformer-realtime-8k-v1",
   },
}
```

## 更新到 0.15版后的需要补充安装的依赖
```shell
pip install infinity-sdk=="0.5.0"
pip install infinity-emb=="0.0.66"

pip install flasgger=="0.9.7.1"
pip install polars=="1.9.0"

pip install xxhash=="3.5.0"
pip install valkey=="6.0.2"
pip install pyodbc=="5.2.0"

sudo apt-get update
sudo apt-get install unixodbc unixodbc-dev
```

## 上游仓库更新后同步到本地
```shell
git remote add upstream https://github.com/infiniflow/ragflow.git --- 只需要执行一次
git fetch upstream
git merge upstream/main
```

## 原始内容

<div align="center">
<a href="https://demo.ragflow.io/">
<img src="web/src/assets/logo-with-text.png" width="350" alt="ragflow logo">
</a>
</div>

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README_zh.md">简体中文</a> |
  <a href="./README_tzh.md">繁体中文</a> |
  <a href="./README_ja.md">日本語</a> |
  <a href="./README_ko.md">한국어</a> |
  <a href="./README_id.md">Bahasa Indonesia</a> |
  <a href="/README_pt_br.md">Português (Brasil)</a>
</p>

<p align="center">
    <a href="https://x.com/intent/follow?screen_name=infiniflowai" target="_blank">
        <img src="https://img.shields.io/twitter/follow/infiniflow?logo=X&color=%20%23f5f5f5" alt="follow on X(Twitter)">
    </a>
    <a href="https://demo.ragflow.io" target="_blank">
        <img alt="Static Badge" src="https://img.shields.io/badge/Online-Demo-4e6b99">
    </a>
    <a href="https://hub.docker.com/r/infiniflow/ragflow" target="_blank">
        <img src="https://img.shields.io/badge/docker_pull-ragflow:v0.19.0-brightgreen" alt="docker pull infiniflow/ragflow:v0.19.0">
    </a>
    <a href="https://github.com/infiniflow/ragflow/releases/latest">
        <img src="https://img.shields.io/github/v/release/infiniflow/ragflow?color=blue&label=Latest%20Release" alt="Latest Release">
    </a>
    <a href="https://github.com/infiniflow/ragflow/blob/main/LICENSE">
        <img height="21" src="https://img.shields.io/badge/License-Apache--2.0-ffffff?labelColor=d4eaf7&color=2e6cc4" alt="license">
    </a>
</p>

<h4 align="center">
  <a href="https://ragflow.io/docs/dev/">Document</a> |
  <a href="https://github.com/infiniflow/ragflow/issues/4214">Roadmap</a> |
  <a href="https://twitter.com/infiniflowai">Twitter</a> |
  <a href="https://discord.gg/NjYzJD3GM3">Discord</a> |
  <a href="https://demo.ragflow.io">Demo</a>
</h4>

## 💡 RAGFlow 是什么？

[RAGFlow](https://ragflow.io/) 是一款基于深度文档理解构建的开源 RAG（Retrieval-Augmented Generation）引擎。RAGFlow 可以为各种规模的企业及个人提供一套精简的 RAG 工作流程，结合大语言模型（LLM）针对用户各类不同的复杂格式数据提供可靠的问答以及有理有据的引用。

## 🎮 Demo 试用

请登录网址 [https://demo.ragflow.io](https://demo.ragflow.io) 试用 demo。

<div align="center" style="margin-top:20px;margin-bottom:20px;">
<img src="https://github.com/infiniflow/ragflow/assets/7248/2f6baa3e-1092-4f11-866d-36f6a9d075e5" width="1200"/>
<img src="https://github.com/user-attachments/assets/504bbbf1-c9f7-4d83-8cc5-e9cb63c26db6" width="1200"/>
</div>

## 🔥 近期更新

- 2025-03-19 PDF和DOCX中的图支持用多模态大模型去解析得到描述.
- 2025-02-28 结合互联网搜索（Tavily），对于任意大模型实现类似 Deep Research 的推理功能.
- 2025-01-26 优化知识图谱的提取和应用，提供了多种配置选择。
- 2024-12-18 升级了 DeepDoc 的文档布局分析模型。
- 2024-11-01 对解析后的 chunk 加入关键词抽取和相关问题生成以提高召回的准确度。
- 2024-08-22 支持用 RAG 技术实现从自然语言到 SQL 语句的转换。

## 🎉 关注项目

⭐️ 点击右上角的 Star 关注 RAGFlow，可以获取最新发布的实时通知 !🌟

<div align="center" style="margin-top:20px;margin-bottom:20px;">
<img src="https://github.com/user-attachments/assets/18c9707e-b8aa-4caf-a154-037089c105ba" width="1200"/>
</div>

## 🌟 主要功能

### 🍭 **"Quality in, quality out"**

- 基于[深度文档理解](./deepdoc/README.md)，能够从各类复杂格式的非结构化数据中提取真知灼见。
- 真正在无限上下文（token）的场景下快速完成大海捞针测试。

### 🍱 **基于模板的文本切片**

- 不仅仅是智能，更重要的是可控可解释。
- 多种文本模板可供选择

### 🌱 **有理有据、最大程度降低幻觉（hallucination）**

- 文本切片过程可视化，支持手动调整。
- 有理有据：答案提供关键引用的快照并支持追根溯源。

### 🍔 **兼容各类异构数据源**

- 支持丰富的文件类型，包括 Word 文档、PPT、excel 表格、txt 文件、图片、PDF、影印件、复印件、结构化数据、网页等。

### 🛀 **全程无忧、自动化的 RAG 工作流**

- 全面优化的 RAG 工作流可以支持从个人应用乃至超大型企业的各类生态系统。
- 大语言模型 LLM 以及向量模型均支持配置。
- 基于多路召回、融合重排序。
- 提供易用的 API，可以轻松集成到各类企业系统。

## 🔎 系统架构

<div align="center" style="margin-top:20px;margin-bottom:20px;">
<img src="https://github.com/infiniflow/ragflow/assets/12318111/d6ac5664-c237-4200-a7c2-a4a00691b485" width="1000"/>
</div>

## 🎬 快速开始

### 📝 前提条件

- CPU >= 4 核
- RAM >= 16 GB
- Disk >= 50 GB
- Docker >= 24.0.0 & Docker Compose >= v2.26.1
- [gVisor](https://gvisor.dev/docs/user_guide/install/): 仅在你打算使用 RAGFlow 的代码执行器（沙箱）功能时才需要安装。

> [!TIP]
> 如果你并没有在本机安装 Docker（Windows、Mac，或者 Linux）, 可以参考文档 [Install Docker Engine](https://docs.docker.com/engine/install/) 自行安装。

### 🚀 启动服务器

1. 确保 `vm.max_map_count` 不小于 262144：

   > 如需确认 `vm.max_map_count` 的大小：
   >
   > ```bash
   > $ sysctl vm.max_map_count
   > ```
   >
   > 如果 `vm.max_map_count` 的值小于 262144，可以进行重置：
   >
   > ```bash
   > # 这里我们设为 262144:
   > $ sudo sysctl -w vm.max_map_count=262144
   > ```
   >
   > 你的改动会在下次系统重启时被重置。如果希望做永久改动，还需要在 **/etc/sysctl.conf** 文件里把 `vm.max_map_count` 的值再相应更新一遍：
   >
   > ```bash
   > vm.max_map_count=262144
   > ```

2. 克隆仓库：

   ```bash
   $ git clone https://github.com/infiniflow/ragflow.git
   ```

3. 进入 **docker** 文件夹，利用提前编译好的 Docker 镜像启动服务器：

> [!CAUTION]
> 请注意，目前官方提供的所有 Docker 镜像均基于 x86 架构构建，并不提供基于 ARM64 的 Docker 镜像。
> 如果你的操作系统是 ARM64 架构，请参考[这篇文档](https://ragflow.io/docs/dev/build_docker_image)自行构建 Docker 镜像。

   > 运行以下命令会自动下载 RAGFlow slim Docker 镜像 `v0.19.0-slim`。请参考下表查看不同 Docker 发行版的描述。如需下载不同于 `v0.19.0-slim` 的 Docker 镜像，请在运行 `docker compose` 启动服务之前先更新 **docker/.env** 文件内的 `RAGFLOW_IMAGE` 变量。比如，你可以通过设置 `RAGFLOW_IMAGE=infiniflow/ragflow:v0.19.0` 来下载 RAGFlow 镜像的 `v0.19.0` 完整发行版。

   ```bash
   $ cd ragflow/docker
   # Use CPU for embedding and DeepDoc tasks:
   $ docker compose -f docker-compose.yml up -d

   # To use GPU to accelerate embedding and DeepDoc tasks:
   # docker compose -f docker-compose-gpu.yml up -d
   ```

   | RAGFlow image tag | Image size (GB) | Has embedding models? | Stable?                  |
   | ----------------- | --------------- | --------------------- | ------------------------ |
   | v0.19.0           | &approx;9       | :heavy_check_mark:    | Stable release           |
   | v0.19.0-slim      | &approx;2       | ❌                    | Stable release           |
   | nightly           | &approx;9       | :heavy_check_mark:    | _Unstable_ nightly build |
   | nightly-slim      | &approx;2       | ❌                     | _Unstable_ nightly build |

   > [!TIP]
   > 如果你遇到 Docker 镜像拉不下来的问题，可以在 **docker/.env** 文件内根据变量 `RAGFLOW_IMAGE` 的注释提示选择华为云或者阿里云的相应镜像。
   >
   > - 华为云镜像名：`swr.cn-north-4.myhuaweicloud.com/infiniflow/ragflow`
   > - 阿里云镜像名：`registry.cn-hangzhou.aliyuncs.com/infiniflow/ragflow`

4. 服务器启动成功后再次确认服务器状态：

   ```bash
   $ docker logs -f ragflow-server
   ```

   _出现以下界面提示说明服务器启动成功：_

   ```bash
        ____   ___    ______ ______ __
       / __ \ /   |  / ____// ____// /____  _      __
      / /_/ // /| | / / __ / /_   / // __ \| | /| / /
     / _, _// ___ |/ /_/ // __/  / // /_/ /| |/ |/ /
    /_/ |_|/_/  |_|\____//_/    /_/ \____/ |__/|__/

    * Running on all addresses (0.0.0.0)
   ```

   > 如果您在没有看到上面的提示信息出来之前，就尝试登录 RAGFlow，你的浏览器有可能会提示 `network anormal` 或 `网络异常`。

5. 在你的浏览器中输入你的服务器对应的 IP 地址并登录 RAGFlow。
   > 上面这个例子中，您只需输入 http://IP_OF_YOUR_MACHINE 即可：未改动过配置则无需输入端口（默认的 HTTP 服务端口 80）。
6. 在 [service_conf.yaml.template](./docker/service_conf.yaml.template) 文件的 `user_default_llm` 栏配置 LLM factory，并在 `API_KEY` 栏填写和你选择的大模型相对应的 API key。

   > 详见 [llm_api_key_setup](https://ragflow.io/docs/dev/llm_api_key_setup)。

   _好戏开始，接着奏乐接着舞！_

## 🔧 系统配置

系统配置涉及以下三份文件：

- [.env](./docker/.env)：存放一些基本的系统环境变量，比如 `SVR_HTTP_PORT`、`MYSQL_PASSWORD`、`MINIO_PASSWORD` 等。
- [service_conf.yaml.template](./docker/service_conf.yaml.template)：配置各类后台服务。
- [docker-compose.yml](./docker/docker-compose.yml): 系统依赖该文件完成启动。

请务必确保 [.env](./docker/.env) 文件中的变量设置与 [service_conf.yaml.template](./docker/service_conf.yaml.template) 文件中的配置保持一致！

如果不能访问镜像站点 hub.docker.com 或者模型站点 huggingface.co，请按照 [.env](./docker/.env) 注释修改 `RAGFLOW_IMAGE` 和 `HF_ENDPOINT`。

> [./docker/README](./docker/README.md) 解释了 [service_conf.yaml.template](./docker/service_conf.yaml.template) 用到的环境变量设置和服务配置。

如需更新默认的 HTTP 服务端口(80), 可以在 [docker-compose.yml](./docker/docker-compose.yml) 文件中将配置 `80:80` 改为 `<YOUR_SERVING_PORT>:80`。

> 所有系统配置都需要通过系统重启生效：
>
> ```bash
> $ docker compose -f docker-compose.yml up -d
> ```

### 把文档引擎从 Elasticsearch 切换成为 Infinity

RAGFlow 默认使用 Elasticsearch 存储文本和向量数据. 如果要切换为 [Infinity](https://github.com/infiniflow/infinity/), 可以按照下面步骤进行:

1. 停止所有容器运行:

   ```bash
   $ docker compose -f docker/docker-compose.yml down -v
   ```
   Note: `-v` 将会删除 docker 容器的 volumes，已有的数据会被清空。

2. 设置 **docker/.env** 目录中的 `DOC_ENGINE` 为 `infinity`.

3. 启动容器:

   ```bash
   $ docker compose -f docker-compose.yml up -d
   ```

> [!WARNING]
> Infinity 目前官方并未正式支持在 Linux/arm64 架构下的机器上运行.

## 🔧 源码编译 Docker 镜像（不含 embedding 模型）

本 Docker 镜像大小约 2 GB 左右并且依赖外部的大模型和 embedding 服务。

```bash
git clone https://github.com/infiniflow/ragflow.git
cd ragflow/
docker build --platform linux/amd64 --build-arg LIGHTEN=1 --build-arg NEED_MIRROR=1 -f Dockerfile -t infiniflow/ragflow:nightly-slim .
```

## 🔧 源码编译 Docker 镜像（包含 embedding 模型）

本 Docker 大小约 9 GB 左右。由于已包含 embedding 模型，所以只需依赖外部的大模型服务即可。

```bash
git clone https://github.com/infiniflow/ragflow.git
cd ragflow/
docker build --platform linux/amd64 --build-arg NEED_MIRROR=1 -f Dockerfile -t infiniflow/ragflow:nightly .
```

## 🔨 以源代码启动服务

1. 安装 uv。如已经安装，可跳过本步骤：

   ```bash
   pipx install uv pre-commit
   export UV_INDEX=https://mirrors.aliyun.com/pypi/simple
   ```

2. 下载源代码并安装 Python 依赖：

   ```bash
   git clone https://github.com/infiniflow/ragflow.git
   cd ragflow/
   uv sync --python 3.10 --all-extras # install RAGFlow dependent python modules
   uv run download_deps.py
   pre-commit install
   ```

3. 通过 Docker Compose 启动依赖的服务（MinIO, Elasticsearch, Redis, and MySQL）：

   ```bash
   docker compose -f docker/docker-compose-base.yml up -d
   ```

   在 `/etc/hosts` 中添加以下代码，目的是将 **conf/service_conf.yaml** 文件中的所有 host 地址都解析为 `127.0.0.1`：

   ```
   127.0.0.1       es01 infinity mysql minio redis sandbox-executor-manager
   ```
4. 如果无法访问 HuggingFace，可以把环境变量 `HF_ENDPOINT` 设成相应的镜像站点：

   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

5. 如果你的操作系统没有 jemalloc，请按照如下方式安装：

   ```bash
   # ubuntu
   sudo apt-get install libjemalloc-dev
   # centos
   sudo yum install jemalloc
   ```

6. 启动后端服务：

   ```bash
   source .venv/bin/activate
   export PYTHONPATH=$(pwd)
   bash docker/launch_backend_service.sh
   ```

7. 安装前端依赖：

   ```bash
   cd web
   npm install
   ```

8. 启动前端服务：

   ```bash
   npm run dev
   ```

   _以下界面说明系统已经成功启动：_

   ![](https://github.com/user-attachments/assets/0daf462c-a24d-4496-a66f-92533534e187)

9. 开发完成后停止 RAGFlow 前端和后端服务：

   ```bash
   pkill -f "ragflow_server.py|task_executor.py"
   ```


## 📚 技术文档

- [Quickstart](https://ragflow.io/docs/dev/)
- [Configuration](https://ragflow.io/docs/dev/configurations)
- [Release notes](https://ragflow.io/docs/dev/release_notes)
- [User guides](https://ragflow.io/docs/dev/category/guides)
- [Developer guides](https://ragflow.io/docs/dev/category/developers)
- [References](https://ragflow.io/docs/dev/category/references)
- [FAQs](https://ragflow.io/docs/dev/faq)

## 📜 路线图

详见 [RAGFlow Roadmap 2025](https://github.com/infiniflow/ragflow/issues/4214) 。

## 🏄 开源社区

- [Discord](https://discord.gg/zd4qPW6t)
- [Twitter](https://twitter.com/infiniflowai)
- [GitHub Discussions](https://github.com/orgs/infiniflow/discussions)

## 🙌 贡献指南

RAGFlow 只有通过开源协作才能蓬勃发展。秉持这一精神,我们欢迎来自社区的各种贡献。如果您有意参与其中,请查阅我们的 [贡献者指南](https://ragflow.io/docs/dev/contributing) 。

## 🤝 商务合作

- [预约咨询](https://aao615odquw.feishu.cn/share/base/form/shrcnjw7QleretCLqh1nuPo1xxh)

## 👥 加入社区

扫二维码添加 RAGFlow 小助手，进 RAGFlow 交流群。

<p align="center">
  <img src="https://github.com/infiniflow/ragflow/assets/7248/bccf284f-46f2-4445-9809-8f1030fb7585" width=50% height=50%>
</p>
