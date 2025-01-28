# base 阶段
FROM ubuntu:22.04 AS base
USER root
SHELL ["/bin/bash", "-c"]

# 定义构建参数以自定义构建过程
ARG NEED_MIRROR=0
ARG LIGHTEN=0
ENV LIGHTEN=${LIGHTEN}

# 设置工作目录
WORKDIR /ragflow

# 创建模型文件存放的路径
RUN mkdir -p /ragflow/rag/res/deepdoc /root/.ragflow

# 将infiniflow/ragflow_deps:latest镜像里面保存的模型权重文件拷贝到 /ragflow/rag/res ， 这些是deepdoc的模型权重文件
RUN --mount=type=bind,from=infiniflow/ragflow_deps:latest,source=/huggingface.co,target=/huggingface.co \
    cp /huggingface.co/InfiniFlow/huqie/huqie.txt.trie /ragflow/rag/res/ && \
    tar --exclude='.*' -cf - \
        /huggingface.co/InfiniFlow/text_concat_xgb_v1.0 \
        /huggingface.co/InfiniFlow/deepdoc \
        | tar -xf - --strip-components=3 -C /ragflow/rag/res/deepdoc 

# 将infiniflow/ragflow_deps:latest镜像里面保存的模型权重文件拷贝到 /root/.ragflow，这些是ntlk、embedding、reranking的模型权重文件
RUN --mount=type=bind,from=infiniflow/ragflow_deps:latest,source=/huggingface.co,target=/huggingface.co \
    if [ "$LIGHTEN" != "1" ]; then \
        (tar -cf - \
            /huggingface.co/BAAI/bge-large-zh-v1.5 \
            /huggingface.co/BAAI/bge-reranker-v2-m3 \
            /huggingface.co/maidalun1020/bce-embedding-base_v1 \
            /huggingface.co/maidalun1020/bce-reranker-base_v1 \
            | tar -xf - --strip-components=2 -C /root/.ragflow) \
    fi

# https://github.com/chrismattmann/tika-python
# 这是唯一可以在没有互联网访问的情况下运行 python-tika 的方法。如果不设置，每次都会检查 tika 版本并从 Apache 拉取最新版本。
RUN --mount=type=bind,from=infiniflow/ragflow_deps:latest,source=/,target=/deps \
    cp -r /deps/nltk_data /root/ && \
    cp /deps/tika-server-standard-3.0.0.jar /deps/tika-server-standard-3.0.0.jar.md5 /ragflow/ && \
    cp /deps/cl100k_base.tiktoken /ragflow/9b5ad71b2ce5302211f9c61530b329a4922fc6a4

ENV TIKA_SERVER_JAR="file:///ragflow/tika-server-standard-3.0.0.jar"
ENV DEBIAN_FRONTEND=noninteractive

# 设置 apt 源和安装依赖项
# Python 包及其隐式依赖项：
# opencv-python: libglib2.0-0 libglx-mesa0 libgl1
# aspose-slides: pkg-config libicu-dev libgdiplus         libssl1.1_1.1.1f-1ubuntu2_amd64.deb
# python-pptx:   default-jdk                              tika-server-standard-3.0.0.jar
# selenium:      libatk-bridge2.0-0                       chrome-linux64-121-0-6167-85
# 构建 C 扩展：libpython3-dev libgtk-4-1 libnss3 xdg-utils libgbm-dev
RUN --mount=type=cache,id=ragflow_apt,target=/var/cache/apt,sharing=locked \
    if [ "$NEED_MIRROR" == "1" ]; then \
        sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list; \
    fi; \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache && \
    chmod 1777 /tmp && \
    apt update && \
    apt --no-install-recommends install -y ca-certificates && \
    apt update && \
    apt install -y libglib2.0-0 libglx-mesa0 libgl1 && \
    apt install -y pkg-config libicu-dev libgdiplus && \
    apt install -y default-jdk && \
    apt install -y libatk-bridge2.0-0 && \
    apt install -y libpython3-dev libgtk-4-1 libnss3 xdg-utils libgbm-dev && \
    apt install -y python3-pip pipx nginx unzip curl wget git vim less

# 如果需要使用镜像源，则配置 pip 使用清华大学的镜像源，并安装 uv 工具
RUN if [ "$NEED_MIRROR" == "1" ]; then \
        pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
        pip3 config set global.trusted-host pypi.tuna.tsinghua.edu.cn; \
        mkdir -p /etc/uv && \
        echo "[[index]]" > /etc/uv/uv.toml && \
        echo 'url = "https://pypi.tuna.tsinghua.edu.cn/simple"' >> /etc/uv/uv.toml && \
        echo "default = true" >> /etc/uv/uv.toml; \
    fi; \
    pipx install uv

ENV PYTHONDONTWRITEBYTECODE=1 DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
ENV PATH=/root/.local/bin:$PATH

# 更新 nodejs 到较新版本（Ubuntu 22.04 自带的 nodejs 12.22 版本过旧）
RUN --mount=type=cache,id=ragflow_apt,target=/var/cache/apt,sharing=locked \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt purge -y nodejs npm && \
    apt autoremove && \
    apt update && \
    apt install -y nodejs cargo 

# 添加 mssql ODBC 驱动
# macOS ARM64 环境下安装 msodbcsql18。
# 一般 x86_64 环境下安装 msodbcsql17。
RUN --mount=type=cache,id=ragflow_apt,target=/var/cache/apt,sharing=locked \
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list > /etc/apt/sources.list.d/mssql-release.list && \
    apt update && \
    if [ -n "$ARCH" ] && [ "$ARCH" = "arm64" ]; then \
        # MacOS ARM64 
        ACCEPT_EULA=Y apt install -y unixodbc-dev msodbcsql18; \
    else \
        # (x86_64)
        ACCEPT_EULA=Y apt install -y unixodbc-dev msodbcsql17; \
    fi || \
    { echo "Failed to install ODBC driver"; exit 1; }

# 添加 selenium 依赖项
RUN --mount=type=bind,from=infiniflow/ragflow_deps:latest,source=/chrome-linux64-121-0-6167-85,target=/chrome-linux64.zip \
    unzip /chrome-linux64.zip && \
    mv chrome-linux64 /opt/chrome && \
    ln -s /opt/chrome/chrome /usr/local/bin/
RUN --mount=type=bind,from=infiniflow/ragflow_deps:latest,source=/chromedriver-linux64-121-0-6167-85,target=/chromedriver-linux64.zip \
    unzip -j /chromedriver-linux64.zip chromedriver-linux64/chromedriver && \
    mv chromedriver /usr/local/bin/ && \
    rm -f /usr/bin/google-chrome

# 安装特定架构的 libssl1.1
# https://forum.aspose.com/t/aspose-slides-for-net-no-usable-version-of-libssl-found-with-linux-server/271344/13
# aspose-slides 在 linux/arm64 上不可用
RUN --mount=type=bind,from=infiniflow/ragflow_deps:latest,source=/,target=/deps \
    if [ "$(uname -m)" = "x86_64" ]; then \
        dpkg -i /deps/libssl1.1_1.1.1f-1ubuntu2_amd64.deb; \
    elif [ "$(uname -m)" = "aarch64" ]; then \
        dpkg -i /deps/libssl1.1_1.1.1f-1ubuntu2_arm64.deb; \
    fi


# builder 阶段
FROM base AS builder
USER root

WORKDIR /ragflow

# 安装依赖项（根据 uv.lock 文件）
COPY pyproject.toml uv.lock ./

# 安装python 依赖库 pip install
RUN --mount=type=cache,id=ragflow_uv,target=/root/.cache/uv,sharing=locked \
    if [ "$LIGHTEN" == "1" ]; then \
        uv sync --python 3.10 --frozen; \
    else \
        uv sync --python 3.10 --frozen --all-extras; \
    fi

# 复制 web 和 docs 目录
COPY web web
COPY docs docs
RUN --mount=type=cache,id=ragflow_npm,target=/root/.npm,sharing=locked \
    cd web && npm install && npm run build

# 复制 Git 信息并生成版本信息
COPY .git /ragflow/.git

RUN version_info=$(git describe --tags --match=v* --first-parent --always); \
    if [ "$LIGHTEN" == "1" ]; then \
        version_info="$version_info slim"; \
    else \
        version_info="$version_info full"; \
    fi; \
    echo "RAGFlow 版本: $version_info"; \
    echo $version_info > /ragflow/VERSION

# production 阶段
FROM base AS production
USER root

WORKDIR /ragflow

# 复制 Python 环境和包
ENV VIRTUAL_ENV=/ragflow/.venv
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

ENV PYTHONPATH=/ragflow/

# 复制项目代码
COPY web web
COPY api api
COPY conf conf
COPY deepdoc deepdoc
COPY rag rag
COPY agent agent
COPY graphrag graphrag
COPY pyproject.toml uv.lock ./

# 复制服务配置模板和入口脚本
COPY docker/service_conf.yaml.template ./conf/service_conf.yaml.template
COPY docker/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# 复制编译后的网页文件
COPY --from=builder /ragflow/web/dist /ragflow/web/dist

# 复制版本信息文件
COPY --from=builder /ragflow/VERSION /ragflow/VERSION
ENTRYPOINT ["./entrypoint.sh"]