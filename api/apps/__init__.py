#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import os
import sys
import logging
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from flask import Blueprint, Flask
from werkzeug.wrappers.request import Request
from flask_cors import CORS
from flasgger import Swagger
from itsdangerous.url_safe import URLSafeTimedSerializer as Serializer

from api.db import StatusEnum
from api.db.db_models import close_connection
from api.db.services import UserService
from api.utils import CustomJSONEncoder, commands

from flask_session import Session
from flask_login import LoginManager
from api import settings
from api.utils.api_utils import server_error_response
from api.constants import API_VERSION

__all__ = ["app"]

Request.json = property(lambda self: self.get_json(force=True, silent=True))

app = Flask(__name__)

# Add this at the beginning of your file to configure Swagger UI
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,  # Include all endpoints
            "model_filter": lambda tag: True,  # Include all models
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/",
}

swagger = Swagger(
    app,
    config=swagger_config,
    template={
        "swagger": "2.0",
        "info": {
            "title": "RAGFlow API",
            "description": "",
            "version": "1.0.0",
        },
        "securityDefinitions": {
            "ApiKeyAuth": {"type": "apiKey", "name": "Authorization", "in": "header"}
        },
    },
)

CORS(app, supports_credentials=True, max_age=2592000)
app.url_map.strict_slashes = False
app.json_encoder = CustomJSONEncoder
app.errorhandler(Exception)(server_error_response)

## convince for dev and debug
# app.config["LOGIN_DISABLED"] = True
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("MAX_CONTENT_LENGTH", 128 * 1024 * 1024)
)

Session(app)
login_manager = LoginManager()
login_manager.init_app(app)

commands.register_commands(app)


def search_pages_path(pages_dir):
    app_path_list = [
        path for path in pages_dir.glob("*_app.py") if not path.name.startswith(".")
    ]
    api_path_list = [
        path for path in pages_dir.glob("*sdk/*.py") if not path.name.startswith(".")
    ]
    app_path_list.extend(api_path_list)
    return app_path_list


def register_page(page_path):
    path = f'{page_path}'

    page_name = page_path.stem.rstrip('_app')
    module_name = '.'.join(page_path.parts[page_path.parts.index('api'):-1] + (page_name,))

    # 动态加载模块
    spec = spec_from_file_location(module_name, page_path)
    page = module_from_spec(spec)

    # 将 Flask 应用实例和 Blueprint 实例绑定到模块
    page.app = app
    page.manager = Blueprint(page_name, module_name)

    # 将模块注册到系统模块列表中
    sys.modules[module_name] = page

    # 执行模块中的代码
    spec.loader.exec_module(page)
    page_name = getattr(page, 'page_name', page_name)
    url_prefix = f'/api/{API_VERSION}' if "/sdk/" in path else f'/{API_VERSION}/{page_name}'

    # 注册 Blueprint 到 Flask 应用中
    app.register_blueprint(page.manager, url_prefix=url_prefix)
    # 返回注册的 URL 前缀
    return url_prefix


# 定义需要搜索的目录路径
pages_dir = [
    # 当前文件所在目录
    Path(__file__).parent,
    Path(__file__).parent.parent / 'api' / 'apps',
    Path(__file__).parent.parent / 'api' / 'apps' / 'sdk',
]

# 动态注册所有找到的模块文件，并收集返回的 URL 前缀
client_urls_prefix = [
    register_page(path) for dir in pages_dir for path in search_pages_path(dir)
]


@login_manager.request_loader
def load_user(web_request):
    """jwt令牌解码为用户access_token
    """
    jwt = Serializer(secret_key=settings.SECRET_KEY)
    authorization = web_request.headers.get("Authorization")
    if authorization:
        try:
            """
            # 如果绑定了过期时间,就要这样写
            payload = jwt.loads(authorization)
            if 'access_token' in payload and 'exp' in payload and payload['exp'] > int(time.time()):
                user = UserService.query(access_token=access_token, status=StatusEnum.VALID.value)
                if user:
                    return user[0]    
                else:
                    return None                            
            """
            access_token = str(jwt.loads(authorization))
            user = UserService.query(
                access_token=access_token, status=StatusEnum.VALID.value
            )
            if user:
                return user[0]
            else:
                return None
        except Exception:
            logging.exception("load_user got exception")
            return None
    else:
        return None


@app.teardown_request
def _db_close(exc):
    close_connection()
