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
import logging
from api import settings
from api.db.services.user_service import TenantService, UserTenantService
from api.utils.file_utils import get_project_base_directory
from rag.llm import EmbeddingModel, CvModel, ChatModel, RerankModel, Seq2txtModel, TTSModel
from api.db import LLMType
from api.db.db_models import DB, LLM, LLMFactories, TenantLLM
from api.db.services.common_service import CommonService
from api.db.services.user_service import TenantService
from rag.llm import ChatModel, CvModel, EmbeddingModel, RerankModel, Seq2txtModel, TTSModel


class LLMFactoriesService(CommonService):
    model = LLMFactories


class LLMService(CommonService):
    """
    封装了对LLM表的操作
    """
    model = LLM


class TenantLLMService(CommonService):
    """
    封装了对TenantLLM表的操作
    提供类方法,创建LLM实例
    """
    model = TenantLLM

    @classmethod
    @DB.connection_context()
    def get_api_key(cls, tenant_id, model_name):
        """根据租户 ID 和模型名称从TenantLLM table 查询模型的配置(不仅仅是api_key)。
        如果模型名称包含 '@' 符号，则将名称分割成两部分，分别对应模型名和工厂名。
        Args:
            tenant_id (_type_): _description_
            model_name (_type_): _description_

        Returns:
            _type_: _description_
        """
        mdlnm, fid = TenantLLMService.split_model_name_and_factory(model_name)
        if not fid:
            objs = cls.query(tenant_id=tenant_id, llm_name=mdlnm)
        else:
            objs = cls.query(tenant_id=tenant_id, llm_name=mdlnm, llm_factory=fid)
        if not objs:
            return
        return objs[0]

    @classmethod
    @DB.connection_context()
    def get_my_llms(cls, tenant_id):
        """查询并返回指定租户所有已配置的 LLM 实例信息，包括模型工厂、标志、标签、模型类型、模型名称和已使用的 token 数量等。

        Args:
            tenant_id (_type_): _description_

        Returns:
            _type_: _description_
        """
        fields = [
            cls.model.llm_factory,
            LLMFactories.logo,
            LLMFactories.tags,
            cls.model.model_type,
            cls.model.llm_name,
            cls.model.used_tokens
        ]
        objs = cls.model.select(*fields).join(LLMFactories, on=(cls.model.llm_factory == LLMFactories.name)).where(
            cls.model.tenant_id == tenant_id, ~cls.model.api_key.is_null()).dicts()

        return list(objs)

    @classmethod
    @DB.connection_context()
    def get_avail_llms(cls, tenant_id):
        """ F8080 获取自己的模型和超级用户的模型。

        Args:
            tenant_id (_type_): _description_

        Returns:
            _type_: _description_
        """
        fields = [
            cls.model.llm_factory,
            LLMFactories.logo,
            LLMFactories.tags,
            cls.model.model_type,
            cls.model.llm_name,
            cls.model.used_tokens
        ]

        super_tenants = UserTenantService.get_tenants_by_is_superuser()
        if len(super_tenants) > 0:
            super_tenant_id = super_tenants[0]['tenant_id']
            objs = cls.model.select(*fields).join(LLMFactories, on=(cls.model.llm_factory == LLMFactories.name)).where(
                (
                    (cls.model.tenant_id == tenant_id)
                    | (cls.model.tenant_id == super_tenant_id)
                )
                & ~cls.model.api_key.is_null()
            ).dicts()
        else:
            objs = cls.model.select(*fields).join(LLMFactories, on=(cls.model.llm_factory == LLMFactories.name)).where(
                cls.model.tenant_id == tenant_id, ~cls.model.api_key.is_null()).dicts()

        return list(objs)


    @staticmethod
    def split_model_name_and_factory(model_name):
        arr = model_name.split("@")
        if len(arr) < 2:
            return model_name, None
        if len(arr) > 2:
            return "@".join(arr[0:-1]), arr[-1]

        # model name must be xxx@yyy
        try:
            model_factories = settings.FACTORY_LLM_INFOS
            model_providers = set([f["name"] for f in model_factories])
            if arr[-1] not in model_providers:
                return model_name, None
            return arr[0], arr[-1]
        except Exception as e:
            logging.exception(f"TenantLLMService.split_model_name_and_factory got exception: {e}")
        return model_name, None

    @classmethod
    @DB.connection_context()
    def get_model_config(cls, tenant_id, llm_type, llm_name=None):
        """
        根据租户 ID、模型类型（如嵌入、语音转文本等）、可选的模型名称和语言创建相应的 LLM 实例。
        这个方法会先尝试从数据库中获取模型配置，如果找不到，则会尝试使用默认值或抛出错误。

        Args:
            tenant_id (int): 租户 ID
            llm_type (str): LLM 类型，例如 EMBEDDING, SPEECH2TEXT 等
            llm_name (str, optional): 模型名称，默认为 None
            lang (str, optional): 语言，默认为 "Chinese"

        Raises:
            LookupError: 租户未找到
            LookupError: 模型类型未设置
            LookupError: 模型未授权

        Returns:
            object: 创建的 LLM 实例,如果是chat model，就返回ChatModel对象
        """
        # 获取租户信息
        e, tenant = TenantService.get_by_id(tenant_id)
        if not e:
            raise LookupError("Tenant not found")

        # 根据 LLM 类型确定模型名称
        if llm_type == LLMType.EMBEDDING.value:
            mdlnm = tenant.embd_id if not llm_name else llm_name
        elif llm_type == LLMType.SPEECH2TEXT.value:
            mdlnm = tenant.asr_id
        elif llm_type == LLMType.IMAGE2TEXT.value:
            mdlnm = tenant.img2txt_id if not llm_name else llm_name
        elif llm_type == LLMType.CHAT.value:
            mdlnm = tenant.llm_id if not llm_name else llm_name
        elif llm_type == LLMType.RERANK:
            mdlnm = tenant.rerank_id if not llm_name else llm_name
        elif llm_type == LLMType.TTS:
            mdlnm = tenant.tts_id if not llm_name else llm_name
        else:
            assert False, "LLM type error"

        # 从 tenant_llm 获取模型配置,不仅仅是api_key
        model_config = cls.get_api_key(tenant_id, mdlnm)
        mdlnm, fid = TenantLLMService.split_model_name_and_factory(mdlnm)

        # 如果找到了配置
        if model_config:
            model_config = model_config.to_dict()

        # 如果 tenant_llm 不存在该模型的配置且为rerank或者embedding,那么从llm表中获取本地的模型（不需要api key）
        if not model_config:
            # 如果是emdedding或者rerank模型
            if llm_type in [LLMType.EMBEDDING, LLMType.RERANK]:
                # 在llm表中查找模型（刚才在tenant_llm表没找到该模型）,fid是大模型厂家名称
                llm = LLMService.query(llm_name=mdlnm) if not fid else LLMService.query(llm_name=mdlnm, fid=fid)
                # 这个是不需要api_key的
                if llm and llm[0].fid in ["Youdao", "FastEmbed", "BAAI"]:
                    model_config = {"llm_factory": llm[0].fid, "api_key": "", "llm_name": mdlnm, "api_base": ""}

            # 如果还没有找到,且模型名称为flag-embedding
            if not model_config:
                if mdlnm == "flag-embedding":
                    model_config = {"llm_factory": "Tongyi-Qianwen", "api_key": "",
                                    "llm_name": llm_name, "api_base": ""}
                else:
                    if not mdlnm:
                        raise LookupError(f"Type of {llm_type} model is not set.")
                    raise LookupError("Model({}) not authorized".format(mdlnm))
        return model_config

    @classmethod
    @DB.connection_context()
    def model_instance(cls, tenant_id, llm_type,
                       llm_name=None, lang="Chinese"):
        """
        根据租户 ID、模型类型（如嵌入、语音转文本等）、可选的模型名称和语言创建相应的 LLM 实例。
        这个方法会先尝试从数据库中获取模型配置，如果找不到，则会尝试使用默认值或抛出错误。

        Args:
            tenant_id (int): 租户 ID
            llm_type (str): LLM 类型，例如 EMBEDDING, SPEECH2TEXT 等
            llm_name (str, optional): 模型名称，默认为 None
            lang (str, optional): 语言，默认为 "Chinese"

        Raises:
            LookupError: 租户未找到
            LookupError: 模型类型未设置
            LookupError: 模型未授权

        Returns:
            object: 创建的 LLM 实例,如果是chat model，就返回ChatModel对象
        """

        model_config = TenantLLMService.get_model_config(tenant_id, llm_type, llm_name)

        if llm_type == LLMType.EMBEDDING.value:
            if model_config["llm_factory"] not in EmbeddingModel:
                return
            return EmbeddingModel[model_config["llm_factory"]](
                model_config["api_key"], model_config["llm_name"], base_url=model_config["api_base"])

        # 创建 rerank 模型实例
        if llm_type == LLMType.RERANK:
            if model_config["llm_factory"] not in RerankModel:
                return
            return RerankModel[model_config["llm_factory"]](
                model_config["api_key"], model_config["llm_name"], base_url=model_config["api_base"])

        # 创建图像转文本模型实例
        if llm_type == LLMType.IMAGE2TEXT.value:
            if model_config["llm_factory"] not in CvModel:
                return
            return CvModel[model_config["llm_factory"]](
                model_config["api_key"], model_config["llm_name"], lang,
                base_url=model_config["api_base"]
            )

        # 创建chat模型实例
        if llm_type == LLMType.CHAT.value:
            if model_config["llm_factory"] not in ChatModel:
                return
            return ChatModel[model_config["llm_factory"]](
                model_config["api_key"], model_config["llm_name"], base_url=model_config["api_base"])

        # 创建语音转文本模型实例
        if llm_type == LLMType.SPEECH2TEXT:
            if model_config["llm_factory"] not in Seq2txtModel:
                return
            return Seq2txtModel[model_config["llm_factory"]](
                key=model_config["api_key"], model_name=model_config["llm_name"],
                lang=lang,
                base_url=model_config["api_base"]
            )

        # 创建文本转语音模型实例
        if llm_type == LLMType.TTS:
            if model_config["llm_factory"] not in TTSModel:
                return
            return TTSModel[model_config["llm_factory"]](
                model_config["api_key"],
                model_config["llm_name"],
                base_url=model_config["api_base"],
            )

    @classmethod
    @DB.connection_context()
    def increase_usage(cls, tenant_id, llm_type, used_tokens, llm_name=None):
        """更新指定租户和模型的已使用 token 数量。
首先确定要更新的模型名称，然后执行数据库更新操作。

        Args:
            tenant_id (_type_): _description_
            llm_type (_type_): _description_
            used_tokens (_type_): _description_
            llm_name (_type_, optional): _description_. Defaults to None.

        Raises:
            LookupError: _description_

        Returns:
            _type_: _description_
        """
        e, tenant = TenantService.get_by_id(tenant_id)
        if not e:
            logging.error(f"Tenant not found: {tenant_id}")
            return 0

        llm_map = {
            LLMType.EMBEDDING.value: tenant.embd_id,
            LLMType.SPEECH2TEXT.value: tenant.asr_id,
            LLMType.IMAGE2TEXT.value: tenant.img2txt_id,
            LLMType.CHAT.value: tenant.llm_id if not llm_name else llm_name,
            LLMType.RERANK.value: tenant.rerank_id if not llm_name else llm_name,
            LLMType.TTS.value: tenant.tts_id if not llm_name else llm_name
        }

        mdlnm = llm_map.get(llm_type)
        if mdlnm is None:
            logging.error(f"LLM type error: {llm_type}")
            return 0

        llm_name, llm_factory = TenantLLMService.split_model_name_and_factory(mdlnm)

        try:
            num = cls.model.update(
                used_tokens=cls.model.used_tokens + used_tokens
            ).where(
                cls.model.tenant_id == tenant_id,
                cls.model.llm_name == llm_name,
                cls.model.llm_factory == llm_factory if llm_factory else True
            ).execute()
        except Exception:
            logging.exception(
                "TenantLLMService.increase_usage got exception,Failed to update used_tokens for tenant_id=%s, llm_name=%s",
                tenant_id, llm_name)
            return 0

        return num

    @classmethod
    @DB.connection_context()
    def get_openai_models(cls):
        """从数据库中查询所有属于 OpenAI 工厂且不是特定嵌入模型的 LLM 实例，并以字典形式返回结果列表。

        Returns:
            _type_: _description_
        """
        objs = cls.model.select().where(
            (cls.model.llm_factory == "OpenAI"),
            ~(cls.model.llm_name == "text-embedding-3-small"),
            ~(cls.model.llm_name == "text-embedding-3-large")
        ).dicts()
        return list(objs)


class LLMBundle:
    """
    从tenant_llm取出模型信息并封装成对象
    """
    def __init__(self, tenant_id, llm_type, llm_name=None, lang="Chinese"):
        """初始化

        Args:
            tenant_id (_type_): _description_
            llm_type (_type_): chat/embedding
            llm_name (_type_, optional): 模型的名称. Defaults to None.
            lang (str, optional): 中文还是英文. Defaults to "Chinese".
        """

        self.tenant_id = tenant_id
        self.llm_type = llm_type
        self.llm_name = llm_name

        # F8080 - 如果本人的模型没有找到，就找超级用户的模型，
        try:
            # 返回factory对象，例如 DeepSeekChat, QWenChat（在chat_model.py中定义的)
            self.mdl = TenantLLMService.model_instance(
                tenant_id, llm_type, llm_name, lang=lang)
            model_config = TenantLLMService.get_model_config(tenant_id, llm_type, llm_name)
        except LookupError as ex:
            # 如果本人的账号找不到模型，就从super user账号找
            super_tenants = UserTenantService.get_tenants_by_is_superuser()
            if len(super_tenants) > 0:
                super_tenant_id = super_tenants[0]['tenant_id']
                self.mdl = TenantLLMService.model_instance(
                    super_tenant_id, llm_type, llm_name, lang=lang)
                model_config = TenantLLMService.get_model_config(super_tenant_id, llm_type, llm_name)

        assert self.mdl, "Can't find model for {}/{}/{}".format(
            tenant_id, llm_type, llm_name)

        # model_config = TenantLLMService.get_model_config(tenant_id, llm_type, llm_name)
        self.max_length = model_config.get("max_tokens", 8192)

    def encode(self, texts: list):
        embeddings, used_tokens = self.mdl.encode(texts)
        if not TenantLLMService.increase_usage(
                self.tenant_id, self.llm_type, used_tokens):
            logging.error(
                "LLMBundle.encode can't update token usage for {}/EMBEDDING used_tokens: {}".format(self.tenant_id, used_tokens))
        return embeddings, used_tokens

    def encode_queries(self, query: str):
        emd, used_tokens = self.mdl.encode_queries(query)
        if not TenantLLMService.increase_usage(
                self.tenant_id, self.llm_type, used_tokens):
            logging.error(
                "LLMBundle.encode_queries can't update token usage for {}/EMBEDDING used_tokens: {}".format(self.tenant_id, used_tokens))
        return emd, used_tokens

    def similarity(self, query: str, texts: list):
        sim, used_tokens = self.mdl.similarity(query, texts)
        if not TenantLLMService.increase_usage(
                self.tenant_id, self.llm_type, used_tokens):
            logging.error(
                "LLMBundle.similarity can't update token usage for {}/RERANK used_tokens: {}".format(self.tenant_id, used_tokens))
        return sim, used_tokens

    def describe(self, image, max_tokens=300):
        txt, used_tokens = self.mdl.describe(image, max_tokens)
        if not TenantLLMService.increase_usage(
                self.tenant_id, self.llm_type, used_tokens):
            logging.error(
                "LLMBundle.describe can't update token usage for {}/IMAGE2TEXT used_tokens: {}".format(self.tenant_id, used_tokens))
        return txt

    def describe_with_prompt(self, image, prompt):
        txt, used_tokens = self.mdl.describe_with_prompt(image, prompt)
        if not TenantLLMService.increase_usage(
                self.tenant_id, self.llm_type, used_tokens):
            logging.error(
                "LLMBundle.describe can't update token usage for {}/IMAGE2TEXT used_tokens: {}".format(self.tenant_id, used_tokens))
        return txt

    def transcription(self, audio):
        txt, used_tokens = self.mdl.transcription(audio)
        if not TenantLLMService.increase_usage(
                self.tenant_id, self.llm_type, used_tokens):
            logging.error(
                "LLMBundle.transcription can't update token usage for {}/SEQUENCE2TXT used_tokens: {}".format(self.tenant_id, used_tokens))
        return txt

    def tts(self, text):
        for chunk in self.mdl.tts(text):
            if isinstance(chunk, int):
                if not TenantLLMService.increase_usage(
                        self.tenant_id, self.llm_type, chunk, self.llm_name):
                    logging.error(
                        "LLMBundle.tts can't update token usage for {}/TTS".format(self.tenant_id))
                return
            yield chunk

    def chat(self, system, history, gen_conf):
        txt, used_tokens = self.mdl.chat(system, history, gen_conf)
        if isinstance(txt, int) and not TenantLLMService.increase_usage(
                self.tenant_id, self.llm_type, used_tokens, self.llm_name):
            logging.error(
                "LLMBundle.chat can't update token usage for {}/CHAT llm_name: {}, used_tokens: {}".format(self.tenant_id, self.llm_name,
                                                                                                           used_tokens))
        return txt

    def chat_streamly(self, system, history, gen_conf):
        for txt in self.mdl.chat_streamly(system, history, gen_conf):
            if isinstance(txt, int):
                # F8080 - 如果返回的是int，这是消耗的token数，在此注释token更新代码
                # if not TenantLLMService.increase_usage(
                #         self.tenant_id, self.llm_type, txt, self.llm_name):
                #     logging.error(
                #         "LLMBundle.chat_streamly can't update token usage for {}/CHAT llm_name: {}, content: {}".format(self.tenant_id, self.llm_name,
                #                                                                                                   txt))
                return
            yield txt
