#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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
import json
import uuid

import valkey as redis
from rag import settings
from rag.utils import singleton
from valkey.lock import Lock

class RedisMsg:
    def __init__(self, consumer, queue_name, group_name, msg_id, message):
        self.__consumer = consumer
        self.__queue_name = queue_name
        self.__group_name = group_name
        self.__msg_id = msg_id
        self.__message = json.loads(message["message"])

    def ack(self):
        try:
            self.__consumer.xack(self.__queue_name, self.__group_name, self.__msg_id)
            return True
        except Exception as e:
            logging.warning("[EXCEPTION]ack" + str(self.__queue_name) + "||" + str(e))
        return False

    def get_message(self):
        return self.__message

    def get_msg_id(self):
        return self.__msg_id


@singleton
class RedisDB:
    def __init__(self):
        self.REDIS = None
        self.config = settings.REDIS
        self.__open__()

    def __open__(self):
        try:
            self.REDIS = redis.StrictRedis(
                host=self.config["host"].split(":")[0],
                port=int(self.config.get("host", ":6379").split(":")[1]),
                db=int(self.config.get("db", 1)),
                password=self.config.get("password"),
                decode_responses=True,
            )
        except Exception:
            logging.warning("Redis can't be connected.")
        return self.REDIS

    def health(self):
        self.REDIS.ping()
        a, b = "xx", "yy"
        self.REDIS.set(a, b, 3)

        if self.REDIS.get(a) == b:
            return True

    def is_alive(self):
        return self.REDIS is not None

    def exist(self, k):
        if not self.REDIS:
            return
        try:
            return self.REDIS.exists(k)
        except Exception as e:
            logging.warning("RedisDB.exist " + str(k) + " got exception: " + str(e))
            self.__open__()

    def get(self, k):
        """
        用于从 Redis 中获取键值的方法。该方法首先检查 self.REDIS 是否已初始化，然后尝试从 Redis 中获取指定键的值。如果在获取过程中发生异常，它会记录警告日志并重新尝试打开 Redis 连接
        """        
        if not self.REDIS:
            return
        try:
            return self.REDIS.get(k)
        except Exception as e:
            logging.warning("RedisDB.get " + str(k) + " got exception: " + str(e))
            self.__open__()

    def set_obj(self, k, obj, exp=3600):
        """
        用于将一个 Python 对象序列化为 JSON 字符串，并将其存储到 Redis 中。如果在设置过程中发生异常，它会记录警告日志并重新尝试打开 Redis 连接。
        """
        try:
            self.REDIS.set(k, json.dumps(obj, ensure_ascii=False), exp)
            return True
        except Exception as e:
            logging.warning("RedisDB.set_obj " + str(k) + " got exception: " + str(e))
            self.__open__()
        return False

    def set(self, k, v, exp=3600):
        try:
            self.REDIS.set(k, v, exp)
            return True
        except Exception as e:
            logging.warning("RedisDB.set " + str(k) + " got exception: " + str(e))
            self.__open__()
        return False

    def sadd(self, key: str, member: str):
        try:
            self.REDIS.sadd(key, member)
            return True
        except Exception as e:
            logging.warning("RedisDB.sadd " + str(key) + " got exception: " + str(e))
            self.__open__()
        return False

    def srem(self, key: str, member: str):
        try:
            self.REDIS.srem(key, member)
            return True
        except Exception as e:
            logging.warning("RedisDB.srem " + str(key) + " got exception: " + str(e))
            self.__open__()
        return False

    def smembers(self, key: str):
        try:
            res = self.REDIS.smembers(key)
            return res
        except Exception as e:
            logging.warning(
                "RedisDB.smembers " + str(key) + " got exception: " + str(e)
            )
            self.__open__()
        return None

    def zadd(self, key: str, member: str, score: float):
        try:
            self.REDIS.zadd(key, {member: score})
            return True
        except Exception as e:
            logging.warning("RedisDB.zadd " + str(key) + " got exception: " + str(e))
            self.__open__()
        return False

    def zcount(self, key: str, min: float, max: float):
        try:
            res = self.REDIS.zcount(key, min, max)
            return res
        except Exception as e:
            logging.warning("RedisDB.zcount " + str(key) + " got exception: " + str(e))
            self.__open__()
        return 0

    def zpopmin(self, key: str, count: int):
        try:
            res = self.REDIS.zpopmin(key, count)
            return res
        except Exception as e:
            logging.warning("RedisDB.zpopmin " + str(key) + " got exception: " + str(e))
            self.__open__()
        return None

    def zrangebyscore(self, key: str, min: float, max: float):
        try:
            res = self.REDIS.zrangebyscore(key, min, max)
            return res
        except Exception as e:
            logging.warning(
                "RedisDB.zrangebyscore " + str(key) + " got exception: " + str(e)
            )
            self.__open__()
        return None

    def transaction(self, key, value, exp=3600):
        """
        用于在 Redis 中执行一个事务，确保多个操作要么全部成功，要么全部失败。该方法使用 Redis 的管道功能来实现事务。如果在执行过程中发生异常，它会记录警告日志并重新尝试打开 Redis 连接。下面是对该方法的详细解释和一些改进建议：
        1. **方法参数**：
        - `key`：要设置的键。
        - `value`：要设置的值。
        - `exp`：过期时间（秒），默认为 3600 秒（1 小时）。

        2. **创建管道**：
        - `pipeline = self.REDIS.pipeline(transaction=True)`：创建一个 Redis 管道，确保操作在一个事务中执行。

        3. **设置键值**：
        - `pipeline.set(key, value, ex=exp, nx=True)`：将键值对设置到 Redis 中，并设置过期时间和 `nx` 选项（只有在键不存在时才设置）。

        4. **执行管道**：
        - `pipeline.execute()`：执行管道中的所有操作。

        5. **异常处理**：
        - `except Exception as e`：捕获在执行事务过程中可能发生的任何异常。
        - `logging.warning(f"[EXCEPTION] set {key} || {e}")`：记录包含键值和异常信息的警告日志。
        - `self.__open__()`：调用 `__open__` 方法重新尝试打开 Redis 连接。

        6. **返回值**：
        - 如果事务成功，返回 `True`。
        - 如果事务失败，返回 `False`。
        """
        try:
            pipeline = self.REDIS.pipeline(transaction=True)
            pipeline.set(key, value, exp, nx=True)
            pipeline.execute()
            return True
        except Exception as e:
            logging.warning(
                "RedisDB.transaction " + str(key) + " got exception: " + str(e)
            )
            self.__open__()
        return False

    def queue_product(self, queue, message) -> bool:
        for _ in range(3):
            try:
                payload = {"message": json.dumps(message)}
                self.REDIS.xadd(queue, payload)
                return True
            except Exception as e:
                logging.exception(
                    "RedisDB.queue_product " + str(queue) + " got exception: " + str(e)
                )
        return False

    def queue_consumer(self, queue_name, group_name, consumer_name, msg_id=b">") -> RedisMsg:
        """https://redis.io/docs/latest/commands/xreadgroup/"""
        try:
            group_info = self.REDIS.xinfo_groups(queue_name)
            if not any(e["name"] == group_name for e in group_info):
                self.REDIS.xgroup_create(queue_name, group_name, id="0", mkstream=True)
            args = {
                "groupname": group_name,
                "consumername": consumer_name,
                "count": 1,
                "block": 5,
                "streams": {queue_name: msg_id},
            }
            messages = self.REDIS.xreadgroup(**args)
            if not messages:
                return None
            stream, element_list = messages[0]
            if not element_list:
                return None
            msg_id, payload = element_list[0]
            res = RedisMsg(self.REDIS, queue_name, group_name, msg_id, payload)
            return res
        except Exception as e:
            if "key" in str(e):
                pass
            else:
                logging.exception(
                    "RedisDB.queue_consumer "
                    + str(queue_name)
                    + " got exception: "
                    + str(e)
                )
        return None

    def get_unacked_iterator(self, queue_names: list[str], group_name, consumer_name):
        try:
            for queue_name in queue_names:
                group_info = self.REDIS.xinfo_groups(queue_name)
                if not any(e["name"] == group_name for e in group_info):
                    continue
                current_min = 0
                while True:
                    payload = self.queue_consumer(queue_name, group_name, consumer_name, current_min)
                    if not payload:
                        break
                    current_min = payload.get_msg_id()
                    logging.info(f"RedisDB.get_unacked_iterator {consumer_name} msg_id {current_min}")
                    yield payload
        except Exception as e:
            if "key" in str(e):
                return
            logging.exception(
                "RedisDB.get_unacked_iterator " + consumer_name + " got exception: "
            )
            self.__open__()

    def queue_info(self, queue, group_name) -> dict | None:
        try:
            groups = self.REDIS.xinfo_groups(queue)
            for group in groups:
                if group["name"] == group_name:
                    return group
        except Exception as e:
            logging.warning(
                "RedisDB.queue_info " + str(queue) + " got exception: " + str(e)
            )
        return None


REDIS_CONN = RedisDB()


class RedisDistributedLock:
    def __init__(self, lock_key, lock_value=None, timeout=10, blocking_timeout=1):
        self.lock_key = lock_key
        if lock_value:
            self.lock_value = lock_value
        else:
            self.lock_value = str(uuid.uuid4())
        self.timeout = timeout
        self.lock = Lock(REDIS_CONN.REDIS, lock_key, timeout=timeout, blocking_timeout=blocking_timeout)

    def acquire(self):
        return self.lock.acquire()

    def release(self):
        return self.lock.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.release()