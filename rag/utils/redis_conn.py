import logging
import json

import valkey as redis
from rag import settings
from rag.utils import singleton

# redis的任务队列消息体
class Payload:
    """
    用于封装从 Redis 流中读取的消息及其相关元数据，并提供确认消息和获取消息内容的方法。下面是对该类的详细解释和一些改进建议：

    1. **初始化方法 `__init__`**：
    - `consumer`：Redis 客户端实例，用于执行 Redis 命令。
    - `queue_name`：队列名称。
    - `group_name`：消费者组名称。
    - `msg_id`：消息 ID。
    - `message`：消息内容，是一个字典，其中包含键 `'message'`，其值为 JSON 字符串。
    - `self.__message = json.loads(message['message'])`：将消息内容从 JSON 字符串解析为 Python 字典。

    2. **确认方法 `ack`**：
    - 尝试使用 `xack` 方法确认消息已被处理。
    - 如果成功，返回 `True`。
    - 如果在确认过程中发生异常，记录警告日志并返回 `False`。

    3. **获取消息方法 `get_message`**：
    - 返回解析后的消息内容。
    """
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

    def queue_product(self, queue, message, exp=settings.SVR_QUEUE_RETENTION) -> bool:
        """
        您提供的 `queue_product` 方法用于将消息推送到 Redis 的流（stream）队列中。该方法使用 Redis 的管道功能来确保操作的原子性，并在遇到异常时尝试重新连接。下面是对该方法的详细解释和一些改进建议：

        1. **方法参数**：
        - `queue`：要推送消息的队列名称。
        - `message`：要推送的消息内容。
        - `exp`：队列的过期时间，默认值为 `settings.SVR_QUEUE_RETENTION`。

        2. **重试机制**：
        - `for _ in range(3)`：最多尝试 3 次。

        3. **构建消息负载**：
        - `payload = {"message": json.dumps(message)}`：将消息内容序列化为 JSON 字符串，并构建负载。

        4. **创建管道**：
        - `pipeline = self.REDIS.pipeline()`：创建一个 Redis 管道，确保操作的原子性。

        5. **向队列中添加消息**：
        - `pipeline.xadd(queue, payload)`：使用 `xadd` 方法将消息添加到队列中。

        6. **执行管道**：
        - `pipeline.execute()`：执行管道中的所有操作。

        7. **异常处理**：
        - `except Exception as e`：捕获在执行过程中可能发生的任何异常。
        - `print(e)`：打印异常信息。
        - `logging.warning(f"[EXCEPTION] producer {queue} || {e}")`：记录包含队列名称和异常信息的警告日志。
        - `self.__open__()`：调用 `__open__` 方法重新尝试打开 Redis 连接。

        8. **返回值**：
        - 如果成功推送消息，返回 `True`。
        - 如果在多次重试后仍然失败，返回 `False`。
        """
        for _ in range(3):
            try:
                payload = {"message": json.dumps(message)}
                pipeline = self.REDIS.pipeline()
                pipeline.xadd(queue, payload)
                # pipeline.expire(queue, exp)
                pipeline.execute()
                return True
            except Exception as e:
                logging.exception(
                    "RedisDB.queue_product " + str(queue) + " got exception: " + str(e)
                )
        return False

    def queue_consumer(self, queue_name, group_name, consumer_name, msg_id=b">") -> Payload:
        """
        用于从 Redis 流（stream）队列中消费消息。该方法首先检查指定的消费者组是否存在，如果不存在则创建该组，然后从队列中读取消息。如果在读取消息过程中发生异常，它会记录警告日志。下面是对该方法的详细解释和一些改进建议：

        1. **Payload 类**：
        - `Payload` 类用于封装从 Redis 流中读取的消息及其相关元数据。
        - `ack` 方法用于确认消息已被处理，调用 `xack` 方法将消息从待处理列表中移除。

        2. **方法参数**：
        - `queue_name`：队列名称。
        - `group_name`：消费者组名称。
        - `consumer_name`：消费者名称。
        - `msg_id`：起始消息 ID，默认为 `b">"`，表示从最新的消息开始读取。

        3. **检查消费者组**：
        - `group_info = self.REDIS.xinfo_groups(queue_name)`：获取队列的所有消费者组信息。
        - `if not any(e["name"] == group_name for e in group_info)`：检查指定的消费者组是否存在，如果不存在则创建该组。
        - `self.REDIS.xgroup_create(queue_name, group_name, id="0", mkstream=True)`：创建消费者组，`id="0"` 表示从最早的可用消息开始读取，`mkstream=True` 表示如果队列不存在则创建队列。

        4. **读取消息**：
        - `args = {...}`：构建 `xreadgroup` 方法的参数。
        - `messages = self.REDIS.xreadgroup(**args)`：从队列中读取消息，阻塞等待 10000 毫秒（10 秒）。
        - `if not messages`：如果没有消息，返回 `None`。
        - `stream, element_list = messages[0]`：解析读取到的消息。
        - `msg_id, payload = element_list[0]`：提取消息 ID 和负载。
        - `res = Payload(self.REDIS, queue_name, group_name, msg_id, payload)`：创建 `Payload` 实例并返回。

        5. **异常处理**：
        - `except Exception as e`：捕获在执行过程中可能发生的任何异常。
        - `if 'key' in str(e): pass`：忽略包含 "key" 的异常，可能是由于队列不存在等正常情况。
        - `else: logging.warning(f"[EXCEPTION] consumer: {queue_name} || {e}")`：记录包含队列名称和异常信息的警告日志。

        6. **返回值**：
        - 如果成功读取消息，返回 `Payload` 实例。
        - 如果在多次重试后仍然失败，返回 `None`。
        """
        try:
            group_info = self.REDIS.xinfo_groups(queue_name)
            if not any(e["name"] == group_name for e in group_info):
                self.REDIS.xgroup_create(queue_name, group_name, id="0", mkstream=True)
            args = {
                "groupname": group_name,
                "consumername": consumer_name,
                "count": 1,
                "block": 10000,
                "streams": {queue_name: msg_id},
            }
            messages = self.REDIS.xreadgroup(**args)
            if not messages:
                return None
            stream, element_list = messages[0]
            msg_id, payload = element_list[0]
            res = Payload(self.REDIS, queue_name, group_name, msg_id, payload)
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

    def get_unacked_for(self, consumer_name, queue_name, group_name):
        """
        用于获取指定消费者组中未确认的消息。该方法首先检查指定的消费者组是否存在，如果存在则查询未确认的消息，并返回 `Payload` 实例。如果在执行过程中发生异常，它会记录警告日志并重新尝试打开 Redis 连接：

        1. **方法参数**：
        - `consumer_name`：消费者名称。
        - `queue_name`：队列名称。
        - `group_name`：消费者组名称。

        2. **检查消费者组**：
        - `group_info = self.REDIS.xinfo_groups(queue_name)`：获取队列的所有消费者组信息。
        - `if not any(e["name"] == group_name for e in group_info)`：检查指定的消费者组是否存在，如果不存在则返回 `None`。

        3. **查询未确认的消息**：
        - `pendings = self.REDIS.xpending_range(queue_name, group_name, min=0, max=10000000000000, count=1, consumername=consumer_name)`：查询指定消费者组中未确认的消息，限制返回一条消息。
        - `if not pendings`：如果没有未确认的消息，返回 `None`。
        - `msg_id = pendings[0]["message_id"]`：提取消息 ID。

        4. **获取消息内容**：
        - `msg = self.REDIS.xrange(queue_name, min=msg_id, count=1)`：从队列中获取指定消息 ID 的消息。
        - `if not msg`：如果没有消息，返回 `None`。
        - `_, payload = msg[0]`：提取消息内容。
        - `return Payload(self.REDIS, queue_name, group_name, msg_id, payload)`：创建 `Payload` 实例并返回。

        5. **异常处理**：
        - `except Exception as e`：捕获在执行过程中可能发生的任何异常。
        - `if 'key' in str(e): return`：忽略包含 "key" 的异常，可能是由于队列不存在等正常情况。
        - `else: logging.warning(f"[EXCEPTION] xpending_range: {consumer_name} || {e}")`：记录包含消费者名称和异常信息的警告日志。
        - `self.__open__()`：调用 `__open__` 方法重新尝试打开 Redis 连接。

        6. **返回值**：
        - 如果成功获取未确认的消息，返回 `Payload` 实例。
        - 如果在多次重试后仍然失败，返回 `None`。
        """
        try:
            group_info = self.REDIS.xinfo_groups(queue_name)
            if not any(e["name"] == group_name for e in group_info):
                return
            pendings = self.REDIS.xpending_range(
                queue_name,
                group_name,
                min=0,
                max=10000000000000,
                count=1,
                consumername=consumer_name,
            )
            if not pendings:
                return
            msg_id = pendings[0]["message_id"]
            msg = self.REDIS.xrange(queue_name, min=msg_id, count=1)
            _, payload = msg[0]
            return Payload(self.REDIS, queue_name, group_name, msg_id, payload)
        except Exception as e:
            if "key" in str(e):
                return
            logging.exception(
                "RedisDB.get_unacked_for " + consumer_name + " got exception: " + str(e)
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
