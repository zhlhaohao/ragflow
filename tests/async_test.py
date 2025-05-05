import asyncio


# 异步函数，周期性生成消息
async def async_producer():
    for i in range(5):
        await asyncio.sleep(1)  # 模拟周期性任务
        yield f"Message {i}"


# 将异步生成器转换为同步生成器的适配器
def async_to_sync_gen(async_gen):
    # policy = asyncio.get_event_loop_policy()
    # 创建新的事件循环
    # policy = asyncio.DefaultEventLoopPolicy()
    # asyncio.set_event_loop_policy(policy)

    # 获取新的事件循环
    # loop = asyncio.get_event_loop_policy().new_event_loop()
    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        while True:
            try:
                # 运行事件循环直到获取下一个消息
                msg = loop.run_until_complete(async_gen.__anext__())
                yield msg
            except StopAsyncIteration:
                break
    finally:
        loop.close()


# 普通同步函数，调用异步生产者并传递消息
def sync_producer():
    # 创建异步生成器实例
    async_gen = async_producer()
    # 将异步生成器转换为同步生成器
    yield from async_to_sync_gen(async_gen)


# 使用示例
if __name__ == "__main__":
    print("=== First run ===")
    for msg in sync_producer():
        print(msg)

    print("\n=== Second run ===")
    for msg in sync_producer():
        print(msg)
