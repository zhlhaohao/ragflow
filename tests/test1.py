import asyncio
import threading
import queue

def A():
    message_queue = queue.Queue()
    result_container = [None]  # 使用列表来共享结果，因为 nonlocal 在嵌套函数中可能有限制

    async def run_b():
        # 执行异步函数B并获取结果
        result = await B(message_queue)
        result_container[0] = result
        # 放入结束标记
        message_queue.put(None)

    def start_event_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_b())

    # 启动异步事件循环的线程
    thread = threading.Thread(target=start_event_loop)
    thread.start()

    while True:
        try:
            msg = message_queue.get(timeout=0.1)
            if msg is None:
                break  # 收到结束信号
            yield msg
        except queue.Empty:
            # 检查线程是否还在运行
            if not thread.is_alive():
                break

    thread.join()
    print("B returned:", result_container[0])

async def B(queue):
    for i in range(3):
        await asyncio.sleep(1)
        queue.put(f"Message {i}")
    return "Done"

def C():
    for msg in A():
        print(f"C received: {msg}")

if __name__ == "__main__":
    C()