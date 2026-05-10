# Python asyncio队列与同步原语


## 📋 asyncio 队列与同步原语


asyncio.Queue 生产者-消费者模式、LifoQueue/PriorityQueue、Event/Condition/Lock 同步原语、工作池模式。


## asyncio.Queue 基础


```
// ========== asyncio.Queue ==========
import asyncio

# Queue: 协程安全的 FIFO 队列
# 用于在协程间传递数据

async def producer(queue):
    """生产者: 往队列放数据"""
    for i in range(5):
        item = f"item-{i}"
        await queue.put(item)  # 队列满时等待
        print(f"生产: {item}")
        await asyncio.sleep(0.3)

async def consumer(queue):
    """消费者: 从队列取数据"""
    while True:
        item = await queue.get()  # 队列空时等待
        print(f"消费: {item}")
        queue.task_done()  # 标记任务完成
        await asyncio.sleep(0.5)

async def main():
    queue = asyncio.Queue(maxsize=3)  # 最大容量 3

    # 创建生产者和消费者(并发运行)
    producer_task = asyncio.create_task(producer(queue))
    consumer_task = asyncio.create_task(consumer(queue))

    # 等待生产者完成
    await producer_task
    # 等待队列中所有项目被处理
    await queue.join()
    # 取消消费者
    consumer_task.cancel()

asyncio.run(main())
```


## Queue 方法与配置


```
// ========== Queue API ==========
import asyncio

async def queue_api():
    q = asyncio.Queue(maxsize=10)

    # 常用方法:
    await q.put(item)          # 放元素 (满时等待)
    item = await q.get()       # 取元素 (空时等待)
    q.put_nowait(item)         # 非阻塞放 (满时抛出 QueueFull)
    item = q.get_nowait()      # 非阻塞取 (空时抛出 QueueEmpty)
    q.task_done()              # 标记一个任务已完成
    await q.join()             # 等待所有任务完成

    # 属性:
    size = q.qsize()           # 当前元素数量
    empty = q.empty()          # 是否为空
    full = q.full()            # 是否已满

    # maxsize=0 表示无上限

# ========== LifoQueue / PriorityQueue ==========
from asyncio import LifoQueue, PriorityQueue

async def lifo_demo():
    """后进先出 (栈)"""
    q = LifoQueue(maxsize=5)
    for i in range(3):
        await q.put(f"task-{i}")
    while not q.empty():
        item = await q.get()
        print(f"LIFO 取: {item}")  # task-2, task-1, task-0

async def priority_demo():
    """优先级队列 (小值优先)"""
    pq = PriorityQueue()
    await pq.put((2, "普通任务"))
    await pq.put((1, "紧急任务"))
    await pq.put((3, "低优先级"))

    while not pq.empty():
        priority, task = await pq.get()
        print(f"[{priority}] {task}")
    # 输出: [1] 紧急任务 → [2] 普通任务 → [3] 低优先级

async def main():
    await lifo_demo()
    print("---")
    await priority_demo()

asyncio.run(main())
```


## 生产者-消费者模式


```
// ========== 多生产者多消费者 ==========
import asyncio
import random

async def producer(queue, pid):
    for i in range(3):
        item = f"P{pid}-item-{i}"
        await queue.put(item)
        print(f"生产者{pid}: {item}")
        await asyncio.sleep(random.uniform(0.1, 0.3))

async def consumer(queue, cid):
    while True:
        item = await queue.get()
        print(f"消费者{cid}: 处理 {item}")
        await asyncio.sleep(random.uniform(0.2, 0.5))
        queue.task_done()

async def main():
    queue = asyncio.Queue(maxsize=10)

    # 2 个生产者, 3 个消费者
    producers = [asyncio.create_task(producer(queue, i)) for i in range(2)]
    consumers = [asyncio.create_task(consumer(queue, i)) for i in range(3)]

    # 等待所有生产者完成
    await asyncio.gather(*producers)
    # 等待队列清空
    await queue.join()
    # 取消所有消费者
    for c in consumers:
        c.cancel()

    print("所有任务完成")

asyncio.run(main())
```


## 异步同步原语


```
// ========== Lock ==========
import asyncio

# Lock: 互斥锁,保护共享资源

shared_counter = 0
lock = asyncio.Lock()

async def increment():
    global shared_counter
    async with lock:  # 获取锁
        temp = shared_counter
        await asyncio.sleep(0.1)  # 模拟耗时
        shared_counter = temp + 1
        print(f"计数器: {shared_counter}")
    # 自动释放锁

async def main():
    tasks = [asyncio.create_task(increment()) for _ in range(5)]
    await asyncio.gather(*tasks)
    print(f"最终值: {shared_counter}")  # 5 (不加锁可能 < 5)

asyncio.run(main())

# ========== Event ==========
# Event: 等待某个事件发生

async def waiter(event, wid):
    print(f"等待者{wid} 等待事件...")
    await event.wait()  # 阻塞直到事件被设置
    print(f"等待者{wid} 收到事件!")

async def setter(event):
    print("2 秒后触发事件...")
    await asyncio.sleep(2)
    event.set()  # 通知所有等待者

async def main():
    event = asyncio.Event()
    waiters = [asyncio.create_task(waiter(event, i)) for i in range(3)]
    await asyncio.gather(*waiters, asyncio.create_task(setter(event)))

asyncio.run(main())
```


## Condition 与 Semaphore


```
// ========== Condition ==========
import asyncio

# Condition: 条件变量,等待特定条件满足

async def consumer(cond, data):
    async with cond:
        while not data["ready"]:  # 防止虚假唤醒
            await cond.wait()     # 等待通知
        print(f"消费: {data['value']}")

async def producer(cond, data):
    await asyncio.sleep(1)
    async with cond:
        data["ready"] = True
        data["value"] = "重要数据"
        cond.notify_all()  # 通知所有等待者

async def main():
    cond = asyncio.Condition()
    data = {"ready": False, "value": None}

    await asyncio.gather(
        consumer(cond, data),
        consumer(cond, data),
        producer(cond, data),
    )

asyncio.run(main())

# ========== Semaphore 高级用法 ==========
sem = asyncio.Semaphore(3)

async def limited_task(wid):
    async with sem:
        print(f"任务{wid} 开始")
        await asyncio.sleep(1)
        print(f"任务{wid} 结束")

async def main():
    await asyncio.gather(*[limited_task(i) for i in range(6)])
    # 同时最多 3 个任务执行,分 2 批完成

asyncio.run(main())

# ========== BoundedSemaphore ==========
# 确保 release 次数不会超过 acquire 次数
bsem = asyncio.BoundedSemaphore(3)
# bsem.release()  # 会报错! 不能超过初始值
```


> **Note:** 💡 Queue 协调生产者和消费者; Lock 保护共享资源; Event 通知多个协程; Condition 等待复杂条件; Semaphore 限制并发。


## 完整示例: 爬虫工作池


```
// ========== 异步爬虫工作池 ==========
import asyncio
import random

class CrawlWorker:
    """异步爬虫工作池"""

    def __init__(self, num_workers=3):
        self.num_workers = num_workers
        self.url_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.workers = []

    async def worker(self, wid):
        """单个工作线程"""
        while True:
            try:
                url = await asyncio.wait_for(
                    self.url_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                break  # 队列空了,退出

            print(f"Worker{wid} 爬取: {url}")
            await asyncio.sleep(random.uniform(0.3, 0.8))  # 模拟网络

            result = {"url": url, "title": f"Title-{url}", "length": len(url) * 10}
            await self.result_queue.put(result)
            self.url_queue.task_done()

    async def crawl(self, urls):
        # 放入 URL
        for url in urls:
            await self.url_queue.put(url)

        # 启动 workers
        self.workers = [
            asyncio.create_task(self.worker(i))
            for i in range(self.num_workers)
        ]

        # 等待所有 URL 处理完成
        await self.url_queue.join()

        # 取消 workers
        for w in self.workers:
            w.cancel()

        # 收集结果
        results = []
        while not self.result_queue.empty():
            results.append(await self.result_queue.get())
        return results

async def main():
    pool = CrawlWorker(num_workers=3)
    urls = [f"https://example.com/page/{i}" for i in range(10)]

    results = await pool.crawl(urls)
    print(f"\n共爬取 {len(results)} 个页面:")
    for r in results:
        print(f"  {r['url']:40} {r['title']}")

asyncio.run(main())
```


## 练习


<!-- Converted from: 110_Python asyncio队列与同步原语.html -->
