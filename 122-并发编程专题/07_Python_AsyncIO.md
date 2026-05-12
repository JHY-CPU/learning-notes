# Python AsyncIO

## async/await 基础

`asyncio` 是 Python 的异步编程框架，使用单线程 + 事件循环实现高并发 IO，不存在 GIL 问题。

### 协程定义

```python
import asyncio

async def greet(name: str) -> str:
    """定义协程函数"""
    await asyncio.sleep(1)  # 非阻塞等待
    return f"Hello, {name}!"

async def main():
    result = await greet("World")
    print(result)

# 运行事件循环
asyncio.run(main())
```

### 多个协程并发执行

```python
import asyncio
import time

async def fetch_data(name: str, delay: float) -> str:
    print(f"{name} 开始获取数据")
    await asyncio.sleep(delay)  # 模拟 IO 操作
    print(f"{name} 获取完成")
    return f"{name} 的数据"

async def main():
    # 方式 1: asyncio.gather - 并发执行多个协程
    start = time.time()
    results = await asyncio.gather(
        fetch_data("API-1", 2),
        fetch_data("API-2", 1),
        fetch_data("API-3", 3),
    )
    print(f"总耗时: {time.time() - start:.2f}s")  # 约 3 秒（最慢的那个）
    print(f"结果: {results}")

asyncio.run(main())
```

## 事件循环的底层原理

事件循环是 `asyncio` 的核心，它不断检查并执行就绪的协程任务。

### 事件循环的内部架构

```
┌─────────────────────────────────────────────────────────────┐
│                     事件循环 (Event Loop)                     │
│                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌────────────────┐  │
│  │ Ready Queue   │  │ Selector      │  │ Timer Heap     │  │
│  │ (就绪任务队列) │  │ (IO 多路复用)  │  │ (定时器堆)     │  │
│  │               │  │               │  │                │  │
│  │ [Task1, Task3]│  │ epoll/kqueue  │  │ [sleep(1),     │  │
│  │               │  │ /select       │  │  timeout(2)]   │  │
│  └───────┬───────┘  └───────┬───────┘  └───────┬────────┘  │
│          │                  │                   │           │
│          └──────────────────┼───────────────────┘           │
│                             │                               │
│                             ▼                               │
│                    ┌─────────────────┐                      │
│                    │  Run Loop Tick  │                      │
│                    │  (每次循环迭代)  │                      │
│                    └─────────────────┘                      │
│                                                             │
│  每次 tick 的工作:                                           │
│  1. 检查定时器堆，唤醒到期的定时器任务                          │
│  2. 调用 selector.select(timeout) 等待 IO 事件                │
│  3. 将就绪的 IO 回调加入 Ready Queue                          │
│  4. 执行 Ready Queue 中的所有任务（直到队列为空）              │
│  5. 回到步骤 1                                               │
└─────────────────────────────────────────────────────────────┘
```

### selector 模块：IO 多路复用的抽象

Python 的 selector 模块是对底层 IO 多路复用系统调用的统一抽象：

```
操作系统      底层调用               最大并发           特点
─────────────────────────────────────────────────────────────────
Linux        epoll                  理论无限           O(1) 事件通知
macOS/BSD    kqueue                 理论无限           O(1) 事件通知
Windows      select (IOCP*)         1024 (FD_SETSIZE)  O(n) 轮询
跨平台       select                 1024               最兼容
```

*注: Windows 上 asyncio 实际使用 ProactorEventLoop (IOCP)，性能优于 SelectorEventLoop

**epoll 的工作流程**：

```
                  应用程序
                     │
                     ▼
            ┌─────────────────┐
            │ epoll_create()  │ ← 创建 epoll 实例
            │ epoll_ctl()     │ ← 注册 fd 和感兴趣的事件
            │ epoll_wait()    │ ← 阻塞等待事件
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │   内核空间       │
            │                 │
            │  epoll 实例      │
            │  ┌────────────┐ │
            │  │ 红黑树      │ │ ← 存储所有注册的 fd
            │  │ (O(logn))  │ │
            │  └────────────┘ │
            │  ┌────────────┐ │
            │  │ 就绪链表    │ │ ← 有事件的 fd 放入此处
            │  │            │ │
            │  └────────────┘ │
            └─────────────────┘
```

与 select/poll 的区别：select/poll 每次需要遍历所有 fd（O(n)），而 epoll 使用回调机制，当 fd 就绪时内核自动将其加入就绪链表，应用程序只需检查就绪链表（O(1)）。

### 协程的本质：生成器 + 状态机

Python 的 async/await 底层被编译为生成器（generator）状态机：

```python
# 用户写的代码
async def fetch(url):
    data = await download(url)
    return process(data)

# 编译器实际上生成了类似这样的代码（简化）
def fetch(url):
    # 状态机: 每个 await 点是一个状态
    # 状态 0: 调用 download(url)，挂起自己
    # 状态 1: download 完成，调用 process(data)
    # 状态 2: 返回结果
    state = 0
    while True:
        if state == 0:
            inner = download(url)
            yield inner           # 挂起，等待 download 完成
            state = 1
        elif state == 1:
            data = inner_result
            result = process(data)
            return result         # 终止
```

`await` 的含义：暂停当前协程的执行，将控制权交还事件循环，等待被等待对象完成后恢复执行。

### 创建与管理任务

```python
import asyncio

async def task_function(name: str, delay: float) -> str:
    print(f"任务 {name} 开始")
    await asyncio.sleep(delay)
    print(f"任务 {name} 完成")
    return f"result-{name}"

async def main():
    # 创建任务（立即调度）
    task1 = asyncio.create_task(task_function("A", 2))
    task2 = asyncio.create_task(task_function("B", 1))

    # 等待任务完成
    result1 = await task1
    result2 = await task2
    print(f"结果: {result1}, {result2}")

    # 取消任务
    task3 = asyncio.create_task(task_function("C", 10))
    await asyncio.sleep(0.5)
    task3.cancel()
    try:
        await task3
    except asyncio.CancelledError:
        print("任务 C 已取消")

asyncio.run(main())
```

### Task 的内部结构

```python
# Task 的简化内部模型
class Task(Future):
    def __init__(self, coro, loop):
        self._coro = coro        # 协程对象
        self._loop = loop        # 事件循环引用
        self._state = PENDING    # PENDING / CANCELLED / FINISHED
        self._result = None
        self._callbacks = []     # 完成时的回调列表

    def _step(self, value=None):
        """推进协程执行一步"""
        try:
            # 继续执行协程到下一个 await 点
            future = self._coro.send(value)
            # 注册 future 的回调，完成后继续 _step
            future.add_done_callback(self._wakeup)
        except StopIteration as exc:
            self._result = exc.value
            self._state = FINISHED
            self._schedule_callbacks()
```

### 超时控制

```python
import asyncio

async def slow_operation():
    await asyncio.sleep(10)
    return "完成"

async def main():
    # wait_for: 设置超时
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=2.0)
    except asyncio.TimeoutError:
        print("操作超时")

    # timeout: 上下文管理器（Python 3.11+）
    try:
        async with asyncio.timeout(2.0):
            await slow_operation()
    except TimeoutError:
        print("超时!")

asyncio.run(main())
```

## 完整工程级示例：异步 Web 爬虫

```python
import asyncio
import aiohttp
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse
import re

@dataclass
class CrawlResult:
    url: str
    status: int
    title: Optional[str] = None
    links: list = field(default_factory=list)
    error: Optional[str] = None
    elapsed_ms: float = 0

class AsyncWebCrawler:
    """高性能异步 Web 爬虫，支持深度限制、并发控制、robots.txt"""

    def __init__(self, max_concurrency: int = 10, max_depth: int = 2,
                 timeout: float = 10):
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.max_depth = max_depth
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.visited: set[str] = set()
        self.results: list[CrawlResult] = []
        self._lock = asyncio.Lock()

    async def crawl(self, start_url: str) -> list[CrawlResult]:
        """从起始 URL 开始爬取"""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            await self._crawl_recursive(session, start_url, depth=0)
        return self.results

    async def _crawl_recursive(self, session: aiohttp.ClientSession,
                                url: str, depth: int):
        """递归爬取（深度优先）"""
        if depth > self.max_depth:
            return

        async with self._lock:
            if url in self.visited:
                return
            self.visited.add(url)

        result = await self._fetch_page(session, url)
        self.results.append(result)

        if result.links and depth < self.max_depth:
            # 并发爬取子链接
            tasks = []
            for link in result.links[:20]:  # 限制每页最多跟踪 20 个链接
                if self._is_same_domain(url, link):
                    tasks.append(
                        self._crawl_recursive(session, link, depth + 1)
                    )
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_page(self, session: aiohttp.ClientSession,
                           url: str) -> CrawlResult:
        """获取单个页面"""
        async with self.semaphore:  # 并发控制
            start = time.monotonic()
            try:
                async with session.get(url) as response:
                    html = await response.text()
                    elapsed = (time.monotonic() - start) * 1000

                    title = self._extract_title(html)
                    links = self._extract_links(url, html)

                    return CrawlResult(
                        url=url,
                        status=response.status,
                        title=title,
                        links=links,
                        elapsed_ms=elapsed,
                    )
            except Exception as e:
                elapsed = (time.monotonic() - start) * 1000
                return CrawlResult(
                    url=url,
                    status=0,
                    error=str(e),
                    elapsed_ms=elapsed,
                )

    def _extract_title(self, html: str) -> Optional[str]:
        match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_links(self, base_url: str, html: str) -> list[str]:
        pattern = r'<a\s+[^>]*href=["\']([^"\']+)["\']'
        links = re.findall(pattern, html, re.IGNORECASE)
        return [urljoin(base_url, link) for link in links
                if link.startswith(('http', '/'))]

    def _is_same_domain(self, url1: str, url2: str) -> bool:
        return urlparse(url1).netloc == urlparse(url2).netloc

async def main():
    crawler = AsyncWebCrawler(max_concurrency=5, max_depth=1)
    results = await crawler.crawl("https://example.com")

    for r in results:
        status_str = r.status if r.status else f"ERROR: {r.error}"
        print(f"[{status_str}] {r.url} ({r.elapsed_ms:.0f}ms) "
              f"links={len(r.links)}")

if __name__ == "__main__":
    asyncio.run(main())
```

## aiohttp

`aiohttp` 是异步 HTTP 客户端/服务器库，适合高并发网络请求。

### 异步 HTTP 请求

```python
import asyncio
import aiohttp

async def fetch_url(session: aiohttp.ClientSession, url: str) -> dict:
    """异步获取 URL 内容"""
    async with session.get(url) as response:
        data = await response.text()
        return {"url": url, "status": response.status, "length": len(data)}

async def main():
    urls = [
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2",
        "https://httpbin.org/delay/1",
    ]

    async with aiohttp.ClientSession() as session:
        # 并发请求
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)

        for r in results:
            print(f"{r['url']}: 状态={r['status']}, 长度={r['length']}")

    # 限制并发数
    semaphore = asyncio.Semaphore(2)  # 最多 2 个并发请求

    async def limited_fetch(session, url):
        async with semaphore:
            return await fetch_url(session, url)

    async with aiohttp.ClientSession() as session:
        tasks = [limited_fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        print("限流结果:", results)

asyncio.run(main())
```

### 异步 POST 请求

```python
import asyncio
import aiohttp
import json

async def post_data(session: aiohttp.ClientSession, url: str, data: dict) -> dict:
    async with session.post(url, json=data) as response:
        return await response.json()

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(5):
            data = {"name": f"user-{i}", "age": 20 + i}
            tasks.append(post_data(session, "https://httpbin.org/post", data))

        results = await asyncio.gather(*tasks)
        for r in results:
            print("响应:", json.dumps(r, indent=2)[:100])

asyncio.run(main())
```

## asyncio.gather / asyncio.wait

### gather: 收集结果

```python
import asyncio

async def compute(n: int) -> int:
    await asyncio.sleep(n * 0.1)
    return n * n

async def main():
    # return_exceptions=False: 异常直接抛出
    results = await asyncio.gather(
        compute(1), compute(2), compute(3),
        return_exceptions=True
    )
    print(f"结果: {results}")  # [1, 4, 9]

asyncio.run(main())
```

### wait: 更灵活的等待

```python
import asyncio

async def task(name: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"{name} 完成"

async def main():
    tasks = {
        asyncio.create_task(task("fast", 0.5)),
        asyncio.create_task(task("slow", 2.0)),
        asyncio.create_task(task("medium", 1.0)),
    }

    # FIRST_COMPLETED: 第一个完成就返回
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for d in done:
        print(f"已完成: {d.result()}")

    # 取消剩余任务
    for p in pending:
        p.cancel()

    # 等待所有任务
    tasks2 = {
        asyncio.create_task(task(f"task-{i}", i * 0.2))
        for i in range(5)
    }
    done2, pending2 = await asyncio.wait(tasks2, return_when=asyncio.ALL_COMPLETED)
    print(f"全部完成: {len(done2)} 个任务")

asyncio.run(main())
```

### gather vs wait vs TaskGroup

```
工具           取消传播    异常处理           适用场景
──────────────────────────────────────────────────────────────
gather         无          return_exceptions  简单并发收集结果
wait           无          手动检查           需要灵活等待策略
TaskGroup      有          自动取消其他任务   Python 3.11+, 严格错误处理
```

### TaskGroup (Python 3.11+)

```python
import asyncio

async def task_with_error(name: str, should_fail: bool):
    await asyncio.sleep(0.1)
    if should_fail:
        raise ValueError(f"{name} 故障")
    return f"{name} 成功"

async def main():
    # TaskGroup: 任一任务失败时自动取消所有其他任务
    try:
        async with asyncio.TaskGroup() as tg:
            t1 = tg.create_task(task_with_error("A", False))
            t2 = tg.create_task(task_with_error("B", True))   # 会失败
            t3 = tg.create_task(task_with_error("C", False))  # 会被取消
    except* ValueError as eg:
        for exc in eg.exceptions:
            print(f"捕获异常: {exc}")

    # 即使 t3 没有失败，也会被取消
    print(f"t1: {t1.done()}, t2: {t2.done()}, t3: {t3.done()}")
    # t1: True, t2: True, t3: True (被取消)

asyncio.run(main())
```

## asyncio.Queue

异步队列用于协程之间的数据传递，实现生产者-消费者模式。

```python
import asyncio
import random

async def producer(queue: asyncio.Queue, name: str):
    """生产者：生成任务放入队列"""
    for i in range(5):
        item = f"{name}-item-{i}"
        await asyncio.sleep(random.uniform(0.1, 0.3))
        await queue.put(item)
        print(f"生产: {item} (队列大小: {queue.qsize()})")

async def consumer(queue: asyncio.Queue, name: str):
    """消费者：从队列取出任务处理"""
    while True:
        item = await queue.get()
        if item is None:  # 结束信号
            queue.task_done()
            break
        print(f"消费者 {name} 处理: {item}")
        await asyncio.sleep(random.uniform(0.2, 0.5))
        queue.task_done()

async def main():
    queue = asyncio.Queue(maxsize=10)

    # 启动生产者
    producers = [
        asyncio.create_task(producer(queue, f"P{i}"))
        for i in range(2)
    ]

    # 启动消费者
    consumers = [
        asyncio.create_task(consumer(queue, f"C{i}"))
        for i in range(3)
    ]

    # 等待所有生产者完成
    await asyncio.gather(*producers)

    # 发送结束信号
    for _ in consumers:
        await queue.put(None)

    # 等待所有消费者完成
    await asyncio.gather(*consumers)

    print("所有任务完成")

asyncio.run(main())
```

### Queue 的底层实现

```python
# asyncio.Queue 的简化实现
class Queue:
    def __init__(self, maxsize=0):
        self._maxsize = maxsize
        self._queue = collections.deque()  # 双端队列
        self._getters = collections.deque()  # 等待 get 的 Future
        self._putters = collections.deque()  # 等待 put 的 Future

    async def put(self, item):
        # 有空位直接放入
        while self.full():
            putter = self._loop.create_future()
            self._putters.append(putter)
            try:
                await putter  # 阻塞等待
            except:
                # 被取消时，如果有空位让给其他 putter
                ...
        self._queue.append(item)
        self._wakeup_next(self._getters)  # 唤醒一个等待的 get

    async def get(self):
        while self.empty():
            getter = self._loop.create_future()
            self._getters.append(getter)
            try:
                await getter  # 阻塞等待
            except:
                ...
        item = self._queue.popleft()
        self._wakeup_next(self._putters)  # 唤醒一个等待的 put
        return item
```

## 异步生成器

```python
import asyncio

async def async_range(start: int, stop: int, delay: float = 0.1):
    """异步生成器"""
    for i in range(start, stop):
        await asyncio.sleep(delay)
        yield i

async def main():
    async for num in async_range(0, 5):
        print(f"收到: {num}")

    # 异步列表推导（Python 3.6+）
    results = [num async for num in async_range(0, 5)]
    print(f"结果: {results}")

asyncio.run(main())
```

### 异步生成器的实际用途：分页数据获取

```python
async def fetch_pages(session, base_url, max_pages=10):
    """异步生成器：逐页获取数据"""
    page = 1
    while page <= max_pages:
        async with session.get(f"{base_url}?page={page}") as resp:
            data = await resp.json()
            if not data["items"]:
                break
            for item in data["items"]:
                yield item
            page += 1

async def main():
    async with aiohttp.ClientSession() as session:
        count = 0
        async for item in fetch_pages(session, "https://api.example.com/data"):
            print(f"处理: {item}")
            count += 1
        print(f"共处理 {count} 条数据")
```

## 异步上下文管理器

```python
import asyncio

class AsyncDatabase:
    def __init__(self, name: str):
        self.name = name

    async def __aenter__(self):
        print(f"连接数据库 {self.name}")
        await asyncio.sleep(0.1)  # 模拟连接
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print(f"关闭数据库 {self.name}")
        await asyncio.sleep(0.05)  # 模拟关闭

    async def query(self, sql: str):
        await asyncio.sleep(0.1)
        return f"查询结果: {sql}"

async def main():
    async with AsyncDatabase("mydb") as db:
        result = await db.query("SELECT * FROM users")
        print(result)

asyncio.run(main())
```

## Semaphore 限制并发

```python
import asyncio
import aiohttp

async def fetch_with_semaphore(sem: asyncio.Semaphore, session, url):
    """使用信号量限制并发数"""
    async with sem:
        print(f"开始请求: {url}")
        async with session.get(url) as resp:
            await asyncio.sleep(0.5)  # 模拟处理
            result = await resp.text()
            print(f"完成请求: {url}, 长度={len(result)}")
            return len(result)

async def main():
    urls = [f"https://httpbin.org/get?id={i}" for i in range(10)]

    # 最多 3 个并发请求
    sem = asyncio.Semaphore(3)

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_with_semaphore(sem, session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        print(f"总长度: {sum(results)}")

asyncio.run(main())
```

### Semaphore 的底层实现

```python
# asyncio.Semaphore 简化实现
class Semaphore:
    def __init__(self, value=1):
        self._value = value         # 当前信号量值
        self._waiters = deque()     # 等待获取的 Future 队列

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *args):
        self.release()

    async def acquire(self):
        while self._value <= 0:
            fut = self._loop.create_future()
            self._waiters.append(fut)
            await fut  # 阻塞等待
        self._value -= 1
        return True

    def release(self):
        self._value += 1
        # 唤醒一个等待者
        if self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.set_result(True)
```

## 同步与异步混用

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

def blocking_io():
    """阻塞 IO 操作"""
    time.sleep(1)
    return "IO 完成"

def cpu_bound():
    """CPU 密集型操作"""
    return sum(i * i for i in range(10_000_000))

async def main():
    loop = asyncio.get_event_loop()

    # 在线程池中执行阻塞 IO
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, blocking_io)
        print(f"线程池结果: {result}")

    # 在进程池中执行 CPU 密集任务
    with ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, cpu_bound)
        print(f"进程池结果: {result}")

    # 默认线程池
    result = await loop.run_in_executor(None, blocking_io)
    print(f"默认线程池: {result}")

asyncio.run(main())
```

### 为什么阻塞代码会破坏事件循环

```
事件循环是单线程的:

  时间线:
  ├── Task A: await sleep(1)  → 挂起，交出控制权 ✓
  ├── Task B: await fetch()   → 挂起，交出控制权 ✓
  ├── Task C: time.sleep(5)   → 阻塞整个线程 5 秒! ✗
  │   └── 期间 Task A、B 的 IO 完成了但无法被处理
  ├── Task D: 已就绪但无法执行
  └── ...

解决方案:
  - run_in_executor: 将阻塞代码扔到线程池执行
  - 使用异步版本的库 (aiohttp 替代 requests, aiofiles 替代 open)
  - spawn_blocking: Tokio 等运行时的类似机制
```

## 异步编程的调试方法论

### 1. 检测未等待的协程

```python
import asyncio
import warnings

# Python 3.8+: 未 await 的协程会触发 RuntimeWarning
# 开启调试模式可以检测更多问题
asyncio.run(main(), debug=True)

# 或者通过环境变量
# PYTHONASYNCIODEBUG=1 python script.py

# 调试模式启用的检查:
# - 未 await 的协程（RuntimeWarning）
# - 执行时间过长的回调（> 100ms）
# - 未正确关闭的资源
# - slow_callback_duration 可配置
```

### 2. 检测协程泄漏

```python
import asyncio

async def main():
    # 启动大量任务但忘记等待
    for i in range(100):
        asyncio.create_task(asyncio.sleep(9999))

    print(f"活跃任务: {len(asyncio.all_tasks())}")  # 101 (包含 main 自己)

    # 优雅地取消所有任务
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
```

### 3. 使用 aiomonitor 监控异步程序

```python
# pip install aiomonitor
import asyncio
import aiomonitor

async def main():
    loop = asyncio.get_event_loop()
    with aiomonitor.start_monitor(loop):
        # 在另一个终端运行: telnet localhost 50101
        # 可以查看所有任务、协程状态
        await asyncio.sleep(3600)

asyncio.run(main())

# aiomonitor 命令:
# tasks     - 列出所有任务
# ps        - 列出所有协程
# where <id>- 查看协程调用栈
# signal <id> - 发送异常给指定协程
```

### 4. 使用 cProfile 分析异步性能

```python
import cProfile
import asyncio
import pstats

def profile_async(coro):
    """分析异步协程的性能"""
    profiler = cProfile.Profile()
    profiler.enable()

    result = asyncio.run(coro)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    return result

# 使用
async def main():
    await some_async_work()

profile_async(main())
```

### 5. 异步死锁检测

```python
import asyncio
import signal

async def detect_deadlock():
    """检测潜在死锁：所有任务都在等待"""
    while True:
        await asyncio.sleep(5)
        tasks = asyncio.all_tasks()
        waiting = sum(1 for t in tasks if t._fut_waiter is not None)
        if waiting == len(tasks):
            print(f"警告: 所有 {len(tasks)} 个任务都在等待，可能存在死锁")
            for t in tasks:
                print(f"  Task: {t.get_name()}, Stack: {t.get_stack()}")

# 常见异步死锁模式:
# 1. 协程等待自己持有的锁
#    lock = asyncio.Lock()
#    async with lock:
#        async with lock:  # 死锁!
#            pass
#
# 2. 循环依赖
#    async def a(): await b_event.wait()
#    async def b(): await a_event.wait()  # 死锁!
```

## GIL 与 AsyncIO 的关系

```
┌─────────────────────────────────────────────────────────────┐
│ Python 的并发模型对比                                       │
├──────────────┬──────────────────────────────────────────────┤
│ threading    │ GIL 限制：同一时刻只有一个线程执行 Python 字节码│
│              │ 适用: IO 密集型（线程在等待 IO 时释放 GIL）    │
│              │ 不适用: CPU 密集型                            │
├──────────────┼──────────────────────────────────────────────┤
│ multiprocessing│ 多进程，每个进程有独立 GIL                   │
│              │ 适用: CPU 密集型                              │
│              │ 代价: 进程间通信开销大                        │
├──────────────┼──────────────────────────────────────────────┤
│ asyncio      │ 单线程 + 协程 + 事件循环                     │
│              │ 完全绕过 GIL 问题                            │
│              │ 适用: 高并发 IO（数万连接）                   │
│              │ 不适用: CPU 密集型（会阻塞事件循环）           │
└──────────────┴──────────────────────────────────────────────┘

GIL 的本质:
  - 全局互斥锁，保护 Python 对象的引用计数
  - 每执行 N 条字节码（默认 100，Python 3.2+ 为 5ms）释放一次
  - IO 操作会释放 GIL，C 扩展可以手动释放 GIL

AsyncIO 完全不涉及 GIL 竞争:
  - 单线程，无需线程间同步
  - 协作式调度，由 await 点显式让出控制权
  - 零上下文切换开销（对比线程）
```

## 10K 连接性能对比

```
场景: 同时处理 10,000 个并发 HTTP 连接

方案               内存占用       创建开销       上下文切换      吞吐量
─────────────────────────────────────────────────────────────────────────
多线程 (threading)  ~80 GB*       ~50 μs/线程    ~5 μs         无法创建
                    (*每线程 8MB 栈)
asyncio             ~200 MB       ~1 μs/协程     ~0.1 μs       ~15K req/s
aiohttp + uvloop    ~150 MB       ~0.5 μs        ~0.05 μs      ~25K req/s
multiprocessing     ~80 GB+       ~100 μs/进程   ~10 μs        无法创建

*实际中会使用线程池，但线程池大小通常 < 1000
```

### uvloop: 加速事件循环

```python
# uvloop 是 libuv (Node.js 底层) 的 Python 封装
# 替换默认事件循环，速度提升 2-4 倍
import uvloop

# 方式 1: 全局替换
uvloop.install()
asyncio.run(main())

# 方式 2: 指定使用
async def main():
    ...

uvloop.run(main())
```

```
性能对比 (echo server, 100 并发连接):

事件循环          请求/秒        延迟 (p99)
────────────────────────────────────────────
asyncio (默认)    ~10,000        ~2 ms
uvloop            ~40,000        ~0.5 ms
Node.js           ~35,000        ~0.6 ms
Go (net/http)     ~45,000        ~0.4 ms
```

## 生产案例

### Instagram 的 Python AsyncIO 使用

Instagram 的后端大量使用 Python，通过 asyncio 处理高并发：
- 使用 `asyncio` + `uvloop` 替代传统的多进程模型
- 单个服务器可以处理数万个并发连接
- 将 CPU 密集型任务卸载到进程池（`run_in_executor`）
- 通过 asyncio.Semaphore 控制对下游服务的并发调用

### Discord 的 Python 异步实践

Discord 早期使用 Python + asyncio 构建聊天后端：
- 单个进程处理数百万 WebSocket 连接
- 使用 `asyncio` 管理每个连接的心跳和消息推送
- 后来将性能关键路径迁移到 Rust（见 Rust 异步编程章节）

### Netflix 的异步网关

Netflix 使用 Python asyncio 构建 API 网关：
- 异步处理请求转发和响应聚合
- 使用 aiohttp 客户端并发调用多个微服务
- 通过 Semaphore 限制对每个微服务的并发数，防止雪崩

## 常见陷阱详解

### 陷阱 1：忘记 await

```python
# BUG: 协程没有被 await，不会执行
async def save_data(data):
    await db.write(data)

# 错误写法
save_data(mydata)  # RuntimeWarning: coroutine was never awaited

# 正确写法
await save_data(mydata)

# 在非 async 函数中调用需要特殊处理
def sync_wrapper(data):
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # 如果已有事件循环在运行（如 Jupyter），创建任务
        task = asyncio.create_task(save_data(data))
        return task
    else:
        return asyncio.run(save_data(data))
```

### 陷阱 2：在异步代码中使用阻塞调用

```python
import time
import requests

# BUG: time.sleep 阻塞整个事件循环
async def bad_fetch(url):
    time.sleep(1)  # 所有其他协程都被阻塞!
    return requests.get(url).text  # requests 是同步库!

# 修复 1: 使用异步替代
async def good_fetch(url):
    await asyncio.sleep(1)
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()

# 修复 2: 必须用同步库时，使用 run_in_executor
async def ok_fetch(url):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, requests.get, url)
```

### 陷阱 3：事件循环嵌套

```python
# BUG: RuntimeError: This event loop is already running
async def inner():
    return 42

async def outer():
    result = asyncio.run(inner())  # 不能在已有循环中调用 asyncio.run!
    return result

# 修复: 直接 await
async def outer_fixed():
    result = await inner()
    return result

# 在 Jupyter 等已有事件循环的环境中
import nest_asyncio
nest_asyncio.apply()  # 允许嵌套事件循环
```

### 陷阱 4：共享可变状态没有同步

```python
# BUG: 虽然是单线程，但某些操作不是原子的
counter = 0

async def increment():
    global counter
    for _ in range(1000):
        temp = counter
        await asyncio.sleep(0)  # 让出控制权，其他协程可能修改 counter
        counter = temp + 1

# 10 个协程各增加 1000 次，期望 10000，实际可能 < 10000

# 修复: 使用 asyncio.Lock
lock = asyncio.Lock()

async def safe_increment():
    global counter
    for _ in range(1000):
        async with lock:
            counter += 1
```

### 陷阱 5：任务取消后未正确处理

```python
# BUG: 忽略 CancelledError 导致资源泄漏
async def buggy_cleanup():
    try:
        await long_operation()
    except asyncio.CancelledError:
        await cleanup()  # 可能导致 CancelledError 被吞掉
        # 应该重新抛出!

# 正确写法
async def correct_cleanup():
    try:
        await long_operation()
    except asyncio.CancelledError:
        await cleanup()
        raise  # 重新抛出，确保取消信号传播
    finally:
        await cleanup()  # finally 是更安全的方式
```

## 总结

| 工具 | 用途 | 说明 |
|------|------|------|
| async/await | 定义协程 | 基于生成器的状态机 |
| asyncio.run() | 运行入口 | 创建事件循环并执行 |
| asyncio.gather | 并发执行 | 收集所有结果 |
| asyncio.wait | 灵活等待 | FIRST_COMPLETED 等策略 |
| asyncio.Queue | 异步队列 | 生产者-消费者模式 |
| asyncio.Semaphore | 并发限制 | 控制并发数 |
| TaskGroup | 结构化并发 | Python 3.11+, 自动取消传播 |
| aiohttp | 异步 HTTP | 高并发网络请求 |
| uvloop | 加速事件循环 | libuv 封装, 2-4x 提升 |
| run_in_executor | 混合同步 | 在线程/进程中运行阻塞代码 |
