# Python asyncio进阶


## ⚡ asyncio 进阶


Future 对象、run_in_executor 线程池/进程池、超时处理 asyncio.wait_for、asyncio.timeout、信号量 Semaphore、取消任务、异常处理。


## Future 对象


```
// ========== Future ==========
import asyncio

# Future: 表示一个未来的结果 (低层级)
# Task 是 Future 的子类

async def set_future(fut):
    await asyncio.sleep(1)
    fut.set_result("Future 完成!")

async def main():
    # 创建 Future
    fut = asyncio.Future()

    # 创建任务设置 Future
    asyncio.create_task(set_future(fut))

    # 等待 Future 结果
    result = await fut
    print(result)  # Future 完成!

    # Future 状态:
    print(fut.done())       # True
    print(fut.result())     # "Future 完成!"
    # print(fut.exception())  # 如果有异常

asyncio.run(main())

# 大多数情况下使用 Task (高层级 API)
```


## run_in_executor (线程池)


```
// ========== 在协程中运行同步代码 ==========
import asyncio
import time

# 问题: 同步函数会阻塞事件循环!
def blocking_io():
    print("同步 I/O 开始")
    time.sleep(2)  # 阻塞! 其他协程不能运行
    print("同步 I/O 完成")
    return "data"

async def main():
    # ❌ 不要直接调用阻塞函数
    # result = blocking_io()  # 会阻塞整个事件循环!

    # ✅ 用 run_in_executor 放到线程池执行
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,               # None = 默认线程池
        blocking_io         # 同步函数
    )
    print(result)

    # 使用 ThreadPoolExecutor:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = await loop.run_in_executor(pool, blocking_io)

    # 使用 ProcessPoolExecutor (CPU 密集型):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, cpu_intensive)

asyncio.run(main())
```


## 超时处理


```
// ========== 超时 ==========
import asyncio

async def slow_operation():
    await asyncio.sleep(10)
    return "完成"

# 方式 1: asyncio.wait_for (Python 3.7+)
async def main():
    try:
        result = await asyncio.wait_for(
            slow_operation(),
            timeout=3.0  # 3 秒超时
        )
        print(result)
    except asyncio.TimeoutError:
        print("操作超时!")

# 方式 2: asyncio.timeout (Python 3.11+)
async def main():
    try:
        async with asyncio.timeout(3.0):
            result = await slow_operation()
            print(result)
    except asyncio.TimeoutError:
        print("操作超时!")

# 方式 3: 带取消的超时
async def main():
    task = asyncio.create_task(slow_operation())

    try:
        result = await asyncio.wait_for(task, timeout=3.0)
    except asyncio.TimeoutError:
        # task 已被自动取消
        print("超时,已取消")
        # 如果需要部分结果:
        # result = task.result()  # 会抛出 CancelledError

# ========== asyncio.sleep(0) 让出控制权 ==========
async def cooperative():
    for i in range(5):
        print(f"步骤 {i}")
        await asyncio.sleep(0)  # 让其他协程运行
```


## 信号量 Semaphore


```
// ========== 限制并发数量 ==========
import asyncio

# Semaphore: 限制同时运行的协程数量
# 防止对某个资源有过多的并发请求

async def fetch_url(sem, url):
    async with sem:  # 获取信号量 (限制并发)
        print(f"请求: {url}")
        await asyncio.sleep(1)  # 模拟网络请求
        return f"响应: {url}"

async def main():
    # 限制最多 3 个并发
    sem = asyncio.Semaphore(3)

    urls = [f"https://api.example.com/{i}" for i in range(10)]

    # 创建所有任务 (但信号量限制实际并发)
    tasks = [fetch_url(sem, url) for url in urls]
    results = await asyncio.gather(*tasks)

    print(f"完成 {len(results)} 个请求")
    # 总耗时: 约 4 秒 (10/3 ≈ 4 批)

asyncio.run(main())

# ========== BoundedSemaphore ==========
# 确保 release() 次数不超过 acquire() 次数
# sem = asyncio.BoundedSemaphore(3)
```


## 取消任务与异常处理


```
// ========== 取消任务 ==========
import asyncio

async def cancellable_work():
    try:
        for i in range(10):
            print(f"工作 {i}")
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        print("任务被取消!")
        # 清理资源...
        raise  # 必须重新抛出或返回

async def main():
    task = asyncio.create_task(cancellable_work())

    await asyncio.sleep(3)
    task.cancel()  # 发送取消请求

    try:
        await task
    except asyncio.CancelledError:
        print("捕获到取消")

# ========== 屏蔽取消 ==========
async def shutdown_only():
    # asyncio.shield: 保护操作不被取消
    try:
        task.cancel()  # 尝试取消
        await asyncio.shield(important_operation())
        # important_operation 不会被取消
    except asyncio.CancelledError:
        print("外部被取消,但重要操作完成")

# ========== 异常处理 ==========
async def error_handling():
    tasks = [
        asyncio.create_task(might_fail(1)),
        asyncio.create_task(might_fail(2)),
    ]

    # gather 默认第一个异常就抛出
    # 用 return_exceptions=True:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            print(f"任务失败: {r}")
        else:
            print(f"成功: {r}")
```


> **Note:** 💡 asyncio 进阶要点: (1) run_in_executor 在协程中运行同步阻塞代码; (2) wait_for/timeout 超时控制; (3) Semaphore 限制并发数量; (4) task.cancel() 取消任务,需捕获 CancelledError; (5) shield() 保护关键操作不被取消。


## 练习


<!-- Converted from: 107_Python asyncio进阶.html -->
