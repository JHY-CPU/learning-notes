# Python 并发模型对比


## 🔄 Python 并发模型对比


threading (I/O 密集型)、multiprocessing (CPU 密集型)、asyncio (高并发 I/O)、concurrent.futures、选型对比。


## threading 多线程


```
// ========== threading ==========
import threading
import time

# Python 线程: 受 GIL 限制
# 适合 I/O 密集型 (文件/网络/数据库)
# 不适合 CPU 密集型 (GIL 限制并行)

def worker(name, delay):
    print(f"线程 {name} 开始")
    time.sleep(delay)
    print(f"线程 {name} 结束")

# 创建线程
threads = []
for i in range(3):
    t = threading.Thread(target=worker, args=(f"W{i}", i))
    threads.append(t)
    t.start()

# 等待所有线程完成
for t in threads:
    t.join()

print("所有线程完成")

# ========== ThreadPoolExecutor ==========
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    # submit: 单个提交
    future = executor.submit(worker, "T1", 1)

    # map: 批量提交
    results = executor.map(
        lambda x: x * 2,
        range(10)
    )

# ========== 线程安全 ==========
import threading

lock = threading.Lock()
counter = 0

def safe_increment():
    global counter
    with lock:  # 获取锁
        counter += 1
    # 自动释放

# threading 其他工具:
# threading.Event()     # 事件通知
# threading.Semaphore() # 信号量
# threading.Condition() # 条件变量
# threading.Barrier()   # 栅栏
# threading.local()     # 线程局部存储
```


## multiprocessing 多进程


```
// ========== multiprocessing ==========
import multiprocessing as mp
import time

# 多进程: 绕过 GIL,利用多核 CPU
# 适合 CPU 密集型 (计算/图像处理)
# 进程间通信复杂,内存开销大

def cpu_heavy(n):
    """CPU 密集型任务"""
    total = 0
    for i in range(n):
        total += i ** 2
    return total

# 进程池
with mp.Pool(processes=4) as pool:
    # map: 分发任务
    results = pool.map(cpu_heavy, [1000000, 2000000, 3000000])

    # async 版本
    result = pool.apply_async(cpu_heavy, (5000000,))
    print(result.get(timeout=10))

# ========== 进程间通信 ==========
# Queue (安全)
def producer(q):
    for i in range(5):
        q.put(i)
        time.sleep(0.1)

def consumer(q):
    while True:
        item = q.get()
        if item is None:
            break
        print(f"消费: {item}")

q = mp.Queue()
p1 = mp.Process(target=producer, args=(q,))
p2 = mp.Process(target=consumer, args=(q,))
p1.start()
p2.start()

p1.join()
q.put(None)  # 哨兵值
p2.join()

# Pipe (更快,但仅两点通信)
# Value/Array (共享内存)
# Manager (高层共享)
```


## asyncio 协程


```
// ========== asyncio ==========
import asyncio

# asyncio: 单线程事件循环
# 适合: 高并发 I/O (数千个连接)
# 不适合: CPU 密集型 (阻塞事件循环)

async def fetch_url(url):
    """模拟异步网络请求"""
    await asyncio.sleep(1)  # 非阻塞等待
    return f"数据: {url}"

async def main():
    # 并发执行多个任务
    tasks = [
        fetch_url(f"https://api.example.com/{i}")
        for i in range(100)
    ]

    # gather: 等待所有完成
    results = await asyncio.gather(*tasks)
    print(f"完成 {len(results)} 个请求")

asyncio.run(main())

# ========== 对比总结 ==========
# 特性        threading   multiprocessing  asyncio
# ------------------------------------------------
# 并行性      ❌ (GIL)    ✅ (多核)         ❌ (单线程)
# I/O 密集型  ✅          不太适合           ✅✅
# CPU 密集型  ❌          ✅                ❌
# 内存开销    小          大                小
# 任务切换    系统级       系统级            用户级(极快)
# 数据共享    容易         困难              容易
# 适合连接数  几十         几十              数千+
# 学习曲线    简单         中等              较难
```


## concurrent.futures 统一接口


```
// ========== concurrent.futures ==========
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
    wait,
    FIRST_COMPLETED,
    ALL_COMPLETED,
)
import time

# 统一接口切换线程/进程

def work(n):
    time.sleep(n)
    return n * 2

# 一键切换线程池 ↔ 进程池:
# Executor = ThreadPoolExecutor   # I/O
# Executor = ProcessPoolExecutor  # CPU

with ThreadPoolExecutor(max_workers=4) as executor:
    # 提交多个任务
    futures = {
        executor.submit(work, n): n
        for n in [3, 1, 4, 1, 5]
    }

    # as_completed: 按完成顺序迭代
    for future in as_completed(futures):
        print(f"完成: {future.result()}")

    # wait: 等待策略
    done, not_done = wait(
        futures,
        timeout=2,
        return_when=FIRST_COMPLETED,
    )

# ========== 混合模式 ==========
# 使用 run_in_executor 在协程中运行阻塞代码
import asyncio

async def main():
    loop = asyncio.get_running_loop()

    # 在线程池执行同步 I/O
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool,
            lambda: time.sleep(1)  # 阻塞函数
        )

    # 在进程池执行 CPU 密集型
    with ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool,
            cpu_heavy, 1000000
        )

asyncio.run(main())
```


## 选型指南


```
// ========== 选型决策树 ==========
# 任务类型 → 推荐方案
#
# ┌─ I/O 密集型?
# │   ├─ 高并发 (1000+ 连接) → asyncio
# │   └─ 低并发 (几十连接)   → threading
# │
# └─ CPU 密集型?
#     ├─ 计算密集 → multiprocessing
#     └─ 混合型 → asyncio + ProcessPoolExecutor

# ========== 实战场景 ==========
# Web 服务 (FastAPI/Flask):
#   → asyncio (处理高并发请求)
#   → 阻塞操作用 run_in_executor

# 爬虫:
#   → asyncio + aiohttp (数千并发)
#   → threading + requests (数百并发)

# 图像/视频处理:
#   → multiprocessing (CPU 并行)
#   → ProcessPoolExecutor (简单)

# 定时任务:
#   → threading (少量任务)
#   → asyncio (大量任务)

# 数据分析:
#   → multiprocessing (并行计算)
#   → asyncio + run_in_executor (混合)

# ========== 性能比较 ==========
# 用 timeit 测试不同模型:
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

# I/O 密集型 (sleep)
N = 100
TASKS = 20

# threading:
def io_threading():
    threads = []
    for _ in range(TASKS):
        t = threading.Thread(target=lambda: time.sleep(0.1))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

# asyncio:
async def io_asyncio():
    await asyncio.gather(*[
        asyncio.sleep(0.1) for _ in range(TASKS)
    ])

# 多线程: ≈ 0.1s (并行 I/O)
# asyncio: ≈ 0.1s (并行 I/O)
# 顺序: ≈ 2.0s (N * 0.1)
💡 并发选型: I/O + 高并发 → asyncio; I/O + 简单 → threading; CPU 密集 → multiprocessing; 混合 → asyncio + ProcessPoolExecutor; GIL 限制 threading 的 CPU 并行。
```


## 练习


## 练习


<!-- Converted from: 129_Python 并发模型对比.html -->
