# Python 并发编程

## threading 模块

Python 的 `threading` 模块提供了对多线程的基本支持。

### 创建线程

```python
import threading
import time

def worker(name, delay):
    """线程工作函数"""
    print(f"线程 {name} 开始")
    time.sleep(delay)
    print(f"线程 {name} 结束")

# 方式 1: 直接传入函数
t1 = threading.Thread(target=worker, args=("A", 1))
t2 = threading.Thread(target=worker, args=("B", 2))
t1.start()
t2.start()
t1.join()
t2.join()

# 方式 2: 继承 Thread 类
class MyThread(threading.Thread):
    def __init__(self, name):
        super().__init__(name=name)
        self.result = None

    def run(self):
        self.result = sum(range(1000000))
        print(f"{self.name} 计算结果: {self.result}")

t = MyThread("计算线程")
t.start()
t.join()
print(f"获取结果: {t.result}")
```

### 线程同步：Lock

```python
import threading

class BankAccount:
    def __init__(self, balance):
        self.balance = balance
        self.lock = threading.Lock()

    def deposit(self, amount):
        with self.lock:  # 自动获取和释放锁
            current = self.balance
            time.sleep(0.001)  # 模拟耗时操作
            self.balance = current + amount

    def withdraw(self, amount):
        with self.lock:
            if self.balance >= amount:
                self.balance -= amount

# 测试线程安全
account = BankAccount(0)
threads = []

for _ in range(100):
    t = threading.Thread(target=account.deposit, args=(10,))
    threads.append(t)

for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"最终余额: {account.balance}")  # 正确结果: 1000
```

### RLock（可重入锁）

```python
import threading

class ReentrantExample:
    def __init__(self):
        self.lock = threading.RLock()  # 可重入锁
        self.data = []

    def outer(self):
        with self.lock:
            print("外层方法获取锁")
            self.inner()  # 同一线程可以再次获取 RLock

    def inner(self):
        with self.lock:
            print("内层方法获取锁")
            self.data.append(1)

obj = ReentrantExample()
obj.outer()
```

### Condition（条件变量）

```python
import threading
from collections import deque

class BoundedQueue:
    def __init__(self, capacity):
        self.queue = deque()
        self.capacity = capacity
        self.lock = threading.Lock()
        self.not_full = threading.Condition(self.lock)
        self.not_empty = threading.Condition(self.lock)

    def put(self, item):
        with self.not_full:
            while len(self.queue) >= self.capacity:
                self.not_full.wait()  # 等待消费者取走元素
            self.queue.append(item)
            self.not_empty.notify()

    def get(self):
        with self.not_empty:
            while len(self.queue) == 0:
                self.not_empty.wait()  # 等待生产者放入元素
            item = self.queue.popleft()
            self.not_full.notify()
            return item

# 使用示例
queue = BoundedQueue(5)

def producer():
    for i in range(10):
        queue.put(i)
        print(f"生产: {i}")

def consumer():
    for _ in range(10):
        item = queue.get()
        print(f"消费: {item}")

p = threading.Thread(target=producer)
c = threading.Thread(target=consumer)
p.start()
c.start()
p.join()
c.join()
```

### Event

```python
import threading
import time

event = threading.Event()

def waiter(name):
    print(f"{name} 等待信号...")
    event.wait()  # 阻塞直到 event 被设置
    print(f"{name} 收到信号!")

def setter():
    time.sleep(2)
    print("发送信号")
    event.set()  # 设置信号，唤醒所有等待的线程

for i in range(3):
    threading.Thread(target=waiter, args=(f"线程-{i}",)).start()

threading.Thread(target=setter).start()
```

## multiprocessing 模块

`multiprocessing` 使用独立进程，绕过 GIL，真正实现并行计算。

### 基本用法

```python
import multiprocessing
import os

def cpu_task(n):
    """CPU 密集型任务"""
    pid = os.getpid()
    result = sum(i * i for i in range(n))
    print(f"进程 {pid} 计算 1..{n} 的平方和 = {result}")
    return result

if __name__ == "__main__":
    # 创建进程
    processes = []
    for n in [1000000, 2000000, 3000000]:
        p = multiprocessing.Process(target=cpu_task, args=(n,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
```

### 进程池

```python
import multiprocessing
import time

def heavy_computation(n):
    return sum(i * i for i in range(n))

if __name__ == "__main__":
    data = [1000000, 2000000, 3000000, 4000000, 5000000]

    # 串行计算
    start = time.time()
    serial = [heavy_computation(n) for n in data]
    print(f"串行: {time.time() - start:.2f}s")

    # 并行计算
    start = time.time()
    with multiprocessing.Pool(processes=4) as pool:
        parallel = pool.map(heavy_computation, data)
    print(f"并行: {time.time() - start:.2f}s")

    # 异步提交
    with multiprocessing.Pool() as pool:
        results = [pool.apply_async(heavy_computation, (n,)) for n in data]
        outputs = [r.get(timeout=10) for r in results]
        print(f"结果: {outputs}")
```

### 进程间通信：Queue 和 Pipe

```python
import multiprocessing

def sender(conn):
    """通过 Pipe 发送数据"""
    conn.send({"key": "value", "numbers": [1, 2, 3]})
    conn.close()

def queue_producer(queue):
    """通过 Queue 发送数据"""
    for i in range(5):
        queue.put(f"item-{i}")
    queue.put(None)  # 结束信号

def queue_consumer(queue):
    """从 Queue 接收数据"""
    while True:
        item = queue.get()
        if item is None:
            break
        print(f"消费: {item}")

if __name__ == "__main__":
    # Pipe 示例
    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=sender, args=(child_conn,))
    p.start()
    print("收到:", parent_conn.recv())
    p.join()

    # Queue 示例
    q = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=queue_producer, args=(q,))
    p2 = multiprocessing.Process(target=queue_consumer, args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
```

## concurrent.futures

`concurrent.futures` 提供了高级的并发执行接口，统一了线程池和进程池。

### ThreadPoolExecutor

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import requests

def fetch_url(url):
    """模拟网络请求"""
    time.sleep(0.5)
    return f"获取 {url} 完成，长度: {len(url)}"

urls = [
    "https://example.com/page1",
    "https://example.com/page2",
    "https://example.com/page3",
    "https://example.com/page4",
    "https://example.com/page5",
]

# map: 按顺序返回结果
with ThreadPoolExecutor(max_workers=3) as executor:
    results = executor.map(fetch_url, urls)
    for url, result in zip(urls, results):
        print(result)

# submit + as_completed: 谁先完成先返回
with ThreadPoolExecutor(max_workers=3) as executor:
    future_to_url = {executor.submit(fetch_url, url): url for url in urls}

    for future in as_completed(future_to_url):
        url = future_to_url[future]
        try:
            data = future.result()
            print(f"{url}: {data}")
        except Exception as e:
            print(f"{url} 出错: {e}")
```

### ProcessPoolExecutor

```python
from concurrent.futures import ProcessPoolExecutor
import math

def is_prime(n):
    """判断是否为素数"""
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def count_primes_in_range(start, end):
    """统计范围内素数个数"""
    return sum(1 for n in range(start, end) if is_prime(n))

if __name__ == "__main__":
    # 将大范围划分为多个子范围
    ranges = [
        (1, 250000),
        (250000, 500000),
        (500000, 750000),
        (750000, 1000000),
    ]

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(count_primes_in_range, s, e) for s, e in ranges]
        total = sum(f.result() for f in futures)
        print(f"1 到 1000000 之间的素数个数: {total}")
```

## GIL 详解

GIL（Global Interpreter Lock，全局解释器锁）是 CPython 解释器中的一把互斥锁，它保证同一时刻只有一个线程执行 Python 字节码。

### GIL 的影响

```python
import threading
import time
import multiprocessing

def cpu_bound():
    """CPU 密集型任务"""
    total = 0
    for i in range(10_000_000):
        total += i
    return total

# 多线程执行 CPU 密集型任务（受 GIL 限制）
start = time.time()
threads = [threading.Thread(target=cpu_bound) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"多线程: {time.time() - start:.2f}s")

# 多进程执行 CPU 密集型任务（绕过 GIL）
if __name__ == "__main__":
    start = time.time()
    processes = [multiprocessing.Process(target=cpu_bound) for _ in range(4)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print(f"多进程: {time.time() - start:.2f}s")
```

### GIL 不影响的场景

```python
import threading
import time

def io_bound():
    """IO 密集型任务（GIL 会在 IO 等待时释放）"""
    time.sleep(1)  # sleep 会释放 GIL

# 多线程处理 IO 密集型任务效果好
start = time.time()
threads = [threading.Thread(target=io_bound) for _ in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"IO 多线程: {time.time() - start:.2f}s")  # 约 1 秒（并行）
```

### 选择策略

```
任务类型          推荐方案
──────────────────────────────────
IO 密集型         threading / asyncio
CPU 密集型         multiprocessing
混合型             concurrent.futures.ProcessPoolExecutor
```

## 线程安全的数据结构

### queue 模块

```python
from queue import Queue, LifoQueue, PriorityQueue
import threading

# Queue: 先进先出
q = Queue(maxsize=10)
q.put("first")
q.put("second")
print(q.get())  # first

# LifoQueue: 后进先出（栈）
lifo = LifoQueue()
lifo.put("a")
lifo.put("b")
print(lifo.get())  # b

# PriorityQueue: 优先级队列
pq = PriorityQueue()
pq.put((2, "低优先级"))
pq.put((1, "高优先级"))
print(pq.get())  # (1, '高优先级')

# 生产者-消费者模式
def producer(queue):
    for i in range(5):
        queue.put(f"任务-{i}")
    queue.put(None)

def consumer(queue):
    while True:
        item = queue.get()
        if item is None:
            break
        print(f"处理: {item}")

queue = Queue()
threading.Thread(target=producer, args=(queue,)).start()
threading.Thread(target=consumer, args=(queue,)).start()
```

## 总结

| 工具 | 用途 | GIL 影响 |
|------|------|---------|
| threading | IO 并发 | 不影响 IO 等待 |
| multiprocessing | CPU 并行 | 绕过 GIL |
| concurrent.futures | 统一接口 | 可选择线程或进程 |
| queue | 线程安全队列 | 内置锁保护 |
| asyncio | 高并发 IO | 单线程，无 GIL 问题 |
