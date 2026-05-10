# API 限流器 (Rate Limiter)

## 项目需求与功能分析

API 限流是保护后端服务的核心手段。本项目实现令牌桶、漏桶、滑动窗口三种经典限流算法，帮助理解速率限制的原理和适用场景。

### 核心功能

- 令牌桶算法 (Token Bucket)
- 漏桶算法 (Leaky Bucket)
- 固定窗口计数器 (Fixed Window)
- 滑动窗口计数器 (Sliding Window)
- 滑动窗口日志 (Sliding Window Log)
- 限流策略性能对比

### 应用场景

- API 网关限流
- 防止 DDoS 攻击
- 第三方 API 调用配额控制
- 游戏中操作频率限制
- 登录尝试次数限制

## 核心算法原理

### 令牌桶 (Token Bucket)

以固定速率向桶中添加令牌，每个请求消耗一个令牌。桶有最大容量限制。

- 允许突发流量（桶中有令牌即可通过）
- 长期平均速率受令牌添加速率限制

### 漏桶 (Leaky Bucket)

请求进入队列（桶），以固定速率流出处理。

- 严格平滑流量，不允许突发
- 桶满则拒绝新请求

### 滑动窗口 (Sliding Window)

统计过去 N 秒内的请求数，若超过阈值则拒绝。使用时间戳列表或加权计算。

## 完整代码实现

```python
import time
import threading
from collections import deque
from typing import Optional
from dataclasses import dataclass


class TokenBucket:
    """令牌桶限流器"""

    def __init__(self, rate: float, capacity: float):
        """
        rate: 每秒添加的令牌数
        capacity: 桶最大容量
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_time = time.time()
        self.lock = threading.Lock()

    def allow(self, tokens: int = 1) -> bool:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_time = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


class LeakyBucket:
    """漏桶限流器"""

    def __init__(self, rate: float, capacity: float):
        """
        rate: 每秒流出的请求数
        capacity: 桶容量
        """
        self.rate = rate
        self.capacity = capacity
        self.water = 0.0
        self.last_time = time.time()
        self.lock = threading.Lock()

    def allow(self) -> bool:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_time
            self.water = max(0, self.water - elapsed * self.rate)
            self.last_time = now

            if self.water < self.capacity:
                self.water += 1
                return True
            return False


class FixedWindowCounter:
    """固定窗口计数器"""

    def __init__(self, max_requests: int, window_seconds: int = 1):
        self.max_requests = max_requests
        self.window = window_seconds
        self.count = 0
        self.window_start = time.time()
        self.lock = threading.Lock()

    def allow(self) -> bool:
        with self.lock:
            now = time.time()
            if now - self.window_start >= self.window:
                self.count = 0
                self.window_start = now
            if self.count < self.max_requests:
                self.count += 1
                return True
            return False


class SlidingWindowCounter:
    """滑动窗口计数器"""

    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: deque = deque()
        self.lock = threading.Lock()

    def allow(self) -> bool:
        with self.lock:
            now = time.time()
            # 移除窗口外的请求
            while self.requests and now - self.requests[0] >= self.window:
                self.requests.popleft()

            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False


class SlidingWindowLog:
    """滑动窗口日志"""

    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.timestamps: deque = deque()
        self.lock = threading.Lock()

    def allow(self) -> bool:
        with self.lock:
            now = time.time()
            while self.timestamps and now - self.timestamps[0] >= self.window:
                self.timestamps.popleft()

            if len(self.timestamps) < self.max_requests:
                self.timestamps.append(now)
                return True
            return False


class RateLimiter:
    """统一限流器接口"""

    def __init__(self, algorithm: str, **kwargs):
        self.algorithms = {
            'token_bucket': lambda: TokenBucket(kwargs.get('rate', 10), kwargs.get('capacity', 20)),
            'leaky_bucket': lambda: LeakyBucket(kwargs.get('rate', 10), kwargs.get('capacity', 20)),
            'fixed_window': lambda: FixedWindowCounter(kwargs.get('max_requests', 10), kwargs.get('window', 1)),
            'sliding_window': lambda: SlidingWindowCounter(kwargs.get('max_requests', 10), kwargs.get('window', 1)),
            'sliding_log': lambda: SlidingWindowLog(kwargs.get('max_requests', 10), kwargs.get('window', 1)),
        }
        if algorithm not in self.algorithms:
            raise ValueError(f"未知算法: {algorithm}")
        self.limiter = self.algorithms[algorithm]()
        self.name = algorithm

    def allow(self) -> bool:
        return self.limiter.allow()


def simulate(rate_limiter, total_requests=100, requests_per_second=20):
    """模拟请求"""
    allowed = 0
    rejected = 0

    for i in range(total_requests):
        if rate_limiter.allow():
            allowed += 1
        else:
            rejected += 1
        time.sleep(1.0 / requests_per_second)

    return allowed, rejected


def compare_algorithms():
    """对比不同限流算法"""
    print("=" * 55)
    print("限流算法对比 (10秒, 每秒20个请求, 限制每秒10个)")
    print("=" * 55)

    configs = [
        ('令牌桶', 'token_bucket', {'rate': 10, 'capacity': 15}),
        ('漏桶', 'leaky_bucket', {'rate': 10, 'capacity': 15}),
        ('固定窗口', 'fixed_window', {'max_requests': 10, 'window': 1}),
        ('滑动窗口', 'sliding_window', {'max_requests': 10, 'window': 1}),
        ('滑动日志', 'sliding_log', {'max_requests': 10, 'window': 1}),
    ]

    print(f"\n{'算法':<12} {'通过':<8} {'拒绝':<8} {'通过率':<10}")
    print("-" * 40)

    for name, algo, kwargs in configs:
        limiter = RateLimiter(algo, **kwargs)
        # 快速模拟（不实际等待）
        allowed = 0
        rejected = 0
        for i in range(100):
            if limiter.allow():
                allowed += 1
            else:
                rejected += 1
            # 模拟时间推进
            if hasattr(limiter.limiter, 'last_time'):
                limiter.limiter.last_time += 0.05
            if hasattr(limiter.limiter, 'window_start'):
                if i % 20 == 0 and i > 0:
                    limiter.limiter.count = 0
                    limiter.limiter.window_start += 1

        rate = allowed / 100 * 100
        print(f"{name:<12} {allowed:<8} {rejected:<8} {rate:>5.1f}%")


def demo():
    # 令牌桶演示
    print("令牌桶限流 (速率=5/s, 容量=10):")
    tb = TokenBucket(rate=5, capacity=10)
    for i in range(15):
        result = "通过" if tb.allow() else "拒绝"
        print(f"  请求 {i+1:>2}: {result}")
    print()

    compare_algorithms()


if __name__ == '__main__':
    demo()
```

## 测试用例

```python
import unittest

class TestRateLimiter(unittest.TestCase):
    def test_token_bucket(self):
        tb = TokenBucket(rate=10, capacity=5)
        # 满桶时前5个应该通过
        for _ in range(5): self.assertTrue(tb.allow())
        # 第6个应被拒绝
        self.assertFalse(tb.allow())

    def test_leaky_bucket(self):
        lb = LeakyBucket(rate=10, capacity=3)
        self.assertTrue(lb.allow())
        self.assertTrue(lb.allow())
        self.assertTrue(lb.allow())
        self.assertFalse(lb.allow())

    def test_fixed_window(self):
        fw = FixedWindowCounter(max_requests=3, window=1)
        self.assertTrue(fw.allow())
        self.assertTrue(fw.allow())
        self.assertTrue(fw.allow())
        self.assertFalse(fw.allow())

    def test_sliding_window(self):
        sw = SlidingWindowCounter(max_requests=2, window=1)
        self.assertTrue(sw.allow())
        self.assertTrue(sw.allow())
        self.assertFalse(sw.allow())

    def test_rate_limiter_interface(self):
        rl = RateLimiter('token_bucket', rate=5, capacity=10)
        self.assertTrue(rl.allow())

if __name__ == '__main__':
    unittest.main()
```

## 扩展方向

1. **分布式限流**：使用 Redis 实现跨节点限流
2. **多级限流**：IP 级、用户级、API 级组合限流
3. **自适应限流**：根据系统负载动态调整限流阈值
4. **优先级队列**：VIP 用户优先通过
5. **限流降级**：超限时返回缓存数据而非拒绝
6. **限流可视化**：实时监控面板
7. **限流规则配置**：动态配置限流策略
