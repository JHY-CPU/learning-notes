# 负载均衡模拟 (Load Balancer Simulator)

## 项目需求与功能分析

负载均衡是分布式系统的核心组件。本项目模拟多种负载均衡算法，可视化请求分配过程和服务器负载，帮助理解不同算法的特性。

### 核心功能

- 服务器集群管理（添加、移除、设置权重）
- 多种负载均衡算法（轮询、加权轮询、最少连接、一致性哈希）
- 请求模拟与分配可视化
- 服务器负载实时统计
- 算法性能对比

## 核心算法原理

### 轮询 (Round Robin)

依次将请求分配给每个服务器，循环往复。最简单的策略。

### 加权轮询 (Weighted Round Robin)

根据服务器权重分配请求。权重高的服务器处理更多请求。

### 最少连接 (Least Connections)

将请求分配给当前连接数最少的服务器。适合处理时间差异大的场景。

### 一致性哈希 (Consistent Hashing)

将请求和服务器映射到哈希环上，请求分配给顺时针方向最近的服务器。服务器增减时只影响少量请求的映射。

## 完整代码实现

```python
import hashlib
import bisect
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import random


@dataclass
class Server:
    name: str
    weight: int = 1
    connections: int = 0
    total_requests: int = 0
    active: bool = True


class LoadBalancer:
    """负载均衡器基类"""
    def __init__(self, servers: List[Server] = None):
        self.servers = servers or []
        self.request_log: List[Tuple[str, str]] = []  # (request_id, server_name)

    def add_server(self, server: Server):
        self.servers.append(server)

    def remove_server(self, name: str):
        self.servers = [s for s in self.servers if s.name != name]

    def route(self, request_id: str) -> Optional[Server]:
        raise NotImplementedError

    def stats(self) -> Dict:
        return {s.name: {'connections': s.connections, 'total': s.total_requests}
                for s in self.servers}


class RoundRobinBalancer(LoadBalancer):
    """轮询负载均衡"""

    def __init__(self, servers=None):
        super().__init__(servers)
        self.index = 0

    def route(self, request_id: str) -> Optional[Server]:
        active = [s for s in self.servers if s.active]
        if not active:
            return None
        server = active[self.index % len(active)]
        self.index += 1
        server.connections += 1
        server.total_requests += 1
        self.request_log.append((request_id, server.name))
        return server


class WeightedRoundRobinBalancer(LoadBalancer):
    """加权轮询"""

    def __init__(self, servers=None):
        super().__init__(servers)
        self._build_weight_list()

    def _build_weight_list(self):
        self.weight_list = []
        for s in self.servers:
            if s.active:
                self.weight_list.extend([s.name] * s.weight)

    def route(self, request_id: str) -> Optional[Server]:
        if not self.weight_list:
            return None
        name = self.weight_list[self.index % len(self.weight_list)]
        self.index += 1
        server = next(s for s in self.servers if s.name == name)
        server.connections += 1
        server.total_requests += 1
        self.request_log.append((request_id, server.name))
        return server


class LeastConnectionsBalancer(LoadBalancer):
    """最少连接"""

    def route(self, request_id: str) -> Optional[Server]:
        active = [s for s in self.servers if s.active]
        if not active:
            return None
        server = min(active, key=lambda s: s.connections)
        server.connections += 1
        server.total_requests += 1
        self.request_log.append((request_id, server.name))
        return server

    def release(self, server: Server):
        """释放连接"""
        server.connections = max(0, server.connections - 1)


class ConsistentHashBalancer(LoadBalancer):
    """一致性哈希"""

    def __init__(self, servers=None, virtual_nodes: int = 150):
        super().__init__(servers)
        self.virtual_nodes = virtual_nodes
        self.ring: List[Tuple[int, str]] = []
        self._build_ring()

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def _build_ring(self):
        self.ring = []
        for server in self.servers:
            if server.active:
                for i in range(self.virtual_nodes):
                    h = self._hash(f"{server.name}:{i}")
                    self.ring.append((h, server.name))
        self.ring.sort()

    def route(self, request_id: str) -> Optional[Server]:
        if not self.ring:
            return None
        h = self._hash(request_id)
        idx = bisect.bisect_left(self.ring, (h, ''))
        idx = idx % len(self.ring)
        name = self.ring[idx][1]
        server = next(s for s in self.servers if s.name == name)
        server.connections += 1
        server.total_requests += 1
        self.request_log.append((request_id, server.name))
        return server


def simulate(balancer: LoadBalancer, num_requests: int = 200,
             with_release: bool = False):
    """模拟请求"""
    for i in range(num_requests):
        req_id = f"req_{i}"
        server = balancer.route(req_id)
        if server and with_release and hasattr(balancer, 'release'):
            # 随机释放一些连接
            if random.random() < 0.7:
                balancer.release(server)

    print(f"\n{'服务器':<12} {'总请求':<10} {'当前连接':<10} {'占比':<10}")
    print("-" * 44)
    stats = balancer.stats()
    total = sum(v['total'] for v in stats.values())
    for name, v in sorted(stats.items()):
        pct = v['total'] / total * 100 if total > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"{name:<12} {v['total']:<10} {v['connections']:<10} {pct:>5.1f}% {bar}")


def compare_algorithms(num_requests=500):
    """对比所有算法"""
    print("=" * 60)
    print(f"负载均衡算法对比 ({num_requests} 次请求)")
    print("=" * 60)

    servers = [
        Server("Server-A", weight=3),
        Server("Server-B", weight=2),
        Server("Server-C", weight=1),
    ]

    balancers = {
        '轮询': RoundBalancer([Server(s.name, s.weight) for s in servers]),
        '加权轮询': WeightedRoundRobinBalancer([Server(s.name, s.weight) for s in servers]),
        '最少连接': LeastConnectionsBalancer([Server(s.name, s.weight) for s in servers]),
        '一致性哈希': ConsistentHashBalancer([Server(s.name, s.weight) for s in servers]),
    }

    for name, balancer in balancers.items():
        print(f"\n--- {name} ---")
        simulate(balancer, num_requests)


# 别名修复
RoundBalancer = RoundRobinBalancer


def demo():
    compare_algorithms(500)


if __name__ == '__main__':
    demo()
```

## 测试用例

```python
import unittest

class TestLoadBalancer(unittest.TestCase):
    def test_round_robin(self):
        servers = [Server("A"), Server("B"), Server("C")]
        balancer = RoundRobinBalancer(servers)
        names = [balancer.route(f"r{i}").name for i in range(6)]
        self.assertEqual(names, ['A','B','C','A','B','C'])

    def test_least_connections(self):
        servers = [Server("A"), Server("B")]
        balancer = LeastConnectionsBalancer(servers)
        balancer.route("r0")  # A: 1
        balancer.route("r1")  # B: 1
        balancer.route("r2")  # A: 2 or B: 2
        # 最少连接应该均匀分配
        self.assertEqual(servers[0].total_requests + servers[1].total_requests, 3)

    def test_consistent_hash(self):
        servers = [Server("A"), Server("B"), Server("C")]
        balancer = ConsistentHashBalancer(servers)
        # 同一请求应该总是路由到同一服务器
        s1 = balancer.route("same_key").name
        s2 = balancer.route("same_key").name
        self.assertEqual(s1, s2)

    def test_weighted_round_robin(self):
        servers = [Server("A", weight=3), Server("B", weight=1)]
        balancer = WeightedRoundRobinBalancer(servers)
        for i in range(8): balancer.route(f"r{i}")
        self.assertEqual(servers[0].total_requests, 6)
        self.assertEqual(servers[1].total_requests, 2)

if __name__ == '__main__':
    unittest.main()
```

## 扩展方向

1. **健康检查**：自动检测并移除故障服务器
2. **加权最少连接**：结合权重和连接数
3. **源地址哈希**：基于客户端 IP 的会话保持
4. **响应时间优先**：根据服务器响应时间动态调整权重
5. **GeoDNS**：根据客户端地理位置分配最近的服务器
6. **熔断机制**：服务器过载时自动熔断
7. **可视化仪表盘**：实时图形化展示负载分布
