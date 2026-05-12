# 06_DNS解析过程

## 核心概念

- **DNS解析（Name Resolution）**：将域名转换为IP地址的过程
- **两种查询方式**：
  - **递归查询（Recursive Query）**：被请求的服务器代替客户端完成全部查询
  - **迭代查询（Iterative Query）**：被请求的服务器返回下一级服务器地址，客户端自己继续查询
- **实际DNS解析过程**：主机到本地DNS用递归，本地DNS到其他服务器用迭代
- **408考试重点**：区分递归查询和迭代查询，画出完整的DNS查询流程图

## 原理分析

### 迭代查询过程（408重点）

以查询`www.example.com`为例：

1. **主机 → 本地DNS服务器**：递归查询请求
2. **本地DNS → 根DNS服务器**：查询`www.example.com`
   - 根DNS返回：`com`的TLD服务器IP地址
3. **本地DNS → com TLD服务器**：查询`www.example.com`
   - com TLD返回：`example.com`的权威DNS服务器IP地址
4. **本地DNS → example.com权威DNS服务器**：查询`www.example.com`
   - 权威DNS返回：`www.example.com`的IP地址
5. **本地DNS → 主机**：返回最终IP地址

### 递归查询过程

1. **主机 → 本地DNS**：递归查询
2. **本地DNS → 根DNS**：递归查询
   - 根DNS代替本地DNS向TLD查询
3. **根DNS → TLD**：递归查询
   - TLD代替根DNS向权威DNS查询
4. **TLD → 权威DNS**：查询并返回结果
5. 逐层返回：权威DNS → TLD → 根DNS → 本地DNS → 主机

### 递归与迭代对比

| 特性 | 递归查询 | 迭代查询 |
|------|---------|---------|
| 负担 | 被查询服务器负担重 | 客户端负担重 |
| 实际使用 | 主机→本地DNS | 本地DNS→其他服务器 |
| 查询方向 | 单向传递 | 客户端多次查询 |
| 服务器压力 | 大 | 小 |

### DNS缓存机制

- **TTL（生存时间）**：每条DNS记录都有TTL值
- 本地DNS缓存查询结果，减少重复查询
- 权威DNS更新后，需要等待TTL过期才能生效
- 缓存命中可以跳过某些查询步骤

## 直观理解

**迭代查询就像问路**：
- 你问路人A："请问xxx在哪里？"
- A说："我不知道，但B可能知道，你去问B"
- 你去问B，B说："我不清楚，但C知道"
- 你去问C，C告诉你答案

**递归查询就像找秘书办事**：
- 你告诉秘书："帮我查一下xxx"
- 秘书自己去问A，A不知道但告诉秘书去问B
- 秘书问B，B告诉秘书去问C
- 秘书问C，拿到答案回来告诉你
- 你只需要等，不用自己跑

**记忆技巧**：
- 递归 = "你帮我做，做完告诉我"（服务器负担大）
- 迭代 = "你告诉我找谁，我自己去"（客户端负担大）
- 实际 = 主机对本地DNS递归，本地DNS对其他服务器迭代（折中方案）

## 代码示例

### 使用 dig +trace 追踪 DNS 迭代查询过程

```bash
# +trace 模拟本地DNS的迭代查询过程（408考试重点）
dig +trace www.example.com

# 输出示例（逐级查询）：
# ① 从根DNS服务器(.)开始
# .           518400  IN  NS  a.root-servers.net.
# ② 查询 .com 的TLD服务器
# com.        172800  IN  NS  a.gtld-servers.net.
# ③ 查询 example.com 的权威DNS服务器
# example.com. 172800 IN  NS  a.iana-servers.net.
# ④ 权威DNS返回最终IP
# www.example.com. 86400 IN A  93.184.216.34
```

### 使用 Python 模拟 DNS 递归解析过程

```python
import dns.resolver

def trace_dns_resolution(domain):
    """模拟DNS解析过程，理解递归/迭代查询"""
    print(f"正在解析: {domain}")
    print("=" * 50)

    # 1. 查询根服务器（根提示文件）
    print("[步骤1] 查询根DNS服务器 → 获取TLD服务器地址")

    # 2. 查询TLD服务器（如 .com）
    tld = domain.split('.')[-1]
    ns_records = dns.resolver.resolve(f'{tld}', 'NS')
    print(f"[步骤2] 查询 .{tld} TLD服务器:")
    for ns in ns_records:
        print(f"  → {ns}")

    # 3. 查询权威DNS服务器
    try:
        ns_records = dns.resolver.resolve(domain, 'NS')
        print(f"[步骤3] 查询 {domain} 权威DNS服务器:")
        for ns in ns_records:
            print(f"  → {ns}")
    except Exception as e:
        print(f"[步骤3] 权威DNS查询: {e}")

    # 4. 获取最终IP地址
    a_records = dns.resolver.resolve(domain, 'A')
    print(f"[步骤4] 获取A记录:")
    for ip in a_records:
        print(f"  → {domain} → {ip}")

# 执行追踪
trace_dns_resolution('www.example.com')
```

### 使用 Python 实现简单的 DNS 缓存模拟

```python
import time

class SimpleDNSCache:
    """模拟本地DNS服务器的缓存机制"""
    def __init__(self):
        self.cache = {}  # {domain: (ip, expire_time)}

    def resolve(self, domain, ttl=300):
        """带缓存的DNS解析"""
        now = time.time()

        # 检查缓存是否命中
        if domain in self.cache:
            ip, expire_time = self.cache[domain]
            if now < expire_time:
                print(f"[缓存命中] {domain} → {ip}")
                return ip
            else:
                print(f"[缓存过期] {domain} TTL已到，重新查询")
                del self.cache[domain]

        # 缓存未命中，模拟向DNS服务器查询
        import socket
        ip = socket.gethostbyname(domain)

        # 存入缓存
        self.cache[domain] = (ip, now + ttl)
        print(f"[查询并缓存] {domain} → {ip} (TTL={ttl}s)")
        return ip

# 演示
cache = SimpleDNSCache()
cache.resolve('www.example.com')      # 第一次：查询并缓存
cache.resolve('www.example.com')      # 第二次：缓存命中
cache.resolve('www.google.com')       # 另一个域名：查询并缓存
```

## 协议关联

- **DNS与UDP**：查询使用UDP，端口53，减少TCP握手开销
- **DNS与TCP**：响应超过512字节或区域传送时用TCP
- **DNS与HTTP**：HTTP请求前必须先完成DNS解析（增加延迟）
- **408陷阱**：实际DNS解析中，主机到本地DNS是递归查询，本地DNS到其他服务器是迭代查询
