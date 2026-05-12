# 74_磁盘调度算法_FCFS

## 核心概念
- **FCFS (First Come First Served)**: 先来先服务
- **原理**: 按照请求到达的先后顺序依次服务
- **特点**: 公平、简单, 但磁头移动距离大, 效率低

## 原理分析

### 算法描述

```
请求队列: 按到达顺序排列
磁头: 从当前位置出发, 依次访问队列中的每个磁道
```

### 示例

**题目**: 磁盘请求队列: 98, 183, 37, 122, 14, 124, 65, 67。磁头当前位置: 53。

**解**:

访问顺序: 53 → 98 → 183 → 37 → 122 → 14 → 124 → 65 → 67

```
磁道号:   0   14  37      53  65 67    98      122 124      183
          |    |   |       |   |  |     |        |   |        |
访问:     |    |   |       ●   |  |     ●        |   |        ●
          |    |   ●       |   |  |     |        |   |        |
          |    ●           |   |  |     |        ●   |        |
          |    |           |   |  ●     |        |   |        |
          |    |           |   ●        |        |   |        |
          |    |           |   |  ●     |        |   ●        |
```

**总移动距离**:
$$|98-53| + |183-98| + |37-183| + |122-37| + |14-122| + |124-14| + |65-124| + |67-65|$$
$$= 45 + 85 + 146 + 85 + 108 + 110 + 59 + 2 = 640 \text{ 磁道}$$

**平均寻道距离**: $640 / 8 = 80$ 磁道

### FCFS分析

**优点**:
- 实现最简单(队列)
- 完全公平, 不会产生饥饿
- 每个请求都能得到服务

**缺点**:
- 磁头来回移动, 寻道距离大
- 效率低, 吞吐量小
- 不考虑请求的位置局部性

**适用场景**:
- 负载轻的系统(请求少)
- 实时性要求高的系统(必须按顺序)
- 对公平性有严格要求的场合

### 与其他算法对比(简要)

对于同样的请求序列, 各算法总寻道距离:
- FCFS: 640 磁道
- SSTF: 约 236 磁道 (后续详解)
- SCAN: 约 236 磁道
- C-SCAN: 约 322 磁道

FCFS效率最低。

## 直观理解

**排队买票类比**:
- FCFS = 先来先排队, 不管你住得远近
- 售票员先处理第一个人, 再处理第二个...即使第二个人就在隔壁
- 如果队伍里有人来自很远的地方, 售票员也要先处理他

### 记忆要点
> **"FCFS最公平, 但磁头来回跑, 效率最低"**

## 知识关联

### 与操作系统的FCFS进程调度对比
- 进程FCFS: 按到达顺序调度, 可能有护航效应
- 磁盘FCFS: 按到达顺序服务, 磁头来回移动(类似护航效应)

## 代码/模拟

### Python实现四种磁盘调度算法

```python
"""磁盘调度算法模拟 - 适用于408考研复习"""

def fcfs(requests, head):
    """FCFS: 先来先服务"""
    total_movement = 0
    current = head
    order = []
    for req in requests:
        total_movement += abs(req - current)
        order.append(req)
        current = req
    return order, total_movement

def sstf(requests, head):
    """SSTF: 最短寻道时间优先"""
    remaining = list(requests)
    total_movement = 0
    current = head
    order = []
    while remaining:
        # 找最近的请求
        nearest = min(remaining, key=lambda x: abs(x - current))
        total_movement += abs(nearest - current)
        order.append(nearest)
        current = nearest
        remaining.remove(nearest)
    return order, total_movement

def scan(requests, head, disk_size=200, direction='up'):
    """SCAN: 电梯算法（扫描到端点再反向）"""
    remaining = sorted(requests)
    total_movement = 0
    current = head
    order = []

    if direction == 'up':
        # 先向磁道号增大方向扫描
        up = [r for r in remaining if r >= current]
        down = [r for r in remaining if r < current][::-1]
        order = up
        if up:
            total_movement += abs(up[-1] - current)
            current = up[-1]
        if down:
            total_movement += abs(disk_size - 1 - current)  # 到端点
            current = disk_size - 1
            total_movement += abs(down[0] - current)
            current = down[0]
            order.extend(down)
    else:
        down = [r for r in remaining if r <= current][::-1]
        up = [r for r in remaining if r > current]
        order = down
        if down:
            total_movement += abs(down[-1] - current)
            current = down[-1]
        if up:
            total_movement += abs(0 - current)
            current = 0
            total_movement += abs(up[0] - current)
            current = up[0]
            order.extend(up)

    return order, total_movement

def c_scan(requests, head, disk_size=200):
    """C-SCAN: 循环扫描（单方向服务，返回时快速回到起点）"""
    remaining = sorted(requests)
    total_movement = 0
    current = head

    up = [r for r in remaining if r >= current]
    down = [r for r in remaining if r < current]
    order = up + down

    if up:
        total_movement += abs(up[-1] - current)
        current = up[-1]
    if down:
        # 到达最高端，跳回0号磁道
        total_movement += abs(disk_size - 1 - current)
        total_movement += (disk_size - 1)  # 快速返回
        current = 0
        total_movement += abs(down[-1] - current)
        current = down[-1]

    return order, total_movement

# 笔记中的经典示例
requests = [98, 183, 37, 122, 14, 124, 65, 67]
head = 53

print(f"请求队列: {requests}")
print(f"磁头位置: {head}\n")

for name, algo in [("FCFS", fcfs), ("SSTF", sstf), ("SCAN", scan), ("C-SCAN", c_scan)]:
    order, movement = algo(requests, head)
    print(f"{name:8}: 访问顺序={order}")
    print(f"{'':8}  总寻道距离={movement} 磁道, 平均={movement/len(requests):.1f} 磁道\n")
```

### 408考点
- FCFS的访问顺序就是请求到达顺序
- 计算总寻道距离(各相邻磁道号差的绝对值之和)
- 磁头当前位置已知, 第一步从当前位置到第一个请求
