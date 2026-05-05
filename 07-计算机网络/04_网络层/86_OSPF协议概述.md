# 50_OSPF协议概述

## 核心概念

- **OSPF（Open Shortest Path First）**：开放最短路径优先协议
  - 类型：**链路状态**路由协议（Link State）
  - 范围：IGP（自治系统内部）
  - 度量：**代价（Cost）**，通常与带宽成反比
  - 使用**Dijkstra算法**计算最短路径
  - 直接封装在**IP数据报**中（协议号**89**）
  - 版本：OSPFv2（IPv4）、OSPFv3（IPv6）

- **OSPF vs RIP的关键区别**：

| 特性 | OSPF | RIP |
|------|------|-----|
| 算法 | 链路状态（Dijkstra） | 距离向量（Bellman-Ford） |
| 交换信息 | 链路状态通告（LSA） | 整个路由表 |
| 交换范围 | 泛洪到整个AS | 仅相邻邻居 |
| 度量 | 代价（带宽） | 跳数 |
| 收敛速度 | 快 | 慢 |
| 网络规模 | 大型网络 | 小型网络 |
| 传输协议 | IP（协议号89） | UDP 520 |
| 层次化 | 支持区域划分 | 不支持 |

- **408考点**：
  - OSPF是链路状态协议
  - 使用Dijkstra算法
  - 泛洪链路状态信息
  - 直接封装在IP中（不使用TCP/UDP）
  - 支持区域划分

## 原理分析

### OSPF的工作过程

```
步骤1: 发现邻居
  每个路由器向邻居发送Hello分组
  建立邻居关系

步骤2: 泛洪链路状态
  每个路由器生成链路状态通告（LSA）
  LSA描述自己连接的链路和代价
  泛洪到整个区域的所有路由器

步骤3: 构建拓扑图
  每个路由器收集所有LSA
  构建完整的网络拓扑图（链路状态数据库LSDB）

步骤4: 运行Dijkstra算法
  以自己为根，计算到所有目的网络的最短路径树
  生成路由表
```

### OSPF的五种分组类型

| 类型 | 名称 | 用途 |
|------|------|------|
| 1 | Hello | 发现和维护邻居关系 |
| 2 | Database Description | 描述LSDB摘要 |
| 3 | Link State Request | 请求特定LSA |
| 4 | Link State Update | 发送LSA（携带LSA） |
| 5 | Link State Acknowledgment | 确认收到LSA |

### OSPF的区域划分

```
自治系统（AS）
  ├── Area 0（骨干区域，Backbone）
  │     ├── ABR（区域边界路由器）
  │     ├── 内部路由器
  │     └── ...
  ├── Area 1
  │     ├── ABR
  │     ├── 内部路由器
  │     └── ...
  └── Area 2
        └── ...
```

- 所有区域必须与Area 0（骨干区域）相连
- ABR（Area Border Router）连接多个区域
- 区域内使用OSPF，区域间通过ABR传递路由汇总信息

## 直观理解

**类比**：OSPF像全城广播的交通信息
- 每个路口（路由器）广播自己连接的道路状况（LSA）
- 所有路口都知道全城的路况（泛洪）
- 每个路口自己计算最佳路线（Dijkstra）
- 与RIP不同：RIP是邻居间传话，OSPF是全城广播

**记忆口诀**：
- "OSPF链路状态全网泛洪，Dijkstra自己算路由"
- "Hello发现邻居，LSA泛洪全网，Dijkstra算最短路"

## 协议关联

### OSPF与IP
- OSPF直接封装在IP中，协议号89
- 不经过TCP/UDP传输层
- 这使OSPF更高效，但也意味着OSPF自己需要处理可靠性

### OSPF与RIP
- OSPF是RIP的升级替代
- 收敛更快，支持更大规模网络
- 没有跳数限制
- 支持负载均衡（等代价多路径）

### 考试要点
- OSPF是链路状态协议，RIP是距离向量协议
- OSPF使用Dijkstra算法
- OSPF直接封装在IP中（协议号89）
- OSPF支持区域划分，Area 0是骨干区域
