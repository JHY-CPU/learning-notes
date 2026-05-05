# 54_BGP协议

## 核心概念

- **BGP（Border Gateway Protocol）**：边界网关协议
  - 类型：**路径向量**路由协议（Path Vector）
  - 范围：**EGP**（自治系统之间）
  - 基于**TCP**，端口号**179**
  - 版本：BGP-4（当前使用）

- **BGP的核心功能**：
  - 在AS之间交换**可达性信息**
  - 不是最短路径，而是**策略路由**（考虑政治、经济、安全等因素）
  - 防止AS间路由环路

- **BGP的两种会话**：
  - **eBGP（External BGP）**：不同AS之间的BGP会话
  - **iBGP（Internal BGP）**：同一AS内部的BGP会话

- **BGP路由信息**：
  - 不仅包含"到某网络可达"，还包含**经过的AS路径（AS-PATH）**
  - AS-PATH = 经过的AS号列表
  - 如果收到的AS-PATH中包含自己的AS号，说明有环路，丢弃

- **408考点**：
  - BGP是路径向量协议
  - BGP使用TCP
  - BGP在AS之间使用（eBGP和iBGP）
  - AS-PATH用于防环
  - BGP是应用层协议（基于TCP）

## 原理分析

### BGP的工作过程

```
1. BGP邻居建立（TCP连接，端口179）
2. 交换OPEN报文，建立BGP会话
3. 交换完整的BGP路由表
4. 之后只发送增量更新（UPDATE报文）
5. 使用KEEPALIVE报文维持连接
6. 出错时发送NOTIFICATION报文
```

### BGP的四种报文

| 报文 | 功能 |
|------|------|
| OPEN | 建立BGP邻居关系 |
| UPDATE | 通告/撤销路由信息 |
| KEEPALIVE | 保活（维持TCP连接） |
| NOTIFICATION | 报告错误，关闭连接 |

### eBGP vs iBGP

```
AS 1 ←——eBGP——→ AS 2
  |               |
 iBGP           iBGP
  |               |
R1←——eBGP——→R2  R3←——eBGP——→R4
```

- eBGP：不同AS的边界路由器之间
- iBGP：同一AS内的路由器之间（确保所有边界路由器知道如何到达外部网络）
- eBGP邻居通常直连
- iBGP邻居不需要直连（通过TCP连接）

### AS-PATH防环

```
AS1通告网络N: AS-PATH = [1]
AS2从AS1学到N: AS-PATH = [2, 1]
AS3从AS2学到N: AS-PATH = [3, 2, 1]
AS1从AS3收到N: AS-PATH = [1, 3, 2, 1]
AS1发现AS-PATH包含自己的AS号(1) → 丢弃！
```

## 直观理解

**类比**：BGP像国家间的外交关系
- 每个AS像一个国家
- eBGP = 国家之间的外交
- iBGP = 国内各省之间的协调
- AS-PATH = 护照上的签证记录
- 策略路由 = 考虑政治因素（某些国家不能经过）

**记忆口诀**：
- "BGP是路径向量，AS-PATH防环"
- "eBGP跨AS，iBGP在AS内"
- "基于TCP 179端口"

## 协议关联

### BGP与其他协议

| 对比 | RIP | OSPF | BGP |
|------|-----|------|-----|
| 类型 | 距离向量 | 链路状态 | 路径向量 |
| 范围 | IGP | IGP | EGP |
| 传输 | UDP | IP | TCP |
| 度量 | 跳数 | 代价 | 策略 |
| 防环 | 水平分割 | 无环路 | AS-PATH |

### 考试要点
- BGP是路径向量协议（不是距离向量，不是链路状态）
- BGP使用TCP 179端口
- eBGP在AS之间，iBGP在AS内部
- AS-PATH包含经过的AS号列表，用于防环和路径选择
- BGP关注策略而非最短路径
