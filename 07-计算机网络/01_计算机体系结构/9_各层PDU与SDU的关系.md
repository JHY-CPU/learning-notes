# 10_各层PDU与SDU的关系

## 核心概念

- **SDU（Service Data Unit）**：服务数据单元，上层传递给本层的数据
- **PDU（Protocol Data Unit）**：协议数据单元，本层封装后向外传递的数据
- **ICI（Interface Control Information）**：接口控制信息，层与层之间交换的控制信息

### 核心公式

$$
\text{PDU} = \text{本层首部} + \text{SDU}
$$

$$
\text{本层的 PDU} = \text{下层的 SDU}
$$

$$
\text{本层的 SDU} = \text{上层的 PDU}
$$

### 各层SDU与PDU对应关系

```
应用层：  SDU = 用户数据          PDU = 报文 = H₇ + 用户数据
传输层：  SDU = 报文              PDU = 报文段 = H₄ + 报文
网络层：  SDU = 报文段            PDU = 数据报 = H₃ + 报文段
数据链路层：SDU = 数据报          PDU = 帧 = H₂ + 数据报 + T₂
```

## 原理分析

### 封装时的SDU-PDU关系

发送端每一层的操作：

1. 接收上层的 **PDU**（作为本层的 **SDU**）
2. 添加本层的 **首部**（控制信息）
3. 形成本层的 **PDU**，传递给下层

$$
\text{PDU}_n = \text{Header}_n + \text{SDU}_n = \text{Header}_n + \text{PDU}_{n+1}
$$

### 分段与拼接

当上层的PDU过大时，本层可能需要将其**分段**：

$$
\text{SDU}_n \xrightarrow{\text{分段}} \text{SDU}_{n,1} + \text{SDU}_{n,2} + \cdots + \text{SDU}_{n,k}
$$

$$
\text{PDU}_{n,i} = \text{Header}_n + \text{SDU}_{n,i} \quad (i = 1,2,\ldots,k)
$$

反之，多个小的SDU也可以**拼接**成一个PDU：

$$
\text{PDU}_n = \text{Header}_n + \text{SDU}_{n,1} + \text{SDU}_{n,2} + \cdots
$$

### IP分片示例

- 以太网MTU（最大传输单元）= 1500字节
- 如果IP数据报总长度 > 1500字节，需要在IP层分片
- 每个分片都是独立的IP数据报，具有各自的IP首部

## 直观理解

- **SDU和PDU的关系**像工厂流水线：
  - 上游车间交给你半成品（SDU）
  - 你给它加上包装和标签（本层首部）
  - 变成成品（PDU）交给下游车间
- **分段**像切蛋糕：一个大蛋糕（大SDU）切成几小块，每块分别包装（各自加首部变成小PDU）
- **拼接**像拼盘：几个小菜（小SDU）拼成一个拼盘（一个PDU），外加一个大罩子（首部）

## 协议关联

- SDU/PDU的概念在408中通常以选择题形式考查
- TCP的分段（segmentation）和IP的分片（fragmentation）是两件不同的事：
  - TCP分段：传输层将应用层数据切分
  - IP分片：网络层将过大的数据报切分
- MTU限制导致的IP分片是网络层的重要考点
- PDU大小受限于下层的MTU，这是分层设计中自上而下约束的体现
