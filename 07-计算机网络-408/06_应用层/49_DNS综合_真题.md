# 49_DNS综合_真题

## 核心概念

- **真题来源**：408统考及各校自命题中涉及DNS综合计算
- **高频考点**：DNS查询时间、缓存影响、记录配置
- **出题规律**：通常结合HTTP和TCP综合考查

## 原理分析

### 真题1（计算题）

**题目**：假设DNS查询中，本地DNS到根DNS的RTT=10ms，到TLD的RTT=20ms，到权威DNS的RTT=30ms。主机到本地DNS的RTT=5ms。无缓存情况下，迭代查询需要多长时间？

**参考答案**：
简化模型（忽略响应时间）：
$$T = RTT_{host-local} + RTT_{local-root} + RTT_{local-tld} + RTT_{local-auth}$$
$$T = 5 + 10 + 20 + 30 = 65ms$$

### 真题2（计算题）

**题目**：本地DNS缓存了.com TLD地址，缓存命中率70%。查询`www.abc.com`，命中时查询时间为25ms，未命中时为65ms。求平均查询时间。

**参考答案**：
$$T_{avg} = 0.7 \times 25 + 0.3 \times 65 = 17.5 + 19.5 = 37ms$$

### 真题3（综合题）

**题目**：用户访问`http://www.example.com`，DNS解析时间40ms，TCP握手时间30ms，HTTP请求响应时间20ms。求总时间。

**参考答案**：
$$T_{total} = T_{DNS} + T_{TCP} + T_{HTTP} = 40 + 30 + 20 = 90ms$$

### 真题4（记录配置）

**题目**：某公司需要配置邮件服务器（mx.company.com, 200.1.1.1），请列出需要的DNS记录。

**参考答案**：
1. A记录：`(mx.company.com, 200.1.1.1, A)`
2. MX记录：`(company.com, mx.company.com, MX, 10)`

## 直观理解

**真题做题技巧**：
- 先画DNS查询流程图
- 标注每段RTT
- 注意缓存跳过的步骤
- 结合HTTP和TCP的总时间

## 协议关联

- **DNS + TCP + HTTP** = Web访问完整时间
- **DNS + SMTP + MX** = 邮件系统
- **408常见组合**：DNS解析 + TCP握手 + HTTP请求 = 总延迟
