# 23_SNMP详解

## 核心概念

- **SNMP（Simple Network Management Protocol）定义**：TCP/IP网络的标准管理协议
- **SNMP版本**：
  - SNMPv1：基本功能，团体名认证（明文），安全性差
  - SNMPv2c：增加GetBulk操作（批量查询），团体名认证
  - SNMPv3：增加USM（用户安全模型），支持加密和认证
- **SNMP端口**：
  - 管理者→代理：UDP 161
  - 代理→管理者（Trap）：UDP 162
- **SNMP五大操作**：
  - GetRequest / GetNextRequest / GetBulkRequest（查询）
  - SetRequest（设置）
  - Trap / Inform（通知）
- **团体名（Community String）**：SNMPv1/v2c的"密码"，明文传输
- **408重点**：SNMP操作、SNMP使用UDP的原因

## 原理分析

### SNMP操作详解

| 操作 | 方向 | 功能 | 版本 |
|------|------|------|------|
| GetRequest | 管理者→代理 | 查询一个或多个OID | v1/v2/v3 |
| GetNextRequest | 管理者→代理 | 查询下一个OID | v1/v2/v3 |
| GetBulkRequest | 管理者→代理 | 批量查询 | v2/v3 |
| SetRequest | 管理者→代理 | 设置OID值 | v1/v2/v3 |
| Response | 代理→管理者 | 回复查询/设置结果 | v1/v2/v3 |
| Trap | 代理→管理者 | 异步告警通知 | v1/v2/v3 |
| Inform | 管理者→管理者 | 带确认的通知 | v2/v3 |

### SNMP为什么用UDP

1. **轻量级**：SNMP操作简单，不需要TCP的复杂机制
2. **开销小**：UDP无连接，不需要建立/释放连接
3. **实时性**：Trap告警需要尽快到达，TCP握手会增加延迟
4. **容忍丢失**：管理查询偶尔丢失可重试，不影响大局

**类比**：SNMP像发短信（UDP），不需要先打电话确认对方在线（TCP），直接发就行。

### SNMPv3安全增强

- **USM（User-based Security Model）**：
  - 用户认证：HMAC-MD5/HMAC-SHA
  - 数据加密：DES/AES
- **VACM（View-based Access Control Model）**：
  - 基于视图的访问控制
  - 不同用户可访问不同的MIB子树

## 直观理解

- **SNMP操作比喻**：
  - GetRequest："你现在CPU利用率多少？"→"85%"
  - SetRequest："把端口1的状态设为down"→"已设置"
  - Trap："警报！CPU利用率超过95%！"
- **为什么用UDP**：管理协议要"轻"，不能因为管理协议本身占用太多资源
- **MIB读取**：像读一个目录，OID是文件路径

## 协议关联

- **与UDP**：SNMP直接使用UDP（端口161/162）
- **与MIB/SMI**：SMI定义对象类型，MIB存储对象实例，SNMP操作MIB
- **与网络层**：MIB中存储IP地址表、路由表等网络层信息
- **408常见陷阱**：
  - "SNMP使用TCP"——错，SNMP使用UDP
  - "SNMP管理者主动发送Trap"——错，Trap是代理主动发给管理者的
  - "SNMPv1很安全"——错，团体名明文传输，非常不安全
