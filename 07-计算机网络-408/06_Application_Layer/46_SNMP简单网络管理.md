# 46_SNMP简单网络管理

## 核心概念

- **SNMP（Simple Network Management Protocol）**：简单网络管理协议
- SNMP使用**UDP**协议：
  - **管理站→代理：端口161**
  - **代理→管理站（Trap）：端口162**
- SNMP是**C/S模式**的应用协议
- SNMP用于**网络设备的监控和管理**
- **408考试重点**：SNMP端口号、基本概念、MIB的作用

## 原理分析

### SNMP体系结构

1. **管理站（Manager/NMS）**：
   - 运行SNMP管理软件
   - 监控和管理网络设备
   - 发送请求，接收Trap

2. **代理（Agent）**：
   - 运行在被管理设备上
   - 响应管理站的请求
   - 主动发送Trap告警

3. **MIB（Management Information Base）**：
   - 管理信息库
   - 存储被管理设备的信息
   - 使用树状结构组织

4. **SMI（Structure of Management Information）**：
   - 管理信息结构
   - 定义MIB中对象的格式

### SNMP操作

| 操作 | 说明 | 端口 |
|------|------|------|
| GetRequest | 获取对象值 | 161 |
| GetNextRequest | 获取下一个对象值 | 161 |
| SetRequest | 设置对象值 | 161 |
| GetResponse | 响应请求 | 161 |
| Trap | 主动告警 | 162 |

### SNMP版本

| 版本 | 安全性 | 特点 |
|------|--------|------|
| SNMPv1 | 无 | 简单，使用团体名认证 |
| SNMPv2c | 弱 | 增加了GetBulk操作 |
| SNMPv3 | 强 | 支持加密、认证、访问控制 |

### MIB结构

MIB使用OID（Object Identifier）标识对象：
- OID是树状结构的数字串
- 如：`1.3.6.1.2.1.1.1.0`（系统描述）
- `1.3.6.1` = ISO.org.dod.internet

## 直观理解

**SNMP就像物业管理系统**：
- **管理站** = 物业管理中心
- **代理** = 每栋楼的管理员
- **MIB** = 每栋楼的信息表（水电表读数等）
- **GetRequest** = 物业查抄水电表
- **SetRequest** = 物业设置某些参数
- **Trap** = 楼栋管理员主动报告问题（如漏水报警）

**记忆技巧**：
- SNMP = "简单网络管理"，端口161/162
- 管理站→代理 = 端口161
- Trap = 端口162（主动告警）
- MIB = "管理信息库"

## 协议关联

- **SNMP与UDP**：SNMP使用UDP，端口161/162
- **SNMP与MIB**：MIB存储被管理设备的信息
- **SNMP与TCP**：SNMP不使用TCP（简单、快速）
- **408考点**：
  - SNMP端口号：161（请求）、162（Trap）
  - SNMP使用UDP
  - MIB的作用
  - SNMP的基本操作
- **陷阱**：SNMP使用UDP，不是TCP；端口161和162用途不同
